//! Shard reads through `loadngo-proactor`.
//!
//! Every read the port makes is a positioned completion read against a handle opened
//! once per shard and submitted to one owned proactor. That is loadngo's portable I/O
//! path: `io_uring` on Linux, IOCP on Windows, kqueue on macOS and iOS, epoll on Android.
//! It takes the place of the C engine's per-OS `pread`, `O_DIRECT` and `F_NOCACHE`
//! shims.
//!
//! A batch is submitted in full before any completion is collected, so a backend that
//! performs file reads asynchronously services the whole batch at once. `io_uring` and
//! IOCP do. The kqueue and epoll backends currently finish a regular-file read inside
//! submission, so on macOS, iOS and Android a batch still runs one read at a time;
//! offloading those reads is loadngo work, and this module needs no change when it
//! lands.

use std::{
    fmt,
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
};

use loadngo_proactor::{IoBuf, IoResult, PlatformPort, Proactor, RawFdCompat};

/// One positioned read: exactly `buffer.len()` bytes from `offset` in `shard`.
///
/// The buffer's allocation travels with the read and comes back filled, so a caller
/// that passes storage it already owns, such as a cache slot, reads with no copy.
#[derive(Debug)]
pub struct ReadRequest {
    /// Index into the paths the [`ShardFiles`] was opened with.
    pub shard: usize,
    /// Absolute byte position in the shard.
    pub offset: u64,
    /// Destination; its length is the number of bytes to read.
    pub buffer: Vec<u8>,
}

/// Every shard of a checkpoint, opened once, with the proactor that reads them.
pub struct ShardFiles {
    paths: Vec<PathBuf>,
    files: Vec<File>,
    proactor: Proactor<PlatformPort>,
    /// Serialises batches. A completion is dispatched by whichever thread polls, so two
    /// threads driving one proactor could each consume the other's last completion and
    /// leave one of them blocked in `poll` indefinitely.
    drive: Mutex<()>,
}

impl fmt::Debug for ShardFiles {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ShardFiles")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

impl ShardFiles {
    /// Opens every shard once and creates this platform's proactor.
    ///
    /// # Errors
    ///
    /// Returns [`ShardIoError`] when a shard cannot be opened or the proactor cannot be
    /// created.
    pub fn open(paths: &[PathBuf]) -> Result<Self, ShardIoError> {
        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            files.push(open_shard(path).map_err(|error| ShardIoError::Open {
                path: path.clone(),
                error: error.to_string(),
            })?);
        }
        let proactor =
            loadngo_proactor::new_platform_proactor().map_err(|error| ShardIoError::Proactor {
                error: error.to_string(),
            })?;
        Ok(Self {
            paths: paths.to_vec(),
            files,
            proactor,
            drive: Mutex::new(()),
        })
    }

    /// Returns the shard paths in shard-index order.
    #[must_use]
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Reads one range; a batch of one.
    ///
    /// # Errors
    ///
    /// Returns [`ShardIoError`] under the same conditions as [`Self::read_batch`].
    pub fn read(
        &self,
        shard: usize,
        offset: u64,
        buffer: Vec<u8>,
    ) -> Result<Vec<u8>, ShardIoError> {
        let mut filled = self.read_batch(vec![ReadRequest {
            shard,
            offset,
            buffer,
        }])?;
        Ok(filled.pop().unwrap_or_default())
    }

    /// Reads every request, submitting the whole batch before collecting completions.
    ///
    /// Returns the filled buffers in request order. A short read resumes where it
    /// stopped; reaching end of file inside a requested range is an error, never a short
    /// buffer.
    ///
    /// # Errors
    ///
    /// Returns [`ShardIoError`] for an unknown shard, a submission or poll failure, a
    /// failed read, or end of file inside a range. Operations already submitted are
    /// completed before a submission error is returned, so no buffer is left with the
    /// kernel.
    pub fn read_batch(&self, requests: Vec<ReadRequest>) -> Result<Vec<Vec<u8>>, ShardIoError> {
        self.wait_batch(self.submit_batch(requests)?)
    }

    /// Submits every request without waiting for any completion, so the operating
    /// system can complete them while the caller does other, non-I/O work.
    ///
    /// Nothing here spawns a thread: submission itself does not drive the proactor,
    /// and completion is only ever observed later, when [`Self::wait_batch`] polls
    /// for it. A caller that wants overlap -- start a prefetch, compute on data
    /// already in hand, then wait -- gets it for free from this split; a caller
    /// that wants the old blocking behavior gets exactly that from
    /// [`Self::read_batch`], which is just this immediately followed by
    /// [`Self::wait_batch`].
    ///
    /// # Errors
    ///
    /// Returns [`ShardIoError::UnknownShard`] if any request names an unopened
    /// shard, or [`ShardIoError::Proactor`] if the proactor refuses a submission
    /// (every already-submitted read is still completed before the error returns,
    /// so no buffer is left with the kernel).
    pub(crate) fn submit_batch(
        &self,
        requests: Vec<ReadRequest>,
    ) -> Result<PendingBatch, ShardIoError> {
        if let Some(request) = requests
            .iter()
            .find(|request| request.shard >= self.files.len())
        {
            return Err(ShardIoError::UnknownShard {
                shard: request.shard,
            });
        }

        let mut outputs = Vec::with_capacity(requests.len());
        let mut round = Vec::with_capacity(requests.len());
        for (index, request) in requests.into_iter().enumerate() {
            if request.buffer.is_empty() {
                outputs.push(request.buffer);
                continue;
            }
            outputs.push(Vec::new());
            round.push(Pending {
                index,
                shard: request.shard,
                offset: request.offset,
                buffer: request.buffer,
            });
        }
        let submitted = self.submit(round);
        Ok(PendingBatch { submitted, outputs })
    }

    /// Drives the proactor until every request `pending` was submitted with has
    /// completed, resuming a short read exactly as [`Self::read_batch`] always has.
    ///
    /// # Errors
    ///
    /// Returns [`ShardIoError`] for a poll failure, a failed read, or end of file
    /// inside a requested range.
    pub(crate) fn wait_batch(&self, pending: PendingBatch) -> Result<Vec<Vec<u8>>, ShardIoError> {
        let PendingBatch {
            submitted,
            mut outputs,
        } = pending;
        let mut round = self.collect(submitted, &mut outputs)?;
        while !round.is_empty() {
            round = self.collect(self.submit(round), &mut outputs)?;
        }
        Ok(outputs)
    }

    /// Drives the proactor until every submitted request completes, folding each
    /// result into `outputs` and returning any short read as a new round to resume.
    fn collect(
        &self,
        submitted: Submitted,
        outputs: &mut [Vec<u8>],
    ) -> Result<Vec<Pending>, ShardIoError> {
        let _drive = self.drive.lock().unwrap_or_else(PoisonError::into_inner);
        while submitted
            .results
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|result| result.is_some())
            .count()
            < submitted.metas.len()
        {
            self.proactor
                .run_once()
                .map_err(|error| ShardIoError::Proactor {
                    error: error.to_string(),
                })?;
        }
        if let Some(error) = submitted.submit_error {
            return Err(ShardIoError::Proactor {
                error: error.to_string(),
            });
        }

        let collected = std::mem::take(
            &mut *submitted
                .results
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        let mut next = Vec::new();
        for (meta, result) in submitted.metas.into_iter().zip(collected) {
            let result = result.ok_or_else(|| ShardIoError::Proactor {
                error: "a submitted read never completed".to_owned(),
            })?;
            let path = &self.paths[meta.shard];
            let transfer = result.map_err(|error| read_error(path, meta.offset, &error))?;
            let bytes = transfer.buf.into_vec();
            if bytes.is_empty() {
                return Err(ShardIoError::UnexpectedEof {
                    path: path.clone(),
                    offset: meta.offset,
                    missing: meta.wanted,
                });
            }
            let got = bytes.len();
            let output = &mut outputs[meta.index];
            if output.is_empty() {
                *output = bytes;
            } else {
                output.extend_from_slice(&bytes);
            }
            if got < meta.wanted {
                let advanced = u64::try_from(got)
                    .ok()
                    .and_then(|got| meta.offset.checked_add(got))
                    .ok_or_else(|| ShardIoError::OffsetOverflow { path: path.clone() })?;
                next.push(Pending {
                    index: meta.index,
                    shard: meta.shard,
                    offset: advanced,
                    buffer: vec![0; meta.wanted - got],
                });
            }
        }
        Ok(next)
    }

    /// Submits every request to the proactor and returns immediately, without
    /// driving it -- so nothing here waits, and no thread is involved. A
    /// submission failure still submits nothing further, but `metas` and
    /// `results` cover everything submitted before it, so [`Self::collect`] can
    /// still drain those completions before the error surfaces.
    fn submit(&self, round: Vec<Pending>) -> Submitted {
        let results: Arc<Mutex<Vec<Option<IoResult>>>> =
            Arc::new(Mutex::new((0..round.len()).map(|_| None).collect()));
        let handle = self.proactor.handle();
        let mut metas = Vec::with_capacity(round.len());
        let mut submit_error = None;

        for (slot, pending) in round.into_iter().enumerate() {
            let Pending {
                index,
                shard,
                offset,
                buffer,
            } = pending;
            let wanted = buffer.len();
            let results = Arc::clone(&results);
            let submitted = handle.read(
                raw_handle(&self.files[shard]),
                IoBuf::from_vec(buffer),
                offset,
                move |result: IoResult| {
                    results.lock().unwrap_or_else(PoisonError::into_inner)[slot] = Some(result);
                },
            );
            if let Err(error) = submitted {
                submit_error = Some(error);
                break;
            }
            metas.push(Meta {
                index,
                shard,
                offset,
                wanted,
            });
        }

        Submitted {
            results,
            metas,
            submit_error,
        }
    }
}

/// Everything [`ShardFiles::submit_batch`] needs [`ShardFiles::wait_batch`] to
/// finish later: what was submitted, plus the output buffers assembled so far
/// (already-empty requests are filled immediately and need no completion).
pub(crate) struct PendingBatch {
    submitted: Submitted,
    outputs: Vec<Vec<u8>>,
}

/// One round of reads submitted to the proactor, not yet drained.
struct Submitted {
    results: Arc<Mutex<Vec<Option<IoResult>>>>,
    metas: Vec<Meta>,
    /// Set when a submission failed partway through a round; surfaced only after
    /// every read that did get submitted has been drained, matching the always-
    /// drain-what-was-submitted contract [`ShardFiles::read_batch`] has always had.
    submit_error: Option<io::Error>,
}

/// Makes `buffer` exactly `len` bytes long, reusing its allocation.
///
/// Shrinking never touches memory, and growing zeroes only the new tail, so a buffer
/// that is reused for same-sized reads is written by the kernel alone.
pub(crate) fn fit(buffer: &mut Vec<u8>, len: usize) {
    if buffer.len() >= len {
        buffer.truncate(len);
    } else {
        buffer.resize(len, 0);
    }
}

/// A shard read failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShardIoError {
    /// A shard could not be opened for reading.
    Open { path: PathBuf, error: String },
    /// The proactor could not be created, accept a submission, or poll.
    Proactor { error: String },
    /// A request named a shard this set was not opened with.
    UnknownShard { shard: usize },
    /// The operating system reported a failed read.
    Read {
        path: PathBuf,
        offset: u64,
        error: String,
    },
    /// End of file arrived before a requested range was filled.
    UnexpectedEof {
        path: PathBuf,
        offset: u64,
        missing: usize,
    },
    /// Resuming a short read would move past the addressable offset range.
    OffsetOverflow { path: PathBuf },
}

impl fmt::Display for ShardIoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open { path, error } => {
                write!(formatter, "cannot open shard {}: {error}", path.display())
            }
            Self::Proactor { error } => write!(formatter, "shard I/O proactor failed: {error}"),
            Self::UnknownShard { shard } => write!(formatter, "no shard with index {shard}"),
            Self::Read {
                path,
                offset,
                error,
            } => write!(
                formatter,
                "read at byte {offset} of {} failed: {error}",
                path.display()
            ),
            Self::UnexpectedEof {
                path,
                offset,
                missing,
            } => write!(
                formatter,
                "{} ended at byte {offset} with {missing} requested bytes unread",
                path.display()
            ),
            Self::OffsetOverflow { path } => {
                write!(formatter, "read offset in {} overflowed", path.display())
            }
        }
    }
}

impl std::error::Error for ShardIoError {}

struct Pending {
    index: usize,
    shard: usize,
    offset: u64,
    buffer: Vec<u8>,
}

struct Meta {
    index: usize,
    shard: usize,
    offset: u64,
    wanted: usize,
}

/// Windows reports a read that starts at or past end of file as the error
/// `ERROR_HANDLE_EOF` rather than as a zero-byte success, which is what `io_uring`,
/// kqueue and epoll return. Both mean the same thing here.
fn read_error(path: &Path, offset: u64, error: &io::Error) -> ShardIoError {
    const ERROR_HANDLE_EOF: i32 = 38;
    if cfg!(windows) && error.raw_os_error() == Some(ERROR_HANDLE_EOF) {
        return ShardIoError::UnexpectedEof {
            path: path.to_path_buf(),
            offset,
            missing: 0,
        };
    }
    ShardIoError::Read {
        path: path.to_path_buf(),
        offset,
        error: error.to_string(),
    }
}

#[cfg(unix)]
fn open_shard(path: &Path) -> io::Result<File> {
    File::open(path)
}

#[cfg(unix)]
fn raw_handle(file: &File) -> RawFdCompat {
    use std::os::fd::AsRawFd;
    file.as_raw_fd()
}

#[cfg(windows)]
fn open_shard(path: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_FLAG_OVERLAPPED: IOCP posts completions only for handles opened with it; a
    // plain handle would finish every read synchronously and never complete.
    const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OVERLAPPED)
        .open(path)
}

#[cfg(windows)]
fn raw_handle(file: &File) -> RawFdCompat {
    use std::os::windows::io::AsRawHandle;
    file.as_raw_handle() as usize as u64
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{ReadRequest, ShardFiles, ShardIoError};

    fn cache_shard() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/cache/model-00001-of-00001.safetensors")
    }

    #[test]
    fn a_batch_returns_every_range_byte_exact_in_request_order() {
        let path = cache_shard();
        let expected = fs::read(&path).expect("fixture reads");
        let files = ShardFiles::open(std::slice::from_ref(&path)).expect("shard opens");

        // Overlapping, out-of-order ranges spread over the whole shard.
        let span = expected.len() - 2048;
        let ranges: Vec<(usize, usize)> = (0..48_usize)
            .rev()
            .map(|index| ((index * 1231) % span, 1024 + index * 7))
            .collect();
        let requests = ranges
            .iter()
            .map(|&(offset, len)| ReadRequest {
                shard: 0,
                offset: offset as u64,
                buffer: vec![0; len],
            })
            .collect();

        let buffers = files.read_batch(requests).expect("batch reads");
        assert_eq!(buffers.len(), ranges.len());
        for (&(offset, len), buffer) in ranges.iter().zip(&buffers) {
            assert_eq!(buffer.as_slice(), &expected[offset..offset + len]);
        }
    }

    #[test]
    fn a_range_past_end_of_file_is_an_error_not_a_short_buffer() {
        let path = cache_shard();
        let size = fs::metadata(&path).expect("fixture metadata").len();
        let files = ShardFiles::open(std::slice::from_ref(&path)).expect("shard opens");

        let error = files
            .read(0, size - 16, vec![0; 64])
            .expect_err("reading past the end must fail");
        assert!(
            matches!(error, ShardIoError::UnexpectedEof { .. }),
            "expected UnexpectedEof, got {error:?}"
        );
    }

    #[test]
    fn a_request_for_an_unknown_shard_is_refused_before_any_read() {
        let files = ShardFiles::open(&[cache_shard()]).expect("shard opens");
        assert_eq!(
            files.read(1, 0, vec![0; 8]),
            Err(ShardIoError::UnknownShard { shard: 1 })
        );
    }

    #[test]
    fn submit_then_wait_reads_the_same_bytes_as_read_batch_with_work_done_in_between() {
        let path = cache_shard();
        let expected = fs::read(&path).expect("fixture reads");
        let files = ShardFiles::open(std::slice::from_ref(&path)).expect("shard opens");

        let span = expected.len() - 2048;
        let ranges: Vec<(usize, usize)> = (0..16_usize)
            .map(|index| ((index * 733) % span, 512 + index * 3))
            .collect();
        let requests = ranges
            .iter()
            .map(|&(offset, len)| ReadRequest {
                shard: 0,
                offset: offset as u64,
                buffer: vec![0; len],
            })
            .collect();

        // The point of the split: real, non-I/O work happens here, between
        // submission and waiting, with the reads already in flight.
        let pending = files.submit_batch(requests).expect("submits");
        let mut unrelated_work = 0_u64;
        for value in 0..1_000_000_u64 {
            unrelated_work = unrelated_work.wrapping_add(value);
        }
        std::hint::black_box(unrelated_work);

        let buffers = files.wait_batch(pending).expect("waits");
        assert_eq!(buffers.len(), ranges.len());
        for (&(offset, len), buffer) in ranges.iter().zip(&buffers) {
            assert_eq!(buffer.as_slice(), &expected[offset..offset + len]);
        }
    }

    #[test]
    fn submit_batch_refuses_an_unknown_shard_before_any_read() {
        let files = ShardFiles::open(&[cache_shard()]).expect("shard opens");
        let requests = vec![ReadRequest {
            shard: 1,
            offset: 0,
            buffer: vec![0; 8],
        }];
        assert_eq!(
            files.submit_batch(requests).err(),
            Some(ShardIoError::UnknownShard { shard: 1 })
        );
    }
}
