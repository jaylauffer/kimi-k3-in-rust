//! Safetensors shard indexing and bounded, portable reads.
//!
//! This keeps the C engine's two important guarantees: shard order is stable and a
//! tensor's declared shape must exactly cover its stored byte span. The JSON header is
//! deserialized entry by entry rather than as a complete DOM, which matters for the
//! released checkpoint's tens-of-megabytes header.

use std::{
    collections::HashMap,
    fmt,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use serde::{
    Deserialize,
    de::{DeserializeSeed, Error as _, IgnoredAny, MapAccess, Visitor},
};

const WIDEN_CHUNK_BYTES: usize = 4 << 20;

/// The scalar representation recorded by a safetensors header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8,
    /// Brain floating point with 16 stored bits.
    Bf16,
    /// IEEE 754 half precision.
    F16,
    /// IEEE 754 single precision.
    F32,
}

impl DType {
    /// Returns the number of bytes occupied by one stored element.
    #[must_use]
    pub const fn element_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "U8" => Some(Self::U8),
            "BF16" => Some(Self::Bf16),
            "F16" => Some(Self::F16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }
}

/// Location and shape of one tensor inside a safetensors shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorInfo {
    /// The exact, decoded tensor name from the header.
    pub name: String,
    /// Zero-based position in [`SafeTensorIndex::shard_paths`].
    pub shard: usize,
    /// The on-disk scalar representation.
    pub dtype: DType,
    /// Tensor dimensions. An empty shape represents one scalar.
    pub shape: Vec<usize>,
    /// Absolute byte position in the shard file.
    pub offset: u64,
    /// Number of stored bytes for the tensor.
    pub nbytes: usize,
    elements: usize,
}

impl TensorInfo {
    /// Returns the number of logical elements, including one for a scalar shape.
    #[must_use]
    pub const fn numel(&self) -> usize {
        self.elements
    }
}

/// An immutable, name-indexed view of all safetensors shards in a directory.
#[derive(Debug)]
pub struct SafeTensorIndex {
    shard_paths: Vec<PathBuf>,
    tensors: Vec<TensorInfo>,
    tensor_indices: HashMap<String, usize>,
}

impl SafeTensorIndex {
    /// Opens and indexes every `*.safetensors` file in `directory`.
    ///
    /// Paths are sorted before they are assigned shard indices. Each header is checked
    /// before this method returns, so later reads cannot run past the tensor's declared
    /// extent or use an unsupported type.
    ///
    /// # Errors
    ///
    /// Returns [`SafeTensorError`] when the directory cannot be read, contains no
    /// shards, or a shard violates the safetensors layout used by the C reference.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, SafeTensorError> {
        let directory = directory.as_ref();
        let entries = fs::read_dir(directory).map_err(|error| SafeTensorError::ReadDirectory {
            path: directory.to_path_buf(),
            error: error.to_string(),
        })?;

        let mut shard_paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| SafeTensorError::ReadDirectory {
                path: directory.to_path_buf(),
                error: error.to_string(),
            })?;
            let path = entry.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "safetensors")
            {
                shard_paths.push(path);
            }
        }
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(SafeTensorError::NoShards {
                path: directory.to_path_buf(),
            });
        }

        let mut tensors = Vec::new();
        for (shard, path) in shard_paths.iter().enumerate() {
            scan_shard(path, shard, &mut tensors)?;
        }

        let mut tensor_indices = HashMap::with_capacity(tensors.len());
        for (index, tensor) in tensors.iter().enumerate() {
            if tensor_indices.insert(tensor.name.clone(), index).is_some() {
                return Err(SafeTensorError::DuplicateName {
                    name: tensor.name.clone(),
                });
            }
        }

        Ok(Self {
            shard_paths,
            tensors,
            tensor_indices,
        })
    }

    /// Returns every shard path in its stable, zero-based shard order.
    #[must_use]
    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    /// Returns every indexed tensor in header discovery order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Finds a tensor by its exact header name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_indices
            .get(name)
            .and_then(|&index| self.tensors.get(index))
    }

    /// Reads exactly the raw stored bytes for `tensor`.
    ///
    /// # Errors
    ///
    /// Returns [`SafeTensorError`] if the tensor did not originate from this index or
    /// its file cannot be reopened and read in full.
    pub fn read_raw(&self, tensor: &TensorInfo) -> Result<Vec<u8>, SafeTensorError> {
        let mut bytes = vec![0; tensor.nbytes];
        self.read_exact(tensor, &mut bytes)?;
        Ok(bytes)
    }

    /// Reads a bounded byte range from one shard.
    ///
    /// This is the primitive used for a contiguous routed-expert run. Higher-level
    /// loaders must derive the range from indexed [`TensorInfo`] values rather than
    /// accepting an unchecked checkpoint offset from a caller.
    ///
    /// # Errors
    ///
    /// Returns [`SafeTensorError`] if `shard` is not part of this index or the requested
    /// bytes cannot be read in full.
    pub fn read_range(
        &self,
        shard: usize,
        offset: u64,
        output: &mut [u8],
    ) -> Result<(), SafeTensorError> {
        let path = self
            .shard_paths
            .get(shard)
            .ok_or_else(|| SafeTensorError::UnknownShard {
                name: format!("range at byte {offset}"),
                shard,
            })?;
        let mut file = File::open(path).map_err(|error| SafeTensorError::OpenShard {
            path: path.clone(),
            error: error.to_string(),
        })?;
        file.seek(SeekFrom::Start(offset))
            .and_then(|_| file.read_exact(output))
            .map_err(|error| SafeTensorError::ShortRead {
                path: path.clone(),
                name: format!("range at byte {offset}"),
                error: error.to_string(),
            })
    }

    /// Reads and widens `tensor` into exactly `output.len()` `f32` values.
    ///
    /// The conversion is bit-exact with the C reader for BF16 and F16, including signed
    /// zero, infinities, NaNs, and half-precision subnormals. Reads are bounded to four
    /// MiB of temporary storage so a multi-gigabyte embedding does not require a second
    /// multi-gigabyte buffer.
    ///
    /// # Errors
    ///
    /// Returns [`SafeTensorError`] if the output does not exactly fit the tensor, or
    /// the shard cannot be read in full.
    pub fn read_f32(&self, tensor: &TensorInfo, output: &mut [f32]) -> Result<(), SafeTensorError> {
        if output.len() != tensor.numel() {
            return Err(SafeTensorError::OutputLength {
                name: tensor.name.clone(),
                expected: tensor.numel(),
                actual: output.len(),
            });
        }
        if output.is_empty() {
            return Ok(());
        }

        let element_size = tensor.dtype.element_size();
        let chunk_elements = WIDEN_CHUNK_BYTES / element_size;
        let mut raw = vec![0_u8; chunk_elements * element_size];
        let path = self.tensor_path(tensor)?;
        let mut file = File::open(path).map_err(|error| SafeTensorError::OpenShard {
            path: path.to_path_buf(),
            error: error.to_string(),
        })?;

        for (element_offset, destination) in output.chunks_mut(chunk_elements).enumerate() {
            let bytes = destination.len() * element_size;
            let relative_offset = element_offset
                .checked_mul(chunk_elements)
                .and_then(|start| start.checked_mul(element_size))
                .ok_or_else(|| SafeTensorError::OffsetOverflow {
                    name: tensor.name.clone(),
                })?;
            let absolute_offset = tensor
                .offset
                .checked_add(u64::try_from(relative_offset).map_err(|_| {
                    SafeTensorError::OffsetOverflow {
                        name: tensor.name.clone(),
                    }
                })?)
                .ok_or_else(|| SafeTensorError::OffsetOverflow {
                    name: tensor.name.clone(),
                })?;
            read_exact_at(&mut file, path, absolute_offset, &mut raw[..bytes], tensor)?;
            widen_into(tensor.dtype, &raw[..bytes], destination);
        }
        Ok(())
    }

    fn read_exact(&self, tensor: &TensorInfo, output: &mut [u8]) -> Result<(), SafeTensorError> {
        if output.len() != tensor.nbytes {
            return Err(SafeTensorError::RawOutputLength {
                name: tensor.name.clone(),
                expected: tensor.nbytes,
                actual: output.len(),
            });
        }
        let path = self.tensor_path(tensor)?;
        let mut file = File::open(path).map_err(|error| SafeTensorError::OpenShard {
            path: path.to_path_buf(),
            error: error.to_string(),
        })?;
        read_exact_at(&mut file, path, tensor.offset, output, tensor)
    }

    fn tensor_path(&self, tensor: &TensorInfo) -> Result<&Path, SafeTensorError> {
        self.shard_paths
            .get(tensor.shard)
            .map(PathBuf::as_path)
            .ok_or_else(|| SafeTensorError::UnknownShard {
                name: tensor.name.clone(),
                shard: tensor.shard,
            })
    }
}

/// A safetensors index or read failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SafeTensorError {
    /// The directory holding shards could not be enumerated.
    ReadDirectory { path: PathBuf, error: String },
    /// The directory contained no files with a `.safetensors` extension.
    NoShards { path: PathBuf },
    /// A shard could not be opened.
    OpenShard { path: PathBuf, error: String },
    /// The shard was too short to contain the eight-byte header length.
    ShortHeaderLength { path: PathBuf },
    /// The header length could not describe a valid header inside the shard.
    InvalidHeaderLength {
        path: PathBuf,
        length: u64,
        file_size: u64,
    },
    /// The JSON header or a tensor description was invalid.
    InvalidHeader { path: PathBuf, detail: String },
    /// A tensor declared a scalar type outside the reader's supported set.
    UnsupportedDType {
        path: PathBuf,
        name: String,
        dtype: String,
    },
    /// A tensor's byte range did not equal its shape times element size.
    InvalidByteSpan {
        path: PathBuf,
        name: String,
        declared: u64,
        expected: usize,
    },
    /// A tensor's declared range extended past the shard's data region.
    TensorPastEnd { path: PathBuf, name: String },
    /// A name appeared in more than one shard.
    DuplicateName { name: String },
    /// A caller passed tensor information that does not point at a known shard.
    UnknownShard { name: String, shard: usize },
    /// A seek position could not be represented or computed safely.
    OffsetOverflow { name: String },
    /// The caller supplied a raw-byte buffer of the wrong length.
    RawOutputLength {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// The caller supplied a widened output buffer of the wrong number of elements.
    OutputLength {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// A read stopped before the required tensor bytes arrived.
    ShortRead {
        path: PathBuf,
        name: String,
        error: String,
    },
}

impl fmt::Display for SafeTensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadDirectory { path, error } => {
                write!(
                    formatter,
                    "cannot read safetensors directory {}: {error}",
                    path.display()
                )
            }
            Self::NoShards { path } => {
                write!(formatter, "no .safetensors files in {}", path.display())
            }
            Self::OpenShard { path, error } => {
                write!(formatter, "cannot open shard {}: {error}", path.display())
            }
            Self::ShortHeaderLength { path } => {
                write!(
                    formatter,
                    "{} is too short for a safetensors header length",
                    path.display()
                )
            }
            Self::InvalidHeaderLength {
                path,
                length,
                file_size,
            } => write!(
                formatter,
                "{} has impossible safetensors header length {length} for {file_size} bytes",
                path.display()
            ),
            Self::InvalidHeader { path, detail } => {
                write!(
                    formatter,
                    "{} has an invalid safetensors header: {detail}",
                    path.display()
                )
            }
            Self::UnsupportedDType { path, name, dtype } => write!(
                formatter,
                "{} uses unsupported dtype {dtype} for tensor {name}",
                path.display()
            ),
            Self::InvalidByteSpan {
                path,
                name,
                declared,
                expected,
            } => write!(
                formatter,
                "{} tensor {name} spans {declared} bytes but its shape requires {expected}",
                path.display()
            ),
            Self::TensorPastEnd { path, name } => {
                write!(
                    formatter,
                    "{} tensor {name} extends past end of shard",
                    path.display()
                )
            }
            Self::DuplicateName { name } => write!(formatter, "duplicate safetensors name {name}"),
            Self::UnknownShard { name, shard } => {
                write!(formatter, "tensor {name} references unknown shard {shard}")
            }
            Self::OffsetOverflow { name } => write!(formatter, "tensor {name} offset overflowed"),
            Self::RawOutputLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "raw output for {name} has {actual} bytes; expected {expected}"
            ),
            Self::OutputLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "output for {name} has {actual} elements; expected {expected}"
            ),
            Self::ShortRead { path, name, error } => write!(
                formatter,
                "short read for tensor {name} from {}: {error}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for SafeTensorError {}

#[derive(Deserialize)]
struct HeaderTensor {
    dtype: String,
    #[serde(default)]
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct HeaderSeed<'a> {
    path: &'a Path,
    shard: usize,
    data_base: u64,
    data_length: u64,
    tensors: &'a mut Vec<TensorInfo>,
}

impl<'de> DeserializeSeed<'de> for HeaderSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(HeaderVisitor { seed: self })
    }
}

struct HeaderVisitor<'a> {
    seed: HeaderSeed<'a>,
}

impl<'de> Visitor<'de> for HeaderVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a safetensors header object")
    }

    fn visit_map<A>(mut self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Some(name) = map.next_key::<String>()? {
            if name == "__metadata__" {
                map.next_value::<IgnoredAny>()?;
                continue;
            }
            let entry = map.next_value::<HeaderTensor>()?;
            self.push(name, entry).map_err(A::Error::custom)?;
        }
        Ok(())
    }
}

impl HeaderVisitor<'_> {
    fn push(&mut self, name: String, entry: HeaderTensor) -> Result<(), String> {
        let path = self.seed.path;
        let dtype = DType::parse(&entry.dtype).ok_or_else(|| {
            format!(
                "unsupported dtype {} for tensor {name} in {}",
                entry.dtype,
                path.display()
            )
        })?;
        if entry.shape.len() > 4 {
            return Err(format!(
                "tensor {name} has rank {} above the four-rank limit",
                entry.shape.len()
            ));
        }
        let elements = entry
            .shape
            .iter()
            .try_fold(1_usize, |product, &dimension| {
                product
                    .checked_mul(dimension)
                    .ok_or_else(|| format!("tensor {name} shape overflows usize"))
            })?;
        let expected = elements
            .checked_mul(dtype.element_size())
            .ok_or_else(|| format!("tensor {name} byte length overflows usize"))?;
        let [start, end] = entry.data_offsets;
        let declared = end
            .checked_sub(start)
            .ok_or_else(|| format!("tensor {name} has descending data_offsets"))?;
        if declared != u64::try_from(expected).map_err(|_| format!("tensor {name} is too large"))? {
            return Err(format!(
                "tensor {name} spans {declared} bytes but its shape requires {expected}"
            ));
        }
        if end > self.seed.data_length {
            return Err(format!("tensor {name} extends past the shard data region"));
        }
        let offset = self
            .seed
            .data_base
            .checked_add(start)
            .ok_or_else(|| format!("tensor {name} absolute offset overflows"))?;
        self.seed.tensors.push(TensorInfo {
            name,
            shard: self.seed.shard,
            dtype,
            shape: entry.shape,
            offset,
            nbytes: expected,
            elements,
        });
        Ok(())
    }
}

fn scan_shard(
    path: &Path,
    shard: usize,
    tensors: &mut Vec<TensorInfo>,
) -> Result<(), SafeTensorError> {
    let mut file = File::open(path).map_err(|error| SafeTensorError::OpenShard {
        path: path.to_path_buf(),
        error: error.to_string(),
    })?;
    let file_size = file
        .metadata()
        .map_err(|error| SafeTensorError::OpenShard {
            path: path.to_path_buf(),
            error: error.to_string(),
        })?
        .len();
    let mut header_length = [0_u8; 8];
    file.read_exact(&mut header_length)
        .map_err(|_| SafeTensorError::ShortHeaderLength {
            path: path.to_path_buf(),
        })?;
    let header_length = u64::from_le_bytes(header_length);
    let data_base =
        8_u64
            .checked_add(header_length)
            .ok_or_else(|| SafeTensorError::InvalidHeaderLength {
                path: path.to_path_buf(),
                length: header_length,
                file_size,
            })?;
    if header_length == 0 || data_base > file_size {
        return Err(SafeTensorError::InvalidHeaderLength {
            path: path.to_path_buf(),
            length: header_length,
            file_size,
        });
    }
    let header_size =
        usize::try_from(header_length).map_err(|_| SafeTensorError::InvalidHeaderLength {
            path: path.to_path_buf(),
            length: header_length,
            file_size,
        })?;
    let mut header = vec![0_u8; header_size];
    file.read_exact(&mut header)
        .map_err(|error| SafeTensorError::InvalidHeader {
            path: path.to_path_buf(),
            detail: format!("could not read {header_length} header bytes: {error}"),
        })?;
    let mut deserializer = serde_json::Deserializer::from_slice(&header);
    HeaderSeed {
        path,
        shard,
        data_base,
        data_length: file_size - data_base,
        tensors,
    }
    .deserialize(&mut deserializer)
    .and_then(|()| deserializer.end())
    .map_err(|error| SafeTensorError::InvalidHeader {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })
}

fn read_exact_at(
    file: &mut File,
    path: &Path,
    offset: u64,
    output: &mut [u8],
    tensor: &TensorInfo,
) -> Result<(), SafeTensorError> {
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(output))
        .map_err(|error| SafeTensorError::ShortRead {
            path: path.to_path_buf(),
            name: tensor.name.clone(),
            error: error.to_string(),
        })
}

fn widen_into(dtype: DType, input: &[u8], output: &mut [f32]) {
    match dtype {
        DType::U8 => {
            for (&value, destination) in input.iter().zip(output) {
                *destination = f32::from(value);
            }
        }
        DType::Bf16 => {
            for (bytes, destination) in input.chunks_exact(2).zip(output) {
                let value = u16::from_le_bytes([bytes[0], bytes[1]]);
                *destination = f32::from_bits(u32::from(value) << 16);
            }
        }
        DType::F16 => {
            for (bytes, destination) in input.chunks_exact(2).zip(output) {
                let value = u16::from_le_bytes([bytes[0], bytes[1]]);
                *destination = f32::from_bits(f16_to_f32_bits(value));
            }
        }
        DType::F32 => {
            for (bytes, destination) in input.chunks_exact(4).zip(output) {
                *destination = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
    }
}

fn f16_to_f32_bits(value: u16) -> u32 {
    let sign = u32::from(value & 0x8000) << 16;
    let exponent = (value >> 10) & 0x1f;
    let mantissa = u32::from(value & 0x03ff);
    match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut shifts = 0_u32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                shifts += 1;
            }
            sign | ((127 - 15 - shifts + 1) << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 112) << 23) | (mantissa << 13),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{DType, SafeTensorIndex};

    fn fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/st")
    }

    #[test]
    fn indexes_the_complete_c_fixture_with_stable_shards_and_offsets() {
        let index = SafeTensorIndex::open(fixture_directory()).expect("fixture index opens");

        assert_eq!(index.shard_paths().len(), 2);
        assert_eq!(index.tensors().len(), 15);
        assert!(index.tensor("__metadata__").is_none());
        assert!(index.tensor("not.present").is_none());
        assert_eq!(
            index.shard_paths()[0]
                .file_name()
                .and_then(|name| name.to_str()),
            Some("model-00001-of-00002.safetensors")
        );

        let f32 = index
            .tensor("plain.f32.2d")
            .expect("f32 fixture is indexed");
        assert_eq!(f32.dtype, DType::F32);
        assert_eq!(f32.shape, [16, 16]);
        assert_eq!(f32.offset, 1241);
        assert_eq!(f32.nbytes, 1024);
        assert_eq!(f32.numel(), 256);

        let escaped = index
            .tensor(r#"weird\name."with"quotes"#)
            .expect("escaped JSON name is decoded exactly");
        assert_eq!(escaped.shard, 1);
        assert_eq!(escaped.offset, 961);
        assert_eq!(escaped.shape, [3]);
    }

    #[test]
    fn widens_special_bf16_and_f16_values_bit_exactly() {
        let index = SafeTensorIndex::open(fixture_directory()).expect("fixture index opens");

        let bf16 = index
            .tensor("plain.bf16.1d")
            .expect("bf16 fixture is indexed");
        let mut bf16_values = vec![0.0; bf16.numel()];
        index
            .read_f32(bf16, &mut bf16_values)
            .expect("bf16 fixture widens");
        assert_eq!(
            bf16_values[..6]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0,
                0x8000_0000,
                0x0080_0000
            ]
        );

        let f16 = index
            .tensor("tricky.f16.1d")
            .expect("f16 fixture is indexed");
        let mut f16_values = vec![0.0; f16.numel()];
        index
            .read_f32(f16, &mut f16_values)
            .expect("f16 fixture widens");
        assert_eq!(
            f16_values[..8]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0,
                0x8000_0000,
                0x3380_0000,
                0x387f_c000,
                0x3880_0000,
            ]
        );
    }

    #[test]
    fn reads_raw_values_scalars_and_empty_tensors() {
        let index = SafeTensorIndex::open(fixture_directory()).expect("fixture index opens");

        let scalar = index
            .tensor("scalar.f32")
            .expect("scalar fixture is indexed");
        assert_eq!(
            index.read_raw(scalar).expect("scalar reads"),
            3.5_f32.to_le_bytes()
        );
        let mut scalar_value = [0.0];
        index
            .read_f32(scalar, &mut scalar_value)
            .expect("scalar widens");
        assert_eq!(scalar_value[0].to_bits(), 3.5_f32.to_bits());

        let empty = index.tensor("empty.f32").expect("empty fixture is indexed");
        assert_eq!(empty.numel(), 0);
        assert!(
            index
                .read_raw(empty)
                .expect("empty tensor reads")
                .is_empty()
        );
        index
            .read_f32(empty, &mut [])
            .expect("empty tensor widens without a read");
    }
}
