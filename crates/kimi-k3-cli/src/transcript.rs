//! Saved chats: an append-only JSON-lines transcript for reviewing what Kimi did, and a
//! resume snapshot of the exact token history, rewritten after every round, so a chat can
//! be picked up after Ctrl-C at the prompt, a crash or a restart (`--resume`).
//!
//! Both live in `~/.loadngo/kimi/transcripts/`: `<stamp>.jsonl` and `<stamp>.state.json`.
//! Writing them never stops the chat; the first failure is reported once.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use loadngo_inference::Session;
use serde_json::{Value, json};

use crate::chat::Held;

/// Tool results longer than this are cut in the transcript (the model still saw them whole).
const MAX_LOGGED_RESULT: usize = 64 * 1024;
const STATE_VERSION: u64 = 1;

/// The default transcript directory, `~/.loadngo/kimi/transcripts`.
#[must_use]
pub fn default_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".loadngo/kimi/transcripts"))
}

/// What a resumed chat starts from.
pub struct Resumed {
    pub session: Session,
    /// Where the latest assistant message begins in the session's tokens.
    pub reply_start: usize,
    pub held: Option<Held>,
}

pub struct Transcript {
    log: PathBuf,
    state: PathBuf,
    file: Option<File>,
    failed: bool,
    format: &'static str,
}

impl Transcript {
    /// Starts a new transcript named after the local time.
    ///
    /// # Errors
    /// When the directory or the log file cannot be created.
    pub fn create(dir: &Path, format: &'static str, model: &Path) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let stamp = chrono::Local::now().format("%Y-%m-%d-%H%M%S").to_string();
        let mut stem = stamp.clone();
        let mut n = 1;
        while dir.join(format!("{stem}.jsonl")).exists() {
            n += 1;
            stem = format!("{stamp}-{n}");
        }
        let mut transcript = Self::open(dir, &stem, format)?;
        transcript.event(json!({"event": "start", "format": format, "model": model}));
        Ok(transcript)
    }

    /// Reopens a saved chat: `latest` (the newest in `dir`), or a `.jsonl`/`.state.json`
    /// path. The transcript goes on appending to the same log.
    ///
    /// # Errors
    /// When there is nothing to resume, the snapshot is unreadable, or it was saved by a
    /// different chat format (its tokens would mean nothing to this model).
    pub fn resume(
        dir: &Path,
        which: &str,
        format: &'static str,
        max_context: usize,
    ) -> Result<(Self, Resumed), String> {
        let state = if which == "latest" {
            let mut states: Vec<PathBuf> = fs::read_dir(dir)
                .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
                .filter_map(|entry| entry.ok().map(|e| e.path()))
                .filter(|p| p.to_string_lossy().ends_with(".state.json"))
                .collect();
            // Names start with the time, so the last in order is the newest.
            states.sort();
            states
                .pop()
                .ok_or_else(|| format!("no saved chats in {}", dir.display()))?
        } else {
            let path = PathBuf::from(which);
            let name = path.to_string_lossy();
            match name.strip_suffix(".jsonl") {
                Some(stem) => PathBuf::from(format!("{stem}.state.json")),
                None => path,
            }
        };
        let text = fs::read_to_string(&state)
            .map_err(|e| format!("cannot read {}: {e}", state.display()))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| format!("{} is not a saved chat: {e}", state.display()))?;
        let resumed = parse_state(&value, format, max_context)
            .map_err(|e| format!("cannot resume {}: {e}", state.display()))?;
        let name = state.file_name().map(|n| n.to_string_lossy().into_owned());
        let stem = name
            .as_deref()
            .and_then(|n| n.strip_suffix(".state.json"))
            .ok_or_else(|| format!("{} is not a .state.json file", state.display()))?;
        let dir = state.parent().unwrap_or(dir);
        let mut transcript = Self::open(dir, stem, format)?;
        transcript.event(json!({
            "event": "resume",
            "context_tokens": resumed.session.tokens().len(),
            "pending_reply": resumed.session.is_pending(),
            "held_calls": resumed.held.as_ref().map_or(0, |h| h.remaining.len()),
        }));
        Ok((transcript, resumed))
    }

    fn open(dir: &Path, stem: &str, format: &'static str) -> Result<Self, String> {
        let log = dir.join(format!("{stem}.jsonl"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .map_err(|e| format!("cannot open {}: {e}", log.display()))?;
        Ok(Self {
            state: dir.join(format!("{stem}.state.json")),
            log,
            file: Some(file),
            failed: false,
            format,
        })
    }

    #[must_use]
    pub fn log_path(&self) -> &Path {
        &self.log
    }

    fn failed(&mut self, what: &str, error: &dyn std::fmt::Display) {
        if !self.failed {
            eprintln!("transcript: cannot write {what}: {error}; the chat goes on unsaved");
            self.failed = true;
        }
    }

    /// Appends one event, stamped with the local time.
    pub fn event(&mut self, mut value: Value) {
        if let Value::Object(map) = &mut value {
            map.insert("at".into(), chrono::Local::now().to_rfc3339().into());
        }
        let Some(file) = &mut self.file else { return };
        let mut line = value.to_string();
        line.push('\n');
        if let Err(e) = file.write_all(line.as_bytes()) {
            self.file = None;
            let log = self.log.display().to_string();
            self.failed(&log, &e);
        }
    }

    /// A tool result, cut to [`MAX_LOGGED_RESULT`] bytes.
    pub fn tool_result(&mut self, name: &str, text: &str) {
        let cut = (0..=MAX_LOGGED_RESULT.min(text.len()))
            .rev()
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(0);
        self.event(json!({
            "event": "tool_result",
            "name": name,
            "bytes": text.len(),
            "text": &text[..cut],
            "cut": cut < text.len(),
        }));
    }

    /// Rewrites the resume snapshot (a temporary file renamed over the old one, so a crash
    /// mid-write leaves the previous snapshot).
    pub fn save_state(&mut self, session: &Session, reply_start: usize, held: Option<&Held>) {
        let value = json!({
            "version": STATE_VERSION,
            "format": self.format,
            "max_context": session.max_context(),
            "tokens": session.tokens(),
            "turns": session.turn_starts(),
            "pending": session.is_pending(),
            "reply_start": reply_start,
            "held": held.map(|h| json!({
                "done": h.done.iter().map(|(id, name, text)| json!([id, name, text])).collect::<Vec<_>>(),
                "remaining": h.remaining.iter().map(|(id, arguments)| json!([id, arguments])).collect::<Vec<_>>(),
            })),
        });
        let temporary = self.state.with_extension("json.tmp");
        let written = fs::write(&temporary, value.to_string())
            .and_then(|()| fs::rename(&temporary, &self.state));
        if let Err(e) = written {
            let state = self.state.display().to_string();
            self.failed(&state, &e);
        }
    }
}

fn strings<const N: usize>(value: &Value) -> Option<[String; N]> {
    let items = value.as_array()?;
    if items.len() != N {
        return None;
    }
    let mut out: [String; N] = std::array::from_fn(|_| String::new());
    for (slot, item) in out.iter_mut().zip(items) {
        item.as_str()?.clone_into(slot);
    }
    Some(out)
}

fn parse_state(value: &Value, format: &str, max_context: usize) -> Result<Resumed, String> {
    if value["version"].as_u64() != Some(STATE_VERSION) {
        return Err("unknown snapshot version".into());
    }
    let saved_format = value["format"].as_str().unwrap_or("?");
    if saved_format != format {
        return Err(format!(
            "it was saved by the {saved_format} chat, not {format}"
        ));
    }
    let numbers = |key: &str| -> Result<Vec<u64>, String> {
        value[key]
            .as_array()
            .ok_or(format!("no {key}"))?
            .iter()
            .map(|n| n.as_u64().ok_or(format!("bad {key}")))
            .collect()
    };
    let tokens = numbers("tokens")?
        .into_iter()
        .map(|n| u32::try_from(n).map_err(|_| "bad token".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let turns = numbers("turns")?
        .into_iter()
        .map(|n| usize::try_from(n).map_err(|_| "bad turn".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.len() > max_context {
        return Err(format!(
            "it holds {} tokens; pass --max-context {} or more",
            tokens.len(),
            tokens.len()
        ));
    }
    let pending = value["pending"].as_bool().unwrap_or(false);
    let reply_start = value["reply_start"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n <= tokens.len())
        .unwrap_or(tokens.len());
    let held = match &value["held"] {
        Value::Null => None,
        held => Some(Held {
            done: held["done"]
                .as_array()
                .ok_or("bad held calls")?
                .iter()
                .map(|v| strings::<3>(v).map(|[a, b, c]| (a, b, c)))
                .collect::<Option<_>>()
                .ok_or("bad held results")?,
            remaining: held["remaining"]
                .as_array()
                .ok_or("bad held calls")?
                .iter()
                .map(|v| strings::<2>(v).map(|[a, b]| (a, b)))
                .collect::<Option<_>>()
                .ok_or("bad held calls")?,
        }),
    };
    let session = Session::restore(max_context, tokens, turns, pending)
        .map_err(|_| "the token history is inconsistent".to_string())?;
    Ok(Resumed {
        session,
        reply_start,
        held,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_saved_chat_resumes_with_its_tokens_turns_and_held_calls() {
        let dir = tempfile::tempdir().unwrap();
        let mut transcript =
            Transcript::create(dir.path(), "kimi-linear", Path::new("/model")).unwrap();
        let mut session = Session::new(64).unwrap();
        session.begin_turn(&[1, 2, 3]).unwrap();
        let held = Held {
            done: vec![(
                "functions.fs_read:0".into(),
                "fs_read".into(),
                "text".into(),
            )],
            remaining: vec![("functions.fs_list:1".into(), "{\"path\":\".\"}".into())],
        };
        transcript.save_state(&session, 3, Some(&held));
        transcript.tool_result("fs_read", &"é".repeat(MAX_LOGGED_RESULT));
        let log = fs::read_to_string(transcript.log_path()).unwrap();
        let last: Value = serde_json::from_str(log.lines().last().unwrap()).unwrap();
        assert_eq!(last["cut"], true);
        assert!(last["text"].as_str().unwrap().len() <= MAX_LOGGED_RESULT);

        let (_, resumed) = Transcript::resume(dir.path(), "latest", "kimi-linear", 64).unwrap();
        assert_eq!(resumed.session.tokens(), [1, 2, 3]);
        assert_eq!(resumed.session.turn_starts(), [0]);
        assert!(resumed.session.is_pending());
        assert_eq!(resumed.reply_start, 3);
        let resumed_held = resumed.held.unwrap();
        assert_eq!(resumed_held.done, held.done);
        assert_eq!(resumed_held.remaining, held.remaining);

        let err = Transcript::resume(dir.path(), "latest", "k3", 64)
            .err()
            .unwrap();
        assert!(err.contains("kimi-linear"), "{err}");
        let err = Transcript::resume(dir.path(), "latest", "kimi-linear", 2)
            .err()
            .unwrap();
        assert!(err.contains("--max-context 3"), "{err}");
        let log = fs::read_to_string(transcript.log_path()).unwrap();
        assert!(log.lines().any(|l| l.contains("\"event\":\"resume\"")));
    }
}
