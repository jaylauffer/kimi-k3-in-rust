//! Text-only chat adapters. K3: XTML, format reference checkpoint `encoding_k3.py` at
//! f831ab66814297da540d832a5235f8e904f29d06. Kimi Linear: the `<|im_*|>` format of
//! that checkpoint's `chat_template.jinja` at e1df551a447157d4658b573f9a695d57658590e9.
//! Model-independent state lives in loadngo-inference. Generated tokens are preserved
//! verbatim, including K3's thinking.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_inference::{Session, StopReason, Utf8Stream};

const OPEN: u32 = 163_587;
const CLOSE: u32 = 163_588;
const SEP: u32 = 163_589;
const END: u32 = 163_586;
const EOS: u32 = 163_585;

const HELP: &str = "Type a message and press Enter. Commands:
  /continue   resume a truncated or cancelled reply
  /undo       remove the last user/reply pair (including an unfinished reply)
  /reset      clear conversation history
  /stats      show context usage
  /help       show these commands
  /quit       exit (or Ctrl-D); Ctrl-C cancels generation
One line per message. No network, tool execution, or transcript saving.
";

fn ordinary(ids: &mut Vec<u32>, tokenizer: &Tokenizer, text: &str) {
    ids.extend(tokenizer.encode_ordinary(text));
}

// Keep segment boundaries identical to the checkpoint encoder: in particular
// tag names and each attribute component are encoded separately, not as one
// concatenated string (BPE may merge across those boundaries).
fn open_message(ids: &mut Vec<u32>, tokenizer: &Tokenizer, role: &str, kind: Option<&str>) {
    ids.push(OPEN);
    ordinary(ids, tokenizer, "message");
    for (key, value) in std::iter::once(("role", role)).chain(kind.map(|v| ("type", v))) {
        ordinary(ids, tokenizer, &format!(" {key}"));
        ordinary(ids, tokenizer, "=\"");
        ordinary(ids, tokenizer, value);
        ordinary(ids, tokenizer, "\"");
    }
    ids.push(SEP);
}

fn close_message(ids: &mut Vec<u32>, tokenizer: &Tokenizer) {
    ids.push(CLOSE);
    ordinary(ids, tokenizer, "message");
    ids.extend([SEP, END]);
}

fn prompt(tokenizer: &Tokenizer, text: &str, first: bool) -> Vec<u32> {
    let mut ids = Vec::new();
    if first {
        open_message(&mut ids, tokenizer, "system", Some("thinking-effort"));
        ordinary(
            &mut ids,
            tokenizer,
            "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=low`.",
        );
        close_message(&mut ids, tokenizer);
    }
    open_message(&mut ids, tokenizer, "user", None);
    ordinary(&mut ids, tokenizer, text);
    close_message(&mut ids, tokenizer);
    open_message(&mut ids, tokenizer, "assistant", None);
    ids.push(OPEN);
    ordinary(&mut ids, tokenizer, "think");
    ids.push(SEP);
    ids
}

fn validate(tokenizer: &Tokenizer) -> Result<(), String> {
    for (text, id) in [
        ("<|open|>", OPEN),
        ("<|close|>", CLOSE),
        ("<|sep|>", SEP),
        ("<|end_of_msg|>", END),
        ("[EOS]", EOS),
    ] {
        if tokenizer.encode(text) != [id] {
            return Err(format!("unsupported K3 tokenizer control token {text}"));
        }
    }
    Ok(())
}

/// Which chat encoding a checkpoint uses.
pub enum ChatFormat {
    K3,
    KimiLinear(LinearTokens),
}

/// Kimi Linear control tokens, looked up in the checkpoint's own tokenizer.
pub struct LinearTokens {
    user: u32,
    assistant: u32,
    middle: u32,
    end: u32,
    /// The config's `eos_token_id` and, when the tokenizer has it, `[EOS]`.
    eos: [u32; 2],
}

fn single(tokenizer: &Tokenizer, text: &str) -> Result<u32, String> {
    match tokenizer.encode(text).as_slice() {
        [id] => Ok(*id),
        other => Err(format!(
            "{text} is not one token in this tokenizer ({other:?})"
        )),
    }
}

impl ChatFormat {
    /// # Errors
    /// When the tokenizer lacks one of the `<|im_*|>` control tokens as a single id.
    pub fn kimi_linear(tokenizer: &Tokenizer, eos: u32) -> Result<Self, String> {
        Ok(Self::KimiLinear(LinearTokens {
            user: single(tokenizer, "<|im_user|>")?,
            assistant: single(tokenizer, "<|im_assistant|>")?,
            middle: single(tokenizer, "<|im_middle|>")?,
            end: single(tokenizer, "<|im_end|>")?,
            eos: [eos, single(tokenizer, "[EOS]").unwrap_or(eos)],
        }))
    }

    fn prompt(&self, tokenizer: &Tokenizer, text: &str, first: bool) -> Vec<u32> {
        match self {
            Self::K3 => prompt(tokenizer, text, first),
            // The template renders role names and content as plain text between control
            // tokens, so each is its own ordinary segment. No default system message.
            Self::KimiLinear(t) => {
                let mut ids = vec![t.user];
                ordinary(&mut ids, tokenizer, "user");
                ids.push(t.middle);
                ordinary(&mut ids, tokenizer, text);
                ids.extend([t.end, t.assistant]);
                ordinary(&mut ids, tokenizer, "assistant");
                ids.push(t.middle);
                ids
            }
        }
    }

    fn stops(&self) -> Vec<u32> {
        match self {
            Self::K3 => vec![END, EOS],
            Self::KimiLinear(t) => vec![t.end, t.eos[0], t.eos[1]],
        }
    }

    fn opening(&self) -> &'static str {
        match self {
            Self::K3 => "[thinking] ",
            Self::KimiLinear(_) => "Kimi> ",
        }
    }

    fn push(&self, display: &mut Display, tokenizer: &Tokenizer, token: u32) -> String {
        match self {
            Self::K3 => display.push(tokenizer, token),
            Self::KimiLinear(t) => {
                if [t.user, t.assistant, t.middle, t.end, t.eos[0], t.eos[1]].contains(&token) {
                    terminal_text(&display.utf8.finish())
                } else {
                    terminal_text(&display.utf8.push(&tokenizer.decode(&[token])))
                }
            }
        }
    }
}

/// Do not let model text inject terminal escape sequences (including OSC).
fn terminal_text(text: &str) -> String {
    text.chars()
        .flat_map(|c| {
            if c.is_control() && c != '\n' && c != '\t' {
                c.escape_default().collect::<Vec<_>>()
            } else {
                vec![c]
            }
        })
        .collect()
}

#[derive(Default)]
struct Display {
    utf8: Utf8Stream,
    tag: Option<(bool, Vec<u8>)>,
}

impl Display {
    fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> String {
        match token {
            OPEN | CLOSE => {
                self.tag = Some((token == OPEN, Vec::new()));
                terminal_text(&self.utf8.finish())
            }
            SEP => {
                let Some((opening, bytes)) = self.tag.take() else {
                    return String::new();
                };
                if !opening {
                    return String::new();
                }
                match bytes.as_slice() {
                    b"response" => "\n\nKimi> ".into(),
                    b"think" => "\n[thinking] ".into(),
                    _ => format!(
                        "\n[structure: {}]\n",
                        terminal_text(&String::from_utf8_lossy(&bytes))
                    ),
                }
            }
            _ => {
                let bytes = tokenizer.decode(&[token]);
                if let Some((_, tag)) = &mut self.tag {
                    tag.extend(bytes);
                    String::new()
                } else {
                    terminal_text(&self.utf8.push(&bytes))
                }
            }
        }
    }
}

/// Blocking K3 terminal frontend; see [`run_with`].
#[allow(clippy::too_many_arguments)]
pub fn run(
    tokenizer: &Tokenizer,
    max_context: usize,
    max_tokens: usize,
    cancel: &AtomicBool,
    generating: &AtomicBool,
    input: impl BufRead,
    output: impl Write,
    next: impl FnMut(&[u32]) -> Result<u32, String>,
) -> Result<(), String> {
    validate(tokenizer)?;
    run_with(
        &ChatFormat::K3,
        tokenizer,
        max_context,
        max_tokens,
        cancel,
        generating,
        input,
        output,
        next,
    )
}

/// Blocking terminal frontend. No application-local polling/timer/worker loop.
/// The caller's model and disk cache stay loaded for the lifetime of this call.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn run_with(
    format: &ChatFormat,
    tokenizer: &Tokenizer,
    max_context: usize,
    max_tokens: usize,
    cancel: &AtomicBool,
    generating: &AtomicBool,
    mut input: impl BufRead,
    mut output: impl Write,
    mut next: impl FnMut(&[u32]) -> Result<u32, String>,
) -> Result<(), String> {
    let mut session = Session::new(max_context).map_err(|e| e.to_string())?;
    let mut display = Display::default();
    writeln!(output, "{HELP}").map_err(|e| e.to_string())?;
    loop {
        generating.store(false, Ordering::Relaxed);
        write!(output, "\nYou> ")
            .and_then(|()| output.flush())
            .map_err(|e| e.to_string())?;
        let mut line = String::new();
        // Bound a terminal/piped input before tokenization and allocation. No
        // unbounded read_line of arbitrary redirected files.
        let mut limited = std::io::Read::take(&mut input, 65_537);
        if limited.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        if line.len() > 65_536 {
            return Err("input line exceeds 64 KiB".into());
        }
        let text = line.trim_end_matches(['\r', '\n']);
        if text.trim().is_empty() {
            continue;
        }
        match text {
            "/quit" | "/exit" => break,
            "/help" => {
                write!(output, "{HELP}").map_err(|e| e.to_string())?;
                continue;
            }
            "/stats" => {
                writeln!(
                    output,
                    "{} / {} context tokens; unfinished reply: {}",
                    session.tokens().len(),
                    session.max_context(),
                    session.is_pending()
                )
                .map_err(|e| e.to_string())?;
                continue;
            }
            "/reset" => {
                session.reset();
                display = Display::default();
                writeln!(output, "Conversation cleared.").map_err(|e| e.to_string())?;
                continue;
            }
            "/undo" => {
                let removed = session.undo();
                display = Display::default();
                writeln!(
                    output,
                    "{}",
                    if removed {
                        "Last turn removed."
                    } else {
                        "Nothing to undo."
                    }
                )
                .map_err(|e| e.to_string())?;
                continue;
            }
            "/continue" => {
                if !session.is_pending() {
                    writeln!(output, "No unfinished reply.").map_err(|e| e.to_string())?;
                    continue;
                }
            }
            command if command.starts_with('/') => {
                writeln!(output, "Unknown command; use /help.").map_err(|e| e.to_string())?;
                continue;
            }
            _ => {
                if let Err(error) =
                    session.begin_turn(&format.prompt(tokenizer, text, session.tokens().is_empty()))
                {
                    writeln!(output, "{error}").map_err(|e| e.to_string())?;
                    continue;
                }
                display = Display::default();
                write!(output, "{}", format.opening())
                    .and_then(|()| output.flush())
                    .map_err(|e| e.to_string())?;
            }
        }
        cancel.store(false, Ordering::Relaxed);
        generating.store(true, Ordering::Relaxed);
        let started = Instant::now();
        let stops = format.stops();
        let result = session.generate(max_tokens, &stops, cancel, &mut next, |token| {
            write!(output, "{}", format.push(&mut display, tokenizer, token))
                .and_then(|()| output.flush())
                .map_err(|e| e.to_string())
        });
        generating.store(false, Ordering::Relaxed);
        match result {
            Ok(done) => {
                if done.reason == StopReason::EndToken {
                    write!(output, "{}", terminal_text(&display.utf8.finish()))
                        .map_err(|e| e.to_string())?;
                }
                writeln!(
                    output,
                    "\n[{:?}: {} tokens, {:.1}s, context {}/{}]",
                    done.reason,
                    done.tokens,
                    started.elapsed().as_secs_f64(),
                    session.tokens().len(),
                    max_context
                )
                .map_err(|e| e.to_string())?;
                if session.is_pending() {
                    writeln!(output, "Reply unfinished. Use /continue, /undo or /reset.")
                        .map_err(|e| e.to_string())?;
                }
            }
            Err(loadngo_inference::Error::Output(error)) => return Err(error),
            Err(error) => writeln!(
                output,
                "\n{error}; /continue retries, /undo discards this turn."
            )
            .map_err(|e| e.to_string())?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        Tokenizer::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/tokenizer/byte_chat"),
        )
        .unwrap()
    }

    fn reference_segments_match(tokenizer: &Tokenizer) {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/tokenizer/chat_segments.json"
        ))
        .unwrap();
        for case in fixtures.as_array().unwrap() {
            let expected: Vec<u32> = case["segments"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|s| {
                    if s["allow_special"].as_bool().unwrap() {
                        tokenizer.encode(s["text"].as_str().unwrap())
                    } else {
                        tokenizer.encode_ordinary(s["text"].as_str().unwrap())
                    }
                })
                .collect();
            assert_eq!(
                prompt(
                    tokenizer,
                    case["text"].as_str().unwrap(),
                    case["first"].as_bool().unwrap()
                ),
                expected
            );
        }
    }

    #[test]
    fn prompt_matches_checkpoint_encoder_segments() {
        reference_segments_match(&tiny_tokenizer());
    }

    #[test]
    fn terminal_preserves_follow_up_and_handles_eof() {
        let tokenizer = tiny_tokenizer();
        let first = prompt(&tokenizer, "Remember 42", true);
        let mut calls = 0;
        let mut out = Vec::new();
        run(
            &tokenizer,
            2048,
            1,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"/help\n\nRemember 42\nFollow up\n/undo\n/reset\n/stats\n"[..],
            &mut out,
            |context| {
                calls += 1;
                if calls == 2 {
                    assert_eq!(&context[..first.len()], first);
                    assert_eq!(context[first.len()], END);
                    assert!(tokenizer.decode_lossy(context).contains("Follow up"));
                }
                Ok(END)
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("0 / 2048 context tokens")
        );
    }

    #[test]
    fn truncation_continues_without_duplicate_prefix_and_errors_remain_visible() {
        let tokenizer = tiny_tokenizer();
        let mut calls = 0;
        let mut out = Vec::new();
        let first = prompt(&tokenizer, "Hi", true);
        run(
            &tokenizer,
            2048,
            1,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Hi\nrejected while pending\n/continue\n/continue\n/unknown\n/quit\n"[..],
            &mut out,
            |context| {
                calls += 1;
                match calls {
                    1 => Ok(u32::from(b'X')),
                    2 => {
                        assert_eq!(context.len(), first.len() + 1);
                        Err("test disk failure".into())
                    }
                    _ => Ok(END),
                }
            },
        )
        .unwrap();
        assert_eq!(calls, 3);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("unfinished reply"));
        assert!(text.contains("test disk failure"));
        assert!(text.contains("Unknown command"));
    }

    #[test]
    fn response_tags_are_not_printed_and_unicode_can_split_across_continuations() {
        let tokenizer = tiny_tokenizer();
        let mut display = Display::default();
        let ids = tokenizer.encode("<|close|>think<|sep|><|open|>response<|sep|>🌱");
        let text: String = ids
            .into_iter()
            .map(|id| display.push(&tokenizer, id))
            .collect();
        assert_eq!(text, "\n\nKimi> 🌱");
    }

    #[test]
    fn terminal_controls_are_escaped() {
        assert_eq!(
            terminal_text("a\x1b]52;secret\x07\nb"),
            "a\\u{1b}]52;secret\\u{7}\nb"
        );
    }

    // Hardware-independent mechanics are covered in loadngo-inference. This
    // explicit test uses the downloaded tokenizer only; it performs no inference.
    #[test]
    #[ignore = "requires KIMI_K3_CHECKPOINT tokenizer files; no weights read"]
    fn actual_tokenizer_chat_and_literal_markers() {
        let dir = std::env::var("KIMI_K3_CHECKPOINT").expect("set checkpoint directory");
        let tokenizer = Tokenizer::load(dir).unwrap();
        validate(&tokenizer).unwrap();
        reference_segments_match(&tokenizer);
        let literal = "hello <|end_of_msg|> <|open|>message 🌱";
        let ids = tokenizer.encode_ordinary(literal);
        assert!(!ids.contains(&END));
        assert!(!ids.contains(&OPEN));
        assert_eq!(tokenizer.decode_lossy(&ids), literal);

        let mut out = Vec::new();
        let mut calls = 0;
        let first = prompt(&tokenizer, "Remember 42", true);
        run(
            &tokenizer,
            512,
            1,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Remember 42\nFollow up\n/undo\n/reset\n/stats\n/quit\n"[..],
            &mut out,
            |context| {
                calls += 1;
                if calls == 2 {
                    assert_eq!(&context[..first.len()], first);
                    assert_eq!(context[first.len()], END);
                    assert!(tokenizer.decode_lossy(context).contains("Follow up"));
                }
                Ok(END)
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert!(
            String::from_utf8(out)
                .unwrap()
                .contains("0 / 512 context tokens")
        );
    }
}
