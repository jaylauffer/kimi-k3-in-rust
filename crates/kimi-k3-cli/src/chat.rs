//! Text-only chat adapters. K3: XTML, format reference checkpoint `encoding_k3.py` at
//! f831ab66814297da540d832a5235f8e904f29d06. Kimi Linear: the `<|im_*|>` format of
//! that checkpoint's `chat_template.jinja` at e1df551a447157d4658b573f9a695d57658590e9.
//! Model-independent state lives in loadngo-inference. Generated tokens are preserved
//! verbatim, including K3's thinking.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_inference::{Session, StopReason, Utf8Stream, tools::Toolbox};

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
    system: u32,
    user: u32,
    assistant: u32,
    middle: u32,
    end: u32,
    section_begin: u32,
    section_end: u32,
    call_begin: u32,
    argument_begin: u32,
    call_end: u32,
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
            system: single(tokenizer, "<|im_system|>")?,
            section_begin: single(tokenizer, "<|tool_calls_section_begin|>")?,
            section_end: single(tokenizer, "<|tool_calls_section_end|>")?,
            call_begin: single(tokenizer, "<|tool_call_begin|>")?,
            argument_begin: single(tokenizer, "<|tool_call_argument_begin|>")?,
            call_end: single(tokenizer, "<|tool_call_end|>")?,
            user: single(tokenizer, "<|im_user|>")?,
            assistant: single(tokenizer, "<|im_assistant|>")?,
            middle: single(tokenizer, "<|im_middle|>")?,
            end: single(tokenizer, "<|im_end|>")?,
            eos: [eos, single(tokenizer, "[EOS]").unwrap_or(eos)],
        }))
    }

    fn prompt(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        first: bool,
        tools: Option<&Toolbox>,
    ) -> Vec<u32> {
        match self {
            Self::K3 => prompt(tokenizer, text, first),
            // The template renders role names and content as plain text between control
            // tokens, so each is its own ordinary segment. With tools, the conversation
            // opens with the template's `tool_declare` message and a short system note.
            Self::KimiLinear(t) => {
                let mut ids = Vec::new();
                if let Some(tools) = tools.filter(|tools| first && !tools.is_empty()) {
                    t.message(&mut ids, tokenizer, "tool_declare", &tools.declaration());
                    t.message(&mut ids, tokenizer, "system", TOOL_GUIDANCE);
                }
                ids.push(t.user);
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

    /// Tool calls in a finished Kimi Linear reply, as `(id, arguments)` text pairs.
    fn tool_calls(&self, tokenizer: &Tokenizer, reply: &[u32]) -> Vec<(String, String)> {
        let Self::KimiLinear(t) = self else {
            return Vec::new();
        };
        let Some(begin) = reply.iter().position(|&x| x == t.section_begin) else {
            return Vec::new();
        };
        let section = &reply[begin + 1..];
        let section = &section[..section
            .iter()
            .position(|&x| x == t.section_end)
            .unwrap_or(section.len())];
        let text = |ids: &[u32]| {
            String::from_utf8_lossy(&tokenizer.decode(ids))
                .trim()
                .to_string()
        };
        let mut calls = Vec::new();
        let mut rest = section;
        while let Some(start) = rest.iter().position(|&x| x == t.call_begin) {
            rest = &rest[start + 1..];
            let Some(arg) = rest.iter().position(|&x| x == t.argument_begin) else {
                break;
            };
            let end = rest
                .iter()
                .position(|&x| x == t.call_end)
                .unwrap_or(rest.len());
            if end < arg {
                break;
            }
            calls.push((text(&rest[..arg]), text(&rest[arg + 1..end])));
            rest = &rest[end..];
        }
        calls
    }

    /// Tool results as the template's tool messages, then the assistant header.
    fn tool_results(
        &self,
        tokenizer: &Tokenizer,
        results: &[(String, String, String)],
    ) -> Vec<u32> {
        let Self::KimiLinear(t) = self else {
            return Vec::new();
        };
        let mut ids = Vec::new();
        for (id, name, result) in results {
            t.message(
                &mut ids,
                tokenizer,
                name,
                &format!("## Return of {id}\n{result}"),
            );
        }
        ids.push(t.assistant);
        ordinary(&mut ids, tokenizer, "assistant");
        ids.push(t.middle);
        ids
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
                if token == t.section_begin {
                    return terminal_text(&display.utf8.finish());
                }
                if token == t.call_begin {
                    return format!("{}\n[tool call ", terminal_text(&display.utf8.finish()));
                }
                if token == t.argument_begin {
                    return format!("{} ", terminal_text(&display.utf8.finish()));
                }
                if token == t.call_end {
                    return format!("{}]", terminal_text(&display.utf8.finish()));
                }
                if [
                    t.system,
                    t.user,
                    t.assistant,
                    t.middle,
                    t.end,
                    t.eos[0],
                    t.eos[1],
                    t.section_end,
                ]
                .contains(&token)
                {
                    terminal_text(&display.utf8.finish())
                } else {
                    terminal_text(&display.utf8.push(&tokenizer.decode(&[token])))
                }
            }
        }
    }
}

const TOOL_GUIDANCE: &str = "You are Kimi, running locally on Jay's Mac mini. You can read \
files with tools: fs_list, fs_read, fs_find and fs_grep read the local drive (read-only); \
cas_list, cas_find, cas_read and cas_grep read the signed loadngo CAS snapshot of the pudding \
workspace, where every file is verified against its signed root. When a question depends on a \
file's contents, read it before answering and name the path you read.";

/// Most tool rounds (calls, results, continued reply) after one user message.
const MAX_TOOL_ROUNDS: usize = 8;

impl LinearTokens {
    /// `<|im_system|>role<|im_middle|>content<|im_end|>`, each text its own segment.
    fn message(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer, role: &str, content: &str) {
        ids.push(self.system);
        ordinary(ids, tokenizer, role);
        ids.push(self.middle);
        ordinary(ids, tokenizer, content);
        ids.push(self.end);
    }
}

/// The tool name in a Kimi call id such as `functions.fs_read:0`.
fn tool_name(id: &str) -> &str {
    let id = id.strip_prefix("functions.").unwrap_or(id);
    id.split(':').next().unwrap_or(id)
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
        None,
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
    tools: Option<&Toolbox>,
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
                if let Err(error) = session.begin_turn(&format.prompt(
                    tokenizer,
                    text,
                    session.tokens().is_empty(),
                    tools,
                )) {
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
        for round in 0..=MAX_TOOL_ROUNDS {
            let reply_start = session.tokens().len();
            generating.store(true, Ordering::Relaxed);
            let started = Instant::now();
            let stops = format.stops();
            let result = session.generate(max_tokens, &stops, cancel, &mut next, |token| {
                write!(output, "{}", format.push(&mut display, tokenizer, token))
                    .and_then(|()| output.flush())
                    .map_err(|e| e.to_string())
            });
            generating.store(false, Ordering::Relaxed);
            let done = match result {
                Ok(done) => done,
                Err(loadngo_inference::Error::Output(error)) => return Err(error),
                Err(error) => {
                    writeln!(
                        output,
                        "\n{error}; /continue retries, /undo discards this turn."
                    )
                    .map_err(|e| e.to_string())?;
                    break;
                }
            };
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
                break;
            }
            let Some(tools) = tools else { break };
            let calls = format.tool_calls(tokenizer, &session.tokens()[reply_start..]);
            if calls.is_empty() {
                break;
            }
            if round == MAX_TOOL_ROUNDS {
                writeln!(
                    output,
                    "[tool limit: {MAX_TOOL_ROUNDS} rounds; ask a narrower question]"
                )
                .map_err(|e| e.to_string())?;
                break;
            }
            let mut results = Vec::new();
            for (id, arguments) in calls {
                let name = tool_name(&id).to_string();
                let text = match tools.call(&name, &arguments) {
                    Ok(text) => text,
                    Err(error) => format!("error: {error}"),
                };
                writeln!(output, "[tool result {name}: {} bytes]", text.len())
                    .map_err(|e| e.to_string())?;
                results.push((id, name, text));
            }
            // Fit the results into what is left of the context, keeping room to answer.
            let mut prompt = format.tool_results(tokenizer, &results);
            let room = max_context.saturating_sub(session.tokens().len() + max_tokens.min(512));
            while prompt.len() > room && results.iter().any(|(_, _, t)| t.len() > 200) {
                for (_, _, text) in results.iter_mut().filter(|(_, _, t)| t.len() > 200) {
                    let keep = text.len() / 2;
                    let cut = (0..=keep)
                        .rev()
                        .find(|&i| text.is_char_boundary(i))
                        .unwrap_or(0);
                    text.truncate(cut);
                    text.push_str("\n[truncated to fit the context]");
                }
                prompt = format.tool_results(tokenizer, &results);
            }
            if let Err(error) = session.begin_turn(&prompt) {
                writeln!(
                    output,
                    "{error}: tool results do not fit; /reset to start over"
                )
                .map_err(|e| e.to_string())?;
                break;
            }
            display = Display::default();
            write!(output, "{}", format.opening())
                .and_then(|()| output.flush())
                .map_err(|e| e.to_string())?;
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
    fn kimi_call_ids_name_their_tool() {
        assert_eq!(tool_name("functions.fs_read:0"), "fs_read");
        assert_eq!(tool_name("functions.cas_grep:12"), "cas_grep");
        assert_eq!(tool_name("fs_list"), "fs_list");
    }

    // Uses only the downloaded Kimi Linear tokenizer files; no weights are read.
    #[test]
    #[ignore = "requires KIMI_LINEAR_CHECKPOINT tokenizer files"]
    fn kimi_linear_tool_calls_parse_from_real_tokens() {
        let dir = std::env::var("KIMI_LINEAR_CHECKPOINT").expect("set checkpoint directory");
        let tokenizer = Tokenizer::load(dir).unwrap();
        let format = ChatFormat::kimi_linear(&tokenizer, 163_586).unwrap();
        let reply = tokenizer.encode(
            "Let me look.<|tool_calls_section_begin|><|tool_call_begin|>functions.fs_read:0\
             <|tool_call_argument_begin|>{\"path\": \"README.md\"}<|tool_call_end|>\
             <|tool_call_begin|>functions.cas_find:1<|tool_call_argument_begin|>{\"pattern\": \"**/*.rs\"}\
             <|tool_call_end|><|tool_calls_section_end|><|im_end|>",
        );
        let calls = format.tool_calls(&tokenizer, &reply);
        assert_eq!(
            calls,
            [
                (
                    "functions.fs_read:0".to_string(),
                    r#"{"path": "README.md"}"#.to_string()
                ),
                (
                    "functions.cas_find:1".to_string(),
                    r#"{"pattern": "**/*.rs"}"#.to_string()
                ),
            ]
        );
        let results = format.tool_results(
            &tokenizer,
            &[(
                "functions.fs_read:0".into(),
                "fs_read".into(),
                "hello".into(),
            )],
        );
        assert!(
            tokenizer
                .decode_lossy(&results)
                .contains("## Return of functions.fs_read:0")
        );
        assert!(
            format
                .tool_calls(&tokenizer, &tokenizer.encode("no calls<|im_end|>"))
                .is_empty()
        );
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
