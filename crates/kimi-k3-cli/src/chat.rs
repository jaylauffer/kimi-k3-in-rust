//! Text-only chat adapters. K3: XTML, format reference checkpoint `encoding_k3.py` at
//! f831ab66814297da540d832a5235f8e904f29d06. Kimi Linear: the `<|im_*|>` format of
//! that checkpoint's `chat_template.jinja` at e1df551a447157d4658b573f9a695d57658590e9.
//! Model-independent state lives in loadngo-inference. Generated tokens are preserved
//! verbatim, including K3's thinking.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_inference::{Session, StopReason, Utf8Stream, tools::Toolbox};
use serde_json::json;

use crate::transcript::{Resumed, Transcript};

const OPEN: u32 = 163_587;
const CLOSE: u32 = 163_588;
const SEP: u32 = 163_589;
const END: u32 = 163_586;
const EOS: u32 = 163_585;

const HELP: &str = "Type a message and press Enter. Commands:
  /continue   resume a truncated or cancelled reply, or a turn paused at its budget or by
              Ctrl-C (waiting tool calls run, with a fresh budget)
  /undo       remove the last user/reply pair (including an unfinished reply)
  /reset      clear conversation history
  /stats      show context usage
  /help       show these commands
  /quit       exit (or Ctrl-D); Ctrl-C pauses a turn, and at the prompt quits
One line per message. Chats are saved to ~/.loadngo/kimi/transcripts (--no-transcript
turns that off); --resume latest picks the last one up. Kimi Linear can read local files and the
signed CAS snapshot, and create/edit claimed workspace .rs, .md and .txt files (--no-tools
disables these).
File edits and terminal command side effects survive /undo and /reset.
Terminal tools run commands, read output, send stdin and stop sessions (no PTY).
Kimi can also search and read public web pages (--no-web turns those tools off;
terminal commands can still use the network).
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
    /// An extra system note for every conversation (for example how to answer aloud).
    note: Option<String>,
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
            note: None,
        }))
    }

    /// Adds a system note that opens every conversation (Kimi Linear only); notes added
    /// later follow earlier ones.
    #[must_use]
    pub fn with_note(mut self, note: &str) -> Self {
        if let Self::KimiLinear(t) = &mut self {
            t.note = Some(match t.note.take() {
                Some(earlier) => format!("{earlier}\n{note}"),
                None => note.to_string(),
            });
        }
        self
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
                let mut ids = if first {
                    self.preamble(tokenizer, tools)
                } else {
                    Vec::new()
                };
                // A fresh clock observation per user turn, outside the cached opening.
                t.message(&mut ids, tokenizer, "system", &current_date_note());
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

    /// What every conversation opens with before the first user message: with tools,
    /// the `tool_declare` message and the tool guidance (empty otherwise). The same for
    /// every conversation, so a session that has consumed it can be reused.
    pub fn preamble(&self, tokenizer: &Tokenizer, tools: Option<&Toolbox>) -> Vec<u32> {
        let mut ids = Vec::new();
        let Self::KimiLinear(t) = self else {
            return ids;
        };
        if let Some(tools) = tools.filter(|t| !t.is_empty()) {
            t.message(&mut ids, tokenizer, "tool_declare", &tools.declaration());
            t.message(&mut ids, tokenizer, "system", TOOL_GUIDANCE);
        }
        if let Some(note) = &t.note {
            t.message(&mut ids, tokenizer, "system", note);
        }
        ids
    }

    /// A Kimi Linear system message, then the start of a user message holding `head`;
    /// [`Self::close_user_message`] finishes it and opens the assistant's reply. The split
    /// lets a caller read one long text once and reuse the session for several endings.
    ///
    /// # Errors
    /// For the K3 format, which has no system role here.
    pub fn open_user_message(
        &self,
        tokenizer: &Tokenizer,
        system: &str,
        head: &str,
    ) -> Result<Vec<u32>, String> {
        let Self::KimiLinear(t) = self else {
            return Err("only the Kimi Linear format splits user messages".into());
        };
        let mut ids = Vec::new();
        t.message(&mut ids, tokenizer, "system", system);
        ids.push(t.user);
        ordinary(&mut ids, tokenizer, "user");
        ids.push(t.middle);
        ordinary(&mut ids, tokenizer, head);
        Ok(ids)
    }

    /// The rest of a user message begun by [`Self::open_user_message`], then the opening
    /// of the assistant's reply.
    ///
    /// # Errors
    /// For the K3 format.
    pub fn close_user_message(
        &self,
        tokenizer: &Tokenizer,
        tail: &str,
    ) -> Result<Vec<u32>, String> {
        let Self::KimiLinear(t) = self else {
            return Err("only the Kimi Linear format splits user messages".into());
        };
        let mut ids = Vec::new();
        ordinary(&mut ids, tokenizer, tail);
        ids.extend([t.end, t.assistant]);
        ordinary(&mut ids, tokenizer, "assistant");
        ids.push(t.middle);
        Ok(ids)
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

    /// The name saved chats record, so one is never resumed with another format's tokens.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::K3 => "k3",
            Self::KimiLinear(_) => "kimi-linear",
        }
    }

    /// Tool results as the template's tool messages.
    fn tool_messages(
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
        ids
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
        let mut ids = self.tool_messages(tokenizer, results);
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

pub const FRESHNESS_GUIDANCE: &str = "Use the current date supplied each turn when researching \
present-day conditions. For current costs, visas, jobs or news, start with a date-neutral \
query or the current year; do not insert a past year unless Jay asks for that period. \
Historical questions keep their requested dates. Check publication and event dates, prefer \
official sources for visa rules, and distinguish old information from verified current facts. \
Search snippets are leads; read the source before relying on its details.";

fn current_date_note() -> String {
    format!(
        "Current local date: {}. Use this date for current research, not a year remembered from training.",
        chrono::Local::now().format("%Y-%m-%d (%Z, UTC%:z)")
    )
}

const TOOL_GUIDANCE: &str = "You are Kimi, running locally on Jay's Mac mini. You can read \
files with tools: fs_list, fs_read, fs_find and fs_grep read the local drive (read-only); \
cas_archives lists every loadngo Archive CAS archive on the attached drives (the same archives \
the Archive CAS browser shows, each marked signed or unsigned), and cas_list, cas_find, cas_read \
and cas_grep read one of them by name, verifying every file against its manifest. When a question depends on a \
file's contents, read it before answering and name the path you read. web_search searches the \
public web and web_fetch reads a page: use them for current events, prices, schedules and \
anything recent or that you are unsure of, and say which site the answer came from. Answer \
everything else from your own knowledge. Before changing files, read applicable AGENTS.md \
and COLLABORATION.md and the board. Claim exact repo-relative paths via board_add_row \
(Active claims, exact repo name, comma-separated paths, status in progress). text_read returns \
a revision; text_edit uses that revision and one unique old_text/new_text replacement. \
text_write creates a new file. Only .rs, .md and .txt files in a workspace Git repository \
can be written. Paths start at the workspace. Never adopt another agent's \
dirty files. terminal_exec starts a shell command and returns a session id; terminal_read \
reads output/status, terminal_write sends stdin or closes it, and terminal_stop cancels. \
Use these to inspect repositories and run builds/tests. Read command exit status before \
claiming success; report checks not run. Commands run with your OS user's permissions, \
including filesystem writes and network access: follow Jay's scope and shared-work claims. \
Never push, publish, delete user data, or alter another agent's work without Jay's authorization. \
Prefer text_edit for claimed source changes. File edits and command side effects survive \
/undo and /reset. Finish with a board handoff. Avoid repeating unchanged tool calls; after a \
successful edit you may read the updated file again. When a search finds nothing, say so plainly.";

/// Limits on the work one user message leads to: every reply and tool round, until Kimi
/// answers without calling a tool. Checked after each reply, before its tool calls run, so
/// a reply in progress always finishes (it is bounded by `--gen` on its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurnBudget {
    /// Wall-clock time from the user's message; `None` is unlimited.
    pub time: Option<Duration>,
    /// Tokens Kimi generates across the turn's replies; `None` is unlimited.
    pub tokens: Option<usize>,
}

impl TurnBudget {
    pub const DEFAULT_MINUTES: u64 = 30;
    pub const DEFAULT_TOKENS: usize = 16_384;

    /// Why the turn should pause, once either limit is reached.
    fn spent(&self, elapsed: Duration, tokens: usize) -> Option<String> {
        if let Some(limit) = self.time.filter(|&limit| elapsed >= limit) {
            return Some(format!("time budget of {} min spent", limit.as_secs() / 60));
        }
        let limit = self.tokens.filter(|&limit| tokens >= limit)?;
        Some(format!("token budget of {limit} generated tokens spent"))
    }
}

impl Default for TurnBudget {
    fn default() -> Self {
        Self {
            time: Some(Duration::from_secs(Self::DEFAULT_MINUTES * 60)),
            tokens: Some(Self::DEFAULT_TOKENS),
        }
    }
}

impl std::fmt::Display for TurnBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.time {
            Some(time) => write!(f, "{} min", time.as_secs() / 60)?,
            None => write!(f, "no time limit")?,
        }
        match self.tokens {
            Some(tokens) => write!(f, ", {tokens} generated tokens"),
            None => write!(f, ", no token limit"),
        }
    }
}

/// Tool calls from a finished reply that have not all run: the turn's budget ran out or
/// Jay pressed Ctrl-C. `/continue` runs the rest; a new message answers them as not run.
#[derive(Clone, Debug, Default)]
pub struct Held {
    /// `(call id, tool name, result)` for the calls that ran.
    pub(crate) done: Vec<(String, String, String)>,
    /// `(call id, arguments)` for the calls still to run.
    pub(crate) remaining: Vec<(String, String)>,
}

const NOT_RUN: &str = "Not run: Jay paused this turn before this call ran and has sent a \
new message instead. Do not assume the call happened.";

impl Held {
    /// Every call's result, the ones not run marked so.
    fn abandoned(&self) -> Vec<(String, String, String)> {
        self.done
            .iter()
            .cloned()
            .chain(
                self.remaining
                    .iter()
                    .map(|(id, _)| (id.clone(), tool_name(id).to_string(), NOT_RUN.to_string())),
            )
            .collect()
    }
}

/// How a chat is budgeted, saved and started.
#[derive(Default)]
pub struct ChatOptions {
    pub budget: TurnBudget,
    /// Where the chat is saved; `None` keeps it in memory only.
    pub transcript: Option<Transcript>,
    /// A saved chat to carry on from.
    pub resumed: Option<Resumed>,
}

/// A reply is ending in a loop once its last tokens are at least this many copies of one
/// block ...
const MIN_REPEAT_COPIES: usize = 4;
/// ... covering at least this many tokens ...
const MIN_REPEAT_TOKENS: usize = 64;
/// ... of a block at most this long.
const MAX_REPEAT_PERIOD: usize = 200;

/// The block length when `reply` ends in [`MIN_REPEAT_COPIES`] or more identical copies of
/// one block, together at least [`MIN_REPEAT_TOKENS`] long. Greedy decoding never leaves
/// such a loop once it is in one, so the reply is ended there.
fn repeating_period(reply: &[u32]) -> Option<usize> {
    (1..=MAX_REPEAT_PERIOD).find(|&period| {
        let span = period * MIN_REPEAT_COPIES.max(MIN_REPEAT_TOKENS.div_ceil(period));
        span <= reply.len() && {
            let tail = &reply[reply.len() - span..];
            tail[period..] == tail[..span - period]
        }
    })
}

/// A tool call's identity: its name and its arguments as JSON, so key order and spacing
/// do not make a repeated call look new.
fn call_key(name: &str, arguments: &str) -> (String, serde_json::Value) {
    let arguments = serde_json::from_str(arguments)
        .unwrap_or_else(|_| serde_json::Value::String(arguments.trim().to_string()));
    (name.to_string(), arguments)
}

// Session reads consume output, and writes/stops can legitimately repeat. Never cache
// them as immutable lookups. Commands can change files asynchronously too.
fn repeatable_tool(name: &str) -> bool {
    matches!(
        name,
        "terminal_exec" | "terminal_read" | "terminal_write" | "terminal_stop"
    )
}

const REPEATED_CALL: &str = "Not run: you already made this exact call in this turn, and its \
result is above. It would return the same thing. Do not call it again. Answer Jay with what you \
have, or tell him plainly what you could not find.";

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
    options: ChatOptions,
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
        options,
    )
}

/// Blocking terminal frontend. No application-local polling/timer/worker loop.
/// The caller's model and disk cache stay loaded for the lifetime of this call.
///
/// One user message may lead to any number of replies and tool rounds. The turn ends
/// when Kimi answers without calling a tool, or pauses when `options.budget` is spent or
/// Jay presses Ctrl-C; `/continue` picks a paused turn up again with a fresh budget.
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
    options: ChatOptions,
) -> Result<(), String> {
    let ChatOptions {
        budget,
        mut transcript,
        resumed,
    } = options;
    let was_resumed = resumed.is_some();
    // `reply_start`: where the latest assistant message begins, so a reply continued
    // after a pause is read for tool calls as a whole. `held`: calls not yet run.
    let (mut session, mut reply_start, mut held) = match resumed {
        Some(saved) => (saved.session, saved.reply_start, saved.held),
        None => (
            Session::new(max_context).map_err(|e| e.to_string())?,
            0,
            None,
        ),
    };
    let mut display = Display::default();
    // The reply being generated, for the repetition check; reused across replies.
    let mut reply = Vec::with_capacity(max_tokens.min(max_context));
    writeln!(output, "{HELP}").map_err(|e| e.to_string())?;
    writeln!(output, "Turn budget: {budget}.").map_err(|e| e.to_string())?;
    if let Some(transcript) = &transcript {
        writeln!(output, "Transcript: {}", transcript.log_path().display())
            .map_err(|e| e.to_string())?;
    }
    if was_resumed {
        writeln!(
            output,
            "Resumed a saved chat: {} context tokens.{}",
            session.tokens().len(),
            if session.is_pending() {
                " The last reply is unfinished: /continue finishes it."
            } else if held.is_some() {
                " Tool calls are waiting: /continue runs them."
            } else {
                ""
            }
        )
        .map_err(|e| e.to_string())?;
    }
    loop {
        generating.store(false, Ordering::Relaxed);
        if let Some(transcript) = &mut transcript {
            transcript.save_state(&session, reply_start, held.as_ref());
        }
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
        let mut run_held = None;
        match text {
            "/quit" | "/exit" => break,
            "/help" => {
                write!(output, "{HELP}").map_err(|e| e.to_string())?;
                continue;
            }
            "/stats" => {
                writeln!(
                    output,
                    "{} / {} context tokens; unfinished reply: {}; waiting tool calls: {}; \
                     turn budget: {budget}",
                    session.tokens().len(),
                    session.max_context(),
                    session.is_pending(),
                    held.as_ref().map_or(0, |h| h.remaining.len()),
                )
                .map_err(|e| e.to_string())?;
                continue;
            }
            "/reset" => {
                session.reset();
                (reply_start, held) = (0, None);
                display = Display::default();
                log(&mut transcript, json!({"event": "command", "text": text}));
                writeln!(output, "Conversation cleared.").map_err(|e| e.to_string())?;
                continue;
            }
            "/undo" => {
                let removed = session.undo();
                (reply_start, held) = (session.tokens().len(), None);
                display = Display::default();
                log(&mut transcript, json!({"event": "command", "text": text}));
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
                    run_held = held.take();
                    if run_held.is_none() {
                        writeln!(output, "Nothing to continue.").map_err(|e| e.to_string())?;
                        continue;
                    }
                }
                log(&mut transcript, json!({"event": "command", "text": text}));
            }
            command if command.starts_with('/') => {
                writeln!(output, "Unknown command; use /help.").map_err(|e| e.to_string())?;
                continue;
            }
            _ => {
                // Calls left waiting are answered as not run, so the history stays a
                // well-formed conversation, then the new message follows.
                let mut ids = held.as_ref().map_or_else(Vec::new, |held| {
                    format.tool_messages(tokenizer, &held.abandoned())
                });
                ids.extend(format.prompt(tokenizer, text, session.tokens().is_empty(), tools));
                if let Err(error) = session.begin_turn(&ids) {
                    writeln!(output, "{error}").map_err(|e| e.to_string())?;
                    continue;
                }
                held = None;
                reply_start = session.tokens().len();
                reply.clear();
                log(&mut transcript, json!({"event": "user", "text": text}));
                display = Display::default();
                write!(output, "{}", format.opening())
                    .and_then(|()| output.flush())
                    .map_err(|e| e.to_string())?;
            }
        }
        cancel.store(false, Ordering::Relaxed);
        // For the whole turn, tool rounds included: Ctrl-C pauses it instead of quitting.
        generating.store(true, Ordering::Relaxed);
        let turn_started = Instant::now();
        let mut turn_tokens = 0;
        let mut rounds = 0;
        let mut earlier_calls = Vec::new();
        let mut repeated_rounds = 0;
        let stop: Option<String> = loop {
            let mut calls = if let Some(calls) = run_held.take() {
                calls
            } else {
                let round_start = session.tokens().len();
                let started = Instant::now();
                let stops = format.stops();
                // Once the reply is looping, the next "token" is the end of the message:
                // the turn closes normally instead of running to the token limit.
                let looping = std::cell::Cell::new(None);
                let result = session.generate(
                    max_tokens,
                    &stops,
                    cancel,
                    |context| match looping.get() {
                        Some(_) => Ok(stops[0]),
                        None => next(context),
                    },
                    |token| {
                        write!(output, "{}", format.push(&mut display, tokenizer, token))
                            .and_then(|()| output.flush())
                            .map_err(|e| e.to_string())?;
                        reply.push(token);
                        if looping.get().is_none() {
                            looping.set(repeating_period(&reply));
                        }
                        Ok(())
                    },
                );
                let done = match result {
                    Ok(done) => done,
                    Err(loadngo_inference::Error::Output(error)) => return Err(error),
                    Err(error) => {
                        writeln!(
                            output,
                            "\n{error}; /continue retries, /undo discards this turn."
                        )
                        .map_err(|e| e.to_string())?;
                        break Some(error.to_string());
                    }
                };
                turn_tokens += done.tokens;
                rounds += 1;
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
                log(
                    &mut transcript,
                    json!({
                        "event": "reply",
                        "text": tokenizer.decode_lossy(&session.tokens()[round_start..]),
                        "tokens": done.tokens,
                        "stop": format!("{:?}", done.reason),
                        "seconds": started.elapsed().as_secs_f64(),
                        "context": session.tokens().len(),
                    }),
                );
                if session.is_pending() {
                    let why = if done.reason == StopReason::ContextLimit {
                        "Context full. Use /undo or /reset."
                    } else {
                        "Reply unfinished. Use /continue, /undo or /reset."
                    };
                    writeln!(output, "{why}").map_err(|e| e.to_string())?;
                    break Some(format!("{:?}", done.reason));
                }
                if let Some(period) = looping.get() {
                    writeln!(
                        output,
                        "[stopped: the reply was repeating a {period}-token block; \
                         /undo removes this turn]"
                    )
                    .map_err(|e| e.to_string())?;
                    break Some("repeating reply".into());
                }
                if tools.is_none() {
                    break None;
                }
                let calls = format.tool_calls(tokenizer, &session.tokens()[reply_start..]);
                if calls.is_empty() {
                    break None;
                }
                if calls.iter().all(|(id, arguments)| {
                    let key = call_key(tool_name(id), arguments);
                    !repeatable_tool(&key.0) && earlier_calls.contains(&key)
                }) {
                    repeated_rounds += 1;
                    if repeated_rounds == 2 {
                        writeln!(
                            output,
                            "[stopped: Kimi repeated the same tool call; ask differently or /reset]"
                        )
                        .map_err(|e| e.to_string())?;
                        break Some("repeated tool call".into());
                    }
                }
                Held {
                    done: Vec::new(),
                    remaining: calls,
                }
            };
            let Some(tools) = tools else { break None };
            if let Some(spent) = budget.spent(turn_started.elapsed(), turn_tokens) {
                writeln!(
                    output,
                    "[paused: {spent}. /continue runs the {} waiting tool call(s) with a \
                     fresh budget; a new message goes on without them]",
                    calls.remaining.len()
                )
                .map_err(|e| e.to_string())?;
                held = Some(calls);
                break Some(spent);
            }
            while !calls.remaining.is_empty() && !cancel.load(Ordering::Relaxed) {
                let (id, arguments) = calls.remaining.remove(0);
                let name = tool_name(&id).to_string();
                let key = call_key(&name, &arguments);
                log(
                    &mut transcript,
                    json!({"event": "tool_call", "name": name, "arguments": arguments}),
                );
                if !repeatable_tool(&name) && earlier_calls.contains(&key) {
                    writeln!(output, "[tool call repeated: {name}; not run again]")
                        .map_err(|e| e.to_string())?;
                    if let Some(transcript) = &mut transcript {
                        transcript.tool_result(&name, REPEATED_CALL);
                    }
                    calls.done.push((id, name, REPEATED_CALL.to_string()));
                    continue;
                }
                let text = match tools.call(&name, &arguments) {
                    Ok(text) => {
                        if matches!(
                            name.as_str(),
                            "text_write"
                                | "text_edit"
                                | "terminal_exec"
                                | "terminal_write"
                                | "terminal_read"
                                | "terminal_stop"
                        ) {
                            // Files changed: previous reads/finds may now have different
                            // results. Keep text mutation keys so an edit is never replayed.
                            earlier_calls.retain(|(tool, _)| {
                                !matches!(
                                    tool.as_str(),
                                    "text_read"
                                        | "fs_read"
                                        | "fs_list"
                                        | "fs_find"
                                        | "fs_grep"
                                        | "terminal_exec"
                                )
                            });
                            repeated_rounds = 0;
                        }
                        text
                    }
                    Err(error) => format!("error: {error}"),
                };
                writeln!(output, "[tool result {name}: {} bytes]", text.len())
                    .map_err(|e| e.to_string())?;
                if let Some(transcript) = &mut transcript {
                    transcript.tool_result(&name, &text);
                }
                calls.done.push((id, name, text));
                earlier_calls.push(key);
            }
            if !calls.remaining.is_empty() {
                writeln!(
                    output,
                    "[paused by Ctrl-C: {} tool call(s) not run. /continue runs them; a new \
                     message goes on without them]",
                    calls.remaining.len()
                )
                .map_err(|e| e.to_string())?;
                held = Some(calls);
                break Some("interrupted".into());
            }
            // Fit the results into what is left of the context, keeping room to answer.
            let mut results = calls.done;
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
                break Some("context full".into());
            }
            reply_start = session.tokens().len();
            reply.clear();
            if let Some(transcript) = &mut transcript {
                transcript.save_state(&session, reply_start, None);
            }
            display = Display::default();
            write!(output, "{}", format.opening())
                .and_then(|()| output.flush())
                .map_err(|e| e.to_string())?;
        };
        log(
            &mut transcript,
            json!({
                "event": "turn_end",
                "stop": stop.as_deref().unwrap_or("answered"),
                "replies": rounds,
                "tokens": turn_tokens,
                "seconds": turn_started.elapsed().as_secs_f64(),
            }),
        );
    }
    generating.store(false, Ordering::Relaxed);
    if let Some(transcript) = &mut transcript {
        transcript.save_state(&session, reply_start, held.as_ref());
        transcript.event(json!({"event": "exit"}));
    }
    Ok(())
}

fn log(transcript: &mut Option<Transcript>, event: serde_json::Value) {
    if let Some(transcript) = transcript {
        transcript.event(event);
    }
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

    // Distinct synthetic controls let the byte fixture exercise Linear chat
    // mechanics in CI without downloading checkpoint tokenizer files.
    fn tiny_linear_format() -> ChatFormat {
        ChatFormat::KimiLinear(LinearTokens {
            system: 200_000,
            user: 200_001,
            assistant: 200_002,
            middle: 200_003,
            end: 200_004,
            section_begin: 200_005,
            section_end: 200_006,
            call_begin: 200_007,
            argument_begin: 200_008,
            call_end: 200_009,
            eos: [200_004; 2],
            note: None,
        })
    }

    #[test]
    fn current_date_is_in_each_user_turn_outside_cached_preamble() {
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format().with_note(FRESHNESS_GUIDANCE);
        let preamble = format.preamble(&tokenizer, None);
        assert!(
            !tokenizer
                .decode_lossy(&preamble)
                .contains("Current local date:")
        );
        for first in [true, false] {
            let before = current_date_note();
            let prompt = format.prompt(&tokenizer, "research", first, None);
            let after = current_date_note();
            let text = tokenizer.decode_lossy(&prompt);
            assert!(text.contains(&before) || text.contains(&after));
            assert_eq!(text.matches("Current local date:").count(), 1);
            assert!(text.find("Current local date:").unwrap() < text.find("userresearch").unwrap());
            assert_eq!(prompt.starts_with(&preamble), first);
        }
    }

    #[test]
    fn terminal_rounds_repeat_and_refresh_file_reads_in_chat() {
        use loadngo_inference::tools::Tool;
        use std::cell::RefCell;
        use std::rc::Rc;
        struct Recorded(&'static str, Rc<RefCell<Vec<&'static str>>>);
        impl Tool for Recorded {
            fn name(&self) -> &'static str {
                self.0
            }
            fn description(&self) -> &'static str {
                "test"
            }
            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type":"object"})
            }
            fn call(&self, _: &serde_json::Value) -> Result<String, String> {
                self.1.borrow_mut().push(self.0);
                Ok("recorded".into())
            }
        }
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let recorded = Rc::new(RefCell::new(Vec::new()));
        let mut tools = Toolbox::default();
        for name in [
            "terminal_exec",
            "terminal_read",
            "terminal_write",
            "terminal_stop",
            "fs_read",
        ] {
            tools.push(Box::new(Recorded(name, Rc::clone(&recorded))));
        }
        let expected = [
            "terminal_exec",
            "terminal_exec",
            "fs_read",
            "terminal_read",
            "terminal_read",
            "fs_read",
            "terminal_write",
            "terminal_write",
            "terminal_stop",
            "terminal_stop",
        ];
        let mut replies: Vec<Vec<u32>> = expected
            .iter()
            .enumerate()
            .map(|(n, name)| {
                let mut ids = vec![t.section_begin, t.call_begin];
                ordinary(&mut ids, &tokenizer, &format!("functions.{name}:{n}"));
                ids.push(t.argument_begin);
                ordinary(&mut ids, &tokenizer, "{}");
                ids.extend([t.call_end, t.section_end, t.end]);
                ids
            })
            .collect();
        replies.insert(8, vec![t.end]); // The first message's turn ends here.
        replies.push(vec![t.end]);
        let mut replies = replies.into_iter().flatten();
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Run checks\nStop the session\n/quit\n"[..],
            &mut output,
            |_| Ok(replies.next().expect("scripted reply")),
            ChatOptions::default(),
        )
        .unwrap();
        assert_eq!(*recorded.borrow(), expected);
        assert!(
            !String::from_utf8(output)
                .unwrap()
                .contains("tool call repeated")
        );
    }

    /// Records each call's arguments; with `cancel`, presses Ctrl-C during its first call.
    struct Step(
        std::rc::Rc<std::cell::RefCell<Vec<String>>>,
        Option<std::sync::Arc<AtomicBool>>,
    );

    impl loadngo_inference::tools::Tool for Step {
        fn name(&self) -> &'static str {
            "fs_read"
        }
        fn description(&self) -> &'static str {
            "test"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        fn call(&self, arguments: &serde_json::Value) -> Result<String, String> {
            if let Some(cancel) = self.1.as_ref().filter(|_| self.0.borrow().is_empty()) {
                cancel.store(true, Ordering::Relaxed);
            }
            self.0.borrow_mut().push(arguments["path"].to_string());
            Ok("read".into())
        }
    }

    /// Scripted replies: `rounds` replies that each call `fs_read` `per_round` times on new
    /// paths, then a plain answer; the model is fed from the returned iterator.
    fn stepping_replies(
        tokenizer: &Tokenizer,
        t: &LinearTokens,
        rounds: usize,
        per_round: usize,
    ) -> std::vec::IntoIter<u32> {
        let mut ids = Vec::new();
        for round in 0..rounds {
            ids.push(t.section_begin);
            for call in 0..per_round {
                ids.push(t.call_begin);
                ordinary(
                    &mut ids,
                    tokenizer,
                    &format!("functions.fs_read:{round}{call}"),
                );
                ids.push(t.argument_begin);
                ordinary(
                    &mut ids,
                    tokenizer,
                    &format!("{{\"path\": \"{round}-{call}\"}}"),
                );
                ids.push(t.call_end);
            }
            ids.extend([t.section_end, t.end]);
        }
        ordinary(&mut ids, tokenizer, "Done.");
        ids.push(t.end);
        ids.into_iter()
    }

    fn stepping_tools(
        cancel: Option<std::sync::Arc<AtomicBool>>,
    ) -> (Toolbox, std::rc::Rc<std::cell::RefCell<Vec<String>>>) {
        let ran = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut tools = Toolbox::default();
        tools.push(Box::new(Step(ran.clone(), cancel)));
        (tools, ran)
    }

    #[test]
    fn a_turn_runs_as_many_tool_rounds_as_the_work_needs() {
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let (tools, ran) = stepping_tools(None);
        let mut replies = stepping_replies(&tokenizer, t, 20, 1);
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Work through it\n/quit\n"[..],
            &mut output,
            |_| Ok(replies.next().expect("scripted reply")),
            ChatOptions::default(),
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(ran.borrow().len(), 20, "{output}");
        assert!(output.contains("Done."));
        assert!(!output.contains("[paused"));
    }

    #[test]
    fn a_spent_budget_pauses_the_turn_and_continue_runs_the_waiting_calls() {
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let (tools, ran) = stepping_tools(None);
        let mut replies = stepping_replies(&tokenizer, t, 3, 2);
        let dir = tempfile::tempdir().unwrap();
        let transcript =
            Transcript::create(dir.path(), "kimi-linear", std::path::Path::new("/m")).unwrap();
        let log = transcript.log_path().to_path_buf();
        let options = ChatOptions {
            // One scripted reply is longer than this, so every round pauses.
            budget: TurnBudget {
                time: None,
                tokens: Some(1),
            },
            transcript: Some(transcript),
            resumed: None,
        };
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Work\n/stats\n/continue\n/continue\n/continue\n/continue\n/quit\n"[..],
            &mut output,
            |_| Ok(replies.next().expect("scripted reply")),
            options,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(
            output
                .matches("[paused: token budget of 1 generated tokens spent")
                .count(),
            3,
            "{output}"
        );
        assert!(output.contains("waiting tool calls: 2"), "{output}");
        assert_eq!(ran.borrow().len(), 6, "{output}");
        assert!(output.contains("Done."));
        assert!(output.contains("Nothing to continue."));
        let log = std::fs::read_to_string(log).unwrap();
        for event in [
            "\"user\"",
            "\"reply\"",
            "\"tool_call\"",
            "\"tool_result\"",
            "\"turn_end\"",
            "\"exit\"",
        ] {
            assert!(
                log.contains(&format!("\"event\":{event}")),
                "{event} missing: {log}"
            );
        }
    }

    #[test]
    fn ctrl_c_during_a_tool_holds_the_rest_and_a_resumed_chat_can_run_them() {
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let (tools, ran) = stepping_tools(Some(cancel.clone()));
        let mut replies = stepping_replies(&tokenizer, t, 1, 3);
        let dir = tempfile::tempdir().unwrap();
        let options = ChatOptions {
            transcript: Some(
                Transcript::create(dir.path(), "kimi-linear", std::path::Path::new("/m")).unwrap(),
            ),
            ..ChatOptions::default()
        };
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &cancel,
            &AtomicBool::new(false),
            &b"Work\n/quit\n"[..],
            &mut output,
            |_| Ok(replies.next().expect("scripted reply")),
            options,
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(ran.borrow().len(), 1, "{output}");
        assert!(
            output.contains("[paused by Ctrl-C: 2 tool call(s) not run"),
            "{output}"
        );

        // A new process: the saved chat, its two waiting calls, then Kimi's answer.
        let (transcript, resumed) =
            Transcript::resume(dir.path(), "latest", "kimi-linear", 32_768).unwrap();
        let history = resumed.session.tokens().to_vec();
        let mut output = Vec::new();
        let mut answer = {
            let mut ids = Vec::new();
            ordinary(&mut ids, &tokenizer, "Done.");
            ids.push(t.end);
            ids.into_iter()
        };
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"/continue\n/quit\n"[..],
            &mut output,
            |context| {
                assert!(context.starts_with(&history));
                Ok(answer.next().expect("scripted reply"))
            },
            ChatOptions {
                transcript: Some(transcript),
                resumed: Some(resumed),
                ..ChatOptions::default()
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("Tool calls are waiting"), "{output}");
        assert_eq!(*ran.borrow(), ["\"0-0\"", "\"0-1\"", "\"0-2\""], "{output}");
        assert!(output.contains("Done."), "{output}");
    }

    #[test]
    fn a_new_message_answers_waiting_calls_as_not_run() {
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let (tools, ran) = stepping_tools(None);
        let mut replies = stepping_replies(&tokenizer, t, 1, 1);
        let mut saw_note = false;
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Work\nNever mind\n/quit\n"[..],
            &mut output,
            |context| {
                let text = tokenizer.decode_lossy(context);
                saw_note |= text.contains(NOT_RUN) && text.ends_with("assistant");
                Ok(replies.next().unwrap_or(t.end))
            },
            ChatOptions {
                budget: TurnBudget {
                    time: None,
                    tokens: Some(1),
                },
                ..ChatOptions::default()
            },
        )
        .unwrap();
        assert!(ran.borrow().is_empty());
        assert!(saw_note, "{}", String::from_utf8(output).unwrap());
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
            ChatOptions::default(),
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
            ChatOptions::default(),
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
    fn repeating_period_finds_loops_but_not_ordinary_repetition() {
        let block: Vec<u32> = (100..130).collect(); // 30 tokens
        let copies = |n: usize| -> Vec<u32> {
            let mut reply = vec![1, 2, 3];
            for _ in 0..n {
                reply.extend(&block);
            }
            reply
        };
        assert_eq!(repeating_period(&copies(4)), Some(30));
        assert_eq!(
            repeating_period(&copies(3)),
            None,
            "three copies can be a list"
        );
        assert_eq!(repeating_period(&vec![7; 64]), Some(1));
        assert_eq!(
            repeating_period(&vec![7; 63]),
            None,
            "a short run is left alone"
        );
        let two: Vec<u32> = [5, 6].repeat(32);
        assert_eq!(repeating_period(&two), Some(2));
        let text: Vec<u32> = (0..1000).collect();
        assert_eq!(repeating_period(&text), None);
        let mut broken = copies(4);
        broken.push(999);
        assert_eq!(repeating_period(&broken), None, "only the tail counts");
    }

    #[test]
    fn a_looping_reply_is_ended_and_the_conversation_goes_on() {
        let tokenizer = tiny_tokenizer();
        let mut calls = 0;
        let mut out = Vec::new();
        run(
            &tokenizer,
            2048,
            500,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Hi\nAgain\n/quit\n"[..],
            &mut out,
            |context| {
                calls += 1;
                if calls <= MIN_REPEAT_TOKENS {
                    return Ok(u32::from(b'a'));
                }
                // The looping reply was closed with the end token, not left pending.
                let a = u32::from(b'a');
                let run = context
                    .windows(MIN_REPEAT_TOKENS)
                    .position(|w| w.iter().all(|&t| t == a))
                    .unwrap();
                let after = run + context[run..].iter().position(|&t| t != a).unwrap();
                assert_eq!(context[after], END);
                Ok(END)
            },
            ChatOptions::default(),
        )
        .unwrap();
        assert_eq!(calls, MIN_REPEAT_TOKENS + 1, "no model call after the loop");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("[stopped: the reply was repeating a 1-token block"));
        assert!(!text.contains("Reply unfinished"));
    }

    #[test]
    fn a_tool_call_is_the_same_whatever_its_json_spacing_or_key_order() {
        assert_eq!(
            call_key("cas_find", r#"{"pattern": "**/*.conf", "limit": 5}"#),
            call_key("cas_find", r#" {"limit":5,"pattern":"**/*.conf"}"#)
        );
        assert_ne!(
            call_key("cas_find", r#"{"pattern": "**/*.conf"}"#),
            call_key("cas_find", r#"{"pattern": "**/*gcp*"}"#)
        );
        assert_ne!(
            call_key("cas_find", r#"{"pattern": "x"}"#),
            call_key("fs_find", r#"{"pattern": "x"}"#)
        );
    }

    struct Counting(std::rc::Rc<std::cell::Cell<usize>>);

    impl loadngo_inference::tools::Tool for Counting {
        fn name(&self) -> &'static str {
            "cas_find"
        }
        fn description(&self) -> &'static str {
            "test"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        fn call(&self, _: &serde_json::Value) -> Result<String, String> {
            self.0.set(self.0.get() + 1);
            Ok("no matches".into())
        }
    }

    // Uses only the downloaded Kimi Linear tokenizer files; no weights are read. The
    // scripted "model" makes the same call every round, as Kimi did on 2026-09-27.
    #[test]
    #[ignore = "requires KIMI_LINEAR_CHECKPOINT tokenizer files"]
    fn a_repeated_tool_call_is_not_run_again_and_a_second_repeat_ends_the_turn() {
        let dir = std::env::var("KIMI_LINEAR_CHECKPOINT").expect("set checkpoint directory");
        let tokenizer = Tokenizer::load(dir).unwrap();
        let format = ChatFormat::kimi_linear(&tokenizer, 163_586).unwrap();
        let ran = std::rc::Rc::new(std::cell::Cell::new(0));
        let mut tools = Toolbox::default();
        tools.push(Box::new(Counting(ran.clone())));
        let reply = |n: usize| {
            tokenizer.encode(&format!(
                "Let me look.<|tool_calls_section_begin|><|tool_call_begin|>functions.cas_find:{n}\
                 <|tool_call_argument_begin|>{{\"pattern\": \"**/*.conf\"}}<|tool_call_end|>\
                 <|tool_calls_section_end|><|im_end|>"
            ))
        };
        let (mut round, mut at, mut saw_note) = (0, 0, false);
        let mut out = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Find wg-gcp.conf\n/quit\n"[..],
            &mut out,
            |context| {
                if at == 0 && round == 2 {
                    saw_note = tokenizer.decode_lossy(context).contains(REPEATED_CALL);
                }
                let tokens = reply(round);
                let token = tokens[at];
                at += 1;
                if at == tokens.len() {
                    (round, at) = (round + 1, 0);
                }
                Ok(token)
            },
            ChatOptions::default(),
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(ran.get(), 1, "the repeat was not run: {text}");
        assert_eq!(round, 3, "the third identical call ended the turn: {text}");
        assert!(text.contains("[tool call repeated: cas_find; not run again]"));
        assert!(text.contains("[stopped: Kimi repeated the same tool call"));
        assert!(saw_note, "the model was told why its call was not run");
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
            ChatOptions::default(),
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
