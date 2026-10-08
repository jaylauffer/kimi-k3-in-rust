//! Text-only chat adapters. K3: XTML, format reference checkpoint `encoding_k3.py` at
//! f831ab66814297da540d832a5235f8e904f29d06. Kimi Linear: the `<|im_*|>` format of
//! that checkpoint's `chat_template.jinja` at e1df551a447157d4658b573f9a695d57658590e9.
//! Model-independent state lives in loadngo-inference. Generated tokens are preserved
//! verbatim, including K3's thinking.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_inference::system_one::{Answer, Question, Request};
use loadngo_inference::{Session, StopReason, Utf8Stream, tools::Toolbox};
use serde_json::json;

use crate::chat_gemma::GemmaTokens;
use crate::transcript::{Resumed, Transcript};

const OPEN: u32 = 163_587;
const CLOSE: u32 = 163_588;
const SEP: u32 = 163_589;
const END: u32 = 163_586;
const EOS: u32 = 163_585;

const HELP: &str = "Type a message and press Enter. Commands:
  /continue   resume a truncated or cancelled reply, or a turn paused at its budget or by
              Ctrl-C (waiting tool calls run, with a fresh budget)
  /undo       remove the last user turn, including all tool rounds and unfinished replies
  /reset      clear conversation history
  /stats      show context usage
  /help       show these commands
  /quit       exit (or Ctrl-D); Ctrl-C pauses a turn, and at the prompt quits
One line per message. Chats are saved to ~/.loadngo/kimi/transcripts (--no-transcript
turns that off); --resume latest picks the last one up. Kimi Linear can read local files and the
signed CAS snapshot, and create/edit workspace UTF-8 text files (--no-tools
disables these).
File edits and terminal command side effects survive /undo and /reset.
Only workspace-root AGENTS.md and CLAUDE.md are protected; repository copies are editable.
Terminal tools run commands, read output, send stdin and stop sessions (no PTY).
Kimi can also search and read public web pages (--no-web turns those tools off;
terminal commands can still use the network).
When a Kimi Linear chat's context is nearly full, Kimi writes a handoff and the context is
rebuilt from it; the turn goes on. A chat saved at the limit continues with /continue.
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
    Gemma(GemmaTokens),
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
    /// Session boundaries include tool-result prompts. Find the last actual user
    /// prompt before removing its entire exchange, including in older saved chats.
    fn undo_user_turn(&self, session: &mut Session) -> bool {
        // Inspect only the prompt, before its assistant header, so generated
        // control tokens cannot masquerade as another user turn.
        let has_user = |prompt: &[u32]| match self {
            Self::KimiLinear(t) => prompt
                .iter()
                .take_while(|&&id| id != t.assistant)
                .any(|&id| id == t.user),
            Self::Gemma(g) => g.has_user(prompt),
            Self::K3 => false,
        };
        if matches!(self, Self::K3) {
            return session.undo();
        }
        let start = session
            .turn_starts()
            .iter()
            .rev()
            .copied()
            .find(|&start| has_user(&session.tokens()[start..]));
        let Some(start) = start else {
            return false;
        };
        while session.tokens().len() > start {
            if !session.undo() {
                break;
            }
        }
        true
    }

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

    /// # Errors
    /// When the tokenizer lacks one of Gemma 4's control tokens as a single id.
    pub fn gemma(tokenizer: &Tokenizer, bos: u32, eos: &[u32]) -> Result<Self, String> {
        Ok(Self::Gemma(GemmaTokens::new(tokenizer, bos, eos)?))
    }

    /// Adds a system note that opens every conversation (not K3); notes added later
    /// follow earlier ones.
    #[must_use]
    pub fn with_note(mut self, note: &str) -> Self {
        let slot = match &mut self {
            Self::KimiLinear(t) => &mut t.note,
            Self::Gemma(g) => &mut g.note,
            Self::K3 => return self,
        };
        *slot = Some(match slot.take() {
            Some(earlier) => format!("{earlier}\n{note}"),
            None => note.to_string(),
        });
        self
    }

    /// Whether this format is a sequence of role messages (Kimi Linear, Gemma), which
    /// tools, context flow and checkpoints need.
    #[must_use]
    pub const fn has_messages(&self) -> bool {
        !matches!(self, Self::K3)
    }

    /// A message in `role` holding `content`.
    fn message(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer, role: &str, content: &str) {
        match self {
            Self::KimiLinear(t) => t.message(ids, tokenizer, role, content),
            Self::Gemma(g) => g.message(ids, tokenizer, role, content),
            Self::K3 => {}
        }
    }

    /// A user message holding `text`.
    fn user_message(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer, text: &str) {
        match self {
            Self::KimiLinear(t) => {
                ids.push(t.user);
                ordinary(ids, tokenizer, "user");
                ids.push(t.middle);
                ordinary(ids, tokenizer, text);
                ids.push(t.end);
            }
            Self::Gemma(g) => g.message(ids, tokenizer, "user", text),
            Self::K3 => {}
        }
    }

    /// The opening of the assistant's reply.
    fn reply_header(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer) {
        match self {
            Self::KimiLinear(t) => {
                ids.push(t.assistant);
                ordinary(ids, tokenizer, "assistant");
                ids.push(t.middle);
            }
            Self::Gemma(g) => g.reply_header(ids, tokenizer),
            Self::K3 => {}
        }
    }

    /// Where a reply begins in `span` (a turn's prompt and reply): past its header.
    fn reply_offset(&self, span: &[u32]) -> Option<usize> {
        match self {
            Self::KimiLinear(t) => {
                let header = span.iter().position(|&id| id == t.assistant)?;
                let middle = span[header..].iter().position(|&id| id == t.middle)?;
                Some(header + middle + 1)
            }
            Self::Gemma(g) => g.reply_start(span),
            Self::K3 => None,
        }
    }

    /// The text of the last user message in `prompt`.
    fn user_text<'s>(&self, prompt: &'s [u32]) -> Option<&'s [u32]> {
        match self {
            Self::KimiLinear(t) => {
                let user = prompt.iter().rposition(|&id| id == t.user)?;
                let body = &prompt[user..];
                let from = body.iter().position(|&id| id == t.middle)?;
                let to = body[from..]
                    .iter()
                    .position(|&id| id == t.end)
                    .map_or(body.len(), |end| from + end);
                Some(&body[from + 1..to])
            }
            Self::Gemma(g) => g.user_text(prompt),
            Self::K3 => None,
        }
    }

    /// The token that opens tool calls in a reply.
    fn calls_open(&self) -> Option<u32> {
        match self {
            Self::KimiLinear(t) => Some(t.section_begin),
            Self::Gemma(g) => Some(g.call),
            Self::K3 => None,
        }
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
            Self::Gemma(g) => {
                let mut ids = if first {
                    self.preamble(tokenizer, tools)
                } else {
                    Vec::new()
                };
                g.message(&mut ids, tokenizer, "system", &current_date_note());
                g.message(&mut ids, tokenizer, "user", text);
                g.reply_header(&mut ids, tokenizer);
                ids
            }
        }
    }

    /// What every conversation opens with before the first user message: with tools,
    /// the `tool_declare` message and the tool guidance (empty otherwise). The same for
    /// every conversation, so a session that has consumed it can be reused.
    pub fn preamble(&self, tokenizer: &Tokenizer, tools: Option<&Toolbox>) -> Vec<u32> {
        let mut ids = Vec::new();
        if let Self::Gemma(g) = self {
            let tools = tools.filter(|t| !t.is_empty());
            let mut system = String::new();
            if tools.is_some() {
                system.push_str(&TOOL_GUIDANCE.replacen("You are Kimi,", "You are Gemma,", 1));
            }
            if let Some(note) = &g.note {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(note);
            }
            let declaration = tools.map(Toolbox::declaration);
            return g
                .preamble(tokenizer, &system, declaration.as_deref())
                .unwrap_or_else(|_| vec![g.bos]);
        }
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
        let mut ids = Vec::new();
        match self {
            Self::KimiLinear(t) => {
                t.message(&mut ids, tokenizer, "system", system);
                ids.push(t.user);
                ordinary(&mut ids, tokenizer, "user");
                ids.push(t.middle);
            }
            Self::Gemma(g) => {
                ids.push(g.bos);
                g.message(&mut ids, tokenizer, "system", system);
                ids.push(g.turn);
                ordinary(&mut ids, tokenizer, "user\n");
            }
            Self::K3 => return Err("the K3 format does not split user messages".into()),
        }
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
        let mut ids = Vec::new();
        ordinary(&mut ids, tokenizer, tail);
        match self {
            Self::KimiLinear(t) => ids.push(t.end),
            Self::Gemma(g) => ids.extend([g.turn_end, g.newline]),
            Self::K3 => return Err("the K3 format does not split user messages".into()),
        }
        self.reply_header(&mut ids, tokenizer);
        Ok(ids)
    }

    /// Tool calls in a finished Kimi Linear reply, as `(id, arguments)` text pairs.
    fn tool_calls(&self, tokenizer: &Tokenizer, reply: &[u32]) -> Vec<(String, String)> {
        if let Self::Gemma(g) = self {
            return g.tool_calls(tokenizer, reply);
        }
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
            Self::Gemma(_) => "gemma",
        }
    }

    /// Tool results as the template's tool messages.
    fn tool_messages(
        &self,
        tokenizer: &Tokenizer,
        results: &[(String, String, String)],
        last: Option<u32>,
    ) -> Vec<u32> {
        if let Self::Gemma(g) = self {
            // Results inside the model's turn, then the turn closes before Jay's message.
            let mut ids = Vec::new();
            g.responses(&mut ids, tokenizer, results, last == Some(g.response));
            ids.extend([g.turn_end, g.newline]);
            return ids;
        }
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
        last: Option<u32>,
    ) -> Vec<u32> {
        if let Self::Gemma(g) = self {
            // The model goes on in the same turn after its results.
            let mut ids = Vec::new();
            g.responses(&mut ids, tokenizer, results, last == Some(g.response));
            return ids;
        }
        let Self::KimiLinear(t) = self else {
            return Vec::new();
        };
        let mut ids = self.tool_messages(tokenizer, results, last);
        ids.push(t.assistant);
        ordinary(&mut ids, tokenizer, "assistant");
        ids.push(t.middle);
        ids
    }

    fn stops(&self) -> Vec<u32> {
        match self {
            Self::K3 => vec![END, EOS],
            Self::KimiLinear(t) => vec![t.end, t.eos[0], t.eos[1]],
            Self::Gemma(g) => {
                let mut stops = vec![g.turn_end, g.response];
                for &id in &g.eos {
                    if !stops.contains(&id) {
                        stops.push(id);
                    }
                }
                stops
            }
        }
    }

    fn opening(&self) -> &'static str {
        match self {
            Self::K3 => "[thinking] ",
            Self::KimiLinear(_) => "Kimi> ",
            Self::Gemma(_) => "Gemma> ",
        }
    }

    fn push(&self, display: &mut Display, tokenizer: &Tokenizer, token: u32) -> String {
        match self {
            Self::K3 => display.push(tokenizer, token),
            Self::Gemma(g) => {
                if token == g.call {
                    return format!("{}\n[tool call ", terminal_text(&display.utf8.finish()));
                }
                if token == g.call_end {
                    return format!("{}]", terminal_text(&display.utf8.finish()));
                }
                if token == g.quote {
                    return format!("{}\"", terminal_text(&display.utf8.finish()));
                }
                if [
                    g.turn,
                    g.turn_end,
                    g.channel,
                    g.channel_end,
                    g.response,
                    g.response_end,
                ]
                .contains(&token)
                    || g.eos.contains(&token)
                {
                    terminal_text(&display.utf8.finish())
                } else {
                    terminal_text(&display.utf8.push(&tokenizer.decode(&[token])))
                }
            }
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
text_write creates a new UTF-8 text file anywhere in the workspace, including its root, \
without requiring Git. Only workspace-root AGENTS.md and CLAUDE.md are protected; \
repository-local instruction files are editable. Paths start at the workspace. Never adopt another agent's \
dirty files. terminal_exec starts a shell command and returns a session id; terminal_read \
reads output/status, terminal_write sends stdin or closes it, and terminal_stop cancels. \
Use these to inspect repositories and run builds/tests. Read command exit status before \
claiming success; report checks not run. Commands run with your OS user's permissions, \
including filesystem writes and network access: follow Jay's scope and shared-work claims. \
Never push, publish, delete user data, or alter another agent's work without Jay's authorization. \
Change source with text_read and text_edit, not shell commands: they take the same \
paths fs_read shows. This Mac's sed, grep and find are the BSD ones. File edits and command side effects survive \
/undo and /reset. Finish with a board handoff. Avoid repeating unchanged tool calls; after a \
successful edit you may read the updated file again. A failed edit did not change the file: \
read its error, correct its path, revision or exact text, and retry only after fixing the cause. \
Never repeat an unchanged failing write. When a search finds nothing, say so plainly.";

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

/// Answers a System One request about the turn: a probability for every option of every
/// question, read from the model without generating anything.
pub type Decide<'a> = Box<dyn FnMut(&Request) -> Result<Vec<(String, Answer)>, String> + 'a>;

/// How a chat is budgeted, saved and started.
#[derive(Default)]
pub struct ChatOptions<'a> {
    pub budget: TurnBudget,
    /// Where the chat is saved; `None` keeps it in memory only.
    pub transcript: Option<Transcript>,
    /// A saved chat to carry on from.
    pub resumed: Option<Resumed>,
    /// Asked where the turn stands at each compaction (shadow mode: the answers are shown
    /// and saved, and nothing acts on them). `None` asks nothing.
    pub checkpoint: Option<Decide<'a>>,
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
/// Drops the read-only calls from a turn's record of calls already made, when their
/// results may have changed or are no longer in the context. Mutations stay, so an edit
/// is never replayed.
fn forget_reads(earlier_calls: &mut Vec<(String, serde_json::Value)>) {
    earlier_calls.retain(|(tool, _)| {
        !matches!(
            tool.as_str(),
            "text_read" | "fs_read" | "fs_list" | "fs_find" | "fs_grep" | "terminal_exec"
        )
    });
}

fn repeatable_tool(name: &str) -> bool {
    matches!(
        name,
        "terminal_exec" | "terminal_read" | "terminal_write" | "terminal_stop"
    )
}

const REPEATED_CALL: &str = "Not run: you already made this exact call in this turn, and its \
result is above. It would return the same thing. Do not call it again. Answer Jay with what you \
have, or tell him plainly what you could not find.";

/// The result of a second round of repeated calls: tools close for the rest of the turn.
const TOOLS_CLOSED: &str = "Not run: you already made this exact call in this turn, and its \
result is above. Your tools are closed for the rest of this turn. Answer Jay now in plain text \
from what you have read: what you found, with paths and line numbers, and say plainly what you \
did not find.";

/// How the answer after closed tools begins; the program writes it, so the reply starts as
/// text. On 2026-10-08, told not to call again, Kimi made the same call a third time.
const ANSWER_OPENING: &str = "I'll stop searching here and answer from what I have read.\n\n";

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

/// What Kimi is asked when her context is nearly full. Her answer is the handoff.
const FLOW_REQUEST: &str = "[Automatic message from the chat program, not from Jay.] Stop \
working for one reply. Your context is nearly full and is about to be cleared. Only the \
handoff you write now, Jay's messages and your most recent tool rounds are carried over; \
everything else you have read is gone afterwards. Do not call tools in this reply. Write a \
handoff to yourself in plain text under these headings. TASK: what Jay asked for, in his \
terms. STANDING: anything Jay asked you to remember or keep doing, including from an earlier \
handoff. DONE: what is finished and checked, with paths. FACTS: the exact paths, line numbers, \
names, commands and results you will need. FAILED: what you tried that did not work and why, \
so you do not repeat it. FILES CHANGED: every file you created or modified, temporary ones \
included. NEXT: the single next step. Be specific and brief, and say so where you are unsure.";

/// How every handoff begins; the program writes it, so the reply starts as text.
const HANDOFF_OPENING: &str = "TASK:";

/// Opens the system message that carries a handoff into the rebuilt context.
const FLOW_NOTE: &str = "The earlier conversation was cleared to free context. Before that you \
wrote the handoff below: it is your own notes. Carry on with the work from it without asking \
Jay to repeat himself. Details that are not in it are gone, so read files again instead of \
guessing.";

/// Most bytes of Jay's earlier messages repeated word for word after a compaction.
const EARLIER_MESSAGES_BYTES: usize = 2048;

/// Context flow: when a Kimi Linear chat's context is nearly full, Kimi writes a handoff,
/// the context is rebuilt from it and the turn goes on. All sizes follow the context size.
#[derive(Clone, Copy, Debug)]
struct Flow {
    enabled: bool,
    max_context: usize,
    /// The context length right after the latest compaction.
    compacted_len: usize,
}

impl Flow {
    /// The context length a compaction keeps the chat under.
    const fn limit(&self) -> usize {
        self.max_context - self.max_context / 4
    }

    /// The longest handoff.
    fn handoff_tokens(&self) -> usize {
        (self.max_context / 32).clamp(64, 1024)
    }

    /// How much of the latest tool rounds is carried over word for word.
    const fn tail_tokens(&self) -> usize {
        self.max_context / 8
    }

    /// Whether the chat has grown enough since the latest compaction for another to help.
    const fn progressed(&self, len: usize) -> bool {
        self.enabled && len >= self.compacted_len + self.max_context / 8
    }

    /// Whether adding `adding` tokens to a context of `len` calls for a compaction first.
    const fn due(&self, len: usize, adding: usize) -> bool {
        self.progressed(len) && len + adding > self.limit()
    }
}

/// The typed questions asked about a turn when its context is compacted. The state is
/// what a loadngo Task submitter would hold: the request (`TaskRequest.summary`) and the
/// worker's latest status note (`TaskStatus.note`, here the handoff). `state` chooses
/// among the assignment states recommended for `TaskStatus.state` in loadngo
/// `docs/TASK_CHECKPOINT_RECOMMENDATIONS.md` and `stuck`, which a worker does not report
/// and a reader of its notes may conclude. `complete` is the claim a `TaskResult` makes,
/// which only the submitter's `TaskAck` settles.
fn checkpoint_request(request: &str, handoff: &str) -> Request {
    let options = |list: &[(&str, &str)]| {
        list.iter()
            .map(|(label, text)| ((*label).to_string(), (*text).to_string()))
            .collect()
    };
    Request {
        state: format!("Request from Jay:\n{request}\n\nThe worker's status notes:\n{handoff}"),
        questions: vec![
            (
                "state".into(),
                Question::Choice {
                    instructions: Some("Where does the work on Jay's request stand?".into()),
                    criteria: options(&[
                        (
                            "in-progress",
                            "Work is under way, and the notes name a next step that differs \
                             from what already failed",
                        ),
                        (
                            "paused",
                            "The work stopped for a reason that has nothing to do with the \
                             task, and can simply be resumed",
                        ),
                        (
                            "needs-input",
                            "The work cannot go on without a decision, an answer or access \
                             from Jay",
                        ),
                        (
                            "needs-help",
                            "The work needs expertise or a capability the worker does not have",
                        ),
                        ("withdrawn", "The worker has given the task up"),
                        (
                            "complete",
                            "Everything Jay asked for is finished and checked",
                        ),
                        (
                            "stuck",
                            "The same step keeps failing, or the notes show effort without \
                             progress",
                        ),
                    ]),
                },
            ),
            (
                "repeating".into(),
                Question::Noul {
                    instructions: "The notes show the same action failing more than once.".into(),
                },
            ),
            (
                "progress".into(),
                Question::Score {
                    instructions: Some("How far has the work on Jay's request come?".into()),
                    criteria: options(&[
                        ("0", "Nothing towards the request has been done"),
                        ("1", "Files were read or explored, and nothing was changed"),
                        ("2", "Part of the requested change has been made"),
                        ("3", "The requested change has been made and checked"),
                    ]),
                },
            ),
        ],
    }
}

/// Asks `decide` where the turn stands, shows the answers and saves them with what the
/// program itself counted (`identical_calls`), which later tells how well they held up.
fn checkpoint(
    decide: &mut Decide<'_>,
    request: &str,
    handoff: &str,
    identical_calls: usize,
    output: &mut impl Write,
    transcript: &mut Option<Transcript>,
) {
    let started = Instant::now();
    match decide(&checkpoint_request(request, handoff)) {
        Ok(answers) => {
            let shown: Vec<String> = answers
                .iter()
                .map(|(id, answer)| {
                    let (best, p) = answer.best();
                    format!("{id} {best} {:.0}%", p * 100.0)
                })
                .collect();
            let _ = writeln!(
                output,
                "[checkpoint, shadow: {}; {identical_calls} identical call(s) counted; \
                 nothing acts on this]",
                shown.join(", ")
            );
            log(
                transcript,
                json!({
                    "event": "checkpoint",
                    "mode": "shadow",
                    "request": request,
                    "handoff": handoff,
                    "answers": loadngo_inference::system_one::response_json(&answers)["answers"],
                    "observed": {"identical_calls": identical_calls},
                    "seconds": started.elapsed().as_secs_f64(),
                }),
            );
        }
        Err(error) => {
            let _ = writeln!(output, "[checkpoint not answered: {error}]");
        }
    }
}

struct Compacted {
    session: Session,
    handoff: String,
    /// Jay's latest message, word for word.
    request: Option<String>,
    /// Tokens Kimi generated for the handoff.
    generated: usize,
}

/// Has Kimi write a handoff and builds the context that replaces `session`'s: the opening,
/// the handoff, Jay's earlier messages and, when a turn is in progress (`in_turn`), his
/// latest message and the newest tool rounds that fit [`Flow::tail_tokens`]. An unfinished
/// reply is dropped: the rebuilt context ends where it began, ready to be written again.
/// `session` itself is not changed, so a failure leaves the chat as it was.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compact(
    format: &ChatFormat,
    tools: Option<&Toolbox>,
    tokenizer: &Tokenizer,
    session: &Session,
    in_turn: bool,
    flow: Flow,
    cancel: &AtomicBool,
    next: &mut impl FnMut(&[u32]) -> Result<u32, String>,
    output: &mut impl Write,
) -> Result<Compacted, String> {
    if !format.has_messages() {
        return Err("only Kimi Linear and Gemma chats are compacted".into());
    }
    let (tokens, turns) = (session.tokens(), session.turn_starts());
    let Some(last) = turns.len().checked_sub(1) else {
        return Err("there is no conversation yet".into());
    };
    let turn_end = |turn: usize| turns.get(turn + 1).copied().unwrap_or(tokens.len());
    // Where a turn's reply begins: just past its prompt's assistant header.
    let reply_start = |turn: usize| {
        let span = &tokens[turns[turn]..turn_end(turn)];
        Some(turns[turn] + format.reply_offset(span)?)
    };
    // Jay's messages, word for word: Kimi's summary of a request is not the request.
    let mut said: Vec<(usize, String)> = Vec::new();
    for turn in 0..turns.len() {
        let prompt = &tokens[turns[turn]..reply_start(turn).unwrap_or_else(|| turn_end(turn))];
        let Some(body) = format.user_text(prompt) else {
            continue;
        };
        said.push((turn, tokenizer.decode_lossy(body)));
    }
    let pending = session.is_pending();
    // The end of what is carried over.
    let end = if pending {
        reply_start(last).ok_or("the unfinished reply has no start")?
    } else {
        tokens.len()
    };

    // The handoff is written at the end of the existing context where there is room, so
    // nothing is read twice; otherwise the newest turns are left out until there is.
    // Asked as a user message, with the reply already begun: on the model, a system
    // message after a run of tool rounds was answered with one more tool call.
    let mut request = Vec::new();
    format.user_message(&mut request, tokenizer, FLOW_REQUEST);
    format.reply_header(&mut request, tokenizer);
    ordinary(&mut request, tokenizer, HANDOFF_OPENING);
    let need = request.len() + flow.handoff_tokens() + 1;
    let cut_at = |kept: usize| turns.get(kept).copied().unwrap_or(tokens.len());
    let mut kept = if pending { last } else { turns.len() };
    while kept > 0 && cut_at(kept) + need > session.max_context() {
        kept -= 1;
    }
    let (mut handoff, mut generated) = (String::new(), 0);
    if kept > 0 {
        let cut = cut_at(kept);
        let mut scratch = Session::restore(
            session.max_context(),
            tokens[..cut].to_vec(),
            turns[..kept].to_vec(),
            false,
        )
        .and_then(|mut scratch| scratch.begin_turn(&request).map(|()| scratch))
        .map_err(|e| e.to_string())?;
        write!(
            output,
            "\n[context {}/{}: Kimi writes a handoff, then the context is rebuilt from it]\n\
             Kimi (handoff)> {HANDOFF_OPENING}",
            tokens.len(),
            session.max_context()
        )
        .and_then(|()| output.flush())
        .map_err(|e| e.to_string())?;
        let mut display = Display::default();
        let done = scratch
            .generate(
                flow.handoff_tokens(),
                &format.stops(),
                cancel,
                &mut *next,
                |token| {
                    write!(output, "{}", format.push(&mut display, tokenizer, token))
                        .and_then(|()| output.flush())
                        .map_err(|e| e.to_string())
                },
            )
            .map_err(|e| e.to_string())?;
        writeln!(output, "{}", terminal_text(&display.utf8.finish())).map_err(|e| e.to_string())?;
        if done.reason == StopReason::Cancelled {
            return Err("interrupted while the handoff was being written".into());
        }
        generated = done.tokens;
        let written = &scratch.tokens()[cut + request.len()..];
        let text = written
            .iter()
            .position(|id| Some(*id) == format.calls_open() || format.stops().contains(id))
            .map_or(written, |stop| &written[..stop]);
        let text = tokenizer.decode_lossy(text);
        if !text.trim().is_empty() {
            handoff = format!("{HANDOFF_OPENING}{}", text.trim_end());
        }
    }

    let request = said.last().map(|(_, text)| text.clone());
    let latest = if in_turn {
        Some(
            said.pop()
                .ok_or("the turn in progress has no message from Jay")?,
        )
    } else {
        None
    };
    let mut note = format!(
        "{FLOW_NOTE}\n\n## Handoff\n{}",
        if handoff.is_empty() {
            "(You wrote no handoff.)"
        } else {
            &handoff
        }
    );
    let mut earlier = Vec::new();
    let mut bytes = 0;
    for (_, text) in said.iter().rev() {
        bytes += text.len();
        if bytes > EARLIER_MESSAGES_BYTES {
            break;
        }
        earlier.push(text.as_str());
    }
    if !earlier.is_empty() {
        note.push_str("\n\n## Jay's earlier messages, oldest first");
        for text in earlier.iter().rev() {
            note.push_str("\n- ");
            note.push_str(text);
        }
    }
    let mut ids = format.preamble(tokenizer, tools);
    format.message(&mut ids, tokenizer, "system", &note);
    let mut new_turns = vec![0];
    if let Some((user_turn, text)) = latest {
        ids.extend(format.prompt(tokenizer, &text, false, tools));
        // The newest rounds that fit, starting at a reply so the history stays well formed.
        // A finished reply whose tool results are about to follow is always kept.
        let starts = (user_turn..=last)
            .map(|turn| reply_start(turn).map(|start| (turn, start)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a turn has no reply")?;
        let (turn, start) = starts
            .iter()
            .copied()
            .find(|&(_, start)| start <= end && end - start <= flow.tail_tokens())
            .or(starts.last().copied().filter(|_| !pending))
            .ok_or("the newest reply cannot be found")?;
        let base = ids.len();
        new_turns.extend(
            turns[turn + 1..]
                .iter()
                .filter(|&&s| s < end)
                .map(|s| base + s - start),
        );
        ids.extend_from_slice(&tokens[start..end]);
    }
    if ids.len() >= flow.limit() {
        return Err(format!(
            "what must be kept is {} tokens, too much for a context of {}",
            ids.len(),
            session.max_context()
        ));
    }
    let session = Session::restore(session.max_context(), ids, new_turns, in_turn && pending)
        .map_err(|e| e.to_string())?;
    Ok(Compacted {
        session,
        handoff,
        request,
        generated,
    })
}

/// Compacts `session` in place and says so; on failure the chat is left as it was and the
/// reason is shown. Returns the tokens Kimi generated for the handoff.
#[allow(clippy::too_many_arguments)]
fn flow_now(
    format: &ChatFormat,
    tools: Option<&Toolbox>,
    tokenizer: &Tokenizer,
    session: &mut Session,
    in_turn: bool,
    flow: &mut Flow,
    cancel: &AtomicBool,
    next: &mut impl FnMut(&[u32]) -> Result<u32, String>,
    output: &mut impl Write,
    transcript: &mut Option<Transcript>,
    decide: &mut Option<Decide<'_>>,
    identical_calls: &mut usize,
) -> Option<usize> {
    let before = session.tokens().len();
    match compact(
        format, tools, tokenizer, session, in_turn, *flow, cancel, next, output,
    ) {
        Ok(compacted) => {
            *session = compacted.session;
            flow.compacted_len = session.tokens().len();
            log(
                transcript,
                json!({
                    "event": "compaction",
                    "before": before,
                    "after": session.tokens().len(),
                    "in_turn": in_turn,
                    "handoff": compacted.handoff,
                }),
            );
            let _ = writeln!(
                output,
                "[context rebuilt from the handoff: {before} -> {} tokens]",
                session.tokens().len()
            );
            if let (Some(decide), Some(request)) = (decide, &compacted.request)
                && !compacted.handoff.is_empty()
            {
                checkpoint(
                    decide,
                    request,
                    &compacted.handoff,
                    std::mem::take(identical_calls),
                    output,
                    transcript,
                );
            }
            Some(compacted.generated)
        }
        Err(error) => {
            let _ = writeln!(output, "[context not compacted: {error}]");
            None
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
    options: ChatOptions<'_>,
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
    options: ChatOptions<'_>,
) -> Result<(), String> {
    let ChatOptions {
        budget,
        mut transcript,
        resumed,
        checkpoint: mut decide,
    } = options;
    // Calls Kimi made with a name and arguments she had already used since the last
    // checkpoint: what the program can count itself to hold the checkpoint's answers to.
    let mut identical_calls = 0;
    let mut calls_made: Vec<(String, serde_json::Value)> = Vec::new();
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
    let mut flow = Flow {
        enabled: format.has_messages(),
        max_context,
        compacted_len: 0,
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
                flow.compacted_len = 0;
                display = Display::default();
                log(&mut transcript, json!({"event": "command", "text": text}));
                writeln!(output, "Conversation cleared.").map_err(|e| e.to_string())?;
                continue;
            }
            "/undo" => {
                let removed = format.undo_user_turn(&mut session);
                (reply_start, held) = (session.tokens().len(), None);
                flow.compacted_len = flow.compacted_len.min(session.tokens().len());
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
                    format.tool_messages(
                        tokenizer,
                        &held.abandoned(),
                        session.tokens().last().copied(),
                    )
                });
                ids.extend(format.prompt(tokenizer, text, session.tokens().is_empty(), tools));
                // A message that would crowd the context starts from a handoff instead.
                if held.is_none()
                    && !session.is_pending()
                    && flow.due(session.tokens().len(), ids.len())
                    && flow_now(
                        format,
                        tools,
                        tokenizer,
                        &mut session,
                        false,
                        &mut flow,
                        cancel,
                        &mut next,
                        &mut output,
                        &mut transcript,
                        &mut decide,
                        &mut identical_calls,
                    )
                    .is_some()
                {
                    ids = format.prompt(tokenizer, text, session.tokens().is_empty(), tools);
                }
                if let Err(error) = session.begin_turn(&ids) {
                    writeln!(output, "{error}").map_err(|e| e.to_string())?;
                    continue;
                }
                held = None;
                (identical_calls, calls_made) = (0, Vec::new());
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
        let mut failed_writes = Vec::new();
        let mut repeated_rounds = 0;
        // After a second round of repeated calls: one last reply, with no tool calls.
        let mut closing = false;
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
                // With tools closed, a tool call ends the reply where it would begin.
                let calls_open = format.calls_open().filter(|_| closing);
                let result = session.generate(
                    max_tokens,
                    &stops,
                    cancel,
                    |context| match looping.get() {
                        Some(_) => Ok(stops[0]),
                        None => next(context).map(|token| {
                            if Some(token) == calls_open {
                                stops[0]
                            } else {
                                token
                            }
                        }),
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
                // A reply that ran into the end of the context is dropped and written again
                // in a context rebuilt from her handoff; the turn goes on.
                if done.reason == StopReason::ContextLimit
                    && flow.progressed(session.tokens().len())
                    && let Some(spent) = flow_now(
                        format,
                        tools,
                        tokenizer,
                        &mut session,
                        true,
                        &mut flow,
                        cancel,
                        &mut next,
                        &mut output,
                        &mut transcript,
                        &mut decide,
                        &mut identical_calls,
                    )
                {
                    turn_tokens += spent;
                    forget_reads(&mut earlier_calls);
                    repeated_rounds = 0;
                    reply_start = session.tokens().len();
                    reply.clear();
                    if let Some(transcript) = &mut transcript {
                        transcript.save_state(&session, reply_start, None);
                    }
                    display = Display::default();
                    write!(output, "{}", format.opening())
                        .and_then(|()| output.flush())
                        .map_err(|e| e.to_string())?;
                    continue;
                }
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
                if closing {
                    writeln!(
                        output,
                        "[tools closed: Kimi repeated the same tool call and answered from what \
                         she had read; ask differently or /reset]"
                    )
                    .map_err(|e| e.to_string())?;
                    break Some("repeated tool call; answered without tools".into());
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
                    closing = repeated_rounds == 2;
                }
                Held {
                    done: Vec::new(),
                    remaining: calls,
                }
            };
            let Some(tools) = tools else { break None };
            // Closing runs no tool, so it is not paused: the answer is what a pause would want.
            if !closing && let Some(spent) = budget.spent(turn_started.elapsed(), turn_tokens) {
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
            let mut write_failure_limit = false;
            while !calls.remaining.is_empty() && !cancel.load(Ordering::Relaxed) {
                let (id, arguments) = calls.remaining.remove(0);
                let name = tool_name(&id).to_string();
                let key = call_key(&name, &arguments);
                if calls_made.contains(&key) {
                    identical_calls += 1;
                } else {
                    calls_made.push(key.clone());
                }
                log(
                    &mut transcript,
                    json!({"event": "tool_call", "name": name, "arguments": arguments}),
                );
                if !repeatable_tool(&name) && earlier_calls.contains(&key) {
                    writeln!(output, "[tool call repeated: {name}; not run again]")
                        .map_err(|e| e.to_string())?;
                    let note = if closing { TOOLS_CLOSED } else { REPEATED_CALL };
                    if let Some(transcript) = &mut transcript {
                        transcript.tool_result(&name, note);
                    }
                    calls.done.push((id, name, note.to_string()));
                    continue;
                }
                let (text, succeeded) = match tools.call(&name, &arguments) {
                    Ok(text) => {
                        if matches!(name.as_str(), "text_write" | "text_edit") {
                            failed_writes.clear();
                        }
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
                            forget_reads(&mut earlier_calls);
                            repeated_rounds = 0;
                        }
                        if name == "board_add_row" {
                            earlier_calls.retain(|(tool, _)| {
                                !matches!(tool.as_str(), "board_read" | "board_sections")
                            });
                            repeated_rounds = 0;
                        }
                        (text, true)
                    }
                    Err(error) => {
                        if matches!(name.as_str(), "text_write" | "text_edit") {
                            let failure = (key.clone(), error.clone());
                            write_failure_limit |= failed_writes.contains(&failure);
                            failed_writes.push(failure);
                        }
                        (format!("error: {error}"), false)
                    }
                };
                writeln!(output, "[tool result {name}: {} bytes]", text.len())
                    .map_err(|e| e.to_string())?;
                if let Some(transcript) = &mut transcript {
                    transcript.tool_result(&name, &text);
                }
                calls.done.push((id, name, text));
                // Failed calls may succeed after their prerequisites change. Cache
                // successful mutations only, so an edit is never replayed.
                if succeeded {
                    earlier_calls.push(key);
                }
                if write_failure_limit {
                    break;
                }
            }
            if !calls.remaining.is_empty() && !write_failure_limit {
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
            if write_failure_limit {
                for (id, _) in calls.remaining.drain(..) {
                    let name = tool_name(&id).to_string();
                    calls.done.push((id, name, NOT_RUN.to_string()));
                }
                // Hold results before opening a reply: /continue and a corrective
                // user message can both consume them without an unfinished reply.
                held = Some(calls);
                writeln!(output, "[paused: repeated write failure, the same write failing twice with the same error. Fix the reported cause before /continue, or send a new instruction.]")
                    .map_err(|e| e.to_string())?;
                break Some("write failure limit".into());
            }
            let mut results = calls.done;
            let mut prompt =
                format.tool_results(tokenizer, &results, session.tokens().last().copied());
            // Results that would crowd the context go into one rebuilt from her handoff.
            if flow.due(session.tokens().len(), prompt.len())
                && let Some(spent) = flow_now(
                    format,
                    Some(tools),
                    tokenizer,
                    &mut session,
                    true,
                    &mut flow,
                    cancel,
                    &mut next,
                    &mut output,
                    &mut transcript,
                    &mut decide,
                    &mut identical_calls,
                )
            {
                turn_tokens += spent;
                forget_reads(&mut earlier_calls);
                repeated_rounds = 0;
            }
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
                prompt = format.tool_results(tokenizer, &results, session.tokens().last().copied());
            }
            if closing {
                ordinary(&mut prompt, tokenizer, ANSWER_OPENING);
                log(
                    &mut transcript,
                    json!({"event": "tools_closed", "opening": ANSWER_OPENING}),
                );
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
            write!(
                output,
                "{}{}",
                format.opening(),
                if closing { ANSWER_OPENING } else { "" }
            )
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
    fn undo_removes_all_tool_rounds_from_restored_completed_or_partial_turns() {
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let cancel = AtomicBool::new(false);
        for partial in [false, true] {
            let mut session = Session::new(32_768).unwrap();
            session
                .begin_turn(&format.prompt(&tokenizer, "Keep this", true, None))
                .unwrap();
            session
                .generate(1, &[t.end], &cancel, |_| Ok(t.end), |_| Ok(()))
                .unwrap();
            let kept = session.tokens().to_vec();
            session
                .begin_turn(&format.prompt(&tokenizer, "Remove this", false, None))
                .unwrap();
            for round in 0..3 {
                session
                    .generate(1, &[t.end], &cancel, |_| Ok(t.end), |_| Ok(()))
                    .unwrap();
                session
                    .begin_turn(&format.tool_results(
                        &tokenizer,
                        &[(
                            format!("functions.fs_read:{round}"),
                            "fs_read".into(),
                            "read".into(),
                        )],
                        None,
                    ))
                    .unwrap();
            }
            // A generated user control must not count as a new user prompt.
            let mut reply = [t.user, t.end].into_iter();
            session
                .generate(
                    if partial { 1 } else { 2 },
                    &[t.end],
                    &cancel,
                    |_| Ok(reply.next().unwrap()),
                    |_| Ok(()),
                )
                .unwrap();
            let mut restored = Session::restore(
                session.max_context(),
                session.tokens().to_vec(),
                session.turn_starts().to_vec(),
                session.is_pending(),
            )
            .unwrap();
            assert!(format.undo_user_turn(&mut restored));
            assert_eq!(restored.tokens(), kept);
            assert_eq!(restored.turn_starts(), [0]);
            assert!(!restored.is_pending());
            assert!(format.undo_user_turn(&mut restored));
            assert_eq!(restored.tokens(), &[] as &[u32]);
            assert!(!format.undo_user_turn(&mut restored));
        }
    }

    #[test]
    fn one_undo_after_tool_work_preserves_prior_turn_and_accepts_a_new_message() {
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let (tools, ran) = stepping_tools(None);
        let mut replies = stepping_replies(&tokenizer, t, 0, 1)
            .chain(stepping_replies(&tokenizer, t, 3, 1))
            .chain(stepping_replies(&tokenizer, t, 0, 1));
        let mut output = Vec::new();
        let mut checked = false;
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            32_768,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Keep this\nRemove this\n/undo\nAfter undo\n/quit\n"[..],
            &mut output,
            |ids| {
                let text = tokenizer.decode_lossy(ids);
                if text.contains("After undo") {
                    assert!(text.contains("Keep this"));
                    assert!(!text.contains("Remove this"));
                    assert!(!ids.contains(&t.section_begin));
                    assert!(!ids.contains(&t.call_begin));
                    checked = true;
                }
                Ok(replies.next().expect("scripted reply"))
            },
            ChatOptions::default(),
        )
        .unwrap();
        assert!(checked);
        assert_eq!(ran.borrow().len(), 3);
        assert_eq!(
            String::from_utf8(output)
                .unwrap()
                .matches("Last turn removed.")
                .count(),
            1
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)] // One scripted regression checks recovery and pause.
    fn failed_writes_can_recover_but_repeated_failures_pause() {
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;
        struct WriteTool {
            name: &'static str,
            ready: Rc<Cell<bool>>,
            writes: Rc<Cell<usize>>,
        }
        impl loadngo_inference::tools::Tool for WriteTool {
            fn name(&self) -> &'static str {
                self.name
            }
            fn description(&self) -> &'static str {
                "test"
            }
            fn parameters(&self) -> serde_json::Value {
                json!({"type":"object"})
            }
            fn call(&self, _: &serde_json::Value) -> Result<String, String> {
                if self.name == "repair" {
                    self.ready.set(true);
                    return Ok("fixed".into());
                }
                self.writes.set(self.writes.get() + 1);
                if self.ready.get() {
                    Ok("wrote".into())
                } else {
                    Err("fix prerequisite first".into())
                }
            }
        }
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        for (recover, new_instruction) in [(false, false), (false, true), (true, false)] {
            let ready = Rc::new(Cell::new(false));
            let writes = Rc::new(Cell::new(0));
            let mut tools = Toolbox::default();
            for name in ["text_write", "repair"] {
                tools.push(Box::new(WriteTool {
                    name,
                    ready: ready.clone(),
                    writes: writes.clone(),
                }));
            }
            let names = if recover {
                vec!["text_write", "repair", "text_write"]
            } else {
                vec!["text_write", "text_write"]
            };
            let mut ids = Vec::new();
            for (i, name) in names.iter().enumerate() {
                ids.extend([t.section_begin, t.call_begin]);
                ordinary(&mut ids, &tokenizer, &format!("functions.{name}:{i}"));
                ids.push(t.argument_begin);
                ordinary(&mut ids, &tokenizer, "{}");
                ids.extend([t.call_end, t.section_end, t.end]);
            }
            ordinary(&mut ids, &tokenizer, "Done.");
            ids.push(t.end);
            let mut replies = ids.into_iter();
            let contexts = RefCell::new(Vec::new());
            let mut output = Vec::new();
            run_with(
                &format,
                Some(&tools),
                &tokenizer,
                32_768,
                200,
                &AtomicBool::new(false),
                &AtomicBool::new(false),
                if recover {
                    &b"Write\n/quit\n"[..]
                } else if new_instruction {
                    &b"Write\n/stats\nUse a different path\n/quit\n"[..]
                } else {
                    &b"Write\n/stats\n/continue\n/quit\n"[..]
                },
                &mut output,
                |ids| {
                    contexts.borrow_mut().push(tokenizer.decode_lossy(ids));
                    Ok(replies.next().expect("scripted reply"))
                },
                ChatOptions::default(),
            )
            .unwrap();
            let output = String::from_utf8(output).unwrap();
            assert_eq!(writes.get(), 2);
            assert!(!output.contains("tool call repeated"));
            assert_eq!(output.contains("[paused: repeated write failure"), !recover);
            assert!(output.contains("Done."));
            if !recover {
                assert!(output.contains("unfinished reply: false"));
                assert!(
                    contexts
                        .borrow()
                        .last()
                        .unwrap()
                        .matches("fix prerequisite first")
                        .count()
                        >= 2
                );
                if new_instruction {
                    assert!(
                        contexts
                            .borrow()
                            .last()
                            .unwrap()
                            .contains("Use a different path")
                    );
                }
            }
        }
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
            ..ChatOptions::default()
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

    /// A scripted Kimi for context-flow tests: writes `HANDOFF` when asked for one, calls
    /// `fs_read` on a new path for each of `rounds` rounds, then answers `Done.`. Records
    /// the first context it was shown after each compaction.
    struct FlowScript<'a> {
        tokenizer: &'a Tokenizer,
        t: &'a LinearTokens,
        rounds: usize,
        called: usize,
        queue: std::collections::VecDeque<u32>,
        longest: usize,
        /// The length of the context last shown; a shorter one follows a compaction.
        shown: usize,
        rebuilt: Vec<String>,
    }

    const HANDOFF: &str = "TASK: read every path. NEXT: the next path.";

    impl FlowScript<'_> {
        fn next(&mut self, context: &[u32]) -> u32 {
            self.longest = self.longest.max(context.len());
            if self.queue.is_empty() {
                let text = self.tokenizer.decode_lossy(context);
                let mut ids = Vec::new();
                if text.ends_with(&format!("{FLOW_REQUEST}assistant{HANDOFF_OPENING}")) {
                    let rest = HANDOFF.strip_prefix(HANDOFF_OPENING).unwrap();
                    ordinary(&mut ids, self.tokenizer, rest);
                } else {
                    if context.len() < self.shown && text.contains(FLOW_NOTE) {
                        self.rebuilt.push(text);
                    }
                    if self.called < self.rounds {
                        ids.extend([self.t.section_begin, self.t.call_begin]);
                        ordinary(
                            &mut ids,
                            self.tokenizer,
                            &format!("functions.fs_read:{}", self.called),
                        );
                        ids.push(self.t.argument_begin);
                        ordinary(
                            &mut ids,
                            self.tokenizer,
                            &format!("{{\"path\": \"file-{}\"}}", self.called),
                        );
                        ids.extend([self.t.call_end, self.t.section_end]);
                        self.called += 1;
                    } else {
                        ordinary(&mut ids, self.tokenizer, "Done.");
                    }
                }
                ids.push(self.t.end);
                self.queue = ids.into();
            }
            self.shown = context.len();
            self.queue.pop_front().unwrap()
        }
    }

    #[test]
    fn a_long_turn_flows_through_handoffs_instead_of_filling_the_context() {
        const CONTEXT: usize = 8192;
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let (tools, ran) = stepping_tools(None);
        let mut script = FlowScript {
            tokenizer: &tokenizer,
            t,
            rounds: 200,
            called: 0,
            queue: std::collections::VecDeque::default(),
            longest: 0,
            shown: 0,
            rebuilt: Vec::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        let transcript =
            Transcript::create(dir.path(), "kimi-linear", std::path::Path::new("/m")).unwrap();
        let log = transcript.log_path().to_path_buf();
        let asked = std::cell::RefCell::new(Vec::new());
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            CONTEXT,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Read every path for me\n/quit\n"[..],
            &mut output,
            |context| Ok(script.next(context)),
            ChatOptions {
                budget: TurnBudget {
                    time: None,
                    tokens: None,
                },
                transcript: Some(transcript),
                resumed: None,
                // A scripted System One model: sure of everything it is asked.
                checkpoint: Some(Box::new(|request| {
                    asked.borrow_mut().push(request.state.clone());
                    Ok(request
                        .questions
                        .iter()
                        .map(|(id, question)| {
                            let mut options = question.options().into_iter();
                            let mut probabilities = vec![(options.next().unwrap().0, 0.9)];
                            probabilities.extend(options.map(|(label, _)| (label, 0.0)));
                            (id.clone(), Answer { probabilities })
                        })
                        .collect())
                })),
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        // Every call ran exactly once, in order, across several compactions.
        let expected: Vec<String> = (0..200).map(|n| format!("\"file-{n}\"")).collect();
        assert_eq!(*ran.borrow(), expected);
        assert!(output.contains("Done."));
        assert!(!output.contains("Context full"));
        let compactions = output.matches("[context rebuilt from the handoff").count();
        assert!(compactions >= 2, "{compactions} compactions");
        // Only a handoff is ever written above the compaction line; the context never fills.
        assert!(script.longest < CONTEXT, "{}", script.longest);
        assert_ne!(script.rebuilt.len(), 0, "nothing was rebuilt");
        for context in &script.rebuilt {
            assert!(context.contains(HANDOFF));
            assert!(context.contains("Read every path for me"));
            // The reply whose results were still to come was carried over.
            assert!(context.contains("functions.fs_read:"), "{context}");
        }
        let log = std::fs::read_to_string(log).unwrap();
        assert_eq!(log.matches("\"event\":\"compaction\"").count(), compactions);
        // Each compaction was followed by a checkpoint over Jay's request and her handoff,
        // shown and saved, and the turn went on whatever it said.
        assert_eq!(asked.borrow().len(), compactions);
        for state in asked.borrow().iter() {
            assert!(state.contains("Read every path for me") && state.contains(HANDOFF));
        }
        assert_eq!(
            output
                .matches("[checkpoint, shadow: state in-progress 90%, repeating true 90%, progress 0 90%; 0 identical call(s) counted")
                .count(),
            compactions,
            "{output}"
        );
        let saved: Vec<serde_json::Value> = log
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|event: &serde_json::Value| event["event"] == "checkpoint")
            .collect();
        assert_eq!(saved.len(), compactions);
        assert_eq!(saved[0]["mode"], "shadow");
        let states = saved[0]["answers"]["state"].as_object().unwrap();
        assert!(
            states.len() == 7 && states["needs-help"].is_number() && states["stuck"].is_number()
        );
        assert_eq!(saved[0]["observed"]["identical_calls"], 0);
    }

    #[test]
    fn a_saved_chat_stopped_at_the_context_limit_resumes_through_a_handoff() {
        const CONTEXT: usize = 8192;
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let (tools, ran) = stepping_tools(None);
        let cancel = AtomicBool::new(false);
        // Jay's request, one tool round whose result nearly fills the context, and a reply
        // cut off by the limit after five tokens: how Kimi's 2026-10-02 turn was saved.
        let mut session = Session::new(CONTEXT).unwrap();
        session
            .begin_turn(&format.prompt(&tokenizer, "Add the waves", true, Some(&tools)))
            .unwrap();
        let mut call = stepping_replies(&tokenizer, t, 1, 1);
        session
            .generate(
                200,
                &[t.end],
                &cancel,
                |_| Ok(call.next().unwrap()),
                |_| Ok(()),
            )
            .unwrap();
        let result = |text: String| {
            format.tool_results(
                &tokenizer,
                &[("functions.fs_read:00".into(), "fs_read".into(), text)],
                None,
            )
        };
        let filler = CONTEXT - 5 - session.tokens().len() - result(String::new()).len();
        session.begin_turn(&result("x".repeat(filler))).unwrap();
        let reply_start = session.tokens().len();
        let stop = session
            .generate(200, &[t.end], &cancel, |_| Ok(u32::from(b'L')), |_| Ok(()))
            .unwrap();
        assert_eq!(stop.reason, StopReason::ContextLimit);

        let mut script = FlowScript {
            tokenizer: &tokenizer,
            t,
            rounds: 1,
            called: 0,
            queue: std::collections::VecDeque::default(),
            longest: 0,
            shown: 0,
            rebuilt: Vec::new(),
        };
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            CONTEXT,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"/continue\n/stats\n/quit\n"[..],
            &mut output,
            |context| Ok(script.next(context)),
            ChatOptions {
                resumed: Some(Resumed {
                    session,
                    reply_start,
                    held: None,
                }),
                ..ChatOptions::default()
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(
            output.contains("[context rebuilt from the handoff: 8192 ->"),
            "{output}"
        );
        assert!(!output.contains("Context full"), "{output}");
        assert!(output.contains("Done."), "{output}");
        assert_eq!(*ran.borrow(), ["\"file-0\""]);
        // The rebuilt context opens the reply again: the request and the handoff, without
        // the cut-off reply or the result too large to carry.
        let rebuilt = &script.rebuilt[0];
        assert!(rebuilt.contains(HANDOFF) && rebuilt.contains("Add the waves"));
        assert!(rebuilt.ends_with("assistant") && !rebuilt.contains("LLLLL"));
        assert!(!rebuilt.contains("xxxx"));
    }

    #[test]
    fn a_new_message_into_a_crowded_context_starts_from_a_handoff() {
        const CONTEXT: usize = 8192;
        let tokenizer = tiny_tokenizer();
        let format = tiny_linear_format();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let cancel = AtomicBool::new(false);
        let mut session = Session::new(CONTEXT).unwrap();
        session
            .begin_turn(&format.prompt(&tokenizer, "Remember the blue door", true, None))
            .unwrap();
        let mut long = std::iter::repeat_n(u32::from(b'y'), 6200).chain([t.end]);
        session
            .generate(
                7000,
                &[t.end],
                &cancel,
                |_| Ok(long.next().unwrap()),
                |_| Ok(()),
            )
            .unwrap();
        let reply_start = session.tokens().len();
        let mut script = FlowScript {
            tokenizer: &tokenizer,
            t,
            rounds: 0,
            called: 0,
            queue: std::collections::VecDeque::default(),
            longest: 0,
            shown: 0,
            rebuilt: Vec::new(),
        };
        let mut output = Vec::new();
        run_with(
            &format,
            None,
            &tokenizer,
            CONTEXT,
            200,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"And now?\n/undo\n/quit\n"[..],
            &mut output,
            |context| Ok(script.next(context)),
            ChatOptions {
                resumed: Some(Resumed {
                    session,
                    reply_start,
                    held: None,
                }),
                ..ChatOptions::default()
            },
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(
            output.contains("[context rebuilt from the handoff"),
            "{output}"
        );
        assert!(output.contains("Last turn removed."), "{output}");
        let rebuilt = &script.rebuilt[0];
        assert!(rebuilt.contains(HANDOFF));
        assert!(rebuilt.contains("- Remember the blue door"), "{rebuilt}");
        assert!(rebuilt.contains("And now?") && !rebuilt.contains("yyyy"));
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
    // scripted "model" makes the same call every round, as Kimi did on 2026-09-27 and
    // 2026-10-08. After the second repeat her tools close: she either answers, or the
    // call she tries again ends her reply where it would begin.
    #[test]
    #[ignore = "requires KIMI_LINEAR_CHECKPOINT tokenizer files"]
    fn a_second_repeated_tool_call_closes_tools_and_kimi_answers() {
        const ANSWER: &str = "wg-gcp.conf is not in the archive; cas_find found no .conf files.";
        let dir = std::env::var("KIMI_LINEAR_CHECKPOINT").expect("set checkpoint directory");
        let tokenizer = Tokenizer::load(dir).unwrap();
        let format = ChatFormat::kimi_linear(&tokenizer, 163_586).unwrap();
        for insists in [false, true] {
            let ran = std::rc::Rc::new(std::cell::Cell::new(0));
            let mut tools = Toolbox::default();
            tools.push(Box::new(Counting(ran.clone())));
            let reply = |n: usize| {
                tokenizer.encode(&if n == 3 && !insists {
                    format!("{ANSWER}<|im_end|>")
                } else {
                    format!(
                        "Let me look.<|tool_calls_section_begin|><|tool_call_begin|>\
                         functions.cas_find:{n}<|tool_call_argument_begin|>\
                         {{\"pattern\": \"**/*.conf\"}}<|tool_call_end|>\
                         <|tool_calls_section_end|><|im_end|>"
                    )
                })
            };
            let (mut round, mut at, mut replies) = (0, 0, 0);
            let (mut saw_repeated, mut saw_closed) = (false, false);
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
                    if at == 0 {
                        replies += 1;
                        let text = tokenizer.decode_lossy(context);
                        if round == 2 {
                            saw_repeated = text.contains(REPEATED_CALL);
                        }
                        if round == 3 {
                            saw_closed =
                                text.contains(TOOLS_CLOSED) && text.ends_with(ANSWER_OPENING);
                        }
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
            assert_eq!(ran.get(), 1, "the repeats were not run: {text}");
            assert_eq!(replies, 4, "the closing reply was the last: {text}");
            // An insisted call ends the reply at its opening token: the history never
            // holds a call without a result.
            assert_eq!(round, if insists { 3 } else { 4 }, "{text}");
            assert!(saw_repeated, "the model was told why its call was not run");
            assert!(
                saw_closed,
                "tools closed and the answer begun for her: {text}"
            );
            assert!(text.contains("[tool call repeated: cas_find; not run again]"));
            assert!(text.contains(ANSWER_OPENING), "{text}");
            assert!(text.contains("[tools closed: Kimi repeated the same tool call"));
            assert_eq!(text.contains(ANSWER), !insists, "{text}");
            assert!(!text.contains("[stopped:"), "{text}");
        }
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
            None,
        );
        assert!(
            tokenizer
                .decode_lossy(&results)
                .contains("## Return of functions.fs_read:0")
        );
        assert_eq!(
            format.tool_calls(&tokenizer, &tokenizer.encode("no calls<|im_end|>")),
            Vec::<(String, String)>::new()
        );
    }

    #[test]
    fn terminal_controls_are_escaped() {
        assert_eq!(
            terminal_text("a\x1b]52;secret\x07\nb"),
            "a\\u{1b}]52;secret\\u{7}\nb"
        );
    }

    /// Compacts a real saved chat with the real tokenizer and prints what the rebuilt
    /// context holds. No weights are read: the handoff is a fixed line.
    /// `KIMI_FLOW_STATE=<chat>.state.json cargo test -p kimi-k3-cli -- --ignored --nocapture a_real_saved_chat`
    #[test]
    #[ignore = "requires KIMI_LINEAR_CHECKPOINT tokenizer files and KIMI_FLOW_STATE"]
    fn a_real_saved_chat_compacts_within_the_context() {
        let dir = std::env::var("KIMI_LINEAR_CHECKPOINT").expect("set checkpoint directory");
        let tokenizer = Tokenizer::load(dir).unwrap();
        let state = std::env::var("KIMI_FLOW_STATE").expect("set a .state.json path");
        let state: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(state).unwrap()).unwrap();
        let numbers = |key: &str| -> Vec<usize> {
            let list = state[key].as_array().unwrap().iter();
            list.map(|n| usize::try_from(n.as_u64().unwrap()).unwrap())
                .collect()
        };
        let max_context = numbers("tokens").len().max(32_768);
        let session = Session::restore(
            max_context,
            numbers("tokens")
                .into_iter()
                .map(|n| u32::try_from(n).unwrap())
                .collect(),
            numbers("turns"),
            state["pending"].as_bool().unwrap(),
        )
        .unwrap();
        let format = ChatFormat::kimi_linear(&tokenizer, 0).unwrap();
        let ChatFormat::KimiLinear(t) = &format else {
            unreachable!()
        };
        let mut handoff = tokenizer.encode_ordinary(HANDOFF.strip_prefix(HANDOFF_OPENING).unwrap());
        handoff.push(t.end);
        let mut handoff = handoff.into_iter();
        let mut read = 0;
        let flow = Flow {
            enabled: true,
            max_context,
            compacted_len: 0,
        };
        let compacted = compact(
            &format,
            None,
            &tokenizer,
            &session,
            true,
            flow,
            &AtomicBool::new(false),
            &mut |context: &[u32]| {
                read = context.len();
                Ok(handoff.next().unwrap())
            },
            &mut Vec::new(),
        )
        .unwrap();
        let rebuilt = compacted.session;
        println!(
            "{} tokens -> {} (handoff written after reading {read}); pending {}; turns {:?}\n{}",
            session.tokens().len(),
            rebuilt.tokens().len(),
            rebuilt.is_pending(),
            rebuilt.turn_starts(),
            tokenizer.decode_lossy(rebuilt.tokens())
        );
        assert!(rebuilt.tokens().len() < flow.limit());
        assert_eq!(rebuilt.is_pending(), session.is_pending());
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
