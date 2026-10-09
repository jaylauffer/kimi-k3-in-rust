//! Kimi Linear, Gemma 4 and K3 on loadngo's shared chat loop (`loadngo_inference::agent`,
//! loadngo `docs/AGENT_LOOP.md`), the loop gpt-oss runs on too. This file connects the
//! models' chat formats (a [`Template`]), their engines (a [`Backend`]) and the terminal
//! or voice; the turn loop, the tools, the guards, Jev's questions, pauses and
//! `/continue`, compaction through a handoff and saved chats are the loop's.
//!
//! Since 2026-10-09 this is the only chat; the one before it (kimi `chat::run_with`) was
//! removed once the loop had everything it had.

use std::cell::Cell;
use std::fmt::Write as _;
use std::io::{BufRead, Read as _, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kimi_k3_core::layer::Accel;
use kimi_k3_core::model::argmax;
use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_inference::agent::transcript::{Tee, Transcript};
use loadngo_inference::agent::workspace::{Workspace, WorkspaceOptions};
use loadngo_inference::agent::{
    self, Agent, Backend, Budget, Call, Ended, Event, Observer, Prompt, Read, Rendered, Template,
    TurnEnd,
};
use loadngo_inference::system_one::LabelModel;

use crate::chat::{ChatFormat, Display, ordinary, terminal_text, tool_name};
use crate::system_one::{KimiLabels, Reader};
use crate::{Args, thermal};

/// The name Kimi's edits are claimed under on the board (for Gemma too: the chat is
/// hers).
const AGENT: &str = "Kimi";

const HELP: &str = "Type a message and press Enter. Commands: /continue (goes on with a paused \
turn), /undo (drops the last exchange; file changes stay), /reset (starts over), /stats, /help, \
/quit. Ctrl-C pauses a turn (/continue goes on; a new message answers waiting tool calls as not \
run); at the prompt it quits, as does Ctrl-D. Chats are saved; --resume latest carries one on.";

/// Where Kimi's chats are saved.
#[must_use]
pub fn transcripts() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".loadngo/kimi/transcripts"))
}

/// The tools Kimi works with in `base`: loadngo's workspace tools (files, editing under
/// COLLABORATION.md, cargo and git, the Archive CAS, notes, the web) and her read-only
/// board tools. Gemma's template test declares exactly these.
///
/// # Errors
/// When the workspace does not exist or the CAS key cannot be read.
pub fn workspace(
    base: &Path,
    memory: Option<PathBuf>,
    web: bool,
    cas_roots: Vec<PathBuf>,
    cas_key: Option<PathBuf>,
) -> Result<Workspace, String> {
    let mut workspace = Workspace::new(&WorkspaceOptions {
        base: base.to_path_buf(),
        agent: AGENT.into(),
        today: agent::clock::now().date,
        edit: true,
        memory,
        web,
        cas_roots,
        cas_key,
    })?;
    for tool in crate::board::Board::new(base.join("AGENT-BOARD.md")).into_tools() {
        workspace.tools.push(tool);
    }
    workspace.described.push(
        "board: board_sections and board_read (read-only; edits claim on the board themselves)"
            .into(),
    );
    Ok(workspace)
}

/// The workspace these flags ask for, or none with `--no-tools`.
///
/// # Errors
/// As [`workspace`].
pub fn workspace_for(args: &Args) -> Result<Option<Workspace>, String> {
    if args.no_tools {
        eprintln!("tools: off (--no-tools)");
        return Ok(None);
    }
    let base = args
        .fs_base
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| ".".into());
    let memory = (!args.no_memory)
        .then(|| std::env::var_os("HOME"))
        .flatten()
        .map(|home| PathBuf::from(home).join(".loadngo/kimi/memory.jsonl"));
    let workspace = workspace(
        &base,
        memory,
        !args.no_web,
        args.cas_root.iter().cloned().collect(),
        args.cas_key.clone(),
    )?;
    for line in &workspace.described {
        eprintln!("{line}");
    }
    Ok(Some(workspace))
}

/// Kimi Linear's or Gemma's format as the loop's template. Both append: the opening
/// (tool declarations and instructions) is read once, and each message adds a system
/// note (the time, facts about the chat), the message and the reply's opening.
pub struct KimiTemplate<'t> {
    format: &'t ChatFormat,
    tokenizer: &'t Tokenizer,
    /// `Kimi` or `Gemma`.
    identity: &'static str,
    /// Notes for every conversation beyond the instructions (current research, voice).
    extra: String,
    stops: Vec<u32>,
}

impl<'t> KimiTemplate<'t> {
    pub fn new(
        format: &'t ChatFormat,
        tokenizer: &'t Tokenizer,
        identity: &'static str,
        extra: String,
    ) -> Self {
        Self {
            format,
            tokenizer,
            identity,
            extra,
            stops: format.stops(),
        }
    }

    /// The reply as the terminal shows it, tool calls as `[tool call …]`.
    fn text(&self, reply: &[u32]) -> String {
        let mut display = Display::default();
        let mut out = String::new();
        for &token in reply {
            out.push_str(&self.format.push(&mut display, self.tokenizer, token));
        }
        out.push_str(&terminal_text(&display.utf8.finish()));
        out.trim().to_owned()
    }
}

impl Template for KimiTemplate<'_> {
    fn opening(&self, instructions: &str, tools: Option<&str>) -> Result<Vec<u32>, String> {
        if matches!(self.format, ChatFormat::K3) {
            // K3's opening is part of its first message (`chat::prompt`).
            return Ok(Vec::new());
        }
        let mut system = format!(
            "You are {}, running locally on Jay's Mac mini.",
            self.identity
        );
        for part in [instructions, self.extra.as_str()] {
            if !part.is_empty() {
                system.push_str("\n\n");
                system.push_str(part);
            }
        }
        self.format.opening_with(self.tokenizer, &system, tools)
    }

    fn render(&self, p: &Prompt<'_>) -> Result<Rendered, String> {
        if matches!(self.format, ChatFormat::K3) {
            let mut ids = Vec::new();
            if !p.history.is_empty() {
                self.format.close_reply(&mut ids);
            }
            ids.extend(crate::chat::prompt(self.tokenizer, p.user, p.first));
            return Ok(Rendered::Append(ids));
        }
        let mut ids = if p.first {
            self.opening(p.instructions, p.tools)?
        } else {
            Vec::new()
        };
        if !p.history.is_empty() {
            self.format.close_reply(&mut ids);
        }
        // The receipts are left out: every call and result is already in the context.
        let mut note = p.now.said.clone();
        if !p.about.is_empty() {
            note.push_str("\n\n");
            note.push_str(p.about);
        }
        self.format
            .message(&mut ids, self.tokenizer, "system", &note);
        self.format.user_message(&mut ids, self.tokenizer, p.user);
        self.format.reply_header(&mut ids, self.tokenizer);
        Ok(Rendered::Append(ids))
    }

    fn stops(&self) -> &[u32] {
        &self.stops
    }

    fn read(&self, reply: &[u32]) -> Read {
        let calls = self
            .format
            .tool_calls(self.tokenizer, reply)
            .into_iter()
            .map(|(id, arguments)| Call {
                name: tool_name(&id).to_owned(),
                id: Some(id),
                arguments,
            })
            .collect();
        Read {
            calls,
            answer: self.text(reply),
            reasoning: String::new(),
            complete: reply.last().is_some_and(|t| self.stops.contains(t)),
        }
    }

    fn results(
        &self,
        _ended_by: Option<u32>,
        results: &[(Call, String)],
    ) -> Result<Vec<u32>, String> {
        let results: Vec<(String, String, String)> = results
            .iter()
            .map(|(call, text)| {
                (
                    call.id.clone().unwrap_or_else(|| call.name.clone()),
                    call.name.clone(),
                    text.clone(),
                )
            })
            .collect();
        // Kimi Linear closes the reply, then each result is a message; Gemma's results
        // go on inside the model's turn, opening with the `<|tool_response>` it stopped on.
        let mut ids = Vec::new();
        if matches!(self.format, ChatFormat::KimiLinear(_)) {
            self.format.close_reply(&mut ids);
        }
        ids.extend(self.format.tool_results(self.tokenizer, &results, None));
        Ok(ids)
    }

    fn note(&self, _ended_by: Option<u32>, text: &str) -> Result<Vec<u32>, String> {
        // As a user message marked automatic: asked as a system message, Kimi answered
        // with one more tool call (2026-10-02).
        let mut ids = Vec::new();
        self.format.close_reply(&mut ids);
        self.format.user_message(
            &mut ids,
            self.tokenizer,
            &format!("[Automatic message from the chat program, not from Jay.] {text}"),
        );
        self.format.reply_header(&mut ids, self.tokenizer);
        Ok(ids)
    }

    fn answer_opening(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        ordinary(&mut ids, self.tokenizer, text);
        Ok(ids)
    }
}

fn pick(logits: &[f32]) -> Result<u32, String> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite logits; refusing to emit a token".into());
    }
    u32::try_from(argmax(logits)).map_err(|e| e.to_string())
}

/// Kimi Linear or Gemma as the loop's backend. The context is kept as tokens; the
/// model's session follows it, fed only what it has not read. After `/undo` or `/reset`
/// the session restarts from the opening it read once (sessions cannot go back), so only
/// the rest is read again.
pub struct Engine<'a, M: Reader> {
    model: &'a mut M,
    tokenizer: &'a Tokenizer,
    format: &'a ChatFormat,
    accel: Accel<'a>,
    cancel: &'a AtomicBool,
    gate: &'a mut thermal::Gate,
    session: M::Session,
    /// The session right after the opening.
    opening: Option<M::Session>,
    /// The context: what the session should have read.
    held: Vec<u32>,
    /// The scores after the last token read, when the session matches `held`.
    logits: Option<Vec<f32>>,
    capacity: usize,
    /// The last reply's speed, in tokens per second.
    pub rate: Option<f64>,
}

impl<'a, M: Reader> Engine<'a, M> {
    pub fn new(
        model: &'a mut M,
        tokenizer: &'a Tokenizer,
        format: &'a ChatFormat,
        accel: Accel<'a>,
        cancel: &'a AtomicBool,
        gate: &'a mut thermal::Gate,
        capacity: usize,
    ) -> Self {
        let session = model.open(capacity);
        Self {
            model,
            tokenizer,
            format,
            accel,
            cancel,
            gate,
            session,
            opening: None,
            held: Vec::new(),
            logits: None,
            capacity,
            rate: None,
        }
    }

    /// Brings the session to `held`: from where it is when it is a prefix with scores to
    /// go on from, else from the opening, else from nothing.
    fn sync(&mut self) -> Result<(), String> {
        let read = M::ids(&self.session);
        let fits = !M::broken(&self.session)
            && self.held.starts_with(read)
            && (read.len() < self.held.len() || self.logits.is_some());
        if !fits {
            self.logits = None;
            self.session = match &self.opening {
                Some(o)
                    if self.held.starts_with(M::ids(o)) && M::ids(o).len() < self.held.len() =>
                {
                    o.clone()
                }
                _ => self.model.open(self.capacity),
            };
        }
        let done = M::ids(&self.session).len();
        if done < self.held.len() {
            self.gate.checkpoint(self.cancel)?;
            let cancel = self.cancel;
            let keep = move || !cancel.load(Ordering::Relaxed);
            self.logits =
                Some(
                    self.model
                        .read(&mut self.session, &self.held[done..], self.accel, &keep)?,
                );
        }
        Ok(())
    }
}

impl<M: Reader> Backend for Engine<'_, M> {
    /// Replaces the context. The first load is the opening every conversation starts
    /// from, kept as a snapshot; a later one (a context rebuilt from a handoff) starts
    /// from that snapshot when it begins with the opening.
    fn load(&mut self, tokens: &[u32]) -> Result<(), String> {
        self.held = tokens.to_vec();
        self.logits = None;
        self.sync()?;
        if self.opening.is_none() {
            self.opening = Some(self.session.clone());
        }
        Ok(())
    }

    fn held(&self) -> &[u32] {
        &self.held
    }

    fn feed(&mut self, tokens: &[u32]) -> Result<(), String> {
        if self.held.len() + tokens.len() > self.capacity {
            return Err("the context is full; /undo or /reset make room".into());
        }
        self.held.extend_from_slice(tokens);
        self.sync()
    }

    fn generate(
        &mut self,
        limit: usize,
        stops: &[u32],
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Vec<u32>, Ended), String> {
        self.sync()?;
        let started = Instant::now();
        let mut out = Vec::new();
        let ended = loop {
            if cancel.load(Ordering::Relaxed) {
                break Ended::Cancelled;
            }
            if out.len() == limit {
                break Ended::Limit;
            }
            if self.held.len() + 1 >= self.capacity {
                break Ended::Context;
            }
            let next = pick(self.logits.as_deref().ok_or("no scores to go on from")?)?;
            out.push(next);
            if stops.contains(&next) {
                break Ended::Stop;
            }
            self.held.push(next);
            let read = self.sync();
            if let Err(e) = read {
                if cancel.load(Ordering::Relaxed) {
                    break Ended::Cancelled;
                }
                return Err(e);
            }
            if !emit(next) {
                break Ended::Halted;
            }
        };
        if out.len() > 1 {
            #[allow(clippy::cast_precision_loss)] // a rate, shown to whole tokens
            let rate = out.len() as f64 / started.elapsed().as_secs_f64();
            self.rate = Some(rate);
        }
        Ok((out, ended))
    }

    fn truncate(&mut self, len: usize) -> Result<(), String> {
        if len < self.held.len() {
            self.held.truncate(len);
            self.logits = None;
        }
        Ok(())
    }

    fn position(&self) -> usize {
        self.held.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn judge(&mut self, _date: &str) -> Option<Box<dyn LabelModel + '_>> {
        Some(Box::new(KimiLabels::new(
            &mut *self.model,
            self.tokenizer,
            self.format,
            self.accel,
            self.cancel,
        )))
    }
}

/// The terminal (or voice): the reply as it is written, after `Kimi> ` (the voice
/// speaks what follows the last one, up to a `\n[` line); everything else on stderr.
struct Screen<'t> {
    output: Box<dyn Write>,
    format: &'t ChatFormat,
    tokenizer: &'t Tokenizer,
    display: Display,
    call: Option<(String, String)>,
    /// The last reply's speed, for what the model is told about itself.
    rate: Rc<Cell<Option<f64>>>,
}

impl Screen<'_> {
    fn out(&mut self, text: &str) {
        // A closed stdout ends the chat at the next read; nothing to do here.
        let _ = self
            .output
            .write_all(text.as_bytes())
            .and_then(|()| self.output.flush());
    }
}

impl Observer for Screen<'_> {
    #[allow(clippy::cast_precision_loss)] // rates, shown to one decimal
    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::User(_) | Event::State(_) => {}
            Event::Prompt { tokens, seconds } => {
                eprintln!(
                    "[{tokens} tokens read in {seconds:.1}s ({:.0} tokens/s)]",
                    tokens as f64 / seconds.max(1e-9)
                );
                self.display = Display::default();
                let opening = self.format.opening();
                self.out(&format!("\n{opening}"));
            }
            Event::Token(token) => {
                let text = self.format.push(&mut self.display, self.tokenizer, token);
                self.out(&text);
            }
            Event::Reply {
                tokens,
                seconds,
                ended,
                ..
            } => {
                let rest = terminal_text(&self.display.utf8.finish());
                self.out(&format!("{rest}\n"));
                if tokens > 1 {
                    self.rate.set(Some(tokens as f64 / seconds.max(1e-9)));
                }
                eprintln!(
                    "[{ended:?}: {tokens} tokens, {seconds:.1}s, {:.1} tokens/s]",
                    tokens as f64 / seconds.max(1e-9)
                );
            }
            Event::Call { name, arguments } => {
                self.call = Some((name.to_owned(), arguments.trim().to_owned()));
            }
            Event::Result { name, text, ok } => {
                let arguments = self
                    .call
                    .take()
                    .filter(|(n, _)| n == name)
                    .map_or_else(String::new, |(_, a)| a);
                eprintln!(
                    "[tool] {name} {arguments} -> {} characters",
                    text.chars().count()
                );
                if !ok {
                    let first: String = text.chars().take(220).collect();
                    eprintln!("        {}", first.replace('\n', " "));
                }
            }
            Event::Note(note) => eprintln!("{note}"),
            Event::TurnEnd {
                stopped,
                replies,
                tokens,
                seconds,
            } => eprintln!(
                "[turn: {}; {replies} replies, {tokens} tokens, {seconds:.0}s]",
                stopped.unwrap_or("answered")
            ),
        }
    }
}

/// What the chat is told about itself each turn.
fn about(
    model: &str,
    engine: &str,
    capacity: usize,
    transcript: Option<&str>,
    rate: Option<f64>,
) -> String {
    let mut about = format!(
        "About this chat: you are {model}, running locally on Jay's Mac mini on the \
         kimi-k3-in-rust engine ({engine}), with a context of {capacity} tokens. Nothing you \
         read leaves this machine except web_search and web_fetch."
    );
    if let Some(path) = transcript {
        let _ = write!(about, " This chat is saved as JSON lines in {path}.");
    }
    if let Some(rate) = rate {
        let _ = write!(
            about,
            " Your last reply was generated at {rate:.0} tokens/s."
        );
    }
    about
}

/// A model as a next-token function over the whole context, as K3's engine is: it keeps
/// its own session and feeds only what it has not read, rebuilding when the context no
/// longer extends it.
pub struct NextToken<F> {
    next: F,
    held: Vec<u32>,
    capacity: usize,
}

impl<F: FnMut(&[u32]) -> Result<u32, String>> NextToken<F> {
    pub fn new(next: F, capacity: usize) -> Self {
        Self {
            next,
            held: Vec::new(),
            capacity,
        }
    }
}

impl<F: FnMut(&[u32]) -> Result<u32, String>> Backend for NextToken<F> {
    fn load(&mut self, tokens: &[u32]) -> Result<(), String> {
        tokens.clone_into(&mut self.held);
        Ok(())
    }

    fn feed(&mut self, tokens: &[u32]) -> Result<(), String> {
        if self.held.len() + tokens.len() > self.capacity {
            return Err("the context is full; /undo or /reset make room".into());
        }
        self.held.extend_from_slice(tokens);
        Ok(())
    }

    fn generate(
        &mut self,
        limit: usize,
        stops: &[u32],
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(Vec<u32>, Ended), String> {
        let mut out = Vec::new();
        Ok(loop {
            if cancel.load(Ordering::Relaxed) {
                break (out, Ended::Cancelled);
            }
            if out.len() == limit {
                break (out, Ended::Limit);
            }
            if self.held.len() + 1 >= self.capacity {
                break (out, Ended::Context);
            }
            let next = match (self.next)(&self.held) {
                Ok(next) => next,
                Err(_) if cancel.load(Ordering::Relaxed) => break (out, Ended::Cancelled),
                Err(e) => return Err(e),
            };
            out.push(next);
            if stops.contains(&next) {
                break (out, Ended::Stop);
            }
            self.held.push(next);
            if !emit(next) {
                break (out, Ended::Halted);
            }
        })
    }

    fn truncate(&mut self, len: usize) -> Result<(), String> {
        self.held.truncate(len);
        Ok(())
    }

    fn held(&self) -> &[u32] {
        &self.held
    }

    fn position(&self) -> usize {
        self.held.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn judge(&mut self, _date: &str) -> Option<Box<dyn LabelModel + '_>> {
        None
    }
}

/// What the chat says about the model.
#[derive(Clone, Copy)]
pub struct About<'a> {
    /// `Kimi Linear 48B-A3B (Moonshot AI, open weights)`.
    pub description: &'static str,
    /// `Kimi`, `Gemma`.
    pub identity: &'static str,
    /// `the GPU`.
    pub engine: &'static str,
    pub format: &'a ChatFormat,
}

/// A model that reads into sessions (Kimi Linear, Gemma).
pub struct Loaded<'a, M: Reader> {
    pub model: &'a mut M,
    pub about: About<'a>,
    pub accel: Accel<'a>,
}

/// The chat for a [`Reader`] model, through an [`Engine`].
///
/// # Errors
/// As [`run`].
#[allow(clippy::too_many_arguments)]
pub fn run_reader<M: Reader>(
    args: &Args,
    loaded: Loaded<'_, M>,
    tokenizer: &Tokenizer,
    extra: String,
    max_context: usize,
    gen_tokens: usize,
    gate: &mut thermal::Gate,
    cancel: &AtomicBool,
    generating: &AtomicBool,
) -> Result<(), String> {
    let Loaded {
        model,
        about,
        accel,
    } = loaded;
    let mut engine = Engine::new(
        model,
        tokenizer,
        about.format,
        accel,
        cancel,
        gate,
        max_context,
    );
    let workspace = workspace_for(args)?;
    run(
        args,
        about,
        tokenizer,
        extra,
        gen_tokens,
        &mut engine,
        workspace,
        cancel,
        generating,
    )
}

/// The interactive chat on the shared loop, over any backend.
///
/// # Errors
/// When the opening, a saved chat or the terminal fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn run(
    args: &Args,
    about_model: About<'_>,
    tokenizer: &Tokenizer,
    extra: String,
    gen_tokens: usize,
    backend: &mut dyn Backend,
    workspace: Option<Workspace>,
    cancel: &AtomicBool,
    generating: &AtomicBool,
) -> Result<(), String> {
    let About {
        description,
        identity,
        engine: engine_name,
        format,
    } = about_model;
    let jev = !args.no_checkpoint;
    eprintln!(
        "jev: {}",
        if jev {
            "checkpoints every 6 tool calls; web gate"
        } else {
            "off (--no-checkpoint)"
        }
    );
    let (input, output) = crate::linear::chat_io(args)?;
    let rate = Rc::new(Cell::new(None));
    let mut observers: Vec<Box<dyn Observer + '_>> = vec![Box::new(Screen {
        output,
        format,
        tokenizer,
        display: Display::default(),
        call: None,
        rate: Rc::clone(&rate),
    })];
    let mut saved_path = None;
    let mut saved = None;
    if !args.no_transcript {
        if let Some(dir) = transcripts() {
            let opened = match &args.resume {
                Some(which) => Transcript::resume(&dir, which, format.name()).map(|(t, state)| {
                    saved = Some(state);
                    t
                }),
                None => {
                    Transcript::create(&dir, format.name(), &args.model_dir.display().to_string())
                }
            };
            match opened {
                Ok(t) => {
                    eprintln!("transcript: {}", t.path().display());
                    saved_path = Some(t.path().display().to_string());
                    observers.push(Box::new(t));
                }
                Err(e) if args.resume.is_some() => return Err(e),
                Err(e) => eprintln!("transcript: {e}; the chat is not saved"),
            }
        }
    }
    let template = KimiTemplate::new(format, tokenizer, identity, extra);
    let mut chat = Agent::new(template, workspace, jev, Box::new(Tee(observers)));
    chat.set_budget(args.budget);
    let started = Instant::now();
    if let Some(saved) = &saved {
        chat.restore(saved, backend)?;
        eprintln!(
            "resumed: {} exchanges, {} context tokens in {:.1?}{}",
            chat.history().len(),
            backend.position(),
            started.elapsed(),
            if chat.pending().is_some() {
                "; a paused turn waits for /continue"
            } else {
                ""
            }
        );
    } else {
        let opening = chat.prepare(backend)?;
        if opening > 0 {
            eprintln!(
                "read the opening ({opening} tokens) in {:.1?}; each conversation starts from it",
                started.elapsed()
            );
        }
    }
    println!("Local {description} on {engine_name}. {HELP}");
    let max_context = backend.capacity();
    let mut input = input;
    let show = |end: &TurnEnd| {
        if let Some(why) = &end.paused {
            eprintln!(
                "[paused: {why}. /continue goes on; a new message answers waiting tool calls as \
                 not run]"
            );
        }
    };
    loop {
        generating.store(false, Ordering::Relaxed);
        print!("\nYou> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        // Bound a terminal or piped line before it is tokenized.
        if (&mut input)
            .take(65_537)
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
        {
            break;
        }
        if line.len() > 65_536 {
            return Err("input line exceeds 64 KiB".into());
        }
        cancel.store(false, Ordering::Relaxed);
        match line.trim() {
            "" => {}
            "/quit" | "/exit" => break,
            "/help" => println!("{HELP}"),
            "/stats" => println!(
                "{} / {max_context} context tokens; {} exchanges; paused turn: {}",
                backend.position(),
                chat.history().len(),
                chat.pending().is_some()
            ),
            "/continue" => {
                generating.store(true, Ordering::Relaxed);
                match agent::resume(&mut chat, backend, gen_tokens, cancel) {
                    Ok(Some(end)) => show(&end),
                    Ok(None) => println!("Nothing to continue."),
                    Err(e) => eprintln!("[{e}; /undo removes this turn]"),
                }
                chat.save(backend);
            }
            "/reset" => {
                let keep = chat.reset();
                backend.truncate(keep)?;
                chat.save(backend);
                println!("Conversation cleared.");
            }
            "/undo" => match chat.undo() {
                Some(at) => {
                    backend.truncate(at)?;
                    chat.save(backend);
                    println!("Last exchange removed; file changes stay.");
                }
                None => println!("Nothing to undo."),
            },
            command if command.starts_with('/') => println!("Unknown command; /help lists them."),
            text => {
                chat.set_about(about(
                    description,
                    engine_name,
                    max_context,
                    saved_path.as_deref(),
                    rate.get(),
                ));
                generating.store(true, Ordering::Relaxed);
                match agent::turn(&mut chat, backend, text, gen_tokens, cancel) {
                    Ok(end) => show(&end),
                    Err(e) => eprintln!("[{e}; /undo removes this turn]"),
                }
            }
        }
    }
    generating.store(false, Ordering::Relaxed);
    let left = chat.finish();
    if !left.is_empty() {
        eprintln!("[board] handed off on AGENT-BOARD.md; uncommitted changes:");
        for (area, file) in left {
            eprintln!("  {area}: {file}");
        }
    }
    Ok(())
}

/// Kimi's turn budget, from `--turn-minutes` (default 30) and `--turn-tokens` (default
/// 16,384); 0 turns either off.
#[must_use]
pub fn default_budget() -> Budget {
    Budget {
        time: Some(std::time::Duration::from_secs(30 * 60)),
        tokens: Some(16_384),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    use loadngo_inference::agent::clock;

    const SYSTEM: u32 = 200_000;
    const USER: u32 = 200_001;
    const ASSISTANT: u32 = 200_002;
    const END: u32 = 200_004;
    const SECTION: u32 = 200_005;
    const SECTION_END: u32 = 200_006;
    const CALL: u32 = 200_007;
    const ARGUMENT: u32 = 200_008;
    const CALL_END: u32 = 200_009;

    fn tokenizer() -> Tokenizer {
        Tokenizer::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tokenizer/byte_chat"),
        )
        .unwrap()
    }

    fn prompt<'a>(
        now: &'a clock::Now,
        user: &'a str,
        history: &'a [agent::Exchange],
    ) -> Prompt<'a> {
        Prompt {
            now,
            instructions: "Work carefully.",
            about: "About this chat.",
            evidence: "RECEIPTS",
            tools: Some("[]"),
            history,
            user,
            first: history.is_empty(),
        }
    }

    #[test]
    fn kimi_linear_turns_close_the_last_reply_and_never_repeat_the_receipts() {
        let tok = tokenizer();
        let format = crate::chat::test_linear_format();
        let t = KimiTemplate::new(&format, &tok, "Kimi", "Research note.".into());
        let now = clock::now();
        let Rendered::Append(first) = t.render(&prompt(&now, "hi", &[])).unwrap() else {
            panic!("Kimi Linear appends")
        };
        let text = tok.decode_lossy(&first);
        assert!(text.starts_with("tool_declare"), "{text}");
        assert!(text.contains("You are Kimi, running locally on Jay's Mac mini."));
        assert!(text.contains("Work carefully.") && text.contains("Research note."));
        assert!(text.contains(&now.said) && text.contains("About this chat."));
        assert!(
            !text.contains("RECEIPTS"),
            "every call is already in the context"
        );
        assert_eq!(first[0], SYSTEM);
        // The opening is what `prepare` reads once; a later message does not repeat it.
        assert!(first.starts_with(&t.opening("Work carefully.", Some("[]")).unwrap()));
        let history = [agent::Exchange {
            user: "hi".into(),
            answer: "hello".into(),
        }];
        let Rendered::Append(next) = t.render(&prompt(&now, "again", &history)).unwrap() else {
            panic!()
        };
        assert_eq!(next[0], END, "the last reply is closed first");
        assert!(!tok.decode_lossy(&next).contains("tool_declare"));
        assert!(next.contains(&USER));
        assert_eq!(next[next.len() - 1], 200_003, "ends in the reply's opening");
        assert!(next.contains(&ASSISTANT));
    }

    #[test]
    fn kimi_linear_calls_results_and_notes() {
        let tok = tokenizer();
        let format = crate::chat::test_linear_format();
        let t = KimiTemplate::new(&format, &tok, "Kimi", String::new());
        let mut reply = tok.encode_ordinary("Let me look.");
        reply.extend([SECTION, CALL]);
        reply.extend(tok.encode_ordinary("functions.fs_read:0"));
        reply.push(ARGUMENT);
        reply.extend(tok.encode_ordinary(r#"{"path":"a.rs"}"#));
        reply.extend([CALL_END, SECTION_END, END]);
        let read = t.read(&reply);
        assert_eq!(
            read.calls,
            [Call {
                id: Some("functions.fs_read:0".into()),
                name: "fs_read".into(),
                arguments: r#"{"path":"a.rs"}"#.into(),
            }]
        );
        assert!(read.complete);
        assert!(read.answer.starts_with("Let me look."));
        let results = t
            .results(Some(END), &[(read.calls[0].clone(), "fn main() {}".into())])
            .unwrap();
        assert_eq!(results[0], END);
        let text = tok.decode_lossy(&results);
        assert!(
            text.contains("## Return of functions.fs_read:0\nfn main() {}"),
            "{text}"
        );
        assert_eq!(results[results.len() - 1], 200_003);
        let note = t.note(Some(END), "Check first.").unwrap();
        assert_eq!(note[0], END);
        assert!(
            tok.decode_lossy(&note)
                .contains("[Automatic message from the chat program, not from Jay.] Check first.")
        );
        assert_eq!(tok.decode_lossy(&t.answer_opening("So:").unwrap()), "So:");
    }

    /// A model that reads one token per position and scores the next from a script,
    /// counting every token it reads.
    struct Fake {
        script: Vec<u32>,
        reads: Rc<Cell<usize>>,
    }

    #[derive(Clone)]
    struct FakeSession(Vec<u32>);

    impl Reader for Fake {
        type Session = FakeSession;
        fn open(&self, _: usize) -> FakeSession {
            FakeSession(Vec::new())
        }
        fn ids(session: &FakeSession) -> &[u32] {
            &session.0
        }
        fn broken(_: &FakeSession) -> bool {
            false
        }
        fn read(
            &mut self,
            session: &mut FakeSession,
            ids: &[u32],
            _: Accel<'_>,
            _: &dyn Fn() -> bool,
        ) -> Result<Vec<f32>, String> {
            self.reads.set(self.reads.get() + ids.len());
            session.0.extend_from_slice(ids);
            let mut logits = vec![0.0; 300_000];
            let next = self.script[session.0.len() % self.script.len()];
            logits[next as usize] = 1.0;
            Ok(logits)
        }
    }

    #[test]
    fn the_engine_reads_the_opening_once_and_restarts_from_it_after_undo() {
        let tok = tokenizer();
        let format = crate::chat::test_linear_format();
        let reads = Rc::new(Cell::new(0));
        let mut model = Fake {
            script: vec![65, 66, END],
            reads: Rc::clone(&reads),
        };
        let cancel = AtomicBool::new(false);
        let mut gate = thermal::Gate::new().unwrap();
        let mut e = Engine::new(&mut model, &tok, &format, None, &cancel, &mut gate, 1_000);
        e.load(&[1; 100]).unwrap();
        assert_eq!(reads.get(), 100);
        // 111 positions: the script answers 65, 66, then ends.
        e.feed(&[2; 11]).unwrap();
        let (reply, ended) = e.generate(50, &[END], &cancel, &mut |_| true).unwrap();
        assert_eq!((reply.as_slice(), ended), (&[65, 66, END][..], Ended::Stop));
        // The ending token is not read; every other one is.
        assert_eq!(e.position(), 113);
        let before = reads.get();
        // Undo back to the opening, then a new message: only the message is read again.
        e.truncate(100).unwrap();
        e.feed(&[3; 5]).unwrap();
        assert_eq!(reads.get() - before, 5);
        assert_eq!(e.position(), 105);
        // Undo inside the conversation: the session cannot go back, so it restarts from
        // the opening and reads what is kept after it.
        e.feed(&[4; 7]).unwrap();
        e.truncate(103).unwrap();
        let before = reads.get();
        e.feed(&[5; 2]).unwrap();
        assert_eq!(reads.get() - before, 5);
        // A limit and a full context end generation without reading past them.
        let (reply, ended) = e.generate(1, &[END], &cancel, &mut |_| true).unwrap();
        assert_eq!((reply.len(), ended), (1, Ended::Limit));
        assert!(e.feed(&[6; 1_000]).is_err());
        // A context rebuilt from a handoff starts from the opening's snapshot.
        let mut rebuilt = vec![1; 100];
        rebuilt.extend([8; 3]);
        let before = reads.get();
        e.load(&rebuilt).unwrap();
        assert_eq!(reads.get() - before, 3);
        assert_eq!(e.held(), rebuilt);
    }

    #[test]
    fn k3_closes_its_last_reply_and_starts_its_first_message_with_the_opening() {
        let tok = tokenizer();
        let format = ChatFormat::K3;
        let t = KimiTemplate::new(&format, &tok, "Kimi", String::new());
        assert_eq!(t.opening("rules", None).unwrap(), Vec::<u32>::new());
        let now = clock::now();
        let Rendered::Append(first) = t.render(&prompt(&now, "hi", &[])).unwrap() else {
            panic!()
        };
        assert_eq!(first, crate::chat::prompt(&tok, "hi", true));
        let history = [agent::Exchange {
            user: "hi".into(),
            answer: "hello".into(),
        }];
        let Rendered::Append(next) = t.render(&prompt(&now, "again", &history)).unwrap() else {
            panic!()
        };
        assert_eq!(next[0], 163_586, "the last reply's end");
        assert_eq!(&next[1..], &crate::chat::prompt(&tok, "again", false)[..]);
    }

    #[test]
    fn a_next_token_model_sees_the_whole_context_and_its_end_token_is_not_fed() {
        let seen = Rc::new(Cell::new(0));
        let counter = Rc::clone(&seen);
        let mut b = NextToken::new(
            move |ids: &[u32]| {
                counter.set(ids.len());
                Ok(if ids.len() >= 5 { 9 } else { 7 })
            },
            100,
        );
        b.feed(&[1, 2, 3]).unwrap();
        let cancel = AtomicBool::new(false);
        let (out, ended) = b.generate(10, &[9], &cancel, &mut |_| true).unwrap();
        assert_eq!((out.as_slice(), ended), (&[7, 7, 9][..], Ended::Stop));
        assert_eq!(b.held(), [1, 2, 3, 7, 7]);
        assert_eq!(seen.get(), 5);
        b.truncate(3).unwrap();
        assert_eq!(b.position(), 3);
        assert!(b.feed(&[0; 100]).is_err());
        cancel.store(true, Ordering::Relaxed);
        assert_eq!(
            b.generate(10, &[9], &cancel, &mut |_| true).unwrap().1,
            Ended::Cancelled
        );
    }
}
