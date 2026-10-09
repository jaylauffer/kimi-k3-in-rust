//! `--system-one`: Kimi Linear as a typed decision model.
//!
//! loadngo-inference's `system_one` turns each question into lettered options and asks a
//! [`LabelModel`] for the next-token scores of those letters; this is that model. The
//! state is read once, inside a user message after a short system instruction, and the
//! session is kept, so each further question costs only its own tokens and one forward
//! pass. The letters' logits are read straight from the model: nothing is generated.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kimi_k3_core::gemma::{GemmaModel, GemmaSession};
use kimi_k3_core::layer::Accel;
use kimi_k3_core::linear::{LinearModel, LinearSession};
use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_inference::system_one::{self, Calibration, LabelModel, Request};

use crate::chat::ChatFormat;

const INSTRUCTION: &str = "You make careful decisions about the text in the user's message. \
Read it, then answer the question after it with the letter of exactly one option.";

/// Room left after the state for a question and its options.
const QUESTION_ROOM: usize = 1024;

/// A model that reads tokens into a session it can snapshot.
pub trait Reader {
    type Session: Clone;
    fn open(&self, capacity: usize) -> Self::Session;
    /// The tokens a session has read.
    fn ids(session: &Self::Session) -> &[u32];
    /// Whether a session was left unusable (a pass cancelled or failed partway).
    fn broken(session: &Self::Session) -> bool;
    /// Feeds `ids` and returns the logits after the last of them.
    ///
    /// # Errors
    /// The model's error, as text.
    fn read(
        &mut self,
        session: &mut Self::Session,
        ids: &[u32],
        accel: Accel<'_>,
        keep: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>, String>;
}

impl Reader for LinearModel {
    type Session = LinearSession;
    fn open(&self, capacity: usize) -> LinearSession {
        self.session(capacity)
    }
    fn ids(session: &LinearSession) -> &[u32] {
        session.ids()
    }
    fn broken(session: &LinearSession) -> bool {
        session.is_broken()
    }
    fn read(
        &mut self,
        session: &mut LinearSession,
        ids: &[u32],
        accel: Accel<'_>,
        keep: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>, String> {
        self.feed(session, ids, accel, keep)
            .map_err(|e| e.to_string())
    }
}

impl Reader for GemmaModel {
    type Session = GemmaSession;
    fn open(&self, capacity: usize) -> GemmaSession {
        self.session(capacity)
    }
    fn ids(session: &GemmaSession) -> &[u32] {
        session.ids()
    }
    fn broken(session: &GemmaSession) -> bool {
        session.is_broken()
    }
    fn read(
        &mut self,
        session: &mut GemmaSession,
        ids: &[u32],
        accel: Accel<'_>,
        keep: &dyn Fn() -> bool,
    ) -> Result<Vec<f32>, String> {
        self.feed(session, ids, accel, keep)
            .map_err(|e| e.to_string())
    }
}

/// A chat model answering lettered options.
pub struct KimiLabels<'a, M: Reader = LinearModel> {
    model: &'a mut M,
    tokenizer: &'a Tokenizer,
    format: &'a ChatFormat,
    accel: Accel<'a>,
    cancel: &'a AtomicBool,
    /// The state last read, and the session right after it.
    read: Option<(String, M::Session)>,
    /// Forward passes spent reading states and questions.
    pub state_tokens: usize,
    pub question_tokens: usize,
}

impl<'a, M: Reader> KimiLabels<'a, M> {
    pub fn new(
        model: &'a mut M,
        tokenizer: &'a Tokenizer,
        format: &'a ChatFormat,
        accel: Accel<'a>,
        cancel: &'a AtomicBool,
    ) -> Self {
        Self {
            model,
            tokenizer,
            format,
            accel,
            cancel,
            read: None,
            state_tokens: 0,
            question_tokens: 0,
        }
    }

    fn label_id(&self, label: &str) -> Result<u32, String> {
        match self.tokenizer.encode(label).as_slice() {
            [id] => Ok(*id),
            other => Err(format!(
                "option label {label:?} is not one token ({other:?})"
            )),
        }
    }
}

impl<M: Reader> LabelModel for KimiLabels<'_, M> {
    fn label_logits(
        &mut self,
        state: &str,
        question: &str,
        labels: &[String],
    ) -> Result<Vec<f32>, String> {
        let ids: Vec<u32> = labels
            .iter()
            .map(|l| self.label_id(l))
            .collect::<Result<_, _>>()?;
        let cancel = self.cancel;
        let keep = move || !cancel.load(Ordering::Relaxed);
        if self.read.as_ref().map(|(s, _)| s.as_str()) != Some(state) {
            let head = self
                .format
                .open_user_message(self.tokenizer, INSTRUCTION, state)?;
            let mut session = self.model.open(head.len() + QUESTION_ROOM);
            self.model.read(&mut session, &head, self.accel, &keep)?;
            self.state_tokens += head.len();
            self.read = Some((state.to_string(), session));
        }
        let (_, read) = self.read.as_ref().expect("state was just read");
        let mut session = read.clone();
        let tail = self
            .format
            .close_user_message(self.tokenizer, &format!("\n\n{question}"))?;
        self.question_tokens += tail.len();
        let logits = self.model.read(&mut session, &tail, self.accel, &keep)?;
        ids.iter()
            .map(|&id| {
                logits
                    .get(id as usize)
                    .copied()
                    .ok_or_else(|| format!("token {id} is outside the vocabulary"))
            })
            .collect()
    }
}

/// Reads the JSON request at `path`, answers it, and prints the response JSON.
pub fn run<M: Reader>(
    path: &Path,
    temperature: f32,
    model: &mut M,
    tokenizer: &Tokenizer,
    format: &ChatFormat,
    accel: Accel<'_>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let request = Request::from_json(&value)?;
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err("--temperature must be a positive number".into());
    }
    let start = Instant::now();
    let mut labels = KimiLabels::new(model, tokenizer, format, accel, cancel);
    let answers = system_one::answer(&mut labels, &request, Calibration { temperature })?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "system_one: {} questions in {seconds:.2} s ({} state tokens read once, {} question tokens)",
        answers.len(),
        labels.state_tokens,
        labels.question_tokens
    );
    for (id, answer) in &answers {
        let (best, p) = answer.best();
        eprintln!("  {id}: {best} ({:.0}%)", p * 100.0);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&system_one::response_json(&answers))
            .map_err(|e| e.to_string())?
    );
    Ok(())
}
