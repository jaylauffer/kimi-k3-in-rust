//! `--voice`: talk to Kimi. On-device speech recognition (loadngo-speech) listens on the
//! default microphone; an utterance that starts with her name ("Kimi, ...", "Hey Kimi,
//! ...") becomes the next chat message, and the rest is ignored. Her reply is printed
//! as usual and then spoken with the system voice, with the microphone paused so she
//! does not hear herself.
//!
//! It plugs into the ordinary chat loop: [`VoiceInput`] is the loop's input (it speaks
//! the finished reply when the loop asks for the next message, then listens), and
//! [`VoiceOutput`] passes the loop's output through while keeping the reply text.

use std::io::{self, BufRead, Read, Write};
use std::sync::{Arc, Mutex, PoisonError};

/// Names the recognizer may hear for "Kimi".
const WAKE_WORDS: [&str; 6] = ["kimi", "kimmy", "kimmie", "kimee", "keemee", "kemi"];

/// The message addressed to Kimi in `heard`, without her name: `Some` only when her
/// name is among the first three words ("Kimi, ...", "Hey Kimi ...", "OK Kimi ...").
pub fn addressed(heard: &str) -> Option<String> {
    let words: Vec<&str> = heard.split_whitespace().collect();
    let clean = |w: &str| {
        w.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };
    let at = words
        .iter()
        .take(3)
        .position(|w| WAKE_WORDS.contains(&clean(w).as_str()))?;
    let rest = words[at + 1..].join(" ");
    let rest = rest.trim_start_matches([',', '.', '!', '?', ' ']).trim();
    // "Kimi." alone: a greeting with nothing asked.
    Some(if rest.is_empty() {
        "Hello.".to_string()
    } else {
        rest.to_string()
    })
}

/// The part of the chat output worth speaking: the last reply after `Kimi> `, up to its
/// statistics line, without markdown symbols.
pub fn spoken_reply(output: &str) -> String {
    let Some(at) = output.rfind("Kimi> ") else {
        return String::new();
    };
    let reply = &output[at + "Kimi> ".len()..];
    let reply = reply.split("\n[").next().unwrap_or(reply);
    reply
        .chars()
        .filter(|c| !matches!(c, '*' | '#' | '`' | '_' | '|'))
        .collect::<String>()
        .trim()
        .to_string()
}

/// Chat output that also keeps what was written since the last message was read.
pub struct VoiceOutput<W> {
    inner: W,
    reply: Arc<Mutex<String>>,
}

impl<W: Write> Write for VoiceOutput<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.reply
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push_str(&String::from_utf8_lossy(&buf[..n]));
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(target_os = "macos")]
pub use apple::VoiceInput;

#[cfg(target_os = "macos")]
mod apple {
    use super::{BufRead, Mutex, PoisonError, Read, addressed, io, spoken_reply};
    use loadngo_speech::{Listener, Speaker};
    use std::sync::Arc;
    use std::time::Duration;

    /// How long a pause ends an utterance.
    const SILENCE: Duration = Duration::from_millis(1200);

    /// The chat loop's input, from speech.
    pub struct VoiceInput {
        listener: Listener,
        speaker: Speaker,
        reply: Arc<Mutex<String>>,
        line: Vec<u8>,
        at: usize,
    }

    impl VoiceInput {
        /// Asks for permission if needed, starts the microphone, and returns the input
        /// with the [`super::VoiceOutput`] to pair it with.
        pub fn start<W: std::io::Write>(
            locale: &str,
            output: W,
        ) -> Result<(Self, super::VoiceOutput<W>), String> {
            loadngo_speech::request_authorization().map_err(|e| e.to_string())?;
            let listener = Listener::start(locale, &["Kimi"]).map_err(|e| e.to_string())?;
            let reply = Arc::new(Mutex::new(String::new()));
            Ok((
                Self {
                    listener,
                    speaker: Speaker::new(locale),
                    reply: Arc::clone(&reply),
                    line: Vec::new(),
                    at: 0,
                },
                super::VoiceOutput {
                    inner: output,
                    reply,
                },
            ))
        }

        /// Speaks the reply written since the last message, if any.
        fn speak_reply(&mut self) {
            let text = {
                let mut written = self.reply.lock().unwrap_or_else(PoisonError::into_inner);
                let text = spoken_reply(&written);
                written.clear();
                text
            };
            if text.is_empty() {
                return;
            }
            self.listener.pause();
            if let Err(e) = self.speaker.speak(&text) {
                eprintln!("(voice: {e})");
            }
            self.listener.resume();
        }
    }

    impl Read for VoiceInput {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let available = self.fill_buf()?;
            let n = available.len().min(out.len());
            out[..n].copy_from_slice(&available[..n]);
            self.consume(n);
            Ok(n)
        }
    }

    impl BufRead for VoiceInput {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            if self.at >= self.line.len() {
                self.speak_reply();
                eprint!("(listening; start with \"Kimi\") ");
                let message = loop {
                    let heard = self
                        .listener
                        .next_utterance(SILENCE)
                        .map_err(|e| io::Error::other(e.to_string()))?;
                    if heard.trim().is_empty() {
                        continue;
                    }
                    match addressed(&heard) {
                        Some(message) => break message,
                        None => eprint!("\n(not for Kimi: {heard}) "),
                    }
                };
                eprintln!();
                println!("{message}");
                self.line = format!("{message}\n").into_bytes();
                self.at = 0;
            }
            Ok(&self.line[self.at..])
        }

        fn consume(&mut self, n: usize) {
            self.at = (self.at + n).min(self.line.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_messages_that_name_kimi_first_are_taken() {
        assert_eq!(
            addressed("Kimi, what is the capital of Japan?").as_deref(),
            Some("what is the capital of Japan?")
        );
        assert_eq!(
            addressed("Hey Kimmy tell me a joke.").as_deref(),
            Some("tell me a joke.")
        );
        assert_eq!(addressed("Kimi.").as_deref(), Some("Hello."));
        assert_eq!(addressed("I was talking to Jim about Kimi yesterday"), None);
        assert_eq!(addressed("What time is it?"), None);
    }

    #[test]
    fn the_spoken_reply_is_the_last_answer_without_markup() {
        let out =
            "\nYou> Kimi> **Tokyo** is the capital.\n[EndToken: 7 tokens, 0.4s, context 20/4096]\n";
        assert_eq!(spoken_reply(out), "Tokyo is the capital.");
        assert_eq!(spoken_reply("nothing yet"), "");
    }
}
