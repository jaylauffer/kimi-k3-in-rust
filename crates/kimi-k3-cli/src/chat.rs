//! Text-only chat adapters. K3: XTML, format reference checkpoint `encoding_k3.py` at
//! f831ab66814297da540d832a5235f8e904f29d06. Kimi Linear: the `<|im_*|>` format of
//! that checkpoint's `chat_template.jinja` at e1df551a447157d4658b573f9a695d57658590e9.
//! Model-independent state lives in loadngo-inference. Generated tokens are preserved
//! verbatim, including K3's thinking.

use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_inference::Utf8Stream;

use crate::chat_gemma::GemmaTokens;

const OPEN: u32 = 163_587;
const CLOSE: u32 = 163_588;
const SEP: u32 = 163_589;
const END: u32 = 163_586;
const EOS: u32 = 163_585;

pub(crate) fn ordinary(ids: &mut Vec<u32>, tokenizer: &Tokenizer, text: &str) {
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

pub(crate) fn prompt(tokenizer: &Tokenizer, text: &str, first: bool) -> Vec<u32> {
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

pub(crate) fn validate(tokenizer: &Tokenizer) -> Result<(), String> {
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

/// A Kimi Linear format with made-up control token ids, for tests with a byte
/// tokenizer.
#[cfg(test)]
pub(crate) fn test_linear_format() -> ChatFormat {
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
    })
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

    /// # Errors
    /// When the tokenizer lacks one of Gemma 4's control tokens as a single id.
    pub fn gemma(tokenizer: &Tokenizer, bos: u32, eos: &[u32]) -> Result<Self, String> {
        Ok(Self::Gemma(GemmaTokens::new(tokenizer, bos, eos)?))
    }

    /// A message in `role` holding `content`.
    pub(crate) fn message(
        &self,
        ids: &mut Vec<u32>,
        tokenizer: &Tokenizer,
        role: &str,
        content: &str,
    ) {
        match self {
            Self::KimiLinear(t) => t.message(ids, tokenizer, role, content),
            Self::Gemma(g) => g.message(ids, tokenizer, role, content),
            Self::K3 => {}
        }
    }

    /// A user message holding `text`.
    pub(crate) fn user_message(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer, text: &str) {
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
    pub(crate) fn reply_header(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer) {
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
    pub(crate) fn tool_calls(&self, tokenizer: &Tokenizer, reply: &[u32]) -> Vec<(String, String)> {
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

    /// What a conversation on the shared loop (`loadngo_inference::agent`) opens with: the
    /// tool declarations and one system message (Kimi Linear), or the system turn holding
    /// both (Gemma).
    ///
    /// # Errors
    /// For K3, which has no tools, or declarations Gemma cannot write.
    pub(crate) fn opening_with(
        &self,
        tokenizer: &Tokenizer,
        system: &str,
        declaration: Option<&str>,
    ) -> Result<Vec<u32>, String> {
        match self {
            Self::KimiLinear(t) => {
                let mut ids = Vec::new();
                if let Some(declaration) = declaration {
                    t.message(&mut ids, tokenizer, "tool_declare", declaration);
                }
                t.message(&mut ids, tokenizer, "system", system);
                Ok(ids)
            }
            Self::Gemma(g) => g.preamble(tokenizer, system, declaration),
            Self::K3 => Err("the K3 chat has no tools or system messages".into()),
        }
    }

    /// Closes the model's last reply as the template writes a finished one, whatever
    /// token ended it.
    pub(crate) fn close_reply(&self, ids: &mut Vec<u32>) {
        match self {
            Self::KimiLinear(t) => ids.push(t.end),
            Self::Gemma(g) => ids.extend([g.turn_end, g.newline]),
            Self::K3 => ids.push(END),
        }
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
    pub(crate) fn tool_results(
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

    pub(crate) fn stops(&self) -> Vec<u32> {
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

    pub(crate) fn opening(&self) -> &'static str {
        match self {
            Self::K3 => "[thinking] ",
            Self::KimiLinear(_) => "Kimi> ",
            Self::Gemma(_) => "Gemma> ",
        }
    }

    pub(crate) fn push(&self, display: &mut Display, tokenizer: &Tokenizer, token: u32) -> String {
        match self {
            Self::K3 => display.push(tokenizer, token),
            Self::Gemma(g) => {
                // A thought channel (empty with thinking off, but the model may still write
                // its name) is not part of the reply.
                if token == g.channel {
                    display.tag = Some((true, Vec::new()));
                    return terminal_text(&display.utf8.finish());
                }
                if token == g.channel_end {
                    display.tag = None;
                    return String::new();
                }
                if display.tag.is_some() {
                    return String::new();
                }
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
pub(crate) fn tool_name(id: &str) -> &str {
    let id = id.strip_prefix("functions.").unwrap_or(id);
    id.split(':').next().unwrap_or(id)
}

/// Do not let model text inject terminal escape sequences (including OSC).
pub(crate) fn terminal_text(text: &str) -> String {
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
pub(crate) struct Display {
    pub(crate) utf8: Utf8Stream,
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

    /// Records each call's arguments; with `cancel`, presses Ctrl-C during its first call.
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
}
