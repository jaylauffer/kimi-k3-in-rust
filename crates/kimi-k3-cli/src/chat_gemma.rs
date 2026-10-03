//! Gemma 4's chat encoding, from the checkpoint's `chat_template.jinja` ("Google Gemma 4
//! Canonical Chat Template", published 2026-07-09):
//!
//! ```text
//! <bos><|turn>system\n{system}<|tool>declaration:NAME{…}<tool|>…<turn|>\n
//! <|turn>user\n{text}<turn|>\n
//! <|turn>model\n<|channel>thought\n<channel|>{reply}<turn|>\n
//! ```
//!
//! Thinking is off: the reply opens with an empty thought channel, as the template's
//! generation prompt writes it. A tool call is `<|tool_call>call:NAME{key:value,…}<tool_call|>`,
//! strings between `<|"|>` tokens and keys bare. The model then writes `<|tool_response>`
//! (a stop token) and each result follows as `response:NAME{value:<|"|>…<|"|>}<tool_response|>`,
//! inside the same model turn, which the model goes on writing.

use kimi_k3_core::tokenizer::Tokenizer;
use std::fmt::Write as _;

use serde_json::{Map, Value};

/// Gemma 4 control tokens, looked up in the checkpoint's own tokenizer.
pub struct GemmaTokens {
    pub bos: u32,
    pub turn: u32,
    pub turn_end: u32,
    pub newline: u32,
    /// The words after `<|turn>`.
    pub user: u32,
    pub model: u32,
    pub channel: u32,
    pub channel_end: u32,
    pub call: u32,
    pub call_end: u32,
    pub response: u32,
    pub response_end: u32,
    pub quote: u32,
    pub tool: u32,
    pub tool_end: u32,
    /// The config's `eos_token_id`s (`<eos>`, `<turn|>`).
    pub eos: Vec<u32>,
    pub note: Option<String>,
}

fn single(tokenizer: &Tokenizer, text: &str) -> Result<u32, String> {
    match tokenizer.encode(text).as_slice() {
        [id] => Ok(*id),
        other => Err(format!(
            "{text} is not one token in this tokenizer ({other:?})"
        )),
    }
}

fn ordinary(ids: &mut Vec<u32>, tokenizer: &Tokenizer, text: &str) {
    ids.extend(tokenizer.encode_ordinary(text));
}

impl GemmaTokens {
    /// # Errors
    /// When the tokenizer lacks one of the control tokens as a single id.
    pub fn new(tokenizer: &Tokenizer, bos: u32, eos: &[u32]) -> Result<Self, String> {
        Ok(Self {
            bos,
            turn: single(tokenizer, "<|turn>")?,
            turn_end: single(tokenizer, "<turn|>")?,
            newline: single(tokenizer, "\n")?,
            user: single(tokenizer, "user")?,
            model: single(tokenizer, "model")?,
            channel: single(tokenizer, "<|channel>")?,
            channel_end: single(tokenizer, "<channel|>")?,
            call: single(tokenizer, "<|tool_call>")?,
            call_end: single(tokenizer, "<tool_call|>")?,
            response: single(tokenizer, "<|tool_response>")?,
            response_end: single(tokenizer, "<tool_response|>")?,
            quote: single(tokenizer, "<|\"|>")?,
            tool: single(tokenizer, "<|tool>")?,
            tool_end: single(tokenizer, "<tool|>")?,
            eos: eos.to_vec(),
            note: None,
        })
    }

    /// `<|turn>{role}\n{content}<turn|>\n`.
    pub fn message(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer, role: &str, content: &str) {
        ids.push(self.turn);
        ordinary(ids, tokenizer, &format!("{role}\n"));
        ordinary(ids, tokenizer, content.trim());
        ids.push(self.turn_end);
        ids.push(self.newline);
    }

    /// The opening of a model turn, with thinking off.
    pub fn reply_header(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer) {
        ids.push(self.turn);
        ordinary(ids, tokenizer, "model\n");
        ids.push(self.channel);
        ordinary(ids, tokenizer, "thought\n");
        ids.push(self.channel_end);
    }

    /// Where the reply begins in `span`: just past its model header.
    pub fn reply_start(&self, span: &[u32]) -> Option<usize> {
        let header = (0..span.len().saturating_sub(1))
            .find(|&i| span[i] == self.turn && span[i + 1] == self.model)?;
        let after = header + 3;
        // Past the empty thought channel, when the header has one.
        match span
            .get(after..)?
            .iter()
            .position(|&id| id == self.channel_end)
        {
            Some(end) if span.get(after) == Some(&self.channel) => Some(after + end + 1),
            _ => Some(after),
        }
    }

    /// The text of the last user message in `prompt`.
    pub fn user_text<'s>(&self, prompt: &'s [u32]) -> Option<&'s [u32]> {
        let at = (0..prompt.len().saturating_sub(1))
            .rev()
            .find(|&i| prompt[i] == self.turn && prompt[i + 1] == self.user)?;
        let body = prompt.get(at + 3..)?;
        let end = body
            .iter()
            .position(|&id| id == self.turn_end)
            .unwrap_or(body.len());
        Some(&body[..end])
    }

    /// Whether `prompt` holds a user message before any model header.
    pub fn has_user(&self, prompt: &[u32]) -> bool {
        prompt
            .windows(2)
            .take_while(|w| !(w[0] == self.turn && w[1] == self.model))
            .any(|w| w[0] == self.turn && w[1] == self.user)
    }

    /// The system turn every conversation opens with: the guidance and notes, then each
    /// tool's declaration.
    pub fn preamble(
        &self,
        tokenizer: &Tokenizer,
        system: &str,
        declaration: Option<&str>,
    ) -> Result<Vec<u32>, String> {
        let mut ids = vec![self.bos];
        if system.is_empty() && declaration.is_none() {
            return Ok(ids);
        }
        ids.push(self.turn);
        ordinary(&mut ids, tokenizer, "system\n");
        ordinary(&mut ids, tokenizer, system.trim());
        if let Some(declaration) = declaration {
            let tools: Value =
                serde_json::from_str(declaration).map_err(|e| format!("tool declaration: {e}"))?;
            for tool in tools.as_array().into_iter().flatten() {
                ids.push(self.tool);
                self.text(&mut ids, tokenizer, &declare(tool));
                ids.push(self.tool_end);
            }
        }
        ids.push(self.turn_end);
        ids.push(self.newline);
        Ok(ids)
    }

    /// `text` with every `<|"|>` as the quote token and everything else ordinary.
    fn text(&self, ids: &mut Vec<u32>, tokenizer: &Tokenizer, text: &str) {
        for (i, part) in text.split(QUOTE).enumerate() {
            if i > 0 {
                ids.push(self.quote);
            }
            ordinary(ids, tokenizer, part);
        }
    }

    /// Tool results as response blocks. `opened`: the model already wrote the first
    /// `<|tool_response>` (it is a stop token).
    pub fn responses(
        &self,
        ids: &mut Vec<u32>,
        tokenizer: &Tokenizer,
        results: &[(String, String, String)],
        mut opened: bool,
    ) {
        for (_, name, result) in results {
            if !std::mem::take(&mut opened) {
                ids.push(self.response);
            }
            ordinary(ids, tokenizer, &format!("response:{name}{{value:"));
            ids.push(self.quote);
            ordinary(ids, tokenizer, result);
            ids.push(self.quote);
            ordinary(ids, tokenizer, "}");
            ids.push(self.response_end);
        }
    }

    /// Tool calls in a finished reply, as Kimi-style `(functions.NAME:N, JSON)` pairs.
    pub fn tool_calls(&self, tokenizer: &Tokenizer, reply: &[u32]) -> Vec<(String, String)> {
        let mut calls = Vec::new();
        let mut rest = reply;
        while let Some(start) = rest.iter().position(|&id| id == self.call) {
            rest = &rest[start + 1..];
            let end = rest
                .iter()
                .position(|&id| id == self.call_end)
                .unwrap_or(rest.len());
            // The quote token decodes as its text, `<|"|>`, which the parser reads.
            let text = tokenizer.decode_lossy(&rest[..end]);
            if let Some(body) = text.trim().strip_prefix("call:") {
                let split = body.find('{').unwrap_or(body.len());
                let (name, arguments) = body.split_at(split);
                let arguments = parse_arguments(arguments)
                    .map_or_else(|| arguments.to_string(), |v| v.to_string());
                calls.push((
                    format!("functions.{}:{}", name.trim(), calls.len()),
                    arguments,
                ));
            }
            rest = &rest[end..];
        }
        calls
    }
}

const QUOTE: &str = "<|\"|>";

/// A tool's declaration, as the template's `format_function_declaration` writes it.
fn declare(tool: &Value) -> String {
    let function = &tool["function"];
    let mut out = format!(
        "declaration:{}{{description:{QUOTE}{}{QUOTE}",
        function["name"].as_str().unwrap_or_default(),
        function["description"].as_str().unwrap_or_default()
    );
    let params = &function["parameters"];
    if params.as_object().is_some_and(|p| !p.is_empty()) {
        out.push_str(",parameters:{");
        if let Some(properties) = params["properties"].as_object().filter(|p| !p.is_empty()) {
            out.push_str("properties:{");
            out.push_str(&parameters(properties, false));
            out.push_str("},");
        }
        if let Some(required) = params["required"].as_array().filter(|r| !r.is_empty()) {
            out.push_str("required:[");
            out.push_str(&quoted_list(required));
            out.push_str("],");
        }
        if let Some(kind) = params["type"].as_str() {
            let _ = write!(out, "type:{QUOTE}{}{QUOTE}}}", kind.to_uppercase());
        }
    }
    out.push('}');
    out
}

fn quoted_list(items: &[Value]) -> String {
    items
        .iter()
        .map(|item| format!("{QUOTE}{}{QUOTE}", item.as_str().unwrap_or_default()))
        .collect::<Vec<_>>()
        .join(",")
}

/// The template's `format_parameters`.
fn parameters(properties: &Map<String, Value>, filter_keys: bool) -> String {
    const STANDARD: [&str; 5] = ["description", "type", "properties", "required", "nullable"];
    let mut keys: Vec<&String> = properties.keys().collect();
    keys.sort();
    let mut entries = Vec::new();
    for key in keys {
        if filter_keys && STANDARD.contains(&key.as_str()) {
            continue;
        }
        let value = &properties[key];
        let kind = value["type"].as_str().unwrap_or_default().to_uppercase();
        let mut parts = Vec::new();
        if let Some(description) = value["description"].as_str().filter(|d| !d.is_empty()) {
            parts.push(format!("description:{QUOTE}{description}{QUOTE}"));
        }
        if kind == "STRING" {
            if let Some(options) = value["enum"].as_array().filter(|e| !e.is_empty()) {
                parts.push(format!(
                    "enum:{}",
                    argument(&Value::Array(options.clone()), true)
                ));
            }
        } else if kind == "ARRAY" {
            if let Some(items) = value["items"].as_object().filter(|i| !i.is_empty()) {
                let mut item_keys: Vec<&String> = items.keys().collect();
                item_keys.sort();
                let mut inner = Vec::new();
                for item_key in item_keys {
                    let item = &items[item_key];
                    if item.is_null() {
                        continue;
                    }
                    inner.push(match item_key.as_str() {
                        "properties" => format!(
                            "properties:{{{}}}",
                            item.as_object()
                                .map(|p| parameters(p, false))
                                .unwrap_or_default()
                        ),
                        "required" => format!(
                            "required:[{}]",
                            quoted_list(item.as_array().map_or(&[], Vec::as_slice))
                        ),
                        "type" => match item {
                            Value::String(s) => {
                                format!("type:{}", argument(&Value::from(s.to_uppercase()), true))
                            }
                            Value::Array(list) => format!(
                                "type:{}",
                                argument(
                                    &Value::Array(
                                        list.iter()
                                            .map(|v| {
                                                Value::from(
                                                    v.as_str().unwrap_or_default().to_uppercase(),
                                                )
                                            })
                                            .collect()
                                    ),
                                    true
                                )
                            ),
                            other => format!("type:{}", argument(other, true)),
                        },
                        _ => format!("{item_key}:{}", argument(item, true)),
                    });
                }
                parts.push(format!("items:{{{}}}", inner.join(",")));
            }
        }
        if value["nullable"].as_bool() == Some(true) {
            parts.push("nullable:true".into());
        }
        if kind == "OBJECT" {
            if let Some(nested) = value["properties"].as_object() {
                parts.push(format!("properties:{{{}}}", parameters(nested, false)));
            } else if let Some(map) = value.as_object() {
                parts.push(format!("properties:{{{}}}", parameters(map, true)));
            }
            if let Some(required) = value["required"].as_array().filter(|r| !r.is_empty()) {
                parts.push(format!("required:[{}]", quoted_list(required)));
            }
        }
        parts.push(format!("type:{QUOTE}{kind}{QUOTE}"));
        entries.push(format!("{key}:{{{}}}", parts.join(",")));
    }
    entries.join(",")
}

/// The template's `format_argument`.
fn argument(value: &Value, escape_keys: bool) -> String {
    match value {
        Value::Null => "null".into(),
        Value::String(s) => format!("{QUOTE}{s}{QUOTE}"),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|v| argument(v, escape_keys))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let inner: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    let key = if escape_keys {
                        format!("{QUOTE}{k}{QUOTE}")
                    } else {
                        k.clone()
                    };
                    format!("{key}:{}", argument(&map[k], escape_keys))
                })
                .collect();
            format!("{{{}}}", inner.join(","))
        }
    }
}

/// Parses a call's `{key:value,…}`: bare or quoted keys, `<|"|>` strings, numbers,
/// booleans, null, lists and nested objects. `None` when it is not that shape.
pub fn parse_arguments(text: &str) -> Option<Value> {
    let mut parser = Parser { text, at: 0 };
    let value = parser.value()?;
    parser.space();
    (parser.at == text.len() && value.is_object()).then_some(value)
}

struct Parser<'a> {
    text: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn rest(&self) -> &str {
        &self.text[self.at..]
    }

    fn space(&mut self) {
        let trimmed = self.rest().trim_start();
        self.at = self.text.len() - trimmed.len();
    }

    fn eat(&mut self, token: &str) -> bool {
        self.space();
        if self.rest().starts_with(token) {
            self.at += token.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<Value> {
        self.space();
        if self.eat("{") {
            let mut map = Map::new();
            if self.eat("}") {
                return Some(Value::Object(map));
            }
            loop {
                let key = self.key()?;
                if !self.eat(":") {
                    return None;
                }
                map.insert(key, self.value()?);
                if self.eat("}") {
                    return Some(Value::Object(map));
                }
                if !self.eat(",") {
                    return None;
                }
            }
        }
        if self.eat("[") {
            let mut items = Vec::new();
            if self.eat("]") {
                return Some(Value::Array(items));
            }
            loop {
                items.push(self.value()?);
                if self.eat("]") {
                    return Some(Value::Array(items));
                }
                if !self.eat(",") {
                    return None;
                }
            }
        }
        if self.eat(QUOTE) {
            let end = self.rest().find(QUOTE)?;
            let s = self.rest()[..end].to_string();
            self.at += end + QUOTE.len();
            return Some(Value::String(s));
        }
        if self.rest().starts_with('"') {
            // A JSON string, in case the model writes one.
            let mut stream = serde_json::Deserializer::from_str(self.rest()).into_iter::<Value>();
            let value = stream.next()?.ok()?;
            self.at += stream.byte_offset();
            return Some(value);
        }
        let end = self
            .rest()
            .find([',', '}', ']'])
            .unwrap_or(self.rest().len());
        let word = self.rest()[..end].trim().to_string();
        self.at += end;
        match word.as_str() {
            "true" => Some(Value::Bool(true)),
            "false" => Some(Value::Bool(false)),
            "null" => Some(Value::Null),
            _ => serde_json::from_str::<serde_json::Number>(&word)
                .ok()
                .map(Value::Number),
        }
    }

    fn key(&mut self) -> Option<String> {
        self.space();
        if self.eat(QUOTE) {
            let end = self.rest().find(QUOTE)?;
            let key = self.rest()[..end].to_string();
            self.at += end + QUOTE.len();
            return Some(key);
        }
        if self.rest().starts_with('"') {
            let mut stream = serde_json::Deserializer::from_str(self.rest()).into_iter::<Value>();
            let key = stream.next()?.ok()?.as_str()?.to_string();
            self.at += stream.byte_offset();
            return Some(key);
        }
        let end = self.rest().find(':')?;
        let key = self.rest()[..end].trim().to_string();
        self.at += end;
        (!key.is_empty()).then_some(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn call_arguments_parse_from_the_template_syntax() {
        let v = parse_arguments(
            "{path:<|\"|>docs/a b.md<|\"|>,limit:20,deep:true,none:null,tags:[<|\"|>x<|\"|>,<|\"|>y,z<|\"|>],opt:{k:<|\"|>v}<|\"|>}}",
        )
        .unwrap();
        assert_eq!(
            v,
            json!({"path": "docs/a b.md", "limit": 20, "deep": true, "none": null,
                   "tags": ["x", "y,z"], "opt": {"k": "v}"}})
        );
        assert_eq!(parse_arguments("{}").unwrap(), json!({}));
        assert_eq!(
            parse_arguments("{\"path\": \"a\", \"n\": 1.5}").unwrap(),
            json!({"path": "a", "n": 1.5})
        );
        assert!(parse_arguments("{path:<|\"|>unterminated}").is_none());
        assert!(parse_arguments("[1]").is_none());
    }

    #[test]
    fn declarations_follow_the_template() {
        let tool = json!({"type": "function", "function": {
            "name": "fs_read", "description": "Read a file.",
            "parameters": {"type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "mode": {"type": "string", "enum": ["a", "b"]},
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer"}},
                "required": ["path"]}}});
        assert_eq!(
            declare(&tool),
            "declaration:fs_read{description:<|\"|>Read a file.<|\"|>,parameters:{properties:{\
             limit:{type:<|\"|>INTEGER<|\"|>},\
             mode:{enum:[<|\"|>a<|\"|>,<|\"|>b<|\"|>],type:<|\"|>STRING<|\"|>},\
             path:{description:<|\"|>File path.<|\"|>,type:<|\"|>STRING<|\"|>},\
             paths:{items:{type:<|\"|>STRING<|\"|>},type:<|\"|>ARRAY<|\"|>}},\
             required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}"
        );
    }
}

#[cfg(test)]
mod template {
    //! The opening rendered here against the checkpoint's own `chat_template.jinja`.
    //! `tests/fixtures/gemma4/opening.json` holds every chat tool's declaration and the
    //! token ids of the template's rendering of a conversation that declares them (Jinja2
    //! 3.1, `tokenizers` 0.23.2; regenerate with `scripts/gemma_template_fixture.py` when
    //! a tool changes). Needs the checkpoint's tokenizer; skipped without it.

    use super::*;
    use loadngo_inference::tools::Toolbox;
    use std::path::{Path, PathBuf};

    fn toolbox(root: &Path) -> Toolbox {
        let mut tools = Toolbox::default();
        for tool in crate::board::Board::new(root.join("AGENT-BOARD.md"), "2026-10-03").into_tools()
        {
            tools.push(tool);
        }
        for tool in crate::text_tools::tools(root).unwrap() {
            tools.push(tool);
        }
        #[cfg(unix)]
        for tool in crate::terminal::tools(root).unwrap() {
            tools.push(tool);
        }
        for tool in loadngo_inference::tools::FsTools::new(root, None).into_tools() {
            tools.push(tool);
        }
        for tool in loadngo_inference::web_tools::WebTools::new().into_tools() {
            tools.push(tool);
        }
        for tool in
            loadngo_inference::memory_tools::MemoryStore::new(root.join("m.jsonl")).into_tools()
        {
            tools.push(tool);
        }
        let archives = loadngo_inference::cas_tools::Archives::new(Vec::new(), None);
        for tool in loadngo_inference::cas_tools::cas_tools(archives) {
            tools.push(tool);
        }
        tools
    }

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/gemma4/opening.json")
    }

    /// `K3_WRITE_GEMMA_DECLARATION=1 cargo test -p kimi-k3-cli gemma_opening -- --nocapture`
    /// writes the declaration the fixture script renders.
    #[test]
    fn gemma_opening_matches_the_chat_template() {
        let dir = std::env::var("GEMMA_CHECKPOINT").map_or_else(
            |_| PathBuf::from("/Volumes/Jarraya/gemma-4-31b-it"),
            PathBuf::from,
        );
        let root = tempfile::tempdir().unwrap();
        let tools = toolbox(root.path());
        if std::env::var_os("K3_WRITE_GEMMA_DECLARATION").is_some() {
            std::fs::write(
                fixture().with_file_name("declaration.json"),
                tools.declaration(),
            )
            .unwrap();
            return;
        }
        if !dir.join("tokenizer.json").is_file() {
            eprintln!("skipping: no Gemma tokenizer in {}", dir.display());
            return;
        }
        let tokenizer = Tokenizer::load(&dir).unwrap();
        let fixture: Value = serde_json::from_slice(&std::fs::read(fixture()).unwrap()).unwrap();
        assert_eq!(
            fixture["declaration"].as_str().unwrap(),
            tools.declaration(),
            "a tool changed: regenerate the fixture"
        );
        let g = GemmaTokens::new(&tokenizer, 2, &[1, 106]).unwrap();
        let system = fixture["system"].as_str().unwrap();
        let mut ids = g
            .preamble(&tokenizer, system, Some(&tools.declaration()))
            .unwrap();
        g.message(
            &mut ids,
            &tokenizer,
            "user",
            fixture["user"].as_str().unwrap(),
        );
        g.reply_header(&mut ids, &tokenizer);
        let want: Vec<u32> = serde_json::from_value(fixture["ids"].clone()).unwrap();
        if ids != want {
            let at = ids
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or(ids.len().min(want.len()));
            panic!(
                "differs at token {at}: here {:?}\ntemplate {:?}",
                tokenizer.decode_lossy(&ids[at.saturating_sub(20)..(at + 20).min(ids.len())]),
                tokenizer.decode_lossy(&want[at.saturating_sub(20)..(at + 20).min(want.len())])
            );
        }
    }
}
