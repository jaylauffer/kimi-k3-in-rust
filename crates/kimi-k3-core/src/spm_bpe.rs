//! The SentencePiece-style BPE of a Hugging Face `tokenizer.json` (Gemma 4): spaces
//! normalised to `▁`, no pre-tokenizer split, merges by rank, unknown characters as
//! `<0xHH>` byte tokens. Written from the `tokenizers` crate's `BPE::merge_word` and
//! `Word::merge_all`, and checked against it (`tests/spm_bpe_parity.rs`).
//!
//! Only the shape Gemma 4's `tokenizer.json` has is accepted: a `BPE` model with
//! `byte_fallback`, a `Replace(" ", "▁")` normalizer, a split on `" "` (which never
//! matches after that normalizer, so the whole span is one word), and the
//! `Replace / ByteFallback / Fuse` decoder. Anything else is refused.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    fs,
    path::Path,
};

use serde_json::Value;

use crate::tokenizer::TokenizerError;

const SPACE: char = '\u{2581}';

pub struct SpmBpe {
    vocab: HashMap<String, u32>,
    /// `(left, right)` -> `(rank, merged id)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// id -> its bytes when decoded (`▁` already a space; a byte token is its byte).
    pieces: Vec<Vec<u8>>,
    /// The `<0xHH>` token of every byte.
    bytes: [u32; 256],
    /// Added tokens, longest first, for [`Self::encode`].
    specials: Vec<(String, u32)>,
}

fn json_error(path: &Path, error: &impl ToString) -> TokenizerError {
    TokenizerError::Json {
        path: path.to_path_buf(),
        error: error.to_string(),
    }
}

/// Whether `dir` holds a `tokenizer.json` in the shape [`SpmBpe`] reads, rather than a
/// tiktoken model.
#[must_use]
pub fn present(dir: &Path) -> bool {
    dir.join("tokenizer.json").is_file() && !dir.join("tiktoken.model").is_file()
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    id: u32,
}

impl Ord for Merge {
    // A min-heap on rank, then position: the `tokenizers` crate's order.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    live: bool,
}

impl SpmBpe {
    /// Reads `dir/tokenizer.json`.
    ///
    /// # Errors
    /// An unreadable file, or a tokenizer of another shape.
    pub fn load(dir: &Path) -> Result<Self, TokenizerError> {
        let path = dir.join("tokenizer.json");
        let text = fs::read_to_string(&path).map_err(|error| TokenizerError::Io {
            path: path.clone(),
            error: error.to_string(),
        })?;
        let root: Value = serde_json::from_str(&text).map_err(|e| json_error(&path, &e))?;
        let refuse = |what: &str| json_error(&path, &format!("unsupported tokenizer.json: {what}"));

        let model = &root["model"];
        if model["type"] != "BPE" || model["byte_fallback"] != true {
            return Err(refuse("model is not a byte-fallback BPE"));
        }
        if !model["continuing_subword_prefix"].is_null() || !model["end_of_word_suffix"].is_null() {
            return Err(refuse("subword prefix or suffix"));
        }
        let normalizer = &root["normalizer"];
        if normalizer["type"] != "Replace"
            || normalizer["pattern"]["String"] != " "
            || normalizer["content"] != "\u{2581}"
        {
            return Err(refuse("normalizer is not Replace(' ', '▁')"));
        }
        let pre = &root["pre_tokenizer"];
        if !(pre.is_null() || pre["type"] == "Split" && pre["pattern"]["String"] == " ") {
            return Err(refuse("pre-tokenizer other than a split on ' '"));
        }

        let vocab_json = model["vocab"]
            .as_object()
            .ok_or_else(|| refuse("no vocab"))?;
        let mut vocab = HashMap::with_capacity(vocab_json.len());
        let mut size = 0;
        for (piece, id) in vocab_json {
            let id = id
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| refuse("vocab id"))?;
            size = size.max(id as usize + 1);
            vocab.insert(piece.clone(), id);
        }
        let added = root["added_tokens"]
            .as_array()
            .ok_or_else(|| refuse("no added_tokens"))?;
        let mut specials = Vec::new();
        for token in added {
            let (Some(content), Some(id)) = (token["content"].as_str(), token["id"].as_u64())
            else {
                return Err(refuse("added token"));
            };
            let id = u32::try_from(id).map_err(|_| refuse("added token id"))?;
            size = size.max(id as usize + 1);
            specials.push((content.to_string(), id));
        }
        specials.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));

        let mut pieces = vec![Vec::new(); size];
        let mut bytes = [u32::MAX; 256];
        for (piece, &id) in &vocab {
            pieces[id as usize] = match byte_token(piece) {
                Some(b) => {
                    bytes[b as usize] = id;
                    vec![b]
                }
                None => piece.replace(SPACE, " ").into_bytes(),
            };
        }
        for (content, id) in &specials {
            pieces[*id as usize] = content.clone().into_bytes();
        }
        if bytes.contains(&u32::MAX) {
            return Err(refuse("a byte has no <0xHH> token"));
        }

        let merge_list = model["merges"]
            .as_array()
            .ok_or_else(|| refuse("no merges"))?;
        let mut merges = HashMap::with_capacity(merge_list.len());
        for (rank, merge) in merge_list.iter().enumerate() {
            let (left, right) = match merge {
                Value::Array(pair) if pair.len() == 2 => (
                    pair[0].as_str().ok_or_else(|| refuse("merge"))?,
                    pair[1].as_str().ok_or_else(|| refuse("merge"))?,
                ),
                Value::String(s) => s.split_once(' ').ok_or_else(|| refuse("merge"))?,
                _ => return Err(refuse("merge")),
            };
            let (Some(&a), Some(&b), Some(&joined)) = (
                vocab.get(left),
                vocab.get(right),
                vocab.get(&format!("{left}{right}")),
            ) else {
                return Err(refuse("a merge names a piece not in the vocab"));
            };
            let rank = u32::try_from(rank).map_err(|_| refuse("merge rank"))?;
            merges.entry((a, b)).or_insert((rank, joined));
        }
        Ok(Self {
            vocab,
            merges,
            pieces,
            bytes,
            specials,
        })
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// The id of exactly this piece or added token.
    #[must_use]
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.specials
            .iter()
            .find(|(text, _)| text == piece)
            .map(|&(_, id)| id)
            .or_else(|| self.vocab.get(piece).copied())
    }

    /// `text` with added tokens recognised as single ids.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let hit = (0..rest.len())
                .filter(|&i| rest.is_char_boundary(i))
                .find_map(|i| {
                    self.specials
                        .iter()
                        .find(|(s, _)| rest[i..].starts_with(s.as_str()))
                        .map(|(s, id)| (i, s.len(), *id))
                });
            let Some((at, len, id)) = hit else {
                self.encode_ordinary_into(rest, &mut out);
                break;
            };
            self.encode_ordinary_into(&rest[..at], &mut out);
            out.push(id);
            rest = &rest[at + len..];
        }
        out
    }

    /// `text` as plain text: control-token spellings in it are ordinary characters.
    #[must_use]
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_ordinary_into(text, &mut out);
        out
    }

    fn encode_ordinary_into(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut symbols: Vec<Symbol> = Vec::with_capacity(text.len());
        let mut buf = [0_u8; 4];
        for c in text.chars() {
            let c = if c == ' ' { SPACE } else { c };
            let s: &str = c.encode_utf8(&mut buf);
            match self.vocab.get(s) {
                Some(&id) => symbols.push(Symbol {
                    id,
                    prev: None,
                    next: None,
                    live: true,
                }),
                None => {
                    for &b in s.as_bytes() {
                        symbols.push(Symbol {
                            id: self.bytes[b as usize],
                            prev: None,
                            next: None,
                            live: true,
                        });
                    }
                }
            }
        }
        let n = symbols.len();
        for (i, s) in symbols.iter_mut().enumerate() {
            s.prev = i.checked_sub(1);
            s.next = (i + 1 < n).then_some(i + 1);
        }
        let mut queue = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            if let Some(&(rank, id)) = self.merges.get(&(symbols[i].id, symbols[i + 1].id)) {
                queue.push(Merge { pos: i, rank, id });
            }
        }
        while let Some(top) = queue.pop() {
            let current = symbols[top.pos];
            if !current.live {
                continue;
            }
            let Some(next) = current.next else { continue };
            match self.merges.get(&(current.id, symbols[next].id)) {
                Some(&(_, id)) if id == top.id => {}
                _ => continue,
            }
            symbols[top.pos].id = top.id;
            symbols[next].live = false;
            let after = symbols[next].next;
            symbols[top.pos].next = after;
            if let Some(after) = after {
                symbols[after].prev = Some(top.pos);
            }
            let before = current
                .prev
                .and_then(|prev| Some((prev, self.merges.get(&(symbols[prev].id, top.id))?)));
            if let Some((pos, &(rank, id))) = before {
                queue.push(Merge { pos, rank, id });
            }
            let beyond = after.and_then(|after| self.merges.get(&(top.id, symbols[after].id)));
            if let Some(&(rank, id)) = beyond {
                queue.push(Merge {
                    pos: top.pos,
                    rank,
                    id,
                });
            }
        }
        let mut at = Some(0);
        while let Some(i) = at {
            out.push(symbols[i].id);
            at = symbols[i].next;
        }
    }

    /// The bytes `ids` decode to: added tokens as their text, byte tokens as their byte.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(piece) = self.pieces.get(id as usize) {
                out.extend_from_slice(piece);
            }
        }
        out
    }
}

/// `<0xHH>` -> the byte.
fn byte_token(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(hex, 16).ok())
        .flatten()
}
