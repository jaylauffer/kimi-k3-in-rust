//! Gemma 4's tokenizer (`spm_bpe`) token for token against the `tokenizers` library.
//!
//! `tests/fixtures/tokenizer/gemma4-parity.json` was written by `tokenizers` 0.23.2
//! (`Tokenizer.from_file(<checkpoint>/tokenizer.json).encode(text, add_special_tokens=False)`)
//! over the K3 parity strings, chat-template control text, whitespace runs, scripts from
//! Arabic to Hangul, emoji sequences, control characters and three real source files.
//!
//! Needs the checkpoint's `tokenizer.json` (`GEMMA_CHECKPOINT`, default
//! `/Volumes/Jarraya/gemma-4-31b-it`); skipped without it.

use std::{env, path::PathBuf};

use kimi_k3_core::tokenizer::Tokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

#[test]
fn gemma_encode_and_decode_match_the_tokenizers_library() {
    let dir = env::var("GEMMA_CHECKPOINT").map_or_else(
        |_| PathBuf::from("/Volumes/Jarraya/gemma-4-31b-it"),
        PathBuf::from,
    );
    if !dir.join("tokenizer.json").is_file() {
        eprintln!(
            "skipping: no tokenizer.json in {} (set GEMMA_CHECKPOINT)",
            dir.display()
        );
        return;
    }
    let tokenizer = Tokenizer::load(&dir).expect("Gemma tokenizer loads");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tokenizer/gemma4-parity.json");
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(cases.len() > 50);
    let mut failures = Vec::new();
    for case in &cases {
        let ids = tokenizer.encode(&case.text);
        if ids != case.ids {
            failures.push(format!(
                "encode {:?}: got {ids:?}, want {:?}",
                case.text, case.ids
            ));
        }
        let decoded = tokenizer.decode_lossy(&case.ids);
        if decoded != case.decoded {
            failures.push(format!("decode {:?}: got {decoded:?}", case.decoded));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
