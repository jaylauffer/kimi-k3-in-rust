//! Token-for-token parity against the real `tiktoken` library, the way
//! `tools/tok_parity.py` gates the C engine's tokenizer.
//!
//! `tests/fixtures/tokenizer/parity.json` holds real encode results: built by
//! loading the actual checkpoint's `tiktoken.model` (163,584 ranks) into a real
//! `tiktoken.Encoding` with the Kimi pre-tokenizer pattern, then encoding a set
//! of strings spanning CJK, emoji, ZWJ sequences, accents, contractions and
//! whitespace runs (see `docs/RUST_PORT.md` for how it was generated). This is
//! independent ground truth: a different encoder implementation (Rust
//! `tiktoken`, not this port's own logic) against the real vocabulary.
//!
//! Needs the real checkpoint locally only to build the `Tokenizer`
//! (`KIMI_K3_CHECKPOINT`, default `/Volumes/Jarraya/kimi-k3`); the fixture
//! itself ships in the repo, so it always runs (not `#[ignore]`d).

use std::env;
use std::path::PathBuf;

use kimi_k3_core::tokenizer::Tokenizer;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

fn checkpoint_dir() -> PathBuf {
    env::var("KIMI_K3_CHECKPOINT")
        .map_or_else(|_| PathBuf::from("/Volumes/Jarraya/kimi-k3"), PathBuf::from)
}

fn fixtures() -> Vec<Fixture> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tokenizer/parity.json");
    let bytes =
        std::fs::read(&path).unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    serde_json::from_slice(&bytes).expect("parity.json parses")
}

#[test]
fn encode_matches_the_real_tiktoken_library_token_for_token() {
    let dir = checkpoint_dir();
    if !dir.join("tiktoken.model").is_file() {
        eprintln!(
            "skipping: no checkpoint at {} (set KIMI_K3_CHECKPOINT)",
            dir.display()
        );
        return;
    }
    let tokenizer = Tokenizer::load(&dir).expect("tokenizer loads from the real checkpoint");

    let cases = fixtures();
    assert!(!cases.is_empty(), "parity.json must not be empty");
    let mut failures = Vec::new();
    for case in &cases {
        let got = tokenizer.encode(&case.text);
        if got != case.ids {
            failures.push(format!(
                "text {:?}: got {:?}, want {:?}",
                case.text, got, case.ids
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} fixtures mismatched:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn decode_matches_the_real_tiktoken_library_round_trip() {
    let dir = checkpoint_dir();
    if !dir.join("tiktoken.model").is_file() {
        eprintln!(
            "skipping: no checkpoint at {} (set KIMI_K3_CHECKPOINT)",
            dir.display()
        );
        return;
    }
    let tokenizer = Tokenizer::load(&dir).expect("tokenizer loads from the real checkpoint");

    let cases = fixtures();
    let mut failures = Vec::new();
    for case in &cases {
        let got = tokenizer.decode_lossy(&case.ids);
        if got != case.decoded {
            failures.push(format!(
                "ids for {:?}: got {:?}, want {:?}",
                case.text, got, case.decoded
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} fixtures mismatched:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

#[test]
fn round_trips_every_fixture_string_through_this_ports_own_encode_and_decode() {
    let dir = checkpoint_dir();
    if !dir.join("tiktoken.model").is_file() {
        eprintln!(
            "skipping: no checkpoint at {} (set KIMI_K3_CHECKPOINT)",
            dir.display()
        );
        return;
    }
    let tokenizer = Tokenizer::load(&dir).expect("tokenizer loads from the real checkpoint");

    for case in fixtures() {
        let ids = tokenizer.encode(&case.text);
        let text = tokenizer.decode_lossy(&ids);
        assert_eq!(
            text, case.text,
            "round trip changed the text: encode then decode of {:?} gave {:?}",
            case.text, text
        );
    }
}
