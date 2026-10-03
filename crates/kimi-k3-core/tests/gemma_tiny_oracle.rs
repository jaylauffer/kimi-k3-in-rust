//! The Gemma 4 forward pass against transformers' own (`Gemma4ForCausalLM`, eager
//! attention, float32) on a random-weight model with the 31B's structure at toy size:
//! `tests/fixtures/gemma4/tiny/`, written by `scripts/gemma_tiny_oracle.py`. The text is
//! fed whole, in uneven pieces and one token at a time, so the sliding window, the ring
//! that holds it (window + 512 slots, wrapped by 600 tokens) and every chunk boundary are
//! exercised.

use std::path::PathBuf;

use kimi_k3_core::gemma::GemmaModel;
use serde::Deserialize;

#[derive(Deserialize)]
struct Oracle {
    ids: Vec<u32>,
    positions: Vec<usize>,
    logits: Vec<Vec<f32>>,
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/gemma4/tiny")
}

fn oracle() -> Oracle {
    serde_json::from_slice(&std::fs::read(dir().join("oracle.json")).unwrap()).unwrap()
}

fn worst(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
fn whole_text_matches_transformers() {
    let o = oracle();
    let model = GemmaModel::load(dir(), None, None, || true).unwrap();
    let mut session = model.session(o.ids.len());
    let all = model.score(&mut session, &o.ids, None, || true).unwrap();
    let vocab = model.config.vocab_size;
    for (&p, want) in o.positions.iter().zip(&o.logits) {
        let error = worst(&all[p * vocab..(p + 1) * vocab], want);
        assert!(error < 2e-3, "position {p}: max logit error {error}");
    }
}

#[test]
fn pieces_and_single_tokens_match_transformers() {
    let o = oracle();
    let model = GemmaModel::load(dir(), None, None, || true).unwrap();
    let mut session = model.session(o.ids.len());
    let mut fed = 0;
    // Pieces of 1, 7, 300 (two passes' worth of ring), then 1 at a time.
    for size in [1, 7, 300, 190].into_iter().chain(std::iter::repeat(1)) {
        if fed == o.ids.len() {
            break;
        }
        let end = (fed + size).min(o.ids.len());
        let logits = model
            .feed(&mut session, &o.ids[fed..end], None, || true)
            .unwrap();
        fed = end;
        if let Some(i) = o.positions.iter().position(|&p| p == end - 1) {
            let error = worst(&logits, &o.logits[i]);
            assert!(error < 2e-3, "after {end} tokens: max logit error {error}");
        }
    }
}
