//! The full-model oracle gate, the Rust counterpart of `tests/unit/k3_model.c`.
//!
//! Loads the tiny 13-layer checkpoint, runs the complete forward pass and holds it to
//! `ref_k3.json` from the pure-torch reference. The bar is the C gate's: exact integers.
//!
//! - Gate 1: teacher-forced argmax over the generated span of ONE forward.
//! - Gate 1b: one reused KDA state slot gives bit-identical logits.
//! - Gate 2: greedy decode by full recompute.
//! - Gate 3: incremental decode, prefill once then one token at a time.
//!
//! Positions inside the random prompt are not expected to match; only the generated
//! span is a real test.

mod common;

use common::{
    PackedExperts, TinyCheckpoint, bind_tiny, bits_hash, read_json, ref_ids, tiny_config,
};
use kimi_k3_core::{
    config::K3Config,
    layer::{Matrix, NoStreamedExperts},
    model::{Model, argmax},
};

struct Oracle {
    config: K3Config,
    checkpoint: TinyCheckpoint,
    experts: PackedExperts,
    prompt: Vec<u32>,
    full: Vec<u32>,
    teacher_forced: Vec<u32>,
}

impl Oracle {
    fn load() -> Self {
        let reference = read_json("ref_k3.json");
        let config = tiny_config();
        let checkpoint = TinyCheckpoint::load();
        let experts = PackedExperts::pack(&checkpoint, &config);
        Self {
            prompt: ref_ids(&reference, "prompt_ids"),
            full: ref_ids(&reference, "full_ids"),
            teacher_forced: ref_ids(&reference, "tf_pred"),
            config,
            checkpoint,
            experts,
        }
    }

    fn model(&self) -> Model<'_> {
        bind_tiny(
            &self.checkpoint,
            &self.config,
            &|name| Matrix::F32(self.checkpoint.w(name)),
            &|layer| self.experts.resident(layer),
        )
    }
}

fn argmax_id(row: &[f32]) -> u32 {
    u32::try_from(argmax(row)).expect("vocab fits u32")
}

#[test]
fn gate1_teacher_forcing_matches_every_generated_position() {
    let oracle = Oracle::load();
    let model = oracle.model();
    let vocab = oracle.config.vocab_size;
    let (t, np) = (oracle.full.len(), oracle.prompt.len());

    let logits = model
        .forward(&oracle.full, &mut NoStreamedExperts)
        .expect("resident model");
    println!("gate1 logits fnv1a {:016x}", bits_hash(&logits));
    let got: Vec<u32> = logits.chunks_exact(vocab).map(argmax_id).collect();
    let span = np - 1..t - 1;
    let matched = span
        .clone()
        .filter(|&i| got[i] == oracle.teacher_forced[i])
        .count();
    println!("gate1 generated span {matched}/{}", span.len());
    assert_eq!(&got[span.clone()], &oracle.teacher_forced[span]);
}

#[test]
fn gate1b_one_reused_kda_slot_gives_bit_identical_logits() {
    let oracle = Oracle::load();
    let model = oracle.model();
    let per_layer = model
        .forward(&oracle.full, &mut NoStreamedExperts)
        .expect("resident model");
    let shared = model
        .forward_shared_kda_slot(&oracle.full, &mut NoStreamedExperts)
        .expect("resident model");
    let identical = per_layer
        .iter()
        .zip(&shared)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(identical, "state reuse changed the logits");
}

#[test]
fn gate2_greedy_decode_by_full_recompute_matches_every_token() {
    let oracle = Oracle::load();
    let model = oracle.model();
    let vocab = oracle.config.vocab_size;
    let mut generated = oracle.prompt.clone();
    while generated.len() < oracle.full.len() {
        let logits = model
            .forward(&generated, &mut NoStreamedExperts)
            .expect("resident model");
        let last = &logits[(generated.len() - 1) * vocab..];
        generated.push(argmax_id(last));
    }
    assert_eq!(generated, oracle.full);
}

#[test]
fn gate3_incremental_decode_matches_every_token_and_the_full_forward_logits() {
    let oracle = Oracle::load();
    let model = oracle.model();
    let vocab = oracle.config.vocab_size;
    let (t, np) = (oracle.full.len(), oracle.prompt.len());
    let recomputed = model
        .forward(&oracle.full, &mut NoStreamedExperts)
        .expect("resident model");

    let mut session = model.session(t);
    let mut generated = oracle.prompt.clone();
    let mut logits = model
        .feed(&mut session, &oracle.prompt, &mut NoStreamedExperts)
        .expect("resident model");
    loop {
        // Carrying state is a restructuring of WHEN work happens, not WHAT is computed,
        // so each step's logits must equal the full forward's at that position.
        let at = session.position() - 1;
        let identical = logits
            .iter()
            .zip(&recomputed[at * vocab..(at + 1) * vocab])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            identical,
            "incremental logits at position {at} differ from full forward"
        );
        if session.position() >= t {
            break;
        }
        let next = argmax_id(&logits);
        generated.push(next);
        logits = model
            .feed(&mut session, &[next], &mut NoStreamedExperts)
            .expect("resident model");
    }
    assert_eq!(&generated[np..], &oracle.full[np..]);
}
