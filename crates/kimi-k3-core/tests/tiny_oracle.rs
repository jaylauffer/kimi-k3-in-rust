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

#![allow(clippy::many_single_char_names, clippy::similar_names)]

mod common;

use std::fs;

use common::{fixtures_dir, read_json, tiny_config};
use kimi_k3_core::{
    config::K3Config,
    layer::{Attention, KdaWeights, LayerWeights, MlaWeights, Mlp, MoeWeights},
    model::{Model, argmax},
};
use serde_json::Value;

/// The exported tiny checkpoint: a flat little-endian f32 blob and a name -> offset map.
struct TinyCheckpoint {
    blob: Vec<f32>,
    tensors: Value,
}

impl TinyCheckpoint {
    fn load() -> Self {
        let manifest = read_json("tiny_k3.json");
        let bytes = fs::read(fixtures_dir().join("tiny_k3.bin")).expect("tiny_k3.bin");
        assert_eq!(
            bytes.len() % 4,
            0,
            "tiny_k3.bin is not a whole number of f32s"
        );
        let blob = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Self {
            blob,
            tensors: manifest["tensors"].clone(),
        }
    }

    /// A tensor by name. Absence is fatal: a missing weight read as zeros still runs.
    fn w(&self, name: &str) -> &[f32] {
        let entry = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("MISSING WEIGHT: {name}"));
        let offset =
            usize::try_from(entry["offset"].as_u64().expect("offset")).expect("offset fits");
        let numel = usize::try_from(entry["numel"].as_u64().expect("numel")).expect("numel fits");
        &self.blob[offset..offset + numel]
    }

    fn maybe(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|_| self.w(name))
    }
}

/// Routed experts packed contiguously per layer, so an expert is indexed at a fixed stride.
struct PackedExperts {
    per_layer: Vec<Option<[Vec<f32>; 3]>>,
}

impl PackedExperts {
    fn pack(ck: &TinyCheckpoint, c: &K3Config) -> Self {
        let per_layer = (0..c.num_hidden_layers)
            .map(|layer| {
                (!c.is_dense(layer)).then(|| {
                    ["w1", "w3", "w2"].map(|which| {
                        (0..c.num_experts)
                            .flat_map(|e| {
                                ck.w(&format!("layers_{layer}_mlp_experts_{e}_{which}_weight"))
                                    .to_vec()
                            })
                            .collect()
                    })
                })
            })
            .collect();
        Self { per_layer }
    }
}

fn bind<'a>(ck: &'a TinyCheckpoint, experts: &'a PackedExperts, c: &K3Config) -> Model<'a> {
    let layers = (0..c.num_hidden_layers)
        .map(|layer| {
            let w = |name: &str| ck.w(&format!("layers_{layer}_{name}"));
            let attention = if c.is_mla(layer) {
                Attention::Mla(MlaWeights {
                    q_a: w("self_attn_q_a_proj_weight"),
                    q_a_norm: w("self_attn_q_a_layernorm_weight"),
                    q_b: w("self_attn_q_b_proj_weight"),
                    kv_a: w("self_attn_kv_a_proj_with_mqa_weight"),
                    kv_a_norm: w("self_attn_kv_a_layernorm_weight"),
                    kv_b: w("self_attn_kv_b_proj_weight"),
                    o: w("self_attn_o_proj_weight"),
                    g: Some(w("self_attn_g_proj_weight")),
                })
            } else {
                Attention::Kda(KdaWeights {
                    q: w("self_attn_q_proj_weight"),
                    k: w("self_attn_k_proj_weight"),
                    v: w("self_attn_v_proj_weight"),
                    q_conv: w("self_attn_q_conv1d_weight"),
                    k_conv: w("self_attn_k_conv1d_weight"),
                    v_conv: w("self_attn_v_conv1d_weight"),
                    f_a: w("self_attn_f_a_proj_weight"),
                    f_b: w("self_attn_f_b_proj_weight"),
                    a_log: w("self_attn_A_log"),
                    dt_bias: w("self_attn_dt_bias"),
                    b: w("self_attn_b_proj_weight"),
                    g: w("self_attn_g_proj_weight"),
                    o_norm: w("self_attn_o_norm_weight"),
                    o: w("self_attn_o_proj_weight"),
                })
            };
            let mlp = match &experts.per_layer[layer] {
                None => Mlp::Dense {
                    gate: w("mlp_gate_proj_weight"),
                    up: w("mlp_up_proj_weight"),
                    down: w("mlp_down_proj_weight"),
                },
                Some([w1, w3, w2]) => Mlp::Moe(MoeWeights {
                    gate: w("mlp_gate_weight"),
                    bias: Some(w("mlp_e_score_correction_bias")),
                    down: w("mlp_down_weight"),
                    up: w("mlp_up_weight"),
                    latent_norm: w("mlp_norm_weight"),
                    shared_w1: w("mlp_shared_w1_weight"),
                    shared_w3: w("mlp_shared_w3_weight"),
                    shared_w2: w("mlp_shared_w2_weight"),
                    w1,
                    w3,
                    w2,
                }),
            };
            LayerWeights {
                in_norm: w("input_layernorm_weight"),
                post_norm: w("post_attention_layernorm_weight"),
                attn_res_norm: w("self_attention_res_norm_weight"),
                attn_res_proj: w("self_attention_res_proj_weight"),
                mlp_res_norm: w("mlp_res_norm_weight"),
                mlp_res_proj: w("mlp_res_proj_weight"),
                attention,
                mlp,
            }
        })
        .collect();

    Model {
        config: c.clone(),
        embed: ck.w("embed_tokens_weight"),
        lm_head: ck.w("lm_head_weight"),
        final_norm: ck.w("norm_weight"),
        out_res: ck
            .maybe("output_attn_res_norm_weight")
            .zip(ck.maybe("output_attn_res_proj_weight")),
        layers,
    }
}

fn ids(reference: &Value, key: &str) -> Vec<u32> {
    reference[key]
        .as_array()
        .unwrap_or_else(|| panic!("ref_k3.json has no {key}"))
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("token id")).expect("token id fits"))
        .collect()
}

fn bits_hash(logits: &[f32]) -> u64 {
    // FNV-1a over the little-endian bytes, so a C harness can print the same number.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in logits.iter().flat_map(|v| v.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

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
            prompt: ids(&reference, "prompt_ids"),
            full: ids(&reference, "full_ids"),
            teacher_forced: ids(&reference, "tf_pred"),
            config,
            checkpoint,
            experts,
        }
    }

    fn model(&self) -> Model<'_> {
        bind(&self.checkpoint, &self.experts, &self.config)
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

    let logits = model.forward(&oracle.full);
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
    let per_layer = model.forward(&oracle.full);
    let shared = model.forward_shared_kda_slot(&oracle.full);
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
        let logits = model.forward(&generated);
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
    let recomputed = model.forward(&oracle.full);

    let mut session = model.session(t);
    let mut generated = oracle.prompt.clone();
    let mut logits = model.feed(&mut session, &oracle.prompt);
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
        logits = model.feed(&mut session, &[next]);
    }
    assert_eq!(&generated[np..], &oracle.full[np..]);
}
