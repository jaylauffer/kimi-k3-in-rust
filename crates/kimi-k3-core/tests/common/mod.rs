//! Fixture access shared by the op and oracle gates. The fixtures are the C engine's,
//! read from `tests/fixtures` at the repository root, never copied.

#![allow(dead_code, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::{fs, path::PathBuf};

use kimi_k3_core::{
    config::K3Config,
    layer::{
        Attention, KdaWeights, LayerWeights, Matrix, MlaWeights, Mlp, MoeWeights, RoutedExperts,
    },
    model::Model,
};
use serde_json::Value;

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

pub fn read_json(relative: &str) -> Value {
    let path = fixtures_dir().join(relative);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not JSON: {error}", path.display()))
}

/// One op fixture: `key -> {shape, data}` arrays plus scalar parameters.
pub struct Fixture {
    name: String,
    root: Value,
}

impl Fixture {
    pub fn load(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            root: read_json(&format!("ops/{name}.json")),
        }
    }

    fn get(&self, key: &str) -> &Value {
        self.root
            .get(key)
            .unwrap_or_else(|| panic!("{}: missing key {key}", self.name))
    }

    /// The flat `data` array of `key`, as the C harness reads it.
    pub fn arr(&self, key: &str) -> Vec<f32> {
        let value = self.get(key);
        let data = value.get("data").unwrap_or(value);
        data.as_array()
            .unwrap_or_else(|| panic!("{}: {key} has no data array", self.name))
            .iter()
            .map(|v| v.as_f64().expect("numeric fixture data") as f32)
            .collect()
    }

    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.get(key)["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: {key} has no shape", self.name))
            .iter()
            .map(|v| v.as_u64().expect("integer shape") as usize)
            .collect()
    }

    pub fn num(&self, key: &str) -> f64 {
        self.get(key)
            .as_f64()
            .unwrap_or_else(|| panic!("{}: {key} is not a number", self.name))
    }

    pub fn usize(&self, key: &str) -> usize {
        self.num(key) as usize
    }

    /// A scalar parameter only some fixtures carry.
    pub fn try_usize(&self, key: &str) -> Option<usize> {
        self.root
            .get(key)
            .and_then(Value::as_f64)
            .map(|v| v as usize)
    }

    pub fn boolean(&self, key: &str) -> bool {
        self.get(key)
            .as_bool()
            .unwrap_or_else(|| panic!("{}: {key} is not a boolean", self.name))
    }
}

/// The tiny model's configuration, which every op fixture was generated from.
pub fn tiny_config() -> K3Config {
    let reference = read_json("ref_k3.json");
    K3Config::from_value(&reference["config"], "ref_k3.json").expect("tiny config parses")
}

/// The published pass criterion from `ops/MANIFEST.json`.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub abs: f64,
    pub rel: f64,
}

pub fn manifest_tolerance() -> Tolerance {
    let manifest = read_json("ops/MANIFEST.json");
    let tolerance = &manifest["tolerance"];
    Tolerance {
        abs: tolerance["fp32_abs"].as_f64().expect("fp32_abs"),
        rel: tolerance["fp32_rel"].as_f64().expect("fp32_rel"),
    }
}

/// Worst `|got - want| / (abs + rel * |want|)` over the whole array; a pass is <= 1.
pub fn worst_ratio(got: &[f32], want: &[f32], tol: Tolerance) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter()
        .zip(want)
        .map(|(&g, &w)| {
            (f64::from(g) - f64::from(w)).abs() / (tol.abs + tol.rel * f64::from(w).abs())
        })
        .fold(0.0, f64::max)
}

/// The exported tiny checkpoint: a flat little-endian f32 blob and a name -> offset map.
pub struct TinyCheckpoint {
    blob: Vec<f32>,
    tensors: Value,
}

impl TinyCheckpoint {
    pub fn load() -> Self {
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
    pub fn w(&self, name: &str) -> &[f32] {
        let entry = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("MISSING WEIGHT: {name}"));
        let offset = entry["offset"].as_u64().expect("offset") as usize;
        let numel = entry["numel"].as_u64().expect("numel") as usize;
        &self.blob[offset..offset + numel]
    }

    pub fn maybe(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|_| self.w(name))
    }

    pub fn names(&self) -> Vec<String> {
        self.tensors
            .as_object()
            .expect("tensor map")
            .keys()
            .cloned()
            .collect()
    }
}

/// One routed expert's fp32 matrices, as the tiny checkpoint stores them.
pub fn expert_weight<'a>(
    ck: &'a TinyCheckpoint,
    layer: usize,
    expert: usize,
    which: &str,
) -> &'a [f32] {
    ck.w(&format!(
        "layers_{layer}_mlp_experts_{expert}_{which}_weight"
    ))
}

/// Routed experts packed contiguously per layer, so an expert is indexed at a fixed stride.
pub struct PackedExperts {
    pub per_layer: Vec<Option<[Vec<f32>; 3]>>,
}

impl PackedExperts {
    pub fn pack(ck: &TinyCheckpoint, c: &K3Config) -> Self {
        let per_layer = (0..c.num_hidden_layers)
            .map(|layer| {
                (!c.is_dense(layer)).then(|| {
                    ["w1", "w3", "w2"].map(|which| {
                        (0..c.num_experts)
                            .flat_map(|e| expert_weight(ck, layer, e, which).to_vec())
                            .collect()
                    })
                })
            })
            .collect();
        Self { per_layer }
    }

    pub fn resident(&self, layer: usize) -> RoutedExperts<'_> {
        let [w1, w3, w2] = self.per_layer[layer].as_ref().expect("a MoE layer");
        RoutedExperts::Resident { w1, w3, w2 }
    }
}

/// Binds the tiny checkpoint the way `k3_model.c` does. `matrix` supplies every weight the
/// C binder keeps narrow (read only through a matmul), so a test chooses its format;
/// everything read elementwise comes from the checkpoint as fp32.
pub fn bind_tiny<'a>(
    ck: &'a TinyCheckpoint,
    c: &K3Config,
    matrix: &dyn Fn(&str) -> Matrix<'a>,
    experts: &dyn Fn(usize) -> RoutedExperts<'a>,
) -> Model<'a> {
    let layers = (0..c.num_hidden_layers)
        .map(|layer| {
            let w = |name: &str| ck.w(&format!("layers_{layer}_{name}"));
            let m = |name: &str| matrix(&format!("layers_{layer}_{name}"));
            let attention = if c.is_mla(layer) {
                Attention::Mla(MlaWeights {
                    q_a: m("self_attn_q_a_proj_weight"),
                    q_a_norm: w("self_attn_q_a_layernorm_weight"),
                    q_b: m("self_attn_q_b_proj_weight"),
                    kv_a: m("self_attn_kv_a_proj_with_mqa_weight"),
                    kv_a_norm: w("self_attn_kv_a_layernorm_weight"),
                    kv_b: m("self_attn_kv_b_proj_weight"),
                    o: m("self_attn_o_proj_weight"),
                    g: Some(m("self_attn_g_proj_weight")),
                })
            } else {
                Attention::Kda(KdaWeights {
                    q: m("self_attn_q_proj_weight"),
                    k: m("self_attn_k_proj_weight"),
                    v: m("self_attn_v_proj_weight"),
                    q_conv: w("self_attn_q_conv1d_weight"),
                    k_conv: w("self_attn_k_conv1d_weight"),
                    v_conv: w("self_attn_v_conv1d_weight"),
                    f_a: m("self_attn_f_a_proj_weight"),
                    f_b: m("self_attn_f_b_proj_weight"),
                    a_log: w("self_attn_A_log"),
                    dt_bias: w("self_attn_dt_bias"),
                    b: m("self_attn_b_proj_weight"),
                    g: m("self_attn_g_proj_weight"),
                    o_norm: w("self_attn_o_norm_weight"),
                    o: m("self_attn_o_proj_weight"),
                })
            };
            let mlp = if c.is_dense(layer) {
                Mlp::Dense {
                    gate: m("mlp_gate_proj_weight"),
                    up: m("mlp_up_proj_weight"),
                    down: m("mlp_down_proj_weight"),
                }
            } else {
                Mlp::Moe(MoeWeights {
                    gate: w("mlp_gate_weight"),
                    bias: Some(w("mlp_e_score_correction_bias")),
                    down: m("mlp_down_weight"),
                    up: m("mlp_up_weight"),
                    latent_norm: w("mlp_norm_weight"),
                    shared_w1: m("mlp_shared_w1_weight"),
                    shared_w3: m("mlp_shared_w3_weight"),
                    shared_w2: m("mlp_shared_w2_weight"),
                    experts: experts(layer),
                })
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
        embed: matrix("embed_tokens_weight"),
        lm_head: matrix("lm_head_weight"),
        final_norm: ck.w("norm_weight"),
        out_res: ck
            .maybe("output_attn_res_norm_weight")
            .zip(ck.maybe("output_attn_res_proj_weight")),
        layers,
    }
}

/// Token ids from `ref_k3.json`.
pub fn ref_ids(reference: &Value, key: &str) -> Vec<u32> {
    reference[key]
        .as_array()
        .unwrap_or_else(|| panic!("ref_k3.json has no {key}"))
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("token id")).expect("token id fits"))
        .collect()
}

/// FNV-1a over the little-endian bytes, so a C harness can print the same number.
pub fn bits_hash(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in values.iter().flat_map(|v| v.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

pub fn assert_close(name: &str, got: &[f32], want: &[f32], tol: Tolerance) {
    let worst = worst_ratio(got, want, tol);
    println!("{name}: n={} worst={worst:.2}x tolerance", got.len());
    assert!(
        worst <= 1.0,
        "{name}: worst element is {worst:.2}x the tolerance"
    );
}
