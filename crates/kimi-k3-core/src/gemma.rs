//! Google Gemma 4 dense text models (`model_type = "gemma4"`, its `text_config`), the
//! released `gemma-4-31B-it`: written from transformers' `modeling_gemma4.py`
//! (`Gemma4TextModel`, `Gemma4TextDecoderLayer`, `Gemma4TextAttention`) and
//! `modeling_rope_utils.py` (`_compute_proportional_rope_parameters`). Text only; the
//! vision tower is not loaded.
//!
//! The decoder, as implemented:
//!
//! - the embedding row times `sqrt(hidden)` rounded to bf16 (73.5 for 5376), as the
//!   reference's bf16 buffer does; the LM head is the same matrix (tied);
//! - every layer: `h += post_attn_norm(attn(input_norm(h)))`, then
//!   `h += post_ff_norm(mlp(pre_ff_norm(h)))`, then `h *= layer_scalar`;
//! - `RMSNorm` multiplies by its weight directly (not `1 + w`); `v_norm` has no weight;
//! - attention: per-head `q_norm`/`k_norm` before rotary, scale 1, grouped-query heads.
//!   Sliding layers attend to the last `sliding_window` positions (the current one
//!   included) with 256-wide heads and default rotary (theta 10,000). Full layers attend
//!   to everything with 512-wide heads, 4 KV heads whose values are the key projection
//!   before `k_norm` (`attention_k_eq_v`), and proportional rotary: only the first
//!   quarter of the frequency pairs rotate (theta 1,000,000), the rest have frequency 0;
//! - the MLP is `down(gelu_tanh(gate(x)) * up(x))`;
//! - logits are soft-capped: `30 * tanh(logits / 30)`.
//!
//! The 31B checkpoint is 62.5 GB of bf16, more than this Mac's memory, so its matrices
//! are read from a directory written by `k3 --convert-mxfp4` (OCP MXFP4, the format the
//! Kimi Linear experts use) wherever that directory has them, and from the checkpoint's
//! bf16 otherwise. Products go through the same [`products`] path as Kimi Linear, so
//! the GPU runs them from memory it shares with the CPU. Attention runs on the CPU,
//! heads split across threads.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::{fs, path::Path};

use serde_json::Value;

use crate::{
    layer::{Accel, DenseAccel, DeviceCache, GroupedJob, Matrix, WeightRef},
    linear::{LinearError, Mul, Stored, Weight, matrix, products, vector},
    ops::rmsnorm,
    safetensors::SafeTensorIndex,
};

/// Positions fed per pass. A sliding layer's cache keeps `sliding_window + CHUNK`
/// positions, so every query of a pass still finds its whole window.
pub const CHUNK: usize = 512;

/// The MXFP4 scale group, as for Kimi Linear's experts.
pub const MXFP4_BLOCK: usize = crate::expert::MXFP4_GROUP_SIZE;

/// The per-layer matrices, by their name after `layers.L.`.
pub const MATRICES: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// A validated Gemma 4 text configuration. Shapes this module does not implement
/// (mixture of experts, per-layer inputs, shared KV layers, attention bias, another
/// activation or rotary type) are refused rather than run with a guessed default.
#[derive(Clone, Debug, PartialEq)]
pub struct GemmaConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_global_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    /// Full-attention layers take their values from the key projection.
    pub attention_k_eq_v: bool,
    pub final_logit_softcapping: Option<f32>,
    pub sliding_rope_theta: f32,
    pub full_rope_theta: f32,
    pub full_partial_rotary_factor: f32,
    /// Per layer: full attention (true) or sliding.
    pub full: Vec<bool>,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
}

fn field<'a>(root: &'a Value, key: &str) -> Result<&'a Value, LinearError> {
    root.get(key)
        .ok_or_else(|| LinearError::Config(format!("gemma4: no `{key}`")))
}

fn usize_of(root: &Value, key: &str) -> Result<usize, LinearError> {
    field(root, key)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| LinearError::Config(format!("gemma4: `{key}` is not a count")))
}

fn f32_of(root: &Value, key: &str) -> Result<f32, LinearError> {
    field(root, key)?
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| LinearError::Config(format!("gemma4: `{key}` is not a number")))
}

/// Fails unless `root[key]` is absent, null, false, 0, or equal to `want`.
fn off(root: &Value, key: &str) -> Result<(), LinearError> {
    match root.get(key) {
        None | Some(Value::Null | Value::Bool(false)) => Ok(()),
        Some(Value::Number(n)) if n.as_f64() == Some(0.0) => Ok(()),
        Some(other) => Err(LinearError::Config(format!(
            "gemma4: `{key}` = {other} is not implemented"
        ))),
    }
}

fn ids_of(value: &Value) -> Vec<u32> {
    match value {
        Value::Array(items) => items.iter().flat_map(ids_of).collect(),
        Value::Number(n) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

impl GemmaConfig {
    /// Whether `config.json` at `path` describes a Gemma 4 model.
    #[must_use]
    pub fn detect(path: impl AsRef<Path>) -> bool {
        fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .is_some_and(|root| {
                matches!(root["model_type"].as_str(), Some("gemma4" | "gemma4_text"))
            })
    }

    /// # Errors
    /// An unreadable file or an unsupported configuration.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, LinearError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| LinearError::Config(format!("{}: {e}", path.display())))?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|e| LinearError::Config(format!("{}: {e}", path.display())))?;
        Self::from_value(&root)
    }

    /// # Errors
    /// An unsupported configuration.
    #[allow(clippy::too_many_lines)] // one field list, read top to bottom
    pub fn from_value(root: &Value) -> Result<Self, LinearError> {
        let text = match root["model_type"].as_str() {
            Some("gemma4") => field(root, "text_config")?,
            Some("gemma4_text") => root,
            other => {
                return Err(LinearError::Config(format!(
                    "model_type {other:?} is not gemma4"
                )));
            }
        };
        for key in [
            "enable_moe_block",
            "hidden_size_per_layer_input",
            "num_kv_shared_layers",
            "attention_bias",
            "use_double_wide_mlp",
        ] {
            off(text, key)?;
        }
        if text["hidden_activation"] != "gelu_pytorch_tanh" {
            return Err(LinearError::Config(format!(
                "gemma4: activation {} is not implemented",
                text["hidden_activation"]
            )));
        }
        if root.get("tie_word_embeddings") == Some(&Value::Bool(false))
            || text.get("tie_word_embeddings") == Some(&Value::Bool(false))
        {
            return Err(LinearError::Config(
                "gemma4: an untied LM head is not implemented".into(),
            ));
        }
        let rope = field(text, "rope_parameters")?;
        let (sliding, full) = (&rope["sliding_attention"], &rope["full_attention"]);
        if sliding["rope_type"] != "default" || full["rope_type"] != "proportional" {
            return Err(LinearError::Config(format!(
                "gemma4: rotary {} / {} is not implemented (default / proportional)",
                sliding["rope_type"], full["rope_type"]
            )));
        }
        if full
            .get("factor")
            .and_then(Value::as_f64)
            .is_some_and(|f| (f - 1.0).abs() > f64::EPSILON)
        {
            return Err(LinearError::Config(
                "gemma4: a rotary factor is not implemented".into(),
            ));
        }
        let layers = usize_of(text, "num_hidden_layers")?;
        let types = field(text, "layer_types")?
            .as_array()
            .ok_or_else(|| LinearError::Config("gemma4: `layer_types` is not a list".into()))?;
        let full_layers = types
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(true),
                Some("sliding_attention") => Ok(false),
                _ => Err(LinearError::Config(format!("gemma4: layer type {t}"))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if full_layers.len() != layers {
            return Err(LinearError::Config(format!(
                "gemma4: {} layer types for {layers} layers",
                full_layers.len()
            )));
        }
        let attention_k_eq_v = text["attention_k_eq_v"] == true;
        let kv_heads = usize_of(text, "num_key_value_heads")?;
        let num_global_key_value_heads = if attention_k_eq_v {
            usize_of(text, "num_global_key_value_heads")?
        } else {
            kv_heads
        };
        let eos_token_ids = {
            let mut ids = ids_of(&root["eos_token_id"]);
            ids.extend(ids_of(&text["eos_token_id"]));
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        let config = Self {
            hidden_size: usize_of(text, "hidden_size")?,
            num_hidden_layers: layers,
            num_attention_heads: usize_of(text, "num_attention_heads")?,
            num_key_value_heads: kv_heads,
            num_global_key_value_heads,
            head_dim: usize_of(text, "head_dim")?,
            global_head_dim: usize_of(text, "global_head_dim")?,
            intermediate_size: usize_of(text, "intermediate_size")?,
            vocab_size: usize_of(text, "vocab_size")?,
            rms_norm_eps: f32_of(text, "rms_norm_eps")?,
            sliding_window: usize_of(text, "sliding_window")?,
            attention_k_eq_v,
            final_logit_softcapping: text["final_logit_softcapping"].as_f64().map(|v| v as f32),
            sliding_rope_theta: f32_of(sliding, "rope_theta")?,
            full_rope_theta: f32_of(full, "rope_theta")?,
            full_partial_rotary_factor: full["partial_rotary_factor"].as_f64().unwrap_or(1.0)
                as f32,
            full: full_layers,
            bos_token_id: u32::try_from(usize_of(text, "bos_token_id")?)
                .map_err(|_| LinearError::Config("gemma4: bos id".into()))?,
            eos_token_ids,
        };
        if !(config.full_partial_rotary_factor > 0.0 && config.full_partial_rotary_factor <= 1.0) {
            return Err(LinearError::Config(
                "gemma4: partial_rotary_factor must be in (0, 1]".into(),
            ));
        }
        if config.num_attention_heads % config.num_key_value_heads != 0
            || config.num_attention_heads % config.num_global_key_value_heads != 0
        {
            return Err(LinearError::Config(
                "gemma4: heads do not group evenly".into(),
            ));
        }
        Ok(config)
    }

    /// Head width, KV heads and whether values come from the keys, for layer `l`.
    #[must_use]
    pub fn attention(&self, l: usize) -> (usize, usize, bool) {
        if self.full[l] {
            (
                self.global_head_dim,
                self.num_global_key_value_heads,
                self.attention_k_eq_v,
            )
        } else {
            (self.head_dim, self.num_key_value_heads, false)
        }
    }

    /// `[out][inp]` of matrix `name` (one of [`MATRICES`]) in layer `l`; `None` for a
    /// value projection the layer does not have.
    #[must_use]
    pub fn matrix_shape(&self, l: usize, name: &str) -> Option<(usize, usize)> {
        let (hd, kvh, k_eq_v) = self.attention(l);
        let (e, heads, inter) = (
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
        );
        Some(match name {
            "self_attn.q_proj" => (heads * hd, e),
            "self_attn.v_proj" if k_eq_v => return None,
            "self_attn.k_proj" | "self_attn.v_proj" => (kvh * hd, e),
            "self_attn.o_proj" => (e, heads * hd),
            "mlp.gate_proj" | "mlp.up_proj" => (inter, e),
            "mlp.down_proj" => (e, inter),
            _ => return None,
        })
    }

    /// Tensor name prefix of layer `l`.
    #[must_use]
    pub fn layer_prefix(l: usize) -> String {
        format!("model.language_model.layers.{l}.")
    }

    /// The rotary inverse frequencies of a layer (`head_dim / 2` of them, zeros past
    /// the rotated pairs).
    fn inv_freq(&self, full: bool) -> Vec<f32> {
        let (dim, base, rotated) = if full {
            let dim = self.global_head_dim;
            // transformers: int(partial_rotary_factor * head_dim // 2), from a validated
            // factor in (0, 1].
            let angles = ((f64::from(self.full_partial_rotary_factor) * dim as f64) as usize) / 2;
            (dim, self.full_rope_theta, angles)
        } else {
            (self.head_dim, self.sliding_rope_theta, self.head_dim / 2)
        };
        (0..dim / 2)
            .map(|j| {
                if j < rotated {
                    1.0 / base.powf((2 * j) as f32 / dim as f32)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

struct Layer {
    full: bool,
    in_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    pre_ff_norm: Vec<f32>,
    post_ff_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    scalar: f32,
    q: Weight,
    k: Weight,
    v: Option<Weight>,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}

impl Layer {
    fn weights_mut(&mut self) -> Vec<&mut Weight> {
        let mut all = vec![
            &mut self.q,
            &mut self.k,
            &mut self.o,
            &mut self.gate,
            &mut self.up,
            &mut self.down,
        ];
        if let Some(v) = &mut self.v {
            all.push(v);
        }
        all
    }
}

/// A loaded Gemma 4 text model, every weight resident.
pub struct GemmaModel {
    pub config: GemmaConfig,
    /// `[vocab][hidden]` bf16: the embedding, and the LM head (tied).
    embed: Weight,
    embed_scale: f32,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    sliding_freq: Vec<f32>,
    full_freq: Vec<f32>,
    threads: usize,
}

/// One layer's keys and values, `[slot][kv_heads][head_dim]`, position `p` in slot
/// `p % slots`. A full layer has a slot for every position the session can hold; a
/// sliding layer keeps `sliding_window + CHUNK`. Rows are written before they are read,
/// so the buffers grow as positions arrive and a reset only forgets the position count.
#[derive(Clone, Default)]
struct Kv {
    k: Vec<f32>,
    v: Vec<f32>,
    slots: usize,
    /// A device's copy of `k` and `v`.
    device: DeviceCache,
}

/// Incremental decoding state. Cloning it snapshots a shared prefix for reuse.
#[derive(Clone)]
pub struct GemmaSession {
    kv: Vec<Kv>,
    ids: Vec<u32>,
    capacity: usize,
    broken: bool,
}

impl GemmaSession {
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    #[must_use]
    pub const fn is_broken(&self) -> bool {
        self.broken
    }

    pub fn reset(&mut self) {
        for kv in &mut self.kv {
            kv.device.0 = None;
        }
        self.ids.clear();
        self.broken = false;
    }
}

/// `x` rounded to the nearest bf16, ties to even.
fn round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// A dot product in eight lanes, so it vectorizes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for k in 0..8 {
            lanes[k] += x[k] * y[k];
        }
    }
    let tail: f32 = a
        .chunks_exact(8)
        .remainder()
        .iter()
        .zip(b.chunks_exact(8).remainder())
        .map(|(x, y)| x * y)
        .sum();
    lanes.iter().sum::<f32>() + tail
}

fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2 / pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// `RMSNorm` without a weight (`v_norm`).
fn rms_unscaled(v: &mut [f32], eps: f32) {
    let ss: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let inv = (1.0 / (ss / v.len() as f64 + f64::from(eps)).sqrt()) as f32;
    for x in v {
        *x *= inv;
    }
}

/// Rotates the two halves of `x` by `position * freq` (transformers' `rotate_half`).
fn rotate(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    let (a, b) = x.split_at_mut(half);
    for j in 0..half {
        let (x1, x2) = (a[j], b[j]);
        a[j] = x1 * cos[j] - x2 * sin[j];
        b[j] = x2 * cos[j] + x1 * sin[j];
    }
}

fn read_weight(
    index: &SafeTensorIndex,
    mxfp4: Option<&SafeTensorIndex>,
    name: &str,
    out: usize,
    inp: usize,
) -> Result<Weight, LinearError> {
    if let Some(converted) = mxfp4 {
        let blocks = converted.tensor(&format!("{name}.blocks"));
        let scales = converted.tensor(&format!("{name}.scales"));
        if let (Some(blocks), Some(scales)) = (blocks, scales) {
            if blocks.shape != [out, inp / 2] || scales.shape != [out, inp / MXFP4_BLOCK] {
                return Err(LinearError::Tensor {
                    name: name.into(),
                    detail: format!(
                        "converted shapes {:?} / {:?} do not fit [{out}, {inp}]",
                        blocks.shape, scales.shape
                    ),
                });
            }
            return Ok(Weight {
                data: Stored::Mxfp4 {
                    packed: converted.read_raw(blocks)?,
                    scales: converted.read_raw(scales)?,
                },
                out,
                inp,
            });
        }
    }
    Ok(Weight::new(matrix(index, name, out, inp)?, out, inp))
}

fn scalar(index: &SafeTensorIndex, name: &str) -> Result<f32, LinearError> {
    Ok(vector(index, name, 1)?[0])
}

impl GemmaModel {
    /// Reads `directory`'s config and weights, each matrix from `mxfp4` when that
    /// converted directory has it. With a `device`, each weight moves into its memory as
    /// soon as it is read, so the heap never holds more than one layer.
    ///
    /// # Errors
    /// An unsupported config, a missing or misshapen tensor, or a read error.
    pub fn load(
        directory: impl AsRef<Path>,
        mxfp4: Option<&Path>,
        device: Option<&dyn DenseAccel>,
        mut keep_running: impl FnMut() -> bool,
    ) -> Result<Self, LinearError> {
        let directory = directory.as_ref();
        let config = GemmaConfig::from_path(directory.join("config.json"))?;
        let index = SafeTensorIndex::open(directory)?;
        let converted = mxfp4.map(SafeTensorIndex::open).transpose()?;
        let c = &config;
        let e = c.hidden_size;
        let embed_name = "model.language_model.embed_tokens.weight";
        let mut embed = Weight::new(
            matrix(&index, embed_name, c.vocab_size, e)?,
            c.vocab_size,
            e,
        );
        if let Some(device) = device {
            embed.share(device);
        }
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for l in 0..c.num_hidden_layers {
            if !keep_running() {
                return Err(LinearError::Cancelled);
            }
            let at = |s: &str| format!("{}{s}", GemmaConfig::layer_prefix(l));
            let (hd, _, _) = c.attention(l);
            let w = |name: &str| -> Result<Option<Weight>, LinearError> {
                let Some((out, inp)) = c.matrix_shape(l, name) else {
                    return Ok(None);
                };
                read_weight(
                    &index,
                    converted.as_ref(),
                    &at(&format!("{name}.weight")),
                    out,
                    inp,
                )
                .map(Some)
            };
            let need = |name: &str| -> Result<Weight, LinearError> {
                w(name)?.ok_or_else(|| LinearError::Missing(at(name)))
            };
            let v = w("self_attn.v_proj")?;
            if v.is_none() && index.tensor(&at("self_attn.v_proj.weight")).is_some() {
                return Err(LinearError::Config(format!(
                    "gemma4: layer {l} has a v_proj, but its config takes values from the keys"
                )));
            }
            let mut layer = Layer {
                full: c.full[l],
                in_norm: vector(&index, &at("input_layernorm.weight"), e)?,
                post_attn_norm: vector(&index, &at("post_attention_layernorm.weight"), e)?,
                pre_ff_norm: vector(&index, &at("pre_feedforward_layernorm.weight"), e)?,
                post_ff_norm: vector(&index, &at("post_feedforward_layernorm.weight"), e)?,
                q_norm: vector(&index, &at("self_attn.q_norm.weight"), hd)?,
                k_norm: vector(&index, &at("self_attn.k_norm.weight"), hd)?,
                scalar: scalar(&index, &at("layer_scalar"))?,
                q: need("self_attn.q_proj")?,
                k: need("self_attn.k_proj")?,
                v,
                o: need("self_attn.o_proj")?,
                gate: need("mlp.gate_proj")?,
                up: need("mlp.up_proj")?,
                down: need("mlp.down_proj")?,
            };
            if let Some(device) = device {
                for weight in layer.weights_mut() {
                    weight.share(device);
                }
            }
            layers.push(layer);
        }
        let norm = vector(&index, "model.language_model.norm.weight", e)?;
        let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        Ok(Self {
            embed_scale: round_bf16((e as f32).sqrt()),
            sliding_freq: config.inv_freq(false),
            full_freq: config.inv_freq(true),
            config,
            embed,
            norm,
            layers,
            threads,
        })
    }

    /// Bytes of weights by storage: `(bf16, mxfp4)`.
    #[must_use]
    pub fn weight_bytes(&self) -> (usize, usize) {
        let mut sizes = (0, 0);
        let mut add = |w: &Weight| match w.as_ref() {
            WeightRef::Bf16(words) => sizes.0 += words.len() * 2,
            WeightRef::Mxfp4 { packed, scales } => sizes.1 += packed.len() + scales.len(),
        };
        add(&self.embed);
        for layer in &self.layers {
            for w in [
                &layer.q,
                &layer.k,
                &layer.o,
                &layer.gate,
                &layer.up,
                &layer.down,
            ] {
                add(w);
            }
            if let Some(v) = &layer.v {
                add(v);
            }
        }
        sizes
    }

    /// A session able to hold `capacity` positions.
    #[must_use]
    pub fn session(&self, capacity: usize) -> GemmaSession {
        // A multiple of 32, so a device can read the ring 32 keys at a time without wrapping.
        let ring = (self.config.sliding_window + CHUNK).next_multiple_of(32);
        GemmaSession {
            kv: self
                .layers
                .iter()
                .map(|layer| Kv {
                    slots: if layer.full { capacity } else { ring },
                    ..Kv::default()
                })
                .collect(),
            ids: Vec::new(),
            capacity,
            broken: false,
        }
    }

    fn check(&self, session: &GemmaSession, ids: &[u32]) -> Result<(), LinearError> {
        if session.broken {
            return Err(LinearError::BrokenSession);
        }
        if ids.is_empty()
            || session.ids.len() + ids.len() > session.capacity
            || ids.iter().any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(LinearError::Capacity {
                have: session.ids.len(),
                add: ids.len(),
                capacity: session.capacity,
            });
        }
        Ok(())
    }

    /// Feeds `ids` after everything `session` consumed and returns the logits for the
    /// last of them.
    ///
    /// # Errors
    /// [`LinearError::Capacity`] (session untouched), or cancellation, after which the
    /// session is marked broken.
    pub fn feed(
        &self,
        session: &mut GemmaSession,
        ids: &[u32],
        accel: Accel<'_>,
        keep_running: impl FnMut() -> bool,
    ) -> Result<Vec<f32>, LinearError> {
        self.forward(session, ids, accel, false, keep_running)
    }

    /// As [`Self::feed`], returning the logits after every one of `ids`
    /// (`[ids][vocab]`): row `i` predicts the token after `ids[i]`.
    ///
    /// # Errors
    /// As [`Self::feed`].
    pub fn score(
        &self,
        session: &mut GemmaSession,
        ids: &[u32],
        accel: Accel<'_>,
        keep_running: impl FnMut() -> bool,
    ) -> Result<Vec<f32>, LinearError> {
        self.forward(session, ids, accel, true, keep_running)
    }

    fn forward(
        &self,
        session: &mut GemmaSession,
        ids: &[u32],
        accel: Accel<'_>,
        all_logits: bool,
        mut keep_running: impl FnMut() -> bool,
    ) -> Result<Vec<f32>, LinearError> {
        self.check(session, ids)?;
        let mut logits = Vec::new();
        for chunk in ids.chunks(CHUNK) {
            match self.run(session, chunk, accel, all_logits, &mut keep_running) {
                Ok(rows) => {
                    session.ids.extend_from_slice(chunk);
                    if all_logits {
                        logits.extend_from_slice(&rows);
                    } else {
                        logits = rows;
                    }
                }
                Err(error) => {
                    session.broken = true;
                    return Err(error);
                }
            }
        }
        Ok(logits)
    }

    fn run(
        &self,
        session: &mut GemmaSession,
        ids: &[u32],
        accel: Accel<'_>,
        all_logits: bool,
        keep_running: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<f32>, LinearError> {
        let c = &self.config;
        let (e, t, start) = (c.hidden_size, ids.len(), session.ids.len());
        let mut h = vec![0.0_f32; t * e];
        let WeightRef::Bf16(embed) = self.embed.as_ref() else {
            unreachable!("the embedding is read as bf16");
        };
        for (row, &id) in h.chunks_exact_mut(e).zip(ids) {
            Matrix::Bf16(embed).row_into(row, id as usize, e);
            for x in row {
                *x *= self.embed_scale;
            }
        }
        let rope = |freq: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let mut cos = Vec::with_capacity(t * freq.len());
            let mut sin = Vec::with_capacity(t * freq.len());
            for i in 0..t {
                let p = (start + i) as f32;
                for &f in freq {
                    let angle = p * f;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
            (cos, sin)
        };
        let ropes = [rope(&self.sliding_freq), rope(&self.full_freq)];
        let mut x = vec![0.0_f32; t * e];
        let mut tmp = vec![0.0_f32; t * e];
        let mut scratch = Scratch::default();
        for (l, (layer, kv)) in self.layers.iter().zip(&mut session.kv).enumerate() {
            if !keep_running() {
                return Err(LinearError::Cancelled);
            }
            for (y, hx) in x.chunks_exact_mut(e).zip(h.chunks_exact(e)) {
                rmsnorm(y, hx, &layer.in_norm, c.rms_norm_eps);
            }
            let (cos, sin) = &ropes[usize::from(layer.full)];
            self.attention(
                &mut tmp,
                &x,
                layer,
                l,
                kv,
                start,
                (cos, sin),
                accel,
                &mut scratch,
            );
            for (hi, a) in h.chunks_exact_mut(e).zip(tmp.chunks_exact_mut(e)) {
                rmsnorm(&mut x[..e], a, &layer.post_attn_norm, c.rms_norm_eps);
                for (hv, &av) in hi.iter_mut().zip(&x[..e]) {
                    *hv += av;
                }
            }
            for (y, hx) in x.chunks_exact_mut(e).zip(h.chunks_exact(e)) {
                rmsnorm(y, hx, &layer.pre_ff_norm, c.rms_norm_eps);
            }
            self.mlp(&mut tmp, &x, layer, t, accel, &mut scratch);
            for (hi, m) in h.chunks_exact_mut(e).zip(tmp.chunks_exact(e)) {
                let normed = &mut scratch.row;
                normed.resize(e, 0.0);
                rmsnorm(normed, m, &layer.post_ff_norm, c.rms_norm_eps);
                for (hv, &mv) in hi.iter_mut().zip(normed.iter()) {
                    *hv = (*hv + mv) * layer.scalar;
                }
            }
        }
        if !keep_running() {
            return Err(LinearError::Cancelled);
        }
        let first = if all_logits { 0 } else { t - 1 };
        let rows = t - first;
        let mut normed = vec![0.0_f32; rows * e];
        for (y, hx) in normed
            .chunks_exact_mut(e)
            .zip(h[first * e..].chunks_exact(e))
        {
            rmsnorm(y, hx, &self.norm, c.rms_norm_eps);
        }
        let mut logits = vec![0.0_f32; rows * c.vocab_size];
        products(
            accel,
            &mut [Mul {
                x: &normed,
                rows,
                parts: vec![(&self.embed, &mut logits)],
            }],
        );
        if let Some(cap) = c.final_logit_softcapping {
            for v in &mut logits {
                *v = cap * (*v / cap).tanh();
            }
        }
        Ok(logits)
    }

    #[allow(clippy::too_many_lines)] // one attention block, read top to bottom
    fn attention(
        &self,
        out: &mut [f32],
        x: &[f32],
        layer: &Layer,
        l: usize,
        kv: &mut Kv,
        start: usize,
        (cos, sin): (&[f32], &[f32]),
        accel: Accel<'_>,
        s: &mut Scratch,
    ) {
        let c = &self.config;
        let t = x.len() / c.hidden_size;
        let (hd, kvh, k_eq_v) = c.attention(l);
        let heads = c.num_attention_heads;
        let (qw, kw) = (heads * hd, kvh * hd);
        s.q.resize(t * qw, 0.0);
        s.k.resize(t * kw, 0.0);
        s.v.resize(t * kw, 0.0);
        {
            let mut parts = vec![(&layer.q, &mut s.q[..]), (&layer.k, &mut s.k[..])];
            if let Some(v) = &layer.v {
                parts.push((v, &mut s.v[..]));
            }
            products(accel, &mut [Mul { x, rows: t, parts }]);
        }
        if k_eq_v {
            s.v.copy_from_slice(&s.k);
        }
        let half = hd / 2;
        for i in 0..t {
            let (cos, sin) = (
                &cos[i * half..(i + 1) * half],
                &sin[i * half..(i + 1) * half],
            );
            for head in s.q[i * qw..(i + 1) * qw].chunks_exact_mut(hd) {
                crate::ops::rmsnorm_in_place(head, &layer.q_norm, c.rms_norm_eps);
                rotate(head, cos, sin);
            }
            for head in s.k[i * kw..(i + 1) * kw].chunks_exact_mut(hd) {
                crate::ops::rmsnorm_in_place(head, &layer.k_norm, c.rms_norm_eps);
                rotate(head, cos, sin);
            }
            for head in s.v[i * kw..(i + 1) * kw].chunks_exact_mut(hd) {
                rms_unscaled(head, c.rms_norm_eps);
            }
        }
        // Store the new keys and values.
        let slots = kv.slots;
        let needed = slots.min(start + t) * kw;
        if kv.k.len() < needed {
            kv.k.resize(needed, 0.0);
            kv.v.resize(needed, 0.0);
        }
        for i in 0..t {
            let slot = (start + i) % slots;
            kv.k[slot * kw..(slot + 1) * kw].copy_from_slice(&s.k[i * kw..(i + 1) * kw]);
            kv.v[slot * kw..(slot + 1) * kw].copy_from_slice(&s.v[i * kw..(i + 1) * kw]);
        }
        let window = if layer.full {
            usize::MAX
        } else {
            c.sliding_window
        };
        s.attn.resize(t * qw, 0.0);
        let rows = slots.min(start + t) * kw;
        let on_device = accel.is_some_and(|device| {
            device.grouped_attention(&mut GroupedJob {
                q: &s.q,
                k: &kv.k[..rows],
                v: &kv.v[..rows],
                out: &mut s.attn,
                t,
                start,
                heads,
                kv_heads: kvh,
                dim: hd,
                window,
                slots,
                scale: 1.0,
                device: &mut kv.device,
            })
        });
        if !on_device {
            kv.device.0 = None;
            self.cpu_attention(s, kv, (t, start, slots), (heads, kvh, hd), window);
        }
        products(
            accel,
            &mut [Mul {
                x: &s.attn,
                rows: t,
                parts: vec![(&layer.o, out)],
            }],
        );
    }

    /// The reference attention, heads split across threads.
    fn cpu_attention(
        &self,
        s: &mut Scratch,
        kv: &Kv,
        (t, start, slots): (usize, usize, usize),
        (heads, kvh, hd): (usize, usize, usize),
        window: usize,
    ) {
        let (qw, kw) = (heads * hd, kvh * hd);
        let group = heads / kvh;
        let per = heads.div_ceil(self.threads);
        let (q, keys, values) = (&s.q, &kv.k, &kv.v);
        let mut by_head: Vec<Vec<f32>> = vec![Vec::new(); heads];
        std::thread::scope(|scope| {
            for (first, outs) in (0..heads).step_by(per).zip(by_head.chunks_mut(per)) {
                scope.spawn(move || {
                    let mut scores = Vec::new();
                    for (offset, o) in outs.iter_mut().enumerate() {
                        let head = first + offset;
                        let g = head / group;
                        o.resize(t * hd, 0.0);
                        for i in 0..t {
                            let p = start + i;
                            let lo = (p + 1).saturating_sub(window);
                            let qv = &q[i * qw + head * hd..i * qw + (head + 1) * hd];
                            scores.clear();
                            let mut max = f32::NEG_INFINITY;
                            for pos in lo..=p {
                                let at = (pos % slots) * kw + g * hd;
                                let score = dot(qv, &keys[at..at + hd]);
                                max = max.max(score);
                                scores.push(score);
                            }
                            let mut total = 0.0_f32;
                            for v in &mut scores {
                                *v = (*v - max).exp();
                                total += *v;
                            }
                            let row = &mut o[i * hd..(i + 1) * hd];
                            row.fill(0.0);
                            for (pos, &w) in (lo..=p).zip(&scores) {
                                let at = (pos % slots) * kw + g * hd;
                                let w = w / total;
                                for (r, &vv) in row.iter_mut().zip(&values[at..at + hd]) {
                                    *r += w * vv;
                                }
                            }
                        }
                    }
                });
            }
        });
        for (head, o) in by_head.iter().enumerate() {
            for i in 0..t {
                s.attn[i * qw + head * hd..i * qw + (head + 1) * hd]
                    .copy_from_slice(&o[i * hd..(i + 1) * hd]);
            }
        }
    }

    fn mlp(
        &self,
        out: &mut [f32],
        x: &[f32],
        layer: &Layer,
        t: usize,
        accel: Accel<'_>,
        s: &mut Scratch,
    ) {
        let inter = self.config.intermediate_size;
        s.gate.resize(t * inter, 0.0);
        s.up.resize(t * inter, 0.0);
        products(
            accel,
            &mut [Mul {
                x,
                rows: t,
                parts: vec![(&layer.gate, &mut s.gate[..]), (&layer.up, &mut s.up[..])],
            }],
        );
        // 11 million tanh per layer in a 512-position pass: split across the cores.
        let per = (s.gate.len().div_ceil(self.threads)).max(1 << 15);
        if s.gate.len() <= per {
            for (g, &u) in s.gate.iter_mut().zip(&s.up) {
                *g = gelu_tanh(*g) * u;
            }
        } else {
            std::thread::scope(|scope| {
                for (g, u) in s.gate.chunks_mut(per).zip(s.up.chunks(per)) {
                    scope.spawn(move || {
                        for (g, &u) in g.iter_mut().zip(u) {
                            *g = gelu_tanh(*g) * u;
                        }
                    });
                }
            });
        }
        products(
            accel,
            &mut [Mul {
                x: &s.gate,
                rows: t,
                parts: vec![(&layer.down, out)],
            }],
        );
    }
}

/// Per-pass intermediates, grown as needed.
#[derive(Default)]
struct Scratch {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    row: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GemmaConfig {
        let text = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/gemma4/config.json"),
        )
        .unwrap();
        GemmaConfig::from_value(&serde_json::from_str(&text).unwrap()).unwrap()
    }

    #[test]
    fn the_31b_config_parses_with_its_layer_pattern() {
        let c = config();
        assert_eq!(
            (c.hidden_size, c.num_hidden_layers, c.vocab_size),
            (5376, 60, 262_144)
        );
        assert_eq!(c.full.iter().filter(|&&f| f).count(), 10);
        assert!(c.full[5] && c.full[59] && !c.full[0]);
        assert_eq!(c.attention(0), (256, 16, false));
        assert_eq!(c.attention(5), (512, 4, true));
        assert_eq!(c.matrix_shape(5, "self_attn.v_proj"), None);
        assert_eq!(c.matrix_shape(0, "self_attn.v_proj"), Some((4096, 5376)));
        assert_eq!(c.final_logit_softcapping, Some(30.0));
        assert_eq!(c.eos_token_ids, [1, 106]);
    }

    #[test]
    fn proportional_rotary_turns_only_the_first_quarter_of_pairs() {
        let c = config();
        let full = c.inv_freq(true);
        assert_eq!(full.len(), 256);
        assert!(full[..64].iter().all(|&f| f > 0.0));
        assert!(full[64..].iter().all(|&f| f == 0.0));
        assert!((full[1] - 1e6_f32.powf(-2.0 / 512.0)).abs() < 1e-9);
        let sliding = c.inv_freq(false);
        assert_eq!(sliding.len(), 128);
        assert!((sliding[127] - 1e4_f32.powf(-254.0 / 256.0)).abs() < 1e-9);
    }

    #[test]
    fn unimplemented_shapes_are_refused() {
        let text = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/gemma4/config.json"),
        )
        .unwrap();
        for (key, value) in [
            ("enable_moe_block", Value::Bool(true)),
            ("hidden_size_per_layer_input", Value::from(256)),
            ("num_kv_shared_layers", Value::from(20)),
            ("hidden_activation", Value::from("silu")),
        ] {
            let mut root: Value = serde_json::from_str(&text).unwrap();
            root["text_config"][key] = value;
            assert!(GemmaConfig::from_value(&root).is_err(), "{key}");
        }
    }

    #[test]
    fn the_embedding_scale_is_rounded_to_bf16_as_the_reference_does() {
        assert_eq!(round_bf16(5376_f32.sqrt()).to_bits(), 73.5_f32.to_bits());
        assert_eq!(round_bf16(1.0).to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn rotation_with_zero_frequency_is_the_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        rotate(&mut x, &[1.0, 1.0], &[0.0, 0.0]);
        assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
        rotate(&mut x, &[0.0, 1.0], &[1.0, 0.0]);
        assert_eq!(x, [-3.0, 2.0, 1.0, 4.0]);
    }
}
