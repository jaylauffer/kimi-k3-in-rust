//! Moonshot Kimi Linear (`model_type = "kimi_linear"`), the released
//! Kimi-Linear-48B-A3B checkpoints: the same KDA and NoPE-MLA attention family as K3 at
//! a size this lab can run interactively. Written from the checkpoint's own
//! `modeling_kimi.py` (revision `e1df551a447157d4658b573f9a695d57658590e9`) and, for
//! the KDA decay gate it imports, fla's `naive_kda_gate`.
//!
//! What differs from K3, and why this is its own module rather than flags through
//! [`crate::layer`] (whose CPU paths are gated bit-exactly against the C engine):
//!
//! - a plain pre-norm residual decoder: no Block Attention Residual aggregation;
//! - `SiLU(gate) * up` MLPs instead of SiTU-GLU;
//! - experts in the hidden space (no latent down/up projection or latent norm), stored
//!   as bf16, one shared expert;
//! - MLA projects queries directly (`q_lora_rank = null`) and has no output gate;
//! - KDA's output gate is low rank (`g_a`, `g_b`), and its decay gate has no lower
//!   bound: `g = -exp(A_log[h]) * softplus(z + dt_bias)` (fla `naive_kda_gate`).
//!
//! Shared with K3: the KDA recurrence, fused-SiLU short convolution, L2/RMS norms, the
//! sigmoid router with a selection-only bias, and the CPU bf16 product
//! ([`Matrix::mul_rows`]), so the CPU path is the reference for both.
//!
//! Products are handed to the optional device ([`Accel`]) a step at a time, every
//! product whose input is ready at once ([`crate::layer::DenseAccel::run_bf16`]): KDA's six input
//! projections, both low-rank halves, the shared and routed experts' gate and up, then
//! their down projections. A device can then prepare one weight while it computes
//! another (the Neural Engine converts bf16 to fp16 on a helper thread meanwhile).
//!
//! Everything except the routed experts (about 2 B parameters) is loaded resident.
//! Routed experts (26 layers x 256 x 14.2 MB for the 48B model) stream through a
//! least-recently-used cache, fetched per layer as one batch of proactor reads.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::{collections::HashMap, fmt, fs, path::Path, time::Instant};

use serde_json::Value;

use crate::{
    io::ReadRequest,
    layer::{Accel, Bf16Job, Matrix},
    ops::{
        kda_step, l2norm_in_place, rmsnorm, rmsnorm_in_place, router, shortconv_in_place, sigmoid,
    },
    safetensors::{DType, SafeTensorError, SafeTensorIndex},
};

/// A validated Kimi Linear configuration. Only the shapes this module implements are
/// accepted; anything else is refused rather than run with a guessed default.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
    pub short_conv_kernel_size: usize,
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub num_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub moe_renormalize: bool,
    pub first_k_dense_replace: usize,
    pub intermediate_size: usize,
    /// One-based layer positions that use MLA; the rest use KDA.
    pub full_attn_layers: Vec<usize>,
    pub eos_token_id: u32,
}

fn get<'a>(root: &'a Value, key: &str) -> Result<&'a Value, LinearError> {
    root.get(key)
        .ok_or_else(|| LinearError::Config(format!("missing `{key}`")))
}

fn usize_field(root: &Value, key: &str) -> Result<usize, LinearError> {
    get(root, key)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| LinearError::Config(format!("`{key}` is not a non-negative integer")))
}

fn f32_field(root: &Value, key: &str) -> Result<f32, LinearError> {
    get(root, key)?
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| LinearError::Config(format!("`{key}` is not a number")))
}

fn expect(root: &Value, key: &str, want: &Value) -> Result<(), LinearError> {
    let got = root.get(key).unwrap_or(&Value::Null);
    if got == want {
        Ok(())
    } else {
        Err(LinearError::Config(format!(
            "`{key}` is {got}; this engine implements only {want}"
        )))
    }
}

impl LinearConfig {
    /// Reads `config.json`.
    ///
    /// # Errors
    /// Returns [`LinearError::Config`] for unreadable JSON or an unsupported shape.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, LinearError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|e| LinearError::Config(format!("{}: {e}", path.display())))?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|e| LinearError::Config(format!("{}: {e}", path.display())))?;
        Self::from_value(&root)
    }

    /// True when the `config.json` at `path` declares `model_type = "kimi_linear"`.
    /// Unreadable or non-JSON files are simply not Kimi Linear.
    #[must_use]
    pub fn detect(path: impl AsRef<Path>) -> bool {
        fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .is_some_and(|root| Self::is_kimi_linear(&root))
    }

    /// True for a `config.json` whose `model_type` is `kimi_linear`.
    #[must_use]
    pub fn is_kimi_linear(root: &Value) -> bool {
        root.get("model_type").and_then(Value::as_str) == Some("kimi_linear")
    }

    /// # Errors
    /// Returns [`LinearError::Config`] for a missing field or an unsupported shape.
    pub fn from_value(root: &Value) -> Result<Self, LinearError> {
        if !Self::is_kimi_linear(root) {
            return Err(LinearError::Config(
                "`model_type` is not kimi_linear".into(),
            ));
        }
        expect(root, "hidden_act", &Value::from("silu"))?;
        expect(root, "q_lora_rank", &Value::Null)?;
        expect(root, "mla_use_nope", &Value::from(true))?;
        expect(root, "moe_router_activation_func", &Value::from("sigmoid"))?;
        expect(root, "num_expert_group", &Value::from(1))?;
        expect(root, "topk_group", &Value::from(1))?;
        expect(root, "moe_layer_freq", &Value::from(1))?;
        expect(root, "tie_word_embeddings", &Value::from(false))?;
        let linear = get(root, "linear_attn_config")?;
        if linear.get("gate_lower_bound").is_some_and(|v| !v.is_null()) {
            return Err(LinearError::Config(
                "`gate_lower_bound` is set: that is K3's safe gate, not this model".into(),
            ));
        }
        let full_attn_layers = get(linear, "full_attn_layers")?
            .as_array()
            .ok_or_else(|| LinearError::Config("`full_attn_layers` is not a list".into()))?
            .iter()
            .map(|v| v.as_u64().and_then(|v| usize::try_from(v).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| LinearError::Config("`full_attn_layers` holds a non-integer".into()))?;
        let config = Self {
            hidden_size: usize_field(root, "hidden_size")?,
            num_hidden_layers: usize_field(root, "num_hidden_layers")?,
            vocab_size: usize_field(root, "vocab_size")?,
            rms_norm_eps: f32_field(root, "rms_norm_eps")?,
            kda_num_heads: usize_field(linear, "num_heads")?,
            kda_head_dim: usize_field(linear, "head_dim")?,
            short_conv_kernel_size: usize_field(linear, "short_conv_kernel_size")?,
            num_attention_heads: usize_field(root, "num_attention_heads")?,
            kv_lora_rank: usize_field(root, "kv_lora_rank")?,
            qk_nope_head_dim: usize_field(root, "qk_nope_head_dim")?,
            qk_rope_head_dim: usize_field(root, "qk_rope_head_dim")?,
            v_head_dim: usize_field(root, "v_head_dim")?,
            num_experts: usize_field(root, "num_experts")?,
            num_experts_per_token: usize_field(root, "num_experts_per_token")?,
            num_shared_experts: usize_field(root, "num_shared_experts")?,
            moe_intermediate_size: usize_field(root, "moe_intermediate_size")?,
            routed_scaling_factor: f32_field(root, "routed_scaling_factor")?,
            moe_renormalize: get(root, "moe_renormalize")?
                .as_bool()
                .ok_or_else(|| LinearError::Config("`moe_renormalize` is not a bool".into()))?,
            first_k_dense_replace: usize_field(root, "first_k_dense_replace")?,
            intermediate_size: usize_field(root, "intermediate_size")?,
            full_attn_layers,
            eos_token_id: u32::try_from(usize_field(root, "eos_token_id")?)
                .map_err(|_| LinearError::Config("`eos_token_id` exceeds u32".into()))?,
        };
        if config.num_experts_per_token == 0
            || config.num_experts_per_token > config.num_experts
            || config.short_conv_kernel_size == 0
            || config
                .full_attn_layers
                .iter()
                .any(|&l| l == 0 || l > config.num_hidden_layers)
        {
            return Err(LinearError::Config(
                "inconsistent layer map or expert counts".into(),
            ));
        }
        Ok(config)
    }

    /// True when zero-based `layer` uses MLA.
    #[must_use]
    pub fn is_mla(&self, layer: usize) -> bool {
        self.full_attn_layers.contains(&(layer + 1))
    }

    /// True when zero-based `layer` has a dense MLP instead of `MoE`.
    #[must_use]
    pub const fn is_dense(&self, layer: usize) -> bool {
        layer < self.first_k_dense_replace
    }

    /// Bytes of one routed expert (three bf16 matrices).
    #[must_use]
    pub const fn expert_bytes(&self) -> usize {
        3 * self.hidden_size * self.moe_intermediate_size * 2
    }
}

/// Any failure loading or running a Kimi Linear checkpoint.
#[derive(Debug)]
pub enum LinearError {
    Config(String),
    Missing(String),
    Tensor {
        name: String,
        detail: String,
    },
    Read(SafeTensorError),
    Capacity {
        have: usize,
        add: usize,
        capacity: usize,
    },
    BrokenSession,
    Cancelled,
}

impl fmt::Display for LinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(detail) => write!(f, "kimi_linear config: {detail}"),
            Self::Missing(name) => write!(f, "checkpoint has no tensor `{name}`"),
            Self::Tensor { name, detail } => write!(f, "tensor `{name}`: {detail}"),
            Self::Read(error) => write!(f, "{error}"),
            Self::Capacity {
                have,
                add,
                capacity,
            } => write!(
                f,
                "session holds {have} tokens; adding {add} needs 1..={} (capacity {capacity})",
                capacity.saturating_sub(*have)
            ),
            Self::BrokenSession => write!(
                f,
                "the session was left partially updated by an earlier error; reset it"
            ),
            Self::Cancelled => write!(f, "generation cancelled"),
        }
    }
}

impl std::error::Error for LinearError {}

impl From<SafeTensorError> for LinearError {
    fn from(error: SafeTensorError) -> Self {
        Self::Read(error)
    }
}

/// Little-endian bytes to bf16 words.
fn to_words(raw: &[u8]) -> Vec<u16> {
    let mut words = Vec::with_capacity(raw.len() / 2);
    words_into(&mut words, raw);
    words
}

/// As [`to_words`] into `words`, reusing its allocation (see [`ExpertStore::spare`]).
fn words_into(words: &mut Vec<u16>, raw: &[u8]) {
    words.clear();
    words.extend(
        raw.chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]])),
    );
}

/// A bf16 `[out][inp]` matrix, read whole.
fn matrix(
    index: &SafeTensorIndex,
    name: &str,
    out: usize,
    inp: usize,
) -> Result<Vec<u16>, LinearError> {
    let tensor = index
        .tensor(name)
        .ok_or_else(|| LinearError::Missing(name.into()))?;
    if tensor.dtype != DType::Bf16 || tensor.shape != [out, inp] {
        return Err(LinearError::Tensor {
            name: name.into(),
            detail: format!(
                "is {:?} {:?}, expected Bf16 [{out}, {inp}]",
                tensor.dtype, tensor.shape
            ),
        });
    }
    Ok(to_words(&index.read_raw(tensor)?))
}

/// Any-dtype tensor of exactly `len` elements, widened to fp32.
fn vector(index: &SafeTensorIndex, name: &str, len: usize) -> Result<Vec<f32>, LinearError> {
    let tensor = index
        .tensor(name)
        .ok_or_else(|| LinearError::Missing(name.into()))?;
    if tensor.numel() != len {
        return Err(LinearError::Tensor {
            name: name.into(),
            detail: format!("has {} elements, expected {len}", tensor.numel()),
        });
    }
    let mut out = vec![0.0; len];
    index.read_f32(tensor, &mut out)?;
    Ok(out)
}

/// A bf16 `[out][inp]` matrix read only through products.
struct Weight {
    words: Vec<u16>,
    out: usize,
    inp: usize,
}

impl Weight {
    const fn new(words: Vec<u16>, out: usize, inp: usize) -> Self {
        Self { words, out, inp }
    }

    fn mul_cpu(&self, y: &mut [f32], x: &[f32], rows: usize) {
        Matrix::Bf16(&self.words).mul_rows(y, x, rows, self.inp, self.out, None);
    }
}

/// Products sharing an input.
struct Mul<'a> {
    x: &'a [f32],
    rows: usize,
    parts: Vec<(&'a Weight, &'a mut [f32])>,
}

/// Runs `muls` on `accel` together, or each product on the CPU when there is none or it
/// declines. Each product's CPU result depends only on its own weight and input, so how
/// products are gathered never changes the reference path's floats.
fn products(accel: Accel<'_>, muls: &mut [Mul<'_>]) {
    if let Some(device) = accel {
        let mut jobs: Vec<Bf16Job<'_>> = muls
            .iter_mut()
            .filter(|m| !m.parts.is_empty())
            .map(|m| Bf16Job {
                x: m.x,
                rows: m.rows,
                inp: m.parts[0].0.inp,
                parts: m
                    .parts
                    .iter_mut()
                    .map(|(w, y)| (&w.words[..], w.out, &mut **y))
                    .collect(),
            })
            .collect();
        if device.run_bf16(&mut jobs) {
            return;
        }
    }
    for m in muls.iter_mut() {
        for (w, y) in &mut m.parts {
            w.mul_cpu(y, m.x, m.rows);
        }
    }
}

fn silu_mul(g: &mut [f32], u: &[f32]) {
    for (gi, &ui) in g.iter_mut().zip(u) {
        *gi = silu(*gi) * ui;
    }
}

struct Kda {
    q: Weight,
    k: Weight,
    v: Weight,
    b: Weight,
    f_a: Weight,
    f_b: Weight,
    g_a: Weight,
    g_b: Weight,
    o: Weight,
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
}

struct Mla {
    q: Weight,
    kv_a: Weight,
    kv_a_norm: Vec<f32>,
    kv_b: Weight,
    o: Weight,
}

enum Attn {
    Kda(Box<Kda>),
    Mla(Box<Mla>),
}

struct Mlp {
    gate: Weight,
    up: Weight,
    down: Weight,
    inter: usize,
}

enum Ffn {
    Dense(Mlp),
    Moe {
        gate: Vec<f32>,
        bias: Vec<f32>,
        shared: Mlp,
    },
}

struct Layer {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
    attn: Attn,
    ffn: Ffn,
}

/// Routed-expert cache counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpertStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub read_s: f64,
    pub slots: usize,
}

struct Slot {
    key: (usize, usize),
    w1: Weight,
    w3: Weight,
    w2: Weight,
    used: u64,
}

/// Rewrites an expert matrix's bf16 words as it enters the cache (`words` is
/// `[rows][cols]`), for evaluating another storage format with this checkpoint: for
/// example rounding each block to MXFP4 and back, which bf16 holds exactly.
pub type ExpertTransform = Box<dyn Fn(&mut [u16], usize)>;

/// Experts read per batch: enough requests in flight to keep the drive busy, while the
/// read buffers stay a bounded pool (96 buffers, ~450 MB for the 48B model).
const READ_CHUNK: usize = 32;

struct ExpertStore {
    capacity: usize,
    slots: Vec<Slot>,
    map: HashMap<(usize, usize), usize>,
    clock: u64,
    stats: ExpertStats,
    transform: Option<ExpertTransform>,
    /// Read buffers kept between batches. Allocating fresh ones for every miss cost
    /// ~14 million page faults (49 s of kernel time) in a 639-token prompt pass.
    spare: Vec<Vec<u8>>,
}

impl ExpertStore {
    /// Makes every expert in `wanted` for `layer` resident, reading the missing ones in
    /// batches of [`READ_CHUNK`]. Evicts least-recently-used slots not wanted now.
    fn ensure(
        &mut self,
        index: &SafeTensorIndex,
        config: &LinearConfig,
        layer: usize,
        wanted: &[usize],
    ) -> Result<(), LinearError> {
        self.clock += 1;
        let now = self.clock;
        let mut missing = Vec::new();
        for &expert in wanted {
            if let Some(&slot) = self.map.get(&(layer, expert)) {
                self.slots[slot].used = now;
                self.stats.hits += 1;
            } else {
                missing.push(expert);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        let (e, inter) = (config.hidden_size, config.moe_intermediate_size);
        for chunk in missing.chunks(READ_CHUNK) {
            let mut requests = Vec::with_capacity(3 * chunk.len());
            for &expert in chunk {
                for (part, shape) in [("w1", [inter, e]), ("w3", [inter, e]), ("w2", [e, inter])] {
                    let name = format!(
                        "model.layers.{layer}.block_sparse_moe.experts.{expert}.{part}.weight"
                    );
                    let tensor = index
                        .tensor(&name)
                        .ok_or_else(|| LinearError::Missing(name.clone()))?;
                    if tensor.dtype != DType::Bf16 || tensor.shape != shape {
                        return Err(LinearError::Tensor {
                            name,
                            detail: format!(
                                "is {:?} {:?}, expected Bf16 {shape:?}",
                                tensor.dtype, tensor.shape
                            ),
                        });
                    }
                    let mut buffer = self.spare.pop().unwrap_or_default();
                    buffer.resize(tensor.nbytes, 0);
                    requests.push(ReadRequest {
                        shard: tensor.shard,
                        offset: tensor.offset,
                        buffer,
                    });
                }
            }
            let start = Instant::now();
            let buffers = index.read_batch(requests)?;
            self.stats.read_s += start.elapsed().as_secs_f64();
            self.stats.misses += chunk.len() as u64;
            self.stats.bytes_read += buffers.iter().map(|b| b.len() as u64).sum::<u64>();
            for (&expert, parts) in chunk.iter().zip(buffers.chunks_exact(3)) {
                let at = if self.slots.len() < self.capacity {
                    self.slots.push(Slot {
                        key: (layer, expert),
                        w1: Weight::new(Vec::new(), inter, e),
                        w3: Weight::new(Vec::new(), inter, e),
                        w2: Weight::new(Vec::new(), e, inter),
                        used: now,
                    });
                    self.slots.len() - 1
                } else {
                    // Oldest slot not stamped `now` (the capacity holds a whole layer, so
                    // one always exists). Its word buffers are reused in place.
                    let victim = self
                        .slots
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| s.used != now)
                        .min_by_key(|(_, s)| s.used)
                        .map(|(i, _)| i)
                        .expect("capacity holds every expert of one layer");
                    self.map.remove(&self.slots[victim].key);
                    let slot = &mut self.slots[victim];
                    slot.key = (layer, expert);
                    slot.used = now;
                    victim
                };
                let slot = &mut self.slots[at];
                words_into(&mut slot.w1.words, &parts[0]);
                words_into(&mut slot.w3.words, &parts[1]);
                words_into(&mut slot.w2.words, &parts[2]);
                if let Some(transform) = &self.transform {
                    for w in [&mut slot.w1, &mut slot.w3, &mut slot.w2] {
                        transform(&mut w.words, w.inp);
                    }
                }
                self.map.insert((layer, expert), at);
            }
            self.spare = buffers;
        }
        self.stats.slots = self.slots.len();
        Ok(())
    }

    fn get(&self, layer: usize, expert: usize) -> &Slot {
        &self.slots[self.map[&(layer, expert)]]
    }
}

/// A loaded checkpoint: resident trunk, streamed experts.
pub struct LinearModel {
    pub config: LinearConfig,
    index: SafeTensorIndex,
    embed: Vec<u16>,
    lm_head: Weight,
    norm: Vec<f32>,
    layers: Vec<Layer>,
    experts: ExpertStore,
}

#[derive(Clone)]
enum State {
    Kda { recurrent: Vec<f32>, conv: Vec<f32> },
    Mla { kv: Vec<f32>, rope: Vec<f32> },
}

/// Incremental decoding state: every layer's attention memory and the tokens consumed.
/// Cloning it snapshots a shared prefix (a chat's tool declarations) for reuse.
#[derive(Clone)]
pub struct LinearSession {
    states: Vec<State>,
    ids: Vec<u32>,
    capacity: usize,
    broken: bool,
}

impl LinearSession {
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    #[must_use]
    pub const fn is_broken(&self) -> bool {
        self.broken
    }

    /// Forgets every consumed token. MLA cache rows are written before they are read.
    pub fn reset(&mut self) {
        for state in &mut self.states {
            if let State::Kda { recurrent, conv } = state {
                recurrent.fill(0.0);
                conv.fill(0.0);
            }
        }
        self.ids.clear();
        self.broken = false;
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// fla's softplus: `x` above the threshold 20, `ln(1 + e^x)` otherwise.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

impl LinearModel {
    /// Indexes `directory`, reads its config and every non-expert tensor.
    /// `expert_budget_bytes` bounds the routed-expert cache; it is raised to hold at
    /// least one layer's full expert set.
    ///
    /// # Errors
    /// Returns [`LinearError`] for an unsupported config or a missing/misshapen tensor.
    #[allow(clippy::too_many_lines)] // one tensor list, read top to bottom
    pub fn load(
        directory: impl AsRef<Path>,
        expert_budget_bytes: usize,
    ) -> Result<Self, LinearError> {
        let directory = directory.as_ref();
        let config = LinearConfig::from_path(directory.join("config.json"))?;
        let index = SafeTensorIndex::open(directory)?;
        let c = &config;
        let e = c.hidden_size;
        let p = c.kda_num_heads * c.kda_head_dim;
        let d = c.kda_head_dim;
        let k = c.short_conv_kernel_size;
        let qh = c.qk_nope_head_dim + c.qk_rope_head_dim;
        let w = |name: &str, out: usize, inp: usize| -> Result<Weight, LinearError> {
            Ok(Weight::new(matrix(&index, name, out, inp)?, out, inp))
        };
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for l in 0..c.num_hidden_layers {
            let at = |s: &str| format!("model.layers.{l}.{s}");
            let attn = if c.is_mla(l) {
                let heads = c.num_attention_heads;
                Attn::Mla(Box::new(Mla {
                    q: w(&at("self_attn.q_proj.weight"), heads * qh, e)?,
                    kv_a: w(
                        &at("self_attn.kv_a_proj_with_mqa.weight"),
                        c.kv_lora_rank + c.qk_rope_head_dim,
                        e,
                    )?,
                    kv_a_norm: vector(
                        &index,
                        &at("self_attn.kv_a_layernorm.weight"),
                        c.kv_lora_rank,
                    )?,
                    kv_b: w(
                        &at("self_attn.kv_b_proj.weight"),
                        heads * (c.qk_nope_head_dim + c.v_head_dim),
                        c.kv_lora_rank,
                    )?,
                    o: w(&at("self_attn.o_proj.weight"), e, heads * c.v_head_dim)?,
                }))
            } else {
                let heads = c.kda_num_heads;
                Attn::Kda(Box::new(Kda {
                    q: w(&at("self_attn.q_proj.weight"), p, e)?,
                    k: w(&at("self_attn.k_proj.weight"), p, e)?,
                    v: w(&at("self_attn.v_proj.weight"), p, e)?,
                    b: w(&at("self_attn.b_proj.weight"), heads, e)?,
                    f_a: w(&at("self_attn.f_a_proj.weight"), d, e)?,
                    f_b: w(&at("self_attn.f_b_proj.weight"), p, d)?,
                    g_a: w(&at("self_attn.g_a_proj.weight"), d, e)?,
                    g_b: w(&at("self_attn.g_b_proj.weight"), p, d)?,
                    o: w(&at("self_attn.o_proj.weight"), e, p)?,
                    q_conv: vector(&index, &at("self_attn.q_conv1d.weight"), p * k)?,
                    k_conv: vector(&index, &at("self_attn.k_conv1d.weight"), p * k)?,
                    v_conv: vector(&index, &at("self_attn.v_conv1d.weight"), p * k)?,
                    a_log: vector(&index, &at("self_attn.A_log"), heads)?,
                    dt_bias: vector(&index, &at("self_attn.dt_bias"), p)?,
                    o_norm: vector(&index, &at("self_attn.o_norm.weight"), d)?,
                }))
            };
            let mlp = |prefix: &str, inter: usize| -> Result<Mlp, LinearError> {
                Ok(Mlp {
                    gate: w(&at(&format!("{prefix}.gate_proj.weight")), inter, e)?,
                    up: w(&at(&format!("{prefix}.up_proj.weight")), inter, e)?,
                    down: w(&at(&format!("{prefix}.down_proj.weight")), e, inter)?,
                    inter,
                })
            };
            let ffn = if c.is_dense(l) {
                Ffn::Dense(mlp("mlp", c.intermediate_size)?)
            } else {
                Ffn::Moe {
                    gate: vector(
                        &index,
                        &at("block_sparse_moe.gate.weight"),
                        c.num_experts * e,
                    )?,
                    bias: vector(
                        &index,
                        &at("block_sparse_moe.gate.e_score_correction_bias"),
                        c.num_experts,
                    )?,
                    shared: mlp(
                        "block_sparse_moe.shared_experts",
                        c.moe_intermediate_size * c.num_shared_experts,
                    )?,
                }
            };
            layers.push(Layer {
                in_norm: vector(&index, &at("input_layernorm.weight"), e)?,
                post_norm: vector(&index, &at("post_attention_layernorm.weight"), e)?,
                attn,
                ffn,
            });
        }
        let embed = matrix(&index, "model.embed_tokens.weight", c.vocab_size, e)?;
        let lm_head = w("lm_head.weight", c.vocab_size, e)?;
        let norm = vector(&index, "model.norm.weight", e)?;
        let capacity = (expert_budget_bytes / c.expert_bytes()).max(c.num_experts);
        Ok(Self {
            experts: ExpertStore {
                capacity,
                slots: Vec::new(),
                map: HashMap::new(),
                clock: 0,
                stats: ExpertStats::default(),
                transform: None,
                spare: Vec::new(),
            },
            config,
            index,
            embed,
            lm_head,
            norm,
            layers,
        })
    }

    /// Applies `transform` to every expert from now on and forgets every cached one,
    /// so no expert read before the change is used after it. The slots' memory stays
    /// allocated and is overwritten as they are reused, not freed and faulted in again.
    pub fn set_expert_transform(&mut self, transform: Option<ExpertTransform>) {
        let store = &mut self.experts;
        store.transform = transform;
        store.map.clear();
        for slot in &mut store.slots {
            // No live key, so evicting it later cannot unmap a fresh copy elsewhere.
            slot.key = (usize::MAX, usize::MAX);
        }
    }

    #[must_use]
    pub fn expert_stats(&self) -> ExpertStats {
        self.experts.stats
    }

    /// A session able to hold `capacity` positions.
    #[must_use]
    pub fn session(&self, capacity: usize) -> LinearSession {
        let c = &self.config;
        let p = c.kda_num_heads * c.kda_head_dim;
        let states = (0..c.num_hidden_layers)
            .map(|l| {
                if c.is_mla(l) {
                    State::Mla {
                        kv: vec![
                            0.0;
                            capacity
                                * c.num_attention_heads
                                * (c.qk_nope_head_dim + c.v_head_dim)
                        ],
                        rope: vec![0.0; capacity * c.qk_rope_head_dim],
                    }
                } else {
                    State::Kda {
                        recurrent: vec![0.0; c.kda_num_heads * c.kda_head_dim * c.kda_head_dim],
                        conv: vec![0.0; 3 * p * (c.short_conv_kernel_size - 1)],
                    }
                }
            })
            .collect();
        LinearSession {
            states,
            ids: Vec::new(),
            capacity,
            broken: false,
        }
    }

    /// Feeds `ids` after everything `session` consumed and returns the logits for the
    /// last of them.
    ///
    /// # Errors
    /// [`LinearError::Capacity`] (checked first, session untouched), or a read,
    /// cancellation or tensor error, after which the session is marked broken.
    pub fn feed(
        &mut self,
        session: &mut LinearSession,
        ids: &[u32],
        accel: Accel<'_>,
        mut keep_running: impl FnMut() -> bool,
    ) -> Result<Vec<f32>, LinearError> {
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
        let result = self.run(session, ids, accel, false, &mut keep_running);
        match result {
            Ok(logits) => {
                session.ids.extend_from_slice(ids);
                Ok(logits)
            }
            Err(error) => {
                session.broken = true;
                Err(error)
            }
        }
    }

    /// As [`Self::feed`], returning the logits after every one of `ids` (`[ids][vocab]`),
    /// for scoring a text: row `i` predicts the token after `ids[i]`.
    ///
    /// # Errors
    /// As [`Self::feed`].
    pub fn score(
        &mut self,
        session: &mut LinearSession,
        ids: &[u32],
        accel: Accel<'_>,
        mut keep_running: impl FnMut() -> bool,
    ) -> Result<Vec<f32>, LinearError> {
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
        match self.run(session, ids, accel, true, &mut keep_running) {
            Ok(logits) => {
                session.ids.extend_from_slice(ids);
                Ok(logits)
            }
            Err(error) => {
                session.broken = true;
                Err(error)
            }
        }
    }

    fn run(
        &mut self,
        session: &mut LinearSession,
        ids: &[u32],
        accel: Accel<'_>,
        all_logits: bool,
        keep_running: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<f32>, LinearError> {
        let c = &self.config;
        let e = c.hidden_size;
        let t = ids.len();
        let cached = session.ids.len();
        let mut h = vec![0.0_f32; t * e];
        for (row, &id) in h.chunks_exact_mut(e).zip(ids) {
            Matrix::Bf16(&self.embed).row_into(row, id as usize, e);
        }
        let mut hin = vec![0.0_f32; t * e];
        let mut tmp = vec![0.0_f32; t * e];
        for (l, (layer, state)) in self.layers.iter().zip(&mut session.states).enumerate() {
            if !keep_running() {
                return Err(LinearError::Cancelled);
            }
            for (y, x) in hin.chunks_exact_mut(e).zip(h.chunks_exact(e)) {
                rmsnorm(y, x, &layer.in_norm, c.rms_norm_eps);
            }
            match (&layer.attn, state) {
                (Attn::Kda(w), State::Kda { recurrent, conv }) => {
                    kda(&mut tmp, &hin, w, c, t, recurrent, conv, accel);
                }
                (Attn::Mla(w), State::Mla { kv, rope }) => {
                    mla(&mut tmp, &hin, w, c, t, kv, rope, cached, accel);
                }
                _ => unreachable!("session states follow the layer map"),
            }
            for (hi, &ti) in h.iter_mut().zip(&tmp) {
                *hi += ti;
            }
            for (y, x) in hin.chunks_exact_mut(e).zip(h.chunks_exact(e)) {
                rmsnorm(y, x, &layer.post_norm, c.rms_norm_eps);
            }
            match &layer.ffn {
                Ffn::Dense(m) => mlp(&mut tmp, &hin, m, t, accel),
                Ffn::Moe { gate, bias, shared } => {
                    moe(
                        &mut tmp,
                        &hin,
                        (gate, bias, shared),
                        c,
                        l,
                        t,
                        &self.index,
                        &mut self.experts,
                        accel,
                    )?;
                }
            }
            for (hi, &ti) in h.iter_mut().zip(&tmp) {
                *hi += ti;
            }
        }
        if !keep_running() {
            return Err(LinearError::Cancelled);
        }
        let first = if all_logits { 0 } else { t - 1 };
        let rows = t - first;
        let mut normed = vec![0.0_f32; rows * e];
        for (y, x) in normed
            .chunks_exact_mut(e)
            .zip(h[first * e..].chunks_exact(e))
        {
            rmsnorm(y, x, &self.norm, c.rms_norm_eps);
        }
        let mut logits = vec![0.0_f32; rows * c.vocab_size];
        products(
            accel,
            &mut [Mul {
                x: &normed,
                rows,
                parts: vec![(&self.lm_head, &mut logits)],
            }],
        );
        Ok(logits)
    }
}

#[allow(clippy::too_many_lines)] // one attention block, read top to bottom
fn kda(
    out: &mut [f32],
    x: &[f32],
    w: &Kda,
    c: &LinearConfig,
    t: usize,
    recurrent: &mut [f32],
    conv: &mut [f32],
    accel: Accel<'_>,
) {
    let heads = c.kda_num_heads;
    let d = c.kda_head_dim;
    let p = heads * d;
    let k = c.short_conv_kernel_size;
    let hist = k - 1;

    let mut q = vec![0.0_f32; t * p];
    let mut kk = vec![0.0_f32; t * p];
    let mut v = vec![0.0_f32; t * p];
    let mut bt = vec![0.0_f32; t * heads];
    let mut fa = vec![0.0_f32; t * d];
    let mut z = vec![0.0_f32; t * p];
    let mut ga = vec![0.0_f32; t * d];
    let mut gb = vec![0.0_f32; t * p];
    // Everything read from `x` (the output gate's g_a too), then both low-rank halves.
    products(
        accel,
        &mut [Mul {
            x,
            rows: t,
            parts: vec![
                (&w.q, &mut q),
                (&w.k, &mut kk),
                (&w.v, &mut v),
                (&w.b, &mut bt),
                (&w.f_a, &mut fa),
                (&w.g_a, &mut ga),
            ],
        }],
    );
    products(
        accel,
        &mut [
            Mul {
                x: &fa,
                rows: t,
                parts: vec![(&w.f_b, &mut z)],
            },
            Mul {
                x: &ga,
                rows: t,
                parts: vec![(&w.g_b, &mut gb)],
            },
        ],
    );

    let (qs, rest) = conv.split_at_mut(p * hist);
    let (ks, vs) = rest.split_at_mut(p * hist);
    shortconv_in_place(&mut q, &w.q_conv, Some(qs), p, k, t);
    shortconv_in_place(&mut kk, &w.k_conv, Some(ks), p, k, t);
    shortconv_in_place(&mut v, &w.v_conv, Some(vs), p, k, t);

    for step in 0..t {
        for hh in 0..heads {
            let at = step * p + hh * d;
            l2norm_in_place(&mut q[at..at + d], 1e-6);
            l2norm_in_place(&mut kk[at..at + d], 1e-6);
        }
    }
    // Decay: g = -exp(A_log[h]) * softplus(z + dt_bias), alpha = exp(g).
    let mut alpha = vec![0.0_f32; t * p];
    for step in 0..t {
        for hh in 0..heads {
            let a = w.a_log[hh].exp();
            for i in hh * d..(hh + 1) * d {
                let g = -a * softplus(z[step * p + i] + w.dt_bias[i]);
                alpha[step * p + i] = g.exp();
            }
        }
    }
    let qscale = 1.0_f32 / (d as f32).sqrt();
    let mut o = vec![0.0_f32; t * p];
    let mut qh = vec![0.0_f32; d];
    for hh in 0..heads {
        let s = &mut recurrent[hh * d * d..(hh + 1) * d * d];
        for step in 0..t {
            let off = step * p + hh * d;
            for (qi, &x) in qh.iter_mut().zip(&q[off..off + d]) {
                *qi = x * qscale;
            }
            kda_step(
                s,
                &mut o[off..off + d],
                &qh,
                &kk[off..off + d],
                &v[off..off + d],
                &alpha[off..off + d],
                sigmoid(bt[step * heads + hh]),
                d,
                d,
            );
        }
    }
    // Output: per-head RMSNorm, times sigmoid of the low-rank gate g_b(g_a(x)).
    for step in 0..t {
        let row = &mut o[step * p..(step + 1) * p];
        for hh in 0..heads {
            rmsnorm_in_place(&mut row[hh * d..(hh + 1) * d], &w.o_norm, c.rms_norm_eps);
        }
        for (oi, &gi) in row.iter_mut().zip(&gb[step * p..(step + 1) * p]) {
            *oi *= sigmoid(gi);
        }
    }
    products(
        accel,
        &mut [Mul {
            x: &o,
            rows: t,
            parts: vec![(&w.o, out)],
        }],
    );
}

fn mla(
    out: &mut [f32],
    x: &[f32],
    w: &Mla,
    c: &LinearConfig,
    t: usize,
    kv: &mut [f32],
    rope: &mut [f32],
    cached: usize,
    accel: Accel<'_>,
) {
    let heads = c.num_attention_heads;
    let qn = c.qk_nope_head_dim;
    let qr = c.qk_rope_head_dim;
    let vh = c.v_head_dim;
    let qh = qn + qr;
    let kvr = c.kv_lora_rank;
    let kvw = kvr + qr;
    let kvd = qn + vh;
    let scale = 1.0_f32 / (qh as f32).sqrt();

    let mut q = vec![0.0_f32; t * heads * qh];
    let mut ct = vec![0.0_f32; t * kvw];
    let mut ckv = vec![0.0_f32; t * kvr];
    products(
        accel,
        &mut [Mul {
            x,
            rows: t,
            parts: vec![(&w.q, &mut q), (&w.kv_a, &mut ct)],
        }],
    );
    for step in 0..t {
        let pos = cached + step;
        let row = &mut ct[step * kvw..(step + 1) * kvw];
        rmsnorm_in_place(&mut row[..kvr], &w.kv_a_norm, c.rms_norm_eps);
        rope[pos * qr..(pos + 1) * qr].copy_from_slice(&row[kvr..]);
        ckv[step * kvr..(step + 1) * kvr].copy_from_slice(&row[..kvr]);
    }
    products(
        accel,
        &mut [Mul {
            x: &ckv,
            rows: t,
            parts: vec![(
                &w.kv_b,
                &mut kv[cached * heads * kvd..(cached + t) * heads * kvd],
            )],
        }],
    );

    let mut acc = vec![0.0_f32; t * heads * vh];
    let mut sc = vec![0.0_f32; cached + t];
    for step in 0..t {
        let pos = cached + step;
        for hh in 0..heads {
            let qt = &q[(step * heads + hh) * qh..(step * heads + hh + 1) * qh];
            let mut m = f32::NEG_INFINITY;
            for s in 0..=pos {
                let ks = &kv[(s * heads + hh) * kvd..];
                let kr = &rope[s * qr..(s + 1) * qr];
                let mut dot = 0.0_f64;
                for i in 0..qn {
                    dot += f64::from(qt[i]) * f64::from(ks[i]);
                }
                for i in 0..qr {
                    dot += f64::from(qt[qn + i]) * f64::from(kr[i]);
                }
                sc[s] = dot as f32 * scale;
                m = m.max(sc[s]);
            }
            let mut zsum = 0.0_f64;
            for score in &mut sc[..=pos] {
                *score = (*score - m).exp();
                zsum += f64::from(*score);
            }
            let o = &mut acc[(step * heads + hh) * vh..(step * heads + hh + 1) * vh];
            for (s, &score) in sc[..=pos].iter().enumerate() {
                let pr = (f64::from(score) / zsum) as f32;
                let at = (s * heads + hh) * kvd + qn;
                for (oj, &vj) in o.iter_mut().zip(&kv[at..at + vh]) {
                    *oj += pr * vj;
                }
            }
        }
    }
    products(
        accel,
        &mut [Mul {
            x: &acc,
            rows: t,
            parts: vec![(&w.o, out)],
        }],
    );
}

/// `down(SiLU(gate x) * up x)` for `t` rows.
fn mlp(out: &mut [f32], x: &[f32], m: &Mlp, t: usize, accel: Accel<'_>) {
    let mut g = vec![0.0_f32; t * m.inter];
    let mut u = vec![0.0_f32; t * m.inter];
    products(
        accel,
        &mut [Mul {
            x,
            rows: t,
            parts: vec![(&m.gate, &mut g), (&m.up, &mut u)],
        }],
    );
    silu_mul(&mut g, &u);
    products(
        accel,
        &mut [Mul {
            x: &g,
            rows: t,
            parts: vec![(&m.down, out)],
        }],
    );
}

#[allow(clippy::too_many_lines)] // routing, both expert halves, then the mix
fn moe(
    out: &mut [f32],
    x: &[f32],
    (gate, bias, shared): (&[f32], &[f32], &Mlp),
    c: &LinearConfig,
    layer: usize,
    t: usize,
    index: &SafeTensorIndex,
    experts: &mut ExpertStore,
    accel: Accel<'_>,
) -> Result<(), LinearError> {
    let e = c.hidden_size;
    let inter = c.moe_intermediate_size;
    let topk = c.num_experts_per_token;
    let mut idx = vec![0_usize; t * topk];
    let mut wt = vec![0.0_f32; t * topk];
    let mut seen = vec![false; c.num_experts];
    let mut uniq = Vec::new();
    for step in 0..t {
        router(
            &mut idx[step * topk..(step + 1) * topk],
            &mut wt[step * topk..(step + 1) * topk],
            &x[step * e..(step + 1) * e],
            gate,
            Some(bias),
            e,
            c.num_experts,
            topk,
            c.moe_renormalize,
            c.routed_scaling_factor,
        );
        for &expert in &idx[step * topk..(step + 1) * topk] {
            if !seen[expert] {
                seen[expert] = true;
                uniq.push(expert);
            }
        }
    }
    experts.ensure(index, c, layer, &uniq)?;

    // Each expert once, over every (token, slot) that selected it, in slot order.
    let mut at = vec![usize::MAX; c.num_experts];
    for (i, &expert) in uniq.iter().enumerate() {
        at[expert] = i;
    }
    let mut slots = vec![Vec::new(); uniq.len()];
    for (slot, &selected) in idx.iter().enumerate() {
        slots[at[selected]].push(slot);
    }
    let zs: Vec<Vec<f32>> = slots
        .iter()
        .map(|s| {
            s.iter()
                .flat_map(|&slot| &x[slot / topk * e..(slot / topk + 1) * e])
                .copied()
                .collect()
        })
        .collect();
    let zeros = |width: usize| -> Vec<Vec<f32>> {
        slots.iter().map(|s| vec![0.0; s.len() * width]).collect()
    };
    let (mut g, mut u, mut ys) = (zeros(inter), zeros(inter), zeros(e));
    let mut sg = vec![0.0_f32; t * shared.inter];
    let mut su = vec![0.0_f32; t * shared.inter];
    let mut sdn = vec![0.0_f32; t * e];
    let mut muls = vec![Mul {
        x,
        rows: t,
        parts: vec![(&shared.gate, &mut sg), (&shared.up, &mut su)],
    }];
    for (((z, gi), ui), &expert) in zs.iter().zip(&mut g).zip(&mut u).zip(&uniq) {
        let w = experts.get(layer, expert);
        muls.push(Mul {
            x: z,
            rows: z.len() / e,
            parts: vec![(&w.w1, gi), (&w.w3, ui)],
        });
    }
    products(accel, &mut muls);
    silu_mul(&mut sg, &su);
    for (gi, ui) in g.iter_mut().zip(&u) {
        silu_mul(gi, ui);
    }
    let mut muls = vec![Mul {
        x: &sg,
        rows: t,
        parts: vec![(&shared.down, &mut sdn)],
    }];
    for ((gi, yi), &expert) in g.iter().zip(&mut ys).zip(&uniq) {
        let w = experts.get(layer, expert);
        muls.push(Mul {
            x: gi,
            rows: gi.len() / inter,
            parts: vec![(&w.w2, yi)],
        });
    }
    products(accel, &mut muls);

    let mut contrib = vec![0.0_f32; t * topk * e];
    for (s, yi) in slots.iter().zip(&ys) {
        for (r, &slot) in s.iter().enumerate() {
            contrib[slot * e..(slot + 1) * e].copy_from_slice(&yi[r * e..(r + 1) * e]);
        }
    }
    for step in 0..t {
        let row = &mut out[step * e..(step + 1) * e];
        row.fill(0.0);
        for j in 0..topk {
            let slot = step * topk + j;
            for (oi, &ci) in row.iter_mut().zip(&contrib[slot * e..(slot + 1) * e]) {
                *oi += wt[slot] * ci;
            }
        }
    }
    for (oi, &si) in out[..t * e].iter_mut().zip(&sdn) {
        *oi += si;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_json() -> Value {
        serde_json::json!({
            "model_type": "kimi_linear", "hidden_act": "silu", "q_lora_rank": null,
            "mla_use_nope": true, "moe_router_activation_func": "sigmoid",
            "num_expert_group": 1, "topk_group": 1, "moe_layer_freq": 1,
            "tie_word_embeddings": false, "hidden_size": 2304, "num_hidden_layers": 27,
            "vocab_size": 163_840, "rms_norm_eps": 1e-5, "num_attention_heads": 32,
            "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
            "v_head_dim": 128, "num_experts": 256, "num_experts_per_token": 8,
            "num_shared_experts": 1, "moe_intermediate_size": 1024,
            "routed_scaling_factor": 2.446, "moe_renormalize": true,
            "first_k_dense_replace": 1, "intermediate_size": 9216, "eos_token_id": 163_586,
            "linear_attn_config": {"full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
                "head_dim": 128, "num_heads": 32, "short_conv_kernel_size": 4}
        })
    }

    #[test]
    fn released_48b_config_parses_with_one_based_layer_map() {
        let c = LinearConfig::from_value(&config_json()).unwrap();
        assert!(c.is_mla(3) && c.is_mla(26) && !c.is_mla(0) && !c.is_mla(4));
        assert!(c.is_dense(0) && !c.is_dense(1));
        assert_eq!(c.expert_bytes(), 3 * 2304 * 1024 * 2);
    }

    #[test]
    fn unimplemented_shapes_are_refused_not_defaulted() {
        for (key, value) in [
            ("hidden_act", serde_json::json!("gelu")),
            ("q_lora_rank", serde_json::json!(1536)),
            ("num_expert_group", serde_json::json!(8)),
            ("model_type", serde_json::json!("kimi_k3")),
        ] {
            let mut root = config_json();
            root[key] = value;
            assert!(LinearConfig::from_value(&root).is_err(), "{key}");
        }
        let mut root = config_json();
        root["linear_attn_config"]["gate_lower_bound"] = serde_json::json!(-5.0);
        assert!(LinearConfig::from_value(&root).is_err());
        let mut root = config_json();
        root.as_object_mut().unwrap().remove("rms_norm_eps");
        assert!(LinearConfig::from_value(&root).is_err());
    }

    #[test]
    fn words_are_little_endian_pairs() {
        assert_eq!(to_words(&[0x34, 0x12, 0xCD, 0xAB, 0xFF]), [0x1234, 0xABCD]);
    }

    /// `cargo test --release -p kimi-k3-core --lib -- --ignored --nocapture word_rate`
    #[test]
    #[ignore = "timing only"]
    fn word_rate() {
        let raw: Vec<u8> = (0..512_u32 << 20).map(|i| (i * 31) as u8).collect();
        let start = Instant::now();
        let words = std::hint::black_box(to_words(std::hint::black_box(&raw)));
        let s = start.elapsed().as_secs_f64();
        assert_eq!(
            words[words.len() - 1],
            u16::from_le_bytes([raw[raw.len() - 2], raw[raw.len() - 1]])
        );
        println!("to_words: {:.1} GB/s", raw.len() as f64 / s / 1e9);
    }

    #[test]
    fn softplus_matches_fla_threshold_and_ln1p() {
        assert!((softplus(25.0) - 25.0).abs() < f32::EPSILON);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!((softplus(-10.0) - (-10.0_f32).exp().ln_1p()).abs() < 1e-12);
    }
}
