//! Binds the released checkpoint's real tensor names into the structures
//! [`crate::layer`]/[`crate::model`] consume.
//!
//! Names and the elementwise-versus-matmul split come from `src/model/k3_bind.c`,
//! cross-checked against the real checkpoint's `model.safetensors.index.json`
//! (`docs/RUST_PORT.md`'s "Real-checkpoint tensor names" section). Every
//! matmul-only weight is bf16 on disk and is bound as [`Matrix::Bf16`], kept in
//! its own bytes and never widened to fp32 -- the same split C's `reqn`
//! (matmul-tagged) versus `reqw` (elementwise fp32) makes.
//!
//! A `MoE` layer's routed experts are never bound resident here: 896 experts per
//! layer cannot fit in memory at once, so every bound layer gets
//! `RoutedExperts::Streamed`, fetched per token through an [`crate::layer::ExpertSource`]
//! (see [`crate::cache::CachedExperts`]).
//!
//! [`BoundStorage`] separates *reading* tensors from *borrowing* them: every
//! tensor a caller wants is read once into an owned buffer, then
//! [`BoundStorage::layer_weights`]/[`BoundStorage::model`] build borrowed views
//! over it with no further I/O. This lets a caller bind one layer (or a small
//! window of layers, for the eventual trunk-ring streaming reader) without
//! reading the whole checkpoint's ~109 GB packed trunk resident at once, which
//! does not fit in memory on any machine in this lab.

use std::collections::HashMap;
use std::fmt;

use crate::{
    config::K3Config,
    io::{PendingBatch, ReadRequest},
    layer::{
        Attention, KdaWeights, LayerWeights, Matrix, MlaWeights, Mlp, MoeWeights, RoutedExperts,
    },
    model::Model,
    safetensors::{SafeTensorError, SafeTensorIndex, TensorInfo, widen_into},
};

/// A single-read safety margin: `read_bf16_into` chunks because a real read past
/// roughly 2 GiB fails with EINVAL (see its own docs). [`BoundStorage::submit_layer`]
/// reads one tensor per request with no such chunking, so it refuses to plan a
/// read past this bound rather than risk the same failure asynchronously, where
/// there is no per-chunk loop to catch it early.
const MAX_SAFE_SINGLE_READ_BYTES: usize = 1 << 30;

/// Whether a bound tensor is read elementwise (kept fp32) or only through a
/// matmul (kept in the checkpoint's own bf16 bytes). Mirrors C's `reqw` versus
/// `reqn` split; see the module docs.
#[derive(Clone, Copy, Debug)]
enum TensorKind {
    F32,
    Bf16,
}

/// The tensors one decoder layer needs, in the exact set [`BoundStorage::load_layer`]
/// reads -- shared with [`BoundStorage::submit_layer`] so the blocking and async
/// paths can never name a different set of tensors for the same layer.
fn layer_plan(
    index: &SafeTensorIndex,
    config: &K3Config,
    layer: usize,
) -> Vec<(String, TensorKind)> {
    let mut plan = Vec::new();
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attention_res_norm.weight",
        "self_attention_res_proj.weight",
        "mlp_res_norm.weight",
        "mlp_res_proj.weight",
    ] {
        plan.push((layer_name(layer, suffix), TensorKind::F32));
    }
    attention_plan(config, layer, &mut plan);
    mlp_plan(index, config, layer, &mut plan);
    plan
}

fn attention_plan(config: &K3Config, layer: usize, plan: &mut Vec<(String, TensorKind)>) {
    if config.is_mla(layer) {
        plan.push((
            layer_name(layer, "self_attn.q_a_proj.weight"),
            TensorKind::Bf16,
        ));
        plan.push((
            layer_name(layer, "self_attn.q_a_layernorm.weight"),
            TensorKind::F32,
        ));
        plan.push((
            layer_name(layer, "self_attn.q_b_proj.weight"),
            TensorKind::Bf16,
        ));
        plan.push((
            layer_name(layer, "self_attn.kv_a_proj_with_mqa.weight"),
            TensorKind::Bf16,
        ));
        plan.push((
            layer_name(layer, "self_attn.kv_a_layernorm.weight"),
            TensorKind::F32,
        ));
        plan.push((
            layer_name(layer, "self_attn.kv_b_proj.weight"),
            TensorKind::Bf16,
        ));
        plan.push((
            layer_name(layer, "self_attn.o_proj.weight"),
            TensorKind::Bf16,
        ));
        if config.mla_use_output_gate {
            plan.push((
                layer_name(layer, "self_attn.g_proj.weight"),
                TensorKind::Bf16,
            ));
        }
        return;
    }
    for suffix in [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.g_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.f_a_proj.weight",
        "self_attn.f_b_proj.weight",
        "self_attn.b_proj.weight",
    ] {
        plan.push((layer_name(layer, suffix), TensorKind::Bf16));
    }
    for suffix in [
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.A_log",
        "self_attn.dt_bias",
        "self_attn.o_norm.weight",
    ] {
        plan.push((layer_name(layer, suffix), TensorKind::F32));
    }
}

fn mlp_plan(
    index: &SafeTensorIndex,
    config: &K3Config,
    layer: usize,
    plan: &mut Vec<(String, TensorKind)>,
) {
    if config.is_dense(layer) {
        for suffix in [
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            plan.push((layer_name(layer, suffix), TensorKind::Bf16));
        }
        return;
    }
    plan.push((
        layer_name(layer, "block_sparse_moe.gate.weight"),
        TensorKind::F32,
    ));
    let bias_name = layer_name(layer, "block_sparse_moe.gate.e_score_correction_bias");
    if index.tensor(&bias_name).is_some() {
        plan.push((bias_name, TensorKind::F32));
    }
    for suffix in [
        "block_sparse_moe.routed_expert_down_proj.weight",
        "block_sparse_moe.routed_expert_up_proj.weight",
    ] {
        plan.push((layer_name(layer, suffix), TensorKind::Bf16));
    }
    plan.push((
        layer_name(layer, "block_sparse_moe.routed_expert_norm.weight"),
        TensorKind::F32,
    ));
    for suffix in [
        "block_sparse_moe.shared_experts.gate_proj.weight",
        "block_sparse_moe.shared_experts.up_proj.weight",
        "block_sparse_moe.shared_experts.down_proj.weight",
    ] {
        plan.push((layer_name(layer, suffix), TensorKind::Bf16));
    }
}

/// Every tensor name lives under this prefix except `lm_head`, which the
/// checkpoint stores one level up (`language_model.lm_head.weight`, no `.model.`).
const PREFIX: &str = "language_model.model";

/// Bound size for one raw bf16 read; see [`BoundStorage::read_bf16_into`] for why
/// this can't just be one `read_raw` call for a multi-gigabyte tensor.
const BF16_CHUNK_BYTES: usize = 4 << 20;

#[derive(Debug)]
pub enum BindError {
    /// The checkpoint has no tensor by this name.
    MissingTensor(String),
    /// The tensor exists but could not be read.
    Read(String, SafeTensorError),
    /// [`BoundStorage::model`] was asked to build a model but a layer was never loaded.
    LayerNotLoaded(usize),
    /// [`BoundStorage::submit_layer`] refused to plan an unchunked single read past
    /// [`MAX_SAFE_SINGLE_READ_BYTES`].
    TensorTooLargeForOneRead { name: String, nbytes: usize },
}

impl fmt::Display for BindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor(name) => {
                write!(formatter, "checkpoint has no tensor named {name:?}")
            }
            Self::Read(name, error) => write!(formatter, "cannot read tensor {name:?}: {error}"),
            Self::LayerNotLoaded(layer) => {
                write!(
                    formatter,
                    "layer {layer} was never loaded into this storage"
                )
            }
            Self::TensorTooLargeForOneRead { name, nbytes } => write!(
                formatter,
                "tensor {name:?} is {nbytes} bytes, too large for one unchunked async read"
            ),
        }
    }
}

impl std::error::Error for BindError {}

/// Every tensor byte a bound [`Model`] or [`LayerWeights`] borrows from. Must
/// outlive whatever it was used to build.
#[derive(Default)]
pub struct BoundStorage {
    f32: HashMap<String, Vec<f32>>,
    bf16: HashMap<String, Vec<u16>>,
}

/// One tensor [`BoundStorage::submit_layer`] has submitted a read for, along
/// with what [`BoundStorage::absorb_layer`] needs to store the result: which map
/// it belongs in, and (for the `F32` path) the source `dtype` to widen from.
struct PlannedRead {
    name: String,
    kind: TensorKind,
    tensor: TensorInfo,
}

/// A whole layer's reads, submitted but not yet waited on. Produced by
/// [`BoundStorage::submit_layer`], consumed by [`BoundStorage::absorb_layer`].
pub(crate) struct PendingLayer {
    reads: Vec<PlannedRead>,
    pending: PendingBatch,
}

impl BoundStorage {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads and caches the embedding, final norm, LM head, and the output
    /// attention-residual aggregator (present on the released checkpoint) --
    /// everything [`Model`] needs outside its layer stack.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::MissingTensor`] if a required tensor is absent, or
    /// [`BindError::Read`] if a present one cannot be read in full.
    pub fn load_top_level(&mut self, index: &SafeTensorIndex) -> Result<(), BindError> {
        self.load_bf16(index, "embed_tokens.weight")?;
        // Not under `full_name`'s prefix: the checkpoint stores this one level up
        // from every other tensor (`language_model.lm_head.weight`, no `.model.`).
        self.read_bf16_into(index, "language_model.lm_head.weight")?;
        self.load_f32(index, "norm.weight")?;
        if index
            .tensor(&full_name("output_attn_res_norm.weight"))
            .is_some()
        {
            self.load_f32(index, "output_attn_res_norm.weight")?;
            self.load_f32(index, "output_attn_res_proj.weight")?;
        }
        Ok(())
    }

    /// Reads and caches every tensor one decoder layer needs, per `config`.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::MissingTensor`] if a required tensor is absent, or
    /// [`BindError::Read`] if a present one cannot be read in full.
    pub fn load_layer(
        &mut self,
        index: &SafeTensorIndex,
        config: &K3Config,
        layer: usize,
    ) -> Result<(), BindError> {
        for (name, kind) in layer_plan(index, config, layer) {
            match kind {
                TensorKind::F32 => self.read_f32_into(index, &name)?,
                TensorKind::Bf16 => self.read_bf16_into(index, &name)?,
            }
        }
        Ok(())
    }

    /// Submits every tensor one layer needs without waiting for any of them, so
    /// the caller can compute on an already-resident layer -- no I/O involved --
    /// while the operating system services these reads in the background.
    /// [`Self::absorb_layer`] collects the results later. This, not a reader
    /// thread, is `crate::trunk`'s prefetch: [`crate::io::ShardFiles`]'s own
    /// submit/wait split, driven from here.
    ///
    /// One read per tensor, never chunked, unlike [`Self::load_layer`]'s
    /// [`Self::read_bf16_into`] -- see [`MAX_SAFE_SINGLE_READ_BYTES`] for why that
    /// is safe for real per-layer tensors and refused rather than risked otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::MissingTensor`] if a required tensor is absent, or
    /// [`BindError::TensorTooLargeForOneRead`] if one exceeds the safe bound.
    pub(crate) fn submit_layer(
        index: &SafeTensorIndex,
        config: &K3Config,
        layer: usize,
    ) -> Result<PendingLayer, BindError> {
        let plan = layer_plan(index, config, layer);
        let mut reads = Vec::with_capacity(plan.len());
        let mut requests = Vec::with_capacity(plan.len());
        for (name, kind) in plan {
            let tensor = tensor_or_missing(index, &name)?.clone();
            if tensor.nbytes > MAX_SAFE_SINGLE_READ_BYTES {
                return Err(BindError::TensorTooLargeForOneRead {
                    name,
                    nbytes: tensor.nbytes,
                });
            }
            requests.push(ReadRequest {
                shard: tensor.shard,
                offset: tensor.offset,
                buffer: vec![0u8; tensor.nbytes],
            });
            reads.push(PlannedRead { name, kind, tensor });
        }
        let pending = index
            .submit_reads(requests)
            .map_err(|error| BindError::Read(format!("layer {layer} batch"), error))?;
        Ok(PendingLayer { reads, pending })
    }

    /// Waits for every read [`Self::submit_layer`] started and stores the results,
    /// exactly as [`Self::load_layer`] would have for the same layer.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::Read`] if the proactor failed or a read did not
    /// complete in full.
    pub(crate) fn absorb_layer(
        &mut self,
        index: &SafeTensorIndex,
        pending: PendingLayer,
    ) -> Result<(), BindError> {
        let PendingLayer { reads, pending } = pending;
        let buffers = index
            .wait_reads(pending)
            .map_err(|error| BindError::Read("layer batch".to_owned(), error))?;
        for (planned, raw) in reads.into_iter().zip(buffers) {
            match planned.kind {
                TensorKind::F32 => {
                    let mut values = vec![0.0f32; planned.tensor.numel()];
                    widen_into(planned.tensor.dtype, &raw, &mut values);
                    self.f32.insert(planned.name, values);
                }
                TensorKind::Bf16 => {
                    let mut values = Vec::with_capacity(planned.tensor.numel());
                    for pair in raw.chunks_exact(2) {
                        values.push(u16::from_le_bytes([pair[0], pair[1]]));
                    }
                    self.bf16.insert(planned.name, values);
                }
            }
        }
        Ok(())
    }

    fn load_f32(&mut self, index: &SafeTensorIndex, suffix: &str) -> Result<(), BindError> {
        self.read_f32_into(index, &full_name(suffix))
    }

    fn load_bf16(&mut self, index: &SafeTensorIndex, suffix: &str) -> Result<(), BindError> {
        self.read_bf16_into(index, &full_name(suffix))
    }

    fn read_f32_into(&mut self, index: &SafeTensorIndex, name: &str) -> Result<(), BindError> {
        if self.f32.contains_key(name) {
            return Ok(());
        }
        let tensor = tensor_or_missing(index, name)?;
        let mut values = vec![0.0f32; tensor.numel()];
        index
            .read_f32(tensor, &mut values)
            .map_err(|error| BindError::Read(name.to_owned(), error))?;
        self.f32.insert(name.to_owned(), values);
        Ok(())
    }

    /// Reads raw bf16 words (never widened): the released checkpoint's own bytes,
    /// reinterpreted two bytes at a time as little-endian `u16`.
    ///
    /// Reads in bounded chunks rather than one `read_raw` call: a multi-gigabyte
    /// tensor (the embedding table and LM head are each roughly 2.35 GB on the
    /// released checkpoint) read as a single request fails with EINVAL, almost
    /// certainly a single-read size limit under two GiB somewhere in the
    /// positioned-read path. `read_f32` already avoids this the same way, through
    /// its own chunk constant.
    fn read_bf16_into(&mut self, index: &SafeTensorIndex, name: &str) -> Result<(), BindError> {
        if self.bf16.contains_key(name) {
            return Ok(());
        }
        let tensor = tensor_or_missing(index, name)?;
        let mut values = Vec::with_capacity(tensor.numel());
        let mut chunk = vec![0u8; BF16_CHUNK_BYTES];
        let mut remaining = tensor.nbytes;
        let mut relative_offset: u64 = 0;
        while remaining > 0 {
            let this_chunk = remaining.min(BF16_CHUNK_BYTES);
            let absolute_offset = tensor.offset.checked_add(relative_offset).ok_or_else(|| {
                BindError::Read(
                    name.to_owned(),
                    SafeTensorError::OffsetOverflow {
                        name: name.to_owned(),
                    },
                )
            })?;
            index
                .read_range(tensor.shard, absolute_offset, &mut chunk[..this_chunk])
                .map_err(|error| BindError::Read(name.to_owned(), error))?;
            for pair in chunk[..this_chunk].chunks_exact(2) {
                values.push(u16::from_le_bytes([pair[0], pair[1]]));
            }
            remaining -= this_chunk;
            relative_offset += u64::try_from(this_chunk).expect("chunk size fits u64");
        }
        self.bf16.insert(name.to_owned(), values);
        Ok(())
    }

    fn f32_slice(&self, name: &str) -> Result<&[f32], BindError> {
        self.f32
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| BindError::MissingTensor(name.to_owned()))
    }

    fn matrix(&self, name: &str) -> Result<Matrix<'_>, BindError> {
        self.bf16
            .get(name)
            .map(|words| Matrix::Bf16(words.as_slice()))
            .ok_or_else(|| BindError::MissingTensor(name.to_owned()))
    }

    /// Builds one decoder layer's weights from tensors already loaded by
    /// [`Self::load_layer`]. Routed experts, if any, are always
    /// [`RoutedExperts::Streamed`] -- see the module docs for why.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::MissingTensor`] if `layer` was never loaded.
    pub fn layer_weights(
        &self,
        config: &K3Config,
        layer: usize,
    ) -> Result<LayerWeights<'_>, BindError> {
        let f = |suffix: &str| self.f32_slice(&layer_name(layer, suffix));
        let m = |suffix: &str| self.matrix(&layer_name(layer, suffix));

        let attention = if config.is_mla(layer) {
            Attention::Mla(MlaWeights {
                q_a: m("self_attn.q_a_proj.weight")?,
                q_a_norm: f("self_attn.q_a_layernorm.weight")?,
                q_b: m("self_attn.q_b_proj.weight")?,
                kv_a: m("self_attn.kv_a_proj_with_mqa.weight")?,
                kv_a_norm: f("self_attn.kv_a_layernorm.weight")?,
                kv_b: m("self_attn.kv_b_proj.weight")?,
                o: m("self_attn.o_proj.weight")?,
                g: if config.mla_use_output_gate {
                    Some(m("self_attn.g_proj.weight")?)
                } else {
                    None
                },
            })
        } else {
            Attention::Kda(KdaWeights {
                q: m("self_attn.q_proj.weight")?,
                k: m("self_attn.k_proj.weight")?,
                v: m("self_attn.v_proj.weight")?,
                q_conv: f("self_attn.q_conv1d.weight")?,
                k_conv: f("self_attn.k_conv1d.weight")?,
                v_conv: f("self_attn.v_conv1d.weight")?,
                f_a: m("self_attn.f_a_proj.weight")?,
                f_b: m("self_attn.f_b_proj.weight")?,
                a_log: f("self_attn.A_log")?,
                dt_bias: f("self_attn.dt_bias")?,
                b: m("self_attn.b_proj.weight")?,
                g: m("self_attn.g_proj.weight")?,
                o_norm: f("self_attn.o_norm.weight")?,
                o: m("self_attn.o_proj.weight")?,
            })
        };

        let mlp = if config.is_dense(layer) {
            Mlp::Dense {
                gate: m("mlp.gate_proj.weight")?,
                up: m("mlp.up_proj.weight")?,
                down: m("mlp.down_proj.weight")?,
            }
        } else {
            let bias_name = layer_name(layer, "block_sparse_moe.gate.e_score_correction_bias");
            Mlp::Moe(MoeWeights {
                gate: f("block_sparse_moe.gate.weight")?,
                bias: self.f32.get(&bias_name).map(Vec::as_slice),
                down: m("block_sparse_moe.routed_expert_down_proj.weight")?,
                up: m("block_sparse_moe.routed_expert_up_proj.weight")?,
                latent_norm: f("block_sparse_moe.routed_expert_norm.weight")?,
                shared_w1: m("block_sparse_moe.shared_experts.gate_proj.weight")?,
                shared_w3: m("block_sparse_moe.shared_experts.up_proj.weight")?,
                shared_w2: m("block_sparse_moe.shared_experts.down_proj.weight")?,
                experts: RoutedExperts::Streamed,
            })
        };

        Ok(LayerWeights {
            in_norm: f("input_layernorm.weight")?,
            post_norm: f("post_attention_layernorm.weight")?,
            attn_res_norm: f("self_attention_res_norm.weight")?,
            attn_res_proj: f("self_attention_res_proj.weight")?,
            mlp_res_norm: f("mlp_res_norm.weight")?,
            mlp_res_proj: f("mlp_res_proj.weight")?,
            attention,
            mlp,
        })
    }

    /// Builds the whole model from tensors already loaded by [`Self::load_top_level`]
    /// and [`Self::load_layer`] for every layer `config` names.
    ///
    /// # Errors
    ///
    /// Returns [`BindError::LayerNotLoaded`] if any layer was never loaded, or
    /// [`BindError::MissingTensor`] if a top-level tensor was never loaded.
    pub fn model(&self, config: &K3Config) -> Result<Model<'_>, BindError> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(
                self.layer_weights(config, layer)
                    .map_err(|_| BindError::LayerNotLoaded(layer))?,
            );
        }
        let norm_name = full_name("output_attn_res_norm.weight");
        let proj_name = full_name("output_attn_res_proj.weight");
        Ok(Model {
            config: config.clone(),
            embed: self.matrix(&full_name("embed_tokens.weight"))?,
            lm_head: self.matrix("language_model.lm_head.weight")?,
            final_norm: self.f32_slice(&full_name("norm.weight"))?,
            out_res: self
                .f32
                .get(&norm_name)
                .zip(self.f32.get(&proj_name))
                .map(|(norm, proj)| (norm.as_slice(), proj.as_slice())),
            layers,
        })
    }
}

fn full_name(suffix: &str) -> String {
    format!("{PREFIX}.{suffix}")
}

fn layer_name(layer: usize, suffix: &str) -> String {
    format!("{PREFIX}.layers.{layer}.{suffix}")
}

fn tensor_or_missing<'i>(
    index: &'i SafeTensorIndex,
    name: &str,
) -> Result<&'i TensorInfo, BindError> {
    index
        .tensor(name)
        .ok_or_else(|| BindError::MissingTensor(name.to_owned()))
}
