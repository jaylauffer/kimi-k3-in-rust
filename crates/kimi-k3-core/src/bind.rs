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
    layer::{
        Attention, KdaWeights, LayerWeights, Matrix, MlaWeights, Mlp, MoeWeights, RoutedExperts,
    },
    model::Model,
    safetensors::{SafeTensorError, SafeTensorIndex, TensorInfo},
};

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
        let f32_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attention_res_norm.weight",
            "self_attention_res_proj.weight",
            "mlp_res_norm.weight",
            "mlp_res_proj.weight",
        ];
        for name in f32_names {
            self.load_layer_f32(index, layer, name)?;
        }

        if config.is_mla(layer) {
            self.load_layer_bf16(index, layer, "self_attn.q_a_proj.weight")?;
            self.load_layer_f32(index, layer, "self_attn.q_a_layernorm.weight")?;
            self.load_layer_bf16(index, layer, "self_attn.q_b_proj.weight")?;
            self.load_layer_bf16(index, layer, "self_attn.kv_a_proj_with_mqa.weight")?;
            self.load_layer_f32(index, layer, "self_attn.kv_a_layernorm.weight")?;
            self.load_layer_bf16(index, layer, "self_attn.kv_b_proj.weight")?;
            self.load_layer_bf16(index, layer, "self_attn.o_proj.weight")?;
            if config.mla_use_output_gate {
                self.load_layer_bf16(index, layer, "self_attn.g_proj.weight")?;
            }
        } else {
            for name in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.g_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.f_a_proj.weight",
                "self_attn.f_b_proj.weight",
                "self_attn.b_proj.weight",
            ] {
                self.load_layer_bf16(index, layer, name)?;
            }
            for name in [
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
                "self_attn.A_log",
                "self_attn.dt_bias",
                "self_attn.o_norm.weight",
            ] {
                self.load_layer_f32(index, layer, name)?;
            }
        }

        if config.is_dense(layer) {
            for name in [
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                self.load_layer_bf16(index, layer, name)?;
            }
        } else {
            self.load_layer_f32(index, layer, "block_sparse_moe.gate.weight")?;
            if index
                .tensor(&layer_name(
                    layer,
                    "block_sparse_moe.gate.e_score_correction_bias",
                ))
                .is_some()
            {
                self.load_layer_f32(
                    index,
                    layer,
                    "block_sparse_moe.gate.e_score_correction_bias",
                )?;
            }
            self.load_layer_bf16(
                index,
                layer,
                "block_sparse_moe.routed_expert_down_proj.weight",
            )?;
            self.load_layer_bf16(
                index,
                layer,
                "block_sparse_moe.routed_expert_up_proj.weight",
            )?;
            self.load_layer_f32(index, layer, "block_sparse_moe.routed_expert_norm.weight")?;
            self.load_layer_bf16(
                index,
                layer,
                "block_sparse_moe.shared_experts.gate_proj.weight",
            )?;
            self.load_layer_bf16(
                index,
                layer,
                "block_sparse_moe.shared_experts.up_proj.weight",
            )?;
            self.load_layer_bf16(
                index,
                layer,
                "block_sparse_moe.shared_experts.down_proj.weight",
            )?;
        }
        Ok(())
    }

    fn load_f32(&mut self, index: &SafeTensorIndex, suffix: &str) -> Result<(), BindError> {
        self.read_f32_into(index, &full_name(suffix))
    }

    fn load_bf16(&mut self, index: &SafeTensorIndex, suffix: &str) -> Result<(), BindError> {
        self.read_bf16_into(index, &full_name(suffix))
    }

    fn load_layer_f32(
        &mut self,
        index: &SafeTensorIndex,
        layer: usize,
        suffix: &str,
    ) -> Result<(), BindError> {
        self.read_f32_into(index, &layer_name(layer, suffix))
    }

    fn load_layer_bf16(
        &mut self,
        index: &SafeTensorIndex,
        layer: usize,
        suffix: &str,
    ) -> Result<(), BindError> {
        self.read_bf16_into(index, &layer_name(layer, suffix))
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
