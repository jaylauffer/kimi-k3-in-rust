//! One decoder layer and its three sub-blocks, ported from `src/core/k3_ops.c`.
//!
//! Weights are borrowed fp32 slices, which is what the tiny oracle checkpoint and the
//! op fixtures carry. The released checkpoint's bf16 trunk and streamed MXFP4 experts are
//! a later slice; they change where weight bytes come from, not the order of arithmetic
//! gated here.
//!
//! Scratch buffers are allocated per call. Allocation changes no float, and the tiny
//! model does not need the C engine's preallocated scratch layout.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use crate::{
    config::K3Config,
    ops::{
        attn_res, kda_decay_in_place, kda_step, l2norm_in_place, matmul, rmsnorm, rmsnorm_in_place,
        router, shortconv_in_place, sigmoid, situ_glu,
    },
};

/// Kimi Delta Attention weights for one layer.
#[derive(Clone, Copy, Debug)]
pub struct KdaWeights<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub q_conv: &'a [f32],
    pub k_conv: &'a [f32],
    pub v_conv: &'a [f32],
    pub f_a: &'a [f32],
    pub f_b: &'a [f32],
    /// `[heads]`, indexed per head.
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub b: &'a [f32],
    pub g: &'a [f32],
    pub o_norm: &'a [f32],
    pub o: &'a [f32],
}

/// The per-sequence memory one KDA layer carries between calls.
#[derive(Clone, Debug, PartialEq)]
pub struct KdaState {
    /// `[heads][head_dim][head_dim]` recurrent matrices.
    pub recurrent: Vec<f32>,
    /// `ShortConv` history for q, k and v, `(conv_k - 1)` inputs per channel each.
    pub conv: Vec<f32>,
}

impl KdaState {
    #[must_use]
    pub fn new(c: &K3Config) -> Self {
        let p = c.kda_num_heads * c.kda_head_dim;
        Self {
            recurrent: vec![0.0; p * c.kda_head_dim],
            conv: vec![0.0; 3 * p * (c.short_conv_kernel_size - 1)],
        }
    }

    pub fn clear(&mut self) {
        self.recurrent.fill(0.0);
        self.conv.fill(0.0);
    }

    /// A placeholder holding no memory, for moving a state out of a slot briefly.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            recurrent: Vec::new(),
            conv: Vec::new(),
        }
    }
}

/// Gated MLA weights for one layer. `g` is `None` when the output gate is disabled.
#[derive(Clone, Copy, Debug)]
pub struct MlaWeights<'a> {
    pub q_a: &'a [f32],
    pub q_a_norm: &'a [f32],
    pub q_b: &'a [f32],
    pub kv_a: &'a [f32],
    pub kv_a_norm: &'a [f32],
    pub kv_b: &'a [f32],
    pub o: &'a [f32],
    pub g: Option<&'a [f32]>,
}

/// Expanded per-head keys and values plus the shared, unrotated rope slot, for every
/// position an MLA layer has seen.
#[derive(Clone, Debug, PartialEq)]
pub struct MlaCache {
    kv: Vec<f32>,
    rope: Vec<f32>,
    capacity: usize,
}

impl MlaCache {
    #[must_use]
    pub fn new(c: &K3Config, capacity: usize) -> Self {
        let kvd = c.qk_nope_head_dim + c.v_head_dim;
        Self {
            kv: vec![0.0; capacity * c.num_attention_heads * kvd],
            rope: vec![0.0; capacity * c.qk_rope_head_dim],
            capacity,
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Stable `LatentMoE` weights with the routed experts resident and packed contiguously:
/// `w1`/`w3` are `[experts][moe_inter][latent]`, `w2` is `[experts][latent][moe_inter]`.
#[derive(Clone, Copy, Debug)]
pub struct MoeWeights<'a> {
    pub gate: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub down: &'a [f32],
    pub up: &'a [f32],
    pub latent_norm: &'a [f32],
    pub shared_w1: &'a [f32],
    pub shared_w3: &'a [f32],
    pub shared_w2: &'a [f32],
    pub w1: &'a [f32],
    pub w3: &'a [f32],
    pub w2: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub enum Attention<'a> {
    Kda(KdaWeights<'a>),
    Mla(MlaWeights<'a>),
}

#[derive(Clone, Copy, Debug)]
pub enum Mlp<'a> {
    Dense {
        gate: &'a [f32],
        up: &'a [f32],
        down: &'a [f32],
    },
    Moe(MoeWeights<'a>),
}

/// Everything one decoder layer reads.
#[derive(Clone, Copy, Debug)]
pub struct LayerWeights<'a> {
    pub in_norm: &'a [f32],
    pub post_norm: &'a [f32],
    pub attn_res_norm: &'a [f32],
    pub attn_res_proj: &'a [f32],
    pub mlp_res_norm: &'a [f32],
    pub mlp_res_proj: &'a [f32],
    pub attention: Attention<'a>,
    pub mlp: Mlp<'a>,
}

/// The attention memory a layer carries: KDA state, or an MLA KV cache.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerState {
    Kda(KdaState),
    Mla(MlaCache),
}

/// Kimi Delta Attention over `t` tokens, updating `state` in place.
pub fn kda_layer(
    out: &mut [f32],
    x: &[f32],
    w: &KdaWeights<'_>,
    c: &K3Config,
    t: usize,
    state: &mut KdaState,
) {
    let e = c.hidden_size;
    let heads = c.kda_num_heads;
    let d = c.kda_head_dim;
    let p = heads * d;
    let k = c.short_conv_kernel_size;
    let hist = k - 1;

    let mut q = vec![0.0_f32; t * p];
    let mut kk = vec![0.0_f32; t * p];
    let mut v = vec![0.0_f32; t * p];
    let mut z = vec![0.0_f32; t * p];
    let mut al = vec![0.0_f32; t * p];
    let mut bt = vec![0.0_f32; t * heads];
    let mut o = vec![0.0_f32; t * p];
    let mut gb = vec![0.0_f32; p];
    let mut wr = vec![0.0_f32; p];
    let mut fa = vec![0.0_f32; d];

    for step in 0..t {
        let xt = &x[step * e..(step + 1) * e];
        matmul(&mut q[step * p..], xt, w.q, e, p);
        matmul(&mut kk[step * p..], xt, w.k, e, p);
        matmul(&mut v[step * p..], xt, w.v, e, p);
        matmul(&mut bt[step * heads..], xt, w.b, e, heads);
        matmul(&mut fa, xt, w.f_a, e, d);
        matmul(&mut z[step * p..], &fa, w.f_b, d, p);
    }

    let (qs, rest) = state.conv.split_at_mut(p * hist);
    let (ks, vs) = rest.split_at_mut(p * hist);
    shortconv_in_place(&mut q, w.q_conv, Some(qs), p, k, t);
    shortconv_in_place(&mut kk, w.k_conv, Some(ks), p, k, t);
    shortconv_in_place(&mut v, w.v_conv, Some(vs), p, k, t);

    for step in 0..t {
        for h in 0..heads {
            let at = step * p + h * d;
            l2norm_in_place(&mut q[at..at + d], 1e-6);
            l2norm_in_place(&mut kk[at..at + d], 1e-6);
        }
    }

    for step in 0..t {
        for b in &mut bt[step * heads..(step + 1) * heads] {
            *b = sigmoid(*b);
        }
        kda_decay_in_place(
            &mut z[step * p..(step + 1) * p],
            &mut al[step * p..(step + 1) * p],
            w.a_log,
            w.dt_bias,
            heads,
            d,
            c.gate_lower_bound,
        );
    }

    let qscale = 1.0_f32 / (d as f32).sqrt();
    for h in 0..heads {
        let wh = &mut wr[h * d..(h + 1) * d];
        for step in 0..t {
            let off = step * p + h * d;
            for (wi, &qi) in wh.iter_mut().zip(&q[off..off + d]) {
                *wi = qi * qscale;
            }
            kda_step(
                &mut state.recurrent[h * d * d..(h + 1) * d * d],
                &mut o[off..off + d],
                wh,
                &kk[off..off + d],
                &v[off..off + d],
                &al[off..off + d],
                bt[step * heads + h],
                d,
                d,
            );
        }
    }

    for step in 0..t {
        let xt = &x[step * e..(step + 1) * e];
        let ot = &mut o[step * p..(step + 1) * p];
        for h in 0..heads {
            rmsnorm_in_place(&mut ot[h * d..(h + 1) * d], w.o_norm, c.rms_norm_eps);
        }
        matmul(&mut gb, xt, w.g, e, p);
        for (oi, &gi) in ot.iter_mut().zip(&gb) {
            *oi *= sigmoid(gi);
        }
        matmul(&mut out[step * e..(step + 1) * e], ot, w.o, p, e);
    }
}

/// Gated MLA (`NoPE`) over `t` new tokens at absolute positions `cached..cached + t`,
/// appending their keys and values to `cache` and attending causally over everything
/// the cache holds up to each position.
///
/// # Panics
///
/// Panics when the last position does not fit the cache. Returning without writing
/// `out` would fold the previous layer's activations into the residual.
pub fn mla(
    out: &mut [f32],
    x: &[f32],
    w: &MlaWeights<'_>,
    c: &K3Config,
    t: usize,
    cache: &mut MlaCache,
    cached: usize,
) {
    let e = c.hidden_size;
    let heads = c.num_attention_heads;
    let qn = c.qk_nope_head_dim;
    let qr = c.qk_rope_head_dim;
    let vh = c.v_head_dim;
    let qh = qn + qr;
    let kvw = c.kv_lora_rank + qr;
    let kvd = qn + vh;
    let scale = 1.0_f32 / (qh as f32).sqrt();
    let last = cached + t - 1;
    assert!(
        last < cache.capacity,
        "MLA KV cache position {last} exceeds the limit of {}",
        cache.capacity.saturating_sub(1)
    );

    let mut q = vec![0.0_f32; t * heads * qh];
    let mut ct = vec![0.0_f32; kvw];
    let mut ql = vec![0.0_f32; c.q_lora_rank];
    let mut acc = vec![0.0_f32; heads * vh];
    let mut gbuf = vec![0.0_f32; heads * vh];
    let mut sc = vec![0.0_f32; last + 1];

    for step in 0..t {
        let pos = cached + step;
        let xt = &x[step * e..(step + 1) * e];
        matmul(&mut ql, xt, w.q_a, e, c.q_lora_rank);
        rmsnorm_in_place(&mut ql, w.q_a_norm, c.rms_norm_eps);
        matmul(
            &mut q[step * heads * qh..],
            &ql,
            w.q_b,
            c.q_lora_rank,
            heads * qh,
        );

        matmul(&mut ct, xt, w.kv_a, e, kvw);
        rmsnorm_in_place(&mut ct[..c.kv_lora_rank], w.kv_a_norm, c.rms_norm_eps);
        cache.rope[pos * qr..(pos + 1) * qr].copy_from_slice(&ct[c.kv_lora_rank..]);
        matmul(
            &mut cache.kv[pos * heads * kvd..(pos + 1) * heads * kvd],
            &ct,
            w.kv_b,
            c.kv_lora_rank,
            heads * kvd,
        );
    }

    for step in 0..t {
        let pos = cached + step;
        for h in 0..heads {
            let qt = &q[(step * heads + h) * qh..(step * heads + h + 1) * qh];
            let mut m = f32::NEG_INFINITY;
            for s in 0..=pos {
                let ks = &cache.kv[(s * heads + h) * kvd..];
                let kr = &cache.rope[s * qr..(s + 1) * qr];
                let mut d = 0.0_f64;
                for i in 0..qn {
                    d += f64::from(qt[i]) * f64::from(ks[i]);
                }
                for i in 0..qr {
                    d += f64::from(qt[qn + i]) * f64::from(kr[i]);
                }
                sc[s] = d as f32 * scale;
                if sc[s] > m {
                    m = sc[s];
                }
            }
            let mut z = 0.0_f64;
            for score in &mut sc[..=pos] {
                *score = (*score - m).exp();
                z += f64::from(*score);
            }

            let o = &mut acc[h * vh..(h + 1) * vh];
            o.fill(0.0);
            for (s, &score) in sc[..=pos].iter().enumerate() {
                let pr = (f64::from(score) / z) as f32;
                let at = (s * heads + h) * kvd + qn;
                for (oj, &vj) in o.iter_mut().zip(&cache.kv[at..at + vh]) {
                    *oj += pr * vj;
                }
            }
        }

        if let Some(g) = w.g {
            matmul(&mut gbuf, &x[step * e..(step + 1) * e], g, e, heads * vh);
            for (ai, &gi) in acc.iter_mut().zip(&gbuf) {
                *ai *= 1.0 / (1.0 + (-gi).exp());
            }
        }
        matmul(&mut out[step * e..(step + 1) * e], &acc, w.o, heads * vh, e);
    }
}

/// Stable `LatentMoE` with resident experts: route on the full width, run the selected
/// experts in latent space, `RMSNorm` the aggregate, up-project, then add the shared
/// expert computed on the original input, unweighted.
pub fn moe(out: &mut [f32], x: &[f32], w: &MoeWeights<'_>, c: &K3Config, t: usize) {
    let e = c.hidden_size;
    let l = c.routed_expert_hidden_size;
    let inter = c.moe_intermediate_size;
    let si = inter * c.num_shared_experts;
    let topk = c.num_experts_per_token;
    let b1 = c.activation_situ_beta;
    let b2 = c.activation_situ_linear_beta;

    let mut idx = vec![0_usize; topk];
    let mut wt = vec![0.0_f32; topk];
    let mut z = vec![0.0_f32; l];
    let mut acc = vec![0.0_f32; l];
    let mut gu = vec![0.0_f32; 2 * inter];
    let mut act = vec![0.0_f32; inter];
    let mut edn = vec![0.0_f32; l];
    let mut sgu = vec![0.0_f32; 2 * si];
    let mut sact = vec![0.0_f32; si];
    let mut sdn = vec![0.0_f32; e];

    for step in 0..t {
        let xt = &x[step * e..(step + 1) * e];
        router(
            &mut idx,
            &mut wt,
            xt,
            w.gate,
            w.bias,
            e,
            c.num_experts,
            topk,
            c.moe_renormalize,
            c.routed_scaling_factor,
        );

        matmul(&mut z, xt, w.down, e, l);
        acc.fill(0.0);
        for (&expert, &wj) in idx.iter().zip(&wt) {
            let e13 = expert * inter * l;
            let e2 = expert * l * inter;
            let (gate, up) = gu.split_at_mut(inter);
            matmul(gate, &z, &w.w1[e13..e13 + inter * l], l, inter);
            matmul(up, &z, &w.w3[e13..e13 + inter * l], l, inter);
            situ_glu(&mut act, &gu, inter, b1, b2);
            matmul(&mut edn, &act, &w.w2[e2..e2 + l * inter], inter, l);
            for (ai, &di) in acc.iter_mut().zip(&edn) {
                *ai += wj * di;
            }
        }

        if c.latent_moe_use_norm {
            rmsnorm_in_place(&mut acc, w.latent_norm, c.rms_norm_eps);
        }
        let ot = &mut out[step * e..(step + 1) * e];
        matmul(ot, &acc, w.up, l, e);

        let (sgate, sup) = sgu.split_at_mut(si);
        matmul(sgate, xt, w.shared_w1, e, si);
        matmul(sup, xt, w.shared_w3, e, si);
        situ_glu(&mut sact, &sgu, si, b1, b2);
        matmul(&mut sdn, &sact, w.shared_w2, si, e);
        for (oi, &di) in ot.iter_mut().zip(&sdn) {
            *oi += di;
        }
    }
}

fn dense_mlp(
    out: &mut [f32],
    x: &[f32],
    weights: (&[f32], &[f32], &[f32]),
    c: &K3Config,
    t: usize,
) {
    let (gate, up, down) = weights;
    let e = c.hidden_size;
    let di = c.intermediate_size;
    let mut dgu = vec![0.0_f32; 2 * di];
    let mut sub = vec![0.0_f32; di];
    for step in 0..t {
        let xt = &x[step * e..(step + 1) * e];
        let (g, u) = dgu.split_at_mut(di);
        matmul(g, xt, gate, e, di);
        matmul(u, xt, up, e, di);
        situ_glu(
            &mut sub,
            &dgu,
            di,
            c.activation_situ_beta,
            c.activation_situ_linear_beta,
        );
        matmul(&mut out[step * e..(step + 1) * e], &sub, down, di, e);
    }
}

/// Folds an `AttnRes` norm gain and scoring projection into the one vector they act as.
fn fold(norm: &[f32], proj: &[f32]) -> Vec<f32> {
    norm.iter().zip(proj).map(|(&n, &p)| n * p).collect()
}

/// Aggregates `[blocks..., prefix]` for every token into `h`.
fn aggregate(
    h: &mut [f32],
    blocks: &[Vec<f32>],
    prefix: &[f32],
    fold: &[f32],
    e: usize,
    t: usize,
    eps: f32,
) {
    let nsrc = blocks.len() + 1;
    let mut src = vec![0.0_f32; nsrc * e];
    for step in 0..t {
        for (b, block) in blocks.iter().enumerate() {
            src[b * e..(b + 1) * e].copy_from_slice(&block[step * e..(step + 1) * e]);
        }
        src[blocks.len() * e..].copy_from_slice(&prefix[step * e..(step + 1) * e]);
        attn_res(&mut h[step * e..(step + 1) * e], &src, fold, nsrc, e, eps);
    }
}

/// One decoder layer over `t` tokens at absolute positions `cached..cached + t`,
/// reproducing `_forward_attn_residual` statement for statement.
///
/// `blocks` is the Block Attention Residual snapshot stack, each entry `[t][hidden]`.
/// On a boundary layer the running residual is pushed and then CLEARED, so it does not
/// also survive as a separate softmax source there.
///
/// # Panics
///
/// Panics when `state` does not match the layer's attention kind.
pub fn decoder_layer(
    h: &mut [f32],
    blocks: &mut Vec<Vec<f32>>,
    w: &LayerWeights<'_>,
    c: &K3Config,
    layer_idx: usize,
    t: usize,
    state: &mut LayerState,
    cached: usize,
) {
    let e = c.hidden_size;
    let eps = c.rms_norm_eps;
    let n = t * e;
    let fold_attn = fold(w.attn_res_norm, w.attn_res_proj);
    let fold_mlp = fold(w.mlp_res_norm, w.mlp_res_proj);

    let mut pref = h[..n].to_vec();
    let mut have_prefix = true;

    if !blocks.is_empty() {
        aggregate(h, blocks, &pref, &fold_attn, e, t, eps);
    }

    if layer_idx % c.attn_res_block_size == 0 {
        blocks.push(pref.clone());
        have_prefix = false;
    }

    let mut hin = vec![0.0_f32; n];
    let mut tmp = vec![0.0_f32; n];
    for step in 0..t {
        rmsnorm(
            &mut hin[step * e..(step + 1) * e],
            &h[step * e..(step + 1) * e],
            w.in_norm,
            eps,
        );
    }
    match (&w.attention, state) {
        (Attention::Kda(kw), LayerState::Kda(st)) => kda_layer(&mut tmp, &hin, kw, c, t, st),
        (Attention::Mla(mw), LayerState::Mla(cache)) => {
            mla(&mut tmp, &hin, mw, c, t, cache, cached);
        }
        _ => panic!("layer {layer_idx}: attention state does not match its weights"),
    }

    if have_prefix {
        for (pi, &ti) in pref.iter_mut().zip(&tmp) {
            *pi += ti;
        }
    } else {
        pref.copy_from_slice(&tmp);
    }

    aggregate(h, blocks, &pref, &fold_mlp, e, t, eps);

    for step in 0..t {
        rmsnorm(
            &mut hin[step * e..(step + 1) * e],
            &h[step * e..(step + 1) * e],
            w.post_norm,
            eps,
        );
    }
    match &w.mlp {
        Mlp::Moe(mw) => moe(&mut tmp, &hin, mw, c, t),
        Mlp::Dense { gate, up, down } => dense_mlp(&mut tmp, &hin, (gate, up, down), c, t),
    }

    for (pi, &ti) in pref.iter_mut().zip(&tmp) {
        *pi += ti;
    }
    h[..n].copy_from_slice(&pref);
}
