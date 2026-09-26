//! One decoder layer and its three sub-blocks, ported from `src/core/k3_ops.c`.
//!
//! Every weight read only through a matmul is a [`Matrix`], tagged with the format it
//! arrived in: fp32 for the op fixtures and the tiny oracle, bf16 for the released
//! checkpoint's trunk. Weights read elementwise (norms, conv kernels, `A_log`,
//! `dt_bias`, the router gate and bias) stay fp32 slices, exactly the C engine's
//! `reqw`/`reqn` split in `src/model/k3_bind.c`. C tags a whole struct; here each matrix
//! carries its own tag, so a layer can never read one format's bytes as another's.
//!
//! Routed experts are either resident fp32 banks (the fixtures) or streamed MXFP4 from an
//! [`ExpertSource`]. A streamed expert that cannot be fetched is an error that stops the
//! token, never a silently dropped contribution.
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

use std::fmt;

use crate::{
    config::K3Config,
    expert::MXFP4_GROUP_SIZE,
    ops::{
        attn_res, bf16_to_f32, kda_decay_in_place, kda_step, l2norm_in_place, matmul, matmul_bf16,
        matmul_mxfp4, rmsnorm, rmsnorm_in_place, router, shortconv_in_place, sigmoid, situ_glu,
    },
};

/// Weight memory a device computes from in place and the CPU can still read: a weight
/// moved there ([`DenseAccel::share_words`]) costs no copy per product, and the CPU
/// reference and every fallback read the same bytes.
pub trait SharedWeight: Send + Sync {
    fn bytes(&self) -> &[u8];
    /// The bytes as little-endian 16-bit words.
    fn words(&self) -> &[u16];
}

/// A weight's shared memory, owned jointly by the model and the device.
pub type Shared = std::sync::Arc<dyn SharedWeight>;

/// An optional device for the trunk's bf16 products over several rows at once (the
/// Apple Neural Engine through loadngo's Core ML engine, on macOS). It computes in its
/// own precision, so a forward through it is not bit-identical to the CPU path, which
/// remains the reference and every fallback.
pub trait DenseAccel {
    /// `y[r][o] = W[o][i] . x[r][i]` for `rows` rows, with `W` bf16 `[out][inp]`,
    /// `x` `[rows][inp]` and `y` `[rows][out]`. Returns false, with `y` in an unspecified
    /// state, to decline; the caller then computes the product on the CPU.
    fn matmul_bf16(
        &self,
        w: &[u16],
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        inp: usize,
        out: usize,
    ) -> bool;

    /// Several products known in advance, so a device can prepare the next weight
    /// while it computes the current one. Returns false, with every `y` unspecified, to
    /// decline them all. The default runs each through [`Self::matmul_bf16`] or
    /// [`Self::matmul_mxfp4`].
    fn run_dense(&self, jobs: &mut [DenseJob<'_>]) -> bool {
        jobs.iter_mut().all(|job| {
            let (x, rows, inp) = (job.x, job.rows, job.inp);
            job.parts.iter_mut().all(|(w, out, y)| match *w {
                WeightRef::Bf16(w) => self.matmul_bf16(w, x, y, rows, inp, *out),
                WeightRef::Mxfp4 { packed, scales } => {
                    self.matmul_mxfp4(packed, scales, x, y, rows, inp, *out)
                }
            })
        })
    }

    /// The same product for a packed MXFP4 matrix with [`MXFP4_GROUP_SIZE`]-element
    /// scale groups: `packed` is `[out][inp / 2]`, `scales` `[out][inp / group]`.
    /// The default declines, keeping experts on the CPU.
    #[allow(unused_variables)]
    fn matmul_mxfp4(
        &self,
        packed: &[u8],
        scales: &[u8],
        x: &[f32],
        y: &mut [f32],
        rows: usize,
        inp: usize,
        out: usize,
    ) -> bool {
        false
    }

    /// Copies bf16 `words` into memory the device computes from in place, or `None` (the
    /// default) to leave the weight where it is.
    #[allow(unused_variables)]
    fn share_words(&self, words: &[u16]) -> Option<Shared> {
        None
    }

    /// As [`Self::share_words`] for raw bytes (MXFP4 codes and scales).
    #[allow(unused_variables)]
    fn share_bytes(&self, bytes: &[u8]) -> Option<Shared> {
        None
    }
}

/// `None` computes every product on the CPU reference kernels.
pub type Accel<'a> = Option<&'a dyn DenseAccel>;

/// A weight `[out][inp]` as stored: bf16 words, or MXFP4 with `packed`
/// `[out][inp / 2]` and `scales` `[out][inp / MXFP4_GROUP_SIZE]`.
#[derive(Clone, Copy, Debug)]
pub enum WeightRef<'a> {
    Bf16(&'a [u16]),
    Mxfp4 { packed: &'a [u8], scales: &'a [u8] },
}

/// Products that share one input `x` (`[rows][inp]`): each part is a weight
/// `[out][inp]`, its `out`, and its result `y` (`[rows][out]`).
pub struct DenseJob<'a> {
    pub x: &'a [f32],
    pub rows: usize,
    pub inp: usize,
    pub parts: Vec<(WeightRef<'a>, usize, &'a mut [f32])>,
}

/// A weight matrix read only through a matmul, in the storage format it arrived in.
#[derive(Clone, Copy, Debug)]
pub enum Matrix<'a> {
    F32(&'a [f32]),
    /// The checkpoint's own bf16 bytes, widened on read and never held at fp32.
    Bf16(&'a [u16]),
}

impl Matrix<'_> {
    /// `y[out] = W[out][inp] . x[inp]` through the kernel for this format.
    pub fn mul(&self, y: &mut [f32], x: &[f32], inp: usize, out: usize) {
        match *self {
            Self::F32(w) => matmul(y, x, w, inp, out),
            Self::Bf16(w) => matmul_bf16(y, x, w, inp, out),
        }
    }

    /// [`Self::mul`] for `rows` contiguous rows: `x` is `[rows][inp]`, `y` `[rows][out]`.
    /// A bf16 matrix goes to `accel` when one is given and accepts it; otherwise each
    /// row is exactly one [`Self::mul`], so the CPU result does not depend on batching.
    pub fn mul_rows(
        &self,
        y: &mut [f32],
        x: &[f32],
        rows: usize,
        inp: usize,
        out: usize,
        accel: Accel<'_>,
    ) {
        if let (Self::Bf16(w), Some(device)) = (*self, accel) {
            if device.matmul_bf16(w, x, y, rows, inp, out) {
                return;
            }
        }
        for r in 0..rows {
            self.mul(
                &mut y[r * out..(r + 1) * out],
                &x[r * inp..(r + 1) * inp],
                inp,
                out,
            );
        }
    }

    /// Copies row `row` of a `width`-column matrix into `dst`, widening bf16. The
    /// embedding is gathered a row at a time rather than multiplied.
    pub fn row_into(&self, dst: &mut [f32], row: usize, width: usize) {
        let at = row * width;
        match *self {
            Self::F32(w) => dst[..width].copy_from_slice(&w[at..at + width]),
            Self::Bf16(w) => {
                for (d, &h) in dst[..width].iter_mut().zip(&w[at..at + width]) {
                    *d = bf16_to_f32(h);
                }
            }
        }
    }
}

/// One packed MXFP4 matrix: `rows` rows of `columns` logical values, two per packed byte,
/// one E8M0 scale per [`MXFP4_GROUP_SIZE`] values.
#[derive(Clone, Copy, Debug)]
pub struct Mxfp4Matrix<'a> {
    pub packed: &'a [u8],
    pub scales: &'a [u8],
    pub rows: usize,
    pub columns: usize,
}

impl Mxfp4Matrix<'_> {
    /// `y[rows] = W . x[columns]`, straight out of the packed bytes.
    pub fn mul(&self, y: &mut [f32], x: &[f32]) {
        matmul_mxfp4(
            y,
            x,
            self.packed,
            self.scales,
            self.columns,
            self.rows,
            MXFP4_GROUP_SIZE,
        );
    }

    /// [`Self::mul`] for `n` contiguous rows of `x` (`[n][columns]`) into `y`
    /// (`[n][rows]`), through `accel` when given and accepted; otherwise one
    /// [`Self::mul`] per row, so the CPU result does not depend on batching.
    pub fn mul_rows(&self, y: &mut [f32], x: &[f32], n: usize, accel: Accel<'_>) {
        if let Some(device) = accel {
            if device.matmul_mxfp4(self.packed, self.scales, x, y, n, self.columns, self.rows) {
                return;
            }
        }
        for r in 0..n {
            self.mul(
                &mut y[r * self.rows..(r + 1) * self.rows],
                &x[r * self.columns..(r + 1) * self.columns],
            );
        }
    }

    fn check(&self, name: &str, rows: usize, columns: usize) -> Result<(), String> {
        let packed = rows * columns / 2;
        let scales = rows * columns.div_ceil(MXFP4_GROUP_SIZE);
        if self.rows != rows || self.columns != columns {
            return Err(format!(
                "{name} is {}x{}, the layer needs {rows}x{columns}",
                self.rows, self.columns
            ));
        }
        if self.packed.len() < packed || self.scales.len() < scales {
            return Err(format!(
                "{name} holds {} packed and {} scale bytes, {rows}x{columns} needs {packed} and {scales}",
                self.packed.len(),
                self.scales.len()
            ));
        }
        Ok(())
    }
}

/// One routed expert as the model consumes it: still MXFP4, never widened.
/// `w1` is the gate and `w3` the up projection, `[moe_inter][latent]`; `w2` is the down
/// projection, `[latent][moe_inter]`.
#[derive(Clone, Copy, Debug)]
pub struct PackedExpert<'a> {
    pub w1: Mxfp4Matrix<'a>,
    pub w2: Mxfp4Matrix<'a>,
    pub w3: Mxfp4Matrix<'a>,
}

impl PackedExpert<'_> {
    fn check(&self, latent: usize, inter: usize) -> Result<(), String> {
        self.w1.check("w1", inter, latent)?;
        self.w3.check("w3", inter, latent)?;
        self.w2.check("w2", latent, inter)
    }
}

/// A routed expert could not be supplied, so the token cannot be computed faithfully.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertFetchError {
    pub layer: usize,
    pub expert: usize,
    pub detail: String,
}

impl fmt::Display for ExpertFetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "routed expert {} of layer {} could not be used: {}",
            self.expert, self.layer, self.detail
        )
    }
}

impl std::error::Error for ExpertFetchError {}

/// Where streamed routed experts come from: a cache over the checkpoint shards in the
/// engine, an in-memory bank in tests.
pub trait ExpertSource {
    /// Brings the experts one token selected resident together, so their reads can
    /// overlap. The default does nothing; [`ExpertSource::expert`] then loads each one.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertFetchError`] when the batch cannot be loaded.
    fn prefetch(&mut self, _layer: usize, _experts: &[usize]) -> Result<(), ExpertFetchError> {
        Ok(())
    }

    /// One expert's packed matrices, valid until the source is next used.
    ///
    /// # Errors
    ///
    /// Returns [`ExpertFetchError`] when the expert cannot be supplied.
    fn expert(&mut self, layer: usize, expert: usize)
    -> Result<PackedExpert<'_>, ExpertFetchError>;
}

/// The source for a model whose routed experts are all resident. A layer asking it for a
/// streamed expert gets an error, not a zero contribution.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoStreamedExperts;

impl ExpertSource for NoStreamedExperts {
    fn expert(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<PackedExpert<'_>, ExpertFetchError> {
        Err(ExpertFetchError {
            layer,
            expert,
            detail: "the layer streams its experts but the model was given no expert source"
                .to_owned(),
        })
    }
}

/// Kimi Delta Attention weights for one layer.
#[derive(Clone, Copy, Debug)]
pub struct KdaWeights<'a> {
    pub q: Matrix<'a>,
    pub k: Matrix<'a>,
    pub v: Matrix<'a>,
    pub q_conv: &'a [f32],
    pub k_conv: &'a [f32],
    pub v_conv: &'a [f32],
    pub f_a: Matrix<'a>,
    pub f_b: Matrix<'a>,
    /// `[heads]`, indexed per head.
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub b: Matrix<'a>,
    pub g: Matrix<'a>,
    pub o_norm: &'a [f32],
    pub o: Matrix<'a>,
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
    pub q_a: Matrix<'a>,
    pub q_a_norm: &'a [f32],
    pub q_b: Matrix<'a>,
    pub kv_a: Matrix<'a>,
    pub kv_a_norm: &'a [f32],
    pub kv_b: Matrix<'a>,
    pub o: Matrix<'a>,
    pub g: Option<Matrix<'a>>,
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

/// Where a `MoE` layer's routed experts live.
#[derive(Clone, Copy, Debug)]
pub enum RoutedExperts<'a> {
    /// fp32 banks packed contiguously: `w1`/`w3` are `[experts][moe_inter][latent]`,
    /// `w2` is `[experts][latent][moe_inter]`. What the fixtures carry.
    Resident {
        w1: &'a [f32],
        w3: &'a [f32],
        w2: &'a [f32],
    },
    /// Packed MXFP4, fetched per token from the [`ExpertSource`] the model is run with.
    Streamed,
}

/// Stable `LatentMoE` weights. The router gate and bias stay fp32: the router carries its
/// own inline dot product, as in C.
#[derive(Clone, Copy, Debug)]
pub struct MoeWeights<'a> {
    pub gate: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub down: Matrix<'a>,
    pub up: Matrix<'a>,
    pub latent_norm: &'a [f32],
    pub shared_w1: Matrix<'a>,
    pub shared_w3: Matrix<'a>,
    pub shared_w2: Matrix<'a>,
    pub experts: RoutedExperts<'a>,
}

#[derive(Clone, Copy, Debug)]
pub enum Attention<'a> {
    Kda(KdaWeights<'a>),
    Mla(MlaWeights<'a>),
}

#[derive(Clone, Copy, Debug)]
pub enum Mlp<'a> {
    Dense {
        gate: Matrix<'a>,
        up: Matrix<'a>,
        down: Matrix<'a>,
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
    kda_layer_with(out, x, w, c, t, state, None);
}

/// [`kda_layer`] with each projection applied to all `t` rows at once, through `accel`
/// when given. With `None` it is float-for-float the same computation.
pub fn kda_layer_with(
    out: &mut [f32],
    x: &[f32],
    w: &KdaWeights<'_>,
    c: &K3Config,
    t: usize,
    state: &mut KdaState,
    accel: Accel<'_>,
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
    let mut gb = vec![0.0_f32; t * p];
    let mut wr = vec![0.0_f32; p];
    let mut fa = vec![0.0_f32; t * d];

    let x = &x[..t * e];
    w.q.mul_rows(&mut q, x, t, e, p, accel);
    w.k.mul_rows(&mut kk, x, t, e, p, accel);
    w.v.mul_rows(&mut v, x, t, e, p, accel);
    w.b.mul_rows(&mut bt, x, t, e, heads, accel);
    w.f_a.mul_rows(&mut fa, x, t, e, d, accel);
    w.f_b.mul_rows(&mut z, &fa, t, d, p, accel);

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

    w.g.mul_rows(&mut gb, x, t, e, p, accel);
    for step in 0..t {
        let ot = &mut o[step * p..(step + 1) * p];
        for h in 0..heads {
            rmsnorm_in_place(&mut ot[h * d..(h + 1) * d], w.o_norm, c.rms_norm_eps);
        }
        for (oi, &gi) in ot.iter_mut().zip(&gb[step * p..(step + 1) * p]) {
            *oi *= sigmoid(gi);
        }
    }
    w.o.mul_rows(out, &o, t, p, e, accel);
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
    mla_with(out, x, w, c, t, cache, cached, None);
}

/// [`mla`] with each projection applied to all `t` rows at once, through `accel` when
/// given. With `None` it is float-for-float the same computation.
///
/// # Panics
///
/// As [`mla`].
pub fn mla_with(
    out: &mut [f32],
    x: &[f32],
    w: &MlaWeights<'_>,
    c: &K3Config,
    t: usize,
    cache: &mut MlaCache,
    cached: usize,
    accel: Accel<'_>,
) {
    let e = c.hidden_size;
    let heads = c.num_attention_heads;
    let qn = c.qk_nope_head_dim;
    let qr = c.qk_rope_head_dim;
    let vh = c.v_head_dim;
    let qh = qn + qr;
    let qlr = c.q_lora_rank;
    let kvr = c.kv_lora_rank;
    let kvw = kvr + qr;
    let kvd = qn + vh;
    let scale = 1.0_f32 / (qh as f32).sqrt();
    let last = cached + t - 1;
    assert!(
        last < cache.capacity,
        "MLA KV cache position {last} exceeds the limit of {}",
        cache.capacity.saturating_sub(1)
    );

    let x = &x[..t * e];
    let mut q = vec![0.0_f32; t * heads * qh];
    let mut ql = vec![0.0_f32; t * qlr];
    let mut ct = vec![0.0_f32; t * kvw];
    let mut ckv = vec![0.0_f32; t * kvr];
    let mut acc = vec![0.0_f32; t * heads * vh];
    let mut sc = vec![0.0_f32; last + 1];

    w.q_a.mul_rows(&mut ql, x, t, e, qlr, accel);
    for row in ql.chunks_exact_mut(qlr) {
        rmsnorm_in_place(row, w.q_a_norm, c.rms_norm_eps);
    }
    w.q_b.mul_rows(&mut q, &ql, t, qlr, heads * qh, accel);

    w.kv_a.mul_rows(&mut ct, x, t, e, kvw, accel);
    for step in 0..t {
        let pos = cached + step;
        let row = &mut ct[step * kvw..(step + 1) * kvw];
        rmsnorm_in_place(&mut row[..kvr], w.kv_a_norm, c.rms_norm_eps);
        cache.rope[pos * qr..(pos + 1) * qr].copy_from_slice(&row[kvr..]);
        ckv[step * kvr..(step + 1) * kvr].copy_from_slice(&row[..kvr]);
    }
    w.kv_b.mul_rows(
        &mut cache.kv[cached * heads * kvd..(cached + t) * heads * kvd],
        &ckv,
        t,
        kvr,
        heads * kvd,
        accel,
    );

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

            let o = &mut acc[(step * heads + h) * vh..(step * heads + h + 1) * vh];
            o.fill(0.0);
            for (s, &score) in sc[..=pos].iter().enumerate() {
                let pr = (f64::from(score) / z) as f32;
                let at = (s * heads + h) * kvd + qn;
                for (oj, &vj) in o.iter_mut().zip(&cache.kv[at..at + vh]) {
                    *oj += pr * vj;
                }
            }
        }
    }

    if let Some(g) = w.g {
        let mut gbuf = vec![0.0_f32; t * heads * vh];
        g.mul_rows(&mut gbuf, x, t, e, heads * vh, accel);
        for (ai, &gi) in acc.iter_mut().zip(&gbuf) {
            *ai *= 1.0 / (1.0 + (-gi).exp());
        }
    }
    w.o.mul_rows(out, &acc, t, heads * vh, e, accel);
}

/// Stable `LatentMoE`: route on the full width, run the selected experts in latent space,
/// `RMSNorm` the aggregate, up-project, then add the shared expert computed on the
/// original input, unweighted.
///
/// Streamed experts are handed to `experts` as a batch per token before the loop, so
/// their reads can overlap, then multiplied straight out of MXFP4 in the router's order.
///
/// # Errors
///
/// Returns [`ExpertFetchError`] when a streamed expert cannot be fetched or has the wrong
/// geometry. The output is then incomplete and must not be used.
pub fn moe(
    out: &mut [f32],
    x: &[f32],
    w: &MoeWeights<'_>,
    c: &K3Config,
    t: usize,
    layer_idx: usize,
    experts: &mut dyn ExpertSource,
) -> Result<(), ExpertFetchError> {
    moe_with(out, x, w, c, t, layer_idx, experts, None)
}

/// [`moe`] with the dense projections (latent down/up and the shared expert) applied to
/// all `t` rows at once, through `accel` when given. Experts stay on the CPU. With
/// `None` it is float-for-float the same computation.
///
/// # Errors
///
/// As [`moe`].
pub fn moe_with(
    out: &mut [f32],
    x: &[f32],
    w: &MoeWeights<'_>,
    c: &K3Config,
    t: usize,
    layer_idx: usize,
    experts: &mut dyn ExpertSource,
    accel: Accel<'_>,
) -> Result<(), ExpertFetchError> {
    let e = c.hidden_size;
    let l = c.routed_expert_hidden_size;
    let inter = c.moe_intermediate_size;
    let topk = c.num_experts_per_token;
    let b1 = c.activation_situ_beta;
    let b2 = c.activation_situ_linear_beta;

    let x = &x[..t * e];
    let mut idx = vec![0_usize; topk];
    let mut wt = vec![0.0_f32; topk];
    let mut zz = vec![0.0_f32; t * l];
    let mut accs = vec![0.0_f32; t * l];
    let mut gu = vec![0.0_f32; 2 * inter];
    let mut act = vec![0.0_f32; inter];
    let mut edn = vec![0.0_f32; l];

    w.down.mul_rows(&mut zz, x, t, e, l, accel);
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

        let z = &zz[step * l..(step + 1) * l];
        let acc = &mut accs[step * l..(step + 1) * l];
        if matches!(w.experts, RoutedExperts::Streamed) {
            experts.prefetch(layer_idx, &idx)?;
        }
        for (&expert, &wj) in idx.iter().zip(&wt) {
            let (gate, up) = gu.split_at_mut(inter);
            match w.experts {
                RoutedExperts::Resident { w1, w3, w2 } => {
                    let e13 = expert * inter * l;
                    let e2 = expert * l * inter;
                    matmul(gate, z, &w1[e13..e13 + inter * l], l, inter);
                    matmul(up, z, &w3[e13..e13 + inter * l], l, inter);
                    situ_glu(&mut act, &gu, inter, b1, b2);
                    matmul(&mut edn, &act, &w2[e2..e2 + l * inter], inter, l);
                }
                RoutedExperts::Streamed => {
                    let packed = experts.expert(layer_idx, expert)?;
                    packed.check(l, inter).map_err(|detail| ExpertFetchError {
                        layer: layer_idx,
                        expert,
                        detail,
                    })?;
                    packed.w1.mul_rows(gate, z, 1, accel);
                    packed.w3.mul_rows(up, z, 1, accel);
                    situ_glu(&mut act, &gu, inter, b1, b2);
                    packed.w2.mul_rows(&mut edn, &act, 1, accel);
                }
            }
            for (ai, &di) in acc.iter_mut().zip(&edn) {
                *ai += wj * di;
            }
        }

        if c.latent_moe_use_norm {
            rmsnorm_in_place(acc, w.latent_norm, c.rms_norm_eps);
        }
    }
    moe_tail(out, x, &accs, w, c, t, accel);
    Ok(())
}

/// The `MoE` tail for `t` rows: up-project the normalized latent aggregates `accs`
/// (`[t][latent]`) into `out`, then add the unweighted shared expert of `x`.
fn moe_tail(
    out: &mut [f32],
    x: &[f32],
    accs: &[f32],
    w: &MoeWeights<'_>,
    c: &K3Config,
    t: usize,
    accel: Accel<'_>,
) {
    let e = c.hidden_size;
    let l = c.routed_expert_hidden_size;
    let si = c.moe_intermediate_size * c.num_shared_experts;
    let b1 = c.activation_situ_beta;
    let b2 = c.activation_situ_linear_beta;

    w.up.mul_rows(out, accs, t, l, e, accel);
    let mut sgate = vec![0.0_f32; t * si];
    let mut sup = vec![0.0_f32; t * si];
    w.shared_w1.mul_rows(&mut sgate, x, t, e, si, accel);
    w.shared_w3.mul_rows(&mut sup, x, t, e, si, accel);
    let mut sgu = vec![0.0_f32; 2 * si];
    let mut sact = vec![0.0_f32; t * si];
    for step in 0..t {
        sgu[..si].copy_from_slice(&sgate[step * si..(step + 1) * si]);
        sgu[si..].copy_from_slice(&sup[step * si..(step + 1) * si]);
        situ_glu(&mut sact[step * si..(step + 1) * si], &sgu, si, b1, b2);
    }
    let mut sdn = vec![0.0_f32; t * e];
    w.shared_w2.mul_rows(&mut sdn, &sact, t, si, e, accel);
    for (oi, &di) in out[..t * e].iter_mut().zip(&sdn) {
        *oi += di;
    }
}

/// Bounds the contribution buffer regardless of how long a prefill chunk runs
/// -- `CHUNK * topk * latent_width` floats, not the whole prompt at once. At
/// `topk=16`, `latent_width=3584` that's 3.7 MB per token of contribution
/// storage; a 32k-token prompt would otherwise want gigabytes of it. Matches
/// C's own `CHUNK` constant in `moe_prefill_chunk`.
const MOE_PREFILL_CHUNK: usize = 64;

/// [`moe`], but for `t > 1` streamed-expert tokens processed together (a
/// prompt prefill): each unique expert the chunk's tokens route to is fetched
/// ONCE and applied to every token that selected it, instead of once per
/// token that selected it. Ported from `k3_moe_prefill`/`moe_prefill_chunk` in
/// `src/core/k3_ops.c`.
///
/// Bit-identical to calling [`moe`] once per token: this only changes fetch
/// order and count, never the arithmetic (`tests/moe_prefill.rs` gates the
/// two paths against each other). Falls straight through to [`moe`] for a
/// single token, or when the experts are resident (nothing to dedup: a
/// resident bank has no per-fetch cost), so a caller can always use this and
/// get the per-token path exactly where batching would not help.
///
/// # Errors
///
/// Returns [`ExpertFetchError`] when a streamed expert cannot be fetched.
pub fn moe_prefill(
    out: &mut [f32],
    x: &[f32],
    w: &MoeWeights<'_>,
    c: &K3Config,
    t: usize,
    layer_idx: usize,
    experts: &mut dyn ExpertSource,
) -> Result<(), ExpertFetchError> {
    moe_prefill_with(out, x, w, c, t, layer_idx, experts, None)
}

/// [`moe_prefill`] with dense projections through `accel` when given; see [`moe_with`].
///
/// # Errors
///
/// As [`moe_prefill`].
pub fn moe_prefill_with(
    out: &mut [f32],
    x: &[f32],
    w: &MoeWeights<'_>,
    c: &K3Config,
    t: usize,
    layer_idx: usize,
    experts: &mut dyn ExpertSource,
    accel: Accel<'_>,
) -> Result<(), ExpertFetchError> {
    if t <= 1 || !matches!(w.experts, RoutedExperts::Streamed) {
        return moe_with(out, x, w, c, t, layer_idx, experts, accel);
    }
    let e = c.hidden_size;
    let mut start = 0;
    while start < t {
        let n = (t - start).min(MOE_PREFILL_CHUNK);
        let chunk_out = &mut out[start * e..(start + n) * e];
        let chunk_x = &x[start * e..(start + n) * e];
        if n == 1 {
            moe_with(chunk_out, chunk_x, w, c, 1, layer_idx, experts, accel)?;
        } else {
            moe_prefill_chunk(chunk_out, chunk_x, w, c, n, layer_idx, experts, accel)?;
        }
        start += n;
    }
    Ok(())
}

/// One prefill chunk of `t` (2..=`MOE_PREFILL_CHUNK`) tokens, batched and
/// deduplicated by unique expert. See [`moe_prefill`].
fn moe_prefill_chunk(
    out: &mut [f32],
    x: &[f32],
    w: &MoeWeights<'_>,
    c: &K3Config,
    t: usize,
    layer_idx: usize,
    experts: &mut dyn ExpertSource,
    accel: Accel<'_>,
) -> Result<(), ExpertFetchError> {
    let e = c.hidden_size;
    let l = c.routed_expert_hidden_size;
    let inter = c.moe_intermediate_size;
    let topk = c.num_experts_per_token;
    let b1 = c.activation_situ_beta;
    let b2 = c.activation_situ_linear_beta;

    // 1. Route every token and down-project it, and collect the chunk's unique
    // experts, expert-major order for step 2 below.
    let x = &x[..t * e];
    let mut ridx = vec![0_usize; t * topk];
    let mut rwt = vec![0.0_f32; t * topk];
    let mut zz = vec![0.0_f32; t * l];
    let mut seen = vec![false; c.num_experts];
    let mut uniq = Vec::with_capacity(t * topk);
    for step in 0..t {
        let xt = &x[step * e..(step + 1) * e];
        let it = &mut ridx[step * topk..(step + 1) * topk];
        let wtt = &mut rwt[step * topk..(step + 1) * topk];
        router(
            it,
            wtt,
            xt,
            w.gate,
            w.bias,
            e,
            c.num_experts,
            topk,
            c.moe_renormalize,
            c.routed_scaling_factor,
        );
        for &expert in it.iter() {
            if !seen[expert] {
                seen[expert] = true;
                uniq.push(expert);
            }
        }
    }
    w.down.mul_rows(&mut zz, x, t, e, l, accel);

    // 2. Expert-major: fetch each unique expert ONCE and apply it to every
    // (token, slot) that selected it.
    experts.prefetch(layer_idx, &uniq)?;
    let mut contrib = vec![0.0_f32; t * topk * l];
    let mut gu = vec![0.0_f32; 2 * inter];
    let mut slots = Vec::with_capacity(t);
    let mut zs = Vec::with_capacity(t * l);
    let mut gates = vec![0.0_f32; t * inter];
    let mut ups = vec![0.0_f32; t * inter];
    let mut acts = vec![0.0_f32; t * inter];
    let mut edns = vec![0.0_f32; t * l];
    for &expert_id in &uniq {
        let packed = experts.expert(layer_idx, expert_id)?;
        packed.check(l, inter).map_err(|detail| ExpertFetchError {
            layer: layer_idx,
            expert: expert_id,
            detail,
        })?;
        // Every (token, slot) that selected this expert, applied as one batch. Each
        // row is the same per-token arithmetic as before, only grouped.
        slots.clear();
        zs.clear();
        for step in 0..t {
            let it = &ridx[step * topk..(step + 1) * topk];
            for (j, &selected) in it.iter().enumerate() {
                if selected == expert_id {
                    slots.push(step * topk + j);
                    zs.extend_from_slice(&zz[step * l..(step + 1) * l]);
                }
            }
        }
        let n = slots.len();
        packed.w1.mul_rows(&mut gates[..n * inter], &zs, n, accel);
        packed.w3.mul_rows(&mut ups[..n * inter], &zs, n, accel);
        for r in 0..n {
            gu[..inter].copy_from_slice(&gates[r * inter..(r + 1) * inter]);
            gu[inter..].copy_from_slice(&ups[r * inter..(r + 1) * inter]);
            situ_glu(&mut acts[r * inter..(r + 1) * inter], &gu, inter, b1, b2);
        }
        packed
            .w2
            .mul_rows(&mut edns[..n * l], &acts[..n * inter], n, accel);
        for (r, &slot) in slots.iter().enumerate() {
            contrib[slot * l..(slot + 1) * l].copy_from_slice(&edns[r * l..(r + 1) * l]);
        }
    }

    // 3. Per token, sum contributions in the ORIGINAL top-k order, then the
    // shared-expert tail exactly as `moe` does it, so every float matches.
    let mut accs = vec![0.0_f32; t * l];
    for step in 0..t {
        let wtt = &rwt[step * topk..(step + 1) * topk];
        let acc = &mut accs[step * l..(step + 1) * l];
        for j in 0..topk {
            let wj = wtt[j];
            let cb = &contrib[(step * topk + j) * l..(step * topk + j + 1) * l];
            for (ai, &ci) in acc.iter_mut().zip(cb) {
                *ai += wj * ci;
            }
        }
        if c.latent_moe_use_norm {
            rmsnorm_in_place(acc, w.latent_norm, c.rms_norm_eps);
        }
    }
    moe_tail(out, x, &accs, w, c, t, accel);
    Ok(())
}

fn dense_mlp(
    out: &mut [f32],
    x: &[f32],
    weights: (Matrix<'_>, Matrix<'_>, Matrix<'_>),
    c: &K3Config,
    t: usize,
    accel: Accel<'_>,
) {
    let (gate, up, down) = weights;
    let e = c.hidden_size;
    let di = c.intermediate_size;
    let x = &x[..t * e];
    let mut g = vec![0.0_f32; t * di];
    let mut u = vec![0.0_f32; t * di];
    gate.mul_rows(&mut g, x, t, e, di, accel);
    up.mul_rows(&mut u, x, t, e, di, accel);
    let mut dgu = vec![0.0_f32; 2 * di];
    let mut sub = vec![0.0_f32; t * di];
    for step in 0..t {
        dgu[..di].copy_from_slice(&g[step * di..(step + 1) * di]);
        dgu[di..].copy_from_slice(&u[step * di..(step + 1) * di]);
        situ_glu(
            &mut sub[step * di..(step + 1) * di],
            &dgu,
            di,
            c.activation_situ_beta,
            c.activation_situ_linear_beta,
        );
    }
    down.mul_rows(out, &sub, t, di, e, accel);
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
/// # Errors
///
/// Returns [`ExpertFetchError`] when the layer's `MoE` cannot fetch a streamed expert; `h`
/// and `state` are then partially updated and must not be used.
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
    experts: &mut dyn ExpertSource,
) -> Result<(), ExpertFetchError> {
    decoder_layer_with(h, blocks, w, c, layer_idx, t, state, cached, experts, None)
}

/// [`decoder_layer`] with every dense projection applied to all `t` rows at once,
/// through `accel` when given. With `None` it is float-for-float the same computation.
///
/// # Errors
///
/// As [`decoder_layer`].
///
/// # Panics
///
/// As [`decoder_layer`].
pub fn decoder_layer_with(
    h: &mut [f32],
    blocks: &mut Vec<Vec<f32>>,
    w: &LayerWeights<'_>,
    c: &K3Config,
    layer_idx: usize,
    t: usize,
    state: &mut LayerState,
    cached: usize,
    experts: &mut dyn ExpertSource,
    accel: Accel<'_>,
) -> Result<(), ExpertFetchError> {
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
        (Attention::Kda(kw), LayerState::Kda(st)) => {
            kda_layer_with(&mut tmp, &hin, kw, c, t, st, accel);
        }
        (Attention::Mla(mw), LayerState::Mla(cache)) => {
            mla_with(&mut tmp, &hin, mw, c, t, cache, cached, accel);
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
        Mlp::Moe(mw) => moe_prefill_with(&mut tmp, &hin, mw, c, t, layer_idx, experts, accel)?,
        Mlp::Dense { gate, up, down } => {
            dense_mlp(&mut tmp, &hin, (*gate, *up, *down), c, t, accel);
        }
    }

    for (pi, &ti) in pref.iter_mut().zip(&tmp) {
        *pi += ti;
    }
    h[..n].copy_from_slice(&pref);
    Ok(())
}
