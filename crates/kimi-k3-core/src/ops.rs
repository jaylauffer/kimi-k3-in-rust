//! Numeric kernels, ported operation for operation from `src/core/k3_ops.c`.
//!
//! FLOATING-POINT CONTRACT. Each kernel keeps the C scalar path's arithmetic exactly:
//! the same accumulator partition and reduction tree, fused products only where C calls
//! `fma`, double accumulators where C uses double, and `f32::exp`/`f32::tanh` where C
//! calls `expf`/`tanhf`. Rust never contracts `a * b + c` on its own, which matches the
//! C build's `-ffp-contract=off`. Change a summation order here and the Rust engine
//! drifts from the C oracle in the last bits, which is how a near-tied argmax flips.
//!
//! The C file's hand-unrolled loops fix those orders; they are reproduced rather than
//! simplified. Names follow the C kernels and the reference model (`q`, `k`, `v`, `z`)
//! so the two can be read side by side, and index loops stay where C indexes, which is
//! also why the f64-to-f32 casts are spelled out: each is a C `(float)` conversion.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

/// Stack bound for the [`kda_step`] temporary. K3 uses `kda_head_dim` 128.
const KDA_STEP_DV: usize = 256;

/// `1 / (1 + exp(-x))` in float, as the C `sigmoidf_`.
#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `y = w * x / sqrt(mean(x^2) + eps)`, accumulated in double with `eps` inside the root.
pub fn rmsnorm(y: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = w.len();
    let inv = rms_inverse(&x[..n], eps);
    for ((yi, &xi), &wi) in y[..n].iter_mut().zip(&x[..n]).zip(w) {
        *yi = wi * xi * inv;
    }
}

/// [`rmsnorm`] where the input and output are the same buffer, as the C engine calls it.
pub fn rmsnorm_in_place(v: &mut [f32], w: &[f32], eps: f32) {
    let n = w.len();
    let inv = rms_inverse(&v[..n], eps);
    for (vi, &wi) in v[..n].iter_mut().zip(w) {
        *vi = wi * *vi * inv;
    }
}

fn rms_inverse(x: &[f32], eps: f32) -> f32 {
    let mut ss = 0.0_f64;
    for &xi in x {
        ss += f64::from(xi) * f64::from(xi);
    }
    (1.0 / (ss / x.len() as f64 + f64::from(eps)).sqrt()) as f32
}

/// L2 normalisation with the SUM of squares and `eps` inside the root (not the mean).
pub fn l2norm_in_place(v: &mut [f32], eps: f32) {
    let mut ss = 0.0_f64;
    for &vi in v.iter() {
        ss += f64::from(vi) * f64::from(vi);
    }
    let inv = (1.0 / (ss + f64::from(eps)).sqrt()) as f32;
    for vi in v {
        *vi *= inv;
    }
}

/// SiTU-GLU over a `2 * n` input laid out as `[gate | up]`. The sigmoid sees the
/// UNCAPPED gate.
pub fn situ_glu(y: &mut [f32], x: &[f32], n: usize, b1: f32, b2: f32) {
    let (gate, up) = x[..2 * n].split_at(n);
    for ((yi, &g), &u) in y[..n].iter_mut().zip(gate).zip(up) {
        let a = b1 * (g / b1).tanh() * sigmoid(g);
        let u = b2 * (u / b2).tanh();
        *yi = a * u;
    }
}

/// Causal depthwise convolution with a fused `SiLU`, in place over `[t][channels]`.
///
/// `w` is `[channels][k]`, taps oldest to newest. `state` holds the `k - 1` previous
/// inputs per channel and is updated in place; `None` treats history as zero and leaves
/// nothing behind. In place is exact: each output overwrites the input it was computed
/// from, after that input has been read.
pub fn shortconv_in_place(
    v: &mut [f32],
    w: &[f32],
    mut state: Option<&mut [f32]>,
    channels: usize,
    k: usize,
    t: usize,
) {
    let hist = k - 1;
    let mut buf = vec![0.0_f32; hist];
    for c in 0..channels {
        match state.as_deref() {
            Some(s) => buf.copy_from_slice(&s[c * hist..(c + 1) * hist]),
            None => buf.fill(0.0),
        }
        let taps = &w[c * k..(c + 1) * k];
        for step in 0..t {
            let at = step * channels + c;
            let cur = v[at];
            let mut acc = taps[hist] * cur;
            for (&tap, &past) in taps[..hist].iter().zip(&buf) {
                acc += tap * past;
            }
            if hist > 0 {
                buf.copy_within(1.., 0);
                buf[hist - 1] = cur;
            }
            v[at] = acc * sigmoid(acc);
        }
        if let Some(s) = state.as_deref_mut() {
            s[c * hist..(c + 1) * hist].copy_from_slice(&buf);
        }
    }
}

/// KDA decay chain, in place: `z` arrives as the raw low-rank output and leaves as `g`.
///
/// `g = lb * sigmoid(exp(A_log[h]) * (z + dt_bias))`, `alpha = exp(g)`. `A_log` is
/// indexed PER HEAD.
pub fn kda_decay_in_place(
    z: &mut [f32],
    alpha: &mut [f32],
    a_log: &[f32],
    dt_bias: &[f32],
    heads: usize,
    d: usize,
    lb: f32,
) {
    for h in 0..heads {
        let a = a_log[h].exp();
        for c in 0..d {
            let i = h * d + c;
            let u = a * (z[i] + dt_bias[i]);
            let gi = lb * sigmoid(u);
            z[i] = gi;
            alpha[i] = gi.exp();
        }
    }
}

/// One KDA recurrence step for one head. `s` is `[dk][dv]`; `q` arrives pre-scaled.
/// Order is load bearing: decay, read `u = S^T k`, delta write, output from the
/// updated state.
pub fn kda_step(
    s: &mut [f32],
    o: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: f32,
    dk: usize,
    dv: usize,
) {
    for (row, &a) in s[..dk * dv].chunks_exact_mut(dv).zip(alpha) {
        for x in row {
            *x *= a;
        }
    }

    let mut stack = [0.0_f32; KDA_STEP_DV];
    let mut heap;
    let u: &mut [f32] = if dv <= KDA_STEP_DV {
        &mut stack[..dv]
    } else {
        heap = vec![0.0_f32; dv];
        &mut heap
    };
    for (row, &ki) in s.chunks_exact(dv).zip(&k[..dk]) {
        if ki == 0.0 {
            continue;
        }
        for (uj, &sj) in u.iter_mut().zip(row) {
            *uj += ki * sj;
        }
    }

    for (row, &ki) in s.chunks_exact_mut(dv).zip(&k[..dk]) {
        if ki == 0.0 {
            continue;
        }
        for ((sj, &vj), &uj) in row.iter_mut().zip(&v[..dv]).zip(u.iter()) {
            *sj += ki * beta * (vj - uj);
        }
    }

    o[..dv].fill(0.0);
    for (row, &qi) in s.chunks_exact(dv).zip(&q[..dk]) {
        if qi == 0.0 {
            continue;
        }
        for (oj, &sj) in o.iter_mut().zip(row) {
            *oj += qi * sj;
        }
    }
}

/// `y[out] = W[out][inp] . x[inp]`, row-major, no bias.
///
/// Sixteen double accumulators with fused products, reduced in the C tree
/// `((a0+a4)+(a8+a12))` per lane then `(b0+b1)+(b2+b3)`, then a fused scalar tail. The
/// NEON and AVX2 C paths are bound to this same partition.
///
/// # Panics
///
/// Panics when `w` holds fewer than `out * inp` values.
pub fn matmul(y: &mut [f32], x: &[f32], w: &[f32], inp: usize, out: usize) {
    assert!(
        w.len() >= inp * out,
        "matmul weight holds {} values, {out}x{inp} needs {}",
        w.len(),
        inp * out
    );
    let x = &x[..inp];
    for (yo, row) in y[..out].iter_mut().zip(w.chunks_exact(inp)) {
        let mut a = [0.0_f64; 16];
        let mut i = 0;
        while i + 16 <= inp {
            for (l, acc) in a.iter_mut().enumerate() {
                *acc = f64::from(row[i + l]).mul_add(f64::from(x[i + l]), *acc);
            }
            i += 16;
        }
        let b0 = (a[0] + a[4]) + (a[8] + a[12]);
        let b1 = (a[1] + a[5]) + (a[9] + a[13]);
        let b2 = (a[2] + a[6]) + (a[10] + a[14]);
        let b3 = (a[3] + a[7]) + (a[11] + a[15]);
        let mut acc = (b0 + b1) + (b2 + b3);
        for j in i..inp {
            acc = f64::from(row[j]).mul_add(f64::from(x[j]), acc);
        }
        *yo = acc as f32;
    }
}

/// One token's routing decision: selected experts in descending selection order and
/// their combining weights.
///
/// Scores are independent sigmoids of `W x`. The bias steers SELECTION only; weights are
/// gathered from the UNBIASED scores, renormalised when asked, then scaled.
pub fn router(
    idx: &mut [usize],
    wt: &mut [f32],
    x: &[f32],
    gate: &[f32],
    bias: Option<&[f32]>,
    hidden: usize,
    n_experts: usize,
    topk: usize,
    renorm: bool,
    routed_scale: f32,
) {
    let mut score = vec![0.0_f32; n_experts];
    let mut choice = vec![0.0_f32; n_experts];
    for e in 0..n_experts {
        let row = &gate[e * hidden..(e + 1) * hidden];
        let mut acc = 0.0_f64;
        for (&r, &xi) in row.iter().zip(&x[..hidden]) {
            acc += f64::from(r) * f64::from(xi);
        }
        score[e] = 1.0 / (1.0 + (-(acc as f32)).exp());
        choice[e] = score[e] + bias.map_or(0.0, |b| b[e]);
    }

    for j in 0..topk {
        let mut best = None;
        let mut bv = f32::NEG_INFINITY;
        for (e, &cv) in choice.iter().enumerate() {
            if cv > bv {
                bv = cv;
                best = Some(e);
            }
        }
        if let Some(e) = best {
            idx[j] = e;
            wt[j] = score[e];
            choice[e] = f32::NEG_INFINITY;
        } else {
            idx[j] = 0;
            wt[j] = 0.0;
        }
    }

    if renorm && topk > 1 {
        let mut s = 0.0_f64;
        for &w in &wt[..topk] {
            s += f64::from(w);
        }
        let inv = (1.0 / (s + 1e-20)) as f32;
        for w in &mut wt[..topk] {
            *w *= inv;
        }
    }
    for w in &mut wt[..topk] {
        *w *= routed_scale;
    }
}

/// Block Attention Residual aggregation over `nsrc` sources of width `n`: softmax of
/// `dot(RMSNorm(source), fold)` applied to the RAW sources.
pub fn attn_res(out: &mut [f32], src: &[f32], fold: &[f32], nsrc: usize, n: usize, eps: f32) {
    let mut score = vec![0.0_f32; nsrc];
    for (s, sc) in score.iter_mut().enumerate() {
        let v = &src[s * n..(s + 1) * n];
        let mut ss = 0.0_f64;
        for &vi in v {
            ss += f64::from(vi) * f64::from(vi);
        }
        let inv = (1.0 / (ss / n as f64 + f64::from(eps)).sqrt()) as f32;
        let mut acc = 0.0_f64;
        for (&vi, &fi) in v.iter().zip(&fold[..n]) {
            acc += f64::from(vi * inv) * f64::from(fi);
        }
        *sc = acc as f32;
    }

    let mut m = score[0];
    for &sc in &score[1..] {
        if sc > m {
            m = sc;
        }
    }
    let mut z = 0.0_f64;
    for sc in &mut score {
        *sc = (*sc - m).exp();
        z += f64::from(*sc);
    }

    out[..n].fill(0.0);
    for (s, &sc) in score.iter().enumerate() {
        let p = (f64::from(sc) / z) as f32;
        for (oi, &vi) in out[..n].iter_mut().zip(&src[s * n..(s + 1) * n]) {
            *oi += p * vi;
        }
    }
}

/// Widens one bf16 value to f32. bf16 IS the top 16 bits of an f32, so this is a pure
/// shift with no rounding, table or exponent rebias.
#[inline]
#[must_use]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits(u32::from(h) << 16)
}

/// [`matmul`] with the weight stored as bf16 and widened on read.
///
/// Same sixteen-accumulator partition, fused products and reduction tree as [`matmul`],
/// and the widen is exact, so on identical values the two agree to the bit. That is the
/// gate: any difference is a wrong stride, a dropped tail element or a mis-shifted widen.
///
/// # Panics
///
/// Panics when `w` holds fewer than `out * inp` values.
pub fn matmul_bf16(y: &mut [f32], x: &[f32], w: &[u16], inp: usize, out: usize) {
    assert!(
        w.len() >= inp * out,
        "matmul_bf16 weight holds {} values, {out}x{inp} needs {}",
        w.len(),
        inp * out
    );
    let x = &x[..inp];
    for (yo, row) in y[..out].iter_mut().zip(w.chunks_exact(inp)) {
        let mut a = [0.0_f64; 16];
        let mut i = 0;
        while i + 16 <= inp {
            for (l, acc) in a.iter_mut().enumerate() {
                *acc = f64::from(bf16_to_f32(row[i + l])).mul_add(f64::from(x[i + l]), *acc);
            }
            i += 16;
        }
        let b0 = (a[0] + a[4]) + (a[8] + a[12]);
        let b1 = (a[1] + a[5]) + (a[9] + a[13]);
        let b2 = (a[2] + a[6]) + (a[10] + a[14]);
        let b3 = (a[3] + a[7]) + (a[11] + a[15]);
        let mut acc = (b0 + b1) + (b2 + b3);
        for j in i..inp {
            acc = f64::from(bf16_to_f32(row[j])).mul_add(f64::from(x[j]), acc);
        }
        *yo = acc as f32;
    }
}

/// OCP MX E2M1, indexed by the 4-bit code; bit 3 is the sign, so code 8 is `-0.0`.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Largest MXFP4 group [`matmul_mxfp4`] accepts, the C kernel's `wf[64]` bound.
pub const MXFP4_MAX_GROUP: usize = 64;

/// An E8M0 scale byte as its power of two, `2^(b - 127)`. 255 is NaN by the OCP MX spec
/// and maps to zero, so one bad byte cannot poison a row.
///
/// Built from bits rather than `powi`, so it is exact by construction: exponent field `b`
/// with a zero mantissa is `2^(b - 127)` for `b` in 1..=254, and `2^-127` is the
/// subnormal with only bit 22 set.
#[inline]
#[must_use]
pub fn e8m0_scale(b: u8) -> f32 {
    match b {
        255 => 0.0,
        0 => f32::from_bits(1 << 22),
        _ => f32::from_bits(u32::from(b) << 23),
    }
}

/// Dequantises MXFP4 rows: `packed` is `[rows][pcols]` (two elements per byte, the LOW
/// nibble is the EVEN element), `scales` is `[rows][ceil(2*pcols/group)]`, and `out` is
/// `[rows][2*pcols]`. A table lookup times a power of two, so it is exact.
pub fn mxfp4_dequant(
    out: &mut [f32],
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    pcols: usize,
    group: usize,
) {
    let width = pcols * 2;
    let ngrp = width.div_ceil(group);
    for r in 0..rows {
        let pr = &packed[r * pcols..(r + 1) * pcols];
        let sr = &scales[r * ngrp..(r + 1) * ngrp];
        let orow = &mut out[r * width..(r + 1) * width];
        for (g, &sb) in sr.iter().enumerate() {
            let mult = e8m0_scale(sb);
            let lo = g * group;
            let hi = (lo + group).min(width);
            for i in lo..hi {
                let byte = pr[i >> 1];
                let nib = if i & 1 == 1 { byte >> 4 } else { byte & 0x0F };
                orow[i] = E2M1[usize::from(nib)] * mult;
            }
        }
    }
}

/// `y[rows] = W[rows][inp] . x[inp]` with `W` read straight out of packed MXFP4, never
/// widened to an fp32 matrix: a streamed K3 expert stays 17.55 MB instead of 132 MB.
///
/// This is the C scalar and NEON arithmetic: per group, the E2M1 values are summed
/// against `x` in eight fused double accumulators (element `i` in lane `i % 8`),
/// reduced as `((s0+s4)+(s1+s5)) + ((s2+s6)+(s3+s7))` with a fused tail, then that group
/// sum times its power-of-two scale is added to the row. A NaN scale (255) skips its
/// group. The C AVX2 path uses a different order; its contract with this one, and with
/// dequantise-then-[`matmul`], is a relative error below 1e-6, not bit identity.
///
/// # Panics
///
/// Panics when `inp` is odd (the packed stride is `inp / 2` bytes) or `group` exceeds
/// [`MXFP4_MAX_GROUP`].
pub fn matmul_mxfp4(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    inp: usize,
    rows: usize,
    group: usize,
) {
    assert!(
        inp % 2 == 0,
        "matmul_mxfp4 needs an even input width (two elements per packed byte), got {inp}"
    );
    assert!(
        group > 0 && group <= MXFP4_MAX_GROUP,
        "matmul_mxfp4 group {group} is outside 1..={MXFP4_MAX_GROUP}"
    );
    let pcols = inp / 2;
    let ngrp = inp.div_ceil(group);
    let gbyte = group / 2;
    // Widened once: float to double is exact, and x does not depend on the row.
    let xd: Vec<f64> = x[..inp].iter().map(|&v| f64::from(v)).collect();

    for (r, yr) in y[..rows].iter_mut().enumerate() {
        let pr = &packed[r * pcols..(r + 1) * pcols];
        let sr = &scales[r * ngrp..(r + 1) * ngrp];
        let mut acc = 0.0_f64;
        for (g, &sb) in sr.iter().enumerate() {
            if sb == 255 {
                continue;
            }
            let pb = &pr[g * gbyte..];
            let n = (inp - g * group).min(group);
            let xdg = &xd[g * group..g * group + n];

            let mut wf = [0.0_f32; MXFP4_MAX_GROUP];
            let half = n >> 1;
            for j in 0..half {
                let byte = pb[j];
                wf[2 * j] = E2M1[usize::from(byte & 0x0F)];
                wf[2 * j + 1] = E2M1[usize::from(byte >> 4)];
            }
            if n & 1 == 1 {
                wf[n - 1] = E2M1[usize::from(pb[half] & 0x0F)];
            }

            let mut s = [0.0_f64; 8];
            let mut i = 0;
            while i + 8 <= n {
                for (l, lane) in s.iter_mut().enumerate() {
                    *lane = f64::from(wf[i + l]).mul_add(xdg[i + l], *lane);
                }
                i += 8;
            }
            let b0 = s[0] + s[4];
            let b1 = s[1] + s[5];
            let b2 = s[2] + s[6];
            let b3 = s[3] + s[7];
            let mut sub = (b0 + b1) + (b2 + b3);
            for j in i..n {
                sub = f64::from(wf[j]).mul_add(xdg[j], sub);
            }
            acc += sub * f64::from(e8m0_scale(sb));
        }
        *yr = acc as f32;
    }
}
