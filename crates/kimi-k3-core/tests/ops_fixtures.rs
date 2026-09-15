//! Per-kernel gates, the Rust counterpart of `tests/unit/test_ops.c`.
//!
//! Every fixture was generated from the pure-torch reference and carries its own
//! weights, inputs and expected outputs. Tolerances come from `ops/MANIFEST.json`, as in
//! the C harness, so the two cannot grade differently. Local names follow the fixture
//! keys and `test_ops.c` (`q`, `k`, `v`, `w`, `x`), and fixture JSON numbers are narrowed
//! to f32 exactly as the C harness's `(float)` casts do.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

mod common;

use common::{Fixture, assert_close, manifest_tolerance, tiny_config};
use kimi_k3_core::{
    config::K3Config,
    layer::{
        Attention, KdaState, KdaWeights, LayerState, LayerWeights, Matrix, MlaCache, MlaWeights,
        Mlp, MoeWeights, NoStreamedExperts, RoutedExperts, decoder_layer, kda_layer, mla, moe,
    },
    ops::{attn_res, kda_decay_in_place, kda_step, rmsnorm, router, shortconv_in_place, situ_glu},
};

#[test]
fn rmsnorm_matches_reference() {
    let f = Fixture::load("rmsnorm");
    let (w, x, want) = (f.arr("weight"), f.arr("in"), f.arr("out"));
    let n = w.len();
    let mut y = vec![0.0; x.len()];
    for (yr, xr) in y.chunks_exact_mut(n).zip(x.chunks_exact(n)) {
        rmsnorm(yr, xr, &w, f.num("eps") as f32);
    }
    assert_close("rmsnorm", &y, &want, manifest_tolerance());
}

#[test]
fn situ_glu_matches_reference_and_holds_its_cap() {
    let f = Fixture::load("situ_glu");
    let (x, want) = (f.arr("in"), f.arr("out"));
    let (din, dout) = (f.shape("in"), f.shape("out"));
    let (rows, width) = (dout[0], dout[1]);
    assert_eq!(din, vec![rows, 2 * width]);
    let (b1, b2) = (f.num("beta") as f32, f.num("linear_beta") as f32);
    let mut y = vec![0.0; want.len()];
    for (yr, xr) in y.chunks_exact_mut(width).zip(x.chunks_exact(2 * width)) {
        situ_glu(yr, xr, width, b1, b2);
    }
    assert_close("situ_glu", &y, &want, manifest_tolerance());
    let max = y.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(max <= b1 * b2 + 1e-2, "|out| = {max} breaks the b1*b2 cap");
}

#[test]
fn shortconv_matches_reference_writes_its_state_and_reads_it_back() {
    let f = Fixture::load("shortconv");
    let tol = manifest_tolerance();
    let (w, x, want) = (f.arr("weight"), f.arr("in"), f.arr("out"));
    let (k, channels) = (f.usize("kernel"), f.usize("channels"));
    let mut state = vec![0.0; channels * (k - 1)];
    let mut y = x.clone();
    shortconv_in_place(
        &mut y,
        &w,
        Some(&mut state),
        channels,
        k,
        x.len() / channels,
    );
    assert_close("shortconv", &y, &want, tol);
    assert_close("conv_state", &state, &f.arr("state_after"), tol);

    // Continuation: only a second call over in_next can see whether the carried
    // history is actually READ.
    let mut next = f.arr("in_next");
    let steps = next.len() / channels;
    shortconv_in_place(&mut next, &w, Some(&mut state), channels, k, steps);
    assert_close("conv_continuation", &next, &f.arr("out_next"), tol);
}

#[test]
fn kda_decay_matches_reference_per_head() {
    let f = Fixture::load("kda_decay");
    let (a_log, dt) = (f.arr("A_log"), f.arr("dt_bias"));
    let (want_g, want_alpha) = (f.arr("g"), f.arr("alpha"));
    let heads = a_log.len();
    let d = dt.len() / heads;
    let mut g = f.arr("z");
    let mut alpha = vec![0.0; want_alpha.len()];
    for (gt, at) in g
        .chunks_exact_mut(dt.len())
        .zip(alpha.chunks_exact_mut(dt.len()))
    {
        kda_decay_in_place(gt, at, &a_log, &dt, heads, d, f.num("lower_bound") as f32);
    }
    let worst = |got: &[f32], want: &[f32]| {
        got.iter()
            .zip(want)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0, f64::max)
    };
    let (dg, da) = (worst(&g, &want_g), worst(&alpha, &want_alpha));
    println!("kda_decay: max|dg|={dg:.3e} max|dalpha|={da:.3e}");
    assert!(dg <= 1e-5 && da <= 1e-5);
}

#[test]
fn attn_res_matches_reference() {
    let f = Fixture::load("attnres");
    let (prefix, blocks, fold) = (
        f.arr("prefix_sum"),
        f.arr("block_residual"),
        f.arr("fold_hint"),
    );
    let n = fold.len();
    let rows = prefix.len() / n;
    let nblk = blocks.len() / (rows * n);
    let mut src = vec![0.0; (nblk + 1) * n];
    let mut y = vec![0.0; rows * n];
    for row in 0..rows {
        src[..nblk * n].copy_from_slice(&blocks[row * nblk * n..(row + 1) * nblk * n]);
        src[nblk * n..].copy_from_slice(&prefix[row * n..(row + 1) * n]);
        attn_res(
            &mut y[row * n..(row + 1) * n],
            &src,
            &fold,
            nblk + 1,
            n,
            f.num("eps") as f32,
        );
    }
    assert_close("attn_res", &y, &f.arr("out"), manifest_tolerance());
}

fn check_recurrence(file: &str) {
    let f = Fixture::load(file);
    let tol = manifest_tolerance();
    let (t, heads, dk) = (f.usize("T"), f.usize("H"), f.usize("d_k"));
    let dv = f.usize("d_v");
    let (q, k, v) = (f.arr("q"), f.arr("k"), f.arr("v"));
    let (alpha, beta) = (f.arr("alpha"), f.arr("beta"));
    let qs = f.num("q_scale") as f32;
    let mut s = vec![0.0; heads * dk * dv];
    let mut out = vec![0.0; t * heads * dv];
    let mut scaled = vec![0.0; dk];
    for step in 0..t {
        for h in 0..heads {
            let off = (step * heads + h) * dk;
            for (si, &qi) in scaled.iter_mut().zip(&q[off..off + dk]) {
                *si = qi * qs;
            }
            kda_step(
                &mut s[h * dk * dv..(h + 1) * dk * dv],
                &mut out[(step * heads + h) * dv..(step * heads + h + 1) * dv],
                &scaled,
                &k[off..off + dk],
                &v[off..off + dv],
                &alpha[off..off + dk],
                beta[step * heads + h],
                dk,
                dv,
            );
        }
    }
    assert_close(&format!("{file}_out"), &out, &f.arr("out"), tol);
    assert_close(&format!("{file}_state"), &s, &f.arr("state_out"), tol);
}

#[test]
fn kda_recurrence_matches_reference_for_one_token() {
    check_recurrence("kda_recur1");
}

#[test]
fn kda_recurrence_matches_reference_over_eight_tokens() {
    check_recurrence("kda_recur8");
}

#[test]
fn router_selects_with_the_bias_and_weights_without_it() {
    let f = Fixture::load("router");
    let tol = manifest_tolerance();
    let (gate, bias, x) = (f.arr("gate_weight"), f.arr("bias"), f.arr("in"));
    let (want_idx, want_wt) = (f.arr("topk_idx"), f.arr("topk_weight"));
    let (topk, experts) = (f.usize("top_k"), f.usize("n_experts"));
    let hidden = gate.len() / experts;
    let mut idx = vec![0; topk];
    let mut wt = vec![0.0; topk];
    for (row, xr) in x.chunks_exact(hidden).enumerate() {
        router(
            &mut idx,
            &mut wt,
            xr,
            &gate,
            Some(&bias),
            hidden,
            experts,
            topk,
            true,
            1.0,
        );
        let want_row = &want_idx[row * topk..(row + 1) * topk];
        for (&e, &w) in idx.iter().zip(&wt) {
            // Reference order is unspecified (topk(sorted=False)): compare as sets.
            let at = want_row
                .iter()
                .position(|&r| r as usize == e)
                .unwrap_or_else(|| panic!("row {row}: picked expert {e}, reference did not"));
            let want = want_wt[row * topk + at];
            let ratio = (f64::from(w) - f64::from(want)).abs()
                / (tol.abs + tol.rel * f64::from(want).abs());
            assert!(ratio <= 1.0, "row {row} expert {e}: weight {w} vs {want}");
        }
    }
}

fn fixture_config(f: &Fixture) -> K3Config {
    let c = tiny_config();
    let checks: &[(&str, usize)] = &[
        ("qk_nope", c.qk_nope_head_dim),
        ("qk_rope", c.qk_rope_head_dim),
        ("kv_lora", c.kv_lora_rank),
        ("q_lora", c.q_lora_rank),
        ("v_head", c.v_head_dim),
        ("n_heads", c.num_attention_heads),
        ("latent", c.routed_expert_hidden_size),
        ("moe_inter", c.moe_intermediate_size),
        ("n_experts", c.num_experts),
        ("top_k", c.num_experts_per_token),
        ("n_shared", c.num_shared_experts),
        ("H", c.kda_num_heads),
        ("d_k", c.kda_head_dim),
        ("conv_k", c.short_conv_kernel_size),
        ("hidden", c.hidden_size),
        ("attn_res_block_size", c.attn_res_block_size),
    ];
    for &(key, expected) in checks {
        if let Some(value) = f.try_usize(key) {
            assert_eq!(
                value, expected,
                "fixture {key} disagrees with the tiny config"
            );
        }
    }
    c
}

#[test]
fn mla_matches_reference() {
    let f = Fixture::load("mla");
    let c = fixture_config(&f);
    let x = f.arr("in");
    let t = f.shape("in")[1];
    let (q_a, q_a_norm, q_b) = (
        f.arr("q_a_proj_weight"),
        f.arr("q_a_layernorm_weight"),
        f.arr("q_b_proj_weight"),
    );
    let (kv_a, kv_a_norm, kv_b) = (
        f.arr("kv_a_proj_with_mqa_weight"),
        f.arr("kv_a_layernorm_weight"),
        f.arr("kv_b_proj_weight"),
    );
    let (o, g) = (f.arr("o_proj_weight"), f.arr("g_proj_weight"));
    let w = MlaWeights {
        q_a: Matrix::F32(&q_a),
        q_a_norm: &q_a_norm,
        q_b: Matrix::F32(&q_b),
        kv_a: Matrix::F32(&kv_a),
        kv_a_norm: &kv_a_norm,
        kv_b: Matrix::F32(&kv_b),
        o: Matrix::F32(&o),
        g: Some(Matrix::F32(&g)),
    };
    let mut cache = MlaCache::new(&c, t);
    let mut y = vec![0.0; t * c.hidden_size];
    mla(&mut y, &x, &w, &c, t, &mut cache, 0);
    assert_close("mla", &y, &f.arr("out"), manifest_tolerance());
}

/// Owned copies of one fixture's `MoE` tensors, keyed by the fixture's prefix.
struct MoeTensors {
    gate: Vec<f32>,
    bias: Vec<f32>,
    down: Vec<f32>,
    up: Vec<f32>,
    norm: Vec<f32>,
    sh1: Vec<f32>,
    sh3: Vec<f32>,
    sh2: Vec<f32>,
    w1: Vec<f32>,
    w3: Vec<f32>,
    w2: Vec<f32>,
}

impl MoeTensors {
    fn load(f: &Fixture, prefix: &str, experts: usize) -> Self {
        let key = |name: &str| f.arr(&format!("{prefix}{name}"));
        let pack = |which: &str| {
            (0..experts)
                .flat_map(|e| key(&format!("experts_{e}_{which}_weight")))
                .collect()
        };
        Self {
            gate: key("gate_weight"),
            bias: key("e_score_correction_bias"),
            down: key("down_weight"),
            up: key("up_weight"),
            norm: key("norm_weight"),
            sh1: key("shared_w1_weight"),
            sh3: key("shared_w3_weight"),
            sh2: key("shared_w2_weight"),
            w1: pack("w1"),
            w3: pack("w3"),
            w2: pack("w2"),
        }
    }

    fn weights(&self) -> MoeWeights<'_> {
        MoeWeights {
            gate: &self.gate,
            bias: Some(&self.bias),
            down: Matrix::F32(&self.down),
            up: Matrix::F32(&self.up),
            latent_norm: &self.norm,
            shared_w1: Matrix::F32(&self.sh1),
            shared_w3: Matrix::F32(&self.sh3),
            shared_w2: Matrix::F32(&self.sh2),
            experts: RoutedExperts::Resident {
                w1: &self.w1,
                w3: &self.w3,
                w2: &self.w2,
            },
        }
    }
}

#[test]
fn moe_matches_reference() {
    let f = Fixture::load("moe");
    let c = fixture_config(&f);
    let x = f.arr("in");
    let t = f.shape("in")[1];
    let tensors = MoeTensors::load(&f, "", c.num_experts);
    let mut y = vec![0.0; t * c.hidden_size];
    moe(
        &mut y,
        &x,
        &tensors.weights(),
        &c,
        t,
        1,
        &mut NoStreamedExperts,
    )
    .expect("resident experts");
    assert_close("moe", &y, &f.arr("out"), manifest_tolerance());
}

/// Owned copies of one fixture's KDA tensors.
struct KdaTensors([Vec<f32>; 14]);

impl KdaTensors {
    fn load(f: &Fixture, names: [&str; 14]) -> Self {
        Self(names.map(|name| f.arr(name)))
    }

    fn weights(&self) -> KdaWeights<'_> {
        let [
            q,
            k,
            v,
            q_conv,
            k_conv,
            v_conv,
            f_a,
            f_b,
            a_log,
            dt_bias,
            b,
            g,
            o_norm,
            o,
        ] = &self.0;
        KdaWeights {
            q: Matrix::F32(q),
            k: Matrix::F32(k),
            v: Matrix::F32(v),
            q_conv,
            k_conv,
            v_conv,
            f_a: Matrix::F32(f_a),
            f_b: Matrix::F32(f_b),
            a_log,
            dt_bias,
            b: Matrix::F32(b),
            g: Matrix::F32(g),
            o_norm,
            o: Matrix::F32(o),
        }
    }
}

fn check_kda_layer(file: &str) {
    let f = Fixture::load(file);
    let c = fixture_config(&f);
    let tol = manifest_tolerance();
    let x = f.arr("in");
    let t = f.shape("in")[1];
    let tensors = KdaTensors::load(
        &f,
        [
            "q_proj", "k_proj", "v_proj", "q_conv", "k_conv", "v_conv", "f_a", "f_b", "A_log",
            "dt_bias", "b_proj", "g_proj", "o_norm", "o_proj",
        ],
    );
    let mut state = KdaState::new(&c);
    let mut y = vec![0.0; t * c.hidden_size];
    kda_layer(&mut y, &x, &tensors.weights(), &c, t, &mut state);
    assert_close(&format!("{file}_out"), &y, &f.arr("out"), tol);
    let want_state = f.arr("state_out");
    assert_close(
        &format!("{file}_state"),
        &state.recurrent[..want_state.len()],
        &want_state,
        tol,
    );
}

#[test]
fn kda_layer_matches_reference_for_one_token() {
    check_kda_layer("kda_layer1");
}

#[test]
fn kda_layer_matches_reference_over_eight_tokens() {
    check_kda_layer("kda_layer8");
}

fn check_decoder_layer(file: &str) {
    let f = Fixture::load(file);
    let c = fixture_config(&f);
    let layer_idx = f.usize("layer_idx");
    let is_mla = f.boolean("is_mla");
    assert_eq!(
        is_mla,
        c.is_mla(layer_idx),
        "fixture layer kind disagrees with the layer map"
    );
    let e = c.hidden_size;
    let x = f.arr("in");
    let t = f.shape("in")[1];
    let block_in = f.arr("block_residual_in");
    let nb_in = f.shape("block_residual_in")[1];

    let norms: Vec<Vec<f32>> = [
        "input_layernorm_weight",
        "post_attention_layernorm_weight",
        "self_attention_res_norm_weight",
        "self_attention_res_proj_weight",
        "mlp_res_norm_weight",
        "mlp_res_proj_weight",
    ]
    .iter()
    .map(|name| f.arr(name))
    .collect();
    let moe_tensors = MoeTensors::load(&f, "mlp_", c.num_experts);

    let kda_tensors;
    let mla_tensors: Vec<Vec<f32>>;
    let (attention, mut state) = if is_mla {
        mla_tensors = [
            "self_attn_q_a_proj_weight",
            "self_attn_q_a_layernorm_weight",
            "self_attn_q_b_proj_weight",
            "self_attn_kv_a_proj_with_mqa_weight",
            "self_attn_kv_a_layernorm_weight",
            "self_attn_kv_b_proj_weight",
            "self_attn_o_proj_weight",
            "self_attn_g_proj_weight",
        ]
        .iter()
        .map(|name| f.arr(name))
        .collect();
        (
            Attention::Mla(MlaWeights {
                q_a: Matrix::F32(&mla_tensors[0]),
                q_a_norm: &mla_tensors[1],
                q_b: Matrix::F32(&mla_tensors[2]),
                kv_a: Matrix::F32(&mla_tensors[3]),
                kv_a_norm: &mla_tensors[4],
                kv_b: Matrix::F32(&mla_tensors[5]),
                o: Matrix::F32(&mla_tensors[6]),
                g: Some(Matrix::F32(&mla_tensors[7])),
            }),
            LayerState::Mla(MlaCache::new(&c, t)),
        )
    } else {
        kda_tensors = KdaTensors::load(
            &f,
            [
                "self_attn_q_proj_weight",
                "self_attn_k_proj_weight",
                "self_attn_v_proj_weight",
                "self_attn_q_conv1d_weight",
                "self_attn_k_conv1d_weight",
                "self_attn_v_conv1d_weight",
                "self_attn_f_a_proj_weight",
                "self_attn_f_b_proj_weight",
                "self_attn_A_log",
                "self_attn_dt_bias",
                "self_attn_b_proj_weight",
                "self_attn_g_proj_weight",
                "self_attn_o_norm_weight",
                "self_attn_o_proj_weight",
            ],
        );
        (
            Attention::Kda(kda_tensors.weights()),
            LayerState::Kda(KdaState::new(&c)),
        )
    };

    let weights = LayerWeights {
        in_norm: &norms[0],
        post_norm: &norms[1],
        attn_res_norm: &norms[2],
        attn_res_proj: &norms[3],
        mlp_res_norm: &norms[4],
        mlp_res_proj: &norms[5],
        attention,
        mlp: Mlp::Moe(moe_tensors.weights()),
    };

    // Fixture block_residual_in is [T][blocks][hidden]; the stack is [blocks][T][hidden].
    let mut blocks: Vec<Vec<f32>> = (0..nb_in)
        .map(|b| {
            (0..t)
                .flat_map(|step| {
                    block_in[(step * nb_in + b) * e..(step * nb_in + b + 1) * e].to_vec()
                })
                .collect()
        })
        .collect();
    let mut h = x;
    decoder_layer(
        &mut h,
        &mut blocks,
        &weights,
        &c,
        layer_idx,
        t,
        &mut state,
        0,
        &mut NoStreamedExperts,
    )
    .expect("resident experts");
    println!(
        "{file}: layer {layer_idx} ({}), blocks {nb_in} -> {}",
        if is_mla { "MLA" } else { "KDA" },
        blocks.len()
    );
    assert_close(file, &h, &f.arr("out"), manifest_tolerance());
}

#[test]
fn kda_decoder_layer_matches_reference() {
    check_decoder_layer("layer_kda");
}

#[test]
fn mla_decoder_layer_on_a_block_boundary_matches_reference() {
    check_decoder_layer("layer_mla");
}
