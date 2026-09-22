//! Gates for the released checkpoint's weight formats wired through the whole model: a
//! bf16 trunk and streamed MXFP4 routed experts, run on the tiny 13-layer checkpoint.
//!
//! - A trunk bound as bf16 must give logits bit-identical to the same values bound as
//!   fp32: `matmul_bf16` is exact against `matmul`, so any difference is a wiring bug
//!   (a matrix read in the wrong format, or a row gathered wrong).
//! - Experts streamed as MXFP4 must match a resident fp32 bank holding their exact
//!   dequantisation: the same argmax at every position and the per-matmul contract of
//!   the C engine carried through the stack.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

mod common;

use std::collections::HashMap;

use common::{
    PackedExperts, TinyCheckpoint, bind_tiny, expert_weight, read_json, ref_ids, tiny_config,
};
use kimi_k3_core::{
    config::K3Config,
    expert::MXFP4_GROUP_SIZE,
    layer::{
        ExpertFetchError, ExpertSource, Matrix, Mxfp4Matrix, NoStreamedExperts, PackedExpert,
        RoutedExperts,
    },
    model::argmax,
    ops::{bf16_to_f32, e8m0_scale, mxfp4_dequant},
};

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn a_bf16_trunk_gives_logits_bit_identical_to_the_same_values_at_fp32() {
    let c = tiny_config();
    let ck = TinyCheckpoint::load();
    let experts = PackedExperts::pack(&ck, &c);
    let reference = read_json("ref_k3.json");
    let ids = ref_ids(&reference, "full_ids");

    // bf16 is the top 16 bits of an f32. Truncating every tensor to bf16 and widening it
    // back gives fp32 values that bf16 represents exactly, so both bindings hold the same
    // numbers in different formats.
    let narrowed: HashMap<String, (Vec<u16>, Vec<f32>)> = ck
        .names()
        .into_iter()
        .map(|name| {
            let bf16: Vec<u16> = ck
                .w(&name)
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect();
            let widened = bf16.iter().map(|&h| bf16_to_f32(h)).collect();
            (name, (bf16, widened))
        })
        .collect();

    let as_bf16 = bind_tiny(&ck, &c, &|name| Matrix::Bf16(&narrowed[name].0), &|layer| {
        experts.resident(layer)
    });
    let as_fp32 = bind_tiny(&ck, &c, &|name| Matrix::F32(&narrowed[name].1), &|layer| {
        experts.resident(layer)
    });

    let narrow = as_bf16
        .forward(&ids, &mut NoStreamedExperts)
        .expect("resident");
    let wide = as_fp32
        .forward(&ids, &mut NoStreamedExperts)
        .expect("resident");
    assert!(
        bits_equal(&narrow, &wide),
        "bf16 trunk logits differ from fp32"
    );

    // The incremental path reads the same matrices through the KV cache and carried KDA
    // state; its last step must agree as well.
    let vocab = c.vocab_size;
    let mut session = as_bf16.session(ids.len());
    let last = as_bf16
        .feed(&mut session, &ids, &mut NoStreamedExperts)
        .expect("resident");
    assert!(bits_equal(&last, &wide[(ids.len() - 1) * vocab..]));
}

/// One matrix quantised to MXFP4 by nearest E2M1 code under a per-group power-of-two
/// scale chosen so the group's largest magnitude fits.
struct QuantMatrix {
    packed: Vec<u8>,
    scales: Vec<u8>,
    rows: usize,
    columns: usize,
}

const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

impl QuantMatrix {
    fn quantise(values: &[f32], rows: usize, columns: usize) -> Self {
        let groups = columns.div_ceil(MXFP4_GROUP_SIZE);
        let mut packed = vec![0_u8; rows * columns / 2];
        let mut scales = vec![0_u8; rows * groups];
        for r in 0..rows {
            let row = &values[r * columns..(r + 1) * columns];
            for g in 0..groups {
                let lo = g * MXFP4_GROUP_SIZE;
                let hi = (lo + MXFP4_GROUP_SIZE).min(columns);
                let max = row[lo..hi].iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                let exponent = if max > 0.0 {
                    (max / 6.0).log2().ceil() as i32
                } else {
                    0
                };
                let byte = (exponent + 127).clamp(0, 254) as u8;
                scales[r * groups + g] = byte;
                let scale = e8m0_scale(byte);
                for (i, &v) in row.iter().enumerate().take(hi).skip(lo) {
                    let q = v / scale;
                    let code = (0..8)
                        .min_by(|&a, &b| {
                            (E2M1_MAGNITUDES[a] - q.abs())
                                .abs()
                                .total_cmp(&(E2M1_MAGNITUDES[b] - q.abs()).abs())
                        })
                        .expect("eight codes") as u8;
                    let nibble = code | if q < 0.0 { 8 } else { 0 };
                    let at = r * (columns / 2) + i / 2;
                    packed[at] |= if i % 2 == 0 { nibble } else { nibble << 4 };
                }
            }
        }
        Self {
            packed,
            scales,
            rows,
            columns,
        }
    }

    fn dequantised(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.columns];
        mxfp4_dequant(
            &mut out,
            &self.packed,
            &self.scales,
            self.rows,
            self.columns / 2,
            MXFP4_GROUP_SIZE,
        );
        out
    }

    fn view(&self) -> Mxfp4Matrix<'_> {
        Mxfp4Matrix {
            packed: &self.packed,
            scales: &self.scales,
            rows: self.rows,
            columns: self.columns,
        }
    }
}

/// Every `MoE` layer's experts quantised to MXFP4, plus the fp32 banks of their exact
/// dequantisation.
struct QuantExperts {
    /// `[layer] -> [expert] -> [w1, w2, w3]`; `None` for the dense layer.
    packed: Vec<Option<Vec<[QuantMatrix; 3]>>>,
    /// `[layer] -> [w1, w3, w2]` banks, as `RoutedExperts::Resident` expects.
    banks: Vec<Option<[Vec<f32>; 3]>>,
}

impl QuantExperts {
    fn build(ck: &TinyCheckpoint, c: &K3Config) -> Self {
        let (latent, inter) = (c.routed_expert_hidden_size, c.moe_intermediate_size);
        let packed: Vec<Option<Vec<[QuantMatrix; 3]>>> = (0..c.num_hidden_layers)
            .map(|layer| {
                (!c.is_dense(layer)).then(|| {
                    (0..c.num_experts)
                        .map(|e| {
                            [
                                QuantMatrix::quantise(
                                    expert_weight(ck, layer, e, "w1"),
                                    inter,
                                    latent,
                                ),
                                QuantMatrix::quantise(
                                    expert_weight(ck, layer, e, "w2"),
                                    latent,
                                    inter,
                                ),
                                QuantMatrix::quantise(
                                    expert_weight(ck, layer, e, "w3"),
                                    inter,
                                    latent,
                                ),
                            ]
                        })
                        .collect()
                })
            })
            .collect();
        let banks = packed
            .iter()
            .map(|layer| {
                layer.as_ref().map(|experts| {
                    let bank = |which: usize| {
                        experts
                            .iter()
                            .flat_map(|m| m[which].dequantised())
                            .collect()
                    };
                    [bank(0), bank(2), bank(1)]
                })
            })
            .collect();
        Self { packed, banks }
    }

    fn resident(&self, layer: usize) -> RoutedExperts<'_> {
        let [w1, w3, w2] = self.banks[layer].as_ref().expect("a MoE layer");
        RoutedExperts::Resident { w1, w3, w2 }
    }
}

/// An in-memory [`ExpertSource`] that counts what the model asks it for.
struct CountingSource<'a> {
    experts: &'a QuantExperts,
    fetched: usize,
    prefetched: usize,
}

impl ExpertSource for CountingSource<'_> {
    fn prefetch(&mut self, _layer: usize, experts: &[usize]) -> Result<(), ExpertFetchError> {
        self.prefetched += experts.len();
        Ok(())
    }

    fn expert(
        &mut self,
        layer: usize,
        expert: usize,
    ) -> Result<PackedExpert<'_>, ExpertFetchError> {
        self.fetched += 1;
        let [w1, w2, w3] = &self.experts.packed[layer].as_ref().expect("a MoE layer")[expert];
        Ok(PackedExpert {
            w1: w1.view(),
            w2: w2.view(),
            w3: w3.view(),
        })
    }
}

#[test]
fn streamed_mxfp4_experts_match_their_dequantised_resident_bank() {
    let c = tiny_config();
    let ck = TinyCheckpoint::load();
    let quant = QuantExperts::build(&ck, &c);
    let reference = read_json("ref_k3.json");
    let ids = ref_ids(&reference, "full_ids");
    let vocab = c.vocab_size;

    let streamed = bind_tiny(&ck, &c, &|name| Matrix::F32(ck.w(name)), &|_| {
        RoutedExperts::Streamed
    });
    let resident = bind_tiny(&ck, &c, &|name| Matrix::F32(ck.w(name)), &|layer| {
        quant.resident(layer)
    });

    let mut source = CountingSource {
        experts: &quant,
        fetched: 0,
        prefetched: 0,
    };
    let got = streamed
        .forward(&ids, &mut source)
        .expect("streamed experts fetch");
    let want = resident
        .forward(&ids, &mut NoStreamedExperts)
        .expect("resident");

    // `moe_prefill` fetches each layer's UNIQUE routed experts once per prefill
    // chunk, not once per (token, slot) that selected them (`layer::moe_prefill`'s
    // whole point) -- so `fetched` is an upper bound, not an exact count, and
    // depends on how much the real router's choices overlap across tokens.
    let moe_layers = (0..c.num_hidden_layers).filter(|&l| !c.is_dense(l)).count();
    let no_dedup_upper_bound = ids.len() * c.num_experts_per_token * moe_layers;
    assert!(
        source.fetched > 0 && source.fetched <= no_dedup_upper_bound,
        "streamed path was not used for every expert (fetched {} not in (0, {}])",
        source.fetched,
        no_dedup_upper_bound
    );
    assert_eq!(
        source.fetched, source.prefetched,
        "every prefetched expert must be exactly the set later fetched, deduplicated the same way"
    );

    let got_tokens: Vec<usize> = got.chunks_exact(vocab).map(argmax).collect();
    let want_tokens: Vec<usize> = want.chunks_exact(vocab).map(argmax).collect();
    assert_eq!(got_tokens, want_tokens, "streamed experts changed a token");

    let scale = want.iter().fold(0.0_f64, |m, v| m.max(f64::from(*v).abs()));
    let worst = got.iter().zip(&want).fold(0.0_f64, |m, (a, b)| {
        m.max((f64::from(*a) - f64::from(*b)).abs())
    });
    let rel = worst / scale;
    println!(
        "streamed vs resident logits: maxrel {rel:.3e}, bit-identical: {}",
        bits_equal(&got, &want)
    );
    assert!(
        rel < 1e-5,
        "logits drift {rel:.3e} beyond the MXFP4 kernel contract"
    );

    // The streamed model's incremental path must reproduce its own full forward exactly.
    let mut session = streamed.session(ids.len());
    let mut source = CountingSource {
        experts: &quant,
        fetched: 0,
        prefetched: 0,
    };
    let np = 12;
    let mut last = streamed
        .feed(&mut session, &ids[..np], &mut source)
        .expect("prefill");
    for (step, &id) in ids.iter().enumerate().skip(np) {
        assert!(bits_equal(&last, &got[(step - 1) * vocab..step * vocab]));
        last = streamed
            .feed(&mut session, &[id], &mut source)
            .expect("decode");
    }
    assert!(bits_equal(&last, &got[(ids.len() - 1) * vocab..]));
}

#[test]
fn a_streamed_layer_without_a_source_is_an_error_not_a_zero_contribution() {
    let c = tiny_config();
    let ck = TinyCheckpoint::load();
    let streamed = bind_tiny(&ck, &c, &|name| Matrix::F32(ck.w(name)), &|_| {
        RoutedExperts::Streamed
    });
    let error = streamed
        .forward(&[1, 2, 3], &mut NoStreamedExperts)
        .expect_err("no expert source");
    assert_eq!(error.layer, 1, "the first MoE layer should report it");
}

#[test]
#[should_panic(expected = "partially updated")]
fn a_session_refuses_tokens_after_an_expert_failure() {
    let c = tiny_config();
    let ck = TinyCheckpoint::load();
    let streamed = bind_tiny(&ck, &c, &|name| Matrix::F32(ck.w(name)), &|_| {
        RoutedExperts::Streamed
    });
    let mut session = streamed.session(8);
    assert!(
        streamed
            .feed(&mut session, &[1, 2], &mut NoStreamedExperts)
            .is_err()
    );
    let _ = streamed.feed(&mut session, &[3], &mut NoStreamedExperts);
}
