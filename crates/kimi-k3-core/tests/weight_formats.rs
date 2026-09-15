//! Gates for the released checkpoint's two weight formats: the bf16 trunk and the MXFP4
//! routed experts. The Rust counterparts of `t_matmul_bf16` and `t_mxfp4` in
//! `tests/unit/test_ops.c` and of the accuracy contract in `tests/unit/test_expert.c`.
//! Comparisons here are exact on purpose: dequantisation and the bf16 widen have no
//! rounding, so any difference at all is a bug.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]

mod common;

use common::read_json;
use kimi_k3_core::ops::{
    bf16_to_f32, e8m0_scale, matmul, matmul_bf16, matmul_mxfp4, mxfp4_dequant,
};
use serde_json::Value;

/// FNV-1a over the little-endian bytes, so a C harness can print the same number.
fn bits_hash(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in values.iter().flat_map(|v| v.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// The C harness's xorshift32, so both generate the same patterns.
struct XorShift(u32);

impl XorShift {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / 8_388_608.0 - 1.0
    }
}

#[test]
fn bf16_matmul_is_bit_identical_to_the_fp32_path_on_the_same_values() {
    // Deliberately not multiples of 16 or 4, so the tail loop and a partial final
    // accumulator block both run.
    let (inp, out) = (257, 129);
    let mut rng = XorShift(0x00C0_FFEE);
    let mut wb = vec![0_u16; inp * out];
    let mut wf = vec![0.0_f32; inp * out];
    for (b, f) in wb.iter_mut().zip(&mut wf) {
        let mut h = (rng.next() >> 8) as u16;
        // Any pattern but the NaN/Inf exponent, which would make every comparison NaN.
        // Denormals and huge exponents stay in: the checkpoint has both.
        if (h >> 7) & 0xFF == 0xFF {
            h &= 0x7F7F;
        }
        *b = h;
        *f = bf16_to_f32(h);
    }
    let x: Vec<f32> = (0..inp).map(|_| rng.unit()).collect();

    let mut wide = vec![0.0; out];
    let mut narrow = vec![0.0; out];
    matmul(&mut wide, &x, &wf, inp, out);
    matmul_bf16(&mut narrow, &x, &wb, inp, out);
    println!("bf16 matmul fnv1a {:016x}", bits_hash(&narrow));
    let differing = wide
        .iter()
        .zip(&narrow)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{differing}/{out} rows differ from the fp32 path"
    );
}

#[test]
fn bf16_widen_is_the_top_sixteen_bits_of_an_f32() {
    assert_eq!(bf16_to_f32(0x3F80).to_bits(), 1.0_f32.to_bits());
    assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(bf16_to_f32(0x0001).to_bits(), 0x0001_0000);
    assert!(bf16_to_f32(0x7F80).is_infinite());
}

#[test]
fn e8m0_scales_are_exact_powers_of_two_and_nan_is_zero() {
    for b in 0..=254_u8 {
        let expected = 2.0_f64.powi(i32::from(b) - 127);
        assert_eq!(f64::from(e8m0_scale(b)), expected, "scale byte {b}");
    }
    assert_eq!(e8m0_scale(255).to_bits(), 0.0_f32.to_bits());
}

struct Mxfp4Fixture {
    rows: usize,
    pcols: usize,
    group: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
    expected: Vec<f32>,
    swapped: Option<Vec<f32>>,
}

fn numbers(value: &Value) -> Vec<f64> {
    let data = value.get("data").unwrap_or(value);
    data.as_array()
        .expect("numeric array")
        .iter()
        .map(|v| v.as_f64().expect("number"))
        .collect()
}

impl Mxfp4Fixture {
    fn load() -> Self {
        let root = read_json("mxfp4.json");
        let usize_of = |key: &str| root[key].as_u64().expect(key) as usize;
        let expected: Vec<f32> = numbers(&root["expected"])
            .into_iter()
            .map(|v| v as f32)
            .collect();
        // The fixture also records what swapping the nibble order produces; find its
        // array without depending on the key name.
        let swapped = root["expected_swapped_nibbles"]
            .as_object()
            .and_then(|object| {
                object.values().find_map(|v| {
                    let candidate = v.get("data").unwrap_or(v).as_array()?;
                    (candidate.len() == expected.len())
                        .then(|| numbers(v).into_iter().map(|x| x as f32).collect())
                })
            });
        Self {
            rows: usize_of("rows"),
            pcols: usize_of("packed_cols"),
            group: usize_of("group_size"),
            packed: numbers(&root["packed"])
                .into_iter()
                .map(|v| v as u8)
                .collect(),
            scales: numbers(&root["scales"])
                .into_iter()
                .map(|v| v as u8)
                .collect(),
            expected,
            swapped,
        }
    }
}

#[test]
fn mxfp4_dequant_is_exact_on_released_checkpoint_bytes() {
    let f = Mxfp4Fixture::load();
    let mut out = vec![0.0_f32; f.rows * f.pcols * 2];
    mxfp4_dequant(&mut out, &f.packed, &f.scales, f.rows, f.pcols, f.group);
    let differing = out
        .iter()
        .zip(&f.expected)
        .filter(|(a, b)| f64::from(**a) != f64::from(**b))
        .count();
    println!(
        "mxfp4 dequant: {} rows x {} elements, {differing} differ",
        f.rows,
        f.pcols * 2
    );
    assert_eq!(differing, 0);
    // The swapped nibble order has identical statistics and wrong positions; make sure
    // the fixture really distinguishes the two orders.
    if let Some(swapped) = &f.swapped {
        assert_ne!(&out, swapped, "dequant matches the SWAPPED nibble order");
    }
}

/// Worst `|a - b|` relative to the largest reference magnitude, as `test_expert.c`.
fn max_relative(got: &[f32], reference: &[f32]) -> f64 {
    let scale = reference
        .iter()
        .fold(0.0_f64, |m, v| m.max(f64::from(*v).abs()));
    let worst = got.iter().zip(reference).fold(0.0_f64, |m, (a, b)| {
        m.max((f64::from(*a) - f64::from(*b)).abs())
    });
    if scale > 0.0 { worst / scale } else { worst }
}

fn dequant_then_matmul(
    packed: &[u8],
    scales: &[u8],
    x: &[f32],
    inp: usize,
    rows: usize,
    group: usize,
) -> Vec<f32> {
    let mut dense = vec![0.0; rows * inp];
    mxfp4_dequant(&mut dense, packed, scales, rows, inp / 2, group);
    let mut y = vec![0.0; rows];
    matmul(&mut y, x, &dense, inp, rows);
    y
}

#[test]
fn mxfp4_matmul_agrees_with_dequantise_then_matmul_on_released_bytes() {
    let f = Mxfp4Fixture::load();
    let inp = f.pcols * 2;
    let mut rng = XorShift(0x1234_5678);
    let x: Vec<f32> = (0..inp).map(|_| rng.unit()).collect();

    let reference = dequant_then_matmul(&f.packed, &f.scales, &x, inp, f.rows, f.group);
    let mut y = vec![0.0; f.rows];
    matmul_mxfp4(&mut y, &x, &f.packed, &f.scales, inp, f.rows, f.group);
    let rel = max_relative(&y, &reference);
    println!("mxfp4 matmul fnv1a {:016x}", bits_hash(&y));
    println!("mxfp4 matmul vs dequant+matmul: maxrel {rel:.3e}");
    assert!(rel < 1e-6, "maxrel {rel:.3e} breaks the 1e-6 contract");
}

#[test]
fn mxfp4_matmul_handles_a_short_final_group_and_skips_a_nan_scale() {
    // 70 inputs at group 32 is groups of 32, 32 and 6.
    let (inp, rows, group) = (70_usize, 3_usize, 32_usize);
    let ngrp = inp.div_ceil(group);
    let mut rng = XorShift(0x0BAD_5EED);
    let mut packed: Vec<u8> = (0..rows * inp / 2)
        .map(|_| (rng.next() >> 24) as u8)
        .collect();
    let mut scales: Vec<u8> = (0..rows * ngrp)
        .map(|_| 120 + (rng.next() >> 29) as u8)
        .collect();
    scales[ngrp + 1] = 255; // row 1, middle group
    let x: Vec<f32> = (0..inp).map(|_| rng.unit()).collect();

    let reference = dequant_then_matmul(&packed, &scales, &x, inp, rows, group);
    let mut y = vec![0.0; rows];
    matmul_mxfp4(&mut y, &x, &packed, &scales, inp, rows, group);
    assert!(max_relative(&y, &reference) < 1e-6);

    // Whatever the NaN-scaled group holds must not reach the output.
    for byte in &mut packed[inp / 2 + group / 2..inp / 2 + group] {
        *byte = !*byte;
    }
    let mut y2 = vec![0.0; rows];
    matmul_mxfp4(&mut y2, &x, &packed, &scales, inp, rows, group);
    assert_eq!(y[1].to_bits(), y2[1].to_bits());
}

#[test]
#[should_panic(expected = "even input width")]
fn mxfp4_matmul_refuses_an_odd_input_width() {
    let mut y = [0.0_f32; 1];
    matmul_mxfp4(&mut y, &[0.0; 3], &[0; 2], &[127], 3, 1, 32);
}
