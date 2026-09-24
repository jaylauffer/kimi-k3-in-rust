//! Evaluating a storage format before converting a checkpoint to it.
//!
//! `--experts mxfp4` rounds every routed expert to OCP MX v1.0 MXFP4 (4-bit E2M1
//! elements, one E8M0 scale per 32) as it enters the cache and writes the result back as
//! bf16, which holds every MXFP4 value exactly. The forward pass therefore computes what
//! a converted checkpoint would, at bf16 speed and memory: this is for judging quality,
//! not for running faster.
//!
//! `--compare mxfp4|cpu` with `--prompt-file` scores one text twice in one process and
//! reports how the second run's next-token distributions differ from the first's:
//! `mxfp4` compares bf16 experts with MXFP4 experts on the `--accel` device; `cpu`
//! compares the CPU reference with the `--accel` device, both bf16, which measures how
//! far the device's own fp16 arithmetic already moves them.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use kimi_k3_core::layer::Accel;
use kimi_k3_core::linear::{ExpertTransform, LinearModel};
use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_weights::mxfp4::{BLOCK_SIZE, e2m1, e8m0, quantize_block};

use crate::thermal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFormat {
    Bf16,
    Mxfp4,
}

impl ExpertFormat {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "mxfp4" => Ok(Self::Mxfp4),
            other => Err(format!("--experts must be bf16 or mxfp4, not {other}")),
        }
    }

    pub fn transform(self) -> Option<ExpertTransform> {
        match self {
            Self::Bf16 => None,
            Self::Mxfp4 => Some(Box::new(mxfp4_round_trip)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    /// bf16 experts, then MXFP4 experts, both on the `--accel` device.
    Mxfp4,
    /// The CPU reference, then the `--accel` device, both with bf16 experts.
    Cpu,
}

impl Comparison {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "mxfp4" => Ok(Self::Mxfp4),
            "cpu" => Ok(Self::Cpu),
            other => Err(format!("--compare must be mxfp4 or cpu, not {other}")),
        }
    }
}

/// Rounds each 32-element block of every `cols`-wide row to MXFP4 and back to bf16.
pub(crate) fn mxfp4_round_trip(words: &mut [u16], cols: usize) {
    let mut block = [0.0_f32; BLOCK_SIZE];
    let mut packed = [0_u8; BLOCK_SIZE / 2];
    for row in words.chunks_exact_mut(cols) {
        for chunk in row.chunks_mut(BLOCK_SIZE) {
            let n = chunk.len();
            for (b, &w) in block.iter_mut().zip(chunk.iter()) {
                *b = f32::from_bits(u32::from(w) << 16);
            }
            let scale = e8m0(quantize_block(&block[..n], &mut packed));
            for (i, w) in chunk.iter_mut().enumerate() {
                let value = e2m1((packed[i / 2] >> (4 * (i % 2))) & 0xF) * scale;
                // An E2M1 value times a power of two has at most one mantissa bit.
                *w = (value.to_bits() >> 16) as u16;
            }
        }
    }
}

/// Per-position differences between two runs' next-token distributions.
struct Stats {
    positions: usize,
    nll: [f64; 2],
    correct: [usize; 2],
    top1_agree: usize,
    kl: Vec<f64>,
}

fn log_softmax(logits: &[f32], out: &mut [f64]) {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let sum: f64 = logits.iter().map(|&v| f64::from(v - max).exp()).sum();
    let log_sum = sum.ln();
    for (o, &v) in out.iter_mut().zip(logits) {
        *o = f64::from(v - max) - log_sum;
    }
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

#[allow(clippy::cast_precision_loss)]
fn compare_logits(ids: &[u32], a: &[f32], b: &[f32], vocab: usize) -> Stats {
    let positions = ids.len() - 1;
    let mut stats = Stats {
        positions,
        nll: [0.0; 2],
        correct: [0; 2],
        top1_agree: 0,
        kl: Vec::with_capacity(positions),
    };
    let (mut la, mut lb) = (vec![0.0_f64; vocab], vec![0.0_f64; vocab]);
    for i in 0..positions {
        log_softmax(&a[i * vocab..(i + 1) * vocab], &mut la);
        log_softmax(&b[i * vocab..(i + 1) * vocab], &mut lb);
        let target = ids[i + 1] as usize;
        let (ta, tb) = (argmax(&la), argmax(&lb));
        stats.nll[0] -= la[target];
        stats.nll[1] -= lb[target];
        stats.correct[0] += usize::from(ta == target);
        stats.correct[1] += usize::from(tb == target);
        stats.top1_agree += usize::from(ta == tb);
        stats
            .kl
            .push(la.iter().zip(&lb).map(|(&p, &q)| p.exp() * (p - q)).sum());
    }
    stats
}

#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub fn compare(
    comparison: Comparison,
    model: &mut LinearModel,
    device: Accel<'_>,
    tokenizer: &Tokenizer,
    text: &str,
    gate: &mut thermal::Gate,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let ids = tokenizer.encode(text);
    if ids.len() < 2 {
        return Err("--compare needs a --prompt-file of at least two tokens".into());
    }
    let (names, runs): ([&str; 2], [(Accel<'_>, Option<ExpertTransform>); 2]) = match comparison {
        Comparison::Mxfp4 => (
            ["bf16 experts", "mxfp4 experts"],
            [(device, None), (device, ExpertFormat::Mxfp4.transform())],
        ),
        Comparison::Cpu => (["cpu", "device"], [(None, None), (device, None)]),
    };
    let keep = || !cancel.load(std::sync::atomic::Ordering::Relaxed);
    let mut logits = Vec::with_capacity(2);
    for (name, (accel, transform)) in names.iter().zip(runs) {
        model.set_expert_transform(transform);
        gate.checkpoint(cancel)?;
        let start = Instant::now();
        let mut session = model.session(ids.len());
        logits.push(
            model
                .score(&mut session, &ids, accel, keep)
                .map_err(|e| e.to_string())?,
        );
        eprintln!(
            "  {name}: {} tokens scored in {:.1?}",
            ids.len(),
            start.elapsed()
        );
    }
    model.set_expert_transform(None);
    let vocab = model.config.vocab_size;
    let s = compare_logits(&ids, &logits[0], &logits[1], vocab);
    let n = s.positions as f64;
    let mut kl = s.kl.clone();
    kl.sort_by(f64::total_cmp);
    let pct = |percent: usize| kl[(kl.len() - 1) * percent / 100];
    let worst =
        s.kl.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i);
    let ppl = |nll: f64| (nll / n).exp();
    println!(
        "{} vs {} over {} predicted tokens:\n\
         \x20 perplexity            {:.4} -> {:.4} ({:+.2}%)\n\
         \x20 next token correct    {:.1}% -> {:.1}%\n\
         \x20 top-1 agreement       {:.1}%\n\
         \x20 KL divergence (nats)  mean {:.5}, median {:.5}, p95 {:.5}, max {:.4} at token {} ({:?})",
        names[0],
        names[1],
        s.positions,
        ppl(s.nll[0]),
        ppl(s.nll[1]),
        (ppl(s.nll[1]) / ppl(s.nll[0]) - 1.0) * 100.0,
        s.correct[0] as f64 / n * 100.0,
        s.correct[1] as f64 / n * 100.0,
        s.top1_agree as f64 / n * 100.0,
        s.kl.iter().sum::<f64>() / n,
        pct(50),
        pct(95),
        kl[kl.len() - 1],
        worst + 1,
        tokenizer.decode_lossy(&ids[worst + 1..worst + 2]),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_mxfp4_values_and_rounds_others_to_them() {
        // One row of 64: block 0 already MXFP4 at scale 2^-4, block 1 arbitrary.
        let bf16 = |v: f32| (v.to_bits() >> 16) as u16;
        let grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let mut words: Vec<u16> = (0..32).map(|i| bf16(grid[i % 8] / 16.0)).collect();
        words.extend((0..32_u16).map(|i| bf16((f32::from(i) * 0.7).sin() * 0.01)));
        let before = words.clone();
        mxfp4_round_trip(&mut words, 64);
        assert_eq!(words[..32], before[..32]);
        let mut packed = [0_u8; 16];
        for (i, &w) in words[32..].iter().enumerate() {
            let v = f32::from_bits(u32::from(w) << 16);
            // Re-quantizing a round-tripped block changes nothing.
            let mut block = [0.0_f32; 32];
            for (b, &x) in block.iter_mut().zip(&words[32..]) {
                *b = f32::from_bits(u32::from(x) << 16);
            }
            let scale = e8m0(quantize_block(&block, &mut packed));
            let again = e2m1((packed[i / 2] >> (4 * (i % 2))) & 0xF) * scale;
            assert_eq!(again.to_bits(), v.to_bits());
        }
    }

    #[test]
    fn identical_runs_compare_as_identical() {
        let ids = [1, 2, 0];
        let logits = [0.5_f32, 2.0, -1.0, 3.0, 0.0, 1.0];
        let s = compare_logits(&ids, &logits, &logits, 3);
        assert_eq!((s.positions, s.top1_agree), (2, 2));
        assert!(s.kl.iter().all(|&k| k.abs() < 1e-12));
        assert!((s.nll[0] - s.nll[1]).abs() < 1e-12);
        assert_eq!(s.correct, [1, 1]);
    }
}
