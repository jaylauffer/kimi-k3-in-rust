//! `k3 <kimi-linear-dir>`: the same CLI over a Kimi Linear checkpoint
//! (`model_type = "kimi_linear"`), detected from its `config.json`.
//!
//! The trunk (everything but the routed experts) is resident; experts stream through
//! an LRU cache bounded by `--cache-gb` (default 24 for this model). Decoding is always
//! incremental. `--accel` applies as for K3.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kimi_k3_core::{linear::LinearModel, model::argmax, tokenizer::Tokenizer};

use crate::{Args, accel, chat};

/// Default routed-expert cache when `--cache-gb` is not given: about a quarter of the
/// 48B model's 94 GB of bf16 experts, leaving room for the ~4 GB trunk on a 64 GB Mac.
const DEFAULT_CACHE_GB: f64 = 24.0;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn gib(value: f64) -> usize {
    (value * 1024.0 * 1024.0 * 1024.0) as usize
}

fn pick(logits: &[f32]) -> Result<u32, String> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite logits; refusing to emit a token".into());
    }
    u32::try_from(argmax(logits)).map_err(|e| e.to_string())
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)] // GB shown to one decimal
pub fn run(
    args: &Args,
    tokenizer: &Tokenizer,
    prompt: Option<&str>,
    cancel: &AtomicBool,
    generating: &AtomicBool,
) -> Result<(), String> {
    if args.recompute || args.layers.is_some() {
        return Err("--recompute and --layers are K3-only; run --help".into());
    }
    let cache_gb = if args.cache_gb_given {
        args.cache_gb
    } else {
        DEFAULT_CACHE_GB
    };
    eprintln!(
        "loading Kimi Linear from {} (expert cache {cache_gb} GiB)...",
        args.model_dir.display()
    );
    let start = Instant::now();
    let mut model = LinearModel::load(&args.model_dir, gib(cache_gb)).map_err(|e| e.to_string())?;
    eprintln!("  resident weights loaded in {:.1?}", start.elapsed());
    let device = accel::Device::open(args.accel)?;
    let mut session = model.session(args.max_context);
    let keep = || !cancel.load(Ordering::Relaxed);

    if args.chat {
        let format = chat::ChatFormat::kimi_linear(tokenizer, model.config.eos_token_id)?;
        println!(
            "Local Kimi Linear 48B-A3B on {}. The first reply is slower while the \
             expert cache warms.",
            match args.accel {
                accel::AccelKind::Ane => "the Apple Neural Engine",
                accel::AccelKind::Cpu => "the CPU",
            }
        );
        let mut last: Option<Vec<f32>> = None;
        return chat::run_with(
            &format,
            tokenizer,
            args.max_context,
            args.gen_tokens,
            cancel,
            generating,
            io::stdin().lock(),
            io::stdout().lock(),
            |ids| {
                // Feed only what the session has not consumed; rebuild after /undo,
                // /reset or a cancelled pass, when the history no longer extends it.
                if session.is_broken() || !ids.starts_with(session.ids()) {
                    session.reset();
                    last = None;
                }
                let new = &ids[session.ids().len()..];
                let logits = if new.is_empty() {
                    last.clone().ok_or("empty context")?
                } else {
                    model
                        .feed(&mut session, new, device.accel(), keep)
                        .map_err(|e| e.to_string())?
                };
                let token = pick(&logits)?;
                last = Some(logits);
                Ok(token)
            },
        );
    }

    let prompt = prompt.ok_or("one-shot mode needs --prompt")?;
    let mut ids = tokenizer.encode(prompt);
    if ids.is_empty() || ids.len() + args.gen_tokens > args.max_context {
        return Err("prompt is empty, or prompt plus --gen exceeds --max-context".into());
    }
    eprintln!(
        "prompt: {} tokens, generating {}...",
        ids.len(),
        args.gen_tokens
    );
    generating.store(true, Ordering::Relaxed);
    let mut decoder = loadngo_inference::Utf8Stream::default();
    print!("{}", tokenizer.decode_lossy(&ids));
    io::stdout().flush().map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut decode_s = 0.0;
    let mut generated = 0;
    for step in 0..args.gen_tokens {
        let pass = Instant::now();
        let new = &ids[session.ids().len()..];
        let logits = model
            .feed(&mut session, new, device.accel(), keep)
            .map_err(|e| format!("forward pass failed at generated token {step}: {e}"))?;
        let next = pick(&logits)?;
        let seconds = pass.elapsed().as_secs_f64();
        if step > 0 {
            decode_s += seconds;
        }
        eprintln!("\n[pass {step}: {} new tokens in {seconds:.2}s]", new.len());
        ids.push(next);
        generated += 1;
        if next == model.config.eos_token_id || tokenizer.encode("[EOS]") == [next] {
            break;
        }
        print!("{}", decoder.push(&tokenizer.decode(&[next])));
        io::stdout().flush().map_err(|e| e.to_string())?;
    }
    generating.store(false, Ordering::Relaxed);
    let s = model.expert_stats();
    eprintln!(
        "\ngenerated {generated} tokens in {:.1?}; decode {:.2} tokens/s after the prompt pass",
        started.elapsed(),
        if generated > 1 {
            f64::from(generated - 1) / decode_s
        } else {
            0.0
        }
    );
    eprintln!(
        "experts: {} hits, {} misses, {:.1} GB read in {:.1}s, {} cached",
        s.hits,
        s.misses,
        s.bytes_read as f64 / 1e9,
        s.read_s,
        s.slots
    );
    let summary = device.summary();
    if !summary.is_empty() {
        eprintln!("{summary}");
    }
    Ok(())
}
