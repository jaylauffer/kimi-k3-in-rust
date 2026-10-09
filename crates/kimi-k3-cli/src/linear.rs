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

use crate::{Args, accel, chat, convert, quality, system_one, thermal};

/// Default routed-expert cache when `--cache-gb` is not given: about a quarter of the
/// 48B model's 94 GB of bf16 experts, leaving room for the ~4 GB trunk on a 64 GB Mac.
const DEFAULT_CACHE_GB: f64 = 24.0;

/// Chat defaults when `--gen`/`--max-context` are not given. K3's 64/512 made replies
/// stop mid-sentence here; this model's context state is ~0.23 MB per token.
const CHAT_GEN: usize = 1024;
/// Kimi Linear reads long documents (the agent board alone is ~15k tokens). Only its 7
/// MLA layers grow with context (~0.23 MB per token in all), and only as it is used.
const CHAT_CONTEXT: usize = 32_768;

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

/// The engine, as the chat tells the model.
pub(crate) const fn engine_name(accel: accel::AccelKind) -> &'static str {
    match accel {
        accel::AccelKind::Ane => "the Apple Neural Engine",
        accel::AccelKind::Gpu => "the GPU",
        accel::AccelKind::Cpu => "the CPU",
    }
}

/// Notes every conversation on the shared loop opens with, beyond the instructions.
pub(crate) fn extra_notes(args: &Args) -> String {
    let mut extra = crate::chat::FRESHNESS_GUIDANCE.to_owned();
    if args.voice {
        extra.push_str("\n\n");
        extra.push_str(VOICE_NOTE);
    }
    extra
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
    if let Some(out) = &args.convert_experts {
        let mut gate = thermal::Gate::new()?;
        return convert::run(&args.model_dir, out, &mut gate, cancel);
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
    let gpu = if args.accel == accel::AccelKind::Gpu {
        Some(device.accel().ok_or("the GPU device did not open")?)
    } else {
        None
    };
    if let Some(dir) = &args.mxfp4_experts {
        if args.experts != quality::ExpertFormat::Bf16 {
            return Err("--mxfp4-experts already reads 4-bit experts; drop --experts".into());
        }
        model.use_mxfp4_experts(dir).map_err(|e| e.to_string())?;
    }
    if let Some(accel) = gpu {
        // The trunk first, before the experts are read: the heap never holds both.
        let start = Instant::now();
        let moved = model.share_weights(accel);
        #[allow(clippy::cast_precision_loss)] // shown to one decimal of a GB
        let gb = moved as f64 / 1e9;
        eprintln!(
            "  {gb:.1} GB of resident weights moved into GPU memory in {:.1?}",
            start.elapsed()
        );
    }
    if let Some(dir) = &args.mxfp4_experts {
        let start = Instant::now();
        let keep = || !cancel.load(Ordering::Relaxed);
        let read = model
            .preload_experts(keep, gpu)
            .map_err(|e| e.to_string())?;
        if read > 0 {
            eprintln!(
                "  {read} MXFP4 experts from {} resident{} in {:.1?}",
                dir.display(),
                if gpu.is_some() { " in GPU memory" } else { "" },
                start.elapsed()
            );
        } else {
            eprintln!(
                "  MXFP4 experts from {} stream through the cache (raise --cache-gb to hold all)",
                dir.display()
            );
        }
    }
    if gpu.is_some() && !model.experts_fit() {
        eprintln!("  routed experts stream from the cache and run on the Neural Engine");
    }
    let max_context = if args.max_context_given {
        args.max_context
    } else {
        CHAT_CONTEXT
    };
    let gen_tokens = if args.gen_given || !args.chat {
        args.gen_tokens
    } else {
        CHAT_GEN
    };
    let mut gate = thermal::Gate::new()?;
    if let Some(request) = &args.system_one {
        let format = chat::ChatFormat::kimi_linear(tokenizer, model.config.eos_token_id)?;
        return system_one::run(
            request,
            args.temperature,
            &mut model,
            tokenizer,
            &format,
            device.accel(),
            cancel,
        );
    }
    if let Some(comparison) = args.compare {
        let text = prompt.ok_or("--compare needs --prompt-file (or --prompt)")?;
        return quality::compare(
            comparison,
            &mut model,
            device.accel(),
            tokenizer,
            text,
            &mut gate,
            cancel,
        );
    }
    if args.experts != quality::ExpertFormat::Bf16 {
        eprintln!(
            "  routed experts rounded to {:?} as they load (quality evaluation)",
            args.experts
        );
        model.set_expert_transform(args.experts.transform());
    }
    if let Some(path) = &args.eval {
        let format = chat::ChatFormat::kimi_linear(tokenizer, model.config.eos_token_id)?;
        return crate::agent_chat::evaluate(
            &mut model,
            tokenizer,
            &format,
            device.accel(),
            cancel,
            path,
            "Kimi Linear 48B-A3B",
        );
    }
    if args.chat {
        let format = chat::ChatFormat::kimi_linear(tokenizer, model.config.eos_token_id)?;
        let accel = device.accel();
        return crate::agent_chat::run_reader(
            args,
            crate::agent_chat::Loaded {
                model: &mut model,
                about: crate::agent_chat::About {
                    description: "Kimi Linear 48B-A3B (Moonshot AI, open weights)",
                    identity: "Kimi",
                    engine: engine_name(args.accel),
                    format: &format,
                },
                accel,
            },
            tokenizer,
            extra_notes(args),
            max_context,
            gen_tokens,
            &mut gate,
            cancel,
            generating,
        );
    }
    let mut session = model.session(max_context);
    let keep = || !cancel.load(Ordering::Relaxed);

    let prompt = prompt.ok_or("one-shot mode needs --prompt")?;
    let mut ids = tokenizer.encode(prompt);
    if ids.is_empty() || ids.len() + gen_tokens > max_context {
        return Err("prompt is empty, or prompt plus --gen exceeds --max-context".into());
    }
    eprintln!("prompt: {} tokens, generating {}...", ids.len(), gen_tokens);
    generating.store(true, Ordering::Relaxed);
    let mut decoder = loadngo_inference::Utf8Stream::default();
    print!("{}", tokenizer.decode_lossy(&ids));
    io::stdout().flush().map_err(|e| e.to_string())?;
    let started = Instant::now();
    let mut decode_s = 0.0;
    let mut generated = 0;
    for step in 0..gen_tokens {
        gate.checkpoint(cancel)?;
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

/// How to answer when replies are spoken aloud (`--voice`).
pub(crate) const VOICE_NOTE: &str = "Your replies are spoken aloud by a text-to-speech voice, and Jay is \
listening, not reading. Answer in one to three short, plain sentences. No lists, headings, code \
or markdown unless he asks for them.";

/// The chat's input and output.
type ChatIo = (Box<dyn io::BufRead>, Box<dyn io::Write>);

/// The chat's input and output: the terminal, or speech with `--voice`.
pub(crate) fn chat_io(args: &Args) -> Result<ChatIo, String> {
    if !args.voice {
        return Ok((Box::new(io::stdin().lock()), Box::new(io::stdout().lock())));
    }
    #[cfg(target_os = "macos")]
    {
        let (input, output) = crate::voice::VoiceInput::start(&args.locale, io::stdout())?;
        eprintln!(
            "voice: on-device recognition ({}); say \"Kimi, ...\" to talk to her",
            args.locale
        );
        Ok((Box::new(input), Box::new(output)))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(format!(
            "--voice (locale {}) needs macOS (on-device speech recognition)",
            args.locale
        ))
    }
}
