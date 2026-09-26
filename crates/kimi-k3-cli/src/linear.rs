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
use loadngo_inference::tools::{FsTools, Toolbox};

use crate::{Args, accel, chat, convert, quality, thermal};

/// Default routed-expert cache when `--cache-gb` is not given: about a quarter of the
/// 48B model's 94 GB of bf16 experts, leaving room for the ~4 GB trunk on a 64 GB Mac.
const DEFAULT_CACHE_GB: f64 = 24.0;

/// Chat defaults when `--gen`/`--max-context` are not given. K3's 64/512 made replies
/// stop mid-sentence here; this model's context state is ~0.23 MB per token.
const CHAT_GEN: usize = 1024;
const CHAT_CONTEXT: usize = 4096;

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

/// Read-only file tools for chat: the local drive, plus a signed CAS snapshot when
/// `--cas-root` and `--cas-key` are given and it verifies. Reported on stderr.
fn toolbox(args: &Args) -> Toolbox {
    let mut tools = Toolbox::default();
    if args.no_tools {
        eprintln!("file tools: off (--no-tools)");
        return tools;
    }
    let base = args
        .fs_base
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| ".".into());
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    eprintln!(
        "file tools: local drive, read-only, relative to {}",
        base.display()
    );
    for tool in FsTools::new(base, home.as_deref()).into_tools() {
        tools.push(tool);
    }
    match (&args.cas_root, &args.cas_key) {
        (Some(root), Some(key)) => {
            let start = Instant::now();
            match data::archive_view::ArchiveView::open_newest_verified(root, key) {
                Ok(view) => {
                    eprintln!(
                        "file tools: CAS snapshot {} ({} files), root {}, signed by {}, verified in {:.1?}",
                        view.archive_id(),
                        view.file_count(),
                        view.root().to_hex(),
                        view.signer(),
                        start.elapsed()
                    );
                    for tool in loadngo_inference::cas_tools::cas_tools(view) {
                        tools.push(tool);
                    }
                }
                Err(error) => eprintln!("file tools: no CAS snapshot ({error:#})"),
            }
        }
        (None, None) => {}
        _ => eprintln!("file tools: --cas-root and --cas-key go together; CAS tools off"),
    }
    tools
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
    if let Some(dir) = &args.mxfp4_experts {
        if args.experts != quality::ExpertFormat::Bf16 {
            return Err("--mxfp4-experts already reads 4-bit experts; drop --experts".into());
        }
        model.use_mxfp4_experts(dir).map_err(|e| e.to_string())?;
        let start = Instant::now();
        let keep = || !cancel.load(Ordering::Relaxed);
        let read = model.preload_experts(keep).map_err(|e| e.to_string())?;
        if read > 0 {
            eprintln!(
                "  {read} MXFP4 experts from {} resident in {:.1?}",
                dir.display(),
                start.elapsed()
            );
        } else {
            eprintln!(
                "  MXFP4 experts from {} stream through the cache (raise --cache-gb to hold all)",
                dir.display()
            );
        }
    }
    let device = accel::Device::open(args.accel)?;
    if args.accel == accel::AccelKind::Gpu {
        let start = Instant::now();
        let accel = device.accel().ok_or("the GPU device did not open")?;
        let moved = model.share_weights(accel);
        #[allow(clippy::cast_precision_loss)] // shown to one decimal of a GB
        let gb = moved as f64 / 1e9;
        eprintln!(
            "  {gb:.1} GB of weights moved into GPU memory in {:.1?}{}",
            start.elapsed(),
            if model.experts_fit() {
                ""
            } else {
                " (routed experts stream from the cache and run on the Neural Engine)"
            }
        );
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
    let mut session = model.session(max_context);
    let keep = || !cancel.load(Ordering::Relaxed);

    if args.chat {
        let format = chat::ChatFormat::kimi_linear(tokenizer, model.config.eos_token_id)?;
        let tools = toolbox(args);
        let tools = Some(&tools).filter(|t| !t.is_empty());
        // Every conversation opens with the same tool declarations (~800 tokens, about a
        // minute on a cold expert cache). Consume them once now and keep a snapshot, so
        // the first question and every /reset start from it instead.
        let preamble = format.preamble(tokenizer, tools);
        let mut opening: Option<kimi_k3_core::linear::LinearSession> = None;
        if !preamble.is_empty() {
            eprintln!(
                "reading the tool declarations once ({} tokens; Ctrl-C quits)...",
                preamble.len()
            );
            let start = Instant::now();
            gate.checkpoint(cancel)?;
            model
                .feed(&mut session, &preamble, device.accel(), keep)
                .map_err(|e| e.to_string())?;
            opening = Some(session.clone());
            eprintln!("  ready in {:.1?}", start.elapsed());
        }
        println!(
            "Local Kimi Linear 48B-A3B on {}. The first reply is slower while the \
             expert cache warms.",
            match args.accel {
                accel::AccelKind::Ane => "the Apple Neural Engine",
                accel::AccelKind::Gpu => "the GPU (prompts on the Apple Neural Engine)",
                accel::AccelKind::Cpu => "the CPU",
            }
        );
        let mut last: Option<Vec<f32>> = None;
        return chat::run_with(
            &format,
            tools,
            tokenizer,
            max_context,
            gen_tokens,
            cancel,
            generating,
            io::stdin().lock(),
            io::stdout().lock(),
            |ids| {
                gate.checkpoint(cancel)?;
                // Feed only what the session has not consumed; rebuild after /undo,
                // /reset or a cancelled pass, when the history no longer extends it,
                // from the opening snapshot when the history still starts with it.
                if session.is_broken() || !ids.starts_with(session.ids()) {
                    match &opening {
                        Some(start) if ids.starts_with(start.ids()) => session = start.clone(),
                        _ => session.reset(),
                    }
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
