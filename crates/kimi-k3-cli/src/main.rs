//! `k3`: run the real, released Kimi K3 checkpoint. Prompt in, tokens out.
//!
//! The Rust port of `src/cli/k3_run.c`'s core path: greedy decode streamed
//! through a [`TrunkRing`] and the real expert cache, incremental by default
//! (a [`TrunkSession`] carries KDA state and the MLA KV cache; `--recompute`
//! keeps the C engine's full-recompute default), with `--accel ane` putting
//! the dense and expert products on the Apple Neural Engine.
//!
//! Deliberately not ported, and flagged in `--help` rather than silently
//! missing: `--spec`/`--draft-trunk`
//! speculative decode, `--save-state`/`--load-state` conversation persistence,
//! the named memory-preset ladder with free-RAM auto-sizing,
//! `--ultra-low-memory`, and `--dump-logits`/`--dump-cache-trace` diagnostics.
//! These are real, useful C-engine features -- deferred, not forgotten; see
//! `docs/RUST_PORT.md`.

use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

mod accel;
mod chat;
mod linear;
mod quality;
mod thermal;

use kimi_k3_core::{
    bind::BoundStorage,
    cache::{CachedExperts, ExpertCache},
    config::K3Config,
    expert::ExpertRef,
    model::argmax,
    safetensors::SafeTensorIndex,
    tokenizer::Tokenizer,
    trunk::{Logits, TrunkRing, TrunkSession},
};

fn main() {
    if let Err(error) = run() {
        eprintln!("k3: {error}");
        std::process::exit(1);
    }
}

#[allow(clippy::struct_excessive_bools)] // one per CLI flag, and whether it was given
struct Args {
    model_dir: PathBuf,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    gen_tokens: usize,
    gen_given: bool,
    pin_layers: usize,
    ring_slots: usize,
    cache_gb: f64,
    cache_gb_given: bool,
    tok_dir: Option<PathBuf>,
    config_path: Option<PathBuf>,
    layers: Option<usize>,
    chat: bool,
    max_context: usize,
    max_context_given: bool,
    accel: accel::AccelKind,
    recompute: bool,
    fs_base: Option<PathBuf>,
    cas_root: Option<PathBuf>,
    cas_key: Option<PathBuf>,
    no_tools: bool,
    experts: quality::ExpertFormat,
    compare: Option<quality::Comparison>,
}

impl Args {
    #[allow(clippy::too_many_lines)] // one flat match arm per flag
    fn parse() -> Result<Self, String> {
        let mut raw = env::args().skip(1);
        let mut model_dir = None;
        let mut prompt = None;
        let mut prompt_file = None;
        let mut gen_tokens = None;
        let mut pin_layers = 2_usize;
        let mut ring_slots = 2_usize;
        let mut cache_gb = 4.0_f64;
        let mut cache_gb_given = false;
        let mut tok_dir = None;
        let mut config_path = None;
        let mut layers = None;
        let mut chat = false;
        let mut max_context = 512;
        let mut max_context_given = false;
        let mut accel = accel::AccelKind::Cpu;
        let mut recompute = false;
        let mut fs_base = None;
        let mut cas_root = None;
        let mut cas_key = None;
        let mut no_tools = false;
        let mut experts = quality::ExpertFormat::Bf16;
        let mut compare = None;

        if env::args().len() <= 1 {
            print_usage();
            std::process::exit(0);
        }

        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--prompt" => prompt = Some(next_value(&mut raw, "--prompt")?),
                "--chat" => chat = true,
                "--max-context" => {
                    max_context = parse_arg(&mut raw, "--max-context")?;
                    max_context_given = true;
                }
                "--prompt-file" => {
                    prompt_file = Some(PathBuf::from(next_value(&mut raw, "--prompt-file")?));
                }
                "--gen" => gen_tokens = Some(parse_arg(&mut raw, "--gen")?),
                "--pin-layers" => pin_layers = parse_arg(&mut raw, "--pin-layers")?,
                "--ring-slots" => ring_slots = parse_arg(&mut raw, "--ring-slots")?,
                "--cache-gb" => {
                    cache_gb = parse_arg(&mut raw, "--cache-gb")?;
                    cache_gb_given = true;
                }
                "--tok" => tok_dir = Some(PathBuf::from(next_value(&mut raw, "--tok")?)),
                "--config" => config_path = Some(PathBuf::from(next_value(&mut raw, "--config")?)),
                "--layers" => layers = Some(parse_arg(&mut raw, "--layers")?),
                "--recompute" => recompute = true,
                "--no-tools" => no_tools = true,
                "--experts" => {
                    experts = quality::ExpertFormat::parse(&next_value(&mut raw, "--experts")?)?;
                }
                "--compare" => {
                    compare = Some(quality::Comparison::parse(&next_value(
                        &mut raw,
                        "--compare",
                    )?)?);
                }
                "--fs-base" => fs_base = Some(PathBuf::from(next_value(&mut raw, "--fs-base")?)),
                "--cas-root" => cas_root = Some(PathBuf::from(next_value(&mut raw, "--cas-root")?)),
                "--cas-key" => cas_key = Some(PathBuf::from(next_value(&mut raw, "--cas-key")?)),
                "--accel" => accel = accel::AccelKind::parse(&next_value(&mut raw, "--accel")?)?,
                other if !other.starts_with('-') && model_dir.is_none() => {
                    model_dir = Some(PathBuf::from(other));
                }
                other => return Err(format!("unknown argument: {other}\nrun --help")),
            }
        }

        let model_dir = model_dir.ok_or_else(|| "missing <model_dir>\nrun --help".to_owned())?;
        match (&prompt, &prompt_file) {
            (None, None) => chat = true,
            (Some(_), Some(_)) => {
                return Err("pass only one of --prompt or --prompt-file".to_owned());
            }
            _ => {}
        }
        if chat && (prompt.is_some() || prompt_file.is_some() || layers.is_some()) {
            return Err("--chat cannot be combined with --prompt, --prompt-file or diagnostic --layers; run --help".into());
        }
        let gen_given = gen_tokens.is_some();
        let gen_tokens = gen_tokens.unwrap_or(if chat { 64 } else { 8 });
        if gen_tokens == 0 || max_context == 0 || ring_slots == 0 || layers == Some(0) {
            return Err(
                "--gen, --max-context, --ring-slots and --layers must be positive; run --help"
                    .into(),
            );
        }
        if !cache_gb.is_finite() || cache_gb <= 0.0 || cache_gb > 64.0 {
            return Err("--cache-gb must be finite and in (0, 64]; run --help".into());
        }

        Ok(Self {
            model_dir,
            prompt,
            prompt_file,
            gen_tokens,
            gen_given,
            pin_layers,
            ring_slots,
            cache_gb,
            cache_gb_given,
            tok_dir,
            config_path,
            layers,
            chat,
            max_context,
            max_context_given,
            accel,
            recompute,
            fs_base,
            cas_root,
            cas_key,
            no_tools,
            experts,
            compare,
        })
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_arg<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let value = next_value(args, flag)?;
    value
        .parse()
        .map_err(|_| format!("{flag}: {value:?} is not a valid value"))
}

fn print_usage() {
    println!(
        "k3, Kimi K3 inference engine (Rust port)\n\
         \n\
         usage: k3 <model_dir> [--chat] [options]\n\
         \x20      k3 <model_dir> --prompt TEXT [options]\n\
         \n\
         model_dir (required): local checkpoint directory; never downloaded\n\
         --chat (optional): interactive multi-turn chat, default without a prompt\n\
         \n\
         one-shot prompt (optional; mutually exclusive with chat):\n\
         \x20 --prompt TEXT         tokenize TEXT and run it\n\
         \x20 --prompt-file PATH    read the prompt from a file\n\
         \n\
         memory:\n\
         \x20 --pin-layers N        optional resident layers (default 2)\n\
         \x20 --ring-slots N        optional ring depth; 2 overlaps prefetch (default 2)\n\
         \x20 --cache-gb X          optional expert cache budget, GiB (default 4.0)\n\
         \n\
         generation:\n\
         \x20 --gen N               optional token limit (chat 64; one-shot 8)\n\
         \x20 --max-context N       optional total token limit, no silent eviction (default 512)\n\
         \n\
         diagnostics:\n\
         \x20 --tok DIR             optional tokenizer dir (default <model_dir>)\n\
         \x20 --config PATH         optional config (default <model_dir>/config.json)\n\
         \x20 --layers N            optional diagnostic layer subset; not allowed in chat\n\
         \x20 --help, -h            show help and exit\n\
         \n\
         example: k3 /Volumes/Jarraya/kimi-k3 --chat --gen 64 --max-context 512\n\
         In chat: /help, /continue, /undo, /reset, /stats, /quit. Ctrl-C cancels\n\
         generation at a safe layer/output boundary; Ctrl-D exits at the prompt.\n\
         \x20 --fs-base DIR        optional: Kimi Linear chat reads local files (read-only)\n\
         \x20                      with relative paths starting here (default: current dir)\n\
         \x20 --cas-root DIR       optional: also offer the newest snapshot in this Archive CAS\n\
         \x20                      root that verifies against --cas-key (a public key)\n\
         \x20 --cas-key PATH       optional: trusted Dilithium public key for --cas-root\n\
         \x20 --no-tools           optional: chat without file tools\n\
         \x20 --experts bf16|mxfp4 optional, Kimi Linear: round routed experts to 4-bit MXFP4\n\
         \x20                      as they load, to judge its quality (not faster; default bf16)\n\
         \x20 --compare mxfp4|cpu  optional, Kimi Linear, with --prompt-file: score the text\n\
         \x20                      twice (bf16 vs mxfp4 experts, or CPU vs --accel) and print\n\
         \x20                      perplexity, top-1 agreement and KL divergence\n\
         \x20 --recompute          optional: recompute the whole context every token (the old,\n\
         \x20                      slow reference path) instead of feeding only new tokens\n\
         \x20 --accel cpu|ane      optional device for the bf16 trunk products (default cpu,\n\
         \x20                      the bit-exact reference). ane: Apple Neural Engine via\n\
         \x20                      Core ML, fp16, macOS 15+, for trunk and expert products.\n\
         \n\
         Not yet ported from the C engine: --spec/--draft-trunk (speculative decode), --save-state/--load-state\n\
         (conversation persistence), the named memory-preset ladder, --ultra-low-memory,\n\
         and --dump-logits/--dump-cache-trace. See docs/RUST_PORT.md.\n\
         \n\
         Decode is incremental: each generated token streams the trunk once for one\n\
         position. On this Mac mini that is about a minute per token with --accel ane,\n\
         bound by reading the 109 GB trunk from disk. See docs/APPLE_NEURAL_ENGINE.md.\n"
    );
}

// Owns model/cache setup and teardown; generation/terminal state is delegated.
#[allow(clippy::too_many_lines)]
fn run() -> Result<(), String> {
    let args = Args::parse().map_err(|error| {
        if error.contains("run --help") {
            error
        } else {
            format!("{error}; run --help")
        }
    })?;

    let config_path = args
        .config_path
        .clone()
        .unwrap_or_else(|| args.model_dir.join("config.json"));

    let tok_dir = args
        .tok_dir
        .clone()
        .unwrap_or_else(|| args.model_dir.clone());
    let tokenizer = Tokenizer::load(&tok_dir)
        .map_err(|error| format!("cannot load tokenizer from {}: {error}", tok_dir.display()))?;

    let prompt = match (&args.prompt, &args.prompt_file) {
        (Some(text), None) => Some(text.clone()),
        (None, Some(path)) => Some(
            std::fs::read_to_string(path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?,
        ),
        _ => None,
    };

    let cancel = Arc::new(AtomicBool::new(false));
    let generating = Arc::new(AtomicBool::new(false));
    let signal_cancel = Arc::clone(&cancel);
    let signal_generating = Arc::clone(&generating);
    ctrlc::set_handler(move || {
        if signal_generating.load(Ordering::Relaxed) {
            signal_cancel.store(true, Ordering::Relaxed);
        } else {
            // Idle/loading: no partial transcript is being persisted.
            std::process::exit(130);
        }
    })
    .map_err(|error| format!("cannot install Ctrl-C handler: {error}"))?;

    if kimi_k3_core::linear::LinearConfig::detect(&config_path) {
        return linear::run(&args, &tokenizer, prompt.as_deref(), &cancel, &generating);
    }
    let mut config = K3Config::from_path(&config_path)
        .map_err(|error| format!("cannot read {}: {error}", config_path.display()))?;
    if let Some(layers) = args.layers {
        config.num_hidden_layers = layers.min(config.num_hidden_layers);
    }

    eprintln!(
        "indexing checkpoint shards under {}...",
        args.model_dir.display()
    );
    let start = Instant::now();
    let index = SafeTensorIndex::open(&args.model_dir)
        .map_err(|error| format!("cannot index {}: {error}", args.model_dir.display()))?;
    eprintln!(
        "  {} tensors across {} shards in {:?}",
        index.tensors().len(),
        index.shard_paths().len(),
        start.elapsed()
    );

    eprintln!(
        "loading the embedding, LM head, and pinning {} of {} layers...",
        args.pin_layers, config.num_hidden_layers
    );
    let start = Instant::now();
    let mut top_storage = BoundStorage::new();
    top_storage
        .load_top_level(&index)
        .map_err(|error| format!("cannot bind top-level tensors: {error}"))?;
    let top = top_storage
        .top_level()
        .map_err(|error| format!("cannot bind top-level tensors: {error}"))?;
    let mut ring = TrunkRing::open(&index, &config, args.pin_layers, args.ring_slots)
        .map_err(|error| format!("cannot open the trunk ring: {error}"))?;
    eprintln!("  ready in {:?}", start.elapsed());
    let device = accel::Device::open(args.accel)?;
    if args.accel == accel::AccelKind::Ane {
        eprintln!("dense trunk products: Apple Neural Engine (fp16, Core ML); experts on the CPU");
    }

    let probe_layer = (0..config.num_hidden_layers)
        .find(|&layer| !config.is_dense(layer))
        .ok_or_else(|| {
            "this checkpoint has no MoE layers to size the expert cache from".to_owned()
        })?;
    let probe = ExpertRef::resolve(&index, probe_layer, 0)
        .map_err(|error| format!("cannot size the expert cache: {error}"))?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let budget_bytes = (args.cache_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let mut cache = ExpertCache::new(
        config.num_hidden_layers,
        config.num_experts,
        config.num_experts_per_token,
        budget_bytes,
        &probe,
    )
    .map_err(|error| format!("cannot size the expert cache: {error}"))?;

    let mut gate = thermal::Gate::new()?;
    let mut session = TrunkSession::new(&config, args.max_context);
    let mut session_logits: Option<Vec<f32>> = None;
    if args.chat {
        println!("Local Kimi K3 -- about a minute per token on this Mac mini with --accel ane.");
        return chat::run(
            &tokenizer,
            args.max_context,
            args.gen_tokens,
            &cancel,
            &generating,
            io::stdin().lock(),
            io::stdout().lock(),
            |ids| {
                gate.checkpoint(&cancel)?;
                let started = Instant::now();
                eprintln!(
                    "forward pass: {} context tokens (Ctrl-C to cancel)...",
                    ids.len()
                );
                let mut experts = CachedExperts::new(&mut cache, &index);
                let keep = || !cancel.load(Ordering::Relaxed);
                let last = if args.recompute {
                    ring.forward_with(
                        &index,
                        &config,
                        &top,
                        ids,
                        &mut experts,
                        device.accel(),
                        Logits::Last,
                        keep,
                    )
                    .map_err(|e| e.to_string())?
                } else {
                    // The chat framework hands over the whole history each time; feed
                    // only what the session has not consumed. After /undo, /reset or a
                    // cancelled pass the history no longer extends the session: rebuild.
                    if session.is_broken() || !ids.starts_with(session.ids()) {
                        session.reset();
                        session_logits = None;
                    }
                    let new = &ids[session.ids().len()..];
                    if new.is_empty() {
                        session_logits.clone().ok_or("empty context")?
                    } else {
                        eprintln!("  {} new of {} context tokens", new.len(), ids.len());
                        ring.feed(
                            &index,
                            &config,
                            &top,
                            &mut session,
                            new,
                            &mut experts,
                            device.accel(),
                            keep,
                        )
                        .map_err(|e| e.to_string())?
                    }
                };
                session_logits = Some(last.clone());
                let last = last.as_slice();
                if last.iter().any(|value| !value.is_finite()) {
                    return Err("non-finite logits; refusing to emit a token".into());
                }
                eprintln!(
                    "forward pass finished in {:.1}s",
                    started.elapsed().as_secs_f64()
                );
                let summary = device.summary();
                if !summary.is_empty() {
                    eprintln!("  {summary}");
                }
                u32::try_from(argmax(last)).map_err(|e| e.to_string())
            },
        );
    }

    let mut ids = tokenizer.encode(prompt.as_deref().expect("one-shot mode has a prompt"));
    if ids.is_empty() {
        return Err("the prompt encoded to zero tokens".to_owned());
    }
    if ids
        .len()
        .checked_add(args.gen_tokens)
        .is_none_or(|len| len > args.max_context)
    {
        return Err(
            "prompt plus --gen exceeds --max-context; increase the limit explicitly".into(),
        );
    }
    eprintln!(
        "prompt: {} tokens, generating {}...",
        ids.len(),
        args.gen_tokens
    );

    let start = Instant::now();
    generating.store(true, Ordering::Relaxed);
    let mut decoder = loadngo_inference::Utf8Stream::default();
    print!("{}", tokenizer.decode_lossy(&ids));
    io::stdout().flush().map_err(|e| e.to_string())?;
    let mut generated = 0;
    for step in 0..args.gen_tokens {
        gate.checkpoint(&cancel)?;
        let mut experts = CachedExperts::new(&mut cache, &index);
        let pass = Instant::now();
        let keep = || !cancel.load(Ordering::Relaxed);
        let last = if args.recompute {
            ring.forward_with(
                &index,
                &config,
                &top,
                &ids,
                &mut experts,
                device.accel(),
                Logits::Last,
                keep,
            )
        } else {
            let new = &ids[session.ids().len()..];
            ring.feed(
                &index,
                &config,
                &top,
                &mut session,
                new,
                &mut experts,
                device.accel(),
                keep,
            )
        }
        .map_err(|error| format!("forward pass failed at generated token {step}: {error}"))?;
        eprintln!(
            "\n[forward pass {step}: {} tokens in {:.1}s]",
            ids.len(),
            pass.elapsed().as_secs_f64()
        );
        if last.iter().any(|value| !value.is_finite()) {
            return Err("non-finite logits; refusing to emit a token".into());
        }
        let next = u32::try_from(argmax(&last)).expect("vocab fits u32");
        ids.push(next);
        generated += 1;
        if [163_585, 163_586].contains(&next) {
            break;
        }
        print!("{}", decoder.push(&tokenizer.decode(&[next])));
        io::stdout().flush().map_err(|e| e.to_string())?;
    }
    generating.store(false, Ordering::Relaxed);
    eprintln!("generated {} tokens in {:?}", generated, start.elapsed());
    let (hits, misses) = ring.stats();
    eprintln!("trunk ring: {hits} hits, {misses} misses");
    let summary = device.summary();
    if !summary.is_empty() {
        eprintln!("{summary}");
    }

    println!("{}", decoder.finish());
    Ok(())
}
