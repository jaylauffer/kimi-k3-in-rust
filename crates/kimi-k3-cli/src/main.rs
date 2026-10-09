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
use std::time::{Duration, Instant};

mod accel;
mod agent_chat;
mod board;
mod chat;
mod chat_gemma;
mod convert;
mod gemma;
mod linear;
mod quality;
mod system_one;
mod thermal;
mod transcript;
// The wake-word parser and reply wrapper are portable and tested everywhere; only macOS
// listens and speaks, so elsewhere they are used by the tests alone.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod voice;

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
    no_web: bool,
    no_memory: bool,
    experts: quality::ExpertFormat,
    compare: Option<quality::Comparison>,
    convert_experts: Option<PathBuf>,
    mxfp4_experts: Option<PathBuf>,
    convert_mxfp4: Option<PathBuf>,
    quantize: gemma::Quantize,
    mxfp4: Option<PathBuf>,
    compare_with: Option<PathBuf>,
    attention: accel::AttentionDevice,
    system_one: Option<PathBuf>,
    temperature: f32,
    voice: bool,
    locale: String,
    budget: chat::TurnBudget,
    no_transcript: bool,
    no_checkpoint: bool,
    resume: Option<String>,
    legacy_chat: bool,
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
        let mut no_web = false;
        let mut no_memory = false;
        let mut experts = quality::ExpertFormat::Bf16;
        let mut compare = None;
        let mut convert_experts = None;
        let mut mxfp4_experts = None;
        let mut convert_mxfp4 = None;
        let mut quantize = gemma::Quantize::All;
        let mut mxfp4 = None;
        let mut compare_with = None;
        let mut attention = accel::AttentionDevice::Gpu;
        let mut system_one = None;
        let mut temperature = 1.0_f32;
        let mut voice = false;
        let mut locale = String::from("en-US");
        let mut budget = chat::TurnBudget::default();
        let mut no_transcript = false;
        let mut no_checkpoint = false;
        let mut resume = None;
        let mut legacy_chat = false;

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
                "--no-web" => no_web = true,
                "--no-memory" => no_memory = true,
                "--experts" => {
                    experts = quality::ExpertFormat::parse(&next_value(&mut raw, "--experts")?)?;
                }
                "--mxfp4-experts" => {
                    mxfp4_experts = Some(PathBuf::from(next_value(&mut raw, "--mxfp4-experts")?));
                }
                "--convert-experts-mxfp4" => {
                    convert_experts = Some(PathBuf::from(next_value(
                        &mut raw,
                        "--convert-experts-mxfp4",
                    )?));
                }
                "--convert-mxfp4" => {
                    convert_mxfp4 = Some(PathBuf::from(next_value(&mut raw, "--convert-mxfp4")?));
                }
                "--quantize" => {
                    quantize = gemma::Quantize::parse(&next_value(&mut raw, "--quantize")?)?;
                }
                "--mxfp4" => mxfp4 = Some(PathBuf::from(next_value(&mut raw, "--mxfp4")?)),
                "--attention" => {
                    attention =
                        accel::AttentionDevice::parse(&next_value(&mut raw, "--attention")?)?;
                }
                "--compare-with" => {
                    compare_with = Some(PathBuf::from(next_value(&mut raw, "--compare-with")?));
                }
                "--compare" => {
                    compare = Some(quality::Comparison::parse(&next_value(
                        &mut raw,
                        "--compare",
                    )?)?);
                }
                "--system-one" => {
                    system_one = Some(PathBuf::from(next_value(&mut raw, "--system-one")?));
                }
                "--temperature" => temperature = parse_arg(&mut raw, "--temperature")?,
                "--voice" => voice = true,
                "--locale" => locale = next_value(&mut raw, "--locale")?,
                "--turn-minutes" => {
                    let minutes: u64 = parse_arg(&mut raw, "--turn-minutes")?;
                    budget.time = (minutes > 0).then(|| Duration::from_secs(minutes * 60));
                }
                "--turn-tokens" => {
                    let tokens: usize = parse_arg(&mut raw, "--turn-tokens")?;
                    budget.tokens = (tokens > 0).then_some(tokens);
                }
                "--no-transcript" => no_transcript = true,
                "--legacy-chat" => legacy_chat = true,
                "--no-checkpoint" => no_checkpoint = true,
                "--resume" => resume = Some(next_value(&mut raw, "--resume")?),
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
        if resume.is_some() && (!chat || no_transcript) {
            return Err(
                "--resume needs chat and saved transcripts (no --no-transcript); run --help".into(),
            );
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
            no_web,
            no_memory,
            experts,
            compare,
            convert_experts,
            mxfp4_experts,
            convert_mxfp4,
            quantize,
            mxfp4,
            compare_with,
            attention,
            system_one,
            temperature,
            voice,
            locale,
            budget,
            no_transcript,
            no_checkpoint,
            resume,
            legacy_chat,
        })
    }
}

/// The chat's budget, and where it is saved or resumed from, as the flags ask.
fn chat_options(
    args: &Args,
    format: &'static str,
    max_context: usize,
) -> Result<chat::ChatOptions<'static>, String> {
    let mut options = chat::ChatOptions {
        budget: args.budget,
        ..chat::ChatOptions::default()
    };
    if args.no_transcript {
        return Ok(options);
    }
    let Some(dir) = transcript::default_dir() else {
        eprintln!("transcript: HOME is not set; the chat is not saved");
        return Ok(options);
    };
    if let Some(which) = &args.resume {
        let (saved, resumed) = transcript::Transcript::resume(&dir, which, format, max_context)?;
        options.transcript = Some(saved);
        options.resumed = Some(resumed);
    } else {
        match transcript::Transcript::create(&dir, format, &args.model_dir) {
            Ok(saved) => options.transcript = Some(saved),
            Err(e) => eprintln!("transcript: {e}; the chat is not saved"),
        }
    }
    Ok(options)
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

#[allow(clippy::too_many_lines)] // one help text
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
         Kimi Linear and Gemma chat on loadngo's shared loop (loadngo docs/AGENT_LOOP.md):\n\
         /help, /undo, /reset, /stats, /quit; Ctrl-C stops a reply (generation stops at a\n\
         safe layer/output boundary). Tools: files (read-only), text edits under\n\
         COLLABORATION.md (claims and handoffs on the board are written for her), cargo and\n\
         git (fixed subcommands, no shell), the Archive CAS, notes, the web, and board\n\
         sections. No terminal commands. At the prompt, Ctrl-C or Ctrl-D exits.\n\
         \x20 --legacy-chat        optional, Kimi Linear or Gemma: the chat before the shared\n\
         \x20                      loop, for what it alone has until step 3: /continue, turn\n\
         \x20                      budgets, --resume and compaction through a handoff\n\
         \x20 --fs-base DIR        optional: workspace for reads and text edits (root AGENTS.md/CLAUDE.md protected)\n\
         \x20                      with relative paths starting here (default: current dir)\n\
         \x20 --cas-root DIR       optional: an Archive CAS root to offer besides those found\n\
         \x20                      on attached drives (cas_archives lists every archive)\n\
         \x20 --cas-key PATH       optional: trusted Dilithium public key; archives it signed\n\
         \x20                      are marked signed, all others unsigned\n\
         \x20 --no-tools           optional: chat without tools\n\
         \x20 --no-web             optional: disable web_search/web_fetch\n\
         \x20 --no-memory          optional: chat without her memory (notes she keeps across\n\
         \x20                      sessions in ~/.loadngo/kimi/memory.jsonl)\n\
         \x20 --experts bf16|mxfp4 optional, Kimi Linear: round routed experts to 4-bit MXFP4\n\
         \x20                      as they load, to judge its quality (not faster; default bf16)\n\
         \x20 --compare mxfp4|cpu|decode|attention  optional, with --prompt-file: score the\n\
         \x20                      text twice (bf16 vs mxfp4 experts, CPU vs --accel, or CPU vs\n\
         \x20                      --accel one position at a time as chat decodes, or, for\n\
         \x20                      Gemma 4, attention on the GPU and the Neural Engine, each\n\
         \x20                      against the CPU reference) and print\n\
         \x20                      perplexity, top-1 agreement and KL divergence\n\
         \x20 --system-one FILE    optional, Kimi Linear: answer the typed questions in a JSON\n\
         \x20                      request ({{\"state\": ..., \"questions\": {{id: {{\"type\": \"noul\" |\n\
         \x20                      \"choice\" | \"score\", ...}}}}}}, TypeSafe's shape) with a probability per\n\
         \x20                      option, never free text; prints the answers as JSON\n\
         \x20 --temperature T      optional, with --system-one: calibration temperature (default 1)\n\
         \x20 --voice              optional, chat (macOS): talk to Kimi. On-device speech\n\
         \x20                      recognition listens; say \"Kimi, ...\" to ask something, and\n\
         \x20                      the reply is also spoken. Nothing leaves the machine.\n\
         \x20 --locale L           optional, with --voice: speech locale (default en-US)\n\
         \x20 --turn-minutes N     optional, --legacy-chat: pause a turn (every reply and tool round\n\
         \x20                      after one message) after N minutes; 0 = no limit (default 30)\n\
         \x20 --turn-tokens N      optional, --legacy-chat: pause a turn once Kimi has generated N\n\
         \x20                      tokens in it; 0 = no limit (default 16384). /continue\n\
         \x20                      resumes a paused turn with a fresh budget\n\
         \x20 --no-transcript      optional, chat: do not save the chat. By default each chat\n\
         \x20                      is logged to ~/.loadngo/kimi/transcripts/<time>.jsonl for\n\
         \x20                      review (--legacy-chat: with a resume snapshot, <time>.state.json)\n\
         \x20 --resume latest|PATH optional, --legacy-chat: carry on a saved chat (the newest, or a\n\
         \x20                      .jsonl/.state.json path) from its exact history\n\
         \x20 --no-checkpoint      optional, chat: without Jev (System One), which judges the\n\
         \x20                      work every 6 tool calls and acts when it is stuck, and gates\n\
         \x20                      web calls (--legacy-chat: its shadow questions at a handoff)\n\
         \x20 --convert-experts-mxfp4 DIR  optional, Kimi Linear: write every routed expert as\n\
         \x20                      MXFP4 into DIR, one file per layer; the checkpoint is only read\n\
         \x20 --mxfp4-experts DIR  optional, Kimi Linear: run with the routed experts converted\n\
         \x20                      into DIR (4-bit, all resident within --cache-gb)\n\
         \x20 --convert-mxfp4 DIR  optional, Gemma 4: write the decoder's matrices as MXFP4 into\n\
         \x20                      DIR, one file per layer; the checkpoint is only read\n\
         \x20 --quantize all|mlp   optional, with --convert-mxfp4: every matrix (default all), or\n\
         \x20                      only the MLPs (attention stays bf16; about twice the memory)\n\
         \x20 --mxfp4 DIR          optional, Gemma 4: run with the matrices converted into DIR\n\
         \x20                      (the rest from the checkpoint's bf16)\n\
         \x20 --attention gpu|npu|cpu  optional, Gemma 4 with --accel gpu: where prompt passes'\n\
         \x20                      attention runs (default gpu; npu: the Neural Engine, fp16,\n\
         \x20                      completions through the proactor; cpu: the reference).\n\
         \x20                      With npu, decoding stays on the GPU\n\
         \x20 --compare-with DIR   optional, Gemma 4, with --mxfp4 and --prompt-file: score the\n\
         \x20                      text with --mxfp4's weights, then with DIR's, and compare\n\
         \x20 --recompute          optional: recompute the whole context every token (the old,\n\
         \x20                      slow reference path) instead of feeding only new tokens\n\
         \x20 --accel cpu|ane|gpu  optional device for the bf16 trunk products (default cpu,\n\
         \x20                      the bit-exact reference). ane: Apple Neural Engine via\n\
         \x20                      Core ML, fp16, macOS 15+, for trunk and expert products.\n\
         \x20                      gpu (Kimi Linear): weights move into GPU memory once and\n\
         \x20                      each generated token's products run there (Metal, fp32);\n\
         \x20                      prompts still go to the Neural Engine.\n\
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
            // Idle at the prompt, or loading: the saved chat is complete up to here.
            std::process::exit(130);
        }
    })
    .map_err(|error| format!("cannot install Ctrl-C handler: {error}"))?;

    if kimi_k3_core::gemma::GemmaConfig::detect(&config_path) {
        return gemma::run(&args, &tokenizer, prompt.as_deref(), &cancel, &generating);
    }
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
        let options = chat_options(&args, "k3", args.max_context)?;
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
            options,
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
