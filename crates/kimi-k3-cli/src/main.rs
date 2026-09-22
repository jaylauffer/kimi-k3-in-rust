//! `k3`: run the real, released Kimi K3 checkpoint. Prompt in, tokens out.
//!
//! The Rust port of `src/cli/k3_run.c`'s core path: greedy decode by full
//! recompute (the C engine's own default, non-`--incremental` mode, which is
//! also what its own full-model oracle validates), streamed through a
//! [`TrunkRing`] and the real expert cache.
//!
//! Deliberately not ported, and flagged in `--help` rather than silently
//! missing: `--incremental`/KV-cache carry-forward, `--spec`/`--draft-trunk`
//! speculative decode, `--save-state`/`--load-state` conversation persistence,
//! the named memory-preset ladder with free-RAM auto-sizing,
//! `--ultra-low-memory`, and `--dump-logits`/`--dump-cache-trace` diagnostics.
//! These are real, useful C-engine features -- deferred, not forgotten; see
//! `docs/RUST_PORT.md`.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use kimi_k3_core::{
    bind::BoundStorage,
    cache::{CachedExperts, ExpertCache},
    config::K3Config,
    expert::ExpertRef,
    model::argmax,
    safetensors::SafeTensorIndex,
    tokenizer::Tokenizer,
    trunk::TrunkRing,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("k3: {error}");
        std::process::exit(1);
    }
}

struct Args {
    model_dir: PathBuf,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    gen_tokens: usize,
    pin_layers: usize,
    ring_slots: usize,
    cache_gb: f64,
    tok_dir: Option<PathBuf>,
    config_path: Option<PathBuf>,
    layers: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut raw = env::args().skip(1);
        let mut model_dir = None;
        let mut prompt = None;
        let mut prompt_file = None;
        let mut gen_tokens = 8_usize;
        let mut pin_layers = 4_usize;
        let mut ring_slots = 2_usize;
        let mut cache_gb = 4.0_f64;
        let mut tok_dir = None;
        let mut config_path = None;
        let mut layers = None;

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
                "--prompt-file" => {
                    prompt_file = Some(PathBuf::from(next_value(&mut raw, "--prompt-file")?));
                }
                "--gen" => gen_tokens = parse_arg(&mut raw, "--gen")?,
                "--pin-layers" => pin_layers = parse_arg(&mut raw, "--pin-layers")?,
                "--ring-slots" => ring_slots = parse_arg(&mut raw, "--ring-slots")?,
                "--cache-gb" => cache_gb = parse_arg(&mut raw, "--cache-gb")?,
                "--tok" => tok_dir = Some(PathBuf::from(next_value(&mut raw, "--tok")?)),
                "--config" => config_path = Some(PathBuf::from(next_value(&mut raw, "--config")?)),
                "--layers" => layers = Some(parse_arg(&mut raw, "--layers")?),
                other if !other.starts_with('-') && model_dir.is_none() => {
                    model_dir = Some(PathBuf::from(other));
                }
                other => return Err(format!("unknown argument: {other}\nrun --help")),
            }
        }

        let model_dir = model_dir.ok_or_else(|| "missing <model_dir>\nrun --help".to_owned())?;
        match (&prompt, &prompt_file) {
            (None, None) => {
                return Err(
                    "exactly one of --prompt or --prompt-file is required\nrun --help".to_owned(),
                );
            }
            (Some(_), Some(_)) => {
                return Err("pass only one of --prompt or --prompt-file".to_owned());
            }
            _ => {}
        }

        Ok(Self {
            model_dir,
            prompt,
            prompt_file,
            gen_tokens,
            pin_layers,
            ring_slots,
            cache_gb,
            tok_dir,
            config_path,
            layers,
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
         usage: k3 <model_dir> --prompt TEXT [options]\n\
         \n\
         prompt (exactly one):\n\
         \x20 --prompt TEXT         tokenize TEXT and run it\n\
         \x20 --prompt-file PATH    read the prompt from a file\n\
         \n\
         memory:\n\
         \x20 --pin-layers N        layers held resident for the whole run (default 4)\n\
         \x20 --ring-slots N        streaming ring depth; 2 enables prefetch overlap (default 2)\n\
         \x20 --cache-gb X          routed-expert cache budget in GiB (default 4.0)\n\
         \n\
         generation:\n\
         \x20 --gen N               tokens to generate (default 8)\n\
         \n\
         diagnostics:\n\
         \x20 --tok DIR             tiktoken.model/tokenizer_config.json dir (default <model_dir>)\n\
         \x20 --config PATH         model config (default <model_dir>/config.json)\n\
         \x20 --layers N            bind only the first N layers (partial shard sets)\n\
         \x20 --help\n\
         \n\
         Not yet ported from the C engine: --incremental (KV-cache carry-forward),\n\
         --spec/--draft-trunk (speculative decode), --save-state/--load-state\n\
         (conversation persistence), the named memory-preset ladder, --ultra-low-memory,\n\
         and --dump-logits/--dump-cache-trace. See docs/RUST_PORT.md.\n\
         \n\
         Decode is full recompute every step (the C engine's own default, non-\n\
         --incremental mode): O(sequence length squared), correct but not the fastest\n\
         available shape. --gen N therefore costs roughly N times a single forward pass.\n"
    );
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;

    let config_path = args
        .config_path
        .clone()
        .unwrap_or_else(|| args.model_dir.join("config.json"));
    let mut config = K3Config::from_path(&config_path)
        .map_err(|error| format!("cannot read {}: {error}", config_path.display()))?;
    if let Some(layers) = args.layers {
        config.num_hidden_layers = layers.min(config.num_hidden_layers);
    }

    let tok_dir = args
        .tok_dir
        .clone()
        .unwrap_or_else(|| args.model_dir.clone());
    let tokenizer = Tokenizer::load(&tok_dir)
        .map_err(|error| format!("cannot load tokenizer from {}: {error}", tok_dir.display()))?;

    let prompt = match (&args.prompt, &args.prompt_file) {
        (Some(text), None) => text.clone(),
        (None, Some(path)) => std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?,
        _ => unreachable!("Args::parse enforces exactly one of --prompt/--prompt-file"),
    };

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

    let mut ids = tokenizer.encode(&prompt);
    if ids.is_empty() {
        return Err("the prompt encoded to zero tokens".to_owned());
    }
    eprintln!(
        "prompt: {} tokens, generating {}...",
        ids.len(),
        args.gen_tokens
    );

    let start = Instant::now();
    for step in 0..args.gen_tokens {
        let mut experts = CachedExperts::new(&mut cache, &index);
        let logits = ring
            .forward(&index, &config, &top, &ids, &mut experts)
            .map_err(|error| format!("forward pass failed at generated token {step}: {error}"))?;
        let vocab = config.vocab_size;
        let last = &logits[(ids.len() - 1) * vocab..];
        let next = u32::try_from(argmax(last)).expect("vocab fits u32");
        ids.push(next);
    }
    eprintln!(
        "generated {} tokens in {:?}",
        args.gen_tokens,
        start.elapsed()
    );
    let (hits, misses) = ring.stats();
    eprintln!("trunk ring: {hits} hits, {misses} misses");

    println!("{}", tokenizer.decode_lossy(&ids));
    Ok(())
}
