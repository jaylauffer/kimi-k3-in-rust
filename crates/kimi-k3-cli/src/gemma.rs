//! `k3 <gemma-4-dir>`: the same CLI over a Gemma 4 dense checkpoint
//! (`model_type = "gemma4"`), detected from its `config.json`. See `docs/GEMMA.md`.
//!
//! `--convert-mxfp4 OUT` writes the decoder's matrices as MXFP4, one file per layer
//! (`--quantize mlp` converts only the MLPs); `--mxfp4 OUT` then runs from them. Every
//! weight is resident; with `--accel gpu` each moves into GPU memory as it is read.

use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use kimi_k3_core::gemma::{GemmaConfig, GemmaModel, MATRICES};
use kimi_k3_core::io::ReadRequest;
use kimi_k3_core::model::argmax;
use kimi_k3_core::safetensors::{DType, SafeTensorIndex};
use kimi_k3_core::tokenizer::Tokenizer;
use loadngo_weights::mxfp4::BLOCK_SIZE;

use crate::{Args, accel, quality, thermal};

/// Chat and one-shot defaults when `--gen`/`--max-context` are not given.
const ONE_SHOT_CONTEXT: usize = 8192;

/// Which matrices `--convert-mxfp4` converts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantize {
    /// All seven per-layer matrices.
    All,
    /// Only the MLP's gate, up and down; attention stays bf16.
    Mlp,
}

impl Quantize {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "mlp" => Ok(Self::Mlp),
            other => Err(format!("--quantize must be all or mlp, not {other}")),
        }
    }

    fn includes(self, name: &str) -> bool {
        self == Self::All || name.starts_with("mlp.")
    }

    const fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Mlp => "mlp",
        }
    }
}

pub fn layer_file(layer: usize) -> String {
    format!("gemma-mxfp4-layer-{layer:02}.safetensors")
}

/// Writes each layer's chosen matrices as MXFP4 (`…weight.blocks` `[rows][cols / 2]`,
/// `…weight.scales` `[rows][cols / 32]`, the Kimi Linear experts' layout). A layer is
/// written under a temporary name and renamed when whole; rerunning skips finished ones.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)] // one layer file, top to bottom
fn convert(
    model_dir: &Path,
    out: &Path,
    quantize: Quantize,
    gate: &mut thermal::Gate,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let config =
        GemmaConfig::from_path(model_dir.join("config.json")).map_err(|e| e.to_string())?;
    let index = SafeTensorIndex::open(model_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(out).map_err(|err| format!("{}: {err}", out.display()))?;
    let started = Instant::now();
    let (mut read, mut written) = (0_u64, 0_u64);
    for layer in 0..config.num_hidden_layers {
        let path = out.join(layer_file(layer));
        if path.exists() {
            eprintln!("layer {layer}: already converted, skipped");
            continue;
        }
        gate.checkpoint(cancel)?;
        let layer_start = Instant::now();
        let mut parts = Vec::new();
        for name in MATRICES.iter().filter(|n| quantize.includes(n)) {
            let Some((rows, cols)) = config.matrix_shape(layer, name) else {
                continue;
            };
            if cols % BLOCK_SIZE != 0 {
                return Err(format!(
                    "{name} width {cols} is not a multiple of {BLOCK_SIZE}"
                ));
            }
            let full = format!("{}{name}.weight", GemmaConfig::layer_prefix(layer));
            let tensor = index
                .tensor(&full)
                .ok_or_else(|| format!("missing {full}"))?;
            if tensor.dtype != DType::Bf16 || tensor.shape != [rows, cols] {
                return Err(format!(
                    "{full} is {:?} {:?}, expected Bf16 [{rows}, {cols}]",
                    tensor.dtype, tensor.shape
                ));
            }
            parts.push((full, rows, cols, tensor.clone()));
        }
        let mut header = serde_json::Map::new();
        let mut offset = 0_usize;
        for (name, rows, cols, _) in &parts {
            for (suffix, width) in [("blocks", cols / 2), ("scales", cols / BLOCK_SIZE)] {
                let bytes = rows * width;
                header.insert(
                    format!("{name}.{suffix}"),
                    serde_json::json!({"dtype": "U8", "shape": [rows, width],
                        "data_offsets": [offset, offset + bytes]}),
                );
                offset += bytes;
            }
        }
        header.insert(
            "__metadata__".into(),
            serde_json::json!({"format": "OCP MX v1.0 MXFP4 (E2M1, E8M0 per 32)",
                "encoder": "loadngo-weights mxfp4::quantize_block",
                "quantize": quantize.name()}),
        );
        let mut json = serde_json::Value::Object(header).to_string().into_bytes();
        while json.len() % 8 != 0 {
            json.push(b' ');
        }
        let partial = out.join(format!("{}.partial", layer_file(layer)));
        let file =
            fs::File::create(&partial).map_err(|err| format!("{}: {err}", partial.display()))?;
        let mut writer = BufWriter::with_capacity(8 << 20, file);
        let io = |err: io::Error| format!("{}: {err}", partial.display());
        writer
            .write_all(&(json.len() as u64).to_le_bytes())
            .map_err(io)?;
        writer.write_all(&json).map_err(io)?;
        let requests = parts
            .iter()
            .map(|(_, _, _, t)| ReadRequest {
                shard: t.shard,
                offset: t.offset,
                buffer: vec![0; t.nbytes],
            })
            .collect();
        let buffers = index.read_batch(requests).map_err(|e| e.to_string())?;
        let (mut blocks, mut scales) = (Vec::new(), Vec::new());
        for (raw, (_, _, cols, _)) in buffers.iter().zip(&parts) {
            read += raw.len() as u64;
            blocks.clear();
            scales.clear();
            crate::convert::quantize(raw, *cols, &mut blocks, &mut scales);
            writer.write_all(&blocks).map_err(io)?;
            writer.write_all(&scales).map_err(io)?;
            written += (blocks.len() + scales.len()) as u64;
        }
        writer.flush().map_err(io)?;
        writer
            .into_inner()
            .map_err(|e| e.to_string())?
            .sync_all()
            .map_err(io)?;
        fs::rename(&partial, &path).map_err(|err| format!("{}: {err}", path.display()))?;
        eprintln!(
            "layer {layer}: {} matrices in {:.1?} ({:.2} GB so far, {:.1?} total)",
            parts.len(),
            layer_start.elapsed(),
            written as f64 / 1e9,
            started.elapsed()
        );
    }
    let record = serde_json::json!({
        "source": model_dir.display().to_string(),
        "quantize": quantize.name(),
        "matrices": MATRICES.iter().filter(|n| quantize.includes(n)).collect::<Vec<_>>(),
        "format": "OCP MX v1.0 MXFP4 (E2M1, E8M0 per 32), loadngo-weights mxfp4::quantize_block",
        "converted_by": format!("k3 {} --convert-mxfp4", env!("CARGO_PKG_VERSION")),
    });
    fs::write(
        out.join("CONVERSION.json"),
        serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "converted: {:.1} GB of bf16 read, {:.1} GB of MXFP4 written in {:.1?}",
        read as f64 / 1e9,
        written as f64 / 1e9,
        started.elapsed()
    );
    Ok(())
}

fn pick(logits: &[f32]) -> Result<u32, String> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err("non-finite logits; refusing to emit a token".into());
    }
    u32::try_from(argmax(logits)).map_err(|e| e.to_string())
}

/// Loads the model, moving weights into the device's memory as they are read.
#[allow(clippy::cast_precision_loss)]
fn load(
    model_dir: &Path,
    mxfp4: Option<&Path>,
    device: Option<&dyn kimi_k3_core::layer::DenseAccel>,
    cancel: &AtomicBool,
) -> Result<GemmaModel, String> {
    let start = Instant::now();
    let model = GemmaModel::load(model_dir, mxfp4, device, || !cancel.load(Ordering::Relaxed))
        .map_err(|e| e.to_string())?;
    let (bf16, mx) = model.weight_bytes();
    eprintln!(
        "  loaded in {:.1?}: {:.1} GB bf16, {:.1} GB MXFP4{}",
        start.elapsed(),
        bf16 as f64 / 1e9,
        mx as f64 / 1e9,
        if device.is_some() {
            ", in GPU memory"
        } else {
            ""
        }
    );
    Ok(model)
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
pub fn run(
    args: &Args,
    tokenizer: &Tokenizer,
    prompt: Option<&str>,
    cancel: &AtomicBool,
    generating: &AtomicBool,
) -> Result<(), String> {
    if args.recompute || args.layers.is_some() || args.mxfp4_experts.is_some() {
        return Err(
            "--recompute, --layers and --mxfp4-experts do not apply to Gemma; run --help".into(),
        );
    }
    let mut gate = thermal::Gate::new()?;
    if let Some(out) = &args.convert_mxfp4 {
        return convert(&args.model_dir, out, args.quantize, &mut gate, cancel);
    }
    let device = accel::Device::open(args.accel)?;
    device.set_attention(args.attention)?;
    let gpu = if args.accel == accel::AccelKind::Gpu {
        Some(device.accel().ok_or("the GPU device did not open")?)
    } else {
        None
    };
    eprintln!(
        "loading Gemma 4 from {}{}...",
        args.model_dir.display(),
        args.mxfp4
            .as_ref()
            .map(|d| format!(" with MXFP4 matrices from {}", d.display()))
            .unwrap_or_default()
    );
    let mut model = load(&args.model_dir, args.mxfp4.as_deref(), gpu, cancel)?;
    let keep = || !cancel.load(Ordering::Relaxed);
    let bos = model.config.bos_token_id;

    if let Some(comparison) = args.compare.or(args
        .compare_with
        .as_ref()
        .map(|_| quality::Comparison::Mxfp4))
    {
        let text = prompt.ok_or("--compare needs --prompt-file (or --prompt)")?;
        let mut ids = vec![bos];
        ids.extend(tokenizer.encode(text));
        let vocab = model.config.vocab_size;
        let score = |model: &GemmaModel, accel, one_at_a_time: bool| -> Result<Vec<f32>, String> {
            let start = Instant::now();
            let mut session = model.session(ids.len());
            let logits = if one_at_a_time {
                let mut all = Vec::with_capacity(ids.len() * vocab);
                for id in &ids {
                    all.extend(
                        model
                            .feed(&mut session, std::slice::from_ref(id), accel, keep)
                            .map_err(|e| e.to_string())?,
                    );
                }
                all
            } else {
                model
                    .score(&mut session, &ids, accel, keep)
                    .map_err(|e| e.to_string())?
            };
            eprintln!(
                "  {} tokens scored in {:.1?}{}",
                ids.len(),
                start.elapsed(),
                if one_at_a_time { ", one at a time" } else { "" }
            );
            Ok(logits)
        };
        let (names, a, b) = match (comparison, &args.compare_with) {
            (quality::Comparison::Cpu, _) => {
                let a = score(&model, None, false)?;
                (["cpu", "device"], a, score(&model, device.accel(), false)?)
            }
            (quality::Comparison::Attention, _) => {
                // Each device against the CPU reference, products on the GPU throughout.
                device.set_attention(accel::AttentionDevice::Cpu)?;
                let reference = score(&model, device.accel(), false)?;
                device.set_attention(accel::AttentionDevice::Gpu)?;
                let gpu_logits = score(&model, device.accel(), false)?;
                quality::report(
                    ["attention on the CPU", "on the GPU"],
                    &ids,
                    &reference,
                    &gpu_logits,
                    vocab,
                    tokenizer,
                );
                drop(gpu_logits);
                device.set_attention(accel::AttentionDevice::Npu)?;
                let npu_logits = score(&model, device.accel(), false)?;
                (
                    ["attention on the CPU", "on the Neural Engine"],
                    reference,
                    npu_logits,
                )
            }
            (quality::Comparison::Decode, _) => {
                let a = score(&model, None, false)?;
                (
                    ["cpu", "device, one at a time"],
                    a,
                    score(&model, device.accel(), true)?,
                )
            }
            (quality::Comparison::Mxfp4, Some(other)) => {
                let a = score(&model, device.accel(), false)?;
                drop(model);
                eprintln!("loading the comparison weights from {}...", other.display());
                model = load(&args.model_dir, Some(other), gpu, cancel)?;
                (
                    ["--mxfp4", "--compare-with"],
                    a,
                    score(&model, device.accel(), false)?,
                )
            }
            (quality::Comparison::Mxfp4, None) => {
                return Err(
                    "for Gemma, compare two converted directories: --mxfp4 A --compare-with B"
                        .into(),
                );
            }
        };
        quality::report(names, &ids, &a, &b, vocab, tokenizer);
        let summary = device.summary();
        if !summary.is_empty() {
            eprintln!("{summary}");
        }
        return Ok(());
    }

    let max_context = if args.max_context_given {
        args.max_context
    } else {
        ONE_SHOT_CONTEXT
    };
    if args.chat {
        return chat(
            args,
            &mut model,
            tokenizer,
            &device,
            max_context,
            &mut gate,
            cancel,
            generating,
        );
    }

    let prompt = prompt.ok_or("one-shot mode needs --prompt")?;
    let mut ids = vec![bos];
    ids.extend(tokenizer.encode(prompt));
    let gen_tokens = args.gen_tokens;
    if ids.len() + gen_tokens > max_context {
        return Err("prompt plus --gen exceeds --max-context".into());
    }
    eprintln!("prompt: {} tokens, generating {}...", ids.len(), gen_tokens);
    let mut session = model.session(max_context);
    generating.store(true, Ordering::Relaxed);
    let mut decoder = loadngo_inference::Utf8Stream::default();
    print!("{}", tokenizer.decode_lossy(&ids[1..]));
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
        if model.config.eos_token_ids.contains(&next) {
            break;
        }
        print!("{}", decoder.push(&tokenizer.decode(&[next])));
        io::stdout().flush().map_err(|e| e.to_string())?;
    }
    generating.store(false, Ordering::Relaxed);
    eprintln!(
        "\ngenerated {generated} tokens in {:.1?}; decode {:.2} tokens/s after the prompt pass",
        started.elapsed(),
        if generated > 1 {
            f64::from(generated - 1) / decode_s
        } else {
            0.0
        }
    );
    let summary = device.summary();
    if !summary.is_empty() {
        eprintln!("{summary}");
    }
    Ok(())
}

/// Chat defaults when `--gen` is not given: replies as long as Kimi Linear's.
const CHAT_GEN: usize = 1024;

/// Interactive chat with tools, on the shared loop as Kimi Linear's.
#[allow(clippy::too_many_arguments)]
fn chat(
    args: &Args,
    model: &mut GemmaModel,
    tokenizer: &Tokenizer,
    device: &accel::Device,
    max_context: usize,
    gate: &mut thermal::Gate,
    cancel: &AtomicBool,
    generating: &AtomicBool,
) -> Result<(), String> {
    let gen_tokens = if args.gen_given {
        args.gen_tokens
    } else {
        CHAT_GEN
    };
    let config = &model.config;
    let format =
        crate::chat::ChatFormat::gemma(tokenizer, config.bos_token_id, &config.eos_token_ids)?;
    crate::agent_chat::run_reader(
        args,
        crate::agent_chat::Loaded {
            model,
            about: crate::agent_chat::About {
                description: "Gemma 4 31B-it (Google, open weights)",
                identity: "Gemma",
                engine: crate::linear::engine_name(args.accel),
                format: &format,
            },
            accel: device.accel(),
        },
        tokenizer,
        crate::linear::extra_notes(args),
        max_context,
        gen_tokens,
        gate,
        cancel,
        generating,
    )
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    /// The GPU path (resident weights, Metal products, grouped attention) against
    /// transformers' logits for the tiny oracle model (see `scripts/gemma_tiny_oracle.py`),
    /// fed whole and then a token at a time. Fails if the GPU declined the work.
    /// As [`gpu_matches_the_transformers_oracle`] with prompt passes' attention on the
    /// Neural Engine (fp16): looser bounds, and every prompt pass must have run there.
    #[test]
    fn npu_attention_matches_the_transformers_oracle() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/gemma4/tiny");
        let oracle: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("oracle.json")).unwrap()).unwrap();
        let ids: Vec<u32> = serde_json::from_value(oracle["ids"].clone()).unwrap();
        let positions: Vec<usize> = serde_json::from_value(oracle["positions"].clone()).unwrap();
        let want: Vec<Vec<f32>> = serde_json::from_value(oracle["logits"].clone()).unwrap();
        let device = accel::Device::open(accel::AccelKind::Gpu).unwrap();
        device.set_attention(accel::AttentionDevice::Npu).unwrap();
        let gpu = device.accel();
        let model = GemmaModel::load(&dir, None, gpu, || true).unwrap();
        let vocab = model.config.vocab_size;
        let mut session = model.session(ids.len());
        // A whole pass, then pieces and single tokens: the Neural Engine's cache must
        // keep up through the GPU's decoding steps.
        let mut fed = 0;
        for size in [512, 40, 1, 1, 33, 13] {
            let end = (fed + size).min(ids.len());
            let all = model
                .score(&mut session, &ids[fed..end], gpu, || true)
                .unwrap();
            for (k, &p) in positions
                .iter()
                .enumerate()
                .filter(|&(_, &p)| p >= fed && p < end)
            {
                let row = &all[(p - fed) * vocab..(p - fed + 1) * vocab];
                let error = row
                    .iter()
                    .zip(&want[k])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(error < 5e-2, "position {p}: {error}");
            }
            fed = end;
        }
        // The whole text in one call: a pair of passes (512 + 88), both started on the
        // Neural Engine and finished through the proactor while the GPU works.
        let mut session = model.session(ids.len());
        let all = model.score(&mut session, &ids, gpu, || true).unwrap();
        for (&p, w) in positions.iter().zip(&want) {
            let row = &all[p * vocab..(p + 1) * vocab];
            let error = row
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(error < 5e-2, "paired passes, position {p}: {error}");
        }
        let summary = device.summary();
        assert!(!summary.contains(" 0 on the Neural Engine"), "{summary}");
        assert!(
            summary.contains(", 0 on the GPU after a Neural Engine failure"),
            "{summary}"
        );
    }

    #[test]
    fn gpu_matches_the_transformers_oracle() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/gemma4/tiny");
        let oracle: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("oracle.json")).unwrap()).unwrap();
        let ids: Vec<u32> = serde_json::from_value(oracle["ids"].clone()).unwrap();
        let positions: Vec<usize> = serde_json::from_value(oracle["positions"].clone()).unwrap();
        let want: Vec<Vec<f32>> = serde_json::from_value(oracle["logits"].clone()).unwrap();
        let device = accel::Device::open(accel::AccelKind::Gpu).unwrap();
        let gpu = device.accel();
        let model = GemmaModel::load(&dir, None, gpu, || true).unwrap();
        let vocab = model.config.vocab_size;
        let worst = |got: &[f32], want: &[f32]| {
            got.iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
        };
        let mut session = model.session(ids.len());
        let all = model.score(&mut session, &ids, gpu, || true).unwrap();
        for (&p, w) in positions.iter().zip(&want) {
            let error = worst(&all[p * vocab..(p + 1) * vocab], w);
            assert!(error < 2e-3, "whole text, position {p}: {error}");
        }
        let mut session = model.session(ids.len());
        for (i, id) in ids.iter().enumerate() {
            let logits = model
                .feed(&mut session, std::slice::from_ref(id), gpu, || true)
                .unwrap();
            if let Some(k) = positions.iter().position(|&p| p == i) {
                let error = worst(&logits, &want[k]);
                assert!(error < 2e-3, "one at a time, position {i}: {error}");
            }
        }
        let summary = device.summary();
        assert!(summary.contains("0 GPU failures"), "{summary}");
        assert!(
            summary.contains(", 0 on the CPU after a GPU failure, "),
            "{summary}"
        );
        assert!(!summary.contains("grouped attention 0 layers"), "{summary}");
        assert!(summary.contains(" 0 steps to the ANE"), "{summary}");
    }
}
