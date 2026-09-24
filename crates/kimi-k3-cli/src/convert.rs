//! `--convert-experts-mxfp4 OUT`: writes every routed expert of a Kimi Linear checkpoint
//! as OCP MX v1.0 MXFP4 into its own directory, one safetensors file per `MoE` layer. The
//! source checkpoint is only read; the trunk stays there and is loaded from it.
//!
//! For each `model.layers.L.block_sparse_moe.experts.E.{w1,w2,w3}.weight` (bf16
//! `[rows][cols]`) the output holds two U8 tensors with the same prefix:
//!
//! - `…weight.blocks`, `[rows][cols / 2]`: E2M1 codes, the even element in the low nibble;
//! - `…weight.scales`, `[rows][cols / 32]`: one E8M0 scale per 32 elements.
//!
//! Encoding is `loadngo-weights` `mxfp4::quantize_block`, the same rounding
//! `--experts mxfp4` applies (and the quality in `docs/KIMI_LINEAR.md` was measured
//! with). Each layer is written to a temporary name and renamed when complete, so an
//! interrupted run leaves only whole layers; rerunning skips layers already present.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use kimi_k3_core::io::ReadRequest;
use kimi_k3_core::linear::LinearConfig;
use kimi_k3_core::safetensors::{DType, SafeTensorIndex};
use loadngo_weights::mxfp4::{BLOCK_SIZE, quantize_block};

use crate::thermal;

/// Experts read per batch (as the runtime cache does): ~450 MB of bf16 in flight.
const READ_CHUNK: usize = 32;

const PARTS: [&str; 3] = ["w1", "w3", "w2"];

/// The output file for one `MoE` layer.
pub fn layer_file(layer: usize) -> String {
    format!("experts-mxfp4-layer-{layer:02}.safetensors")
}

/// Quantizes one bf16 `[rows][cols]` matrix into `blocks` and `scales`.
fn quantize(raw: &[u8], cols: usize, blocks: &mut Vec<u8>, scales: &mut Vec<u8>) {
    let mut values = [0.0_f32; BLOCK_SIZE];
    let mut packed = [0_u8; BLOCK_SIZE / 2];
    for row in raw.chunks_exact(cols * 2) {
        for block in row.chunks_exact(BLOCK_SIZE * 2) {
            for (v, pair) in values.iter_mut().zip(block.chunks_exact(2)) {
                *v = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            }
            scales.push(quantize_block(&values, &mut packed));
            blocks.extend_from_slice(&packed);
        }
    }
}

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn run(
    model_dir: &Path,
    out: &Path,
    gate: &mut thermal::Gate,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let config =
        LinearConfig::from_path(model_dir.join("config.json")).map_err(|e| e.to_string())?;
    let (e, inter) = (config.hidden_size, config.moe_intermediate_size);
    if e % BLOCK_SIZE != 0 || inter % BLOCK_SIZE != 0 {
        return Err(format!(
            "expert widths {e} and {inter} are not multiples of {BLOCK_SIZE}"
        ));
    }
    let index = SafeTensorIndex::open(model_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(out).map_err(|err| format!("{}: {err}", out.display()))?;
    let shapes = [(inter, e), (inter, e), (e, inter)];
    let started = Instant::now();
    let (mut read, mut written) = (0_u64, 0_u64);
    for layer in (0..config.num_hidden_layers).filter(|&l| !config.is_dense(l)) {
        let path = out.join(layer_file(layer));
        if path.exists() {
            eprintln!("layer {layer}: already converted, skipped");
            continue;
        }
        gate.checkpoint(cancel)?;
        let layer_start = Instant::now();
        // Header: every tensor's name, shape and byte range, in write order.
        let mut header = serde_json::Map::new();
        let mut offset = 0_usize;
        for expert in 0..config.num_experts {
            for (part, (rows, cols)) in PARTS.iter().zip(shapes) {
                let name =
                    format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.{part}.weight");
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
        }
        header.insert(
            "__metadata__".into(),
            serde_json::json!({"format": "OCP MX v1.0 MXFP4 (E2M1, E8M0 per 32)",
                "encoder": "loadngo-weights mxfp4::quantize_block"}),
        );
        let mut json = serde_json::Value::Object(header).to_string().into_bytes();
        while json.len() % 8 != 0 {
            json.push(b' ');
        }
        let partial = out.join(format!("{}.partial", layer_file(layer)));
        let file =
            fs::File::create(&partial).map_err(|err| format!("{}: {err}", partial.display()))?;
        let mut writer = BufWriter::with_capacity(8 << 20, file);
        let io = |err: std::io::Error| format!("{}: {err}", partial.display());
        writer
            .write_all(&(json.len() as u64).to_le_bytes())
            .map_err(io)?;
        writer.write_all(&json).map_err(io)?;
        let mut blocks = Vec::new();
        let mut scales = Vec::new();
        let experts: Vec<usize> = (0..config.num_experts).collect();
        for chunk in experts.chunks(READ_CHUNK) {
            let mut requests = Vec::with_capacity(3 * chunk.len());
            for &expert in chunk {
                for (part, (rows, cols)) in PARTS.iter().zip(shapes) {
                    let name = format!(
                        "model.layers.{layer}.block_sparse_moe.experts.{expert}.{part}.weight"
                    );
                    let tensor = index
                        .tensor(&name)
                        .ok_or_else(|| format!("missing {name}"))?;
                    if tensor.dtype != DType::Bf16 || tensor.shape != [rows, cols] {
                        return Err(format!(
                            "{name} is {:?} {:?}, expected Bf16 [{rows}, {cols}]",
                            tensor.dtype, tensor.shape
                        ));
                    }
                    requests.push(ReadRequest {
                        shard: tensor.shard,
                        offset: tensor.offset,
                        buffer: vec![0; tensor.nbytes],
                    });
                }
            }
            let buffers = index.read_batch(requests).map_err(|e| e.to_string())?;
            for (raw, (_, cols)) in buffers.iter().zip(shapes.iter().cycle()) {
                read += raw.len() as u64;
                blocks.clear();
                scales.clear();
                quantize(raw, *cols, &mut blocks, &mut scales);
                writer.write_all(&blocks).map_err(io)?;
                writer.write_all(&scales).map_err(io)?;
                written += (blocks.len() + scales.len()) as u64;
            }
        }
        writer.flush().map_err(io)?;
        writer
            .into_inner()
            .map_err(|e| e.to_string())?
            .sync_all()
            .map_err(io)?;
        fs::rename(&partial, &path).map_err(|err| format!("{}: {err}", path.display()))?;
        eprintln!(
            "layer {layer}: {} experts in {:.1?} ({:.2} GB so far, {:.1?} total)",
            config.num_experts,
            layer_start.elapsed(),
            written as f64 / 1e9,
            started.elapsed()
        );
    }
    eprintln!(
        "converted: {:.1} GB of bf16 experts read, {:.1} GB of MXFP4 written in {:.1?}",
        read as f64 / 1e9,
        written as f64 / 1e9,
        started.elapsed()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use loadngo_weights::mxfp4::{e2m1, e8m0};

    /// `cargo test --release -p kimi-k3-cli -- --ignored --nocapture converted_experts`
    #[test]
    #[ignore = "needs the Kimi Linear checkpoint and its converted experts on Jarraya"]
    fn converted_experts_decode_to_the_evaluated_rounding() {
        let source = Path::new("/Volumes/Jarraya/kimi-linear-48b-a3b-instruct");
        let converted = Path::new("/Volumes/Jarraya/kimi-linear-48b-a3b-instruct-mxfp4-experts");
        let config = LinearConfig::from_path(source.join("config.json")).unwrap();
        let (src, out) = (
            SafeTensorIndex::open(source).unwrap(),
            SafeTensorIndex::open(converted).unwrap(),
        );
        let layers = (0..config.num_hidden_layers)
            .filter(|&l| !config.is_dense(l))
            .count();
        assert_eq!(
            out.tensors().len(),
            layers * config.num_experts * 3 * 2,
            "every expert, both halves"
        );
        let (e, inter) = (config.hidden_size, config.moe_intermediate_size);
        let mut checked = 0;
        for (layer, expert) in [(1, 0), (2, 17), (13, 128), (26, 255)] {
            for (part, (rows, cols)) in PARTS.iter().zip([(inter, e), (inter, e), (e, inter)]) {
                let name =
                    format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.{part}.weight");
                let raw = src.read_raw(src.tensor(&name).unwrap()).unwrap();
                let mut want: Vec<u16> = raw
                    .chunks_exact(2)
                    .map(|p| u16::from_le_bytes([p[0], p[1]]))
                    .collect();
                crate::quality::mxfp4_round_trip(&mut want, cols);
                let blocks = out
                    .read_raw(out.tensor(&format!("{name}.blocks")).unwrap())
                    .unwrap();
                let scales = out
                    .read_raw(out.tensor(&format!("{name}.scales")).unwrap())
                    .unwrap();
                assert_eq!(
                    (blocks.len(), scales.len()),
                    (rows * cols / 2, rows * cols / BLOCK_SIZE)
                );
                for (i, &w) in want.iter().enumerate() {
                    let code = (blocks[i / 2] >> (4 * (i % 2))) & 0xF;
                    let value = e2m1(code) * e8m0(scales[i / BLOCK_SIZE]);
                    assert_eq!((value.to_bits() >> 16) as u16, w, "{name} element {i}");
                }
                checked += want.len();
            }
        }
        eprintln!("{checked} weights decode bit-identically to the evaluated rounding");
    }

    #[test]
    fn quantized_bytes_are_blocks_then_scales_row_by_row() {
        // 2 rows x 64: each row two blocks; row 1 all 1.0 (scale 2^-2, so 4.0: code 6).
        let mut raw = Vec::new();
        for r in 0..2 {
            for i in 0..64 {
                let v: f32 = if r == 0 {
                    if i < 32 { 6.0 } else { 0.0 }
                } else {
                    1.0
                };
                raw.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        let (mut blocks, mut scales) = (Vec::new(), Vec::new());
        quantize(&raw, 64, &mut blocks, &mut scales);
        assert_eq!(blocks.len(), 2 * 32);
        assert_eq!(scales, [127, 0, 125, 125]);
        assert!(blocks[..16].iter().all(|&b| b == 0x77));
        assert!(blocks[16..32].iter().all(|&b| b == 0));
        assert!(blocks[32..].iter().all(|&b| b == 0x66));
    }
}
