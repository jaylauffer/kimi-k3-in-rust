# Apple Neural Engine

Status, 2026-09-24 (Claude Code, taking over Codex's bring-up): the released K3 checkpoint
runs on this Mac mini with every dense trunk projection, the LM head and every routed
expert's matrices executed on the Neural Engine through public Core ML APIs, and with
incremental decoding through the trunk ring. It is still slow: about a minute per token.

```sh
~/pudding/launch-kimi-k3.sh            # chat, --accel ane is the launcher default
target/release/k3 /Volumes/Jarraya/kimi-k3 --prompt "..." --gen 8 --accel ane
target/release/k3 ... --accel cpu      # bit-exact CPU reference (the CLI default)
target/release/k3 ... --recompute      # old full-recompute path, for comparison
```

## What runs where

| Work | Device |
|---|---|
| bf16 trunk projections (KDA q/k/v/b/f/g/o, MLA q_a/q_b/kv_a/kv_b/g/o, latent down/up, shared expert, dense MLP), LM head | ANE, all positions of a pass in one product per matrix |
| Routed experts (MXFP4 w1/w3/w2) | ANE, each expert once over the tokens that selected it |
| bf16 and MXFP4 to fp16 expansion into IOSurfaces | CPU (NEON) |
| Routing, KDA recurrence, MLA attention, norms, activations, residual aggregation | CPU |
| Weight streaming (trunk ring, expert cache) | loadngo proactor, as before |

The device code is loadngo's `coreml::dense::DenseEngine` (see
`loadngo/docs/NPU_ACCELERATION.md` for how it works and why weights go through
IOSurfaces). Kimi's side stays platform-agnostic and free of unsafe code: `kimi-k3-core`
defines `layer::DenseAccel`, `Matrix::mul_rows` and `Mxfp4Matrix::mul_rows`, and the CLI's
`accel.rs` adapts the trait to the engine on macOS. With no accelerator, every batched
product is exactly one per-row CPU product, so the CPU path is unchanged float for float
(all fixture, oracle and prefill-parity gates pass).

## Measured on the released checkpoint

Same command each time: `--prompt "The capital of France is" --pin-layers 2 --ring-slots 2`,
all 93 layers, Mac mini M4 Pro, checkpoint on the external PCIe drive `Jarraya`
(3.5 GB/s sequential, measured with `F_NOCACHE`).

| Configuration | Prompt pass (5 tokens) | Each further token |
|---|---|---|
| CPU, full recompute (2026-09-23 baseline) | 224.6 s | ~250 s, growing with context |
| ANE trunk, full recompute, first fp16 conversion | 184.4 s | 203.8, 225.5 s |
| ANE trunk, NEON conversion, full recompute | 165.0 s | 185.0, 205.3 s |
| ANE trunk, incremental decode | 163.7 s | 72.7-73.7 s (5 steps) |
| ANE trunk and experts, incremental decode | **106.4 s** | **63.1-63.9 s** (5 steps) |

Every run produced the same tokens: `Paris.",` then `+` then `            "The`, the
same as the CPU baseline. The last run pushed 888 GB of weights through 46,480 ANE
predictions; Core ML planned all 41 compiled shapes on the Neural Engine and no product
fell back to the CPU. It used 173 s of CPU time for 6 tokens, against 381 s for 3 tokens
on the trunk-only ANE run: most of the arithmetic has left the CPU, which also means much
less heat per token. macOS recorded no thermal or performance warning. Peak RSS rose to
18.5 GB (about 8 GB on the CPU path); that growth is not yet explained.

`ring_feed_matches_full_recompute_token_by_token` (`tests/real_checkpoint_trunk.rs`,
ignored, needs the checkpoint) checks incremental decoding on five real layers covering
dense, `MoE`, KDA and MLA: a three-token prefill and then single tokens, each step's CPU
logits bit-identical to full recompute.

### Where a CPU full-recompute pass went

A 25-second `sample` of the main thread during the 224.6 s baseline pass: 38% scalar
MXFP4 expert products, 30% scalar bf16 products, 21% waiting on disk reads, 10% weight
unpacking. The ANE removed the first two; the remaining floor is I/O.

## Why it is still about a minute per token

The trunk is 109 GB of bf16 and this machine has 64 GB, so every token streams nearly
all of it from the drive: at 3.5 GB/s that is ~31 s before any expert read. The ANE
consumes weights at 24-29 GB/s, eight times faster than they arrive. No accelerator
changes this floor; only less data per token can:

- pinning more layers in RAM (each 1.17 GB pinned saves ~0.33 s/token, at a memory risk
  already seen once at `--pin-layers 20`);
- a smaller trunk on disk (int8/fp8/4-bit), which changes the model's numerics and is
  Jay's decision, not an engineering default;
- a faster drive, or the checkpoint on internal storage.

## Numerics

The ANE computes in fp16. bf16 weights convert exactly except below fp16's normal range
(rounded as hardware does); MXFP4 values expand exactly when the scaled value is
fp16-normal. Activations are rounded to fp16 on the way in. Anything outside fp16's
finite range is refused and that one product runs on the CPU (counted in the run summary;
zero so far). Agreement so far is token-level on one prompt, not a logit-level or
quality evaluation; `--accel cpu` remains the reference.

## Not done

- Chat mode's session reuse (feed only new tokens; rebuild after `/undo`, `/reset` or a
  cancelled pass) compiles and follows the tested `feed` contract, but has not been run
  live against the full model.
- Logit-level ANE-vs-CPU comparison on the full model, and an answer-quality check.
- Overlapping fp16 conversion with ANE execution (async prediction).
- The 18.5 GB peak RSS.
- `k3-ane` (Codex's probe binary) is unchanged; it still benchmarks one baked projection.
