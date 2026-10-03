# Gemma 4 31B-it

Google's dense Gemma 4 (`google/gemma-4-31B-it`, `model_type = "gemma4"`) as a chat
model beside Kimi Linear, on the same loadngo stack: shard reads through the loadngo
proactor, weights resident in GPU memory and products on loadngo's Metal kernels
(`loadngo-metal-compute`), and the same chat loop, tools, transcripts, context handoffs
and System One checkpoints. Text only: the vision tower is not loaded.

```sh
~/pudding/launch-kimi-k3.sh --gemma                         # chat
~/pudding/launch-kimi-k3.sh --gemma "The capital of France is" --gen 16
target/release/k3 /Volumes/Jarraya/gemma-4-31b-it --mxfp4 /Volumes/Jarraya/gemma-4-31b-it-mxfp4 --accel gpu --chat
```

`k3` recognises the checkpoint from its `config.json`.

## Checkpoint and conversion

- `/Volumes/Jarraya/gemma-4-31b-it`: 2 shards, 62.5 GB of bf16, copied 2026-10-03 from
  Jay's download on Loadngo Archive Staging (`gemma-4-31B-it`; `rsync` checksums
  verified): 1,188 tensors, 832 of them the text model (`model.language_model.`).
- Licence: `license: apache-2.0` in the model card's front matter (terms at
  https://ai.google.dev/gemma/docs/gemma_4_license). `tests/fixtures/gemma4/config.json`
  is the checkpoint's own config.
- The bf16 weights are more than this Mac mini's 64 GB of memory, so the decoder's seven
  matrices per layer are converted to OCP MXFP4, the format of Kimi Linear's experts:
  `k3 <dir> --convert-mxfp4 <out>` wrote `/Volumes/Jarraya/gemma-4-31b-it-mxfp4/`, one
  file per layer, 15.6 GB, from 58.6 GB of bf16 in 79 s. `CONVERSION.json` records
  the source and the encoder. `--quantize mlp` converts only the MLPs (attention stays
  bf16, about 31 GB in all). The embedding (also the LM head, tied) and the norms
  are read from the checkpoint as bf16.
- Resident: 2.8 GB bf16 plus 15.6 GB MXFP4, loaded into GPU memory in 2-7 s.

## Architecture, as implemented (`kimi-k3-core::gemma`)

Written from transformers' `modeling_gemma4.py` and `modeling_rope_utils.py`:

- the embedding row times `sqrt(5376)` rounded to bf16 (73.5);
- per layer: `h += post_attn_norm(attn(input_norm(h)))`, then
  `h += post_ff_norm(mlp(pre_ff_norm(h)))`, then `h *= layer_scalar`;
- RMSNorm multiplies by its weight directly; `v_norm` has no weight;
- attention: per-head `q_norm`/`k_norm` before rotary, scale 1, grouped heads. 50
  sliding layers (window 1024, 256-wide heads, 16 KV heads, rotary theta 10,000) and
  10 full layers (512-wide heads, 4 KV heads whose values are the key projection
  before `k_norm`, proportional rotary: the first 64 of 256 frequency pairs at theta
  10^6, the rest unrotated);
- MLP `down(gelu_tanh(gate) * up)`; logits `30 * tanh(logits / 30)`.

The config parser refuses what is not implemented (mixture of experts, per-layer
inputs, shared KV layers, attention bias, another activation or rotary type).

Prompts are fed in passes of at most 512 positions. A sliding layer's keys and values
live in a ring of 1024 + 512 slots, so memory for them does not grow with the
conversation; a full layer keeps every position (16 KB per position per layer as f32).

Attention runs on the GPU in two new loadngo kernels (`attention_grouped` for decoding,
`attention_grouped_tiled` on the matrix units for passes of 32 or more positions; see
loadngo `docs/METAL_COMPUTE_PLAN.md`), with the layer's keys and values kept in GPU
memory and only each pass's new rows copied in. The ring has `sliding_window + 512`
slots rounded up to a multiple of 32 (1,536), so the tiled kernel never reads across
its end. The CPU reference is used without a GPU.

## Tokenizer and chat format

- `kimi-k3-core::spm_bpe`: the `tokenizer.json` BPE (spaces as `▁`, merges by rank,
  byte fallback), behind the same `Tokenizer` the chat uses.
- `kimi-k3-cli::chat_gemma`: the checkpoint's `chat_template.jinja` with thinking off:
  `<|turn>role\n…<turn|>\n`, the reply opened with an empty thought channel, tool
  declarations in the system turn, calls as `<|tool_call>call:NAME{…}<tool_call|>`,
  results as `response:NAME{value:<|"|>…<|"|>}<tool_response|>` inside the model's turn.

## Evidence

- Tokenizer: token for token with `tokenizers` 0.23.2 on 84 texts, 5,700 tokens
  (`tests/gemma_tokenizer_parity.rs`).
- Chat opening: the system turn, all 30-odd chat tool declarations, a user turn and the
  reply header match the template rendered by Jinja2 as transformers does, 2,705
  tokens (`chat_gemma::template`, fixture from `scripts/gemma_template_fixture.py`).
- Forward pass: a random-weight model with the 31B's structure at toy size against
  transformers' `Gemma4ForCausalLM` (5.19.0.dev0, float32 eager): logits within 2e-3,
  fed whole, in pieces and one token at a time over 600 tokens, so the window, the ring
  and every pass boundary are exercised (`tests/gemma_tiny_oracle.rs`, fixture from
  `scripts/gemma_tiny_oracle.py`). Three deliberate bugs (values after `k_norm`, no
  layer scalar, a window one too wide) each fail it by 0.3-2.1. The same oracle on
  the GPU path (`gemma::tests::gpu_matches_the_transformers_oracle`) passes with every
  product and attention on the GPU.
- `attention_grouped` against a float64 reference on the 31B's shapes, including ring
  wrap-around and 512-position passes: max error under 2e-5 (loadngo
  `metal-compute/tests/attention_grouped.rs`).
- On the real checkpoint (MXFP4), 2026-10-03, thermal state nominal:
  - "The capital of France is" -> " Paris." then end of text; decode 6.7 tokens/s,
    weights read at 179 GB/s, 0 steps to the Neural Engine.
  - Chat, no tools: "What is the capital of Japan?" -> "The capital of Japan is Tokyo.";
    the follow-up on its population -> "roughly 37 to 38 million"; `/undo`, then "Name
    one famous temple in Kyoto" -> "Kinkaku-ji, also known as the Golden Pavilion."
  - Chat with tools, asked to read `docs/KIMI_LINEAR.md` and report the GPU path's
    decode speed and the file read: it called `fs_read{path: "docs/KIMI_LINEAR.md"}`,
    then `fs_read{line_start: 280, …}` for the rest, and answered "14-20 tokens/s for
    short context (and up to 32 tokens/s in later updates). I read the file
    `docs/KIMI_LINEAR.md`." That run took 16.4 minutes on the first attention kernel
    (the 2,581-token opening alone 122 s). Rerun on the final kernels, the same calls
    and the same answer took 5.2 minutes: opening 59 s, then replies of 8.9 s, 172 s
    (reading the 6,000-token file) and 70 s.
- Speed on a 6,014-token prompt (this file's sibling `KIMI_LINEAR.md`), measured after
  each change:

  | | Prompt | Tokens/s | Next token at 6k context |
  |---|---|---|---|
  | First attention kernel | 373-392 s | 15-16 | 1.0-1.1 s |
  | Attention split over 8 simdgroups (decoding) | 392 s | 15 | 0.26-0.37 s |
  | Prompt attention on the matrix units | 176 s | 34 | 0.26-0.39 s |
  | GELU split across the CPU cores | 141 s | 43 | 0.26-0.39 s |

  In the last run products took about 93 s of GPU time (about 4.3 TFLOPS) and
  attention about 34 s. Decoding at short context is 6.7 tokens/s (0.15 s per token,
  weights read at 179 GB/s).

Quality of the 4-bit conversion against bf16 is not measured yet: the bf16 weights do
not fit in memory. `--mxfp4 A --compare-with B --prompt-file F` compares two
conversions (for example `--quantize all` against `--quantize mlp`).

## The Neural Engine

Nothing in this path uses it yet. With `--accel gpu` every weight is in GPU memory, so
the GPU never hands work to the Neural Engine. Decoding a dense 31B reads every weight
for every token, and `loadngo-coreml`'s weight-streaming engine reads at 24-29 GB/s
against the GPU's ~180-228 GB/s, so the Neural Engine is a poor fit for decoding.

A prototype for attention exists (2026-10-03, at Jay's request): loadngo-coreml's
`AttentionEngine` runs Gemma's grouped attention on the Neural Engine, with each
prediction's completion delivered through the loadngo proactor. On a full window
(a 512-position pass over the 1,536-slot ring), it takes 9.75 ms against the GPU kernel's
21.9 ms. It runs concurrently with GPU work: 79 ms together against 141-144 ms in
sequence. Errors are fp16-sized (RMS ~1.5e-3). Single tokens are slower than on the
GPU. Figures and the four steps an integration would take are in loadngo
`docs/NPU_ACCELERATION.md`, "Grouped-query attention on the Neural Engine, through the
proactor".

## Next

1. One GPU submission per layer, the GELU on the GPU inside it (now about 5
   submissions per layer, each ~0.4-1.3 ms of overhead when decoding).
2. Prompt products are about 4.3 TFLOPS of fp32; the remaining CPU work in a pass
   (norms, rotary, copies) is next after that.
3. Save the opening's session to disk, keyed by checkpoint and declaration.
4. Measure 4-bit quality (`--compare-with` between conversions).
