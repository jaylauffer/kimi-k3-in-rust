# Kimi Linear 48B-A3B

The interactive model beside K3 (Jay, 2026-09-24: K3's ~1 minute per token is "still too
far from usable"). `moonshotai/Kimi-Linear-48B-A3B-Instruct` is the same KDA + NoPE-MLA
family as K3, with about 3B parameters active per token instead of about 100B.

```sh
~/pudding/launch-kimi-k3.sh                     # chat; the launcher's default model
~/pudding/launch-kimi-k3.sh "The capital of France is" --gen 16
target/release/k3 /Volumes/Jarraya/kimi-linear-48b-a3b-instruct --chat --accel ane
```

`k3` recognises the checkpoint from `model_type = "kimi_linear"` in its `config.json`.
`--accel ane|cpu` and `--cache-gb` apply (default 24 GiB for this model); `--recompute`,
`--layers`, `--pin-layers` and `--ring-slots` are K3's.

## Checkpoint

- `/Volumes/Jarraya/kimi-linear-48b-a3b-instruct`, revision
  `e1df551a447157d4658b573f9a695d57658590e9`, 20 shards, 98.3 GB bf16, 20,493 tensors.
  Downloaded 2026-09-24 with `hf download` pinned to that revision (figures excluded).
- Licence: `license: mit` in the model card's front matter. The repository has no
  separate LICENSE file; check with Moonshot before relying on it commercially.

## Architecture, as implemented (`kimi-k3-core::linear`)

Written from the checkpoint's `modeling_kimi.py` and, for the KDA decay gate it imports
from fla, fla's `naive_kda_gate`. Differences from K3:

| | K3 | Kimi Linear |
|---|---|---|
| Residual | Block Attention Residual aggregation | plain pre-norm residual |
| MLP activation | SiTU-GLU | `SiLU(gate) * up` |
| Experts | MXFP4, in a 3584-wide latent space with down/up projections and a latent norm | bf16, in the hidden space (2304), 256 per layer, 8 active, one shared |
| MLA queries | `q_a` -> norm -> `q_b`, optional output gate | direct `q_proj`, no gate |
| KDA output gate | full rank | low rank `g_b(g_a(x))` |
| KDA decay | `lb * sigmoid(exp(A_log) * (z + dt_bias))`, lb = -5 | `-exp(A_log) * softplus(z + dt_bias)` |

Shared with K3: the KDA recurrence, the fused-SiLU short convolution, L2/RMS norms, the
sigmoid router (selection-only bias, renormalise, scale 2.446), and every bf16 product
through `Matrix::mul_rows`, so the Neural Engine path is the same engine. The config
parser refuses shapes it does not implement (grouped routing, `q_lora_rank`, a gate lower
bound, another activation) rather than defaulting them.

About 2B parameters (~4 GB: attention, shared experts, router, the dense first layer,
embedding and LM head) are resident, loaded in 1-2 s. The 94 GB of routed experts stream
through an LRU cache, each layer's misses fetched as one batch of proactor reads.

## Evidence, 2026-09-24, this Mac mini

- Unit tests: config parsing and refusals, fla's softplus.
- `--accel cpu`, "The capital of France is", 8 tokens: `Paris. The capital of Italy is
  Rome`; 1.0-1.6 s per token.
- `--accel ane`, same prompt, 32 tokens: `Paris. The capital of Italy is Rome. The capital
  of Germany is Berlin. The capital of Spain is Madrid. The capital of the United Kingdom
  is London.` Prompt pass 4.6 s, then 0.65-0.98 s per token (1.33 tokens/s). 31,374 ANE
  predictions; all 29 compiled shapes planned on the NPU; none fell back to the CPU.
  Peak RSS 23.9 GB, mostly the expert cache. No thermal warning recorded.
- Chat through the launcher, piped input: "What is the capital of Japan?" -> "The capital
  of Japan is Tokyo."; the follow-up "And what is its population, roughly?" -> "Tokyo's
  metropolitan area is home to roughly 37 million people, ..."; `/undo` returned the
  context to 27 tokens; the next question ("Name one famous temple in Kyoto") -> "Kinkaku-ji,
  the Golden Pavilion, is a famous temple in Kyoto." Whole replies took 27-48 s: each new
  prompt routes to many experts not yet cached (14.2 MB each).

There is no PyTorch reference on this machine (the reference needs fla's Triton kernels
and more than 98 GB of memory), so correctness rests on coherent, factual output across
prompts plus CPU/ANE agreement, not on logit parity.

## Where the time goes, and what would make it faster

Per decoded token on the ANE path: ~0.37 s in about 980 Core ML predictions (most of
them small expert matrices, where fixed per-call overhead dominates), ~0.23 s converting
the resident bf16 weights to fp16 again (6.4 GB per token), and the rest CPU-side
attention, routing and expert reads. Not yet done, in expected order of payoff:

1. Keep resident weights (and cached experts) as prepared fp16 surfaces instead of
   converting them on every token.
2. Fewer, larger predictions: one fused gate/up product for all eight selected experts,
   and a batched down projection.
3. Warm the expert cache in the background, or pin the most-routed experts.
4. The GPU (Metal) for decode, where memory bandwidth rather than per-call overhead
   limits; 4-bit experts to keep all 256 per layer resident (a numerics decision).

Plans: [loadngo `docs/METAL_COMPUTE_PLAN.md`](https://github.com/jaylauffer/loadngo/blob/dev/docs/METAL_COMPUTE_PLAN.md)
(GPU decode and MXFP4 experts) and
[loadngo `docs/LOCAL_MODEL_CAS_TOOLS.md`](https://github.com/jaylauffer/loadngo/blob/dev/docs/LOCAL_MODEL_CAS_TOOLS.md)
(read-only file access through a signed Archive CAS snapshot, the default capability).
