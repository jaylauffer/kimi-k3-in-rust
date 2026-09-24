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

### Per-token costs, fixed 2026-09-24

Measured back to back on `--accel ane`, "The capital of France is", with the same output
text before and after (the CPU path also gave identical tokens). "After" is two runs, the
second on the committed build; macOS thermal state stayed nominal throughout:

| Measure | Before | After |
|---|---|---|
| Decode speed, 48 tokens | 1.31 tokens/s | 1.68-1.75 tokens/s |
| Steady state, last 24 tokens | 0.70 s/token | 0.51-0.55 s/token |
| Time converting weights | 11.4 s | 4.1 s |
| CPU user time | 18.1 s | 13.2 s |
| 639-token prompt pass | 88 s (91 s on the older build) | 58-59 s |
| Page faults in that pass | 14.0 million | 6.9-7.2 million |
| Kernel time in that pass | 49 s | 27-28 s |

What changed:

1. **Conversion overlaps the Neural Engine.** The forward pass hands over every
   product whose input is ready at once, through `DenseAccel::run_bf16`:
   - KDA's six input projections;
   - both low-rank second halves;
   - the shared and routed experts' gate and up projections, then their down
     projections.

   loadngo's engine converts the next weight on a helper thread while the current one
   runs.
2. **No allocation churn on cache misses.** Read buffers come from a bounded pool, misses
   are read in batches of 32, and an evicted expert's storage is overwritten in place.
   This is the rule in loadngo `docs/PROACTOR_ENGINE_ADOPTION.md`, "Allocation churn in
   hot paths is not acceptable".

Tried and rejected, with measurements in loadngo `docs/NPU_ACCELERATION.md`:

- **fp16 weights prepared once as their own surfaces.** No faster: the Neural Engine is
  slow on memory it has not read recently.
- **Fewer, larger predictions.** Slower: call count was not the cost.

### Chat opening: tool declarations read once per launch

With file tools on, every conversation opens with the same `tool_declare` message and
guidance, 819 tokens. Chat used to process them before the first question and again
after every `/reset`. It now consumes them once, right after loading, and keeps a
snapshot of the session (a `LinearSession` clone, under 1 GB with the 4096-token
context). The first question and every `/reset` start from the snapshot.

Measured through `launch-kimi-k3.sh`, piping "What is the capital of Japan?", `/reset`,
"What is the capital of France?", thermal state nominal:

- the snapshot is ready 66 s after launch;
- the first reply, "The capital of Japan is Tokyo.", took 21.2 s, against 75 s before;
- the reply after `/reset`, "The capital of France is Paris.", took 17.9 s.

The 66 s is paid once per launch. Saving the snapshot to disk, keyed by checkpoint,
accelerator and declaration, would remove it too; that is not built.

### 4-bit (MXFP4) routed experts: quality, measured 2026-09-25

No checkpoint has been converted. `--experts mxfp4` rounds each routed expert to OCP MX
v1.0 MXFP4 as it enters the cache and writes it back as bf16. bf16 holds every MXFP4
value exactly, so the forward pass computes what a converted checkpoint would.

`--compare mxfp4 --prompt-file F` scores a text twice in one process: bf16 experts,
then MXFP4 experts. Both runs use `--accel ane`, so the only difference is the experts.
`--compare cpu` measures the Neural Engine's own fp16 effect against the CPU reference,
both bf16, as a baseline.

The encoder is `loadngo-weights` `mxfp4::quantize_block`: the spec's shared scale,
round-to-nearest-even, clamped at +-6. Every run was paced, and macOS thermal state
was 0 (nominal) before each one.

| Text | Tokens | Perplexity, bf16 -> MXFP4 | Top-1 agreement | Mean KL (nats) | Next token correct |
|---|---|---|---|---|---|
| English prose (loadngo README) | 552 | 21.24 -> 21.56 (+1.5%) | 91.5% | 0.020 | 48.3% -> 47.4% |
| Rust code (loadngo `proactor/src/lib.rs`) | 627 | 4.02 -> 4.10 (+2.0%) | 96.2% | 0.016 | 70.9% -> 70.8% |
| Chinese (《桃花源记》 + a modern paragraph) | 318 | 2.08 -> 2.11 (+1.5%) | 97.8% | 0.011 | 85.5% -> 85.5% |
| README start, same 130 tokens as the baseline | 130 | 75.97 -> 77.15 (+1.6%) | 88.4% | 0.030 | 30.2% -> 28.7% |
| Baseline, CPU vs Neural Engine, both bf16 (README start) | 130 | 78.12 -> 75.97 (-2.8%) | 93.0% | 0.012 | 29.5% -> 30.2% |

What the numbers say:

- **4-bit experts cost about 1.5-2% perplexity.** The chance of predicting the actual
  next token changed by at most 0.9 points.
- **The change is larger than the fp16 Neural Engine path already makes.** On the same
  130 tokens, MXFP4 moved the distributions about 2.5 times as far as fp16 does
  against the CPU (mean KL 0.030 against 0.012; top-1 agreement 88.4% against 93.0%).
  Across the longer texts the top choice agrees at 91.5-97.8% of positions.
- **Caveats.** The Chinese passage is famous and likely memorised, so it is an easy
  case. Perplexity does not measure answer quality over a long generation. The
  listening test (below) is not run yet.

Checks:

- Rerunning the Chinese comparison after the encoder was rewritten branchless reproduced
  every number exactly.
- That rewrite made rounding an expert about 3x faster: the 4-bit pass took 125 s
  instead of 390 s.

Speed was measured after converting (next section).

Listening test, ready to run (bf16 then MXFP4, the same four questions, two in English
and two in Mandarin, with `/reset` between them):

```
k3 <kimi-linear-dir> --chat --accel ane --no-tools --gen 160 --experts bf16  < questions.txt
k3 <kimi-linear-dir> --chat --accel ane --no-tools --gen 160 --experts mxfp4 < questions.txt
```

The Mandarin answers can be played aloud with `say -v Tingting "<text>"`.

### 4-bit experts in use (2026-09-25): converted, resident, measured

Jay decided to convert. `k3 <dir> --convert-experts-mxfp4 <out>` wrote
`/Volumes/Jarraya/kimi-linear-48b-a3b-instruct-mxfp4-experts/`:

- one safetensors file per `MoE` layer, 25.0 GB in all;
- read 94.2 GB of bf16 and wrote it in 124 s;
- `CONVERSION.json` records the source revision, the encoder commit (loadngo
  `0d3a5203`) and SHA-256 of every file;
- the original checkpoint is untouched.

An ignored test (`converted_experts_decode_to_the_evaluated_rounding`) confirms:

- all 39,936 tensors are present;
- 28.3 M sampled weights decode bit-identically to the `--experts mxfp4` rounding the
  quality above was measured with.

`--mxfp4-experts <out>` (the launcher passes it whenever the directory exists) reads
the experts from there. They load at launch in about 8 s and all 6,656 stay resident
in the default 24 GiB cache. The engine expands MXFP4 to fp16 on its helper thread,
overlapped with the Neural Engine, as it does for bf16.

Measured on the same prompts as the bf16 numbers above, with thermal state 0
throughout:

| Measure | bf16 experts | MXFP4 experts |
|---|---|---|
| Decode, 48 tokens | 1.75 tokens/s | 2.61 tokens/s |
| Steady state, last 24 tokens | 0.51 s/token | 0.38 s/token |
| 639-token prompt pass | 58 s | 21.5 s |
| Kernel time in that pass | 28 s | 11 s |
| Launcher: launch to ready | ~66 s | 36 s |
| First one-sentence reply | 21.2 s | 6.0 s |
| Reply after `/reset` | 17.9 s | 5.5 s |

Notes:

- The decode text was identical to the simulated 4-bit run.
- Memory: 31.6 GB peak footprint.
- `KIMI_BF16_EXPERTS=1` makes the launcher use the bf16 experts.

### What limits it now

- **Decoding** is about 0.4 s/token of Neural Engine time, spent streaming ~6 GB of
  weights at 15-25 GB/s, plus about 0.15 s of single-threaded CPU work: attention,
  routing, and expert reads on cache misses.
- **Prompt processing** is bound by reading experts. A few hundred tokens route to
  nearly every expert in every layer, 86 GB for 639 tokens, and the 24 GB cache holds
  about a quarter of them. That is what makes tool replies slow.

Next, in expected order of payoff:

1. **4-bit (MXFP4) routed experts,** about 25 GB, so all of them stay in memory. This
   removes the drive from both prompt processing and decoding, and it helps the Neural
   Engine path as well as Metal. The engine already multiplies MXFP4. It changes the
   model's numbers, so it waits on a quality report and Jay's decision.
2. **The GPU (Metal) for decoding,** where memory bandwidth is roughly ten times the
   Neural Engine's streaming rate.
3. **Multi-threaded CPU work,** for the router and MLA attention in long prompts.

Plans: [loadngo `docs/METAL_COMPUTE_PLAN.md`](https://github.com/jaylauffer/loadngo/blob/dev/docs/METAL_COMPUTE_PLAN.md)
(GPU decode and MXFP4 experts) and
[loadngo `docs/LOCAL_MODEL_CAS_TOOLS.md`](https://github.com/jaylauffer/loadngo/blob/dev/docs/LOCAL_MODEL_CAS_TOOLS.md)
(read-only file access through a signed Archive CAS snapshot, the default capability).
