# Rust Port

**2026-09-24 update:** [Local interactive chat](CHAT.md) documents the new
Loadngo-backed conversation path and its explicit performance/acceptance limits.
The older "base model" interpretation below is incorrect: the downloaded
checkpoint has a structured chat encoder with preserved thinking history.
The older download-not-started and CLI-still-open paragraphs are historical;
the checkpoint and CLI are present locally. Do not treat them as current blockers.

The repository retains the C implementation as the behavioral oracle while a
clean Rust implementation is built beside it. The first Rust release must not
silently call into C: portability and memory safety are the point of the port.

## Compatibility contract

For every implemented subsystem, Rust must consume the same fixture and either
produce the same output or reject the same invalid input as C. A new Rust
implementation is not considered complete merely because it compiles.

| C subsystem | Rust destination | Parity gate |
|---|---|---|
| `include/k3/k3_cfg.h` | `kimi-k3-core::config` | flat/nested configs plus every negative config fixture |
| `src/io/k3_st.c` | `kimi-k3-core::safetensors` | existing dtype, offset, escaped-name, and short-read fixtures |
| `src/cache/k3_cache.c` | `kimi-k3-core::expert_cache` | prefetch, eviction, and mixed access tests |
| `src/io/k3_trunk.c` | `kimi-k3-core::trunk` | one-slot guard and streaming-concurrency tests |
| `src/core/k3_ops.c` | `kimi-k3-core::ops`, `kimi-k3-core::layer` | the existing numeric fixture manifest and full-size scale test |
| `tests/unit/k3_model.c`, `src/model/k3_bind.c` | `kimi-k3-core::model` | tiny-model teacher forcing, greedy, and incremental oracle |
| `src/cli/k3_run.c` | `kimi-k3-cli` | CLI result and memory-budget parity |

## Implemented slices

`kimi-k3-core::config` is a strict reader for both checkpoint JSON shapes. It
collects missing required fields, preserves the C reader's optional-boolean
defaults, keeps the checkpoint's one-based MLA layer map explicit, and enforces
the same structural limits (`MAX_TOPK`, at least one KDA layer, valid layer
positions, positive convolution and residual block sizes).

The port intentionally starts here because a permissive reader can construct a
model that loads and emits plausible text while executing the wrong architecture.

`kimi-k3-core::safetensors` now indexes every `*.safetensors` shard in stable
lexical order and provides exact-name lookup plus raw and bounded `f32` reads.
Its header visitor processes one tensor entry at a time instead of building a DOM for
the released checkpoint's large JSON header. It rejects inconsistent byte spans and
out-of-range tensor data before a model can bind it. The current fixture gate covers
both shards, escaped names, rank zero through four, empty tensors, and bit-exact BF16
and F16 special values/subnormals.

`kimi-k3-core::expert` resolves the six U8 tensors for a routed MXFP4 expert,
validates its packed/scaled geometry, and checks whether its spans form one contiguous
shard range. Normal K3 experts therefore load with one bounded read; a structurally
valid repacked checkpoint falls back to the canonical six-read layout. The Rust gate
uses the C cache fixture to verify real contiguous layout, per-expert byte identity,
and a missing-expert refusal.

`kimi-k3-core::cache` now provides the bounded packed-byte LRU contract: its capacity
reserves the C direct-I/O slot stride, it refuses an under-sized top-k working set,
evicts only unpinned entries, records histogram/trace data, and separates batch-prefetch
disk reads from demand hits. The current batch reader is deliberately sequential while
the no-aliasing and accounting gates are established; the later direct-I/O parallel
implementation must preserve these public semantics and fixture tests.

`kimi-k3-core::ops`, `layer` and `model` run the whole model on resident fp32 weights:
RMSNorm, SiTU-GLU, `ShortConv`, the KDA decay and recurrence, the router, Block
Attention Residuals and the matmul kernel; the KDA and Gated MLA layers (MLA with a KV
cache), Stable `LatentMoE`, the dense MLP and the decoder layer; then embedding, the
model-level aggregator, final norm and head. Each kernel keeps the C scalar path's
summation order, fused products and float/double conversions, so results match C to
the bit rather than to a tolerance.

- `tests/ops_fixtures.rs` runs every `tests/fixtures/ops` fixture at the manifest
  tolerance, including the `ShortConv` continuation, per-head `A_log`, router set and
  unbiased weights, and both decoder-layer fixtures (a non-boundary KDA layer and an
  MLA layer on a block boundary).
- `tests/tiny_oracle.rs` holds the tiny 13-layer checkpoint to `ref_k3.json` exactly, as
  `k3_model.c` does: 20/20 teacher-forced positions, bit-identical logits with one reused
  KDA state slot, 20/20 greedy tokens by full recompute, and 20/20 incremental tokens.
  It goes one step further than the C gate: every incremental step's logits must be
  bit-identical to the full forward's at that position.
- Bit parity with C, checked 2026-09-15 by hashing the gate-1 logits (all 32 positions
  x 256, FNV-1a) from both engines on the same machine, C built with the Makefile's
  `-O3 -mcpu=native -ffp-contract=off`:
  - M4 Pro Mac mini (Apple clang, Apple libm): `68a4648a958e36a1` from both.
  - `agnes`, Raspberry Pi 4, Debian aarch64 (gcc 14.2, glibc): `de363746ad9bf8ea` from
    both, with all 37 Rust tests passing there.

  The two platforms differ from each other and each engine agrees with the other on
  both, which places the difference in the platform's `expf`/`tanhf` rather than in the
  port. The committed gates are therefore the exact-token ones, which hold everywhere.

The released checkpoint's two weight formats have their kernels in `ops` too:
`matmul_bf16` for the bf16 trunk (widened on read), and `mxfp4_dequant` plus
`matmul_mxfp4`, which multiplies straight out of packed MXFP4 so a routed expert is
never widened to fp32. `tests/weight_formats.rs` gates them as the C suite does:

- `matmul_bf16` is bit-identical to `matmul` on the same values, over generated bf16
  patterns that include denormals and huge exponents, at a 257x129 shape that exercises
  every tail.
- `mxfp4_dequant` is exact on `tests/fixtures/mxfp4.json`, released K3 expert bytes
  (64 x 3584), and differs from the swapped-nibble order the fixture records.
- `matmul_mxfp4` meets the C contract of relative error below 1e-6 against
  dequantise-then-matmul on those bytes (measured 0), handles a short final group, skips
  a NaN (255) scale group, and refuses an odd input width.
- Bit parity with C on the Mac mini, same generator and inputs: `matmul_bf16` FNV-1a
  `eea07659fdebcb0e` and `matmul_mxfp4` `ee6d38bf1b9f4046` from both engines. The Rust
  MXFP4 kernel reproduces the C scalar/NEON summation order; the C AVX2 path uses a
  different one and is bound to it only by the 1e-6 contract.

Those formats are wired through the model. Every weight read only through a matmul is a
`layer::Matrix` tagged `F32` or `Bf16` (the embedding row gather included); weights read
elementwise (norms, conv kernels, `A_log`, `dt_bias`, router gate and bias) stay fp32,
the same split as `k3_bind.c`'s `reqw`/`reqn`, but tagged per matrix rather than per
struct. A MoE layer's routed experts are either `RoutedExperts::Resident` fp32 banks or
`Streamed` MXFP4 from a `layer::ExpertSource`, which the model's entry points take; the
engine's source is `cache::CachedExperts`, the `ExpertCache` over the checkpoint shards.
An expert that cannot be fetched is an `ExpertFetchError` that stops the token (and marks
an incremental `Session` unusable) instead of C's counted drop.
`tests/tagged_weights.rs` gates the wiring on the tiny checkpoint:

- A bf16 trunk gives logits bit-identical to the same values bound as fp32, through
  both the full forward and incremental decode.
- Experts quantised to MXFP4 and streamed through a source match a resident fp32 bank of
  their exact dequantisation: same argmax everywhere, measured bit-identical logits (this
  gate predates prefill batching below and still passes unchanged under it, since
  batching only changes fetch order and count, never the arithmetic). The streamed
  model's incremental decode reproduces its own full forward bit for bit.
- A streamed layer with no source is an error at its layer, and a session refuses
  further tokens after one.

`CachedExperts` is gated in `cache.rs` against direct loads of the cache fixture under
eviction pressure and batch prefetch: identical packed bytes, scales and products.

Not ported yet: preallocated scratch. Threading is out of scope deliberately,
per Jay's direction to use the loadngo proactor instead (see `trunk`'s own
section below). Binding the released checkpoint's safetensors names, the
trunk ring, wiring the real streamed-expert cache into a forward pass, and
prefill expert batching are all done -- see below.

### Prefill expert batching, done 2026-09-23

`layer::moe_prefill`, ported from `k3_moe_prefill`/`moe_prefill_chunk` in
`src/core/k3_ops.c`: for `t > 1` streamed-expert tokens processed together (a
prompt prefill), routes and down-projects every token first, then fetches each
UNIQUE expert across the whole chunk exactly once and applies it to every
(token, slot) that selected it, instead of fetching an expert again for every
token that happened to pick it. Chunked at 64 tokens (matching C's own
`CHUNK`) so the contribution buffer stays bounded regardless of prompt length.
Falls straight through to the existing per-token `moe` for a single token or
resident experts, where there is nothing to dedup -- `decoder_layer`'s one MoE
call site goes through `moe_prefill` unconditionally now, so `Model::forward`,
`TrunkRing::forward`, and `Session::feed` (whose own `T = 1` calls hit the
same fallback) all get this with no caller change.

Bit-identical to calling `moe` once per token, not just faster: the existing
tiny-checkpoint streamed-expert gate (`tagged_weights.rs`) now exercises this
path (a multi-token forward through streamed MXFP4 experts always did) and
still reports `maxrel 0.000e0, bit-identical: true` against the resident
reference. That gate's fetch-count assertion had assumed no deduplication
ever happens -- true before this, false by design after -- fixed to check
that `fetched` is a nonzero upper bound and always equals `prefetched` (both
count the same deduplicated set), instead of an exact count that depended on
there being no dedup to begin with. Also re-verified on the real checkpoint:
`real_checkpoint_trunk.rs`'s streamed-expert-cache test exercises this path
for real three-token, three-MoE-layer input and still gets bit-identical
ring-vs-direct logits.

### Real-checkpoint tensor names, confirmed 2026-09-20

`~/k3model/` holds the released checkpoint's metadata only (`config.json`,
`model.safetensors.index.json`, the tokenizer files) fetched from
`moonshotai/Kimi-K3` at commit `f831ab66814297da540d832a5235f8e904f29d06` and
verified sha256-exact against the values Hugging Face's API reports for
each file -- not the 1.56 TB of weight shards themselves. Storage now
exists: the external NVMe formerly earmarked for this (previously "Untitled",
1.1 TiB of unrelated Windows/personal data) was reclaimed and wiped
2026-09-21/22, and is now an empty 2 TB APFS volume named `Jarraya`,
mounted at `/Volumes/Jarraya` on the Mac mini (~2.0 TB free, confirmed via
`diskutil`). That fits the 1.56 TB checkpoint plus 109 GB packed trunk with
room to spare. The actual download from `moonshotai/Kimi-K3` at the pinned
commit above has not started -- this only resolves where it will land.

Read against the index's 497,220 real tensor names, `k3_moe`
(`src/core/k3_ops.c:537-656`, itself verified there against
`modeling_kimi_linear.py:815-838`) resolves what first looked like two
competing representations of the routed-expert weights into one coherent
picture -- there is no discrepancy to design around, just two distinct
pieces `k3_bind.c` already names correctly:

- **One shared per-layer bottleneck**, applied identically regardless of
  which experts are routed to: `block_sparse_moe.routed_expert_down_proj.weight`
  (H=7168 -> latent=3584, `w->down`), `routed_expert_norm.weight` (RMSNorm
  of the *summed* expert output, never per-expert, `w->latent_norm`), and
  `routed_expert_up_proj.weight` (latent -> H, `w->up`). Plain matrices --
  these bind as ordinary `layer::Matrix` (`Bf16` or `F32`), the same as
  every other non-expert weight already wired through the model.
- **896 per-expert MXFP4 cores**, each entirely inside that 3584-dim
  latent space (3584 -> `moe_intermediate_size`=3072 -> 3584):
  `block_sparse_moe.experts.<id>.{w1,w2,w3}.{weight_packed,weight_scale}`.
  This is exactly the six-tensor-per-expert shape `kimi-k3-core::expert`
  and `cache::CachedExperts` already implement and gate against the C
  cache fixture -- nothing new needed there.
- A separate, always-on **shared expert** (`num_shared_experts`=2, fused
  into one wider MLP, intermediate `moe_intermediate_size * n_shared`
  =6144) runs on the *original*, non-latent-projected input and is added
  with no routing weight at all: `shared_experts.{gate_proj,up_proj,down_proj}.weight`.
  Also plain matrices, not expert-indexed.

Real config values behind the numbers above, from `text_config` in the
downloaded `config.json`: `hidden_size`=7168, `routed_expert_hidden_size`
(the latent width)=3584, `moe_intermediate_size`=3072, `num_experts`=896,
`num_experts_per_token`=16, `num_shared_experts`=2,
`latent_moe_use_norm`=true (matches `k3_cfg.h`'s `latent_norm` field
exactly).

### Real-checkpoint binding, done 2026-09-23

`kimi-k3-core::bind` reads the checkpoint's real `language_model.model.layers.N.*`
names (the three shared per-layer matrices above, plus every other trunk weight)
into `LayerWeights`/`Model`. `BoundStorage` separates reading (`load_top_level`,
`load_layer`, owned buffers) from borrowing (`layer_weights`, `model`, pure
lookups with no further I/O), so a caller can bind one layer or a small window
without the whole ~109 GB packed trunk resident at once -- nothing in this lab
has that much free RAM. A `MoE` layer's routed experts always bind
`RoutedExperts::Streamed`; 896 experts per layer is never resident.

Validated directly against the real checkpoint on `/Volumes/Jarraya` (not just a
synthetic fixture, now that storage and the full download both exist):
`cargo test -p kimi-k3-core --test real_checkpoint_binding -- --ignored --nocapture`
indexes all 497,220 real tensors across 96 shards in ~100ms, reads the ~4.7 GB
embedding table and LM head correctly, and binds layers 0 (dense) and 1 (first
MoE) with every shape matching `config.json`. That run caught two real bugs
before landing: `read_raw`'s single unchunked read fails with EINVAL past
roughly 2 GiB (`embed_tokens`/`lm_head` are each ~2.35 GB), fixed by chunking
the bf16 read the same way `read_f32` already does; and `lm_head.weight` was
initially loaded under the wrong, prefixed name, since the checkpoint stores it
one level up from every other tensor (`language_model.lm_head.weight`, no
`.model.`).

Binding all 93 layers into one resident `Model` still is not attempted or
planned as a real inference path: the packed bf16 trunk alone is roughly
109 GB, which does not fit in memory on any machine in this lab, Mac mini
included. Real end-to-end inference against the released checkpoint needs the
trunk ring (below) to stream layers through a bounded window instead.

### Trunk ring, done 2026-09-23

`kimi-k3-core::trunk::TrunkRing` is the Rust port of `src/io/k3_trunk.c`: a
pinned layer prefix held resident for the ring's whole lifetime, plus the rest
streamed through a small ring, walked in the fixed forward layer order every
token visits (layer 0, 1, ..., `num_hidden_layers - 1`) -- see that C file's own
header for why that fixed order is what makes one-layer-ahead prefetch safe
with nothing to predict, and why a cyclic LRU would be the worst possible
policy for this access pattern.

Prefetch is the loadngo proactor's own async submit/wait split, deliberately
never a reader thread (Jay: "we should be utilizing the loadngo proactor and
avoiding threading"). That split didn't exist in `kimi-k3-core::io` before this
-- `ShardFiles` only exposed blocking submit-then-wait -- so `read_batch` is
now `submit_batch` immediately followed by `wait_batch` (a behavior-preserving
refactor; its own existing tests are unchanged, plus a new one proving real
work can happen between submitting and waiting). `bind.rs`'s per-layer tensor
list moved out of `load_layer`'s body into a shared `layer_plan`, so the new
async pair (`BoundStorage::submit_layer`/`absorb_layer`) and the existing
blocking `load_layer` can never name a different set of tensors for the same
layer.

Caller contract: call `prefetch(L + 1)` *before* `bind(L)`, not after --
`bind`'s returned `LayerWeights` borrows from `&mut self`, so nothing can call
`prefetch` again (which also needs `&mut self`) while that borrow is still
alive for the compute pass. Prefetching the next layer first, then binding and
computing on the current one, gets the overlap anyway: the read is already
submitted to the operating system by the time `bind`'s wait (if any) and the
caller's compute run, so both proceed concurrently with it regardless of Rust's
own aliasing rules on the Rust-side calls.

Not ported from the C engine: budget-based automatic sizing of the pinned
prefix (`k3_trunk_open`'s `budget_bytes`) and preallocated, uniformly-sized
ring slots for `O_DIRECT`. A caller here states `pin_layers`/`ring_slots`
directly; ring slots are ordinary per-layer allocations, freed and replaced as
layers stream through, consistent with this port not yet implementing direct
I/O or aligned slot memory anywhere.

Validated against the real checkpoint (`real_checkpoint_trunk.rs`, `#[ignore]`d,
run explicitly): opened a ring with layer 0 pinned and a two-slot streaming
ring, walked layers 0..=3 using the prefetch-then-bind pattern, and confirmed
every layer's weights exactly match what `BoundStorage`'s already-validated
direct path produces for the same layers (fingerprinted by matrix lengths plus
a running hash over `in_norm`, since `LayerWeights` has no equality check) --
not simulated, and with genuine prefetch hits observed (2 hits, 1 miss across 3
non-pinned layers). That run caught a real bug in the first draft: `bind`'s
hit/miss labels were backwards, counting every successful prefetch as a "miss"
because absorbing an already-in-flight read was conflated with never having
prefetched at all.

### Forward pass wired to the ring, done 2026-09-23

`TrunkRing::forward` mirrors `model::Model::run`'s full-recompute path exactly
-- embedding, per-layer state, the model-level attention-residual aggregator,
final norm, LM head -- but sources each decoder layer from the ring via
`prefetch(L + 1)` then `bind(L)` instead of indexing a pre-materialized
`Vec<LayerWeights>`. `decoder_layer` itself is the same already-tested free
function either way, so this only risked the orchestration loop around it, not
the numerics. New `TopLevelWeights` bundles `embed`/`lm_head`/`final_norm`/
`out_res`: not part of the ring, since every one of them is needed once per
position rather than once per layer, unlike the trunk.

Validated against the real checkpoint (same file, `#[ignore]`d): built a
one-layer view of the real config pinned to layer 0 -- the checkpoint's one
dense, non-MoE layer, so `layer::NoStreamedExperts` is honestly correct there
rather than standing in for the unimplemented expert-cache wiring -- ran the
same token ids through both `TrunkRing::forward` and the existing, already-
validated `Model::forward`, and got bit-identical logits.

**The real streamed-expert path is also done, same day.** `TrunkRing::forward`
already took `experts: &mut dyn ExpertSource` generically, and
`cache::CachedExperts` (the engine's real `ExpertSource`, independently tested
in `cache.rs`) already implements that trait -- so wiring it in needed no
change to `trunk.rs` at all. A second real-checkpoint test
(`ring_forward_matches_model_forward_with_the_real_streamed_expert_cache`)
proves that: layers 0..=3 (one dense, three MoE) through both
`TrunkRing::forward` and direct `Model::forward`, each with its own fresh
`ExpertCache` over the same index, land on bit-identical logits -- real
per-token top-k routing from the real gate weights, not simulated.

Prefill expert batching and the CLI/tokenizer are what is left before this is
a real, runnable inference path end to end.

## I/O: loadngo leads

Every shard read goes through `loadngo-proactor` (`kimi-k3-core::io`), pinned to an
exact loadngo commit in `crates/kimi-k3-core/Cargo.toml`. Shards are opened once
(overlapped on Windows) and each read is a positioned completion read. That one path
is `io_uring` on Linux, IOCP on Windows, kqueue on macOS and iOS, and epoll on
Android, in place of the C engine's per-OS `pread`/`O_DIRECT`/`F_NOCACHE` shims and
prefetch threads.

- `ShardFiles::read_batch` submits a whole batch before collecting completions,
  resumes short reads, and treats end of file inside a range as an error (including
  Windows' `ERROR_HANDLE_EOF`).
- Buffers travel with the read and come back filled, so `ExpertCache` reads each
  expert straight into its slot's storage: no zeroed temporary and no copy.
  Slots reserve capacity rather than zeroing the budget, so pages are committed by
  the first read, as with the C arena.
- `ExpertCache::prefetch_many` follows the C three-phase shape: reserve distinct,
  pinned slots serially; read every expert in one batch; publish.

Concurrency is a property of the backend, not of this crate. `io_uring` and IOCP
service a batch concurrently, and since loadngo `98d58d5a` (pinned here from
`0971c29e`) the kqueue and epoll backends hand regular-file reads to a worker pool, so
a batch runs concurrently on macOS, iOS and Android too. On the Mac mini a batch of 128
uncached 64 KiB reads went from 17.1 ms to 2.7 ms; the port needed no change.

Direct I/O (`O_DIRECT`/`F_NOCACHE`) and aligned slot memory are not implemented yet;
reads use the page cache.

## Port order

1. Configuration parsing and safetensors I/O. *Done.*
2. Expert-cache and trunk-streaming contracts. *Done: expert cache, and the trunk ring
   streams per-layer weights through the loadngo proactor's async submit/wait split,
   validated against the real checkpoint.*
3. Pure numerical kernels, checked against the C fixture manifest. *Done, including the
   bf16 and MXFP4 matmuls, wired through the layers with streamed experts.*
4. Tensor binding and the tiny end-to-end model oracle. *Done: tiny oracle bit-identical
   to C, and the released checkpoint's real names bind and validate against the actual
   downloaded checkpoint. Full-model resident binding is not the target -- see the trunk
   ring, next.*
5. CLI, tokenizer, full-memory modes, and released-checkpoint validation.
   *Done: the `k3` CLI runs a real prompt through the real released checkpoint
   end to end -- tokenize, stream 91 of 93 layers per forward pass through
   `TrunkRing` and the real streamed-expert cache with prefill batching,
   greedy-decode, detokenize -- and produces correct output; see "End-to-end
   generation" below. Still open: the named memory-preset ladder with free-RAM
   auto-sizing, `--incremental`/KV-cache carry-forward (decode is full recompute
   only, matching the C engine's own default mode), `--spec`/`--draft-trunk`
   speculative decode, `--save-state`/`--load-state`, `--ultra-low-memory`, and
   preallocated scratch.*

### End-to-end generation, verified 2026-09-23

`k3 <checkpoint-dir> --prompt "The capital of France is" --gen 3 --pin-layers 2
--ring-slots 2` ran against the real, full 1.4 TB checkpoint on `/Volumes/Jarraya`
and produced `The capital of France is Paris.",` followed by a lone `+` on the
next line. This was a raw completion without the released checkpoint's chat
format, not a validated assistant reply. The earlier interpretation that this
proved a base-model release was incorrect: `encoding_k3.py` and the checkpoint
README explicitly define chat and preserved thinking history. Indexed 497,220
tensors, tokenized the prompt to 5
real ids, streamed 91 of the 93 layers fresh through a two-slot ring on every
one of the 3 forward passes (136 hits, 137 misses total, matching `~3x` the
91 non-pinned layers), and detokenized the generated ids back to exactly that
text. This is the first real, non-simulated confirmation that binding, the
trunk ring, the real streamed-expert cache, prefill batching, and the
tokenizer all compose correctly against the actual checkpoint, not just
against each other in isolation.

Took 771 seconds for 3 tokens: full recompute (`O(sequence length squared)`,
the C engine's own default, per `src/cli/k3_run.c`'s own "DECODE STRATEGY"
note) with only 2 of 93 layers pinned is the
deliberately slowest, smallest-memory corner of the tradeoff space -- proving
correctness at minimal footprint first, not a performance result. A first
attempt at the same command with `--pin-layers 20` (~23 GB pinned) pushed the
process to 33 GB RSS + 24 GB compressed on a machine running other real work
at the time and made it briefly nearly unusable; killed immediately, memory
recovered instantly, and `--pin-layers 2` (~2.3 GB) was used for the run
above instead. Incremental state through the ring (not yet ported) would avoid
repeating prefix compute, but would still re-touch streamed weights for each
generated token. It does not by itself establish affordable conversational
latency. Measure that separately; see [CHAT.md](CHAT.md).

### Tokenizer, done 2026-09-23

`kimi-k3-core::tokenizer` ports `third_party/tok.h` (vendored, Apache-2.0 --
byte-level BPE: vocabulary, pre-tokenizer, merge algorithm, encode/decode) and
`src/tokenizer/k3_tok.h` (first-party -- the K3-specific loader that populates
it directly from the released `tiktoken.model`/`tokenizer_config.json`, since
K3 ships no `tokenizer.json`). Only the one fixed configuration K3 actually
uses is ported: the Kimi pre-tokenizer (`\p{Han}`-run rule, Han excluded from
the letter classes) and the `rankbpe` merge rule (merge the pair whose
concatenation has the lowest vocabulary id -- there is no merges list in this
format), not the general cl100k/o200k/merges-list machinery `tok.h` also
supports for other tokenizer families.

No external regex engine, and none needed: the C source hand-writes every
pre-tokenizer rule as explicit greedy character-class scanning rather than
calling a real regex engine, which is what makes a faithful, dependency-free
Rust port possible at all -- the pattern's lookaheads and `&&` class
intersection aren't expressible in Rust's `std` `regex` crate, and adding
`fancy-regex` for one file was rejected. The Unicode range tables (`uni_L`,
`uni_N`, `uni_S` from `tok_unicode.h`; `uni_U`, `uni_X` from
`tok_unicode_o200k.h`; `is_han` from `tok.h`) are embedded verbatim: every
`(lo, hi)` pair was extracted programmatically from the C headers, not
hand-transcribed, and checked sorted, non-overlapping, and matching each
table's own declared entry count before use.

Validated against real, independent ground truth, not just this project's own
C engine: `tests/fixtures/tokenizer/parity.json` holds real `tiktoken` library
encode results, built by loading the actual checkpoint's 163,584-rank
`tiktoken.model` into a real `tiktoken.Encoding` with the Kimi regex pattern
and encoding 65 strings spanning CJK, emoji, ZWJ sequences, RTL text, accents,
contractions, numeric-grouping boundaries, consecutive special tokens, and
non-BMP codepoints. All 65 match token-for-token
(`crates/kimi-k3-core/tests/tokenizer_parity.rs`, not `#[ignore]`d -- the
fixture ships in the repo, only building the real `Tokenizer` needs the
checkpoint locally), decode round-trips exactly, and this port's own
encode-then-decode reproduces every fixture string. Also confirmed the harness
actually catches a mismatch rather than passing vacuously, by corrupting one
fixture's expected ids and watching it fail with the right diagnostic before
reverting.

`cargo test` is the Rust gate for completed slices. `make test` remains the C
baseline until the Rust end-to-end oracle and CLI are complete.
