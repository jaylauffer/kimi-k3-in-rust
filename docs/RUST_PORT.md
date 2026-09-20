# Rust Port

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
  their exact dequantisation: same argmax everywhere, measured bit-identical logits, and
  the source is asked for exactly top-k experts per token per MoE layer. The streamed
  model's incremental decode reproduces its own full forward bit for bit.
- A streamed layer with no source is an error at its layer, and a session refuses
  further tokens after one.

`CachedExperts` is gated in `cache.rs` against direct loads of the cache fixture under
eviction pressure and batch prefetch: identical packed bytes, scales and products.

Not ported yet: binding the released checkpoint's safetensors names into these
structures, the trunk ring, prefill expert batching (`k3_moe_prefill`), threading, and
preallocated scratch.

### Real-checkpoint tensor names, confirmed 2026-09-20

`~/k3model/` holds the released checkpoint's metadata only (`config.json`,
`model.safetensors.index.json`, the tokenizer files) fetched from
`moonshotai/Kimi-K3` at commit `f831ab66814297da540d832a5235f8e904f29d06` and
verified sha256-exact against the values Hugging Face's API reports for
each file -- not the 1.56 TB of weight shards themselves, which no local
storage exists for yet (the external drive earmarked for them is a
separate, still-open decision).

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

The remaining real-checkpoint-binding work is therefore: extend
`kimi-k3-core::model`'s tensor-binding layer to read the three shared
per-layer matrices above by name (trivial -- same shape as any other
`layer::Matrix`) alongside the already-implemented per-expert MXFP4 path,
using `~/k3model/model.safetensors.index.json` as the name/shard map. This
does not need the actual weight bytes to implement or unit-test against a
synthetic fixture shaped like the real names; it needs them only to
validate against the real checkpoint once storage exists.

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
2. Expert-cache and trunk-streaming contracts. *Expert cache done; trunk streaming open.*
3. Pure numerical kernels, checked against the C fixture manifest. *Done, including the
   bf16 and MXFP4 matmuls, wired through the layers with streamed experts.*
4. Tensor binding and the tiny end-to-end model oracle. *Tiny oracle done, bit-identical
   to C; binding the released checkpoint's safetensors names open.*
5. CLI, tokenizer, full-memory modes, and released-checkpoint validation.

`cargo test` is the Rust gate for completed slices. `make test` remains the C
baseline until the Rust end-to-end oracle and CLI are complete.
