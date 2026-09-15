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
| `src/core/k3_ops.c` | `kimi-k3-core::ops` | the existing numeric fixture manifest and full-size scale test |
| `src/model/k3_bind.c` | `kimi-k3-core::model` | tiny-model teacher forcing, greedy, and incremental oracle |
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

## Port order

1. Configuration parsing and safetensors I/O.
2. Expert-cache and trunk-streaming contracts.
3. Pure numerical kernels, checked against the C fixture manifest.
4. Tensor binding and the tiny end-to-end model oracle.
5. CLI, tokenizer, full-memory modes, and released-checkpoint validation.

`cargo test` is the Rust gate for completed slices. `make test` remains the C
baseline until the Rust end-to-end oracle and CLI are complete.
