# Local interactive chat

## Run

The Rust CLI now defaults to a terminal conversation when given a model directory
without a one-shot prompt. Model weights/index/cache are loaded once per process.
It never downloads weights or sends prompts off the machine.

```sh
# Required layout: parent/loadngo/inference and parent/kimi-k3-in-rust
cd kimi-k3-in-rust
cargo build --locked --release -p kimi-k3-cli
target/release/k3 /Volumes/Jarraya/kimi-k3 --chat --gen 64 --max-context 512
```

Add `--offline` to Cargo when dependencies are already cached. On Jay's Mac the
root `~/pudding/launch-kimi-k3.sh` builds offline and starts chat with no arguments.
An initial positional text argument retains the old raw-completion mode.

Type one message per line. Commands: `/help`, `/stats`, `/continue`, `/undo`,
`/reset`, `/quit`. Ctrl-D exits while waiting for input. Ctrl-C requests cancellation
during generation; it is checked between layers/output rows, not inside a kernel
or in-flight disk read. At the idle prompt or during loading Ctrl-C exits.
No conversation files are written: closing the process loses its history.

`--gen` limits each generation/continuation, including thinking and structure tokens.
A truncated/cancelled response remains **unfinished**. Use `/continue` or discard it
with `/undo` or `/reset`; do not append a new user message into half a K3 message.
`--max-context` bounds the entire token history. There is no silent eviction.
`--layers` is forbidden in chat: a diagnostic subset is not a working chatbot.
Memory defaults remain conservative: two pinned layers, two ring slots, 4 GiB expert
cache. These settings are not a guarantee against memory pressure with long prompts.

## Responsibility and format

- Loadngo: exact bounded token history, turn lifecycle, cooperative generation,
  cancellation, output backpressure, UTF-8 stream decoding. Fresh BSD-3-Clause code.
- Kimi: model, tokenizer, XTML prompt encoding, rendering thinking/response sections,
  CPU inference and terminal adapter. Existing Apache licensing is preserved.
- Disk reads remain on the existing Loadngo proactor. The terminal blocks for user
  input; it adds no polling/timer loop or GUI scheduler. `--accel ane` (the launcher
  default) runs dense and expert products on the Neural Engine; see
  [APPLE_NEURAL_ENGINE.md](APPLE_NEURAL_ENGINE.md).

The format reference is the downloaded checkpoint's `encoding_k3.py` and
`tokenization_kimi.py`, revision `f831ab66814297da540d832a5235f8e904f29d06`.
Text-only messages use XTML `message role=...` with a `think` generation prefix;
the initial system message requests `thinking_effort=low`. Ordinary user text is
encoded separately from structural tokens, matching the reference segment boundaries.
Pasted control-marker spellings remain ordinary text. Generated tokens (including
thinking and end markers) remain exact in history, not decoded and re-tokenized.
No tools are advertised or executed. Terminal escape/control characters are escaped.

This corrects the older port note calling the release a raw base model: the local
checkpoint explicitly provides chat encoding and preserved thinking-history support.
Three plausible raw completion tokens did not prove chat quality.

## What is still missing

Decoding is still **slow**. 2026-09-24, five-token raw prompt, full model: the prompt
pass took 106 s and each further token about 63 s with `--accel ane` and incremental
decoding (2026-09-23: 771 s for three tokens on the CPU with full recompute). The floor
is streaming the 109 GB trunk from the external drive for every token; see
[APPLE_NEURAL_ENGINE.md](APPLE_NEURAL_ENGINE.md). Chat reuses one `TrunkSession`,
feeding only tokens it has not consumed and rebuilding after `/undo`, `/reset` or a
cancelled pass; that path has **not yet been run live** against the full model.
The interface work does not establish satisfactory speed or answer quality.

Remaining inference work: logit-level ANE-vs-CPU agreement and answer quality on the
full model, less data per token (more pinned layers, or a smaller trunk, which is a
numerics decision for Jay), and the ANE path's peak memory.
Also absent: persistent transcripts, multiline editor, tool calls, images, sampling
controls and a GUI. Do not describe any of those as shipped.

## Verification and publication

```sh
cd ../loadngo
cargo test --offline -p loadngo-inference
cargo clippy --offline -p loadngo-inference --all-targets -- -D warnings
cd ../kimi-k3-in-rust
cargo test --offline --locked -p kimi-k3-core -p kimi-k3-cli
cargo clippy --offline --locked -p kimi-k3-core -p kimi-k3-cli --all-targets -- -D warnings
KIMI_K3_CHECKPOINT=/Volumes/Jarraya/kimi-k3 cargo test --offline -p kimi-k3-cli -- --ignored
```

Normal tests must run without model weights; the explicit ignored tokenizer test
uses only local tokenizer files, with a fake token source for terminal mechanics.
It does **not** prove full-checkpoint conversation quality. Full-model acceptance
requires an actual coherent reply and follow-up, with token latency and peak memory
recorded, separately from these protocol tests.

### Local verification, 2026-09-24

- 63 normal Kimi core/CLI tests passed; the explicit real-tokenizer chat test also
  passed. Four existing full-weight binding/trunk tests remained ignored.
- Six Loadngo session tests passed on macOS and natively on Dolores/Linux using
  an isolated dependency-free `rustc --test` build, cleaned up afterward.
- Strict Clippy passed for the changed crates on macOS, aarch64 Linux and x86_64
  Windows targets. Linux/Windows cross-checks are not native CLI runtime tests.
- The launcher indexed all 96 local shards, loaded two pinned layers in 2.64 s,
  and reached `You>`. `/stats`, Ctrl-C during a real 92-token chat forward,
  `/undo`, `/reset`, and `/quit` were exercised. Cancellation returned to the
  same prompt with its unfinished turn retained; undo returned context to zero.
- That forward was deliberately cancelled (27.9 s elapsed total; **not** measured
  cancellation latency) before a generated token. No complete real-model reply
  or follow-up was obtained. End-to-end conversation quality remains unverified.
- Sampled RSS was about 8.0 GiB during inference and 7.95 GiB idle; CPU was 99.3%
  active and 0.0% at a later idle sample. These are samples, not peak-memory or
  thermal-safety certification. `pmset` had no recorded thermal/performance warning;
  the timing wrapper's detailed resource query was sandbox-blocked on exit.
- No weights/dependencies downloaded, no cloud inference, no model/service left
  running, and no commit or push performed.

The new shared crate is a sibling path dependency, not a private absolute path or
a copied implementation. CI checks out both repositories in that layout. Publication
must be coordinated: publish `loadngo/inference` first, pin the CI checkout to that
published Loadngo revision, then publish this consumer. Until Jay authorizes that
sequence, this is local work; no remote green-CI or public-release claim is made.
