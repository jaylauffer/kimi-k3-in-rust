# Local interactive chat

## Run

Since 2026-09-24 the root launcher chats with **Kimi Linear 48B-A3B** by default (about
one token per second here; see [KIMI_LINEAR.md](KIMI_LINEAR.md)); `--k3` as its first
argument selects the full K3. The same CLI and chat commands serve both.

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
`/reset`, `/quit`. Ctrl-D exits while waiting for input. Ctrl-C pauses the turn in
progress: generation stops at the next check (between layers/output rows, not inside a
kernel or in-flight disk read), a running terminal command is stopped, and tool calls
not yet run wait for `/continue`. At the idle prompt or during loading Ctrl-C exits.

### Turns, budgets and pauses (since 2026-09-30)

One message can lead to any number of replies and tool rounds; the turn ends when
Kimi answers without calling a tool. There is no round limit (the old limit of
eight rounds, 2026-09-24 to 09-30, stopped real work partway). A turn instead pauses
when its budget is spent: `--turn-minutes` (default 30) of wall-clock time, or
`--turn-tokens` (default 16384) tokens generated across its replies; 0 turns either
off. The budget is checked after each reply, before its tool calls run, so a reply in
progress always finishes. A paused turn keeps its waiting tool calls:

- `/continue` runs them and carries on with a fresh budget;
- a new message answers them as not run (so Kimi never assumes they happened) and
  goes on from there;
- `/undo` or `/reset` drops them.

`/undo` removes the last user message and every reply and tool-result round that
followed it, including an unfinished reply. One invocation removes one whole user
turn; it preserves earlier user turns. This also applies to older saved chats.
File changes and board rows are not reversed by `/undo` or `/reset`.

Before the 2026-10-01 fix, tool-result prompts were incorrectly treated as user
turns by `/undo`. It removed only the last round, potentially leaving a tool call
without its result in the conversation. Regression tests cover multiple rounds,
unfinished replies, resumed state, and a new message after undo.
Validation on macOS: `cargo test --workspace --all-features --offline --locked`,
`cargo clippy --workspace --all-targets --all-features --offline --locked -- -D warnings`,
`cargo fmt --all --check`, and the offline locked release CLI build passed.
The tests use scripted replies; the running model was not restarted for validation.

The guards against a stuck model stay: a reply repeating one block is ended, and the
same read-only call made twice in a row ends the turn.

## Context flow (Kimi Linear, 2026-10-02)

`--max-context` (32768 for Kimi Linear) no longer ends a turn. When the context
passes three quarters of it, Kimi writes a handoff to herself and the context is
rebuilt from it, and the turn goes on:

1. A system message asks for the handoff (TASK, STANDING, DONE, FACTS, FAILED, FILES
   CHANGED, NEXT; no tools). It is written at the end of the existing context, so
   nothing is read twice. It is limited to 1/32 of the context (1024 tokens).
2. The new context is the opening (tool declarations, guidance, memory), a system
   message holding the handoff and Jay's earlier messages word for word (the newest
   2 KiB of them), Jay's latest message as he typed it, and the newest tool rounds
   that fit an eighth of the context (4096 tokens), tool results included.
3. The model reads the new context once (a few thousand tokens) and carries on.

It happens at three points: before tool results that would cross the line are added;
when a reply runs into the end of the context (the cut-off reply is dropped and
written again); and before a new message from Jay that would cross the line. A saved
chat that stopped at the limit resumes the same way: `--resume latest`, then
`/continue`. Another compaction waits until the context has grown by an eighth since
the last, so a context that cannot be made smaller stops as before ("Context full").

What is kept by the program and what by Kimi: Jay's words and the newest rounds are
copied by the program; everything else survives only if her handoff says it. Reads
made before a compaction may be made again after it. Each compaction is a
`compaction` event in the transcript (`before`, `after`, `handoff`), and the handoff
is shown on the terminal as she writes it. `/undo` after a compaction removes the
whole rebuilt turn; file changes are not reversed. K3 chats (512 tokens) are not
compacted.

Checked 2026-10-02 on macOS with scripted replies: a 200-round turn in an 8192-token
context runs every call once across several compactions and never fills; a chat saved
at the limit mid-reply continues after `/continue`; a new message into a crowded
context starts from a handoff. Kimi's saved 2026-10-02 chat (32768/32768, mid-reply)
rebuilds to 3438 tokens plus the tool declarations, with the real tokenizer and a
fixed handoff line. The handoffs the model itself writes have not been judged yet.

Since 2026-10-01, failed writes can be retried after fixing the reported cause;
they are not treated as successful duplicate mutations. The same write failing
with the same error twice, or three write failures without a successful text
mutation, pauses the turn. Results and calls marked not run are held with the
saved chat. `/continue` gives the model another chance after the cause is
fixed; a new instruction, `/undo`, and `/reset` remain available.

### Saved chats and resume

Every chat is saved in `~/.loadngo/kimi/transcripts/` unless `--no-transcript` is
given; the path is printed at the start:

- `<time>.jsonl`, one JSON event per line for review: `start`/`resume`, `user`,
  `reply` (decoded text including tool-call markup, tokens, stop reason, seconds,
  context size), `tool_call` (name, arguments), `tool_result` (the text, cut at 64 KiB
  with `"cut": true`), `command`, `turn_end` (why it stopped, replies, tokens,
  seconds) and `exit`. Each event carries its local time in `at`.
- `<time>.state.json`, the exact token history, turn boundaries and waiting calls,
  rewritten (via a temporary file) after every round and at each prompt.

`--resume latest` (or a `.jsonl`/`.state.json` path) rebuilds the conversation from
the snapshot and appends to the same log, so a chat survives Ctrl-C at the prompt, a
crash or a restart. The model re-reads the saved history on the first reply. A
snapshot is only resumed by the same chat format (Kimi Linear or K3), and needs a
`--max-context` at least as large as its history. Saved chats hold whatever Kimi read
(files, command output, web pages); they stay on this Mac.

`--gen` limits each generation/continuation, including thinking and structure tokens.
A truncated/cancelled response remains **unfinished**. Use `/continue` or discard it
with `/undo` or `/reset`; do not append a new user message into half a K3 message.
`--max-context` bounds the entire token history. Nothing is dropped silently: a Kimi
Linear chat is compacted through a handoff she writes (see Context flow above).
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
K3 chat advertises no tools (Kimi Linear's are in [FILE_EDITING.md](FILE_EDITING.md)
and [TERMINAL.md](TERMINAL.md)). Terminal escape/control characters are escaped.

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
Also absent: multiline editor, K3 tool calls, images, sampling controls and a GUI.
Do not describe any of those as shipped.

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
