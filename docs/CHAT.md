# Local interactive chat

## Since 2026-10-09: the shared chat loop

Every model here (Kimi Linear, Gemma 4 and K3) chats on loadngo's shared loop
(`loadngo-inference::agent`, loadngo `docs/AGENT_LOOP.md`), the one gpt-oss runs on. `crates/kimi-k3-cli/src/agent_chat.rs`
connects them: `KimiTemplate` (both formats), `Engine` (any `Reader`: the context is kept
as tokens and the session follows it, restarting from a snapshot after the opening when
`/undo` or `/reset` go back) and the terminal or voice (`Kimi> ` then the reply; status
lines on stderr).

What changed for Kimi:

- **Tools**: loadngo's workspace tools (files read-only; `text_read`, `text_edit`,
  `text_write`, `text_format`; `cargo` and `git`; the Archive CAS; notes; the web) and her
  read-only `board_sections`/`board_read`. Her own text tools, the terminal tools (Jay:
  dropped) and `board_add_row` are gone: the editing tools claim a repository on the board
  at her first write and hand it off when the chat ends, listing the uncommitted files and
  the chat's own `cargo check`/`test`.
- **Guards**: the loop's (repeated calls not run again and then the tools closed, write
  failure limits, notes when one tool keeps failing or an edit undoes another, a looping
  reply halted, the check before an answer after unchecked changes, the archive coverage
  check), and Jev acting every 6 calls instead of in shadow at a handoff (`--no-checkpoint`
  turns Jev off).
- **Each turn** a system note gives the local date and time and says which model she is,
  on which engine, her context, where this chat is saved and her last reply's speed.
- **Commands**: `/continue`, `/undo`, `/reset`, `/stats`, `/help`, `/quit`. Ctrl-C pauses a
  turn; so do a spent budget (`--turn-minutes` 30, `--turn-tokens` 16,384) and a reply
  reaching `--gen`. `/continue` goes on; a new message answers waiting tool calls as not run.
- **Context flow**: past three quarters of `--max-context` she writes a handoff and the
  context is rebuilt from it; a rebuild after work that added nothing new closes her tools
  so she answers (loadngo `docs/AGENT_LOOP.md`, "Context flow").
- **Saved chats**: `~/.loadngo/kimi/transcripts/<time>.jsonl` with the same event names,
  and `<time>.state.json` beside it; `--resume latest` carries a chat on, a paused turn
  included. Snapshots saved before 2026-10-09 (by the old chat) cannot be resumed.
- **K3** runs on the loop through a next-token adapter (`NextToken`), with no tools.

The chat before the loop (`chat::run_with`, about 2,600 lines, and its own transcript
module) was removed on 2026-10-09 once the loop had everything it had.

Checked 2026-10-09 on the GPU: Kimi Linear answered where her chat is saved and which model
and engine she is, then fixed the seeded `median` bug (12 calls, 77 s, the chat's own check
passing): her first finished code task. Gemma answered a one-call question. Evidence:
loadngo `docs/AGENT_LOOP.md`, "Evidence, step 2".

The rest of this file describes the chat before the loop, for reading older transcripts
and the reasons behind behaviours the loop took over.

## Run

Since 2026-09-24 the root launcher chats with **Kimi Linear 48B-A3B** by default (about
one token per second here; see [KIMI_LINEAR.md](KIMI_LINEAR.md)); `--k3` as its first
argument selects the full K3, and since 2026-10-03 `--gemma` selects Google's Gemma 4
31B-it on the GPU (see [GEMMA.md](GEMMA.md)). The same CLI and chat commands serve all
three; Gemma's chat uses its own template and tool-call syntax (`chat_gemma.rs`).

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

The guards against a stuck model stay: a reply repeating one block is ended, and
repeated tool calls close her tools (below).

### Repeated tool calls (since 2026-10-08)

A call whose name and arguments match a call already made in this turn is not run
again (terminal tools excepted, since their output changes; a successful edit or
command, and a compaction, clear the record of reads). Its result tells her it was not run and
to answer with what she has. The first round made only of such calls gets that note.
The second one closes her tools for the rest of the turn:

- each call's result says her tools are closed and asks for a plain answer: what she
  found, with paths and line numbers, and what she did not find;
- her reply is begun for her with "I'll stop searching here and answer from what I
  have read." (the same device as the handoff opening: on the model, an instruction
  alone was answered with one more tool call);
- if she still starts a tool call, that token ends the reply, so the history never
  holds a call without a result;
- the turn ends with `[tools closed: ...]`, stop reason `repeated tool call; answered
  without tools` and a `tools_closed` event in the saved chat. The turn budget does not
  pause this reply, since it runs no tool.

Until 2026-10-08 the second round ended the turn with `[stopped: Kimi repeated the same
tool call]` and no answer. That morning (chat `2026-10-08-073103`) Jay asked her to
start in `loadngo/proactor` with no question attached; from the earlier "we're building
a better git" she looked for git code, grepped `git` (which also matches words such as
"digit"), and found only the `git check-ignore` call in loadngo
`inference/src/edit_tools.rs`. Her handoff at the 24k-token compaction set the next step
as "how the git commands are integrated with the proactor system", which does not
exist. She re-read the proactor `Cargo.toml` and `lib.rs`, then `edit_tools.rs` lines
240-290 three times; the third was refused with the note, and her next reply made the
same call word for word. 27 replies and 230 s ended with nothing for Jay.

Checked on macOS with scripted replies and the real Kimi Linear tokenizer
(`a_second_repeated_tool_call_closes_tools_and_kimi_answers`): the call runs once;
the closing reply's context holds the closed-tools result and ends with the opening;
an answer is shown, and a call tried instead ends the reply at its first token. Four
mutations fail it (no call-token cut, never closing, no opening, the old note). Not yet
run on the model: Jay's running Kimi has the old binary.

## Context flow (Kimi Linear, 2026-10-02)

`--max-context` (32768 for Kimi Linear) no longer ends a turn. When the context
passes three quarters of it, Kimi writes a handoff to herself and the context is
rebuilt from it, and the turn goes on:

1. A message marked as automatic asks for the handoff (TASK, STANDING, DONE, FACTS,
   FAILED, FILES CHANGED, NEXT; no tools), and her reply is begun for her with `TASK:`.
   (Asked as a system message, the model answered with one more tool call.) It is written at the end of the existing context, so
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

### Checkpoint (System One, shadow mode)

After each compaction the same model is asked three typed questions about Jay's
request and the handoff she just wrote (loadngo `system_one`; see loadngo
`docs/SYSTEM_ONE.md`, "Checkpoints and the Task model"). Nothing is generated: each
answer is a probability per option, read from one forward pass.

| Question | Type | Options |
|---|---|---|
| `state` | choice | `in-progress`, `paused`, `needs-input`, `needs-help`, `withdrawn`, `complete`, `stuck` |
| `repeating` | true/false | the notes show the same action failing more than once |
| `progress` | score | 0 nothing done, 1 only read, 2 partly changed, 3 changed and checked |

The `state` options are the assignment states recommended for `TaskStatus.state` in
loadngo `docs/TASK_CHECKPOINT_RECOMMENDATIONS.md`, plus `stuck`, which a worker does not
report about itself. Until 2026-10-02 (`c96309b` and before) the options were
`in-progress`, `blocked`, `complete`; saved events from then carry those names.

The answers are shown on one line and saved as a `checkpoint` event (`request`,
`handoff`, `answers`, `observed`, `seconds`). **Nothing acts on them**: the turn goes on
whatever they say. `observed.identical_calls` is counted by the program (calls made
with a name and arguments already used since the last checkpoint), so `repeating` can
be compared with what happened without anyone labelling it. `--no-checkpoint` turns
the questions off.

It is in shadow mode because the probabilities have not been measured on this kind of
decision, and because the model is judging notes it wrote itself. A wrong `blocked`
would stop good work; a wrong `in-progress` would be false comfort. The saved events
are what a threshold will be set from.

Checked 2026-10-02 on macOS with scripted replies: a 200-round turn in an 8192-token
context runs every call once across several compactions and never fills; a chat saved
at the limit mid-reply continues after `/continue`; a new message into a crowded
context starts from a handoff. Kimi's saved 2026-10-02 chat (32768/32768, mid-reply)
rebuilds to 3438 tokens plus the tool declarations, with the real tokenizer and a
fixed handoff line.

Run on the model 2026-10-02 (Mac mini, `--accel gpu`), on a copy of that chat with
`--turn-tokens 1` so no tool ran: `/continue` re-read the history, she wrote a
127-word handoff, the context went 32768 -> 6656 tokens and the turn went on; about
2.5 minutes in all, the checkpoint 2.0 s of it. The handoff named the task and the
next step but no paths or line numbers, and called five failed `sed` calls an "initial
attempt". The checkpoint, reading only that handoff, answered (with the three options of that day) `blocked` 53% /
`in-progress` 46%, `repeating` false 67% and `progress` 2 at 93%; in fact the same
call had failed repeatedly and nothing had been changed (progress 1). Her next reply
carried on with the shell approach.

In the 2026-10-08 chat above, the checkpoint after the compaction answered `stuck` 53% /
`in-progress` 28%, `repeating` false 65% and `progress` 2 at 73%, 80 s before the
repeat guard ended her turn. `stuck` was right; she had only read files (progress 1),
and was circling the same lines with different ranges, which `observed.identical_calls`
(0) does not count.

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
