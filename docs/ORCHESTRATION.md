# Kimi as orchestrator: proposal and findings

Written 2026-10-04 by Claude Code at Jay's request, at the end of the session that added
Gemma 4 and Neural Engine attention. Nothing here is built. It records what Jay wants
from the local model, what the measurements say about getting there, and a proposed
first step. It ends with the decisions that are Jay's.

## The goal (Jay, 2026-10-04)

The model running on this Mac mini should dispatch tasks to Claude Code and Codex, then
review their results reasonably and identify the areas that need Jay's attention.

The local model would not write loadngo code itself. It would route the work, check the
results and summarise them. That role suits a model that fits in 64 GB much better than
coding does.

## Findings from this session

Measured, with where the evidence is:

- **Speed is not what limits the local model.** Kimi Linear (48B, 3B active) decodes at
  about 32 tokens/s and read a 2,789-token prompt in 6.87 s (about 406 tokens/s; run of
  2026-10-04 on `docs/GEMMA.md`). Her runs on the agent board failed on tool friction and
  judgment instead: edits that did not apply, a wrong approach to a wave task, thin
  handoffs. See the handoffs of 2026-10-01 and 10-02 on the board and `docs/CHAT.md`.
- **Gemma 4 31B-it works but is too slow to use.** It is a dense 31B model, so every
  token reads all 15.6 GB of 4-bit weights: 6.7 tokens/s at best, and 112-141 s for a
  6,014-token prompt. Jay found it too slow. No attention work changes that; only a
  smaller model would. See `docs/GEMMA.md`.
- **Neural Engine attention gave little.** It made Gemma's prompts 20% faster (140.5 s to
  112.6 s) and left them slow. For Kimi Linear it would help little:
  - Kimi Linear has no sliding-window attention. 20 of its 27 layers are KDA (linear
    attention with a fixed-size state), which the Core ML attention model does not fit.
  - The 7 MLA layers are ordinary attention and could use it.
  - MLA blocks, projections included, were about 13% of the 2,789-token prompt; routed
    experts were about 46% and KDA about 27%. So the gain is at most about 5-7% today.
  - The share grows only near the 32k context limit.
- **5-bit weights would be tuning, not capability.** 4-bit experts cost Kimi Linear
  1.5-2% perplexity (`docs/KIMI_LINEAR.md`, 2026-09-25). A 5-bit format adds 24% more
  bytes and slows decoding to match. There is no standard 5-bit MX element type; MXFP6
  is the next standard step. If quality ever matters more, measure first by rounding in
  the cache, as was done for the 4-bit experts, before writing kernels.
- **One finding from this session was broadly useful:**
  - **A proactor bug.** loadngo's `run_once` could block forever after its opening
    dispatch fired the last deadline (loadngo `853279ef`). It hung kimi's Windows CI and
    could hit any loadngo app with a short timer.
  - **CI.** kimi CI is green on Linux, macOS and Windows again (`ccc0840`, run
    37145401654).

The conclusion: stop optimising model speed and quantization. Gemma and the Neural
Engine work stay as they are, committed and documented.

## What already exists for the orchestrator role

- **Headless workers on this Mac.**
  - `claude -p` (Claude Code 2.1.288) takes `--max-budget-usd`, `--permission-mode`,
    `--allowedTools` and `--output-format json`/`stream-json`.
  - `codex exec` (codex-cli 0.160.0) takes `--sandbox`, `--cd`, `--json` and
    `-o/--output-last-message`.
- **The loadngo Task protocol.**
  - Messages: `TaskRequest`, `TaskOffer`, `TaskAccept`, `TaskStatus`, `TaskResult` and
    `TaskAck`.
  - Code: loadngo `network/src/bin/task_submitter.rs`, `task_worker.rs` and
    `task-node.rs`.
  - Docs: `docs/TASK_OFFER_PROTOCOL.md` and `docs/TASK_CHECKPOINT_RECOMMENDATIONS.md`.
  - The submitter verifies the result before `TaskAck`. That verification is the review
    step.
- **Kimi's chat.**
  - Tools: terminal, file read and edit, search, and the board tool (claims and
    handoffs, signed Kimi).
  - Saved transcripts and `--resume`.
  - The System One checkpoint, which judges a task's state after compaction and runs in
    shadow mode today (loadngo `docs/SYSTEM_ONE.md`).

## Proposal

### The loop

1. **Dispatch.** Jay states what he wants. Kimi writes the task:
   - repository and paths;
   - the goal;
   - success criteria as commands that must pass (the repository's CI gates at least);
   - a cost cap;
   - the worker: Claude Code or Codex.

   Jay approves each dispatch with one keystroke until the reports have earned trust.
2. **Run.** The worker runs as a child process. Its exit arrives as a completion through
   the loadngo proactor; nothing polls. The worker claims and hands off on the board as
   it does today.
3. **Verify independently.** Kimi re-runs the success criteria herself and never takes
   the worker's report as proof (COLLABORATION.md rule 5). Mechanical checks over the
   diff and the run find most of what needs Jay:
   - files changed outside the claimed paths, or another agent's uncommitted work touched;
   - tests deleted, weakened, or newly `#[ignore]`d;
   - new clippy `allow`s or lint suppressions;
   - gates green on one platform only, or a shared enum or trait changed (the
     cross-platform rule in AGENTS.md);
   - a push, tag, release or version bump;
   - the cost cap reached, or the run stopped early;
   - a question the worker asked, or a final report that does not state what was
     verified.
4. **Report.** Kimi writes a short digest per task, in one of four classes:
   - verified;
   - done but not verified;
   - failed;
   - needs Jay's decision.

   Each gets one line saying why, with commit SHAs and the failing command. Anything Kimi
   is unsure of goes in "needs Jay's decision": a missed problem costs Jay more than an
   extra line to read.

### Where the model's judgment is trusted

Barely, at first. Kimi Linear misjudged her own handoff in the checkpoint test of
2026-10-02 (`docs/CHAT.md`). The first version therefore classifies by the mechanical
checks above and uses the model only to summarise and to word the digest. The model's
own "no attention needed" call is trusted only after it has been measured, as below.

### Measuring it without waiting weeks

Jay has no weeks for a live trial. The history is already enough to measure on:

- The board, its archive, `reviews/` and git history record weeks of finished tasks: what
  each agent reported, and what Jay then had to act on. Examples:
  - the dfs defects Codex found;
  - Kimi's unfinished wave task;
  - pushes blocked for approval;
  - CI that failed after a "green on macOS" handoff.
- About 20-30 of these, replayed as finished tasks (the report, the diff, the gate
  results), form a test set. Its answer key is whether that task needed Jay.
- Running the classifier and Kimi's digest over them gives a number in a day. It also
  shows whether a different local model does better: a coding- and tool-tuned
  mixture-of-experts model of similar size might. Which one is best now is past Claude
  Code's knowledge, and adding a model family to this engine took about a day for
  Gemma.

### First step

A dispatch-and-verify loop for one worker, Claude Code via `claude -p`, on small,
test-checkable loadngo tasks. Jay approves each dispatch, and the classifier and digest
above report back. Before using it, score the digests on the historical test set. Add
Codex (`codex exec`) once the loop works.

## Decisions that are Jay's

- **Roles.** COLLABORATION.md says agents take work only from Jay and neither directs
  the other. Kimi dispatching to Claude Code and Codex needs Jay to name her orchestrator,
  acting on his approval, in the board's Roles table.
- **Cost.** Every dispatched run is billed. Set a cap per task and per day, and show
  spend in each digest.
- **Approval.** Decide which actions keep needing Jay even after trust is earned.
  Suggested: pushes outside standing approvals, releases and tags, rule changes, and
  anything touching another agent's work.
- **Whether to do this now,** against the income work (games shipping and store presence)
  that it competes with for time.
