# Kimi command sessions and current information

> **Removed 2026-10-09 (Jay).** Kimi no longer runs shell commands: `terminal_exec`,
> `terminal_read`, `terminal_write` and `terminal_stop` (`terminal.rs`) are gone, in the
> shared loop and in `--legacy-chat`. She checks work with loadngo's `cargo` (check,
> test, clippy, build, fmt --check) and `git` (status, diff, log, show) tools, which run
> no shell and refuse arguments that reach outside the workspace. Her saved chats had
> used the terminal mostly for `git log` and `ps`. The rest of this file records how the
> sessions behaved, for reading old transcripts; the current-information guidance at
> its end still applies.

Kimi Linear chat on macOS and Linux exposes `terminal_exec`, `terminal_read`,
`terminal_write`, and `terminal_stop` by default. `--no-tools` disables them with other tools.
Restart through `../launch-kimi-k3.sh` to rebuild and use them.

`terminal_exec` runs `/bin/sh -c` on Unix. On macOS it is launched through
`/usr/bin/sandbox-exec`, with a narrow profile that denies writes, deletion and
replacement of workspace-root `AGENTS.md` and `CLAUDE.md`. Reads are allowed;
repository-local instruction files and other workspace files remain writable.
The profile also prevents renaming workspace ancestors to change the protected
paths. Child processes inherit it. A pre-existing hard-link alias of either
protected file causes command launch to fail rather than leave a write bypass.
If sandbox initialization fails, the command fails; no unprotected fallback runs.

Linux terminal tools are unavailable when the workspace contains either protected
root instruction file, pending equivalent OS protection. Text tools still work.
Linux terminal sessions in workspaces without these files retain their existing
lifecycle. Windows terminal tools are unavailable
until process-tree ownership and pipe shutdown have a supported implementation;
the rest of the CLI remains available there.
Its `cwd` defaults to the workspace (`--fs-base`); relative and absolute
directories are supported. It returns a session id, combined stdout/stderr,
running status, and eventual exit status. `terminal_read` consumes further
output. `terminal_write` sends up to 16 KiB of stdin; `close_stdin: true`
delivers EOF after accepted queued input. `terminal_stop` terminates the
command. Commands, builds, tests, scripts, and line-oriented input work;
there is no PTY, terminal emulator, or support for full-screen editors.

Example tool sequence:

```json
{"command":"cargo test --offline --locked -p kimi-k3-cli","cwd":"kimi-k3-in-rust","timeout_seconds":300}
{"session_id":1,"wait_ms":1000}
```

Only one command may run at a time. Drain its output before starting the
next. Deadlines default to 120 seconds and cap at 1800 seconds; the platform
Loadngo proactor dispatches the deadline even while Kimi is generating or
waiting for the user. Pipe reads/writes and child waiting use at most four
blocking workers, plus one proactor dispatcher. They block on actual work,
with no busy polling or short-sleep loop. The reusable output buffer holds
64 KiB; each read returns about 6 KiB. Overflow discards oldest unread bytes
and reports `dropped_bytes`. For complete large logs, redirect output to a
file and read relevant portions. Stdin has a bounded four-message queue
and reports backpressure.

Ctrl-C also stops the active command; normal conversation shutdown kills
and reaps it and joins its workers. Unix commands use their own process
group; cancellation kills that group. Ordinary background descendants are
also killed when the shell exits. Deliberately detached processes are
outside this lifecycle guarantee, and can retain pipes and delay shutdown. Commands
must remain in their assigned process group; do not launch detached daemons.

Terminal calls may repeat: output reads consume new data and commands can be
rerun after completion. Terminal activity invalidates cached file-read results.
Each turn is bounded by its time and token budget, not a round count; see
[CHAT.md](CHAT.md#turns-budgets-and-pauses-since-2026-09-30).

Commands inherit Kimi's OS permissions and environment, subject to the two-file
macOS protection. This is not a general filesystem or network sandbox: commands
can write other files and access the network.
`--no-web` disables dedicated web tools, not shell networking. Existing
AGENTS.md, collaboration claims, preservation of other agents' changes,
and Jay's authorization requirements for pushes/destructive work still
apply. Prefer revision-aware text tools for claimed source edits. Report
actual test results, including failure exit status. Command side effects
survive `/undo` and `/reset`.

The current local ISO date and timezone are injected as a system message
on each user turn, outside the reusable opening snapshot. Guidance directs
current research to date-neutral searches or the current year, preserves
explicit historical requests, checks source dates, and prefers official
visa sources. Queries are not silently rewritten or blocked. This improves
grounding; it does not guarantee model compliance.

## Validation — 2026-09-30

- macOS: `cargo test --workspace --offline --locked` passed 98 tests
  (11 checkpoint-dependent tests skipped by default). Three explicit chat
  tokenizer tests and the text-edit chat integration test also passed using
  the local K3/Linear tokenizer files, without loading weights.
- `cargo fmt --all --check`, `git diff --check`, and
  `cargo clippy --workspace --all-targets --all-features --offline --locked -- -D warnings`
  passed. `cargo build --offline --locked --release -p kimi-k3-cli` rebuilt
  `target/release/k3`; CLI and launcher help both passed.
- Dolores, native aarch64 Linux, Rust 1.98.1: the exact terminal module's five
  tests passed in an isolated temporary Cargo harness against its installed
  Loadngo proactor/inference sources (`loadngo` revision `31d0b8f9`). This covers
  stdin/nonzero exit status, repeated sessions, deadlines, cancellation,
  background pipe holders, shutdown, bounded output and split Unicode.
  This was a terminal-module test, not a full Linux workspace build.
- The full Windows cross-target Clippy attempt stopped in the existing
  `blake3` build script because `ml64.exe` is not installed on this Mac.
  A separate minimal harness then passed strict all-target Clippy for
  `x86_64-pc-windows-msvc` using the actual `chat.rs`, the actual unsupported
  terminal adapter from `main.rs`, Core, Inference and Chrono. This checks the
  changed portable code without the CLI's C dependencies; it is not a full
  Windows build or runtime test. Terminal tools are explicitly unavailable there.
- A bounded release chat with the real Linear GPU model, `--no-web` and
  `--no-memory`, called `terminal_exec` and `terminal_read`. It correctly
  reported `terminal-smoke-ok`, exit status **7**, and the supplied current
  date **2026-09-30 (+07:00)**, then exited with status 0. The session used a
  temporary working directory. No inference or command-test process remained.

The regression tests also exercise current-date injection on both initial and
follow-up prompts outside the cached opening, repeated terminal calls through
the chat dispatcher, and refreshed file reads after terminal activity. Live
web-search query choice remains model-dependent; the smoke test did not fetch
web pages or claim that every stale-year search is eliminated.
