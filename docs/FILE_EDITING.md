# File editing in Kimi Linear chat

Kimi Linear exposes `text_read`, `text_write` (create only) and `text_edit`
(one exact replacement) for UTF-8 text throughout the workspace. Since
2026-10-01, root files and non-Git directories are writable, and there is no
extension whitelist: Rust, Markdown, TOML, JSON and extensionless text work.
The workspace root is `--fs-base` (the pudding directory in the launcher).
`--no-tools` disables them with the other file tools. Full K3 chat has no tools.

Only `AGENTS.md` and `CLAUDE.md` directly at the workspace root are protected
against text-tool writes (case-insensitive name check). They remain readable.
Copies inside repositories are editable. Jay explicitly requested this scope;
the earlier Git-only restriction was removed. macOS terminal commands also
protect these two root files; see [TERMINAL.md](TERMINAL.md).

Before editing, Kimi must read the applicable AGENTS.md and COLLABORATION.md,
inspect the board, and add an Active claims row with `board_add_row`. Use the
exact repo name, comma-separated exact repo-relative file paths, and status
`in progress`; root files use `pudding` as the area and workspace-relative paths.
For example:

```json
{"section":"Active claims","cells":["sng-mahjong","src/sim/mod.rs","Add pressed tile state","in progress"]}
```

Then `text_read` returns numbered text and a BLAKE3 `revision`. Pass an exact unique
`old_text` and the replacement `new_text` to `text_edit`. `revision` is optional:
without it the edit checks the revision `text_read` last showed for that path, or the
one Kimi's own last write produced, so an unread or since-changed file is still
refused. (Kimi read a file, got its revision, then sent the edit without it.)
`text_write` takes `path` and `content` and refuses to overwrite existing files.
Paths are relative to the workspace, e.g. `sng-mahjong/src/sim/mod.rs`.

Writes reject overlapping active claims from other agents when a board is present.
Ambiguous other-agent scope in the same area is treated as a conflict.
Kimi's own claim is a coordination requirement, rather than a runtime prerequisite:
a missing or descriptive own claim no longer prevents an authorized write.
Files already dirty in Git are refused, unless their current bytes match a write
made by these tools in this process. After restart those dirty files require
human reconciliation; adding a claim does not adopt existing dirty work.
Non-Git files use revisions and peer claims without requiring Git status.

Parents must exist. Symlink paths, dot directories, build output and ignored files
are rejected. Reads/files are limited to 1 MiB, edit/create payloads to 64 KiB.
Reads return at most 16 KiB; use line windows. A stale revision or a missing/nonunique
old string fails without changing the file. Each error names its fix: an unread or
changed file points to `text_read` (`fs_read` does not count), and an old string says
whether it occurs zero times or several. Writes stage in a unique sibling
temporary file and publish atomically; existing permissions are preserved.
The path, claims, Git status and revision are rechecked immediately before publish.
These checks coordinate cooperative agents; they are not an OS sandbox against
malicious processes racing filesystem operations.

Tools report the resulting revision and byte count. They do not compile, format,
commit, push, delete, or run model-supplied shell commands. Kimi should report the
edit as untested until a real build/test result is supplied. `/undo` and `/reset`
change conversation history only and do not undo file edits. Finish with a board
handoff; the existing board tool adds rows but cannot remove an old claim.

Failed writes are not cached as successful calls: the same arguments can be retried
after their prerequisites change. Read the error and fix the cause first. A second
identical write failure, or three failed writes without a successful text mutation,
pauses the turn with all results preserved. `/continue` resumes after Jay addresses
the cause; a new instruction or `/undo` is also available. Successful writes retain
duplicate-call protection so a completed mutation is never blindly replayed.

## Validation — 2026-10-01

- macOS workspace all-feature tests, strict all-target/all-feature Clippy,
  formatting, workspace check and the release CLI build passed.
- New regressions verify root/non-Git writes, TOML and extensionless files,
  readable protected root instructions, editable repository copies, descriptive
  own claims, root peer conflicts, retries after repair, and pausing repeat failures.
  Both `/continue` and a new corrective user message work after a failure pause.
- The actual-tokenizer text read/edit/reread/create chat integration passed
  explicitly using the local Linear tokenizer, with no model weights loaded.
- The macOS shell protection test passed explicitly outside the nested agent
  sandbox: overwrite, case alias, deletion, rename, replacement and link attempts
  were denied; ordinary files and repository copies remained writable. Existing
  hard-link aliases cause command launch to fail. The real root instruction files
  each had one link and were not modified by validation.
- The exact terminal module passed strict all-target Clippy for
  `aarch64-unknown-linux-gnu` in an isolated portable harness against the current
  local Loadngo dependencies. This is a cross-check, not native Linux execution or
  a full Linux CLI build. Linux command protection remains unavailable as described
  in TERMINAL.md; there is no unprotected fallback for a workspace with root instructions.
- Root `launch-kimi-k3.sh` help and shell syntax checks passed. Start a fresh chat
  through that launcher to load the new tool declarations and guidance.

These tests verify tool dispatch, disk effects and OS protection. They do not
establish model intelligence or guarantee autonomous work quality.

## Earlier validation

Validation (2026-09-29, `.rs` only): seven filesystem/ownership regression tests and a
scripted chat using the actual Kimi Linear tokenizer cover create, edit, reread
after mutation, stale revisions, repeated writes, dirty/staged/deleted files,
claims, paths and symlinks. This verifies dispatch and disk effects without loading
model weights; autonomous model behavior is not established by the scripted test.
The macOS workspace tests and strict Clippy passed. Linux cross-checking was
attempted but stopped in existing crypto dependency build scripts because
`aarch64-linux-gnu-gcc` is absent; Linux execution remains unverified.

Peer claims count as explicit when every path names a file with an extension or a
directory ending in `/`; anything else is treated as the whole repository.

Validation of `.md`/`.txt` (2026-09-29, Claude Code): a new test creates, reads and
edits one file of each type, and checks that a peer claim naming a Markdown file
blocks only that file. fmt, strict Clippy and the workspace tests pass on macOS.
