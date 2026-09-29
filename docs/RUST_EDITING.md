# Rust editing in Kimi Linear chat

Kimi Linear exposes `rust_read`, `rust_write` (create only) and `rust_edit`
(one exact replacement). These use loadngo's existing `Tool`/`Toolbox` interface.
The workspace root is `--fs-base` (the pudding directory in the launcher).
`--no-tools` disables them with the other file tools. Full K3 chat has no tools.

Before editing, Kimi must read the applicable AGENTS.md and COLLABORATION.md,
inspect the board, and add an Active claims row with `board_add_row`. Use the
exact repo name, comma-separated exact repo-relative file paths, and status
`in progress`. For example:

```json
{"section":"Active claims","cells":["sng-mahjong","src/sim/mod.rs","Add pressed tile state","in progress"]}
```

Then `rust_read` returns numbered text and a BLAKE3 `revision`. Pass that revision,
an exact unique `old_text` and the replacement `new_text` to `rust_edit`.
`rust_write` takes `path` and `content` and refuses to overwrite existing files.
Paths are relative to the workspace, e.g. `sng-mahjong/src/sim/mod.rs`.

Writes require an exact active Kimi claim and reject overlapping active claims
from other agents. Ambiguous other-agent scope in the same repository is treated
as a conflict. Free-form board prose cannot prove ownership; use explicit paths.
Files already dirty in Git are refused, unless their current bytes match a write
made by these tools in this process. After restart those dirty files require
human reconciliation; adding a claim does not adopt existing dirty work.

Only UTF-8 `.rs` files under Git repos inside the workspace can be written.
Parents must exist. Symlink paths, dot directories, build output and ignored files
are rejected. Reads/files are limited to 1 MiB, edit/create payloads to 64 KiB.
Reads return at most 16 KiB; use line windows. A stale revision or a missing/nonunique
old string fails without changing the file. Writes stage in a unique sibling
temporary file and publish atomically; existing permissions are preserved.
The path, claims, Git status and revision are rechecked immediately before publish.
These checks coordinate cooperative agents; they are not an OS sandbox against
malicious processes racing filesystem operations.

Tools report the resulting revision and byte count. They do not compile, format,
commit, push, delete, or run model-supplied shell commands. Kimi should report the
edit as untested until a real build/test result is supplied. `/undo` and `/reset`
change conversation history only and do not undo file edits. Finish with a board
handoff; the existing board tool adds rows but cannot remove an old claim.

Validation (2026-09-29): seven filesystem/ownership regression tests and a
scripted chat using the actual Kimi Linear tokenizer cover create, edit, reread
after mutation, stale revisions, repeated writes, dirty/staged/deleted files,
claims, paths and symlinks. This verifies dispatch and disk effects without loading
model weights; autonomous model behavior is not established by the scripted test.
The macOS workspace tests and strict Clippy passed. Linux cross-checking was
attempted but stopped in existing crypto dependency build scripts because
`aarch64-linux-gnu-gcc` is absent; Linux execution remains unverified.
