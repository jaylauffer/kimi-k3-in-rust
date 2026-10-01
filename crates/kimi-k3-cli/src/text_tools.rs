//! Workspace-scoped editing of UTF-8 text files, with peer-claim protection,
//! revision checks and atomic writes.
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::rc::Rc;

use data::cas::CasHash;
use loadngo_inference::tools::{Tool, numbered_lines};
use serde_json::{Value, json};

const MAX_FILE: usize = 1024 * 1024;
const MAX_CHANGE: usize = 64 * 1024;
const MAX_WRITTEN: usize = 128;
struct Workspace {
    root: PathBuf,
    written: RefCell<HashMap<PathBuf, CasHash>>,
    /// The revision `text_read` last showed, or a write produced, for each path: what
    /// `text_edit` checks when the call carries no `revision`.
    seen: RefCell<HashMap<PathBuf, CasHash>>,
}

/// No writes occur during registration. The board is checked at each mutation.
pub fn tools(root: &Path) -> Result<Vec<Box<dyn Tool>>, String> {
    let root = root.canonicalize().map_err(err)?;
    let workspace = Rc::new(Workspace {
        root,
        written: RefCell::default(),
        seen: RefCell::default(),
    });
    Ok(vec![
        Box::new(TextRead(Rc::clone(&workspace))),
        Box::new(TextWrite(Rc::clone(&workspace))),
        Box::new(TextEdit(workspace)),
    ])
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn string<'a>(args: &'a Value, name: &str) -> Result<&'a str, String> {
    args.get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string `{name}`"))
}

fn read(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path).map_err(err)?;
    if !file.metadata().map_err(err)?.is_file() {
        return Err("expected a regular file".into());
    }
    let mut text = String::new();
    file.take((MAX_FILE + 1) as u64)
        .read_to_string(&mut text)
        .map_err(err)?;
    if text.len() > MAX_FILE || text.contains('\0') {
        return Err("files must be UTF-8 text without NUL, at most 1 MiB".into());
    }
    Ok(text)
}

fn valid_change(text: &str) -> Result<(), String> {
    if text.len() > MAX_CHANGE || text.contains('\0') {
        return Err("change must be at most 64 KiB without NUL".into());
    }
    Ok(())
}

impl Workspace {
    fn remember(&self, path: PathBuf, revision: CasHash) {
        let mut seen = self.seen.borrow_mut();
        if seen.len() >= MAX_WRITTEN && !seen.contains_key(&path) {
            // Forgetting only costs a text_read, which the error asks for.
            seen.clear();
        }
        seen.insert(path, revision);
    }

    /// The file a tool's `path` names. Paths start at the workspace; an absolute path
    /// inside it (what `fs_read` and `fs_find` show) and `.`/`..` that stay inside it
    /// mean the same file.
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let given = Path::new(path);
        let relative = given.strip_prefix(&self.root).unwrap_or(given);
        let mut real = self.root.clone();
        let mut parts = Vec::new();
        for part in relative.components() {
            match part {
                Component::CurDir => {}
                Component::Normal(name) => parts.push(name),
                Component::ParentDir if parts.pop().is_some() => {}
                _ => {
                    return Err(format!(
                        "{path} is outside the workspace {}; these tools only reach files \
                         inside it",
                        self.root.display()
                    ));
                }
            }
        }
        if parts.is_empty() {
            return Err("empty path".into());
        }
        for (index, part) in parts.iter().enumerate() {
            let name = part.to_string_lossy();
            if name.starts_with('.') || matches!(name.as_ref(), "target" | "node_modules") {
                return Err("hidden and build directories are not writable".into());
            }
            real.push(name.as_ref());
            match fs::symlink_metadata(&real) {
                Ok(meta) if meta.file_type().is_symlink() => {
                    return Err("symlink paths are not writable".into());
                }
                Ok(meta) if index + 1 < parts.len() && !meta.is_dir() => {
                    return Err("parent is not a directory".into());
                }
                Ok(meta) if index + 1 == parts.len() && !meta.is_file() => {
                    return Err("target is not a regular file".into());
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && index + 1 == parts.len() => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(self.missing(path, &parts.iter().collect::<PathBuf>()));
                }
                Err(e) => return Err(err(e)),
            }
        }
        Ok(real)
    }

    /// Says that `path` does not exist and, when a repository directly under the
    /// workspace has a file at that path, where it is: a path copied from inside a
    /// repository lacks the repository's name.
    fn missing(&self, path: &str, relative: &Path) -> String {
        let mut found: Vec<String> = fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
            .map(|entry| Path::new(&entry.file_name()).join(relative))
            .filter(|candidate| self.root.join(candidate).is_file())
            .map(|candidate| candidate.to_string_lossy().into_owned())
            .collect();
        found.sort();
        if found.is_empty() {
            format!(
                "{path} does not exist; paths start at the workspace, with the repository's name first"
            )
        } else {
            format!(
                "{path} does not exist; paths start at the workspace. Did you mean {}?",
                found.join(" or ")
            )
        }
    }

    /// [`Self::resolve`] for a file that must already exist.
    fn existing(&self, path: &str) -> Result<PathBuf, String> {
        let real = self.resolve(path)?;
        if real.exists() {
            Ok(real)
        } else {
            Err(self.missing(path, real.strip_prefix(&self.root).unwrap_or(&real)))
        }
    }

    fn authorize(&self, path: &Path, previous: Option<CasHash>) -> Result<(), String> {
        let workspace_path = path.strip_prefix(&self.root).map_err(err)?;
        let name = workspace_path.to_string_lossy().replace('\\', "/");
        if ["AGENTS.md", "CLAUDE.md"]
            .iter()
            .any(|protected| name.eq_ignore_ascii_case(protected))
        {
            return Err("workspace-root AGENTS.md and CLAUDE.md are protected; repository-local instruction files are editable".into());
        }
        let (scope, claim_path) = name.split_once('/').unwrap_or((
            self.root
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("pudding"),
            &name,
        ));
        let board_path = self.root.join("AGENT-BOARD.md");
        if board_path.exists() {
            check_conflicts(&read_board(&board_path)?, scope, claim_path)?;
        }
        let parent = path.parent().ok_or("missing parent")?;
        let repo = parent
            .ancestors()
            .take_while(|p| p.starts_with(&self.root))
            .find(|p| p.join(".git").exists());
        if let Some(repo) = repo {
            let relative = path.strip_prefix(repo).map_err(err)?;
            let ignored = Command::new("git")
                .arg("-C")
                .arg(repo)
                .args(["check-ignore", "--quiet", "--no-index", "--"])
                .arg(relative)
                .status()
                .map_err(err)?;
            if ignored.code() != Some(1) {
                return Err("ignored files cannot be edited (or Git ignore check failed)".into());
            }
            let owned = previous.is_some_and(|hash| self.written.borrow().get(path) == Some(&hash));
            if !owned {
                let status = Command::new("git")
                    .env("GIT_OPTIONAL_LOCKS", "0")
                    .arg("--literal-pathspecs")
                    .arg("-C")
                    .arg(repo)
                    .args(["status", "--porcelain=v1", "--untracked-files=all", "--"])
                    .arg(relative)
                    .output()
                    .map_err(err)?;
                if !status.status.success() || !status.stdout.is_empty() {
                    return Err(
                        "file has pre-existing Git changes; do not adopt another agent's work"
                            .into(),
                    );
                }
            }
        }
        if self.written.borrow().len() >= MAX_WRITTEN && !self.written.borrow().contains_key(path) {
            return Err("session write limit reached (128 files)".into());
        }
        Ok(())
    }

    fn publish(&self, name: &str, content: &str, previous: Option<&str>) -> Result<String, String> {
        if content.len() > MAX_FILE {
            return Err("result exceeds 1 MiB".into());
        }
        let path = self.resolve(name)?;
        let revision = previous.map(|text| CasHash::digest(text.as_bytes()));
        self.authorize(&path, revision)?;
        let mut temp =
            tempfile::NamedTempFile::new_in(path.parent().ok_or("missing parent")?).map_err(err)?;
        if previous.is_some() {
            temp.as_file()
                .set_permissions(fs::metadata(&path).map_err(err)?.permissions())
                .map_err(err)?;
        }
        temp.write_all(content.as_bytes()).map_err(err)?;
        temp.as_file().sync_all().map_err(err)?;
        // Revalidate after staging: a stale read never intentionally overwrites a peer.
        self.resolve(name)?;
        self.authorize(&path, revision)?;
        if let Some(expected) = previous {
            if read(&path)? != expected {
                return Err("file changed since text_read; read again before editing".into());
            }
            temp.persist(&path).map_err(err)?;
        } else {
            temp.persist_noclobber(&path).map_err(err)?;
        }
        let hash = CasHash::digest(content.as_bytes());
        self.written.borrow_mut().insert(path.clone(), hash);
        self.remember(path, hash);
        Ok(format!(
            "wrote {name}: {} bytes, revision {hash}; not compiled or tested",
            content.len()
        ))
    }
}

fn read_board(path: &Path) -> Result<String, String> {
    let mut text = String::new();
    fs::File::open(path)
        .map_err(err)?
        .take(1024 * 1024 + 1)
        .read_to_string(&mut text)
        .map_err(err)?;
    if text.len() > MAX_FILE {
        return Err("board exceeds 1 MiB".into());
    }
    Ok(text)
}

/// Unparseable peer scope is conservatively repo-wide. Kimi's own claim text
/// never blocks an authorized write; ownership is still checked against peers.
fn check_conflicts(board: &str, repo: &str, path: &str) -> Result<(), String> {
    let active = board
        .split("## Active claims")
        .nth(1)
        .ok_or("board has no Active claims")?
        .split("\n## ")
        .next()
        .unwrap_or_default();
    for line in active.lines().filter(|line| line.starts_with('|')) {
        let escaped = line.replace("\\|", "\u{1f}");
        let cells: Vec<_> = escaped
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect();
        if cells.len() < 3 || cells[1] == "Kimi" || cells[1] == "Agent" || cells[1].starts_with('-')
        {
            continue;
        }
        let scope = cells[2].replace('`', "");
        if scope != repo
            && !scope
                .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                .any(|s| s == repo)
        {
            continue;
        }
        if cells.len() != 6 {
            return Err(
                "cannot parse a claim for this repository; resolve its board row first".into(),
            );
        }
        let status = cells[5].to_lowercase();
        if ["done", "complete", "released", "cancelled"]
            .iter()
            .any(|s| status.starts_with(s))
        {
            continue;
        }
        let paths: Vec<_> = cells[3]
            .split(',')
            .map(|p| p.trim().trim_matches('`'))
            .collect();
        let explicit = paths.iter().all(|p| {
            !p.is_empty()
                && !p.contains(char::is_whitespace)
                && (Path::new(p).extension().is_some() || p.ends_with('/'))
        });
        if !explicit
            || paths
                .iter()
                .any(|p| *p == path || (p.ends_with('/') && path.starts_with(p)))
        {
            return Err(format!(
                "overlapping or ambiguous active claim by {}; resolve it on the board first",
                cells[1]
            ));
        }
    }
    Ok(())
}

/// Where `old` is in `text` when only indentation differs: whole lines that match once
/// leading and trailing whitespace is ignored, in exactly one place, every nonblank line
/// off by the same number of columns. Returns the byte range to replace, `new` shifted
/// by that many columns, and the shift (file minus `old`).
fn reindented(text: &str, old: &str, new: &str) -> Option<(std::ops::Range<usize>, String, isize)> {
    let ends_line = old.ends_with('\n');
    let wanted: Vec<&str> = old.strip_suffix('\n').unwrap_or(old).split('\n').collect();
    if wanted.iter().all(|line| line.trim().is_empty()) {
        return None;
    }
    // Each file line: where it starts, and its text without the line ending.
    let mut lines = Vec::new();
    let mut at = 0;
    for line in text.split_inclusive('\n') {
        lines.push((at, line.trim_end_matches(['\n', '\r'])));
        at += line.len();
    }
    let mut places = (0..(lines.len() + 1).saturating_sub(wanted.len())).filter(|&first| {
        wanted
            .iter()
            .zip(&lines[first..])
            .all(|(want, (_, have))| want.trim() == have.trim())
    });
    let first = places.next()?;
    if places.next().is_some() {
        return None;
    }
    let indent = |line: &str| line.len() - line.trim_start().len();
    let signed = |columns: usize| isize::try_from(columns).unwrap_or(isize::MAX);
    let mut shifts = wanted
        .iter()
        .zip(&lines[first..])
        .filter(|(want, _)| !want.trim().is_empty())
        .map(|(want, (_, have))| signed(indent(have)) - signed(indent(want)));
    let by = shifts.next()?;
    if shifts.any(|shift| shift != by) {
        return None;
    }
    let pad = lines[first..first + wanted.len()]
        .iter()
        .find_map(|(_, have)| have.chars().next().filter(|c| c.is_whitespace()))
        .unwrap_or(' ');
    let mut shifted = String::with_capacity(new.len());
    for line in new.split_inclusive('\n') {
        if line.trim().is_empty() {
            shifted.push_str(line);
        } else if by >= 0 {
            shifted.extend(std::iter::repeat_n(pad, by.unsigned_abs()));
            shifted.push_str(line);
        } else {
            // A line of new_text with less indentation than is to be removed cannot be
            // placed; the edit is refused instead of guessed.
            let cut = by.unsigned_abs();
            if indent(line) < cut || !line.is_char_boundary(cut) {
                return None;
            }
            shifted.push_str(&line[cut..]);
        }
    }
    let (last_start, last) = lines[first + wanted.len() - 1];
    let mut end = last_start + last.len();
    if ends_line {
        end = lines
            .get(first + wanted.len())
            .map_or(text.len(), |next| next.0);
    }
    Some((lines[first].0..end, shifted, by))
}

struct TextRead(Rc<Workspace>);
struct TextWrite(Rc<Workspace>);
struct TextEdit(Rc<Workspace>);

impl Tool for TextRead {
    fn name(&self) -> &'static str {
        "text_read"
    }
    fn description(&self) -> &'static str {
        "Read a workspace UTF-8 text file and its revision before editing. Each line is shown as number|text; the text, with its indentation, starts right after the bar. At most 16 KiB."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"path":{"type":"string"},"line_start":{"type":"integer"},"line_count":{"type":"integer"}},"required":["path"]})
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        let name = string(args, "path")?;
        let path = self.0.existing(name)?;
        let text = read(&path)?;
        let revision = CasHash::digest(text.as_bytes());
        self.0.remember(path, revision);
        let number = |key, default| {
            args.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(default)
                .clamp(1, MAX_FILE)
        };
        Ok(format!(
            "{name}: {} bytes, revision {revision}\n{}",
            text.len(),
            numbered_lines(&text, number("line_start", 1), number("line_count", 100))
        ))
    }
}

impl Tool for TextWrite {
    fn name(&self) -> &'static str {
        "text_write"
    }
    fn description(&self) -> &'static str {
        "Create a new UTF-8 text file anywhere in the workspace, including its root. Never overwrites. Only root AGENTS.md and CLAUDE.md are protected; respect peer claims."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]})
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        let content = string(args, "content")?;
        valid_change(content)?;
        self.0.publish(string(args, "path")?, content, None)
    }
}

impl Tool for TextEdit {
    fn name(&self) -> &'static str {
        "text_edit"
    }
    fn description(&self) -> &'static str {
        "Edit a workspace UTF-8 text file, including root files, with one old_text/new_text replacement. old_text is whole text copied from the file, without line numbers; if only its indentation differs and it matches one place, the edit is applied there and new_text is shifted to fit. text_read the file first; revision is optional and defaults to the one text_read last showed. Root AGENTS.md/CLAUDE.md protected; refuses stale reads, peer conflicts and pre-existing Git changes."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"path":{"type":"string"},"revision":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"]})
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        let name = string(args, "path")?;
        let path = self.0.existing(name)?;
        let text = read(&path)?;
        let current = CasHash::digest(text.as_bytes());
        // Kimi text_read the file, got its revision, then sent the edit without it. The
        // revision only proves she saw the current text, which the tools already know.
        let revision = match args.get("revision").and_then(Value::as_str) {
            Some(given) => given.to_string(),
            None => self
                .0
                .seen
                .borrow()
                .get(&path)
                .map(|seen| seen.to_hex())
                .ok_or_else(|| {
                    format!(
                        "{name} has not been read with text_read in this chat (fs_read does \
                         not count): text_read it, then retry this edit"
                    )
                })?,
        };
        if revision != current.to_hex() {
            return Err(format!(
                "{name} changed since you read it: text_read it again, then retry this edit"
            ));
        }
        let old = string(args, "old_text")?;
        let new = string(args, "new_text")?;
        valid_change(old)?;
        valid_change(new)?;
        match (old.is_empty(), text.matches(old).count()) {
            (true, _) => return Err("old_text must be nonempty".into()),
            (false, 0) => {
                if let Some((range, shifted, by)) = reindented(&text, old, new) {
                    let mut edited = text.clone();
                    edited.replace_range(range, &shifted);
                    return self.0.publish(name, &edited, Some(&text)).map(|done| {
                        format!(
                            "{done}. old_text matched one place once indentation was \
                             ignored (the file's is {by} column(s) {}); new_text was shifted \
                             to fit",
                            if by < 0 { "shallower" } else { "deeper" },
                            by = by.abs()
                        )
                    });
                }
                return Err(format!(
                    "old_text is not in {name}: text_read the lines you mean to change and \
                     copy them exactly, including indentation"
                ));
            }
            (false, 1) => {}
            (false, n) => {
                return Err(format!(
                    "old_text occurs {n} times in {name}; include surrounding lines so it \
                     matches once"
                ));
            }
        }
        self.0
            .publish(name, &text.replacen(old, new, 1), Some(&text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loadngo_inference::tools::Toolbox;

    const ORIGINAL: &str = "pub fn value() -> u32 { 1 }\n";

    fn board(agent: &str, paths: &str) -> String {
        format!(
            "# Board\n## Active claims\n| Since | Agent | Repo | Paths | Task | Status |\n|---|---|---|---|---|---|\n| today | {agent} | demo | {paths} | task | in progress |\n## Handoffs\n"
        )
    }

    fn git(repo: &Path, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args([
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "core.hooksPath=disabled-test-hooks",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn fixture() -> (tempfile::TempDir, Toolbox) {
        let root = tempfile::tempdir().unwrap();
        let repo = root.path().join("demo");
        fs::create_dir_all(repo.join("src")).unwrap();
        git(&repo, &["init", "--quiet", "--template="]);
        fs::write(repo.join("src/lib.rs"), ORIGINAL).unwrap();
        fs::write(repo.join(".gitignore"), "ignored.rs\n").unwrap();
        git(&repo, &["add", "src/lib.rs", ".gitignore"]);
        git(&repo, &["commit", "--quiet", "-m", "fixture"]);
        fs::write(
            root.path().join("AGENT-BOARD.md"),
            board(
                "Kimi",
                "src/lib.rs, src/new.rs, ignored.rs, README.md, notes.txt",
            ),
        )
        .unwrap();
        let mut toolbox = Toolbox::default();
        for tool in tools(root.path()).unwrap() {
            toolbox.push(tool);
        }
        (root, toolbox)
    }

    fn edit(toolbox: &Toolbox, text: &str, old: &str, new: &str) -> Result<String, String> {
        toolbox.call(
            "text_edit",
            &json!({"path":"demo/src/lib.rs",
            "revision": CasHash::digest(text.as_bytes()).to_hex(),
            "old_text":old,"new_text":new})
            .to_string(),
        )
    }

    #[test]
    fn edit_errors_say_which_call_fixes_them() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("plan.md"), "twice\ntwice\nonce\n").unwrap();
        let mut toolbox = Toolbox::default();
        for tool in tools(root.path()).unwrap() {
            toolbox.push(tool);
        }
        let edit = |args: Value| toolbox.call("text_edit", &args.to_string()).unwrap_err();
        let revision = CasHash::digest(b"twice\ntwice\nonce\n").to_hex();
        let unread = edit(json!({"path":"plan.md","old_text":"once","new_text":"x"}));
        assert!(unread.contains("text_read it, then retry"), "{unread}");
        let absent =
            edit(json!({"path":"plan.md","revision":revision,"old_text":"thrice","new_text":"x"}));
        assert!(absent.contains("not in plan.md"), "{absent}");
        let repeated =
            edit(json!({"path":"plan.md","revision":revision,"old_text":"twice","new_text":"x"}));
        assert!(repeated.contains("occurs 2 times"), "{repeated}");
    }

    #[test]
    fn an_edit_without_revision_uses_the_last_one_text_read_showed() {
        // Kimi's 2026-10-01 session: text_read, then the same edit without `revision`.
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("Cargo.toml"), "[dependencies]\na = 1\n").unwrap();
        let mut toolbox = Toolbox::default();
        for tool in tools(root.path()).unwrap() {
            toolbox.push(tool);
        }
        let edit = |old: &str, new: &str| {
            toolbox.call(
                "text_edit",
                &json!({"path":"Cargo.toml","old_text":old,"new_text":new}).to_string(),
            )
        };
        toolbox
            .call("text_read", r#"{"path":"Cargo.toml"}"#)
            .unwrap();
        edit("a = 1", "a = 2").unwrap();
        // Her own write counts as seen, so a second edit needs no new read.
        edit("a = 2", "a = 3").unwrap();
        assert_eq!(
            fs::read_to_string(root.path().join("Cargo.toml")).unwrap(),
            "[dependencies]\na = 3\n"
        );
        // A change she has not seen is still refused.
        fs::write(root.path().join("Cargo.toml"), "[dependencies]\na = 9\n").unwrap();
        let stale = edit("a = 9", "a = 4").unwrap_err();
        assert!(stale.contains("changed since you read it"), "{stale}");
        assert_eq!(
            fs::read_to_string(root.path().join("Cargo.toml")).unwrap(),
            "[dependencies]\na = 9\n"
        );
    }

    #[test]
    fn root_and_non_git_text_files_are_writable_without_claims() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("notes")).unwrap();
        let mut toolbox = Toolbox::default();
        for tool in tools(root.path()).unwrap() {
            toolbox.push(tool);
        }
        for path in ["plan.md", "notes/config.toml", "notes/extensionless"] {
            toolbox
                .call(
                    "text_write",
                    &json!({"path":path,"content":"before\n"}).to_string(),
                )
                .unwrap();
            toolbox.call("text_edit", &json!({"path":path,"revision":CasHash::digest(b"before\n").to_hex(),"old_text":"before","new_text":"after"}).to_string()).unwrap();
            assert_eq!(
                fs::read_to_string(root.path().join(path)).unwrap(),
                "after\n"
            );
        }
        // An existing root file is editable after reading its current revision.
        fs::write(root.path().join("existing.md"), "old\n").unwrap();
        toolbox.call("text_edit", &json!({"path":"existing.md","revision":CasHash::digest(b"old\n").to_hex(),"old_text":"old","new_text":"new"}).to_string()).unwrap();
        assert!(toolbox.call("text_edit", &json!({"path":"existing.md","revision":CasHash::digest(b"old\n").to_hex(),"old_text":"new","new_text":"stale"}).to_string()).is_err());
        assert_eq!(
            fs::read_to_string(root.path().join("existing.md")).unwrap(),
            "new\n"
        );
    }

    #[test]
    fn only_root_instruction_files_are_protected_and_remain_readable() {
        let (root, toolbox) = fixture();
        for name in ["AGENTS.md", "CLAUDE.md"] {
            fs::write(root.path().join(name), "keep\n").unwrap();
            assert!(
                toolbox
                    .call("text_read", &json!({"path":name}).to_string())
                    .is_ok()
            );
            for path in [name.to_string(), name.to_lowercase()] {
                assert!(
                    toolbox
                        .call(
                            "text_write",
                            &json!({"path":path,"content":"overwrite"}).to_string()
                        )
                        .unwrap_err()
                        .contains("protected")
                );
            }
            assert!(toolbox.call("text_edit", &json!({"path":name,"revision":CasHash::digest(b"keep\n").to_hex(),"old_text":"keep","new_text":"changed"}).to_string()).unwrap_err().contains("protected"));
            let local = format!("demo/{name}");
            toolbox
                .call(
                    "text_write",
                    &json!({"path":local,"content":"local\n"}).to_string(),
                )
                .unwrap();
            toolbox.call("text_edit", &json!({"path":local,"revision":CasHash::digest(b"local\n").to_hex(),"old_text":"local","new_text":"editable"}).to_string()).unwrap();
            assert_eq!(
                fs::read_to_string(root.path().join(name)).unwrap(),
                "keep\n"
            );
            assert_eq!(
                fs::read_to_string(root.path().join(local)).unwrap(),
                "editable\n"
            );
        }
    }

    #[test]
    fn malformed_own_claim_does_not_block_a_write_but_root_peer_claim_does() {
        let (root, toolbox) = fixture();
        fs::write(
            root.path().join("AGENT-BOARD.md"),
            board(
                "Kimi",
                "README.md documentation update - correct mobile build status",
            ),
        )
        .unwrap();
        toolbox
            .call(
                "text_write",
                r#"{"path":"demo/README.md","content":"hello"}"#,
            )
            .unwrap();
        let scope = root.path().file_name().unwrap().to_str().unwrap();
        let root_board = board("Claude", "plan.md").replace("| demo |", &format!("| {scope} |"));
        fs::write(root.path().join("AGENT-BOARD.md"), root_board).unwrap();
        assert!(
            toolbox
                .call("text_write", r#"{"path":"plan.md","content":"hello"}"#)
                .unwrap_err()
                .contains("claim")
        );
        assert!(!root.path().join("plan.md").exists());
    }

    #[test]
    fn clean_claimed_file_can_be_read_edited_and_edited_again() {
        let (root, toolbox) = fixture();
        let read = toolbox
            .call("text_read", r#"{"path":"demo/src/lib.rs"}"#)
            .unwrap();
        assert!(read.contains(&CasHash::digest(ORIGINAL.as_bytes()).to_hex()));
        assert!(read.contains("1|pub fn value"));
        edit(&toolbox, ORIGINAL, "{ 1 }", "{ 2 }").unwrap();
        let next = ORIGINAL.replace("{ 1 }", "{ 2 }");
        edit(&toolbox, &next, "{ 2 }", "{ 3 }").unwrap();
        assert_eq!(
            fs::read_to_string(root.path().join("demo/src/lib.rs")).unwrap(),
            ORIGINAL.replace("{ 1 }", "{ 3 }")
        );
    }

    #[test]
    fn create_is_noclobber_and_subsequent_edits_are_owned() {
        let (root, toolbox) = fixture();
        let args = json!({"path":"demo/src/new.rs","content":"// new\n"}).to_string();
        toolbox.call("text_write", &args).unwrap();
        assert!(toolbox.call("text_write", &args).is_err());
        toolbox.call("text_edit", &json!({"path":"demo/src/new.rs", "revision":CasHash::digest(b"// new\n").to_hex(), "old_text":"new", "new_text":"edited"}).to_string()).unwrap();
        assert_eq!(
            fs::read_to_string(root.path().join("demo/src/new.rs")).unwrap(),
            "// edited\n"
        );
    }

    #[test]
    fn dirty_staged_deleted_and_peer_modified_files_are_preserved() {
        let (root, toolbox) = fixture();
        let file = root.path().join("demo/src/lib.rs");
        fs::write(&file, "// peer\n").unwrap();
        assert!(
            edit(&toolbox, "// peer\n", "peer", "mine")
                .unwrap_err()
                .contains("pre-existing")
        );
        git(&root.path().join("demo"), &["add", "src/lib.rs"]);
        assert!(edit(&toolbox, "// peer\n", "peer", "mine").is_err());
        fs::remove_file(&file).unwrap();
        assert!(
            toolbox
                .call(
                    "text_write",
                    &json!({"path":"demo/src/lib.rs","content":ORIGINAL}).to_string()
                )
                .is_err()
        );
        assert!(!file.exists());
        // A separate initially clean workspace: another agent changes our own output.
        let (root, toolbox) = fixture();
        edit(&toolbox, ORIGINAL, "{ 1 }", "{ 2 }").unwrap();
        fs::write(root.path().join("demo/src/lib.rs"), "// peer\n").unwrap();
        assert!(edit(&toolbox, "// peer\n", "peer", "mine").is_err());
    }

    #[test]
    fn markdown_and_text_files_can_be_created_and_edited() {
        let (root, toolbox) = fixture();
        for (path, content) in [("demo/README.md", "# Demo\n"), ("demo/notes.txt", "one\n")] {
            toolbox
                .call(
                    "text_write",
                    &json!({"path":path,"content":content}).to_string(),
                )
                .unwrap();
            let read = toolbox
                .call("text_read", &json!({"path":path}).to_string())
                .unwrap();
            assert!(read.contains(&CasHash::digest(content.as_bytes()).to_hex()));
            toolbox
                .call(
                    "text_edit",
                    &json!({"path":path,"revision":CasHash::digest(content.as_bytes()).to_hex(),
                    "old_text":content.trim_end(),"new_text":"edited"})
                    .to_string(),
                )
                .unwrap();
            assert_eq!(
                fs::read_to_string(root.path().join(path)).unwrap(),
                "edited\n"
            );
        }
        // A peer claim naming a Markdown file is explicit, not repo-wide.
        let rows = board("Kimi", "README.md").replace(
            "## Handoffs",
            "| today | Claude | demo | docs/PLAN.md | task | in progress |\n## Handoffs",
        );
        assert!(check_conflicts(&rows, "demo", "README.md").is_ok());
        assert!(
            check_conflicts(
                &rows.replace("docs/PLAN.md", "README.md"),
                "demo",
                "README.md"
            )
            .is_err()
        );
    }

    #[test]
    fn stale_missing_ambiguous_and_oversized_edits_do_not_write() {
        let (root, toolbox) = fixture();
        assert!(
            edit(&toolbox, "stale", "{ 1 }", "{ 2 }")
                .unwrap_err()
                .contains("changed since you read it")
        );
        for old in ["", "not present", "u"] {
            assert!(edit(&toolbox, ORIGINAL, old, "x").is_err());
        }
        assert!(edit(&toolbox, ORIGINAL, "{ 1 }", &"x".repeat(MAX_CHANGE + 1)).is_err());
        assert_eq!(
            fs::read_to_string(root.path().join("demo/src/lib.rs")).unwrap(),
            ORIGINAL
        );
    }

    #[test]
    fn peer_claims_are_rechecked_after_each_write() {
        let (root, toolbox) = fixture();
        let path = root.path().join("AGENT-BOARD.md");
        fs::write(&path, board("Claude", "src/lib.rs")).unwrap();
        assert!(
            edit(&toolbox, ORIGINAL, "1", "2")
                .unwrap_err()
                .contains("claim")
        );
        fs::write(&path, board("Kimi", "src/lib.rs")).unwrap();
        edit(&toolbox, ORIGINAL, "1", "2").unwrap();
        fs::write(&path, board("Claude", "src/lib.rs")).unwrap();
        assert!(edit(&toolbox, &ORIGINAL.replace('1', "2"), "2", "3").is_err());
        assert!(check_conflicts(&board("Kimi", "src/"), "demo", "src/lib.rs").is_ok());
        let rows = board("Kimi", "src/lib.rs").replace(
            "## Handoffs",
            "| today | Claude | demo | as listed | task | in progress |\n## Handoffs",
        );
        assert!(check_conflicts(&rows, "demo", "src/lib.rs").is_err());
        let split = rows.replace("as listed", "src/other.rs");
        assert!(check_conflicts(&split, "demo", "src/lib.rs").is_ok());
    }

    #[test]
    fn old_text_with_the_wrong_indentation_is_placed_and_new_text_shifted_to_fit() {
        // Kimi's 2026-10-02 edit: the file's lines, each copied two columns too deep.
        let file = "formations: [\n    enemies: [\n        (rusher),\n        (ranged),\n    ],\n    waves: [\n    ],\n]\n";
        let old = "      enemies: [\n          (rusher),\n          (ranged),\n      ],";
        let new = "      enemies: [\n          (ranged),\n          (rusher),\n\n          (rusher),\n      ],";
        let (range, shifted, by) = reindented(file, old, new).unwrap();
        assert_eq!(by, -2);
        assert_eq!(
            &file[range.clone()],
            "    enemies: [\n        (rusher),\n        (ranged),\n    ],"
        );
        assert_eq!(
            shifted,
            "    enemies: [\n        (ranged),\n        (rusher),\n\n        (rusher),\n    ],"
        );
        // Too shallow works the same way, and a trailing newline takes the line ending.
        let (range, shifted, by) = reindented(file, "waves: [\n],\n", "waves: [],\n").unwrap();
        assert_eq!((by, shifted.as_str()), (4, "    waves: [],\n"));
        assert_eq!(&file[range], "    waves: [\n    ],\n");
        // Refused: two places, lines off by different amounts, a fragment of a line,
        // and new_text that cannot lose the indentation.
        assert!(reindented("  a\n  b\n  a\n", "a", "c").is_none());
        assert!(reindented(file, "  enemies: [\n          (rusher),", "x").is_none());
        assert!(reindented(file, "  (rusher", "x").is_none());
        assert!(reindented(file, old, "enemies: []").is_none());

        let (root, toolbox) = fixture();
        let path = root.path().join("demo/src/lib.rs");
        fs::write(&path, "fn a() {\n    one();\n}\n").unwrap();
        git(
            &root.path().join("demo"),
            &["commit", "--quiet", "-am", "indented"],
        );
        toolbox
            .call("text_read", &json!({"path":"demo/src/lib.rs"}).to_string())
            .unwrap();
        let done = toolbox
            .call(
                "text_edit",
                &json!({"path":"demo/src/lib.rs","old_text":"      one();","new_text":"      two();"})
                    .to_string(),
            )
            .unwrap();
        assert!(done.contains("2 column(s) shallower"), "{done}");
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            "fn a() {\n    two();\n}\n"
        );
    }

    #[test]
    fn a_path_inside_the_workspace_works_in_any_form_and_a_missing_one_says_where_it_is() {
        let (root, toolbox) = fixture();
        let absolute = root.path().canonicalize().unwrap().join("demo/src/lib.rs");
        for path in [
            absolute.to_str().unwrap(),
            "./demo/src/lib.rs",
            "demo/src/../src/lib.rs",
        ] {
            let shown = toolbox
                .call("text_read", &json!({"path": path}).to_string())
                .unwrap();
            assert!(shown.contains("pub fn value"), "{path}: {shown}");
        }
        // Read by its absolute path, edited without a revision by its relative one.
        toolbox
            .call(
                "text_edit",
                &json!({"path":"demo/src/lib.rs","old_text":"1","new_text":"2"}).to_string(),
            )
            .unwrap();
        // The path as a command run inside the repository shows it.
        for (tool, arguments) in [
            ("text_read", json!({"path":"src/lib.rs"})),
            (
                "text_edit",
                json!({"path":"src/lib.rs","old_text":"2","new_text":"3"}),
            ),
        ] {
            let error = toolbox.call(tool, &arguments.to_string()).unwrap_err();
            assert!(error.contains("Did you mean demo/src/lib.rs?"), "{error}");
        }
        let error = toolbox
            .call("text_read", &json!({"path":"/etc/hosts"}).to_string())
            .unwrap_err();
        assert!(error.contains("outside the workspace"), "{error}");
    }

    #[test]
    fn unsafe_paths_ignored_files_and_untracked_existing_files_are_refused() {
        let (root, toolbox) = fixture();
        for path in [
            "../outside.rs",
            "/tmp/outside.rs",
            "demo/.git/hooks/x.rs",
            "demo/target/x.rs",
            "demo/missing/x.rs",
            "demo/ignored.rs",
        ] {
            assert!(
                toolbox
                    .call(
                        "text_write",
                        &json!({"path":path,"content":"// x"}).to_string()
                    )
                    .is_err(),
                "{path}"
            );
        }
        fs::write(root.path().join("demo/src/new.rs"), "// peer").unwrap();
        assert!(toolbox.call("text_edit", &json!({"path":"demo/src/new.rs","revision":CasHash::digest(b"// peer").to_hex(),"old_text":"peer","new_text":"mine"}).to_string()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_paths_are_rejected_and_permissions_survive_replacement() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let (root, toolbox) = fixture();
        symlink("lib.rs", root.path().join("demo/src/link.rs")).unwrap();
        assert!(
            toolbox
                .call("text_read", r#"{"path":"demo/src/link.rs"}"#)
                .is_err()
        );
        symlink("src", root.path().join("demo/alias")).unwrap();
        assert!(
            toolbox
                .call(
                    "text_write",
                    r#"{"path":"demo/alias/new.rs","content":"// x"}"#
                )
                .is_err()
        );
        let file = root.path().join("demo/src/lib.rs");
        fs::set_permissions(&file, fs::Permissions::from_mode(0o640)).unwrap();
        edit(&toolbox, ORIGINAL, "{ 1 }", "{ 2 }").unwrap();
        assert_eq!(
            fs::metadata(file).unwrap().permissions().mode() & 0o777,
            0o640
        );
    }

    /// Exercises real tool-call framing and the repeated-read guard, without weights.
    #[test]
    #[ignore = "requires KIMI_LINEAR_CHECKPOINT tokenizer files"]
    fn chat_reads_edits_rereads_and_creates_files() {
        use crate::chat::{ChatFormat, ChatOptions, run_with};
        use kimi_k3_core::tokenizer::Tokenizer;
        use std::sync::atomic::AtomicBool;
        let (root, tools) = fixture();
        let tokenizer = Tokenizer::load(std::env::var("KIMI_LINEAR_CHECKPOINT").unwrap()).unwrap();
        let format = ChatFormat::kimi_linear(&tokenizer, 163_586).unwrap();
        let read = ("text_read", json!({"path":"demo/src/lib.rs"}));
        let edit = (
            "text_edit",
            json!({"path":"demo/src/lib.rs",
            "revision":CasHash::digest(ORIGINAL.as_bytes()).to_hex(),
            "old_text":"{ 1 }", "new_text":"{ 2 }"}),
        );
        let create = (
            "text_write",
            json!({"path":"demo/src/new.rs","content":"pub fn created() {}\n"}),
        );
        let replies: Vec<Vec<u32>> = [read.clone(), edit.clone(), read, create, edit].iter()
            .enumerate().map(|(i, (name, args))| tokenizer.encode(&format!(
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{i}<|tool_call_argument_begin|>{args}<|tool_call_end|><|tool_calls_section_end|><|im_end|>"
            ))).chain(std::iter::once(tokenizer.encode("Edited; not tested.<|im_end|>"))).collect();
        let (mut round, mut at) = (0, 0);
        let mut output = Vec::new();
        run_with(
            &format,
            Some(&tools),
            &tokenizer,
            8192,
            512,
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &b"Edit and create the claimed Rust files\n/quit\n"[..],
            &mut output,
            |_| {
                let token = replies[round][at];
                at += 1;
                if at == replies[round].len() {
                    round += 1;
                    at = 0;
                }
                Ok(token)
            },
            ChatOptions::default(),
        )
        .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert_eq!(
            output.matches("[tool result text_read:").count(),
            2,
            "{output}"
        );
        assert_eq!(
            output.matches("[tool result text_edit:").count(),
            1,
            "{output}"
        );
        assert!(
            output.contains("[tool call repeated: text_edit; not run again]"),
            "{output}"
        );
        assert_eq!(
            fs::read_to_string(root.path().join("demo/src/lib.rs")).unwrap(),
            ORIGINAL.replace("{ 1 }", "{ 2 }")
        );
        assert_eq!(
            fs::read_to_string(root.path().join("demo/src/new.rs")).unwrap(),
            "pub fn created() {}\n"
        );
    }
}
