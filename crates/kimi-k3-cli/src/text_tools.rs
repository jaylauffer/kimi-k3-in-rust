//! Workspace-scoped editing of Rust, Markdown and plain-text files, with board claims,
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
/// File types Kimi may create and edit: Rust source, Markdown and plain text.
const EDITABLE: [&str; 3] = ["rs", "md", "txt"];

fn editable(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| EDITABLE.iter().any(|e| ext.eq_ignore_ascii_case(e)))
}

struct Workspace {
    root: PathBuf,
    written: RefCell<HashMap<PathBuf, CasHash>>,
}

/// No writes occur during registration. The board is checked at each mutation.
pub fn tools(root: &Path) -> Result<Vec<Box<dyn Tool>>, String> {
    let root = root.canonicalize().map_err(err)?;
    let workspace = Rc::new(Workspace {
        root,
        written: RefCell::default(),
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
    fn resolve(&self, path: &str) -> Result<PathBuf, String> {
        let relative = Path::new(path);
        if !editable(relative) {
            return Err("only .rs, .md and .txt files may be edited".into());
        }
        let mut real = self.root.clone();
        let parts: Vec<_> = relative.components().collect();
        if parts.is_empty() {
            return Err("empty path".into());
        }
        for (index, part) in parts.iter().enumerate() {
            let Component::Normal(name) = part else {
                return Err("use a workspace-relative path without .. or absolute prefixes".into());
            };
            let name = name.to_string_lossy();
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
                Err(e) => return Err(err(e)),
            }
        }
        Ok(real)
    }

    fn authorize(&self, path: &Path, previous: Option<CasHash>) -> Result<(), String> {
        let parent = path.parent().ok_or("missing parent")?;
        let output = Command::new("git")
            .args(["-C"])
            .arg(parent)
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .map_err(err)?;
        if !output.status.success() {
            return Err("writes require a Git repository inside the workspace".into());
        }
        let repo = PathBuf::from(String::from_utf8(output.stdout).map_err(err)?.trim())
            .canonicalize()
            .map_err(err)?;
        let repo_name = repo.strip_prefix(&self.root).map_err(err)?;
        if repo_name.components().count() != 1 {
            return Err("use an immediate workspace Git repository".into());
        }
        let relative = path.strip_prefix(&repo).map_err(err)?;
        let relative_text = relative.to_string_lossy().replace('\\', "/");
        let board = read_board(&self.root.join("AGENT-BOARD.md"))?;
        check_claims(&board, &repo_name.to_string_lossy(), &relative_text)?;
        let ignored = Command::new("git")
            .arg("-C")
            .arg(&repo)
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
                .arg(&repo)
                .args(["status", "--porcelain=v1", "--untracked-files=all", "--"])
                .arg(relative)
                .output()
                .map_err(err)?;
            if !status.status.success() || !status.stdout.is_empty() {
                return Err(
                    "file has pre-existing Git changes; do not adopt another agent's work".into(),
                );
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
        self.written.borrow_mut().insert(path, hash);
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

/// Explicit Kimi claims only. Unparseable peer scope is conservatively repo-wide.
fn check_claims(board: &str, repo: &str, path: &str) -> Result<(), String> {
    let active = board
        .split("## Active claims")
        .nth(1)
        .ok_or("board has no Active claims")?
        .split("\n## ")
        .next()
        .unwrap_or_default();
    let mut claimed = false;
    for line in active.lines().filter(|line| line.starts_with('|')) {
        let escaped = line.replace("\\|", "\u{1f}");
        let cells: Vec<_> = escaped
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect();
        if cells.len() < 3 || cells[1] == "Agent" || cells[1].starts_with('-') {
            continue;
        }
        let scope = cells[2].replace('`', "");
        if !scope
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
        if cells[1] == "Kimi" {
            if scope == repo
                && matches!(status.as_str(), "active" | "in progress")
                && paths.contains(&path)
            {
                claimed = true;
            }
        } else {
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
    }
    if claimed {
        Ok(())
    } else {
        Err(format!(
            "claim {repo} / {path} first: board_add_row Active claims, exact repo and comma-separated paths, status in progress"
        ))
    }
}

struct TextRead(Rc<Workspace>);
struct TextWrite(Rc<Workspace>);
struct TextEdit(Rc<Workspace>);

impl Tool for TextRead {
    fn name(&self) -> &'static str {
        "text_read"
    }
    fn description(&self) -> &'static str {
        "Read a workspace .rs, .md or .txt file and its revision before editing. Numbered text, at most 16 KiB."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"path":{"type":"string"},"line_start":{"type":"integer"},"line_count":{"type":"integer"}},"required":["path"]})
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        let name = string(args, "path")?;
        let text = read(&self.0.resolve(name)?)?;
        let number = |key, default| {
            args.get(key)
                .and_then(Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(default)
                .clamp(1, MAX_FILE)
        };
        Ok(format!(
            "{name}: {} bytes, revision {}\n{}",
            text.len(),
            CasHash::digest(text.as_bytes()),
            numbered_lines(&text, number("line_start", 1), number("line_count", 100))
        ))
    }
}

impl Tool for TextWrite {
    fn name(&self) -> &'static str {
        "text_write"
    }
    fn description(&self) -> &'static str {
        "Create a new .rs, .md or .txt file in an existing workspace directory. Requires an exact active Kimi board claim. Never overwrites."
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
        "Replace one unique exact string in a claimed .rs, .md or .txt file. Supply revision from text_read, old_text and new_text. Refuses stale or pre-existing dirty files."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"path":{"type":"string"},"revision":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","revision","old_text","new_text"]})
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        let name = string(args, "path")?;
        let text = read(&self.0.resolve(name)?)?;
        if string(args, "revision")? != CasHash::digest(text.as_bytes()).to_hex() {
            return Err("stale revision; text_read again".into());
        }
        let old = string(args, "old_text")?;
        let new = string(args, "new_text")?;
        valid_change(old)?;
        valid_change(new)?;
        if old.is_empty() || text.find(old).is_none() || text.find(old) != text.rfind(old) {
            return Err("old_text must be nonempty and match exactly once".into());
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
    fn clean_claimed_file_can_be_read_edited_and_edited_again() {
        let (root, toolbox) = fixture();
        let read = toolbox
            .call("text_read", r#"{"path":"demo/src/lib.rs"}"#)
            .unwrap();
        assert!(read.contains(&CasHash::digest(ORIGINAL.as_bytes()).to_hex()));
        assert!(read.contains("1  pub fn value"));
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
        assert!(check_claims(&rows, "demo", "README.md").is_ok());
        assert!(
            check_claims(
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
                .contains("stale")
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
    fn claims_are_required_and_rechecked_after_each_write() {
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
        assert!(check_claims(&board("Kimi", "src/"), "demo", "src/lib.rs").is_err());
        let rows = board("Kimi", "src/lib.rs").replace(
            "## Handoffs",
            "| today | Claude | demo | as listed | task | in progress |\n## Handoffs",
        );
        assert!(check_claims(&rows, "demo", "src/lib.rs").is_err());
        let split = rows.replace("as listed", "src/other.rs");
        assert!(check_claims(&split, "demo", "src/lib.rs").is_ok());
    }

    #[test]
    fn unsafe_paths_ignored_files_and_untracked_existing_files_are_refused() {
        let (root, toolbox) = fixture();
        for path in [
            "../outside.rs",
            "/tmp/outside.rs",
            "demo/src/no.json",
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
        use crate::chat::{ChatFormat, run_with};
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
