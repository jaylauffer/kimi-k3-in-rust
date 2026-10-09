//! Board tools: how Kimi reads the workspace's coordination board (`AGENT-BOARD.md` in
//! the pudding folder, see its `COLLABORATION.md`), a section at a time.
//!
//! The board is larger than her whole context, so she reads it a section at a time, a
//! few rows at once. She no longer writes it: since 2026-10-09 her edits go through
//! loadngo's editing tools, which claim a repository on the board at her first write and
//! turn the claim into a handoff when the chat ends (`loadngo docs/AGENT_LOOP.md`).

use std::fmt::Write as _;
use std::path::PathBuf;

use loadngo_inference::tools::Tool;
use serde_json::{Value, json};

/// Longest row text returned for one row; handoff rows can run to thousands of
/// characters.
const MAX_ROW_CHARS: usize = 1200;
const MAX_ROWS: usize = 12;

pub struct Board {
    path: PathBuf,
}

impl Board {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// `board_sections` and `board_read`.
    pub fn into_tools(self) -> Vec<Box<dyn Tool>> {
        let shared = std::rc::Rc::new(self);
        vec![
            Box::new(Sections(std::rc::Rc::clone(&shared))),
            Box::new(Read(shared)),
        ]
    }

    fn text(&self) -> Result<String, String> {
        std::fs::read_to_string(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))
    }
}

/// `(heading, first line, end line)` of every `## ` section.
fn sections(text: &str) -> Vec<(String, usize, usize)> {
    let lines: Vec<&str> = text.lines().collect();
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("## "))
        .map(|(i, _)| i)
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(n, &at)| {
            let end = starts.get(n + 1).copied().unwrap_or(lines.len());
            (lines[at][3..].trim().to_string(), at, end)
        })
        .collect()
}

fn is_separator(line: &str) -> bool {
    line.starts_with("|---") || line.starts_with("| ---")
}

/// The table rows of a section, without header or separator rows.
fn rows<'a>(lines: &[&'a str], from: usize, to: usize) -> Vec<&'a str> {
    let mut out = Vec::new();
    for i in from..to {
        let line = lines[i];
        if !line.starts_with('|') || is_separator(line) {
            continue;
        }
        if lines.get(i + 1).is_some_and(|next| is_separator(next)) {
            continue; // a header row
        }
        out.push(line);
    }
    out
}

fn find<'a>(
    all: &'a [(String, usize, usize)],
    name: &str,
) -> Result<&'a (String, usize, usize), String> {
    let wanted = name.trim().to_lowercase();
    all.iter()
        .find(|(h, _, _)| h.to_lowercase() == wanted)
        .or_else(|| {
            all.iter()
                .find(|(h, _, _)| h.to_lowercase().contains(&wanted))
        })
        .ok_or_else(|| {
            format!(
                "no section {name:?}; sections: {}",
                all.iter()
                    .map(|s| s.0.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

struct Sections(std::rc::Rc<Board>);
struct Read(std::rc::Rc<Board>);

impl Tool for Sections {
    fn name(&self) -> &'static str {
        "board_sections"
    }
    fn description(&self) -> &'static str {
        "List the sections of the coordination board (AGENT-BOARD.md) and how many rows each has."
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}})
    }
    fn call(&self, _: &Value) -> Result<String, String> {
        let text = self.0.text()?;
        let lines: Vec<&str> = text.lines().collect();
        let mut out = String::new();
        for (heading, from, to) in sections(&text) {
            let n = rows(&lines, from, to).len();
            let _ = writeln!(out, "{heading}: {n} rows");
        }
        out.push_str("Rows are newest first.");
        Ok(out)
    }
}

impl Tool for Read {
    fn name(&self) -> &'static str {
        "board_read"
    }
    fn description(&self) -> &'static str {
        "Read rows of one board section (for example \"Active claims\", \"Handoffs\", \"Jay's huddle\", \"Shared devices\"), newest first, at most 12 per call; long rows are shortened."
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "section": {"type": "string"},
            "start": {"type": "integer", "description": "first row, 0 = newest (default 0)"},
            "count": {"type": "integer", "description": "rows to return, at most 12 (default 6)"}},
            "required": ["section"]})
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        let name = args
            .get("section")
            .and_then(Value::as_str)
            .ok_or("`section` is required")?;
        let number = |key: &str, default: usize| {
            args.get(key)
                .and_then(Value::as_u64)
                .map_or(default, |n| usize::try_from(n).unwrap_or(usize::MAX))
        };
        let start = number("start", 0);
        let count = number("count", 6).clamp(1, MAX_ROWS);
        let text = self.0.text()?;
        let all = sections(&text);
        let (heading, from, to) = find(&all, name)?;
        let lines: Vec<&str> = text.lines().collect();
        let rows = rows(&lines, *from, *to);
        let mut out = format!(
            "{heading}: rows {start}..{} of {}\n",
            (start + count).min(rows.len()),
            rows.len()
        );
        for row in rows.iter().skip(start).take(count) {
            let shown: String = row.chars().take(MAX_ROW_CHARS).collect();
            out.push_str(&shown);
            if row.chars().count() > MAX_ROW_CHARS {
                out.push_str(" …(shortened)");
            }
            out.push('\n');
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOARD: &str = "# Board\n\n## Active claims\n\n| Since | Agent | Repo | Paths | Task | Status |\n|---|---|---|---|---|---|\n| d1 | Claude | r | p | t | active |\n\n## Handoffs\n\n| d2 | Claude | what | state | ok | none |\n| Date | Agent | What | State | Verified | Open |\n|---|---|---|---|---|---|\n| d0 | Codex | old | s | v | o |\n\n## Jay's huddle\n\n| When | From | Topic | Note |\n|---|---|---|---|\n";

    #[test]
    fn rows_skip_headers_and_come_newest_first() {
        let lines: Vec<&str> = BOARD.lines().collect();
        let all = sections(BOARD);
        let (_, f, t) = find(&all, "handoffs").unwrap();
        let rows = rows(&lines, *f, *t);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains("d2") && rows[1].contains("d0"));
        assert!(find(&all, "huddle").is_ok());
        assert!(find(&all, "nowhere").is_err());
    }

    #[test]
    fn the_tools_only_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("AGENT-BOARD.md");
        std::fs::write(&path, BOARD).unwrap();
        let tools = Board::new(&path).into_tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert_eq!(names, ["board_sections", "board_read"]);
        let read = tools[1].call(&json!({"section": "Handoffs"})).unwrap();
        assert!(read.starts_with("Handoffs: rows 0..2 of 2"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), BOARD);
    }
}
