//! Board tools: how Kimi reads and adds to the workspace's coordination board
//! (`AGENT-BOARD.md` in the pudding folder, see its `COLLABORATION.md`).
//!
//! The board is ~150 KB, more than her whole context, so she reads it a section at a
//! time, a few rows at once. She may add rows (never edit or delete them) to two
//! sections, Active claims and Handoffs, always dated and signed "Kimi"; Jay's huddle
//! is Jay's, and the other sections are only read. Each addition rewrites the file
//! atomically (a temporary file renamed over it).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use loadngo_inference::tools::Tool;
use serde_json::{Value, json};

/// Longest row text returned for one row; handoff rows can run to thousands of
/// characters.
const MAX_ROW_CHARS: usize = 1200;
const MAX_ROWS: usize = 12;
/// Longest cell she may write.
const MAX_CELL_CHARS: usize = 1500;

/// Sections she may add rows to, with the cells she supplies (the date and "Kimi"
/// come first automatically).
const WRITABLE: [(&str, &[&str]); 2] = [
    ("Active claims", &["repo / area", "paths", "task", "status"]),
    ("Handoffs", &["what", "state", "verified", "open"]),
];

pub struct Board {
    path: PathBuf,
    today: String,
}

impl Board {
    /// The board at `path`, with `today` (for example `2026-09-27`) for new rows.
    pub fn new(path: impl Into<PathBuf>, today: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            today: today.into(),
        }
    }

    pub fn into_tools(self) -> Vec<Box<dyn Tool>> {
        let shared = std::rc::Rc::new(self);
        vec![
            Box::new(Sections(std::rc::Rc::clone(&shared))),
            Box::new(Read(std::rc::Rc::clone(&shared))),
            Box::new(AddRow(shared)),
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

/// `cell` made safe for one table cell: one line, pipes escaped, bounded.
fn cell(text: &str) -> Result<String, String> {
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > MAX_CELL_CHARS {
        return Err(format!("a cell is limited to {MAX_CELL_CHARS} characters"));
    }
    Ok(one_line.replace('|', "\\|"))
}

/// `text` with `row` added as the newest row of section `name`: after the header when
/// the section's table starts with one, otherwise before its first row.
pub fn with_row(text: &str, name: &str, row: &str) -> Result<String, String> {
    let all = sections(text);
    let (_, from, to) = find(&all, name)?;
    let lines: Vec<&str> = text.lines().collect();
    let first = (*from..*to)
        .find(|&i| lines[i].starts_with('|'))
        .ok_or_else(|| format!("section {name:?} has no table"))?;
    let at = if lines.get(first + 1).is_some_and(|l| is_separator(l)) {
        first + 2
    } else {
        first
    };
    let mut out: Vec<&str> = Vec::with_capacity(lines.len() + 1);
    out.extend_from_slice(&lines[..at]);
    out.push(row);
    out.extend_from_slice(&lines[at..]);
    let mut joined = out.join("\n");
    if text.ends_with('\n') {
        joined.push('\n');
    }
    Ok(joined)
}

struct Sections(std::rc::Rc<Board>);
struct Read(std::rc::Rc<Board>);
struct AddRow(std::rc::Rc<Board>);

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
        out.push_str("Rows are newest first. You may add rows to Active claims and Handoffs.");
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

impl Tool for AddRow {
    fn name(&self) -> &'static str {
        "board_add_row"
    }
    fn description(&self) -> &'static str {
        "Add a row, signed Kimi and dated today, to the top of a board section. Active claims takes 4 cells: repo / area, paths, task, status. Handoffs takes 4: what, state, verified, open. Rows are never edited or deleted."
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {
            "section": {"type": "string", "enum": ["Active claims", "Handoffs"]},
            "cells": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4}},
            "required": ["section", "cells"]})
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        let name = args
            .get("section")
            .and_then(Value::as_str)
            .ok_or("`section` is required")?;
        let (section, labels) = WRITABLE
            .iter()
            .find(|(s, _)| s.eq_ignore_ascii_case(name.trim()))
            .ok_or_else(|| {
                format!("rows can be added only to Active claims or Handoffs, not {name:?}")
            })?;
        let cells: Vec<String> = args
            .get("cells")
            .and_then(Value::as_array)
            .ok_or("`cells` must be a list of text")?
            .iter()
            .map(|c| {
                c.as_str()
                    .ok_or("every cell must be text")
                    .map(str::to_string)
            })
            .collect::<Result<_, _>>()?;
        if cells.len() != labels.len() {
            return Err(format!(
                "{section} takes {} cells: {}",
                labels.len(),
                labels.join(", ")
            ));
        }
        let mut row = format!("| {} | Kimi |", self.0.today);
        for c in &cells {
            let _ = write!(row, " {} |", cell(c)?);
        }
        let text = self.0.text()?;
        let updated = with_row(&text, section, &row)?;
        let tmp = self.0.path.with_extension("md.kimi-tmp");
        std::fs::write(&tmp, updated).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &self.0.path)
            .map_err(|e| format!("{}: {e}", self.0.path.display()))?;
        Ok(format!("added to {section}: {row}"))
    }
}

/// The board in `base`, if there is one.
pub fn board_in(base: &Path) -> Option<PathBuf> {
    let path = base.join("AGENT-BOARD.md");
    path.is_file().then_some(path)
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
    fn new_rows_go_under_the_header_or_before_the_first_row() {
        let claims = with_row(BOARD, "Active claims", "| NEW |").unwrap();
        let at_new = claims.find("| NEW |").unwrap();
        assert!(at_new > claims.find("|---").unwrap() && at_new < claims.find("| d1 |").unwrap());
        let handoffs = with_row(BOARD, "Handoffs", "| NEW |").unwrap();
        assert!(handoffs.find("| NEW |").unwrap() < handoffs.find("| d2 |").unwrap());
        assert!(handoffs.ends_with('\n'));
    }

    #[test]
    fn cells_are_one_line_with_pipes_escaped() {
        assert_eq!(cell("a | b\nc").unwrap(), "a \\| b c");
        assert!(cell(&"x".repeat(MAX_CELL_CHARS + 1)).is_err());
    }

    #[test]
    fn adding_is_limited_to_claims_and_handoffs_and_signed_kimi() {
        let dir = std::env::temp_dir().join(format!("kimi-board-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("AGENT-BOARD.md");
        std::fs::write(&path, BOARD).unwrap();
        let tools = Board::new(&path, "2026-09-27").into_tools();
        let add = &tools[2];
        let huddle = add.call(&json!({"section": "Jay's huddle", "cells": ["a", "b", "c", "d"]}));
        assert!(
            huddle
                .unwrap_err()
                .contains("only to Active claims or Handoffs")
        );
        assert!(
            add.call(&json!({"section": "Handoffs", "cells": ["a"]}))
                .is_err()
        );
        add.call(&json!({"section": "Handoffs", "cells": ["Drafted outreach", "done", "read back", "none"]}))
            .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            text.contains("| 2026-09-27 | Kimi | Drafted outreach | done | read back | none |")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
