//! One bounded shell session: pipe I/O offloaded to at most four workers,
//! with a hard deadline dispatched by the platform proactor. No polling loop.
//! Pipe-based command interaction, not a PTY or a filesystem sandbox.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak, mpsc};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use loadngo_inference::tools::Tool;
use loadngo_proactor::{CompletionKind, PlatformPort, ProactorHandle, new_platform_proactor};
use serde_json::{Value, json};

const CAPACITY: usize = 64 * 1024;
const PAGE: usize = 6 * 1024;
const MAX_INPUT: usize = 16 * 1024;

struct Output {
    bytes: VecDeque<u8>,
    dropped: usize,
    exit: Option<String>,
    reason: Option<String>,
}

type SharedOutput = Arc<(Mutex<Output>, Condvar)>;
type CancelTarget = (u32, Weak<(Mutex<Output>, Condvar)>);
static ACTIVE: OnceLock<Mutex<Option<CancelTarget>>> = OnceLock::new();

/// Called by the existing Ctrl-C handler, including during generation.
pub fn interrupt() {
    if let Some(active) = ACTIVE.get() {
        if let Ok(active) = active.lock() {
            if let Some((pid, output)) = active.as_ref() {
                if let Some(output) = output.upgrade() {
                    stop(*pid, &output, "cancelled by Ctrl-C");
                }
            }
        }
    }
}

#[cfg(unix)]
fn kill_tree(pid: u32) {
    use nix::sys::signal::{Signal, killpg};
    use nix::unistd::Pid;
    if let Ok(pid) = i32::try_from(pid) {
        let _ = killpg(Pid::from_raw(pid), Signal::SIGKILL);
    }
}

fn stop(pid: u32, output: &SharedOutput, reason: &str) {
    let mut state = output
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.exit.is_none() {
        state.reason.get_or_insert_with(|| reason.to_string());
        kill_tree(pid);
    }
}

fn capture(mut reader: impl Read, output: &SharedOutput) {
    let mut buf = [0_u8; 4099];
    let mut tail = 0;
    loop {
        match reader.read(&mut buf[tail..]) {
            Ok(0) => break,
            Ok(n) => {
                let len = tail + n;
                // Hold a split UTF-8 code point in this pipe reader, so another
                // pipe or a terminal_read cannot interleave with its bytes.
                let mut complete = 0;
                while complete < len {
                    match std::str::from_utf8(&buf[complete..len]) {
                        Ok(_) => {
                            complete = len;
                        }
                        Err(error) => {
                            complete += error.valid_up_to();
                            match error.error_len() {
                                Some(n) => complete += n,
                                None => break,
                            }
                        }
                    }
                }
                append_output(output, &buf[..complete]);
                buf.copy_within(complete..len, 0);
                tail = len - complete;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    append_output(output, &buf[..tail]);
}

fn append_output(output: &SharedOutput, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let mut state = output
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let discard = (state.bytes.len() + bytes.len()).saturating_sub(CAPACITY);
    state.bytes.drain(..discard);
    state.dropped += discard;
    state.bytes.extend(bytes);
    if discard > 0 {
        while state.bytes.front().is_some_and(|b| b & 0xc0 == 0x80) {
            state.bytes.pop_front();
            state.dropped += 1;
        }
    }
    output.1.notify_all();
}

struct Session {
    id: u64,
    pid: u32,
    output: SharedOutput,
    input: Option<mpsc::SyncSender<String>>,
    workers: Vec<JoinHandle<()>>,
    proactor: ProactorHandle<PlatformPort>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // The output allocation is reused: an old PID must never refer to the
        // next session's reset running state in the Ctrl-C handler.
        if let Some(active) = ACTIVE.get() {
            let mut active = active
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if active.as_ref().is_some_and(|(pid, _)| *pid == self.pid) {
                active.take();
            }
        }
        stop(self.pid, &self.output, "session closed");
        self.input.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        let _ = self.proactor.stop();
    }
}

struct Terminal {
    root: PathBuf,
    session: RefCell<Option<Session>>,
    next: Cell<u64>,
    // Retain and reuse this bounded buffer across commands.
    output: SharedOutput,
}

/// Register bounded command sessions for a workspace.
///
/// # Errors
/// If the workspace cannot be resolved, or root instruction-file protection is
/// required on a platform without a supported command sandbox.
pub fn tools(root: &Path) -> Result<Vec<Box<dyn Tool>>, String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    #[cfg(not(target_os = "macos"))]
    if ["AGENTS.md", "CLAUDE.md"]
        .iter()
        .any(|name| root.join(name).exists())
    {
        return Err("terminal commands in this workspace need root instruction-file protection, currently implemented only on macOS".into());
    }
    let terminal = Rc::new(Terminal {
        root,
        session: RefCell::new(None),
        next: Cell::new(1),
        output: Arc::new((
            Mutex::new(Output {
                bytes: VecDeque::with_capacity(CAPACITY),
                dropped: 0,
                exit: None,
                reason: None,
            }),
            Condvar::new(),
        )),
    });
    Ok([
        "terminal_exec",
        "terminal_read",
        "terminal_write",
        "terminal_stop",
    ]
    .into_iter()
    .map(|name| Box::new(TerminalTool(Rc::clone(&terminal), name)) as Box<dyn Tool>)
    .collect())
}

fn string<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {key}"))
}

fn integer(args: &Value, key: &str, default: u64, max: u64) -> Result<u64, String> {
    let n = match args.get(key) {
        None => default,
        Some(v) => v
            .as_u64()
            .ok_or_else(|| format!("{key} must be a nonnegative integer"))?,
    };
    if n > max {
        return Err(format!("{key} must be at most {max}"));
    }
    Ok(n)
}

impl Terminal {
    #[allow(clippy::too_many_lines)]
    fn execute(&self, args: &Value) -> Result<String, String> {
        let command = string(args, "command")?;
        if command.trim().is_empty() || command.len() > MAX_INPUT || command.contains('\0') {
            return Err("command must be nonempty, at most 16 KiB, without NUL".into());
        }
        let timeout = integer(args, "timeout_seconds", 120, 1800)?;
        if timeout == 0 {
            return Err("timeout_seconds must be at least 1".into());
        }
        let cwd = args
            .get("cwd")
            .map_or(Ok("."), |v| v.as_str().ok_or("cwd must be a string"))?;
        let cwd = self
            .root
            .join(cwd)
            .canonicalize()
            .map_err(|e| format!("cwd: {e}"))?;
        if !cwd.is_dir() {
            return Err("cwd is not a directory".into());
        }
        let mut slot = self.session.borrow_mut();
        if let Some(s) = slot.as_ref() {
            let state = s.output.0.lock().map_err(|e| e.to_string())?;
            if state.exit.is_none() || !state.bytes.is_empty() {
                return Err(format!(
                    "session {} still runs or has unread output; read/stop it first",
                    s.id
                ));
            }
        }
        slot.take();
        let proactor = new_platform_proactor().map_err(|e| e.to_string())?;
        let handle = proactor.handle();
        let mut cmd = shell(&self.root, command)?;
        cmd.current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().map_err(|e| format!("start command: {e}"))?;
        let pid = child.id();
        {
            let mut state = self.output.0.lock().map_err(|e| e.to_string())?;
            state.bytes.clear();
            state.dropped = 0;
            state.exit = None;
            state.reason = None;
        }
        let output = Arc::clone(&self.output);
        let timeout_output = Arc::clone(&output);
        if let Err(e) = handle.defer_for(
            Duration::from_secs(timeout),
            CompletionKind::Timer,
            0,
            move |_| {
                stop(pid, &timeout_output, "deadline exceeded");
            },
        ) {
            kill_tree(pid);
            let _ = child.wait();
            return Err(e.to_string());
        }
        let stdout = child.stdout.take().ok_or("missing stdout pipe")?;
        let stderr = child.stderr.take().ok_or("missing stderr pipe")?;
        let mut stdin = child.stdin.take().ok_or("missing stdin pipe")?;
        let (send, recv) = mpsc::sync_channel::<String>(4);
        let out = Arc::clone(&output);
        let reader = thread::spawn(move || capture(stdout, &out));
        let out = Arc::clone(&output);
        let errors = thread::spawn(move || capture(stderr, &out));
        let writer = thread::spawn(move || {
            while let Ok(text) = recv.recv() {
                if stdin
                    .write_all(text.as_bytes())
                    .and_then(|()| stdin.flush())
                    .is_err()
                {
                    break;
                }
            }
        });
        let completed = Arc::clone(&output);
        let finish = handle.clone();
        let waiter = thread::spawn(move || {
            let status = child
                .wait()
                .map_or_else(|e| format!("wait error: {e}"), |s| s.to_string());
            // Do not retain background descendants or pipes after the shell exits.
            kill_tree(pid);
            let _ = reader.join();
            let _ = errors.join();
            // Publish even if the dispatcher failed. The condition variable
            // wakes terminal_read; stopping wakes the blocked proactor.
            completed
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .exit = Some(status);
            completed.1.notify_all();
            let _ = finish.stop();
        });
        let dispatcher = thread::spawn(move || {
            while proactor.handle().is_running() {
                if proactor.run_once().is_err() {
                    stop(pid, &output, "proactor failed");
                    break;
                }
            }
        });
        let id = self.next.get();
        self.next.set(id + 1);
        *ACTIVE
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|e| e.to_string())? = Some((pid, Arc::downgrade(&self.output)));
        *slot = Some(Session {
            id,
            pid,
            output: Arc::clone(&self.output),
            input: Some(send),
            workers: vec![writer, waiter, dispatcher],
            proactor: handle,
        });
        drop(slot);
        self.read(id, 1000)
    }

    fn read(&self, id: u64, wait_ms: u64) -> Result<String, String> {
        let slot = self.session.borrow();
        let session = slot
            .as_ref()
            .filter(|s| s.id == id)
            .ok_or("unknown terminal session")?;
        let (lock, changed) = &*session.output;
        let state = lock.lock().map_err(|e| e.to_string())?;
        let (mut state, _) = changed
            .wait_timeout_while(state, Duration::from_millis(wait_ms), |s| {
                s.bytes.is_empty() && s.exit.is_none()
            })
            .map_err(|e| e.to_string())?;
        let count = PAGE.min(state.bytes.len());
        let mut bytes: Vec<u8> = state.bytes.drain(..count).collect();
        while state.bytes.front().is_some_and(|b| b & 0xc0 == 0x80) {
            bytes.push(state.bytes.pop_front().expect("front exists"));
        }
        let dropped = std::mem::take(&mut state.dropped);
        Ok(json!({
            "session_id": id, "running": state.exit.is_none(),
            "status": state.exit, "stop_reason": state.reason,
            "output": String::from_utf8_lossy(&bytes), "more_output": !state.bytes.is_empty(),
            "dropped_bytes": dropped
        })
        .to_string())
    }
}

#[cfg(target_os = "macos")]
fn shell(root: &Path, command: &str) -> Result<Command, String> {
    use std::os::unix::fs::MetadataExt;
    let mut protected = Vec::new();
    for name in ["AGENTS.md", "CLAUDE.md"] {
        let path = root.join(name);
        protected.push(path.clone());
        match std::fs::metadata(&path) {
            Ok(meta) => {
                if meta.nlink() != 1 {
                    return Err(format!(
                        "{} has hard-link aliases; cannot protect terminal writes",
                        path.display()
                    ));
                }
                let real = path.canonicalize().map_err(|e| e.to_string())?;
                if real != path {
                    protected.push(real);
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    let literals = |paths: &[PathBuf]| {
        paths
            .iter()
            .map(|p| {
                format!(
                    "(literal {})",
                    serde_json::to_string(&p.to_string_lossy()).expect("path string")
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    // Prevent moving the workspace or an ancestor to give protected files new paths.
    let ancestors = root.ancestors().map(Path::to_path_buf).collect::<Vec<_>>();
    let profile = format!(
        "(version 1)(allow default)(deny file-write* {})(deny file-write-unlink {})",
        literals(&protected),
        literals(&ancestors)
    );
    let mut shell = Command::new("/usr/bin/sandbox-exec");
    shell.args(["-p", &profile, "/bin/sh", "-c", command]);
    Ok(shell)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn shell(root: &Path, command: &str) -> Result<Command, String> {
    if ["AGENTS.md", "CLAUDE.md"]
        .iter()
        .any(|name| root.join(name).exists())
    {
        return Err("root instruction-file protection for terminal commands is currently implemented only on macOS".into());
    }
    let mut shell = Command::new("/bin/sh");
    shell.args(["-c", command]);
    Ok(shell)
}

struct TerminalTool(Rc<Terminal>, &'static str);

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::time::Instant;

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "run explicitly outside a nested sandbox to initialize sandbox-exec"]
    fn shell_protects_root_instructions_and_allows_repo_copies() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        for name in ["AGENTS.md", "CLAUDE.md"] {
            std::fs::write(root.join(name), "keep\n").unwrap();
        }
        std::fs::create_dir(root.join("repo")).unwrap();
        for command in [
            "printf changed > AGENTS.md",
            "printf changed > agents.md",
            "rm CLAUDE.md",
            "mv AGENTS.md moved.md",
            "printf replacement > replacement.md; mv replacement.md AGENTS.md",
            "ln -s AGENTS.md alias.md; printf changed > alias.md",
            "ln CLAUDE.md hard.md && printf changed > hard.md",
        ] {
            let output = shell(&root, command)
                .unwrap()
                .current_dir(&root)
                .output()
                .unwrap();
            assert!(!output.status.success(), "{command}: {output:?}");
            for name in ["AGENTS.md", "CLAUDE.md"] {
                assert_eq!(std::fs::read_to_string(root.join(name)).unwrap(), "keep\n");
            }
        }
        let output = shell(&root, "printf allowed > ordinary.md; printf local > repo/AGENTS.md; printf local > repo/CLAUDE.md")
            .unwrap().current_dir(&root).output().unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            std::fs::read_to_string(root.join("ordinary.md")).unwrap(),
            "allowed"
        );
        for name in ["AGENTS.md", "CLAUDE.md"] {
            assert_eq!(
                std::fs::read_to_string(root.join("repo").join(name)).unwrap(),
                "local"
            );
        }
        // Refuse a pre-existing hard-link alias instead of leaving a bypass open.
        std::fs::hard_link(root.join("AGENTS.md"), root.join("preexisting.md")).unwrap();
        assert!(shell(&root, "true").unwrap_err().contains("hard-link"));
    }

    #[allow(clippy::needless_pass_by_value)] // Inline json! keeps tool sequences legible.
    fn call(tools: &[Box<dyn Tool>], name: &str, args: Value) -> Value {
        serde_json::from_str(
            &tools
                .iter()
                .find(|t| t.name() == name)
                .unwrap()
                .call(&args)
                .unwrap(),
        )
        .unwrap()
    }

    fn finish(tools: &[Box<dyn Tool>], mut result: Value) -> (String, Value) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut text = String::new();
        loop {
            text.push_str(result["output"].as_str().unwrap());
            if result["running"] == false && result["more_output"] == false {
                return (text, result);
            }
            assert!(
                Instant::now() < deadline,
                "session did not finish: {result}"
            );
            result = call(
                tools,
                "terminal_read",
                json!({"session_id":result["session_id"],"wait_ms":100}),
            );
        }
    }

    #[test]
    fn command_stdin_exit_status_and_repeated_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let tools = tools(dir.path()).unwrap();
        let first = call(
            &tools,
            "terminal_exec",
            json!({"command":"read line; printf '%s\\n' \"$line\"; printf 'error\\n' >&2; exit 7"}),
        );
        assert_eq!(first["running"], true);
        tools[2].call(&json!({"session_id":first["session_id"],"input":"café 海 🦀\n","close_stdin":true})).unwrap();
        let (text, result) = finish(&tools, first);
        assert!(
            text.contains("café 海 🦀") && text.contains("error"),
            "{text}"
        );
        assert_eq!(result["status"], "exit status: 7");
        let second = call(&tools, "terminal_exec", json!({"command":"pwd"}));
        assert_ne!(second["session_id"], result["session_id"]);
        let (text, _) = finish(&tools, second);
        assert_eq!(
            text.trim(),
            dir.path().canonicalize().unwrap().to_str().unwrap()
        );
        assert!(tools[1].call(&json!({"session_id":9999})).is_err());
    }

    #[test]
    fn deadline_and_stop_kill_silent_commands_and_children() {
        let dir = tempfile::tempdir().unwrap();
        let tools = tools(dir.path()).unwrap();
        let start = Instant::now();
        let first = call(
            &tools,
            "terminal_exec",
            json!({"command":"sleep 60 & wait","timeout_seconds":1}),
        );
        let (_, result) = finish(&tools, first);
        assert_eq!(result["stop_reason"], "deadline exceeded");
        assert!(start.elapsed() < Duration::from_secs(5));
        let first = call(
            &tools,
            "terminal_exec",
            json!({"command":"sleep 60 & wait"}),
        );
        tools[3]
            .call(&json!({"session_id":first["session_id"]}))
            .unwrap();
        let (_, result) = finish(&tools, first);
        assert_eq!(result["stop_reason"], "stopped by tool");
    }

    #[test]
    fn output_is_bounded_and_drop_cleans_up_running_session() {
        let dir = tempfile::tempdir().unwrap();
        let tools = tools(dir.path()).unwrap();
        let first = call(&tools, "terminal_exec", json!({"command":"yes abcdef"}));
        let id = first["session_id"].clone();
        assert!(tools[0].call(&json!({"command":"echo second"})).is_err());
        assert!(
            tools[2]
                .call(&json!({"session_id":id,"input":"x".repeat(MAX_INPUT + 1)}))
                .is_err()
        );
        tools[3].call(&json!({"session_id":id})).unwrap();
        let _ = finish(&tools, first);
        let result = call(&tools, "terminal_exec", json!({"command":"sleep 60"}));
        assert_eq!(result["running"], true);
        let start = Instant::now();
        drop(tools);
        assert!(start.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn capture_preserves_split_unicode_and_bounds_unread_output() {
        struct ByteReader<'a>(&'a [u8]);
        impl Read for ByteReader<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.0.read(&mut buf[..1])
            }
        }
        let output = Arc::new((
            Mutex::new(Output {
                bytes: VecDeque::with_capacity(CAPACITY),
                dropped: 0,
                exit: None,
                reason: None,
            }),
            Condvar::new(),
        ));
        let text = "海🦀café".repeat(10_000);
        capture(ByteReader(text.as_bytes()), &output);
        let state = output.0.lock().unwrap();
        let retained: Vec<_> = state.bytes.iter().copied().collect();
        assert!(retained.len() <= CAPACITY);
        assert_eq!(state.dropped + retained.len(), text.len());
        let retained = std::str::from_utf8(&retained).unwrap();
        assert!(text.ends_with(retained));
        assert!(state.dropped > 0);
    }

    #[test]
    fn shell_exit_kills_background_pipe_holders_and_rejects_late_input() {
        let dir = tempfile::tempdir().unwrap();
        let tools = tools(dir.path()).unwrap();
        let start = Instant::now();
        let first = call(
            &tools,
            "terminal_exec",
            json!({"command":"sleep 60 & printf done"}),
        );
        let (text, result) = finish(&tools, first);
        assert_eq!(text, "done");
        assert_eq!(result["status"], "exit status: 0");
        assert!(start.elapsed() < Duration::from_secs(3));
        assert!(
            tools[2]
                .call(&json!({"session_id":result["session_id"],"input":"late"}))
                .unwrap_err()
                .contains("exited")
        );
    }
}

impl Tool for TerminalTool {
    fn name(&self) -> &'static str {
        self.1
    }
    fn description(&self) -> &'static str {
        match self.1 {
            "terminal_exec" => {
                "Run a shell command with your OS user's permissions. One session at a time; returns output and session_id. Relative cwd starts at the workspace. Use for builds/tests. No PTY; stdout/stderr combined. Default deadline 120s, maximum 1800s. Side effects persist."
            }
            "terminal_read" => {
                "Read more command output and exit status. May repeat; wait_ms defaults to 1000, maximum 10000. About 6 KiB returned, 64 KiB retained; dropped_bytes reports overflow."
            }
            "terminal_write" => {
                "Send input to a running session's stdin; include newline to submit a line. close_stdin sends EOF after queued input. At most 16 KiB per call."
            }
            _ => {
                "Stop the command session and its ordinary descendants. Read afterward for final output/status."
            }
        }
    }
    fn parameters(&self) -> Value {
        match self.1 {
            "terminal_exec" => {
                json!({"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"},"timeout_seconds":{"type":"integer","minimum":1,"maximum":1800}},"required":["command"]})
            }
            "terminal_read" => {
                json!({"type":"object","properties":{"session_id":{"type":"integer"},"wait_ms":{"type":"integer","minimum":0,"maximum":10000}},"required":["session_id"]})
            }
            "terminal_write" => {
                json!({"type":"object","properties":{"session_id":{"type":"integer"},"input":{"type":"string"},"close_stdin":{"type":"boolean"}},"required":["session_id"]})
            }
            _ => {
                json!({"type":"object","properties":{"session_id":{"type":"integer"}},"required":["session_id"]})
            }
        }
    }
    fn call(&self, args: &Value) -> Result<String, String> {
        if self.1 == "terminal_exec" {
            return self.0.execute(args);
        }
        let id = args
            .get("session_id")
            .and_then(Value::as_u64)
            .ok_or("missing integer session_id")?;
        if self.1 == "terminal_read" {
            return self.0.read(id, integer(args, "wait_ms", 1000, 10_000)?);
        }
        let mut slot = self.0.session.borrow_mut();
        let session = slot
            .as_mut()
            .filter(|s| s.id == id)
            .ok_or("unknown terminal session")?;
        if self.1 == "terminal_stop" {
            stop(session.pid, &session.output, "stopped by tool");
            return Ok("stop requested; terminal_read returns final status".into());
        }
        if session
            .output
            .0
            .lock()
            .map_err(|e| e.to_string())?
            .exit
            .is_some()
        {
            return Err("command has exited; stdin is closed".into());
        }
        let input = args
            .get("input")
            .map_or(Ok(""), |v| v.as_str().ok_or("input must be a string"))?;
        if input.len() > MAX_INPUT {
            return Err("input exceeds 16 KiB".into());
        }
        let close = args.get("close_stdin").map_or(Ok(false), |value| {
            value.as_bool().ok_or("close_stdin must be a boolean")
        })?;
        let sender = session.input.as_ref().ok_or("stdin already closed")?;
        if !input.is_empty() {
            sender
                .try_send(input.to_string())
                .map_err(|e| format!("stdin queue: {e}"))?;
        }
        if close {
            session.input.take();
        }
        Ok("input accepted; terminal_read returns output/status".into())
    }
}
