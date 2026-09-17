//! Independent in-memory Pool/Backend model for D1 oracle tests.
//!
//! Slow and obvious. No Firecracker, no sockets, no live network, no wall clock.
//! Production (`rl/envs/src`) must never import this module.
//!
//! Tool payload shapes the implementer must accept (tiny interpreters):
//! - shell: `{"command": "echo hello"}` / `{"command": "cat /path"}`; `echo a > /path` writes
//! - python: `{"code": "print(1+1)"}` / `print("hi")` / `print(open('/path').read())`
//! - editor/notes: `{"op":"read"|"write","path":"...","content":"..."}`
//! - browser: `{"url":"..."}`
//! - lean: `{"code":"..."}`
//! - subagent: `{"task":"..."}` — forks a child from a snapshot of current files;
//!   `stdout_artifact.path` is the child `SandboxId` string

#![allow(dead_code)]
#![allow(clippy::needless_pass_by_value)]

use std::collections::BTreeMap;

use prometheus_envs::{
    Artifact, CallId, Error, ForkResult, GpuConfig, Image, ImageId, NowMs, OfflineWeb, PoolConfig,
    Result, SandboxId, SnapshotId, TamperFlag, Tool, ToolRequest, ToolResponse, GRADER_ROOT,
    SCHEMA_TOOL_REQUEST, SCHEMA_TOOL_RESPONSE, SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

const NETWORK_BINS: &[&str] = &[
    "curl",
    "wget",
    "nc",
    "netcat",
    "ssh",
    "scp",
    "sftp",
    "nmap",
    "ping",
    "telnet",
    "ftp",
    "host",
    "dig",
    "nslookup",
    "traceroute",
];

pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    hex_encode(&d)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Lowercase hex SHA-256 of sorted `(path + NUL + bytes)` per hidden-test file.
pub fn hidden_tests_hash(hidden: &BTreeMap<String, Vec<u8>>) -> String {
    let mut h = Sha256::new();
    for (path, bytes) in hidden {
        h.update(path.as_bytes());
        h.update([0u8]);
        h.update(bytes);
    }
    hex_encode(&h.finalize())
}

pub fn credential_like_key(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    if upper.contains("SECRET")
        || upper.contains("TOKEN")
        || upper.contains("PASSWORD")
        || upper.contains("CREDENTIAL")
    {
        return true;
    }
    upper.starts_with("AWS_") || upper.starts_with("SSH_")
}

pub fn is_grader_path(path: &str) -> bool {
    path == GRADER_ROOT || path.starts_with("/grader/")
}

pub fn artifact_for(bytes: &[u8]) -> Artifact {
    Artifact {
        content_hash: sha256_hex(bytes),
        bytes: bytes.len() as u64,
        path: None,
        media_type: None,
    }
}

fn empty_stderr() -> Artifact {
    artifact_for(b"")
}

pub struct RefPool {
    cfg: PoolConfig,
    images: BTreeMap<String, Image>,
    snapshots: BTreeMap<String, RefSnapshot>,
    sandboxes: BTreeMap<String, RefSandbox>,
    tamper: Vec<TamperFlag>,
    next: u64,
}

struct RefSnapshot {
    files: BTreeMap<String, Vec<u8>>,
    env: BTreeMap<String, String>,
}

struct RefSandbox {
    snapshot_id: SnapshotId,
    files: BTreeMap<String, Vec<u8>>,
    env: BTreeMap<String, String>,
    live: bool,
    gpu_used_s: u32,
}

impl RefPool {
    pub fn new(cfg: PoolConfig) -> Self {
        Self {
            cfg,
            images: BTreeMap::new(),
            snapshots: BTreeMap::new(),
            sandboxes: BTreeMap::new(),
            tamper: Vec::new(),
            next: 1,
        }
    }

    pub fn config(&self) -> &PoolConfig {
        &self.cfg
    }

    pub fn live_count(&self) -> usize {
        self.sandboxes.values().filter(|s| s.live).count()
    }

    pub fn tamper_flags(&self) -> &[TamperFlag] {
        &self.tamper
    }

    pub fn offline_web(&self) -> &OfflineWeb {
        &self.cfg.offline_web
    }

    pub fn gpu(&self) -> Option<&GpuConfig> {
        self.cfg.gpu.as_ref()
    }

    fn alloc(&mut self, prefix: &str) -> String {
        let id = format!("{prefix}-{}", self.next);
        self.next += 1;
        id
    }

    pub fn register_image(&mut self, mut image: Image) -> Result<ImageId> {
        for name in image.env.keys() {
            if credential_like_key(name) {
                return Err(Error::CredentialInSandbox { name: name.clone() });
            }
        }
        for path in image.hidden_tests.keys() {
            if !is_grader_path(path) {
                return Err(Error::HiddenTestsVisible);
            }
        }
        for path in image.agent_files.keys() {
            if is_grader_path(path) {
                return Err(Error::HiddenTestsVisible);
            }
        }
        image.hidden_tests_hash = hidden_tests_hash(&image.hidden_tests);
        let id = image.id.clone();
        self.images.insert(id.0.clone(), image);
        Ok(id)
    }

    pub fn snapshot_from_image(&mut self, image: &ImageId, _now: NowMs) -> Result<SnapshotId> {
        let img = self.images.get(&image.0).ok_or(Error::ImageNotFound)?;
        let mut files = img.agent_files.clone();
        files.extend(img.hidden_tests.clone());
        let env = img.env.clone();
        let sid = SnapshotId(self.alloc("snap"));
        self.snapshots
            .insert(sid.0.clone(), RefSnapshot { files, env });
        Ok(sid)
    }

    fn fork_one(&mut self, snapshot: &SnapshotId) -> Result<(SandboxId, u64)> {
        let snap = self
            .snapshots
            .get(&snapshot.0)
            .ok_or(Error::SnapshotNotFound)?;
        let files = snap.files.clone();
        let env = snap.env.clone();
        let id = SandboxId(self.alloc("sb"));
        self.sandboxes.insert(
            id.0.clone(),
            RefSandbox {
                snapshot_id: snapshot.clone(),
                files,
                env,
                live: true,
                gpu_used_s: 0,
            },
        );
        Ok((id, 0))
    }

    pub fn fork_group(
        &mut self,
        snapshot: &SnapshotId,
        n: usize,
        _now: NowMs,
    ) -> Result<ForkResult> {
        if !self.snapshots.contains_key(&snapshot.0) {
            return Err(Error::SnapshotNotFound);
        }
        if self.live_count() + n > self.cfg.capacity {
            return Err(Error::PoolExhausted {
                capacity: self.cfg.capacity,
            });
        }
        let mut sandboxes = Vec::with_capacity(n);
        let mut durations_ms = Vec::with_capacity(n);
        for _ in 0..n {
            let (id, d) = self.fork_one(snapshot)?;
            sandboxes.push(id);
            durations_ms.push(d);
        }
        Ok(ForkResult {
            sandboxes,
            durations_ms,
        })
    }

    pub fn fork_pair(
        &mut self,
        snapshot: &SnapshotId,
        now: NowMs,
    ) -> Result<(SandboxId, SandboxId)> {
        let g = self.fork_group(snapshot, 2, now)?;
        Ok((g.sandboxes[0].clone(), g.sandboxes[1].clone()))
    }

    pub fn snapshot(&mut self, sandbox: &SandboxId, _now: NowMs) -> Result<SnapshotId> {
        let sb = self
            .sandboxes
            .get(&sandbox.0)
            .ok_or(Error::SandboxNotFound)?;
        if !sb.live {
            return Err(Error::SandboxNotFound);
        }
        let files = sb.files.clone();
        let env = sb.env.clone();
        let sid = SnapshotId(self.alloc("snap"));
        self.snapshots
            .insert(sid.0.clone(), RefSnapshot { files, env });
        Ok(sid)
    }

    pub fn kill(&mut self, sandbox: &SandboxId, _now: NowMs) -> Result<()> {
        let sb = self
            .sandboxes
            .get_mut(&sandbox.0)
            .ok_or(Error::SandboxNotFound)?;
        if !sb.live {
            return Err(Error::SandboxNotFound);
        }
        sb.live = false;
        Ok(())
    }

    pub fn agent_view(&self, sandbox: &SandboxId) -> Result<BTreeMap<String, Vec<u8>>> {
        let sb = self.live_sb(sandbox)?;
        Ok(sb
            .files
            .iter()
            .filter(|(p, _)| !is_grader_path(p))
            .map(|(p, b)| (p.clone(), b.clone()))
            .collect())
    }

    pub fn grader_view(&self, sandbox: &SandboxId) -> Result<BTreeMap<String, Vec<u8>>> {
        let sb = self.live_sb(sandbox)?;
        Ok(sb
            .files
            .iter()
            .filter(|(p, _)| is_grader_path(p))
            .map(|(p, b)| (p.clone(), b.clone()))
            .collect())
    }

    fn live_sb(&self, sandbox: &SandboxId) -> Result<&RefSandbox> {
        let sb = self
            .sandboxes
            .get(&sandbox.0)
            .ok_or(Error::SandboxNotFound)?;
        if !sb.live {
            return Err(Error::SandboxNotFound);
        }
        Ok(sb)
    }

    fn live_sb_mut(&mut self, sandbox: &SandboxId) -> Result<&mut RefSandbox> {
        let sb = self
            .sandboxes
            .get_mut(&sandbox.0)
            .ok_or(Error::SandboxNotFound)?;
        if !sb.live {
            return Err(Error::SandboxNotFound);
        }
        Ok(sb)
    }

    pub fn call(
        &mut self,
        sandbox: &SandboxId,
        req: ToolRequest,
        now: NowMs,
    ) -> Result<ToolResponse> {
        let _ = self.live_sb(sandbox)?;
        if req.schema_id != SCHEMA_TOOL_REQUEST || req.schema_version != SCHEMA_VERSION {
            return Err(Error::BadToolRequest(format!(
                "schema {} v{}",
                req.schema_id, req.schema_version
            )));
        }
        let timeout_s = req.timeout_s.unwrap_or(self.cfg.default_timeout_s);
        if timeout_s == 0 {
            return Err(Error::Timeout);
        }
        match req.tool {
            Tool::Shell => self.tool_shell(sandbox, &req, now, timeout_s),
            Tool::Python => self.tool_python(sandbox, &req, now, timeout_s),
            Tool::Editor => self.tool_editor(sandbox, &req, now, false),
            Tool::Notes => self.tool_editor(sandbox, &req, now, true),
            Tool::Browser => self.tool_browser(&req),
            Tool::Lean => self.tool_lean(&req),
            Tool::Subagent => self.tool_subagent(sandbox, &req, now),
        }
    }

    fn ok_resp(
        &self,
        req: &ToolRequest,
        stdout: &[u8],
        extra_path: Option<String>,
    ) -> ToolResponse {
        let mut stdout_artifact = artifact_for(stdout);
        stdout_artifact.path = extra_path;
        ToolResponse {
            schema_id: SCHEMA_TOOL_RESPONSE.to_string(),
            schema_version: SCHEMA_VERSION,
            call_id: req.call_id.clone(),
            ok: true,
            exit_code: Some(0),
            stdout_artifact,
            stderr_artifact: empty_stderr(),
            truncated: false,
            duration_ms: 0,
            snapshot_id_after: req.env_snapshot_id.clone(),
            error: None,
        }
    }

    fn fail_resp(&self, req: &ToolRequest, stderr: &str) -> ToolResponse {
        ToolResponse {
            schema_id: SCHEMA_TOOL_RESPONSE.to_string(),
            schema_version: SCHEMA_VERSION,
            call_id: req.call_id.clone(),
            ok: false,
            exit_code: Some(1),
            stdout_artifact: artifact_for(b""),
            stderr_artifact: artifact_for(stderr.as_bytes()),
            truncated: false,
            duration_ms: 0,
            snapshot_id_after: req.env_snapshot_id.clone(),
            error: Some(stderr.to_string()),
        }
    }

    fn payload_str(req: &ToolRequest, key: &str) -> Option<String> {
        req.payload.get(key)?.as_str().map(str::to_string)
    }

    fn flag_write(&mut self, sandbox: &SandboxId, path: &str, req: &ToolRequest, now: NowMs) {
        self.tamper.push(TamperFlag {
            sandbox: sandbox.clone(),
            path: path.to_string(),
            call_id: req.call_id.clone(),
            now_ms: now,
        });
    }

    fn gpu_ish(text: &str) -> bool {
        let l = text.to_ascii_lowercase();
        l.contains("cuda") || l.contains("nvidia") || l.contains("rocminfo")
    }

    fn fail_closed_gpu(&self, req: &ToolRequest, text: &str, timeout_s: u32) -> Result<()> {
        if !Self::gpu_ish(text) && req.payload.get("gpu").and_then(|v| v.as_bool()) != Some(true) {
            return Ok(());
        }
        match &self.cfg.gpu {
            None => Err(Error::Backend("gpu not configured".into())),
            Some(gpu) => {
                let work = req
                    .payload
                    .get("work_s")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as u32;
                if work > timeout_s {
                    return Err(Error::Timeout);
                }
                if work > gpu.walltime_s {
                    return Err(Error::GpuTimeLimit);
                }
                Ok(())
            }
        }
    }

    fn charge_gpu(&mut self, sandbox: &SandboxId, req: &ToolRequest) -> Result<()> {
        let gpu = match &self.cfg.gpu {
            None => return Ok(()),
            Some(g) => g.clone(),
        };
        let text = format!(
            "{}{}",
            Self::payload_str(req, "code").unwrap_or_default(),
            Self::payload_str(req, "command").unwrap_or_default()
        );
        if !Self::gpu_ish(&text) && req.payload.get("gpu").and_then(|v| v.as_bool()) != Some(true) {
            return Ok(());
        }
        let work = req
            .payload
            .get("work_s")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let sb = self.live_sb_mut(sandbox)?;
        if sb.gpu_used_s.saturating_add(work) > gpu.walltime_s {
            return Err(Error::GpuTimeLimit);
        }
        sb.gpu_used_s = sb.gpu_used_s.saturating_add(work);
        Ok(())
    }

    fn tool_shell(
        &mut self,
        sandbox: &SandboxId,
        req: &ToolRequest,
        now: NowMs,
        timeout_s: u32,
    ) -> Result<ToolResponse> {
        let command = Self::payload_str(req, "command").unwrap_or_default();
        self.fail_closed_gpu(req, &command, timeout_s)?;
        self.charge_gpu(sandbox, req)?;
        let tokens = tokenize(&command);
        if tokens.is_empty() {
            return Ok(self.fail_resp(req, "empty command"));
        }
        if NETWORK_BINS.contains(&tokens[0].as_str()) {
            return Err(Error::EgressDenied);
        }
        if tokens[0] == "sleep" {
            let secs: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            if secs > timeout_s {
                return Err(Error::Timeout);
            }
            let mut r = self.ok_resp(req, b"", None);
            r.duration_ms = u64::from(secs) * 1000;
            return Ok(r);
        }
        if let Some(path) = redirect_path(&tokens) {
            if is_grader_path(path) {
                self.flag_write(sandbox, path, req, now);
                return Err(Error::TestFileWrite {
                    path: path.to_string(),
                });
            }
            let content = echo_body_before_redirect(&tokens);
            let sb = self.live_sb_mut(sandbox)?;
            sb.files.insert(path.to_string(), content);
            return Ok(self.ok_resp(req, b"", None));
        }
        match tokens[0].as_str() {
            "echo" => {
                let mut out = tokens[1..].join(" ");
                out.push('\n');
                Ok(self.ok_resp(req, out.as_bytes(), None))
            }
            "cat" => {
                if tokens.len() < 2 {
                    return Ok(self.fail_resp(req, "cat: missing path"));
                }
                let path = &tokens[1];
                if is_grader_path(path) {
                    return Ok(self.fail_resp(req, "not found"));
                }
                let sb = self.live_sb(sandbox)?;
                match sb.files.get(path) {
                    Some(b) => Ok(self.ok_resp(req, b, None)),
                    None => Ok(self.fail_resp(req, "not found")),
                }
            }
            _ => Ok(self.fail_resp(req, "unsupported command")),
        }
    }

    fn tool_python(
        &mut self,
        sandbox: &SandboxId,
        req: &ToolRequest,
        now: NowMs,
        timeout_s: u32,
    ) -> Result<ToolResponse> {
        let code = Self::payload_str(req, "code").unwrap_or_default();
        self.fail_closed_gpu(req, &code, timeout_s)?;
        self.charge_gpu(sandbox, req)?;
        let lower = code.to_ascii_lowercase();
        if lower.contains("import urllib")
            || lower.contains("from urllib")
            || lower.contains("import requests")
            || lower.contains("import socket")
            || lower.contains("from socket")
            || lower.contains("http.client")
            || lower.contains("import httpx")
            || lower.contains("aiohttp")
        {
            return Err(Error::LiveInternet);
        }
        if lower.contains("https://") || lower.contains("http://") {
            return Err(Error::LiveInternet);
        }
        if let Some(path) = python_write_path(&code) {
            if is_grader_path(&path) {
                self.flag_write(sandbox, &path, req, now);
                return Err(Error::TestFileWrite { path });
            }
        }
        match eval_python_print(&code, &self.agent_view(sandbox)?) {
            PythonEval::Stdout(bytes) => Ok(self.ok_resp(req, &bytes, None)),
            PythonEval::Fail(msg) => Ok(self.fail_resp(req, &msg)),
        }
    }

    fn tool_editor(
        &mut self,
        sandbox: &SandboxId,
        req: &ToolRequest,
        now: NowMs,
        notes: bool,
    ) -> Result<ToolResponse> {
        let op = Self::payload_str(req, "op").unwrap_or_else(|| {
            if req.payload.get("content").is_some() {
                "write".into()
            } else {
                "read".into()
            }
        });
        let default_path = if notes { "/notes/memo.txt" } else { "" };
        let path = Self::payload_str(req, "path").unwrap_or_else(|| default_path.to_string());
        if path.is_empty() {
            return Err(Error::BadToolRequest("missing path".into()));
        }
        if op == "write" {
            if is_grader_path(&path) {
                self.flag_write(sandbox, &path, req, now);
                return Err(Error::TestFileWrite { path });
            }
            let content = Self::payload_str(req, "content").unwrap_or_default();
            let sb = self.live_sb_mut(sandbox)?;
            sb.files.insert(path, content.into_bytes());
            return Ok(self.ok_resp(req, b"", None));
        }
        if is_grader_path(&path) {
            return Ok(self.fail_resp(req, "not found"));
        }
        let sb = self.live_sb(sandbox)?;
        match sb.files.get(&path) {
            Some(b) => Ok(self.ok_resp(req, b, None)),
            None => Ok(self.fail_resp(req, "not found")),
        }
    }

    fn tool_browser(&self, req: &ToolRequest) -> Result<ToolResponse> {
        let url = match Self::payload_str(req, "url") {
            Some(u) if !u.is_empty() => u,
            _ => return Ok(self.fail_resp(req, "missing url")),
        };
        let web = &self.cfg.offline_web;
        if let Some(page) = web.pages.get(&url) {
            if page.snapshot_unix_s <= web.cutoff_unix_s {
                let mut r = self.ok_resp(req, &page.body, None);
                r.stdout_artifact.media_type = Some(page.media_type.clone());
                return Ok(r);
            }
            return Ok(self.fail_resp(req, "page after cutoff"));
        }
        let live = url.starts_with("http://") || url.starts_with("https://");
        if live {
            return Err(Error::EgressDenied);
        }
        Ok(self.fail_resp(req, "missing url"))
    }

    fn tool_lean(&self, req: &ToolRequest) -> Result<ToolResponse> {
        let code = Self::payload_str(req, "code").unwrap_or_default();
        if code.trim().is_empty() {
            return Ok(self.fail_resp(req, "empty lean"));
        }
        let lower = code.to_ascii_lowercase();
        if lower.contains("sorry") || lower.contains("admit") {
            return Ok(self.fail_resp(req, "incomplete proof"));
        }
        Ok(self.ok_resp(req, b"ok\n", None))
    }

    fn tool_subagent(
        &mut self,
        sandbox: &SandboxId,
        req: &ToolRequest,
        now: NowMs,
    ) -> Result<ToolResponse> {
        let _ = req.parent_call_id.as_ref();
        let snap = self.snapshot(sandbox, now)?;
        if self.live_count() + 1 > self.cfg.capacity {
            return Err(Error::PoolExhausted {
                capacity: self.cfg.capacity,
            });
        }
        let (child, _) = self.fork_one(&snap)?;
        let mut stdout = child.0.clone();
        stdout.push('\n');
        Ok(self.ok_resp(req, stdout.as_bytes(), Some(child.0)))
    }
}

enum PythonEval {
    Stdout(Vec<u8>),
    Fail(String),
}

fn eval_python_print(code: &str, agent_files: &BTreeMap<String, Vec<u8>>) -> PythonEval {
    let trimmed = code.trim();
    let rest = match trimmed
        .strip_prefix("print(")
        .and_then(|s| s.strip_suffix(')'))
    {
        Some(inner) => inner.trim(),
        None => return PythonEval::Fail("unsupported python".into()),
    };
    if let Some(path) = parse_open_read(rest) {
        if is_grader_path(&path) {
            return PythonEval::Fail("not found".into());
        }
        return match agent_files.get(&path) {
            Some(b) => {
                let mut out = b.clone();
                if !out.ends_with(b"\n") {
                    out.push(b'\n');
                }
                PythonEval::Stdout(out)
            }
            None => PythonEval::Fail("not found".into()),
        };
    }
    if let Some(s) = parse_quoted(rest) {
        let mut out = s.into_bytes();
        out.push(b'\n');
        return PythonEval::Stdout(out);
    }
    if let Some(n) = eval_int_expr(rest) {
        let mut out = n.to_string().into_bytes();
        out.push(b'\n');
        return PythonEval::Stdout(out);
    }
    PythonEval::Fail("unsupported python".into())
}

fn parse_open_read(inner: &str) -> Option<String> {
    let t = inner.trim();
    let t = t.strip_prefix("open(")?.trim_start();
    let (q, rest) = if let Some(r) = t.strip_prefix('"') {
        ('"', r)
    } else if let Some(r) = t.strip_prefix('\'') {
        ('\'', r)
    } else {
        return None;
    };
    let end = rest.find(q)?;
    let path = rest[..end].to_string();
    let after = rest[end + 1..].trim_start();
    let after = after.strip_prefix(')')?.trim_start();
    let after = after.strip_prefix('.')?.trim_start();
    if after.starts_with("read()") {
        Some(path)
    } else {
        None
    }
}

fn parse_quoted(s: &str) -> Option<String> {
    let t = s.trim();
    if t.len() >= 2 {
        let b = t.as_bytes();
        if (b[0] == b'"' && *b.last().unwrap() == b'"')
            || (b[0] == b'\'' && *b.last().unwrap() == b'\'')
        {
            return Some(t[1..t.len() - 1].to_string());
        }
    }
    None
}

fn eval_int_expr(s: &str) -> Option<i64> {
    let t = s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    if t.is_empty() {
        return None;
    }
    let mut acc: i64 = 0;
    let mut sign: i64 = 1;
    let mut cur = String::new();
    let mut started = false;
    for c in t.chars() {
        match c {
            '+' | '-' if !cur.is_empty() || started => {
                acc = acc.checked_add(sign * cur.parse::<i64>().ok()?)?;
                cur.clear();
                sign = if c == '+' { 1 } else { -1 };
            }
            d if d.is_ascii_digit() => {
                cur.push(d);
                started = true;
            }
            _ => return None,
        }
    }
    if cur.is_empty() {
        return None;
    }
    acc.checked_add(sign * cur.parse::<i64>().ok()?)
}

fn tokenize(command: &str) -> Vec<String> {
    command.split_whitespace().map(str::to_string).collect()
}

fn redirect_path(tokens: &[String]) -> Option<&str> {
    let pos = tokens.iter().position(|t| t == ">")?;
    tokens.get(pos + 1).map(String::as_str)
}

fn echo_body_before_redirect(tokens: &[String]) -> Vec<u8> {
    let pos = tokens.iter().position(|t| t == ">").unwrap_or(tokens.len());
    let start = if tokens.first().map(String::as_str) == Some("echo") {
        1
    } else {
        0
    };
    let mut s = tokens[start..pos].join(" ");
    s.push('\n');
    s.into_bytes()
}

fn python_write_path(code: &str) -> Option<String> {
    let rest = code.trim().strip_prefix("open(")?;
    let writey = rest.contains("write") || rest.contains("'w'") || rest.contains("\"w\"");
    if !writey {
        return None;
    }
    let rest = rest.trim_start();
    let q = rest.chars().next()?;
    if q != '"' && q != '\'' {
        return None;
    }
    let rest = &rest[1..];
    let end = rest.find(q)?;
    Some(rest[..end].to_string())
}

/// Unused CallId import kept out of the public surface; silence via type check in tests.
pub fn call_id(s: &str) -> CallId {
    CallId(s.to_string())
}
