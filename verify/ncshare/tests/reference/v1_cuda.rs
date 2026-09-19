//! Independent V1 CUDA extra checker (spec 16.2, F4-v1-cuda).
//!
//! Slow and obvious: no toml crate, no regex crate. Production code must never
//! import this module. Tests feed it repo-root `pyproject.toml` and
//! `render_job(Stage::V1, ...)` output.

#![allow(dead_code)]

use std::path::PathBuf;

/// Extra name the V1 job must enable. Same string as `V1_CUDA_EXTRA`.
pub const EXTRA_NAME: &str = "cuda";

/// Site prefix that must not appear in the extra or in the new install lines.
pub const FORBIDDEN_SITE_PREFIX: &str = "/work/ttennant1";

/// `pip install -e ".[{extra}]"` extras syntax (PEP 508 extras on a path).
pub fn pip_editable_extra_syntax(extra: &str) -> String {
    format!(".[{extra}]")
}

/// Repo-root `pyproject.toml`, relative to this crate's `CARGO_MANIFEST_DIR`.
///
/// Crate lives at `verify/ncshare/`, so the file is `../../pyproject.toml`.
pub fn production_pyproject_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../pyproject.toml")
}

pub fn read_production_pyproject() -> String {
    let path = production_pyproject_path();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// TOML (PEP 621 subset)
// ---------------------------------------------------------------------------

fn strip_toml_comment(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars().peekable();
    let mut quote: Option<char> = None;
    while let Some(c) = chars.next() {
        if let Some(q) = quote {
            out.push(c);
            if c == '\\' && q == '"' {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
                continue;
            }
            if c == q {
                quote = None;
            }
            continue;
        }
        if c == '#' {
            break;
        }
        if c == '"' || c == '\'' {
            quote = Some(c);
        }
        out.push(c);
    }
    out
}

fn is_table_header_line(stripped: &str, name: &str) -> bool {
    stripped.trim() == format!("[{name}]")
}

/// Body of `[name]` until the next table header. `None` if the table is absent.
fn table_body(toml: &str, name: &str) -> Option<String> {
    let mut found = false;
    let mut body = String::new();
    for line in toml.lines() {
        let stripped = strip_toml_comment(line);
        if !found {
            if is_table_header_line(&stripped, name) {
                found = true;
            }
            continue;
        }
        let trim = stripped.trim();
        if trim.starts_with('[') && trim.ends_with(']') {
            break;
        }
        body.push_str(&stripped);
        body.push('\n');
    }
    if found {
        Some(body)
    } else {
        None
    }
}

fn parse_toml_quoted(src: &str) -> Option<(String, &str)> {
    let s = src.trim_start();
    let mut chars = s.chars();
    let quote = chars.next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let mut out = String::new();
    if quote == '"' {
        while let Some(c) = chars.next() {
            if c == '\\' {
                if let Some(n) = chars.next() {
                    out.push(n);
                }
                continue;
            }
            if c == '"' {
                let consumed = s.len() - chars.as_str().len();
                return Some((out, &s[consumed..]));
            }
            out.push(c);
        }
        return None;
    }
    for c in chars.by_ref() {
        if c == '\'' {
            let consumed = s.len() - chars.as_str().len();
            return Some((out, &s[consumed..]));
        }
        out.push(c);
    }
    None
}

fn parse_toml_string_array(raw: &str) -> Option<Vec<String>> {
    let mut s = raw.trim();
    s = s.strip_prefix('[')?;
    let mut items = Vec::new();
    loop {
        s = s.trim_start();
        if s.is_empty() {
            return None;
        }
        if s.starts_with(']') {
            return Some(items);
        }
        if s.starts_with('#') {
            s = s.split_once('\n').map(|(_, r)| r).unwrap_or("");
            continue;
        }
        let (item, rest) = parse_toml_quoted(s)?;
        items.push(item);
        s = rest.trim_start();
        if s.starts_with(',') {
            s = s[1..].trim_start();
        }
    }
}

fn matching_bracket_value(src: &str) -> Option<&str> {
    let start = src.find('[')?;
    let bytes: Vec<char> = src[start..].chars().collect();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut skip_next = false;
    for (i, c) in bytes.iter().copied().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if let Some(q) = quote {
            if c == '\\' && q == '"' {
                skip_next = true;
                continue;
            }
            if c == q {
                quote = None;
            }
            continue;
        }
        if c == '"' || c == '\'' {
            quote = Some(c);
            continue;
        }
        if c == '[' {
            depth += 1;
        } else if c == ']' {
            depth -= 1;
            if depth == 0 {
                return Some(&src[start..start + i + 1]);
            }
        }
    }
    None
}

fn assignment_raw(body: &str, key: &str) -> Option<String> {
    let lines: Vec<&str> = body.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let stripped = strip_toml_comment(lines[i]);
        i += 1;
        let trim = stripped.trim();
        if trim.is_empty() {
            continue;
        }
        let Some((left, right)) = trim.split_once('=') else {
            continue;
        };
        let left = left.trim().trim_matches('"').trim_matches('\'');
        if left != key {
            continue;
        }
        let right = right.trim();
        if !right.contains('[') {
            return Some(right.to_string());
        }
        let mut collected = right.to_string();
        loop {
            if let Some(v) = matching_bracket_value(&collected) {
                return Some(v.to_string());
            }
            if i >= lines.len() {
                return None;
            }
            collected.push('\n');
            collected.push_str(&strip_toml_comment(lines[i]));
            i += 1;
        }
    }
    None
}

/// Items of `[project.optional-dependencies].{extra}`. `None` if missing.
pub fn optional_extra_items(toml: &str, extra: &str) -> Option<Vec<String>> {
    let body = table_body(toml, "project.optional-dependencies")?;
    let raw = assignment_raw(&body, extra)?;
    parse_toml_string_array(&raw)
}

/// `[project] dependencies` list. Empty if the table or key is missing.
pub fn project_dependencies(toml: &str) -> Vec<String> {
    let Some(body) = table_body(toml, "project") else {
        return Vec::new();
    };
    let Some(raw) = assignment_raw(&body, "dependencies") else {
        return Vec::new();
    };
    parse_toml_string_array(&raw).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Requirement / path predicates
// ---------------------------------------------------------------------------

pub fn has_forbidden_site_marker(text: &str) -> bool {
    text.to_ascii_lowercase()
        .contains(&FORBIDDEN_SITE_PREFIX.to_ascii_lowercase())
}

pub fn is_local_or_site_path(item: &str) -> bool {
    let t = item.trim();
    if t.is_empty() {
        return false;
    }
    if has_forbidden_site_marker(t) {
        return true;
    }
    let lower = t.to_ascii_lowercase();
    if lower.contains("file://") {
        return true;
    }
    if t.starts_with('/') || t.starts_with("./") || t.starts_with("../") {
        return true;
    }
    if let Some((_, url)) = lower.split_once('@') {
        let url = url.trim();
        if url.starts_with('/') || url.starts_with("file:") || url.starts_with("./") {
            return true;
        }
    }
    false
}

/// Portable GPU JAX extra: `jax[cuda12]` / `jax[cuda13]` (optional `_pip` suffix).
/// Rejects empty, CPU-only `jax`, local paths, and `*_local` CUDA extras.
pub fn is_portable_jax_cuda_requirement(item: &str) -> bool {
    if is_local_or_site_path(item) {
        return false;
    }
    let lower = item.trim().to_ascii_lowercase();
    let main = lower
        .split(';')
        .next()
        .unwrap_or("")
        .split('@')
        .next()
        .unwrap_or("")
        .trim();
    let Some(bracket) = main.find('[') else {
        return false;
    };
    let name = main[..bracket].trim();
    if name != "jax" {
        return false;
    }
    let rest = &main[bracket + 1..];
    let Some(end) = rest.find(']') else {
        return false;
    };
    rest[..end].split(',').any(|extra| {
        let extra = extra.trim();
        if extra.contains("local") {
            return false;
        }
        extra == "cuda12"
            || extra == "cuda13"
            || extra.starts_with("cuda12_")
            || extra.starts_with("cuda13_")
    })
}

pub fn cuda_extra_lists_portable_gpu_jax(toml: &str, extra: &str) -> bool {
    let Some(items) = optional_extra_items(toml, extra) else {
        return false;
    };
    if items.is_empty() {
        return false;
    }
    if items.iter().any(|i| is_local_or_site_path(i)) {
        return false;
    }
    items.iter().any(|i| is_portable_jax_cuda_requirement(i))
}

/// Default `[project] dependencies` must not install GPU JAX / the V1 extra.
pub fn default_dependencies_are_cpu_jax(toml: &str) -> bool {
    let deps = project_dependencies(toml);
    if deps.iter().any(|d| is_portable_jax_cuda_requirement(d)) {
        return false;
    }
    if deps
        .iter()
        .any(|d| is_local_or_site_path(d) && has_forbidden_site_marker(d))
    {
        return false;
    }
    for d in &deps {
        let lower = d.to_ascii_lowercase();
        if let Some(idx) = lower.find('[') {
            if let Some(end) = lower[idx + 1..].find(']') {
                let extras = &lower[idx + 1..idx + 1 + end];
                if extras.split(',').any(|e| e.trim() == EXTRA_NAME) {
                    return false;
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Rendered V1 script
// ---------------------------------------------------------------------------

fn is_shell_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('#') && !t.starts_with("#SBATCH")
}

fn contains_word(haystack: &str, word: &str) -> bool {
    let h = haystack.as_bytes();
    let w = word.as_bytes();
    if w.is_empty() || h.len() < w.len() {
        return false;
    }
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut i = 0;
    while i + w.len() <= h.len() {
        if h[i..i + w.len()].eq_ignore_ascii_case(w) {
            let before_ok = i == 0 || !ident(h[i - 1]);
            let after = i + w.len();
            let after_ok = after == h.len() || !ident(h[after]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn line_has_pip_extra(line: &str, extra: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let extra = extra.to_ascii_lowercase();
    let mut rest = lower.as_str();
    while let Some(idx) = rest.find('[') {
        let after = &rest[idx + 1..];
        let Some(end) = after.find(']') else {
            break;
        };
        let inner = after[..end].trim();
        if inner.split(',').any(|e| e.trim() == extra) {
            return true;
        }
        rest = &after[end + 1..];
    }
    false
}

fn line_has_pip_install(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if lower.contains("pip install") {
        return true;
    }
    let tokens: Vec<&str> = line.split_whitespace().collect();
    tokens.windows(2).any(|w| {
        w[0].trim_matches(|c: char| c == '"' || c == '\'')
            .ends_with("pip")
            && w[1].trim_matches(|c: char| c == '"' || c == '\'') == "install"
    })
}

fn line_has_editable_flag(line: &str) -> bool {
    line.split_whitespace().any(|tok| {
        let tok = tok.trim_matches(|c: char| c == '"' || c == '\'');
        tok == "-e" || tok == "--editable"
    })
}

/// Non-comment `pip install -e` of the local project with extras `[{extra}]`.
pub fn line_is_editable_pip_with_extra(line: &str, extra: &str) -> bool {
    if is_shell_comment(line) {
        return false;
    }
    if !line_has_pip_install(line) {
        return false;
    }
    if !line_has_editable_flag(line) {
        return false;
    }
    if !line_has_pip_extra(line, extra) {
        return false;
    }
    true
}

pub fn script_installs_editable_with_extra(script: &str, extra: &str) -> bool {
    script
        .lines()
        .any(|line| line_is_editable_pip_with_extra(line, extra))
}

/// True if there is an editable pip install and none of them enable `extra`.
pub fn script_editable_install_lacks_extra(script: &str, extra: &str) -> bool {
    let mut saw_editable = false;
    for line in script.lines() {
        if is_shell_comment(line) {
            continue;
        }
        if line_has_pip_install(line) && line_has_editable_flag(line) {
            saw_editable = true;
            if line_has_pip_extra(line, extra) {
                return false;
            }
        }
    }
    saw_editable
}

fn jax_gpu_fail_closed(blob: &str) -> bool {
    let lower = blob.to_ascii_lowercase();
    if !contains_word(&lower, "jax") {
        return false;
    }
    let device_api = contains_word(&lower, "devices")
        || lower.contains("default_backend")
        || lower.contains("get_backend")
        || contains_word(&lower, "platform");
    if !device_api {
        return false;
    }
    if !contains_word(&lower, "gpu") && !contains_word(&lower, "cuda") {
        return false;
    }
    contains_word(&lower, "assert")
        || contains_word(&lower, "raise")
        || lower.contains("sys.exit")
        || lower.contains("systemexit")
        || contains_word(&lower, "exit")
        || lower.contains("[0]")
}

/// After the extra install, the job must fail if JAX is not on a GPU.
///
/// Comments do not count. The pip extra name on the install line does not count
/// as a device check. `run_v1(gpus=...)` is not a JAX backend check (`gpu` vs
/// `gpus` is word-bounded).
pub fn script_fails_if_jax_not_on_gpu_after_install(script: &str, extra: &str) -> bool {
    let mut after_install = false;
    let mut blob = String::new();
    for line in script.lines() {
        if is_shell_comment(line) {
            continue;
        }
        if line_is_editable_pip_with_extra(line, extra) {
            after_install = true;
            continue;
        }
        if after_install {
            blob.push_str(line);
            blob.push('\n');
        }
    }
    if !after_install {
        return false;
    }
    jax_gpu_fail_closed(&blob)
}

pub fn extra_install_lines_have_no_site_paths(script: &str, extra: &str) -> bool {
    let mut found = false;
    for line in script.lines() {
        if !line_is_editable_pip_with_extra(line, extra) {
            continue;
        }
        found = true;
        if has_forbidden_site_marker(line) {
            return false;
        }
        for tok in line.split_whitespace() {
            let tok = tok.trim_matches(|c: char| c == '"' || c == '\'');
            if tok.starts_with('/') {
                return false;
            }
        }
    }
    found
}

pub fn extra_items_have_no_site_paths(toml: &str, extra: &str) -> bool {
    let Some(items) = optional_extra_items(toml, extra) else {
        return false;
    };
    if items.is_empty() {
        return false;
    }
    items
        .iter()
        .all(|i| !is_local_or_site_path(i) && !has_forbidden_site_marker(i))
}
