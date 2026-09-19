//! Oracle tests that V1 independent logits are torch-only (F4-v1-retire-numpy).
//!
//! Spec 16.2 V1: JAX vs PyTorch reference logits, 1e-5 FP32.
//! `prometheus.verify._numpy_forward` is retired: that module must not exist.
//! `templates/v1.sh` must fail-closed on `import torch` after the JAX GPU
//! assert, the same way it fail-closes if JAX is not using a GPU.
//! `V1_INDEPENDENT_FORWARD` stays `"torch"`.
//!
//! Production is reached through `render_job(Stage::V1, ...)` and by reading
//! repo-root `prometheus/verify/_numpy_forward.py` relative to
//! `CARGO_MANIFEST_DIR`. Do not inspect a stub template comment alone.
//!
//! Every test in this file must fail on the pre-retirement tree (numpy forward
//! still present, `v1.sh` has no `import torch`). Do not add a check that
//! already holds (`V1_INDEPENDENT_FORWARD == "torch"` alone, torch already a
//! default CPU dep).
//!
//! These tests do not require a GPU and are not `#[cfg(feature = "gpu")]`.

use std::path::{Path, PathBuf};

use prometheus_verify_ncshare::{render_job, Stage, V1_INDEPENDENT_FORWARD};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root relative to CARGO_MANIFEST_DIR")
}

fn numpy_forward_py() -> PathBuf {
    repo_root().join("prometheus/verify/_numpy_forward.py")
}

fn numpy_forward_pkg() -> PathBuf {
    repo_root().join("prometheus/verify/_numpy_forward")
}

fn v1_job() -> String {
    render_job(Stage::V1, "run42", "04:00:00", 1)
        .unwrap_or_else(|e| panic!("render_job(Stage::V1, ...) must succeed: {e}"))
}

fn is_comment_line(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

fn strip_trailing_hash_comment(line: &str) -> String {
    let mut out = String::new();
    let mut in_s: Option<char> = None;
    let mut it = line.chars().peekable();
    while let Some(c) = it.next() {
        if let Some(q) = in_s {
            out.push(c);
            if c == '\\' && (q == '"' || q == '\'') {
                if let Some(n) = it.next() {
                    out.push(n);
                }
            } else if c == q {
                in_s = None;
            }
            continue;
        }
        if c == '#' {
            break;
        }
        if c == '"' || c == '\'' {
            in_s = Some(c);
        }
        out.push(c);
    }
    out.trim_end().to_string()
}

fn code_lines(script: &str) -> Vec<String> {
    script
        .lines()
        .filter(|l| !is_comment_line(l))
        .map(strip_trailing_hash_comment)
        .collect()
}

fn is_jax_gpu_fail_closed_blob(blob: &str) -> bool {
    let low = blob.to_ascii_lowercase();
    if !low.contains("jax") {
        return false;
    }
    if !["devices", "default_backend", "get_backend", "platform"]
        .iter()
        .any(|t| low.contains(t))
    {
        return false;
    }
    if !["gpu", "cuda"].iter().any(|t| low.contains(t)) {
        return false;
    }
    ["assert", "raise", "sys.exit", "systemexit"]
        .iter()
        .any(|t| low.contains(t))
}

fn is_import_torch_stmt(stmt: &str) -> bool {
    let t = stmt.trim();
    if t.starts_with("from torch ") || t.starts_with("from torch.") {
        return true;
    }
    let Some(rest) = t.strip_prefix("import ") else {
        return false;
    };
    rest.split(',').any(|part| {
        let tok = part.trim();
        let name = tok.split_whitespace().next().unwrap_or("");
        name == "torch" || name.starts_with("torch.")
    })
}

fn python_c_payloads(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let low = line.to_ascii_lowercase();
    for marker in ["python -c ", "python3 -c "] {
        let mut search_from = 0;
        while let Some(rel) = low[search_from..].find(marker) {
            let start = search_from + rel + marker.len();
            let bytes = line.as_bytes();
            if start >= bytes.len() {
                break;
            }
            let quote = bytes[start] as char;
            if quote != '\'' && quote != '"' {
                search_from = start;
                continue;
            }
            if let Some(end_rel) = line[start + 1..].find(quote) {
                out.push(line[start + 1..start + 1 + end_rel].to_string());
                search_from = start + 1 + end_rel + 1;
            } else {
                break;
            }
        }
    }
    out
}

fn line_has_top_level_import_torch(line: &str) -> bool {
    if line.starts_with(' ') || line.starts_with('\t') {
        return false;
    }
    let mut payloads = vec![line.to_string()];
    payloads.extend(python_c_payloads(line));
    payloads
        .iter()
        .any(|p| p.split(';').any(is_import_torch_stmt))
}

/// Live top-level `import torch` after the JAX GPU fail-closed check.
/// Comments do not count. An import only *before* the JAX GPU assert does not.
fn fail_closed_on_import_torch_after_jax_gpu(script: &str) -> bool {
    let lines = code_lines(script);
    let mut acc = String::new();
    let mut jax_end = None;
    for (i, line) in lines.iter().enumerate() {
        acc.push_str(line);
        acc.push('\n');
        if is_jax_gpu_fail_closed_blob(&acc) {
            jax_end = Some(i);
            break;
        }
    }
    let Some(jax_end) = jax_end else {
        return false;
    };
    let same = &lines[jax_end];
    if same.contains(';') {
        let after = same.splitn(2, ';').nth(1).unwrap_or("");
        if !same.starts_with(' ')
            && !same.starts_with('\t')
            && after.split(';').any(is_import_torch_stmt)
        {
            return true;
        }
    }
    lines[jax_end + 1..]
        .iter()
        .any(|line| line_has_top_level_import_torch(line))
}

fn assert_forward_stays_torch() {
    assert_eq!(
        V1_INDEPENDENT_FORWARD, "torch",
        "V1_INDEPENDENT_FORWARD must stay \"torch\""
    );
}

fn assert_numpy_forward_absent(py: &Path, pkg: &Path) {
    assert!(
        !py.exists(),
        "prometheus.verify._numpy_forward is retired: {} must not exist",
        py.display()
    );
    assert!(
        !pkg.exists(),
        "prometheus.verify._numpy_forward is retired: {} must not exist",
        pkg.display()
    );
}

// ---------------------------------------------------------------------------
// Group: prometheus.verify._numpy_forward must not exist
// ---------------------------------------------------------------------------

#[test]
fn numpy_forward_py_must_not_exist() {
    assert_forward_stays_torch();
    assert_numpy_forward_absent(&numpy_forward_py(), &numpy_forward_pkg());
}

#[test]
fn numpy_forward_retired_and_independent_forward_is_torch() {
    assert_eq!(
        V1_INDEPENDENT_FORWARD, "torch",
        "V1_INDEPENDENT_FORWARD must stay \"torch\" after retiring numpy"
    );
    let py = numpy_forward_py();
    assert!(
        !py.is_file(),
        "prometheus.verify._numpy_forward is retired (must not exist); found {}",
        py.display()
    );
}

// ---------------------------------------------------------------------------
// Group: rendered v1.sh fail-closed on import torch after JAX GPU assert
// ---------------------------------------------------------------------------

#[test]
fn v1_job_keeps_jax_gpu_fail_closed_and_imports_torch_after_it() {
    assert_forward_stays_torch();
    let script = v1_job();
    let blob = code_lines(&script).join("\n");
    assert!(
        is_jax_gpu_fail_closed_blob(&blob),
        "rendered v1.sh must keep a JAX GPU fail-closed check\n{script}"
    );
    assert!(
        fail_closed_on_import_torch_after_jax_gpu(&script),
        "rendered v1.sh must fail-closed on `import torch` after the JAX GPU \
         assert (same style as the jax GPU check: top-level, not a comment, \
         not only before the JAX assert). Torch is already a default CPU dep; \
         this is not a new optional extra.\n--- v1.sh ---\n{script}"
    );
}

#[test]
fn v1_job_import_torch_is_not_only_a_comment() {
    assert_forward_stays_torch();
    let script = v1_job();
    let live = fail_closed_on_import_torch_after_jax_gpu(&script);
    assert!(
        live,
        "rendered v1.sh must contain a live top-level `import torch` after the \
         JAX GPU assert; a comment does not fail-close the job\n--- v1.sh ---\n{script}"
    );
}
