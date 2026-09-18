//! Slow template substitution for V0–V10 job scripts.

use prometheus_verify_ncshare::{template_path, Error, Result, Stage};

/// Substitute `{{RUN_ID}}`, `{{WALLTIME}}`, `{{GPUS}}` in `templates/{stage}.sh`.
pub fn render_job(stage: Stage, run_id: &str, walltime: &str, gpus: u32) -> Result<String> {
    if gpus == 0 {
        return Err(Error::Other(
            "gpus must be > 0 (V10 callers pass a dummy count; the V10 template does not interpolate {{GPUS}})"
                .into(),
        ));
    }
    if run_id.is_empty() {
        return Err(Error::Other("empty run_id".into()));
    }
    if walltime.is_empty() {
        return Err(Error::Other("empty walltime".into()));
    }
    let path = template_path(stage);
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::Other(format!("read template {}: {e}", path.display())))?;
    Ok(raw
        .replace("{{RUN_ID}}", run_id)
        .replace("{{WALLTIME}}", walltime)
        .replace("{{GPUS}}", &gpus.to_string()))
}
