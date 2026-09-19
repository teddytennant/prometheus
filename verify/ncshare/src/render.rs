//! Template substitution for V0-V10 job scripts.

use crate::{template_path, Error, Result, Stage};

/// Substitute `{{RUN_ID}}`, `{{WALLTIME}}`, `{{GPUS}}` in `templates/{stage}.sh`.
pub(crate) fn render_job(stage: Stage, run_id: &str, walltime: &str, gpus: u32) -> Result<String> {
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

/// Substitute `{{RUN_ID}}` and `{{WALLTIME}}` in `templates/v0-2node.sh`.
///
/// Distinct from `render_job(Stage::V0, ...)`, which reads `templates/v0.sh`.
/// Empty `run_id` or `walltime` is `Error::Other` before any template read.
/// GPU counts are fixed by the template (`#SBATCH --nodes=2`, `gpu:h200:4`);
/// there is no `{{GPUS}}` placeholder.
pub(crate) fn render_v0_2node(run_id: &str, walltime: &str) -> Result<String> {
    if run_id.is_empty() {
        return Err(Error::Other("empty run_id".into()));
    }
    if walltime.is_empty() {
        return Err(Error::Other("empty walltime".into()));
    }
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("templates")
        .join("v0-2node.sh");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| Error::Other(format!("read template {}: {e}", path.display())))?;
    Ok(raw
        .replace("{{RUN_ID}}", run_id)
        .replace("{{WALLTIME}}", walltime))
}
