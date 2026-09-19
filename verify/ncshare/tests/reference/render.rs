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

/// Substitute `{{RUN_ID}}` and `{{WALLTIME}}` in the independent 2-node stub
/// (`v0_2node.sh` next to this file, not production `templates/`).
///
/// Empty `run_id` or `walltime` is `Error::Other`, matching `render_job`.
/// GPU counts are fixed by the stub (`#SBATCH --nodes=2`, `gpu:h200:4`,
/// `--ntasks=8`, `--ntasks-per-node=4`; never `--gpus-per-task`); there is
/// no `{{GPUS}}` placeholder. Launch is `srun --mpi=pmix` of
/// `all_reduce_perf_mpi` (not `mpirun`; override `NCCL_TESTS_ALL_REDUCE_MPI`)
/// with the NCShare PMIx env (`PMIX_MCA_gds=hash`,
/// `unset OMPI_MCA_mca_base_component_path`, system OpenMPI `openmpi/lib`
/// on `LD_LIBRARY_PATH`).
pub fn render_v0_2node(run_id: &str, walltime: &str) -> Result<String> {
    if run_id.is_empty() {
        return Err(Error::Other("empty run_id".into()));
    }
    if walltime.is_empty() {
        return Err(Error::Other("empty walltime".into()));
    }
    let raw = include_str!("v0_2node.sh");
    Ok(raw
        .replace("{{RUN_ID}}", run_id)
        .replace("{{WALLTIME}}", walltime))
}
