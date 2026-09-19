//! Oracle tests for the V1 GPU JAX extra (F4-v1-cuda, spec 16.2).
//!
//! Production is reached through `render_job(Stage::V1, ...)` and by reading
//! repo-root `pyproject.toml` relative to `CARGO_MANIFEST_DIR`. Do not inspect
//! a stub template file alone (that would pass today if someone copied a
//! comment into `templates/v1.sh`).
//!
//! The iface-only tree has no `[project.optional-dependencies]` extra and
//! `v1.sh` still `pip install -e "$ROOT"`. Every test in this file must fail
//! on that tree. Do not add a check that already holds (constant name, CPU
//! `run_v1`, absence of `/work/ttennant1` with no extra present).
//!
//! `run_v1` stays callable on CPU JAX for `tests/test_v1_parity.py`. These
//! tests do not require a GPU and are not `#[cfg(feature = "gpu")]`.

mod reference;

use prometheus_verify_ncshare::{render_job, Stage, V1_CUDA_EXTRA};
use reference::v1_cuda as ref_cuda;

fn pyproject() -> String {
    ref_cuda::read_production_pyproject()
}

fn v1_job() -> String {
    render_job(Stage::V1, "run42", "04:00:00", 1)
        .unwrap_or_else(|e| panic!("render_job(Stage::V1, ...) must succeed: {e}"))
}

// ---------------------------------------------------------------------------
// Group: pyproject optional extra
// ---------------------------------------------------------------------------

#[test]
fn prod_pyproject_has_project_optional_dependencies_cuda_extra() {
    let toml = pyproject();
    let items = ref_cuda::optional_extra_items(&toml, V1_CUDA_EXTRA);
    assert!(
        items.is_some(),
        "pyproject.toml (relative to CARGO_MANIFEST_DIR) must have \
         [project.optional-dependencies] extra named {V1_CUDA_EXTRA:?} \
         (V1_CUDA_EXTRA). Got no such extra. pyproject starts:\n{}",
        toml.chars().take(400).collect::<String>()
    );
}

#[test]
fn prod_cuda_extra_lists_portable_gpu_jax_not_empty_not_path() {
    let toml = pyproject();
    let items = ref_cuda::optional_extra_items(&toml, V1_CUDA_EXTRA);
    assert!(
        items.is_some(),
        "missing [project.optional-dependencies].{V1_CUDA_EXTRA} extra"
    );
    let items = items.unwrap();
    assert!(
        !items.is_empty(),
        "{V1_CUDA_EXTRA} extra must not be empty: {items:?}"
    );
    assert!(
        items
            .iter()
            .any(|i| ref_cuda::is_portable_jax_cuda_requirement(i)),
        "{V1_CUDA_EXTRA} extra must list a portable GPU JAX extra \
         (jax[cuda12] or jax[cuda13], not a local path): {items:?}"
    );
    assert!(
        items.iter().all(|i| !ref_cuda::is_local_or_site_path(i)),
        "{V1_CUDA_EXTRA} extra must not use a local/site path: {items:?}"
    );
    assert!(
        ref_cuda::cuda_extra_lists_portable_gpu_jax(&toml, V1_CUDA_EXTRA),
        "reference checker rejected the {V1_CUDA_EXTRA} extra: {items:?}"
    );
}

// ---------------------------------------------------------------------------
// Group: default [project] dependencies stay CPU JAX
// ---------------------------------------------------------------------------

#[test]
fn prod_default_project_dependencies_do_not_pull_in_cuda_extra() {
    let toml = pyproject();
    // Extra must exist so CPU CI can omit it; absence is not "staying CPU".
    assert!(
        ref_cuda::optional_extra_items(&toml, V1_CUDA_EXTRA).is_some(),
        "declare [project.optional-dependencies].{V1_CUDA_EXTRA} so default \
         [project] dependencies can stay CPU JAX (CI / uv run pytest)"
    );
    assert!(
        ref_cuda::default_dependencies_are_cpu_jax(&toml),
        "default [project] dependencies must not pull in jax[cuda12]/jax[cuda13] \
         or extra {V1_CUDA_EXTRA:?}; CPU CI must not need CUDA. deps={:?}",
        ref_cuda::project_dependencies(&toml)
    );
}

// ---------------------------------------------------------------------------
// Group: render_job(Stage::V1) pip-installs the extra
// ---------------------------------------------------------------------------

#[test]
fn prod_render_job_v1_pip_installs_editable_with_cuda_extra() {
    let extra = V1_CUDA_EXTRA;
    let syntax = ref_cuda::pip_editable_extra_syntax(extra);
    for run_id in ["run42", "abc"] {
        let script = render_job(Stage::V1, run_id, "01:00:00", 1)
            .unwrap_or_else(|e| panic!("render_job V1: {e}"));
        assert!(
            ref_cuda::script_installs_editable_with_extra(&script, extra),
            "render_job(Stage::V1, ...) must pip install -e {syntax:?} \
             (editable project with V1_CUDA_EXTRA={extra:?}), not only \
             -e \"$ROOT\". Comments do not count.\n{script}"
        );
        assert!(
            !ref_cuda::script_editable_install_lacks_extra(&script, extra),
            "editable pip install still lacks extra {extra:?}:\n{script}"
        );
    }
}

// ---------------------------------------------------------------------------
// Group: job template fails if JAX is not using a GPU
// ---------------------------------------------------------------------------

#[test]
fn prod_render_job_v1_fails_if_jax_not_using_gpu_after_install() {
    let extra = V1_CUDA_EXTRA;
    let script = v1_job();
    assert!(
        ref_cuda::script_installs_editable_with_extra(&script, extra),
        "GPU check is after pip install -e \".[{extra}]\"; that install is missing:\n{script}"
    );
    assert!(
        ref_cuda::script_fails_if_jax_not_on_gpu_after_install(&script, extra),
        "rendered V1 job must fail if JAX is not using a GPU (jax devices / \
         default_backend / get_backend check with assert/raise/exit), after \
         the extra install. Comments and run_v1(gpus=...) do not count.\n{script}"
    );
}

// ---------------------------------------------------------------------------
// Group: no site paths in the extra or the new install lines
// ---------------------------------------------------------------------------

#[test]
fn prod_cuda_extra_and_v1_install_have_no_site_paths() {
    let extra = V1_CUDA_EXTRA;
    let toml = pyproject();
    let script = v1_job();
    assert!(
        ref_cuda::extra_items_have_no_site_paths(&toml, extra),
        "{extra} extra must exist, be non-empty, and contain no site paths \
         (no {}, no file://, no absolute/relative paths)",
        ref_cuda::FORBIDDEN_SITE_PREFIX
    );
    assert!(
        ref_cuda::extra_install_lines_have_no_site_paths(&script, extra),
        "render_job(Stage::V1) extra install lines must exist and must not \
         hardcode {} or other absolute site paths; use pip install -e \".[{extra}]\".\n{script}",
        ref_cuda::FORBIDDEN_SITE_PREFIX
    );
}
