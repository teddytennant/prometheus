//! Independent F4 reference. Production (`prometheus_verify_ncshare`) must never
//! import this module. Tests compare public functions against these slow parsers
//! and checkers.
//!
//! # `parse_busbw_gbps`
//!
//! nccl-tests all_reduce stdout is a `#`-commented banner plus a table with
//! out-of-place then in-place columns:
//! `size count type redop root time algbw busbw #wrong time algbw busbw #wrong`.
//! The reference walks lines, keeps the last row whose `size` and `count` parse
//! as integers and that has at least 9 whitespace fields, and returns that row's
//! **out-of-place busbw** (field index 7) as GB/s. It does **not** use the
//! `# Avg bus bandwidth` summary or the in-place busbw.
//!
//! Empty input, comment-only input, or no valid data row → `Error::Other`.
//!
//! # `kvm_available`
//!
//! Node-facts text is line-oriented. Blank lines and `#` comments are skipped.
//!
//! 1. An explicit `kvm: <value>` line (case-insensitive key and value) wins.
//!    True values: `yes`, `true`, `1`, `present`. False values: `no`, `false`,
//!    `0`, `absent`. Any other value is `Error::Other`.
//! 2. Else a `/dev/kvm` listing whose first token starts with `c` (character
//!    device, e.g. `crw-rw---- ... /dev/kvm`) or that contains
//!    `character device` / `character special` → `Ok(true)`.
//! 3. Else a `/dev/kvm` line with `no such file`, `not found`, `cannot stat`,
//!    `missing`, or `absent` → `Ok(false)`.
//! 4. Empty / whitespace-only facts → `Error::Other`.
//! 5. Non-empty facts that never mention kvm → `Error::Other` (unknown, not
//!    "absent").
//!
//! # `render_job`
//!
//! Reads `templates/{v0..v10}.sh` via `template_path`, substitutes
//! `{{RUN_ID}}`, `{{WALLTIME}}`, `{{GPUS}}`. `gpus == 0` is `Error::Other` for
//! every stage, including V10 (shared/soak): pass a dummy `gpus >= 1`; V10's
//! template does not interpolate `{{GPUS}}`. Empty `run_id` or `walltime` is
//! `Error::Other`. Stage is an enum; there is no unknown-stage path here.
//!
//! After substitution the job name line from the stock templates is
//! `#SBATCH -J fv-<run_id>-v{n}` (`JOB_PREFIX` is `fv`).
//!
//! # `render_v0_2node`
//!
//! Independent of production `templates/`. Substitutes `{{RUN_ID}}` and
//! `{{WALLTIME}}` in `tests/reference/v0_2node.sh`. Empty `run_id` or
//! `walltime` is `Error::Other`. The stub requests 2 nodes × 4 H200s, 8 MPI
//! ranks (one per GPU via `--ntasks=8` / `--ntasks-per-node=4`, never
//! `--gpus-per-task`), intra-node non-MPI `all_reduce_perf -g 4` (override
//! `NCCL_TESTS_ALL_REDUCE`; stdout not `busbw_gbps` /
//! `nccl_allreduce_2node.txt`; not `srun -N 2` of that binary),
//! `srun --mpi=pmix` of `all_reduce_perf_mpi` (not `mpirun`; override
//! `NCCL_TESTS_ALL_REDUCE_MPI`), PMIx env (`PMIX_MCA_gds=hash`,
//! `unset OMPI_MCA_mca_base_component_path`, system OpenMPI `openmpi/lib`
//! on `LD_LIBRARY_PATH`), writes `busbw_gbps` from the 2-node log and
//! `node_facts.txt` (`/dev/kvm` and NVMe on every node via `srun -N 2`
//! `--ntasks-per-node=1`), and builds the venv inside the job.
//!
//! # `check_exit` on-disk layout
//!
//! Checkers read `output_dir` only. Files named `summary`, `summary.json`,
//! `agent_summary`, or `agent_summary.json` are **ignored**.
//!
//! ## V0
//!
//! * `busbw_gbps` — either a single finite float or nccl-tests stdout that
//!   `parse_busbw_gbps` accepts. Missing file → `MissingOutput("busbw_gbps")`.
//!   Present but unparseable / non-finite → `ExitFailed` (not MissingOutput).
//! * `node_facts.txt` — facts for `kvm_available`. Missing file →
//!   `MissingOutput("node_facts.txt")`. Present but unreadable/empty/unknown →
//!   `ExitFailed`. `Ok(true)` and `Ok(false)` both **pass** (the criterion is
//!   "known whether Firecracker can run there").
//!
//! ## V1–V10
//!
//! One JSON object `{stage}.json` (e.g. `v1.json`). Missing file →
//! `MissingOutput`. Invalid JSON / non-object → `ExitFailed`. Missing key →
//! `MissingOutput(key)`. Wrong JSON type → `ExitFailed`. Values that miss the
//! 16.2 threshold → `ExitFailed`.
//!
//! | stage | file    | keys / pass rule |
//! |-------|---------|------------------|
//! | V1    | v1.json | `logits_max_diff` ≤ 1e-5; `grad_ok`; `overfit_ok` |
//! | V2    | v2.json | `loss_rel_diff` ≤ 1e-6; `routing_identical` |
//! | V3    | v3.json | `fp8_loss_rel_diff` ≤ 0.005 (0.5%); `nvfp4_numerics_ok` |
//! | V4    | v4.json | `resumed_bitwise_equal`; `sdc_caught_flip`; `spike_rollback_skipped_shard` |
//! | V5    | v5.json | `loss_curve_matches_ladder`; `checkpoint_resume_ok` |
//! | V6    | v6.json | `curriculum_no_collapse`; `accuracy_rises_with_latent_budget`; `thoughts_decode` |
//! | V7    | v7.json | `reward_rises`; `logprob_drift_halted`; `planted_write_flagged` |
//! | V8    | v8.json | `logprob_within_threshold`; `tiered_restore_matches` |
//! | V9    | v9.json | `planted_positive_found`; `planted_positive_replicated`; `planted_negative_recorded` |
//! | V10   | v10.json | `hours` ≥ 72; `lost_tasks` == 0; `duplicated_outputs` == 0; `dead_tokens` == 0 |
//!
//! Extra JSON keys are ignored. `output_dir` that is not a directory →
//! `MissingOutput`.
//!
//! # V1 CUDA extra (`v1_cuda`)
//!
//! Independent of production `templates/v1.sh` and of the crate sources. Knows
//! extra name `cuda` (`V1_CUDA_EXTRA`) and pip extra syntax `.[cuda]`. Walks
//! PEP 621 `pyproject.toml` tables by hand (no toml crate): the extra lives
//! under `[project.optional-dependencies]`, lists a portable GPU JAX extra
//! (`jax[cuda12]` / `jax[cuda13]`, not empty, not a site path), and is **not**
//! pulled in by default `[project] dependencies` (CPU CI / `uv run pytest`).
//! A rendered V1 job must `pip install -e ".[cuda]"` and then fail closed if
//! JAX is not using a GPU. `/work/ttennant1` and other site paths are rejected.

#![allow(dead_code, unused_imports)]

mod check;
mod kvm;
mod parse;
mod render;
pub mod v1_cuda;

use std::path::Path;

use prometheus_verify_ncshare::{Error, Result, Stage};

pub use check::check_exit;
pub use kvm::kvm_available;
pub use parse::parse_busbw_gbps;
pub use render::{render_job, render_v0_2node};

pub const NCCL_INTRA: &str = include_str!("../fixtures/nccl/allreduce_intra.txt");
pub const NCCL_2NODE: &str = include_str!("../fixtures/nccl/allreduce_2node.txt");
pub const NCCL_HEADER_ONLY: &str = include_str!("../fixtures/nccl/header_only.txt");
pub const NCCL_TOO_FEW_COLUMNS: &str = include_str!("../fixtures/nccl/too_few_columns.txt");

/// Out-of-place busbw of the last data row in `NCCL_INTRA`.
pub const INTRA_OOP_BUSBW: f64 = 52.20;
/// Out-of-place busbw of the last data row in `NCCL_2NODE`.
pub const CROSS_OOP_BUSBW: f64 = 412.35;

/// Decoy in-place busbw on the last intra row (must not be returned).
pub const INTRA_INPLACE_DECOY: f64 = 99.99;
/// Decoy `# Avg bus bandwidth` on the intra golden.
pub const INTRA_AVG_DECOY: f64 = 17.423333;

pub const FACTS_KVM_CHAR: &str = include_str!("../fixtures/node_facts/kvm_char_device.txt");
pub const FACTS_KVM_YES: &str = include_str!("../fixtures/node_facts/kvm_yes.txt");
pub const FACTS_KVM_NO: &str = include_str!("../fixtures/node_facts/kvm_no.txt");
pub const FACTS_KVM_ABSENT: &str = include_str!("../fixtures/node_facts/kvm_absent.txt");
pub const FACTS_NO_KVM: &str = include_str!("../fixtures/node_facts/no_kvm_mention.txt");

pub const BUSBW_TOL: f64 = 1e-6;

pub fn err_kind(err: &Error) -> &'static str {
    match err {
        Error::UnknownStage(_) => "unknown_stage",
        Error::ExitFailed(_) => "exit_failed",
        Error::MissingOutput(_) => "missing_output",
        Error::Other(_) => "other",
    }
}

pub fn assert_close(got: f64, expected: f64) {
    assert!(
        (got - expected).abs() <= BUSBW_TOL,
        "busbw {got} != {expected} (tol {BUSBW_TOL})"
    );
}

pub fn write_v0(dir: &Path, busbw_body: &str, node_facts: &str) {
    std::fs::write(dir.join("busbw_gbps"), busbw_body).expect("write busbw_gbps");
    std::fs::write(dir.join("node_facts.txt"), node_facts).expect("write node_facts.txt");
}

pub fn write_stage_json(dir: &Path, stage: Stage, json: &str) {
    let name = format!("{}.json", stage.as_str());
    std::fs::write(dir.join(name), json).expect("write stage json");
}

pub fn pass_json(stage: Stage) -> &'static str {
    check::pass_json(stage)
}

pub fn fail_json(stage: Stage) -> &'static str {
    check::fail_json(stage)
}

pub fn json_filename(stage: Stage) -> String {
    format!("{}.json", stage.as_str())
}

pub fn assert_ok(result: Result<()>) {
    result.unwrap_or_else(|e| panic!("expected Ok, got Err({e:?})"));
}

pub fn assert_missing(result: Result<()>, needle: &str) {
    match result {
        Err(Error::MissingOutput(name)) => {
            assert!(
                name.contains(needle),
                "MissingOutput({name:?}) does not contain {needle:?}"
            );
        }
        other => panic!("expected MissingOutput containing {needle:?}, got {other:?}"),
    }
}

pub fn assert_exit_failed(result: Result<()>) {
    match result {
        Err(Error::ExitFailed(_)) => {}
        other => panic!("expected ExitFailed, got {other:?}"),
    }
}

pub fn assert_same_kind(prod: Result<()>, refer: Result<()>) {
    match (prod, refer) {
        (Ok(()), Ok(())) => {}
        (Err(p), Err(r)) => assert_eq!(
            err_kind(&p),
            err_kind(&r),
            "error kind mismatch: prod={p:?} ref={r:?}"
        ),
        (Ok(()), Err(r)) => panic!("prod Ok, ref Err({r:?})"),
        (Err(p), Ok(())) => panic!("prod Err({p:?}), ref Ok"),
    }
}

pub fn json_stages() -> &'static [Stage] {
    &[
        Stage::V1,
        Stage::V2,
        Stage::V3,
        Stage::V4,
        Stage::V5,
        Stage::V6,
        Stage::V7,
        Stage::V8,
        Stage::V9,
        Stage::V10,
    ]
}
