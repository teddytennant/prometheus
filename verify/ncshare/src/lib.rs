//! V0 to V10 templates and exit checkers (spec 16, 15.5 F4).
//!
//! Each stage has a job template under `templates/` and an exit checker
//! that reads the job's own output files. The harness never takes an
//! agent's summary as the criterion (16.2).
//!
//! GPU counts and the exit text below are copied from the 16.2 table.
//! 8-GPU windows (V2, V5) are real: the caller stops gpu-opportunist
//! around them. Allowed day-to-day counts stay 1, 2, or 4 (H7).
//!
//! V-stage exit checks are written by the oracle, not by this interface.
//! Every checker here is `unimplemented!`.
//!
//! Job-name prefix is `fv-`, matching H7.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unknown stage {0}")]
    UnknownStage(String),
    #[error("exit criterion failed: {0}")]
    ExitFailed(String),
    #[error("missing output {0}")]
    MissingOutput(String),
    #[error("verify: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub const JOB_PREFIX: &str = "fv";

/// Spec 16.2 stages, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stage {
    V0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V7,
    V8,
    V9,
    V10,
}

impl Stage {
    pub const ALL: [Stage; 11] = [
        Stage::V0,
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
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Stage::V0 => "v0",
            Stage::V1 => "v1",
            Stage::V2 => "v2",
            Stage::V3 => "v3",
            Stage::V4 => "v4",
            Stage::V5 => "v5",
            Stage::V6 => "v6",
            Stage::V7 => "v7",
            Stage::V8 => "v8",
            Stage::V9 => "v9",
            Stage::V10 => "v10",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "v0" | "V0" => Ok(Stage::V0),
            "v1" | "V1" => Ok(Stage::V1),
            "v2" | "V2" => Ok(Stage::V2),
            "v3" | "V3" => Ok(Stage::V3),
            "v4" | "V4" => Ok(Stage::V4),
            "v5" | "V5" => Ok(Stage::V5),
            "v6" | "V6" => Ok(Stage::V6),
            "v7" | "V7" => Ok(Stage::V7),
            "v8" | "V8" => Ok(Stage::V8),
            "v9" | "V9" => Ok(Stage::V9),
            "v10" | "V10" => Ok(Stage::V10),
            other => Err(Error::UnknownStage(other.to_string())),
        }
    }
}

/// One row of the 16.2 table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageSpec {
    pub stage: Stage,
    /// GPU count the template requests. `None` means shared / soak.
    pub gpus: Option<u32>,
    pub what: &'static str,
    pub exit: &'static str,
}

/// Spec 16.2, verbatim intent. GPU numbers are the primary column.
pub const STAGES: [StageSpec; 11] = [
    StageSpec {
        stage: Stage::V0,
        gpus: Some(1),
        what: "venv inside a job, sbatch --test-only, nccl-tests all_reduce intra-node and across 2 nodes, check /dev/kvm and NVMe on compute nodes, measure queue waits",
        exit: "measured bus bandwidth recorded; known whether Firecracker can run there",
    },
    StageSpec {
        stage: Stage::V1,
        gpus: Some(1),
        what: "~10M-param model with the flagship shape, each custom kernel",
        exit: "JAX vs PyTorch reference logits match to 1e-5 in FP32; grad checks pass; overfits one batch",
    },
    StageSpec {
        stage: Stage::V2,
        gpus: Some(8),
        what: "same model on 1 GPU vs EP=8, FSDP=8, PP=2 with 4 GPUs per stage, CP=2; then DP across 2 nodes",
        exit: "loss matches single-device to 1e-6 relative in FP32 over 200 steps; routing identical",
    },
    StageSpec {
        stage: Stage::V3,
        gpus: Some(8),
        what: "BF16 and FP8 at 0.1 to 0.5B for 2k to 5k steps; NVFP4 by fake-quant emulation",
        exit: "FP8 loss within 0.5% of BF16; NVFP4 checked for numerics only, never speed",
    },
    StageSpec {
        stage: Stage::V4,
        gpus: Some(4),
        what: "kill -9 a rank mid-step, elastic DP continues; in-memory checkpoint restore; injected bit flip; injected bad shard; host-RAM stand-in for Grace offload",
        exit: "resumed run bitwise-equal to uninterrupted run; SDC hash catches the flip within N steps; spike rollback skips the shard",
    },
    StageSpec {
        stage: Stage::V5,
        gpus: Some(8),
        what: "0.1B / 1B MoE, 20B tokens; rung-1 shape at 10B tokens only",
        exit: "loss curve matches the ladder's small-scale fit; checkpoint/resume across job boundaries works",
    },
    StageSpec {
        stage: Stage::V6,
        gpus: Some(4),
        what: "Stage A then Stage B compression at 0.1 to 0.5B on verified math; latent budget sweep 1x to 8x",
        exit: "curriculum trains without collapse; held-out accuracy rises with latent budget on problems the model can't one-shot; thoughts decode to their steps",
    },
    StageSpec {
        stage: Stage::V7,
        gpus: Some(8),
        what: "GSPO/DAPO loss, async staleness, routing replay, weight sync, parity halt, reward-hacking controls",
        exit: "reward rises on a tiny verifiable task; an injected log-prob drift halts RL; a planted test-file write is flagged",
    },
    StageSpec {
        stage: Stage::V8,
        gpus: Some(1),
        what: "SGLang fork loads a JAX checkpoint; latent decode, recurrence buckets, MTP speculative decode, KV tiering",
        exit: "SGLang vs JAX log-probs within threshold; tiered session restore matches unswapped output",
    },
    StageSpec {
        stage: Stage::V9,
        gpus: Some(1),
        what: "one full 14.6 cycle through the Slurm backend",
        exit: "a planted known-positive idea is found and replicated; a planted negative is recorded as negative",
    },
    StageSpec {
        stage: Stage::V10,
        gpus: None,
        what: "harness chaos suite (15.2) while V5 to V9 jobs run",
        exit: "72h with no lost task, no duplicated output, no dead token",
    },
];

pub fn spec(stage: Stage) -> StageSpec {
    STAGES
        .iter()
        .copied()
        .find(|s| s.stage == stage)
        .expect("STAGES covers every Stage")
}

/// Path to the sbatch template for `stage`, relative to this crate's `templates/`.
pub fn template_path(stage: Stage) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("templates")
        .join(format!("{}.sh", stage.as_str()))
}

/// Rendered job script. Substitutes run-id, walltime, gpus into the template.
pub fn render_job(stage: Stage, run_id: &str, walltime: &str, gpus: u32) -> Result<String> {
    let _ = (stage, run_id, walltime, gpus);
    unimplemented!("F4 render_job")
}

/// Read the job output dir and decide pass/fail from the files, never a summary.
pub fn check_exit(stage: Stage, output_dir: &Path) -> Result<()> {
    let _ = (stage, output_dir);
    unimplemented!("F4 check_exit")
}

/// Parse a measured bus-bandwidth number from nccl-tests stdout. V0.
pub fn parse_busbw_gbps(nccl_stdout: &str) -> Result<f64> {
    let _ = nccl_stdout;
    unimplemented!("F4 parse_busbw_gbps")
}

/// True if `/dev/kvm` is present in the recorded node facts. V0 Firecracker gate.
pub fn kvm_available(node_facts: &str) -> Result<bool> {
    let _ = node_facts;
    unimplemented!("F4 kvm_available")
}
