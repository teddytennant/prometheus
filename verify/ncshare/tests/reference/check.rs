//! Slow exit checkers. File layout is documented in `reference/mod.rs`.

use std::path::Path;

use prometheus_verify_ncshare::{Error, Result, Stage};
use serde_json::Value;

use super::{kvm_available, parse_busbw_gbps};

const LOGITS_MAX: f64 = 1e-5;
const LOSS_REL_MAX: f64 = 1e-6;
const FP8_REL_MAX: f64 = 0.005;
const SOAK_HOURS_MIN: f64 = 72.0;

pub fn check_exit(stage: Stage, output_dir: &Path) -> Result<()> {
    if !output_dir.is_dir() {
        return Err(Error::MissingOutput(format!(
            "output_dir {}",
            output_dir.display()
        )));
    }
    match stage {
        Stage::V0 => check_v0(output_dir),
        Stage::V1
        | Stage::V2
        | Stage::V3
        | Stage::V4
        | Stage::V5
        | Stage::V6
        | Stage::V7
        | Stage::V8
        | Stage::V9
        | Stage::V10 => check_json_stage(stage, output_dir),
    }
}

fn check_v0(output_dir: &Path) -> Result<()> {
    let busbw_path = output_dir.join("busbw_gbps");
    if !busbw_path.is_file() {
        return Err(Error::MissingOutput("busbw_gbps".into()));
    }
    let busbw_text = std::fs::read_to_string(&busbw_path)
        .map_err(|e| Error::Other(format!("read busbw_gbps: {e}")))?;
    let busbw = parse_busbw_file(&busbw_text)?;
    if !busbw.is_finite() {
        return Err(Error::ExitFailed(format!("non-finite busbw {busbw}")));
    }

    let facts_path = output_dir.join("node_facts.txt");
    if !facts_path.is_file() {
        return Err(Error::MissingOutput("node_facts.txt".into()));
    }
    let facts = std::fs::read_to_string(&facts_path)
        .map_err(|e| Error::Other(format!("read node_facts.txt: {e}")))?;
    match kvm_available(&facts) {
        Ok(_known) => Ok(()),
        Err(e) => Err(Error::ExitFailed(format!("kvm facts: {e}"))),
    }
}

/// A recorded float, or nccl-tests stdout. Unparseable → `ExitFailed`.
fn parse_busbw_file(text: &str) -> Result<f64> {
    let trimmed = text.trim();
    if let Ok(v) = trimmed.parse::<f64>() {
        if v.is_finite() {
            return Ok(v);
        }
        return Err(Error::ExitFailed(format!("non-finite busbw {v}")));
    }
    match parse_busbw_gbps(text) {
        Ok(v) => Ok(v),
        Err(_) => Err(Error::ExitFailed("busbw unparseable".into())),
    }
}

fn check_json_stage(stage: Stage, output_dir: &Path) -> Result<()> {
    let name = format!("{}.json", stage.as_str());
    let path = output_dir.join(&name);
    if !path.is_file() {
        return Err(Error::MissingOutput(name));
    }
    let text =
        std::fs::read_to_string(&path).map_err(|e| Error::Other(format!("read {name}: {e}")))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|e| Error::ExitFailed(format!("invalid json in {name}: {e}")))?;
    let obj = value
        .as_object()
        .ok_or_else(|| Error::ExitFailed(format!("{name} must be a JSON object")))?;

    match stage {
        Stage::V1 => {
            let diff = require_f64(obj, "logits_max_diff")?;
            let grad = require_bool(obj, "grad_ok")?;
            let overfit = require_bool(obj, "overfit_ok")?;
            require_le("logits_max_diff", diff, LOGITS_MAX)?;
            require_true("grad_ok", grad)?;
            require_true("overfit_ok", overfit)?;
        }
        Stage::V2 => {
            let rel = require_f64(obj, "loss_rel_diff")?;
            let routing = require_bool(obj, "routing_identical")?;
            require_le("loss_rel_diff", rel, LOSS_REL_MAX)?;
            require_true("routing_identical", routing)?;
        }
        Stage::V3 => {
            let rel = require_f64(obj, "fp8_loss_rel_diff")?;
            let nvfp4 = require_bool(obj, "nvfp4_numerics_ok")?;
            require_le("fp8_loss_rel_diff", rel, FP8_REL_MAX)?;
            require_true("nvfp4_numerics_ok", nvfp4)?;
        }
        Stage::V4 => {
            require_true(
                "resumed_bitwise_equal",
                require_bool(obj, "resumed_bitwise_equal")?,
            )?;
            require_true("sdc_caught_flip", require_bool(obj, "sdc_caught_flip")?)?;
            require_true(
                "spike_rollback_skipped_shard",
                require_bool(obj, "spike_rollback_skipped_shard")?,
            )?;
        }
        Stage::V5 => {
            require_true(
                "loss_curve_matches_ladder",
                require_bool(obj, "loss_curve_matches_ladder")?,
            )?;
            require_true(
                "checkpoint_resume_ok",
                require_bool(obj, "checkpoint_resume_ok")?,
            )?;
        }
        Stage::V6 => {
            require_true(
                "curriculum_no_collapse",
                require_bool(obj, "curriculum_no_collapse")?,
            )?;
            require_true(
                "accuracy_rises_with_latent_budget",
                require_bool(obj, "accuracy_rises_with_latent_budget")?,
            )?;
            require_true("thoughts_decode", require_bool(obj, "thoughts_decode")?)?;
        }
        Stage::V7 => {
            require_true("reward_rises", require_bool(obj, "reward_rises")?)?;
            require_true(
                "logprob_drift_halted",
                require_bool(obj, "logprob_drift_halted")?,
            )?;
            require_true(
                "planted_write_flagged",
                require_bool(obj, "planted_write_flagged")?,
            )?;
        }
        Stage::V8 => {
            require_true(
                "logprob_within_threshold",
                require_bool(obj, "logprob_within_threshold")?,
            )?;
            require_true(
                "tiered_restore_matches",
                require_bool(obj, "tiered_restore_matches")?,
            )?;
        }
        Stage::V9 => {
            require_true(
                "planted_positive_found",
                require_bool(obj, "planted_positive_found")?,
            )?;
            require_true(
                "planted_positive_replicated",
                require_bool(obj, "planted_positive_replicated")?,
            )?;
            require_true(
                "planted_negative_recorded",
                require_bool(obj, "planted_negative_recorded")?,
            )?;
        }
        Stage::V10 => {
            let hours = require_f64(obj, "hours")?;
            let lost = require_f64(obj, "lost_tasks")?;
            let dup = require_f64(obj, "duplicated_outputs")?;
            let dead = require_f64(obj, "dead_tokens")?;
            if hours < SOAK_HOURS_MIN {
                return Err(Error::ExitFailed(format!(
                    "hours {hours} < {SOAK_HOURS_MIN}"
                )));
            }
            require_zero("lost_tasks", lost)?;
            require_zero("duplicated_outputs", dup)?;
            require_zero("dead_tokens", dead)?;
        }
        Stage::V0 => unreachable!("V0 is not a JSON stage"),
    }
    Ok(())
}

fn require_f64(obj: &serde_json::Map<String, Value>, key: &str) -> Result<f64> {
    match obj.get(key) {
        None => Err(Error::MissingOutput(key.into())),
        Some(Value::Number(n)) => n
            .as_f64()
            .ok_or_else(|| Error::ExitFailed(format!("{key} is not a finite number"))),
        Some(_) => Err(Error::ExitFailed(format!("{key} is not a number"))),
    }
}

fn require_bool(obj: &serde_json::Map<String, Value>, key: &str) -> Result<bool> {
    match obj.get(key) {
        None => Err(Error::MissingOutput(key.into())),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(Error::ExitFailed(format!("{key} is not a boolean"))),
    }
}

fn require_le(key: &str, got: f64, max: f64) -> Result<()> {
    if got > max {
        Err(Error::ExitFailed(format!("{key} {got} > {max}")))
    } else {
        Ok(())
    }
}

fn require_true(key: &str, got: bool) -> Result<()> {
    if got {
        Ok(())
    } else {
        Err(Error::ExitFailed(format!("{key} is false")))
    }
}

fn require_zero(key: &str, got: f64) -> Result<()> {
    if got == 0.0 {
        Ok(())
    } else {
        Err(Error::ExitFailed(format!("{key} {got} != 0")))
    }
}

pub fn pass_json(stage: Stage) -> &'static str {
    match stage {
        Stage::V0 => panic!("V0 has no JSON fixture"),
        Stage::V1 => r#"{"logits_max_diff":1e-6,"grad_ok":true,"overfit_ok":true}"#,
        Stage::V2 => r#"{"loss_rel_diff":1e-7,"routing_identical":true}"#,
        Stage::V3 => r#"{"fp8_loss_rel_diff":0.004,"nvfp4_numerics_ok":true}"#,
        Stage::V4 => {
            r#"{"resumed_bitwise_equal":true,"sdc_caught_flip":true,"spike_rollback_skipped_shard":true}"#
        }
        Stage::V5 => r#"{"loss_curve_matches_ladder":true,"checkpoint_resume_ok":true}"#,
        Stage::V6 => {
            r#"{"curriculum_no_collapse":true,"accuracy_rises_with_latent_budget":true,"thoughts_decode":true}"#
        }
        Stage::V7 => {
            r#"{"reward_rises":true,"logprob_drift_halted":true,"planted_write_flagged":true}"#
        }
        Stage::V8 => r#"{"logprob_within_threshold":true,"tiered_restore_matches":true}"#,
        Stage::V9 => {
            r#"{"planted_positive_found":true,"planted_positive_replicated":true,"planted_negative_recorded":true}"#
        }
        Stage::V10 => r#"{"hours":72,"lost_tasks":0,"duplicated_outputs":0,"dead_tokens":0}"#,
    }
}

pub fn fail_json(stage: Stage) -> &'static str {
    match stage {
        Stage::V0 => panic!("V0 has no JSON fixture"),
        Stage::V1 => r#"{"logits_max_diff":1e-4,"grad_ok":true,"overfit_ok":true}"#,
        Stage::V2 => r#"{"loss_rel_diff":1e-3,"routing_identical":true}"#,
        Stage::V3 => r#"{"fp8_loss_rel_diff":0.02,"nvfp4_numerics_ok":true}"#,
        Stage::V4 => {
            r#"{"resumed_bitwise_equal":false,"sdc_caught_flip":true,"spike_rollback_skipped_shard":true}"#
        }
        Stage::V5 => r#"{"loss_curve_matches_ladder":false,"checkpoint_resume_ok":true}"#,
        Stage::V6 => {
            r#"{"curriculum_no_collapse":false,"accuracy_rises_with_latent_budget":true,"thoughts_decode":true}"#
        }
        Stage::V7 => {
            r#"{"reward_rises":true,"logprob_drift_halted":true,"planted_write_flagged":false}"#
        }
        Stage::V8 => r#"{"logprob_within_threshold":false,"tiered_restore_matches":true}"#,
        Stage::V9 => {
            r#"{"planted_positive_found":true,"planted_positive_replicated":true,"planted_negative_recorded":false}"#
        }
        Stage::V10 => r#"{"hours":24,"lost_tasks":1,"duplicated_outputs":0,"dead_tokens":0}"#,
    }
}
