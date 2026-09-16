//! `parse_state` against common `sacct`/`squeue` tokens.
//!
//! These tests call the unimplemented production function and must fail on
//! the stub. They must pass once `parse_state` is real.

use prometheus_slurm::{parse_state, JobState};

#[test]
fn parse_state_maps_common_sacct_tokens() {
    let cases: &[(&str, Option<JobState>)] = &[
        ("PENDING", Some(JobState::Pending)),
        ("RUNNING", Some(JobState::Running)),
        ("COMPLETED", Some(JobState::Completed)),
        ("FAILED", Some(JobState::Failed)),
        ("CANCELLED", Some(JobState::Cancelled)),
        ("TIMEOUT", Some(JobState::Timeout)),
        ("PREEMPTED", Some(JobState::Preempted)),
        ("NODE_FAIL", Some(JobState::NodeFail)),
        ("OUT_OF_MEMORY", Some(JobState::OutOfMemory)),
    ];
    for (raw, expected) in cases {
        assert_eq!(
            parse_state(raw),
            *expected,
            "parse_state({raw:?}) should map the sacct token"
        );
    }
}

#[test]
fn parse_state_completing_is_not_a_jobstate_variant() {
    // COMPLETING is a real Slurm state but has no JobState arm. The public
    // docstring says None for unrecognised tokens. wait() must keep polling
    // (Unknown/None is non-terminal).
    assert_eq!(parse_state("COMPLETING"), None);
}

#[test]
fn parse_state_empty_and_garbage_are_none() {
    assert_eq!(parse_state(""), None);
    assert_eq!(parse_state("   "), None);
    assert_eq!(parse_state("garbage"), None);
    assert_eq!(parse_state("NOT_A_STATE"), None);
    assert_eq!(parse_state("???"), None);
}

#[test]
fn parse_state_accepts_common_sacct_noise() {
    // sacct pads fields, appends '+' when steps disagree, and CANCELLED may
    // include "by <uid>".
    assert_eq!(parse_state(" PENDING "), Some(JobState::Pending));
    assert_eq!(parse_state("RUNNING\n"), Some(JobState::Running));
    assert_eq!(parse_state("COMPLETED+"), Some(JobState::Completed));
    assert_eq!(parse_state("CANCELLED by 12345"), Some(JobState::Cancelled));
    assert_eq!(parse_state("FAILED+"), Some(JobState::Failed));
}
