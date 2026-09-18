//! Oracle tests for `check_exit`.
//!
//! Checkers read `output_dir` only — never an agent summary. Fixture layout is
//! documented in `tests/reference/mod.rs`.
//!
//! V0 unparseable `busbw_gbps` → `Error::ExitFailed` (file is present).
//! Missing `busbw_gbps` / `node_facts.txt` / `{stage}.json` → `MissingOutput`.
//!
//! `ref_*` pass on the stub. `prod_*` call
//! `prometheus_verify_ncshare::check_exit` only and must panic
//! `unimplemented!` until F4 is implemented. Do not `#[should_panic]`.

mod reference;

use std::path::Path;

use prometheus_verify_ncshare::{check_exit, Error, Stage};
use reference::{
    assert_exit_failed, assert_missing, assert_ok, assert_same_kind, fail_json, json_filename,
    json_stages, pass_json, write_stage_json, write_v0, FACTS_KVM_NO, FACTS_KVM_YES, NCCL_INTRA,
};

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().expect("tempdir")
}

fn ignore_summaries(dir: &Path) {
    std::fs::write(
        dir.join("summary.json"),
        r#"{"pass":true,"note":"agent summary must be ignored"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("agent_summary"), "PASS\n").unwrap();
}

// ---------------------------------------------------------------------------
// Reference
// ---------------------------------------------------------------------------

#[test]
fn ref_check_exit_v0_pass_numeric_and_kvm_yes() {
    let dir = tmp();
    write_v0(dir.path(), "412.35\n", FACTS_KVM_YES);
    assert_ok(reference::check_exit(Stage::V0, dir.path()));
}

#[test]
fn ref_check_exit_v0_pass_numeric_and_kvm_no() {
    let dir = tmp();
    write_v0(dir.path(), "52.20\n", FACTS_KVM_NO);
    assert_ok(reference::check_exit(Stage::V0, dir.path()));
}

#[test]
fn ref_check_exit_v0_pass_nccl_stdout_file() {
    let dir = tmp();
    write_v0(dir.path(), NCCL_INTRA, FACTS_KVM_YES);
    assert_ok(reference::check_exit(Stage::V0, dir.path()));
}

#[test]
fn ref_check_exit_v0_missing_busbw() {
    let dir = tmp();
    std::fs::write(dir.path().join("node_facts.txt"), FACTS_KVM_YES).unwrap();
    assert_missing(reference::check_exit(Stage::V0, dir.path()), "busbw_gbps");
}

#[test]
fn ref_check_exit_v0_missing_node_facts() {
    let dir = tmp();
    std::fs::write(dir.path().join("busbw_gbps"), "412.35\n").unwrap();
    assert_missing(
        reference::check_exit(Stage::V0, dir.path()),
        "node_facts.txt",
    );
}

#[test]
fn ref_check_exit_v0_unparseable_busbw_is_exit_failed() {
    let dir = tmp();
    write_v0(dir.path(), "not-a-bandwidth\n", FACTS_KVM_YES);
    assert_exit_failed(reference::check_exit(Stage::V0, dir.path()));
}

#[test]
fn ref_check_exit_v0_empty_facts_is_exit_failed() {
    let dir = tmp();
    write_v0(dir.path(), "1.0\n", "   \n");
    assert_exit_failed(reference::check_exit(Stage::V0, dir.path()));
}

#[test]
fn ref_check_exit_json_stages_pass() {
    for &stage in json_stages() {
        let dir = tmp();
        write_stage_json(dir.path(), stage, pass_json(stage));
        assert_ok(reference::check_exit(stage, dir.path()));
    }
}

#[test]
fn ref_check_exit_json_stages_fail_criteria() {
    for &stage in json_stages() {
        let dir = tmp();
        write_stage_json(dir.path(), stage, fail_json(stage));
        assert_exit_failed(reference::check_exit(stage, dir.path()));
    }
}

#[test]
fn ref_check_exit_json_stages_missing_file() {
    for &stage in json_stages() {
        let dir = tmp();
        assert_missing(
            reference::check_exit(stage, dir.path()),
            &json_filename(stage),
        );
    }
}

#[test]
fn ref_check_exit_v1_missing_key() {
    let dir = tmp();
    write_stage_json(
        dir.path(),
        Stage::V1,
        r#"{"grad_ok":true,"overfit_ok":true}"#,
    );
    assert_missing(
        reference::check_exit(Stage::V1, dir.path()),
        "logits_max_diff",
    );
}

#[test]
fn ref_check_exit_v1_boundary_logits() {
    let dir = tmp();
    write_stage_json(
        dir.path(),
        Stage::V1,
        r#"{"logits_max_diff":1e-5,"grad_ok":true,"overfit_ok":true}"#,
    );
    assert_ok(reference::check_exit(Stage::V1, dir.path()));
}

#[test]
fn ref_check_exit_v2_routing_false() {
    let dir = tmp();
    write_stage_json(
        dir.path(),
        Stage::V2,
        r#"{"loss_rel_diff":1e-9,"routing_identical":false}"#,
    );
    assert_exit_failed(reference::check_exit(Stage::V2, dir.path()));
}

#[test]
fn ref_check_exit_v3_fp8_boundary() {
    let dir = tmp();
    write_stage_json(
        dir.path(),
        Stage::V3,
        r#"{"fp8_loss_rel_diff":0.005,"nvfp4_numerics_ok":true}"#,
    );
    assert_ok(reference::check_exit(Stage::V3, dir.path()));
}

#[test]
fn ref_check_exit_v10_hours_short() {
    let dir = tmp();
    write_stage_json(
        dir.path(),
        Stage::V10,
        r#"{"hours":71.9,"lost_tasks":0,"duplicated_outputs":0,"dead_tokens":0}"#,
    );
    assert_exit_failed(reference::check_exit(Stage::V10, dir.path()));
}

#[test]
fn ref_check_exit_ignores_agent_summary() {
    let dir = tmp();
    write_stage_json(dir.path(), Stage::V1, fail_json(Stage::V1));
    ignore_summaries(dir.path());
    assert_exit_failed(reference::check_exit(Stage::V1, dir.path()));
}

#[test]
fn ref_check_exit_summary_only_is_missing_output() {
    let dir = tmp();
    ignore_summaries(dir.path());
    assert_missing(reference::check_exit(Stage::V1, dir.path()), "v1.json");
}

#[test]
fn ref_check_exit_malformed_json_is_exit_failed() {
    let dir = tmp();
    write_stage_json(dir.path(), Stage::V2, "{not json");
    assert_exit_failed(reference::check_exit(Stage::V2, dir.path()));
}

#[test]
fn ref_check_exit_missing_output_dir() {
    let dir = tmp();
    let gone = dir.path().join("nope");
    match reference::check_exit(Stage::V0, &gone) {
        Err(Error::MissingOutput(_)) => {}
        other => panic!("expected MissingOutput, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Production public API — must fail on unimplemented! stub
// ---------------------------------------------------------------------------

#[test]
fn prod_check_exit_v0_pass_numeric_and_kvm_yes() {
    let dir = tmp();
    write_v0(dir.path(), "412.35\n", FACTS_KVM_YES);
    assert_ok(check_exit(Stage::V0, dir.path()));
}

#[test]
fn prod_check_exit_v0_pass_kvm_no_is_still_known() {
    let dir = tmp();
    write_v0(dir.path(), "52.20\n", FACTS_KVM_NO);
    assert_ok(check_exit(Stage::V0, dir.path()));
}

#[test]
fn prod_check_exit_v0_pass_nccl_stdout_file() {
    let dir = tmp();
    write_v0(dir.path(), NCCL_INTRA, FACTS_KVM_YES);
    assert_ok(check_exit(Stage::V0, dir.path()));
}

#[test]
fn prod_check_exit_v0_missing_busbw() {
    let dir = tmp();
    std::fs::write(dir.path().join("node_facts.txt"), FACTS_KVM_YES).unwrap();
    assert_missing(check_exit(Stage::V0, dir.path()), "busbw_gbps");
}

#[test]
fn prod_check_exit_v0_unparseable_busbw_is_exit_failed() {
    let dir = tmp();
    write_v0(dir.path(), "not-a-bandwidth\n", FACTS_KVM_YES);
    assert_exit_failed(check_exit(Stage::V0, dir.path()));
}

#[test]
fn prod_check_exit_json_stages_pass() {
    for &stage in json_stages() {
        let dir = tmp();
        write_stage_json(dir.path(), stage, pass_json(stage));
        assert_ok(check_exit(stage, dir.path()));
    }
}

#[test]
fn prod_check_exit_json_stages_fail_criteria() {
    for &stage in json_stages() {
        let dir = tmp();
        write_stage_json(dir.path(), stage, fail_json(stage));
        assert_exit_failed(check_exit(stage, dir.path()));
    }
}

#[test]
fn prod_check_exit_json_stages_missing_file() {
    for &stage in json_stages() {
        let dir = tmp();
        assert_missing(check_exit(stage, dir.path()), &json_filename(stage));
    }
}

#[test]
fn prod_check_exit_v1_missing_key() {
    let dir = tmp();
    write_stage_json(
        dir.path(),
        Stage::V1,
        r#"{"grad_ok":true,"overfit_ok":true}"#,
    );
    assert_missing(check_exit(Stage::V1, dir.path()), "logits_max_diff");
}

#[test]
fn prod_check_exit_ignores_agent_summary() {
    let dir = tmp();
    write_stage_json(dir.path(), Stage::V1, fail_json(Stage::V1));
    ignore_summaries(dir.path());
    assert_exit_failed(check_exit(Stage::V1, dir.path()));
}

#[test]
fn prod_check_exit_summary_only_is_missing_output() {
    let dir = tmp();
    ignore_summaries(dir.path());
    assert_missing(check_exit(Stage::V1, dir.path()), "v1.json");
}

#[test]
fn prod_check_exit_matches_reference_on_fixtures() {
    let dir = tmp();
    write_v0(dir.path(), "412.35\n", FACTS_KVM_YES);
    assert_same_kind(
        check_exit(Stage::V0, dir.path()),
        reference::check_exit(Stage::V0, dir.path()),
    );

    for &stage in json_stages() {
        let pass_dir = tmp();
        write_stage_json(pass_dir.path(), stage, pass_json(stage));
        assert_same_kind(
            check_exit(stage, pass_dir.path()),
            reference::check_exit(stage, pass_dir.path()),
        );
        let fail_dir = tmp();
        write_stage_json(fail_dir.path(), stage, fail_json(stage));
        assert_same_kind(
            check_exit(stage, fail_dir.path()),
            reference::check_exit(stage, fail_dir.path()),
        );
        let empty = tmp();
        assert_same_kind(
            check_exit(stage, empty.path()),
            reference::check_exit(stage, empty.path()),
        );
    }
}
