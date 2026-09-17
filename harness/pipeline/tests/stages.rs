//! Group: stage machine, WrongStage, stub-must-fail, retries, planted veto.

mod common;
mod reference;

use common::{
    assert_no_candidate, assert_planted_bad, assert_state, assert_stub_not_failed,
    assert_stub_passed, assert_wrong_stage, assert_wrong_stage_expected, cand, cfg,
    default_pipeline_config, default_queue_config, drive_to_implement, drive_to_oracle,
    drive_to_review, drive_to_stub_must_fail, good_patch, merge_record, module, payloads_with_role,
    pick, planted_patch, reject_all, unwrap_err, NOW,
};
use prometheus_pipeline::{Pipeline, Stage, ROLE_IMPLEMENTER};
use reference::RefPipeline;

fn pair() -> (tempfile::TempDir, std::path::PathBuf, Pipeline, RefPipeline) {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let m = module();
    let cfg = default_pipeline_config();
    let qcfg = default_queue_config();
    let p = Pipeline::create(&prod_dir, m.clone(), cfg.clone(), qcfg.clone()).expect("prod create");
    let r = RefPipeline::create(&ref_dir, m, cfg, qcfg).expect("ref create");
    (parent, prod_dir, p, r)
}

#[test]
fn start_oracle_interface_to_oracle() {
    let (_t, _d, mut p, mut r) = pair();
    let a = p.start_oracle(NOW).expect("prod");
    let b = r.start_oracle(NOW).expect("ref");
    assert_eq!(a.0, b.0);
    assert_eq!(p.stage(), Stage::Oracle);
    assert_eq!(p.round(), 0);
    assert!(!p.stub_failed());
    assert_state(&p, &r, "after start_oracle");
}

#[test]
fn record_oracle_false_to_stub_must_fail() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_oracle(&mut p, NOW);
    drive_to_oracle(&mut r, NOW);
    p.record_oracle(false, NOW).expect("prod");
    r.record_oracle(false, NOW).expect("ref");
    assert_eq!(p.stage(), Stage::StubMustFail);
    assert_eq!(r.stage(), Stage::StubMustFail);
    assert!(p.stub_failed());
    assert_eq!(p.round(), 0);
    assert_state(&p, &r, "record_oracle false");
}

#[test]
fn record_oracle_true_is_stub_passed_stays_oracle() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_oracle(&mut p, NOW);
    drive_to_oracle(&mut r, NOW);
    assert_stub_passed(unwrap_err(p.record_oracle(true, NOW), "prod stub passed"));
    assert_stub_passed(unwrap_err(r.record_oracle(true, NOW), "ref stub passed"));
    assert_eq!(p.stage(), Stage::Oracle);
    assert!(!p.stub_failed());
    assert_state(&p, &r, "stub passed");
}

#[test]
fn start_implementers_before_stub_fail_is_stub_not_failed() {
    let (_t, _d, mut p, mut r) = pair();
    assert_eq!(p.stage(), Stage::Interface);
    assert_stub_not_failed(unwrap_err(p.start_implementers(NOW), "prod"));
    assert_stub_not_failed(unwrap_err(r.start_implementers(NOW), "ref"));
    assert_eq!(p.stage(), Stage::Interface);
    drive_to_oracle(&mut p, NOW);
    drive_to_oracle(&mut r, NOW);
    assert_stub_not_failed(unwrap_err(p.start_implementers(NOW), "prod at oracle"));
    assert_eq!(p.stage(), Stage::Oracle);
}

#[test]
fn start_implementers_advances_to_implement() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_stub_must_fail(&mut p, NOW);
    drive_to_stub_must_fail(&mut r, NOW);
    let ids = p.start_implementers(NOW).expect("prod");
    let rids = r.start_implementers(NOW).expect("ref");
    assert_eq!(ids.len(), 3);
    assert_eq!(ids, rids);
    assert_eq!(p.stage(), Stage::Implement);
    assert_state(&p, &r, "implement");
}

#[test]
fn submit_unknown_candidate_is_no_candidate() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_implement(&mut p, NOW);
    drive_to_implement(&mut r, NOW);
    let c = cand("nope", "x", good_patch());
    assert_no_candidate(
        unwrap_err(p.submit_candidate(c.clone(), NOW), "prod"),
        "nope",
    );
    assert_no_candidate(unwrap_err(r.submit_candidate(c, NOW), "ref"), "nope");
    assert!(p.candidates().expect("cands").is_empty());
    assert_eq!(p.stage(), Stage::Implement);
}

#[test]
fn submit_records_in_first_submit_order_last_write_wins() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_implement(&mut p, NOW);
    drive_to_implement(&mut r, NOW);
    p.submit_candidate(cand("1", "first-1", b"a"), NOW).unwrap();
    p.submit_candidate(cand("0", "first-0", b"b"), NOW).unwrap();
    p.submit_candidate(cand("1", "updated-1", b"c"), NOW)
        .unwrap();
    r.submit_candidate(cand("1", "first-1", b"a"), NOW).unwrap();
    r.submit_candidate(cand("0", "first-0", b"b"), NOW).unwrap();
    r.submit_candidate(cand("1", "updated-1", b"c"), NOW)
        .unwrap();
    let got = p.candidates().expect("cands");
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].id.0, "1");
    assert_eq!(got[0].angle, "updated-1");
    assert_eq!(got[0].patch, b"c");
    assert_eq!(got[1].id.0, "0");
    assert_state(&p, &r, "submit order");
}

#[test]
fn planted_candidate_may_be_submitted() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_implement(&mut p, NOW);
    drive_to_implement(&mut r, NOW);
    let patch = planted_patch();
    p.submit_candidate(cand("0", "evil", &patch), NOW).unwrap();
    r.submit_candidate(cand("0", "evil", &patch), NOW).unwrap();
    assert_eq!(p.candidates().unwrap()[0].patch, patch);
    assert_eq!(p.stage(), Stage::Implement);
}

#[test]
fn start_review_from_implement() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_implement(&mut p, NOW);
    drive_to_implement(&mut r, NOW);
    p.submit_candidate(cand("0", "a", good_patch()), NOW)
        .unwrap();
    r.submit_candidate(cand("0", "a", good_patch()), NOW)
        .unwrap();
    p.start_review(NOW).unwrap();
    r.start_review(NOW).unwrap();
    assert_eq!(p.stage(), Stage::Review);
    assert_state(&p, &r, "review");
}

#[test]
fn record_verdict_pick_good_goes_merged() {
    let (_t, _d, mut p, mut r) = pair();
    let patches = [good_patch(), good_patch(), good_patch()];
    drive_to_review(&mut p, NOW, &patches);
    drive_to_review(&mut r, NOW, &patches);
    let st = p.record_verdict(pick("1"), NOW).expect("prod pick");
    let rst = r.record_verdict(pick("1"), NOW).expect("ref pick");
    assert_eq!(st, Stage::Merged);
    assert_eq!(rst, Stage::Merged);
    assert_state(&p, &r, "merged");
}

#[test]
fn record_verdict_pick_planted_stays_review() {
    let (_t, _d, mut p, mut r) = pair();
    let evil = planted_patch();
    let patches: [&[u8]; 3] = [good_patch(), evil.as_slice(), good_patch()];
    drive_to_review(&mut p, NOW, &patches);
    drive_to_review(&mut r, NOW, &patches);
    assert_planted_bad(unwrap_err(p.record_verdict(pick("1"), NOW), "prod planted"));
    assert_planted_bad(unwrap_err(r.record_verdict(pick("1"), NOW), "ref planted"));
    assert_eq!(p.stage(), Stage::Review);
    assert_state(&p, &r, "planted pick");
}

#[test]
fn record_verdict_pick_unknown_is_no_candidate() {
    let (_t, _d, mut p, mut r) = pair();
    let patches = [good_patch(), good_patch(), good_patch()];
    drive_to_review(&mut p, NOW, &patches);
    drive_to_review(&mut r, NOW, &patches);
    assert_no_candidate(unwrap_err(p.record_verdict(pick("9"), NOW), "prod"), "9");
    assert_eq!(p.stage(), Stage::Review);
}

#[test]
fn record_verdict_reject_all_retries_implement_and_bumps_round() {
    let (_t, _d, mut p, mut r) = pair();
    let patches = [good_patch(), good_patch(), good_patch()];
    drive_to_review(&mut p, NOW, &patches);
    drive_to_review(&mut r, NOW, &patches);
    let st = p
        .record_verdict(reject_all(&["none good"]), NOW)
        .expect("prod reject");
    let rst = r
        .record_verdict(reject_all(&["none good"]), NOW)
        .expect("ref reject");
    assert_eq!(st, Stage::Implement);
    assert_eq!(rst, Stage::Implement);
    assert_eq!(p.round(), 1);
    assert!(p.candidates().unwrap().is_empty());
    assert_state(&p, &r, "retry");
    let impls = payloads_with_role(p.queue(), ROLE_IMPLEMENTER);
    assert_eq!(impls.len(), 6, "3 first round + 3 retry");
    let round1: Vec<_> = impls.iter().filter(|v| v["round"] == 1).collect();
    assert_eq!(round1.len(), 3);
}

#[test]
fn reject_all_at_max_rounds_blocks() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let cfg = cfg(1, 1);
    let mut p =
        Pipeline::create(&prod_dir, module(), cfg.clone(), default_queue_config()).expect("create");
    let mut r = RefPipeline::create(
        parent.path().join("refer"),
        module(),
        cfg,
        default_queue_config(),
    )
    .expect("ref");
    drive_to_review(&mut p, NOW, &[good_patch()]);
    drive_to_review(&mut r, NOW, &[good_patch()]);
    let st = p
        .record_verdict(reject_all(&["no"]), NOW)
        .expect("prod block");
    let rst = r
        .record_verdict(reject_all(&["no"]), NOW)
        .expect("ref block");
    assert_eq!(st, Stage::Blocked);
    assert_eq!(rst, Stage::Blocked);
    assert_eq!(p.round(), 1);
    assert_state(&p, &r, "blocked");
}

#[test]
fn three_rejects_with_default_max_rounds_block() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_implement(&mut p, NOW);
    drive_to_implement(&mut r, NOW);
    for expected_round in 0..3u32 {
        assert_eq!(p.round(), expected_round);
        p.start_review(NOW).unwrap();
        r.start_review(NOW).unwrap();
        let st = p.record_verdict(reject_all(&["x"]), NOW).unwrap();
        let rst = r.record_verdict(reject_all(&["x"]), NOW).unwrap();
        if expected_round + 1 >= 3 {
            assert_eq!(st, Stage::Blocked);
            assert_eq!(rst, Stage::Blocked);
            assert_eq!(p.round(), 3);
        } else {
            assert_eq!(st, Stage::Implement);
            assert_eq!(p.round(), expected_round + 1);
        }
        assert_state(&p, &r, &format!("reject round {expected_round}"));
    }
}

#[test]
fn record_merge_at_merged_stays_merged() {
    let (_t, _d, mut p, mut r) = pair();
    let patches = [good_patch(), good_patch(), good_patch()];
    drive_to_review(&mut p, NOW, &patches);
    drive_to_review(&mut r, NOW, &patches);
    p.record_verdict(pick("0"), NOW).unwrap();
    r.record_verdict(pick("0"), NOW).unwrap();
    p.record_merge(merge_record(p.module(), 1, 0), NOW).unwrap();
    r.record_merge(merge_record(r.module(), 1, 0), NOW).unwrap();
    assert_eq!(p.stage(), Stage::Merged);
    assert_state(&p, &r, "after merge");
}

#[test]
fn wrong_stage_matrix_at_interface() {
    let (_t, _d, mut p, _r) = pair();
    assert_eq!(p.stage(), Stage::Interface);
    assert_wrong_stage_expected(
        unwrap_err(p.record_oracle(false, NOW), "record_oracle"),
        Stage::Oracle,
        Stage::Interface,
    );
    assert_wrong_stage_expected(
        unwrap_err(p.submit_candidate(cand("0", "a", b"x"), NOW), "submit"),
        Stage::Implement,
        Stage::Interface,
    );
    assert_wrong_stage_expected(
        unwrap_err(p.start_review(NOW), "start_review"),
        Stage::Implement,
        Stage::Interface,
    );
    assert_wrong_stage_expected(
        unwrap_err(p.record_verdict(pick("0"), NOW), "verdict"),
        Stage::Review,
        Stage::Interface,
    );
    assert_wrong_stage_expected(
        unwrap_err(p.record_merge(merge_record(p.module(), 0, 0), NOW), "merge"),
        Stage::Merged,
        Stage::Interface,
    );
}

#[test]
fn wrong_stage_after_implement() {
    let (_t, _d, mut p, _r) = pair();
    drive_to_implement(&mut p, NOW);
    assert_wrong_stage(
        unwrap_err(p.start_oracle(NOW), "start_oracle"),
        Stage::Implement,
    );
    assert_wrong_stage(
        unwrap_err(p.record_oracle(false, NOW), "record_oracle"),
        Stage::Implement,
    );
    assert_wrong_stage_expected(
        unwrap_err(p.start_implementers(NOW), "start_implementers again"),
        Stage::StubMustFail,
        Stage::Implement,
    );
    assert_wrong_stage(
        unwrap_err(p.record_verdict(pick("0"), NOW), "verdict"),
        Stage::Implement,
    );
}

#[test]
fn blocked_rejects_all_mutations() {
    let parent = tempfile::tempdir().expect("tempdir");
    let mut p = Pipeline::create(
        parent.path().join("prod"),
        module(),
        cfg(1, 1),
        default_queue_config(),
    )
    .expect("create");
    drive_to_review(&mut p, NOW, &[good_patch()]);
    p.record_verdict(reject_all(&["x"]), NOW).unwrap();
    assert_eq!(p.stage(), Stage::Blocked);
    assert_wrong_stage(unwrap_err(p.start_oracle(NOW), "oracle"), Stage::Blocked);
    assert_wrong_stage(
        unwrap_err(p.start_implementers(NOW), "impl"),
        Stage::Blocked,
    );
    assert_wrong_stage(unwrap_err(p.start_review(NOW), "review"), Stage::Blocked);
    assert_wrong_stage(
        unwrap_err(p.submit_candidate(cand("0", "a", b"x"), NOW), "submit"),
        Stage::Blocked,
    );
    assert_wrong_stage(
        unwrap_err(p.record_verdict(pick("0"), NOW), "verdict"),
        Stage::Blocked,
    );
    assert_wrong_stage(
        unwrap_err(p.record_merge(merge_record(p.module(), 0, 0), NOW), "merge"),
        Stage::Blocked,
    );
}

#[test]
fn second_start_oracle_is_wrong_stage() {
    let (_t, _d, mut p, _r) = pair();
    p.start_oracle(NOW).unwrap();
    assert_wrong_stage_expected(
        unwrap_err(p.start_oracle(NOW), "second"),
        Stage::Interface,
        Stage::Oracle,
    );
}

#[test]
fn start_review_allowed_with_zero_submits() {
    let (_t, _d, mut p, mut r) = pair();
    drive_to_implement(&mut p, NOW);
    drive_to_implement(&mut r, NOW);
    p.start_review(NOW).unwrap();
    r.start_review(NOW).unwrap();
    assert_eq!(p.stage(), Stage::Review);
    assert!(p.candidates().unwrap().is_empty());
    assert_state(&p, &r, "empty review");
}
