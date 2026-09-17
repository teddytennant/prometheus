//! Group: drop + open reconstructs stage, round, stub_failed, candidates.

mod common;
mod reference;

use common::{
    assert_fresh_interface, assert_state, cand, cfg, default_pipeline_config, default_queue_config,
    drive_to_implement, drive_to_review, drive_to_stub_must_fail, fresh_dir, good_patch,
    merge_record, module, pick, planted_patch, reject_all, NOW,
};
use prometheus_pipeline::{Pipeline, PipelineConfig, Stage};
use reference::RefPipeline;

fn open_both(
    prod_dir: &std::path::Path,
    ref_dir: &std::path::Path,
    cfg: PipelineConfig,
) -> (Pipeline, RefPipeline) {
    let q = default_queue_config();
    let p = Pipeline::open(prod_dir, module(), cfg.clone(), q.clone()).expect("prod open");
    let r = RefPipeline::open(ref_dir, module(), cfg, q).expect("ref open");
    (p, r)
}

#[test]
fn replay_fresh_create_is_interface() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    {
        let p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q.clone()).expect("rc");
        assert_state(&p, &r, "pre-drop");
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg.clone());
    assert_fresh_interface(&p, &prod_dir, &module(), &cfg);
    assert_state(&p, &r, "open fresh");
}

#[test]
fn replay_after_oracle_started() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    {
        let mut p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let mut r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q).expect("rc");
        p.start_oracle(NOW).unwrap();
        r.start_oracle(NOW).unwrap();
        assert_eq!(p.stage(), Stage::Oracle);
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg);
    assert_eq!(p.stage(), Stage::Oracle);
    assert_eq!(p.round(), 0);
    assert!(!p.stub_failed());
    assert_state(&p, &r, "replay oracle");
}

#[test]
fn replay_after_stub_must_fail() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    {
        let mut p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let mut r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q).expect("rc");
        drive_to_stub_must_fail(&mut p, NOW);
        drive_to_stub_must_fail(&mut r, NOW);
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg);
    assert_eq!(p.stage(), Stage::StubMustFail);
    assert!(p.stub_failed());
    assert_state(&p, &r, "replay stub");
}

#[test]
fn replay_candidates_after_submit() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    let patch = planted_patch();
    {
        let mut p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let mut r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q).expect("rc");
        drive_to_implement(&mut p, NOW);
        drive_to_implement(&mut r, NOW);
        p.submit_candidate(cand("2", "z", good_patch()), NOW).unwrap();
        p.submit_candidate(cand("0", "evil", &patch), NOW).unwrap();
        r.submit_candidate(cand("2", "z", good_patch()), NOW).unwrap();
        r.submit_candidate(cand("0", "evil", &patch), NOW).unwrap();
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg);
    assert_eq!(p.stage(), Stage::Implement);
    let c = p.candidates().expect("cands");
    assert_eq!(c.len(), 2);
    assert_eq!(c[0].id.0, "2");
    assert_eq!(c[1].id.0, "0");
    assert_eq!(c[1].patch, patch);
    assert_state(&p, &r, "replay submits");
}

#[test]
fn replay_after_pick_and_merge() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    {
        let mut p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let mut r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q).expect("rc");
        let patches = [good_patch(), good_patch(), good_patch()];
        drive_to_review(&mut p, NOW, &patches);
        drive_to_review(&mut r, NOW, &patches);
        p.record_verdict(pick("0"), NOW).unwrap();
        r.record_verdict(pick("0"), NOW).unwrap();
        p.record_merge(merge_record(p.module(), 3, 0), NOW).unwrap();
        r.record_merge(merge_record(r.module(), 3, 0), NOW).unwrap();
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg);
    assert_eq!(p.stage(), Stage::Merged);
    assert_eq!(p.round(), 0);
    assert!(p.stub_failed());
    assert_eq!(p.candidates().unwrap().len(), 3);
    assert_state(&p, &r, "replay merge");
}

#[test]
fn replay_after_reject_all_new_round() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    {
        let mut p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let mut r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q).expect("rc");
        let patches = [good_patch(), good_patch(), good_patch()];
        drive_to_review(&mut p, NOW, &patches);
        drive_to_review(&mut r, NOW, &patches);
        p.record_verdict(reject_all(&["no"]), NOW).unwrap();
        r.record_verdict(reject_all(&["no"]), NOW).unwrap();
        assert_eq!(p.round(), 1);
        assert_eq!(p.stage(), Stage::Implement);
        assert!(p.candidates().unwrap().is_empty());
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg);
    assert_eq!(p.stage(), Stage::Implement);
    assert_eq!(p.round(), 1);
    assert!(p.candidates().unwrap().is_empty());
    assert!(p.stub_failed());
    assert_state(&p, &r, "replay reject");
}

#[test]
fn replay_blocked() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = cfg(1, 1);
    let q = default_queue_config();
    {
        let mut p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let mut r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q).expect("rc");
        drive_to_review(&mut p, NOW, &[good_patch()]);
        drive_to_review(&mut r, NOW, &[good_patch()]);
        p.record_verdict(reject_all(&["x"]), NOW).unwrap();
        r.record_verdict(reject_all(&["x"]), NOW).unwrap();
        assert_eq!(p.stage(), Stage::Blocked);
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg);
    assert_eq!(p.stage(), Stage::Blocked);
    assert_eq!(p.round(), 1);
    assert_state(&p, &r, "replay blocked");
}

#[test]
fn replay_failed_stub_passed_does_not_advance() {
    let parent = tempfile::tempdir().expect("tempdir");
    let prod_dir = parent.path().join("prod");
    let ref_dir = parent.path().join("refer");
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    {
        let mut p = Pipeline::create(&prod_dir, module(), cfg.clone(), q.clone()).expect("c");
        let mut r = RefPipeline::create(&ref_dir, module(), cfg.clone(), q).expect("rc");
        p.start_oracle(NOW).unwrap();
        r.start_oracle(NOW).unwrap();
        let _ = p.record_oracle(true, NOW);
        let _ = r.record_oracle(true, NOW);
        assert_eq!(p.stage(), Stage::Oracle);
        assert!(!p.stub_failed());
    }
    let (p, r) = open_both(&prod_dir, &ref_dir, cfg);
    assert_eq!(p.stage(), Stage::Oracle);
    assert!(!p.stub_failed());
    assert_state(&p, &r, "replay stub-passed");
}

#[test]
fn open_then_continue_from_implement() {
    let (_parent, dir) = fresh_dir();
    let cfg = default_pipeline_config();
    let q = default_queue_config();
    {
        let mut p = Pipeline::create(&dir, module(), cfg.clone(), q.clone()).expect("c");
        drive_to_implement(&mut p, NOW);
        p.submit_candidate(cand("0", "a", good_patch()), NOW).unwrap();
    }
    let mut p = Pipeline::open(&dir, module(), cfg, q).expect("open");
    assert_eq!(p.stage(), Stage::Implement);
    p.submit_candidate(cand("1", "b", good_patch()), NOW)
        .expect("submit after open");
    p.start_review(NOW).expect("review after open");
    assert_eq!(p.stage(), Stage::Review);
    assert_eq!(p.candidates().unwrap().len(), 2);
}
