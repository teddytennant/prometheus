//! Group: pure `review` / `merge_allowed` vs the independent reference.

mod common;
mod reference;

use common::{assert_candidate_failed, assert_no_candidate, assert_other, assert_planted_bad, cand};
use prometheus_pipeline::{merge_allowed, review, Candidate, CandidateId, Verdict, PLANTED_BAD_MARKER};
use reference::{ref_is_planted_bad, ref_merge_allowed, ref_review};

fn same_review(cands: &[Candidate], tests_ok: &[bool]) {
    let prod = review(cands, tests_ok);
    let refer = ref_review(cands, tests_ok);
    match (prod, refer) {
        (Ok(Verdict::Pick(a)), Ok(Verdict::Pick(b))) => {
            assert_eq!(a, b, "Pick id vs reference")
        }
        (Ok(Verdict::RejectAll { defects: d1 }), Ok(Verdict::RejectAll { defects: d2 })) => {
            assert!(!d1.is_empty(), "production RejectAll defects must be non-empty");
            assert!(!d2.is_empty(), "reference RejectAll defects must be non-empty");
        }
        (Err(Error::Other(_)), Err(Error::Other(_))) => {}
        (a, b) => panic!("review mismatch prod={a:?} ref={b:?}"),
    }
}

use prometheus_pipeline::Error;

fn same_merge(verdict: &Verdict, cands: &[Candidate], tests_ok: &[bool]) {
    let prod = merge_allowed(verdict, cands, tests_ok);
    let refer = ref_merge_allowed(verdict, cands, tests_ok);
    match (prod, refer) {
        (Ok(()), Ok(())) => {}
        (Err(a), Err(b)) => match (a, b) {
            (Error::PlantedBad, Error::PlantedBad)
            | (Error::CandidateFailed, Error::CandidateFailed)
            | (Error::Other(_), Error::Other(_)) => {}
            (Error::NoCandidate(x), Error::NoCandidate(y)) => assert_eq!(x, y),
            (a, b) => panic!("merge_allowed err mismatch {a:?} vs {b:?}"),
        },
        (a, b) => panic!("merge_allowed mismatch prod={a:?} ref={b:?}"),
    }
}

fn planted() -> Vec<u8> {
    PLANTED_BAD_MARKER.to_vec()
}

#[test]
fn review_picks_first_passing_non_planted() {
    let cands = vec![
        cand("0", "a", b"fail"),
        cand("1", "b", b"ok-one"),
        cand("2", "c", b"ok-two"),
    ];
    let tests = [false, true, true];
    same_review(&cands, &tests);
    match review(&cands, &tests).expect("review") {
        Verdict::Pick(id) => assert_eq!(id.0, "1"),
        other => panic!("expected Pick(1), got {other:?}"),
    }
}

#[test]
fn review_skips_planted_bad_even_if_tests_passed() {
    let cands = vec![
        cand("0", "evil", &planted()),
        cand("1", "good", b"honest"),
    ];
    let tests = [true, true];
    same_review(&cands, &tests);
    match review(&cands, &tests).expect("review") {
        Verdict::Pick(id) => assert_eq!(id.0, "1"),
        other => panic!("expected Pick(1), got {other:?}"),
    }
    assert!(ref_is_planted_bad(&cands[0].patch));
}

#[test]
fn review_never_picks_only_planted_passers() {
    let cands = vec![cand("0", "evil", &planted())];
    let tests = [true];
    same_review(&cands, &tests);
    match review(&cands, &tests).expect("review") {
        Verdict::RejectAll { defects } => assert!(!defects.is_empty()),
        other => panic!("expected RejectAll, got {other:?}"),
    }
}

#[test]
fn review_all_failing_rejects() {
    let cands = vec![cand("0", "a", b"x"), cand("1", "b", b"y")];
    let tests = [false, false];
    same_review(&cands, &tests);
    match review(&cands, &tests).expect("review") {
        Verdict::RejectAll { defects } => assert!(!defects.is_empty()),
        other => panic!("expected RejectAll, got {other:?}"),
    }
}

#[test]
fn review_empty_rejects_with_defects() {
    same_review(&[], &[]);
    match review(&[], &[]).expect("review") {
        Verdict::RejectAll { defects } => assert!(!defects.is_empty()),
        other => panic!("expected RejectAll, got {other:?}"),
    }
}

#[test]
fn review_length_mismatch_is_other() {
    let cands = vec![cand("0", "a", b"x")];
    same_review(&cands, &[]);
    same_review(&cands, &[true, true]);
    assert_other(review(&cands, &[]).expect_err("short"));
    assert_other(review(&[], &[true]).expect_err("long"));
}

#[test]
fn review_order_is_slice_order_not_id_order() {
    let cands = vec![
        cand("9", "z", b"later-id-but-first"),
        cand("0", "a", b"earlier-id"),
    ];
    let tests = [true, true];
    same_review(&cands, &tests);
    match review(&cands, &tests).expect("review") {
        Verdict::Pick(id) => assert_eq!(id.0, "9"),
        other => panic!("expected first slice entry, got {other:?}"),
    }
}

#[test]
fn review_mix_fail_planted_then_pass() {
    let cands = vec![
        cand("0", "f", b"no"),
        cand("1", "p", &planted()),
        cand("2", "ok", b"yes"),
    ];
    let tests = [false, true, true];
    same_review(&cands, &tests);
    match review(&cands, &tests).expect("review") {
        Verdict::Pick(id) => assert_eq!(id.0, "2"),
        other => panic!("expected Pick(2), got {other:?}"),
    }
}

#[test]
fn merge_allowed_ok_only_for_good_pick() {
    let cands = vec![cand("0", "a", b"ok"), cand("1", "b", b"also")];
    let tests = [true, true];
    let v = Verdict::Pick(CandidateId("0".into()));
    same_merge(&v, &cands, &tests);
    merge_allowed(&v, &cands, &tests).expect("allowed");
}

#[test]
fn merge_allowed_planted_pick_is_planted_bad() {
    let cands = vec![cand("0", "evil", &planted())];
    let tests = [true];
    let v = Verdict::Pick(CandidateId("0".into()));
    same_merge(&v, &cands, &tests);
    assert_planted_bad(merge_allowed(&v, &cands, &tests).expect_err("planted"));
}

#[test]
fn merge_allowed_planted_beats_failing() {
    let cands = vec![cand("0", "evil", &planted())];
    let tests = [false];
    let v = Verdict::Pick(CandidateId("0".into()));
    same_merge(&v, &cands, &tests);
    assert_planted_bad(merge_allowed(&v, &cands, &tests).expect_err("planted"));
}

#[test]
fn merge_allowed_failing_pick_is_candidate_failed() {
    let cands = vec![cand("0", "a", b"ok")];
    let tests = [false];
    let v = Verdict::Pick(CandidateId("0".into()));
    same_merge(&v, &cands, &tests);
    assert_candidate_failed(merge_allowed(&v, &cands, &tests).expect_err("fail"));
}

#[test]
fn merge_allowed_reject_all_is_err() {
    let cands = vec![cand("0", "a", b"ok")];
    let tests = [true];
    let v = Verdict::RejectAll {
        defects: vec!["none".into()],
    };
    same_merge(&v, &cands, &tests);
    assert_other(merge_allowed(&v, &cands, &tests).expect_err("reject"));
}

#[test]
fn merge_allowed_unknown_pick_is_no_candidate() {
    let cands = vec![cand("0", "a", b"ok")];
    let tests = [true];
    let v = Verdict::Pick(CandidateId("nope".into()));
    same_merge(&v, &cands, &tests);
    assert_no_candidate(
        merge_allowed(&v, &cands, &tests).expect_err("unknown"),
        "nope",
    );
}

#[test]
fn merge_allowed_length_mismatch_is_other() {
    let cands = vec![cand("0", "a", b"ok")];
    let v = Verdict::Pick(CandidateId("0".into()));
    same_merge(&v, &cands, &[]);
    assert_other(merge_allowed(&v, &cands, &[]).expect_err("len"));
}
