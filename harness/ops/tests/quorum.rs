//! Group: `has_quorum` vs the independent reference.

mod common;
mod reference;

use common::{one_sig_release, release, sig, v1_digest, v1_signed};
use prometheus_ops::{has_quorum, DEFAULT_QUORUM};
use reference::ref_has_quorum;

fn assert_both(rel: &prometheus_ops::Release, quorum: usize, expect: bool) {
    assert_eq!(ref_has_quorum(rel, quorum), expect, "reference has_quorum");
    assert_eq!(has_quorum(rel, quorum), expect, "ops has_quorum");
    assert_eq!(has_quorum(rel, quorum), ref_has_quorum(rel, quorum));
}

#[test]
fn two_distinct_nonempty_meets_default_quorum() {
    assert_both(&v1_signed(), DEFAULT_QUORUM, true);
    assert_both(&v1_signed(), 2, true);
}

#[test]
fn one_signer_is_not_quorum_of_two() {
    assert_both(&one_sig_release("v1", v1_digest()), 2, false);
}

#[test]
fn zero_signers_is_not_quorum_of_two() {
    let rel = release("v1", v1_digest(), vec![]);
    assert_both(&rel, 2, false);
}

#[test]
fn duplicate_humans_count_once() {
    let rel = release(
        "v1",
        v1_digest(),
        vec![sig("alice", b"one"), sig("alice", b"two")],
    );
    assert_both(&rel, 2, false);
    assert_both(&rel, 1, true);
}

#[test]
fn empty_signature_bytes_do_not_count() {
    let rel = release(
        "v1",
        v1_digest(),
        vec![sig("alice", b""), sig("bob", b"ok")],
    );
    assert_both(&rel, 2, false);
    assert_both(&rel, 1, true);
}

#[test]
fn two_empty_signatures_count_zero() {
    let rel = release("v1", v1_digest(), vec![sig("alice", b""), sig("bob", b"")]);
    assert_both(&rel, 1, false);
    assert_both(&rel, 0, true);
}

#[test]
fn mixed_duplicate_and_empty() {
    let rel = release(
        "v1",
        v1_digest(),
        vec![
            sig("alice", b"a"),
            sig("alice", b""),
            sig("bob", b"b"),
            sig("carol", b""),
        ],
    );
    assert_both(&rel, 2, true);
    assert_both(&rel, 3, false);
}

#[test]
fn quorum_zero_is_true_without_signatures() {
    let rel = release("v1", v1_digest(), vec![]);
    assert_both(&rel, 0, true);
}

#[test]
fn quorum_three_needs_three_humans() {
    let two = v1_signed();
    assert_both(&two, 3, false);
    let three = release(
        "v1",
        v1_digest(),
        vec![sig("a", b"1"), sig("b", b"2"), sig("c", b"3")],
    );
    assert_both(&three, 3, true);
}

#[test]
fn extra_signatures_beyond_quorum_still_true() {
    let four = release(
        "v1",
        v1_digest(),
        vec![
            sig("a", b"1"),
            sig("b", b"2"),
            sig("c", b"3"),
            sig("d", b"4"),
        ],
    );
    assert_both(&four, 2, true);
}
