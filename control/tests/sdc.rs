//! Group: SDC `report_shard_hash` / `check_sdc`.

mod common;
mod reference;

use common::*;
use prometheus_control::ReplicaState;

fn report_all(
    prod: &mut prometheus_control::Controller,
    refer: &mut reference::RefController,
    ids: &[&str],
    shard: &str,
    hex: &str,
    step: u64,
) {
    for id in ids {
        assert_both_ok(
            prod.report_shard_hash(&rid(id), shard, hex, step),
            refer.report_shard_hash(&rid(id), shard, hex, step),
        );
    }
}

#[test]
fn matching_hashes_at_period_multiple_ok() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    report_all(
        &mut prod,
        &mut refer,
        &["live-0", "live-1"],
        "w0",
        "abc",
        10,
    );
    assert_both_ok(prod.check_sdc(10), refer.check_sdc(10));
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn different_hex_quarantines_offender_rack_and_drops() {
    let cfg = default_config();
    let specs = n_live_m_spare(2, 0);
    let (mut prod, mut refer) = pair(cfg.clone(), specs);
    assert_both_ok(
        prod.report_shard_hash(&rid("live-0"), "w0", "aaa", 10),
        refer.report_shard_hash(&rid("live-0"), "w0", "aaa", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("live-1"), "w0", "bbb", 10),
        refer.report_shard_hash(&rid("live-1"), "w0", "bbb", 10),
    );
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_hash_mismatch(&err, &rid("live-1"), "w0", "aaa", "bbb");
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Quarantined
    );
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Live
    );
    assert_eq!(prod.n_live().unwrap(), 1);
    assert_eq!(prod.grad_accumulation().unwrap(), 8);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn mismatch_quarantines_every_replica_on_that_rack() {
    let cfg = default_config();
    let specs = vec![
        spec("a", "rack-a", false),
        spec("b", "rack-b", false),
        spec("c", "rack-b", false),
        spec("s", "rack-b", true),
    ];
    let (mut prod, mut refer) = pair(cfg.clone(), specs);
    // Majority "ok" from a+c vs b "bad"? Wait n=3, need count>1.
    // a=ok, b=bad, c=ok => majority ok, offender b, rack-b includes b,c,s.
    assert_both_ok(
        prod.report_shard_hash(&rid("a"), "w", "ok", 10),
        refer.report_shard_hash(&rid("a"), "w", "ok", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("b"), "w", "bad", 10),
        refer.report_shard_hash(&rid("b"), "w", "bad", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("c"), "w", "ok", 10),
        refer.report_shard_hash(&rid("c"), "w", "ok", 10),
    );
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_hash_mismatch(&err, &rid("b"), "w", "ok", "bad");
    assert_eq!(
        prod.replica_state(&rid("b")).unwrap(),
        ReplicaState::Quarantined
    );
    assert_eq!(
        prod.replica_state(&rid("c")).unwrap(),
        ReplicaState::Quarantined
    );
    assert_eq!(
        prod.replica_state(&rid("s")).unwrap(),
        ReplicaState::Quarantined
    );
    assert_eq!(prod.replica_state(&rid("a")).unwrap(), ReplicaState::Live);
    assert_eq!(prod.n_live().unwrap(), 1);
    assert_ctrl_tokens(&prod, &cfg);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn missing_hash_from_live_replica_at_check_step() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    assert_both_ok(
        prod.report_shard_hash(&rid("live-0"), "w0", "aaa", 10),
        refer.report_shard_hash(&rid("live-0"), "w0", "aaa", 10),
    );
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_missing_hash(&err, &rid("live-1"), "w0");
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn no_hashes_at_check_step_missing_hash_first_live_empty_shard() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_missing_hash(&err, &rid("live-0"), "");
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn check_sdc_non_multiple_is_ok_noop() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    // No hashes, but step 5 is not a multiple of 10.
    assert_both_ok(prod.check_sdc(5), refer.check_sdc(5));
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn sdc_period_zero_never_checks() {
    let mut cfg = default_config();
    cfg.sdc_period_steps = 0;
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    assert_both_ok(
        prod.report_shard_hash(&rid("live-0"), "w0", "aaa", 10),
        refer.report_shard_hash(&rid("live-0"), "w0", "aaa", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("live-1"), "w0", "bbb", 10),
        refer.report_shard_hash(&rid("live-1"), "w0", "bbb", 10),
    );
    assert_both_ok(prod.check_sdc(10), refer.check_sdc(10));
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Live
    );
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Live
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn mismatch_that_would_drop_last_live_does_not_change_state() {
    let cfg = default_config();
    // Two live on the same rack: quarantining the rack would leave 0 live.
    let specs = vec![spec("a", "r", false), spec("b", "r", false)];
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(cfg, specs);
    assert_both_ok(
        prod.report_shard_hash(&rid("a"), "w", "aaa", 10),
        refer.report_shard_hash(&rid("a"), "w", "aaa", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("b"), "w", "bbb", 10),
        refer.report_shard_hash(&rid("b"), "w", "bbb", 10),
    );
    let before = membership(&prod, &ids);
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_hash_mismatch(&err, &rid("b"), "w", "aaa", "bbb");
    assert_eq!(membership(&prod, &ids), before);
    assert_eq!(prod.replica_state(&rid("a")).unwrap(), ReplicaState::Live);
    assert_eq!(prod.replica_state(&rid("b")).unwrap(), ReplicaState::Live);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn hashes_at_wrong_step_do_not_count() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    report_all(&mut prod, &mut refer, &["live-0", "live-1"], "w0", "abc", 9);
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_missing_hash(&err, &rid("live-0"), "");
}

#[test]
fn hex_compare_is_case_sensitive() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    assert_both_ok(
        prod.report_shard_hash(&rid("live-0"), "w", "aa", 10),
        refer.report_shard_hash(&rid("live-0"), "w", "aa", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("live-1"), "w", "AA", 10),
        refer.report_shard_hash(&rid("live-1"), "w", "AA", 10),
    );
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_hash_mismatch(&err, &rid("live-1"), "w", "aa", "AA");
    assert_eq!(
        prod.replica_state(&rid("live-1")).unwrap(),
        ReplicaState::Quarantined
    );
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn report_from_spare_is_not_live() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
    let err = assert_both_err(
        prod.report_shard_hash(&rid("spare-0"), "w", "x", 10),
        refer.report_shard_hash(&rid("spare-0"), "w", "x", 10),
    );
    assert_not_live(&err, &rid("spare-0"));
}

#[test]
fn report_unknown_is_not_found() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let id = rid("nope");
    let err = assert_both_err(
        prod.report_shard_hash(&id, "w", "x", 10),
        refer.report_shard_hash(&id, "w", "x", 10),
    );
    assert_replica_not_found(&err, &id);
}

#[test]
fn op_on_quarantined_is_quarantined_error() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    assert_both_ok(
        prod.report_shard_hash(&rid("live-0"), "w", "aaa", 10),
        refer.report_shard_hash(&rid("live-0"), "w", "aaa", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("live-1"), "w", "bbb", 10),
        refer.report_shard_hash(&rid("live-1"), "w", "bbb", 10),
    );
    let _ = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    let err = assert_both_err(
        prod.fail_replica(&rid("live-1")),
        refer.fail_replica(&rid("live-1")),
    );
    assert_quarantined(&err, &rid("live-1"));
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn majority_picks_minority_as_offender_even_if_first() {
    let cfg = default_config();
    let specs = n_live_m_spare(3, 0);
    let (mut prod, mut refer) = pair(cfg, specs);
    assert_both_ok(
        prod.report_shard_hash(&rid("live-0"), "w", "x", 10),
        refer.report_shard_hash(&rid("live-0"), "w", "x", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("live-1"), "w", "y", 10),
        refer.report_shard_hash(&rid("live-1"), "w", "y", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("live-2"), "w", "y", 10),
        refer.report_shard_hash(&rid("live-2"), "w", "y", 10),
    );
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_hash_mismatch(&err, &rid("live-0"), "w", "y", "x");
    assert_eq!(
        prod.replica_state(&rid("live-0")).unwrap(),
        ReplicaState::Quarantined
    );
    assert_eq!(prod.n_live().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn union_of_shard_names_requires_every_live_replica() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    assert_both_ok(
        prod.report_shard_hash(&rid("live-0"), "w", "aaa", 10),
        refer.report_shard_hash(&rid("live-0"), "w", "aaa", 10),
    );
    assert_both_ok(
        prod.report_shard_hash(&rid("live-1"), "v", "aaa", 10),
        refer.report_shard_hash(&rid("live-1"), "v", "aaa", 10),
    );
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    // lex first shard is "v"; live-0 is missing it.
    assert_missing_hash(&err, &rid("live-0"), "v");
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn step_zero_is_a_check_when_period_positive() {
    let mut cfg = default_config();
    cfg.sdc_period_steps = 10;
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    let err = assert_both_err(prod.check_sdc(0), refer.check_sdc(0));
    assert_missing_hash(&err, &rid("live-0"), "");
}
