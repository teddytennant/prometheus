//! Group: heal weight copy (CPU analog of spec 5.5 elastic DP).
//!
//! Real IB is S3. This file checks the in-memory copy only: `heal_spare`
//! records the source, `offer_weight_copy` stages owned bytes, `rejoin`
//! installs them. No GPU tests (nothing here needs a device).
//!
//! Each test compares `Controller` to `reference::RefController`. Offer check
//! order, locked here and in the reference:
//! spare exists (`ReplicaNotFound`), spare is `Healing` (`NoHealInProgress`),
//! source liveness (the same error `heal_spare` would return for that source),
//! source identity (`HealSourceMismatch`), then shards in order.
//!
//! A same-id second `heal_spare` after rejoin stays `NotSpare` (existing
//! membership rule, unchanged). The "second heal replaces the install only at
//! rejoin" rule is observed as: offers never write the installed copy; a
//! later spare's copy appears only at that spare's rejoin; a failed second
//! attempt on an already-healed id does not replace its install. The reference
//! keeps staged bytes separate from the install so a future Spare cycle would
//! replace only at rejoin.

mod common;
mod reference;

use common::*;
use prometheus_control::{ControlError, Controller, ReplicaId, ReplicaState, WeightShard};
use reference::RefController;

fn shard(name: &str, bytes: &[u8]) -> WeightShard {
    WeightShard {
        name: name.to_string(),
        bytes: bytes.to_vec(),
    }
}

fn assert_no_heal(err: &ControlError, id: &ReplicaId) {
    match err {
        ControlError::NoHealInProgress(got) => assert_eq!(got, id),
        other => panic!("expected NoHealInProgress({id:?}), got {other:?}"),
    }
}

fn assert_not_installed(err: &ControlError, id: &ReplicaId) {
    match err {
        ControlError::WeightCopyNotInstalled(got) => assert_eq!(got, id),
        other => panic!("expected WeightCopyNotInstalled({id:?}), got {other:?}"),
    }
}

fn assert_mismatch(err: &ControlError, spare: &ReplicaId, expected: &ReplicaId, got: &ReplicaId) {
    match err {
        ControlError::HealSourceMismatch {
            spare: s,
            expected: e,
            got: g,
        } => {
            assert_eq!(s, spare, "HealSourceMismatch.spare");
            assert_eq!(e, expected, "HealSourceMismatch.expected");
            assert_eq!(g, got, "HealSourceMismatch.got");
        }
        other => panic!("expected HealSourceMismatch, got {other:?}"),
    }
}

fn assert_duplicate(err: &ControlError, name: &str) {
    match err {
        ControlError::DuplicateWeightShard(got) => assert_eq!(got, name),
        other => panic!("expected DuplicateWeightShard({name:?}), got {other:?}"),
    }
}

fn assert_empty_name(err: &ControlError) {
    match err {
        ControlError::EmptyWeightShardName => {}
        other => panic!("expected EmptyWeightShardName, got {other:?}"),
    }
}

fn heal(prod: &mut Controller, refer: &mut RefController, spare: &str, source: &str) {
    assert_both_ok(
        prod.heal_spare(&rid(spare), &rid(source)),
        refer.heal_spare(&rid(spare), &rid(source)),
    );
}

fn rejoin_at(prod: &mut Controller, refer: &mut RefController, id: &str, step: u64) {
    assert_both_ok(prod.rejoin(&rid(id), step), refer.rejoin(&rid(id), step));
}

fn offer_ok(
    prod: &mut Controller,
    refer: &mut RefController,
    spare: &str,
    source: &str,
    shards: &[WeightShard],
) {
    refer
        .offer_weight_copy(&rid(spare), &rid(source), shards)
        .expect("ref offer");
    prod.offer_weight_copy(&rid(spare), &rid(source), shards)
        .expect("prod offer");
}

fn both_offer_err(
    prod: &mut Controller,
    refer: &mut RefController,
    spare: &str,
    source: &str,
    shards: &[WeightShard],
) -> ControlError {
    let refer_err = refer
        .offer_weight_copy(&rid(spare), &rid(source), shards)
        .expect_err("ref offer");
    let prod_err = prod
        .offer_weight_copy(&rid(spare), &rid(source), shards)
        .expect_err("prod offer");
    assert_err_eq(&prod_err, &refer_err);
    prod_err
}

fn expect_source(prod: &Controller, refer: &RefController, spare: &str, source: &str) {
    let want = rid(source);
    let refer_src = refer.heal_source(&rid(spare)).expect("ref heal_source");
    assert_eq!(refer_src, want);
    let prod_src = prod.heal_source(&rid(spare)).expect("prod heal_source");
    assert_eq!(prod_src, refer_src);
}

fn expect_no_heal(prod: &Controller, refer: &RefController, id: &str) {
    let id = rid(id);
    let refer_err = refer.heal_source(&id).expect_err("ref heal_source");
    assert_no_heal(&refer_err, &id);
    let prod_err = prod.heal_source(&id).expect_err("prod heal_source");
    assert_err_eq(&prod_err, &refer_err);
    assert_no_heal(&prod_err, &id);
}

fn expect_not_installed(prod: &Controller, refer: &RefController, id: &str) {
    let id = rid(id);
    let refer_err = refer
        .healed_weight_shards(&id)
        .expect_err("ref healed_weight_shards");
    assert_not_installed(&refer_err, &id);
    let prod_err = prod
        .healed_weight_shards(&id)
        .expect_err("prod healed_weight_shards");
    assert_err_eq(&prod_err, &refer_err);
    assert_not_installed(&prod_err, &id);
}

fn expect_shards(prod: &Controller, refer: &RefController, id: &str, expected: &[WeightShard]) {
    let refer_shards = refer
        .healed_weight_shards(&rid(id))
        .expect("ref healed_weight_shards");
    assert_eq!(refer_shards, expected, "reference installed copy");
    let prod_shards = prod
        .healed_weight_shards(&rid(id))
        .expect("prod healed_weight_shards");
    assert_eq!(prod_shards, refer_shards);
    assert_eq!(prod_shards, expected);
}

fn quarantine(prod: &mut Controller, refer: &mut RefController, offender: &str, peers: &[&str]) {
    assert_both_ok(
        prod.report_shard_hash(&rid(offender), "w", "bad", 10),
        refer.report_shard_hash(&rid(offender), "w", "bad", 10),
    );
    for peer in peers {
        assert_both_ok(
            prod.report_shard_hash(&rid(peer), "w", "ok", 10),
            refer.report_shard_hash(&rid(peer), "w", "ok", 10),
        );
    }
    let err = assert_both_err(prod.check_sdc(10), refer.check_sdc(10));
    assert_hash_mismatch(&err, &rid(offender), "w", "ok", "bad");
    assert_eq!(
        prod.replica_state(&rid(offender)).unwrap(),
        ReplicaState::Quarantined
    );
    assert_prod_matches_ref(prod, refer);
}

/// `heal_spare` records the source. `heal_source` returns it only while Healing.
#[test]
fn heal_source_while_healing_and_absent_outside() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));

    let missing = rid("no-such");
    let refer_err = refer.heal_source(&missing).expect_err("ref missing");
    assert_replica_not_found(&refer_err, &missing);
    let prod_err = prod.heal_source(&missing).expect_err("prod missing");
    assert_err_eq(&prod_err, &refer_err);
    assert_replica_not_found(&prod_err, &missing);

    expect_no_heal(&prod, &refer, "spare-0");
    expect_no_heal(&prod, &refer, "live-0");

    heal(&mut prod, &mut refer, "spare-0", "live-0");
    expect_source(&prod, &refer, "spare-0", "live-0");
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Healing
    );

    // A rejected second heal_spare must not retarget the recorded source.
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("live-1")),
        refer.heal_spare(&rid("spare-0"), &rid("live-1")),
    );
    assert_not_spare(&err, &rid("spare-0"));
    expect_source(&prod, &refer, "spare-0", "live-0");

    // Source may die; the record remains until rejoin.
    assert_both_ok(
        prod.fail_replica(&rid("live-0")),
        refer.fail_replica(&rid("live-0")),
    );
    expect_source(&prod, &refer, "spare-0", "live-0");

    rejoin_at(&mut prod, &mut refer, "spare-0", 4);
    expect_no_heal(&prod, &refer, "spare-0");
    expect_no_heal(&prod, &refer, "live-0");
    expect_no_heal(&prod, &refer, "live-1");
    assert_prod_matches_ref(&prod, &refer);
}

/// Mutating the caller's `WeightShard` after offer must not change the install.
#[test]
fn offer_copies_bytes_independent_of_caller() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(1, 1));
    heal(&mut prod, &mut refer, "spare-0", "live-0");

    let mut batch = vec![
        shard("w", &[0, 1, 255, 9]),
        shard("empty-bytes", &[]),
        shard("zeros", &[0, 0, 0]),
    ];
    refer
        .offer_weight_copy(&rid("spare-0"), &rid("live-0"), &batch)
        .expect("ref offer");
    prod.offer_weight_copy(&rid("spare-0"), &rid("live-0"), &batch)
        .expect("prod offer");

    batch[0].bytes[0] = 42;
    batch[0].bytes.push(8);
    batch[0].name = "mutated".into();
    batch[1].bytes = vec![7, 7];
    batch[2].name.clear();
    batch.push(shard("extra", &[3, 3]));

    // A later offer of a new name must still see the original first payload.
    let more = vec![shard("tail", &[4])];
    offer_ok(&mut prod, &mut refer, "spare-0", "live-0", &more);
    rejoin_at(&mut prod, &mut refer, "spare-0", 1);

    expect_shards(
        &prod,
        &refer,
        "spare-0",
        &[
            shard("w", &[0, 1, 255, 9]),
            shard("empty-bytes", &[]),
            shard("zeros", &[0, 0, 0]),
            shard("tail", &[4]),
        ],
    );
}

/// Append order is offer order, then shard order. Empty slice stages nothing.
/// Offers do not install; rejoin does.
#[test]
fn offer_appends_in_offer_then_shard_order() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(1, 1));
    heal(&mut prod, &mut refer, "spare-0", "live-0");
    expect_not_installed(&prod, &refer, "spare-0");

    offer_ok(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("b", &[1]), shard("a", &[2, 2])],
    );
    expect_not_installed(&prod, &refer, "spare-0");

    offer_ok(&mut prod, &mut refer, "spare-0", "live-0", &[]);
    expect_not_installed(&prod, &refer, "spare-0");

    offer_ok(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("c", &[3, 0, 3])],
    );
    // Second offer must not install either.
    expect_not_installed(&prod, &refer, "spare-0");

    rejoin_at(&mut prod, &mut refer, "spare-0", 2);
    expect_shards(
        &prod,
        &refer,
        "spare-0",
        &[
            shard("b", &[1]),
            shard("a", &[2, 2]),
            shard("c", &[3, 0, 3]),
        ],
    );
}

/// Empty name and duplicates fail in shard order. A failed call stages nothing.
#[test]
fn empty_name_and_duplicate_stage_nothing() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(1, 1));
    heal(&mut prod, &mut refer, "spare-0", "live-0");

    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("", &[1]), shard("a", &[1]), shard("a", &[1])],
    );
    assert_empty_name(&err);

    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("a", &[1]), shard("a", &[2]), shard("", &[3])],
    );
    assert_duplicate(&err, "a");

    // Nothing from the failed calls was staged: rejoin would install empty.
    // Stage one good shard, then fail follow-ups that start with a good name.
    offer_ok(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("kept", &[9, 9])],
    );
    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("new", &[1]), shard("kept", &[2])],
    );
    assert_duplicate(&err, "kept");
    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("also-new", &[4]), shard("", &[5])],
    );
    assert_empty_name(&err);
    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("z", &[1]), shard("z", &[1])],
    );
    assert_duplicate(&err, "z");
    // Duplicate of a staged name wins over a later empty name in the same call.
    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("kept", &[3]), shard("", &[5])],
    );
    assert_duplicate(&err, "kept");

    rejoin_at(&mut prod, &mut refer, "spare-0", 1);
    expect_shards(&prod, &refer, "spare-0", &[shard("kept", &[9, 9])]);
}

/// Spare and source errors, including "same error heal_spare would return".
#[test]
fn offer_source_and_spare_errors() {
    let sample = vec![shard("w", &[1]), shard("", &[2])];

    // Unknown spare beats source and shard errors.
    {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(1, 1));
        let err = both_offer_err(&mut prod, &mut refer, "missing", "live-0", &sample);
        assert_replica_not_found(&err, &rid("missing"));
        let err = both_offer_err(&mut prod, &mut refer, "missing", "also-missing", &[]);
        assert_replica_not_found(&err, &rid("missing"));
    }

    // Not Healing beats source and shard errors. Empty slice is not a bypass.
    {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
        assert_both_ok(
            prod.fail_replica(&rid("live-1")),
            refer.fail_replica(&rid("live-1")),
        );
        let err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-1", &sample);
        assert_no_heal(&err, &rid("spare-0"));
        let err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-0", &[]);
        assert_no_heal(&err, &rid("spare-0"));
        let err = both_offer_err(&mut prod, &mut refer, "live-0", "live-0", &sample);
        assert_no_heal(&err, &rid("live-0"));
    }

    // Wrong live source: HealSourceMismatch, and the failed call stages nothing.
    {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 1));
        heal(&mut prod, &mut refer, "spare-0", "live-0");
        let err = both_offer_err(
            &mut prod,
            &mut refer,
            "spare-0",
            "live-1",
            &[shard("nope", &[9])],
        );
        assert_mismatch(&err, &rid("spare-0"), &rid("live-0"), &rid("live-1"));
        let err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-1", &[]);
        assert_mismatch(&err, &rid("spare-0"), &rid("live-0"), &rid("live-1"));
        rejoin_at(&mut prod, &mut refer, "spare-0", 1);
        expect_shards(&prod, &refer, "spare-0", &[]);
    }

    // Non-live source: same error heal_spare would return, even on mismatch,
    // and even if a shard name is also empty. Checked before identity.
    {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(3, 2));
        heal(&mut prod, &mut refer, "spare-0", "live-0");
        assert_both_ok(
            prod.fail_replica(&rid("live-1")),
            refer.fail_replica(&rid("live-1")),
        );

        let heal_err = assert_both_err(
            prod.heal_spare(&rid("spare-1"), &rid("live-1")),
            refer.heal_spare(&rid("spare-1"), &rid("live-1")),
        );
        assert_not_live(&heal_err, &rid("live-1"));
        let offer_err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-1", &sample);
        assert_not_live(&offer_err, &rid("live-1"));

        // Recorded source dies after the heal started.
        assert_both_ok(
            prod.fail_replica(&rid("live-0")),
            refer.fail_replica(&rid("live-0")),
        );
        expect_source(&prod, &refer, "spare-0", "live-0");
        let heal_err = assert_both_err(
            prod.heal_spare(&rid("spare-1"), &rid("live-0")),
            refer.heal_spare(&rid("spare-1"), &rid("live-0")),
        );
        assert_not_live(&heal_err, &rid("live-0"));
        let offer_err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-0", &sample);
        assert_not_live(&offer_err, &rid("live-0"));

        // spare-1 is still Spare (both heal_spare calls failed).
        let heal_err = assert_both_err(
            prod.heal_spare(&rid("spare-1"), &rid("no-such")),
            refer.heal_spare(&rid("spare-1"), &rid("no-such")),
        );
        assert_replica_not_found(&heal_err, &rid("no-such"));
        let offer_err = both_offer_err(&mut prod, &mut refer, "spare-0", "no-such", &[]);
        assert_replica_not_found(&offer_err, &rid("no-such"));

        // A still-live wrong source is a mismatch, not NotLive. Stages nothing.
        let err = both_offer_err(
            &mut prod,
            &mut refer,
            "spare-0",
            "live-2",
            &[shard("skip", &[1])],
        );
        assert_mismatch(&err, &rid("spare-0"), &rid("live-0"), &rid("live-2"));
        rejoin_at(&mut prod, &mut refer, "spare-0", 8);
        expect_shards(&prod, &refer, "spare-0", &[]);
    }

    // Quarantined source, recorded and not.
    {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(3, 2));
        heal(&mut prod, &mut refer, "spare-0", "live-0");
        quarantine(&mut prod, &mut refer, "live-1", &["live-0", "live-2"]);
        let heal_err = assert_both_err(
            prod.heal_spare(&rid("spare-1"), &rid("live-1")),
            refer.heal_spare(&rid("spare-1"), &rid("live-1")),
        );
        assert_quarantined(&heal_err, &rid("live-1"));
        let offer_err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-1", &sample);
        assert_quarantined(&offer_err, &rid("live-1"));
        // Failed offer staged nothing. A later good offer is impossible while
        // the recorded source is still live-0; rejoin installs only what was
        // staged before this error (nothing).
        rejoin_at(&mut prod, &mut refer, "spare-0", 3);
        expect_shards(&prod, &refer, "spare-0", &[]);
        let _ = heal_err;
    }

    // Recorded source quarantined. Three lives, spare on its own rack.
    {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(3, 1));
        heal(&mut prod, &mut refer, "spare-0", "live-0");
        quarantine(&mut prod, &mut refer, "live-0", &["live-1", "live-2"]);
        expect_source(&prod, &refer, "spare-0", "live-0");
        let offer_err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-0", &sample);
        assert_quarantined(&offer_err, &rid("live-0"));
        // Wrong live source is still a mismatch (liveness passed).
        let err = both_offer_err(&mut prod, &mut refer, "spare-0", "live-1", &[]);
        assert_mismatch(&err, &rid("spare-0"), &rid("live-0"), &rid("live-1"));
        rejoin_at(&mut prod, &mut refer, "spare-0", 10);
        expect_shards(&prod, &refer, "spare-0", &[]);
    }

    // Source that is Spare, and source that is Healing: NotLive, same as heal_spare.
    {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 3));
        heal(&mut prod, &mut refer, "spare-0", "live-0");
        heal(&mut prod, &mut refer, "spare-1", "live-1");

        let heal_err = assert_both_err(
            prod.heal_spare(&rid("spare-2"), &rid("spare-2")),
            refer.heal_spare(&rid("spare-2"), &rid("spare-2")),
        );
        assert_not_live(&heal_err, &rid("spare-2"));
        // spare-2 is still Spare. Offering it as the source of spare-0:
        let offer_err = both_offer_err(&mut prod, &mut refer, "spare-0", "spare-2", &sample);
        assert_not_live(&offer_err, &rid("spare-2"));

        let heal_err = assert_both_err(
            prod.heal_spare(&rid("spare-2"), &rid("spare-1")),
            refer.heal_spare(&rid("spare-2"), &rid("spare-1")),
        );
        assert_not_live(&heal_err, &rid("spare-1"));
        let offer_err = both_offer_err(&mut prod, &mut refer, "spare-0", "spare-1", &[]);
        assert_not_live(&offer_err, &rid("spare-1"));
    }
}

/// Offer, success or failure, does not change replica state, n_live, or accum.
#[test]
fn offer_does_not_change_membership() {
    let specs = n_live_m_spare(2, 1);
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(default_config(), specs);
    heal(&mut prod, &mut refer, "spare-0", "live-0");
    let before = membership(&prod, &ids);
    let state = prod.replica_state(&rid("spare-0")).unwrap();
    let n_live = prod.n_live().unwrap();
    let accum = prod.grad_accumulation().unwrap();
    assert_eq!(state, ReplicaState::Healing);

    offer_ok(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("w", &[1, 2])],
    );
    assert_eq!(membership(&prod, &ids), before);
    assert_eq!(prod.replica_state(&rid("spare-0")).unwrap(), state);
    assert_eq!(prod.n_live().unwrap(), n_live);
    assert_eq!(prod.grad_accumulation().unwrap(), accum);
    assert_prod_matches_ref(&prod, &refer);

    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-1",
        &[shard("other", &[9])],
    );
    assert_mismatch(&err, &rid("spare-0"), &rid("live-0"), &rid("live-1"));
    assert_eq!(membership(&prod, &ids), before);
    assert_eq!(prod.n_live().unwrap(), n_live);
    assert_eq!(prod.grad_accumulation().unwrap(), accum);
    assert_prod_matches_ref(&prod, &refer);
    expect_not_installed(&prod, &refer, "spare-0");
}

/// Rejoin installs empty when nothing was staged, and the staged vec otherwise.
/// Before any completed heal the copy is `WeightCopyNotInstalled`, including
/// after offers. A failed rejoin does not install an empty vec.
#[test]
fn rejoin_installs_empty_or_staged_and_not_before() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 2));
    expect_not_installed(&prod, &refer, "spare-0");
    expect_not_installed(&prod, &refer, "spare-1");
    expect_not_installed(&prod, &refer, "live-0");

    let missing = rid("no-such");
    let refer_err = refer
        .healed_weight_shards(&missing)
        .expect_err("ref missing");
    assert_replica_not_found(&refer_err, &missing);
    let prod_err = prod
        .healed_weight_shards(&missing)
        .expect_err("prod missing");
    assert_err_eq(&prod_err, &refer_err);
    assert_replica_not_found(&prod_err, &missing);

    // Failed rejoin must not install an empty vec.
    let err = assert_both_err(
        prod.rejoin(&rid("spare-0"), 1),
        refer.rejoin(&rid("spare-0"), 1),
    );
    assert_not_spare(&err, &rid("spare-0"));
    assert!(refer.recorded_rejoin_step(&rid("spare-0")).is_none());
    expect_not_installed(&prod, &refer, "spare-0");

    // No offer, then rejoin: Ok of an empty vec, not WeightCopyNotInstalled.
    heal(&mut prod, &mut refer, "spare-0", "live-0");
    offer_ok(&mut prod, &mut refer, "spare-0", "live-0", &[]);
    expect_not_installed(&prod, &refer, "spare-0");
    rejoin_at(&mut prod, &mut refer, "spare-0", 0);
    expect_shards(&prod, &refer, "spare-0", &[]);
    assert_eq!(refer.recorded_rejoin_step(&rid("spare-0")), Some(0));
    expect_no_heal(&prod, &refer, "spare-0");

    // Staged shards install at rejoin, not before. Step is not a clock.
    heal(&mut prod, &mut refer, "spare-1", "live-1");
    offer_ok(
        &mut prod,
        &mut refer,
        "spare-1",
        "live-1",
        &[shard("p", &[9]), shard("q", &[8, 7])],
    );
    expect_not_installed(&prod, &refer, "spare-1");
    rejoin_at(&mut prod, &mut refer, "spare-1", u64::MAX);
    expect_shards(
        &prod,
        &refer,
        "spare-1",
        &[shard("p", &[9]), shard("q", &[8, 7])],
    );
    assert_eq!(refer.recorded_rejoin_step(&rid("spare-1")), Some(u64::MAX));
    // The other id's empty install is untouched.
    expect_shards(&prod, &refer, "spare-0", &[]);
    // A live replica that was never a spare still has no install.
    expect_not_installed(&prod, &refer, "live-0");

    // Death after a completed heal does not clear the installed copy.
    assert_both_ok(
        prod.fail_replica(&rid("spare-1")),
        refer.fail_replica(&rid("spare-1")),
    );
    assert_eq!(
        prod.replica_state(&rid("spare-1")).unwrap(),
        ReplicaState::Dead
    );
    expect_shards(
        &prod,
        &refer,
        "spare-1",
        &[shard("p", &[9]), shard("q", &[8, 7])],
    );
    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-1",
        "live-0",
        &[shard("later", &[1])],
    );
    assert_no_heal(&err, &rid("spare-1"));
    expect_shards(
        &prod,
        &refer,
        "spare-1",
        &[shard("p", &[9]), shard("q", &[8, 7])],
    );
}

/// Owned return value. A second heal installs only at its rejoin, not at offer.
#[test]
fn second_heal_replaces_installed_copy_only_at_rejoin() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 2));
    heal(&mut prod, &mut refer, "spare-0", "live-0");
    offer_ok(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("a", &[1, 2])],
    );
    // Second offer of the first heal must not install.
    offer_ok(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("b", &[3])],
    );
    expect_not_installed(&prod, &refer, "spare-0");
    rejoin_at(&mut prod, &mut refer, "spare-0", 5);

    let expected_a = vec![shard("a", &[1, 2]), shard("b", &[3])];
    let refer_held = refer
        .healed_weight_shards(&rid("spare-0"))
        .expect("ref held");
    assert_eq!(refer_held, expected_a);
    let mut held = prod
        .healed_weight_shards(&rid("spare-0"))
        .expect("prod held");
    assert_eq!(held, refer_held);

    // Caller-owned: mutating the returned vec does not change the controller.
    held[0].bytes[0] = 0xff;
    held.push(shard("caller-only", &[9]));
    expect_shards(&prod, &refer, "spare-0", &expected_a);

    // Second heal (other spare). Offer must not install B and must not replace A.
    heal(&mut prod, &mut refer, "spare-1", "live-1");
    let mut batch_b = vec![shard("b-only", &[7, 7, 7])];
    offer_ok(&mut prod, &mut refer, "spare-1", "live-1", &batch_b);
    batch_b[0].bytes[0] = 0;
    expect_not_installed(&prod, &refer, "spare-1");
    expect_shards(&prod, &refer, "spare-0", &expected_a);
    assert_eq!(
        held,
        vec![
            {
                let mut s = shard("a", &[1, 2]);
                s.bytes[0] = 0xff;
                s
            },
            shard("b", &[3]),
            shard("caller-only", &[9]),
        ]
    );

    // Still not replaced at a second offer of the second heal.
    offer_ok(
        &mut prod,
        &mut refer,
        "spare-1",
        "live-1",
        &[shard("b-tail", &[8])],
    );
    expect_not_installed(&prod, &refer, "spare-1");
    expect_shards(&prod, &refer, "spare-0", &expected_a);

    rejoin_at(&mut prod, &mut refer, "spare-1", 6);
    let expected_b = vec![shard("b-only", &[7, 7, 7]), shard("b-tail", &[8])];
    expect_shards(&prod, &refer, "spare-1", &expected_b);
    expect_shards(&prod, &refer, "spare-0", &expected_a);
    assert_ne!(held, expected_a);

    // Same id cannot start another heal (NotSpare). Failed offer/rejoin must
    // not replace the install either.
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("live-1")),
        refer.heal_spare(&rid("spare-0"), &rid("live-1")),
    );
    assert_not_spare(&err, &rid("spare-0"));
    let err = both_offer_err(
        &mut prod,
        &mut refer,
        "spare-0",
        "live-0",
        &[shard("replace", &[1])],
    );
    assert_no_heal(&err, &rid("spare-0"));
    let err = assert_both_err(
        prod.rejoin(&rid("spare-0"), 99),
        refer.rejoin(&rid("spare-0"), 99),
    );
    assert_not_spare(&err, &rid("spare-0"));
    expect_shards(&prod, &refer, "spare-0", &expected_a);
    assert_eq!(refer.recorded_rejoin_step(&rid("spare-0")), Some(5));
}

/// Existing heal_spare / rejoin errors (not a spare, source not live) unchanged.
#[test]
fn existing_heal_errors_still_match_reference() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(3, 2));

    let err = assert_both_err(
        prod.heal_spare(&rid("live-0"), &rid("live-1")),
        refer.heal_spare(&rid("live-0"), &rid("live-1")),
    );
    assert_not_spare(&err, &rid("live-0"));

    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("spare-1")),
        refer.heal_spare(&rid("spare-0"), &rid("spare-1")),
    );
    assert_not_live(&err, &rid("spare-1"));

    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("missing")),
        refer.heal_spare(&rid("spare-0"), &rid("missing")),
    );
    assert_replica_not_found(&err, &rid("missing"));

    assert_both_ok(
        prod.fail_replica(&rid("live-2")),
        refer.fail_replica(&rid("live-2")),
    );
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("live-2")),
        refer.heal_spare(&rid("spare-0"), &rid("live-2")),
    );
    assert_not_live(&err, &rid("live-2"));

    quarantine(&mut prod, &mut refer, "live-1", &["live-0"]);
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("live-1")),
        refer.heal_spare(&rid("spare-0"), &rid("live-1")),
    );
    assert_quarantined(&err, &rid("live-1"));

    // spare-0 is still Spare: every heal_spare above failed.
    assert_eq!(
        prod.replica_state(&rid("spare-0")).unwrap(),
        ReplicaState::Spare
    );
    let err = assert_both_err(
        prod.rejoin(&rid("spare-0"), 1),
        refer.rejoin(&rid("spare-0"), 1),
    );
    assert_not_spare(&err, &rid("spare-0"));

    heal(&mut prod, &mut refer, "spare-0", "live-0");
    let err = assert_both_err(
        prod.heal_spare(&rid("spare-0"), &rid("live-0")),
        refer.heal_spare(&rid("spare-0"), &rid("live-0")),
    );
    assert_not_spare(&err, &rid("spare-0"));
    rejoin_at(&mut prod, &mut refer, "spare-0", 2);
    let err = assert_both_err(
        prod.rejoin(&rid("spare-0"), 3),
        refer.rejoin(&rid("spare-0"), 3),
    );
    assert_not_spare(&err, &rid("spare-0"));
    assert_prod_matches_ref(&prod, &refer);

    // New methods still agree with the reference on these untouched ids.
    expect_no_heal(&prod, &refer, "spare-0");
    expect_no_heal(&prod, &refer, "spare-1");
    expect_shards(&prod, &refer, "spare-0", &[]);
    expect_not_installed(&prod, &refer, "spare-1");
    expect_not_installed(&prod, &refer, "live-0");
}

/// `rejoin` accepts any step, including 0. It records the step; it does not
/// compare it to a clock. Two controllers, two steps, same installed bytes.
#[test]
fn rejoin_step_is_recorded_not_compared_to_a_clock() {
    let shards = vec![shard("w", &[1, 0, 2])];
    for step in [0u64, 1, 7, u64::MAX] {
        let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(1, 1));
        heal(&mut prod, &mut refer, "spare-0", "live-0");
        assert!(refer.recorded_rejoin_step(&rid("spare-0")).is_none());
        offer_ok(&mut prod, &mut refer, "spare-0", "live-0", &shards);
        rejoin_at(&mut prod, &mut refer, "spare-0", step);
        assert_eq!(refer.recorded_rejoin_step(&rid("spare-0")), Some(step));
        expect_shards(&prod, &refer, "spare-0", &shards);
        assert_eq!(prod.n_live().unwrap(), refer.n_live().unwrap());
        assert_prod_matches_ref(&prod, &refer);
    }
}
