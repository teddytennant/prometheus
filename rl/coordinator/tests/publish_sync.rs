//! Group 5: publish/sync version protocol
//! (`NotNextVersion`, `Unpublished`, `NotRollout`, `RackNotFound`).

mod common;
mod reference;

use common::*;
use prometheus_coordinator::{CoordError, RackRole};

#[test]
fn first_publish_from_zero_must_be_one() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let err = assert_both_err(prod.publish_weights(0), refer.publish_weights(0));
    match err {
        CoordError::NotNextVersion => {}
        other => panic!("expected NotNextVersion, got {other:?}"),
    }
    assert_eq!(prod.trainer_version().unwrap(), 0);
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    assert_eq!(prod.trainer_version().unwrap(), 1);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn publish_skips_and_repeats_are_not_next_version() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let err = assert_both_err(prod.publish_weights(2), refer.publish_weights(2));
    match err {
        CoordError::NotNextVersion => {}
        other => panic!("expected NotNextVersion, got {other:?}"),
    }
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    let err = assert_both_err(prod.publish_weights(1), refer.publish_weights(1));
    match err {
        CoordError::NotNextVersion => {}
        other => panic!("repeat 1 should be NotNextVersion, got {other:?}"),
    }
    let err = assert_both_err(prod.publish_weights(3), refer.publish_weights(3));
    match err {
        CoordError::NotNextVersion => {}
        other => panic!("skip to 3 should be NotNextVersion, got {other:?}"),
    }
    assert_both_ok(prod.publish_weights(2), refer.publish_weights(2));
    assert_eq!(prod.trainer_version().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn publish_does_not_update_rack_versions() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 0);
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    assert_eq!(prod.trainer_version().unwrap(), 1);
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 0);
    assert_eq!(prod.rack_version(&rid("r01")).unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn version_zero_batches_are_valid_before_any_publish() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(
        prod.enqueue_batch(batch("v0", "p", 0, 2, false)),
        refer.enqueue_batch(batch("v0", "p", 0, 2, false)),
    );
    let got = prod.consume_batch().unwrap().unwrap();
    let got_r = refer.consume_batch().unwrap().unwrap();
    assert_eq!(got, got_r);
    assert_eq!(got.policy_version, 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn sync_requires_current_trainer_version() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let err = assert_both_err(
        prod.sync_rack(&rid("r00"), 1),
        refer.sync_rack(&rid("r00"), 1),
    );
    match err {
        CoordError::Unpublished(v) => assert_eq!(v, 1),
        other => panic!("expected Unpublished(1), got {other:?}"),
    }
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 0);
    // version 0 equals trainer_version 0: allowed, no-op-ish set.
    assert_both_ok(
        prod.sync_rack(&rid("r00"), 0),
        refer.sync_rack(&rid("r00"), 0),
    );
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn sync_after_publish_sets_only_that_rack() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    assert_both_ok(
        prod.sync_rack(&rid("r00"), 1),
        refer.sync_rack(&rid("r00"), 1),
    );
    assert_eq!(prod.rack_version(&rid("r00")).unwrap(), 1);
    assert_eq!(prod.rack_version(&rid("r01")).unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn sync_unpublished_after_publish_still_must_match_trainer() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    let err = assert_both_err(
        prod.sync_rack(&rid("r00"), 0),
        refer.sync_rack(&rid("r00"), 0),
    );
    match err {
        CoordError::Unpublished(v) => assert_eq!(v, 0),
        other => panic!("expected Unpublished(0), got {other:?}"),
    }
    let err = assert_both_err(
        prod.sync_rack(&rid("r00"), 2),
        refer.sync_rack(&rid("r00"), 2),
    );
    match err {
        CoordError::Unpublished(v) => assert_eq!(v, 2),
        other => panic!("expected Unpublished(2), got {other:?}"),
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn sync_unknown_rack_is_not_found() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    let id = rid("ghost");
    let err = assert_both_err(prod.sync_rack(&id, 0), refer.sync_rack(&id, 0));
    match err {
        CoordError::RackNotFound(got) => assert_eq!(got, id),
        other => panic!("expected RackNotFound, got {other:?}"),
    }
}

#[test]
fn sync_trainer_rack_is_not_rollout() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    // r02 is trainer under the default 2+2 split.
    assert_eq!(prod.rack_role(&rid("r02")).unwrap(), RackRole::Trainer);
    let err = assert_both_err(
        prod.sync_rack(&rid("r02"), 0),
        refer.sync_rack(&rid("r02"), 0),
    );
    match err {
        CoordError::NotRollout(id) => assert_eq!(id, rid("r02")),
        other => panic!("expected NotRollout, got {other:?}"),
    }
    match prod.rack_version(&rid("r02")) {
        Err(CoordError::NotRollout(id)) => assert_eq!(id, rid("r02")),
        other => panic!("rack_version on trainer: {other:?}"),
    }
    assert_prod_matches_ref(&prod, &refer, &racks);
}

#[test]
fn rack_not_found_beats_unpublished_on_unknown_id() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    // Unknown rack with a version that would also be Unpublished.
    let err = assert_both_err(
        prod.sync_rack(&rid("ghost"), 7),
        refer.sync_rack(&rid("ghost"), 7),
    );
    match err {
        CoordError::RackNotFound(id) => assert_eq!(id, rid("ghost")),
        other => panic!("expected RackNotFound before Unpublished, got {other:?}"),
    }
}

#[test]
fn sequential_publish_and_sync_all_rollout_racks() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    for v in 1..=3 {
        assert_both_ok(prod.publish_weights(v), refer.publish_weights(v));
        assert_both_ok(
            prod.sync_rack(&rid("r00"), v),
            refer.sync_rack(&rid("r00"), v),
        );
        assert_both_ok(
            prod.sync_rack(&rid("r01"), v),
            refer.sync_rack(&rid("r01"), v),
        );
        assert_eq!(prod.rack_version(&rid("r00")).unwrap(), v);
        assert_eq!(prod.rack_version(&rid("r01")).unwrap(), v);
        assert_prod_matches_ref(&prod, &refer, &racks);
    }
}

#[test]
fn reads_on_unknown_rack_after_publish() {
    let racks = default_racks();
    let (mut prod, mut refer) = pair(default_config(), racks.clone());
    assert_both_ok(prod.publish_weights(1), refer.publish_weights(1));
    let err = assert_both_err(
        prod.rack_version(&rid("nope")),
        refer.rack_version(&rid("nope")),
    );
    match err {
        CoordError::RackNotFound(id) => assert_eq!(id, rid("nope")),
        other => panic!("expected RackNotFound, got {other:?}"),
    }
}
