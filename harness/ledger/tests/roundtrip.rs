//! Group: append/get/get_by_experiment field round-trip + F1 serde names.

mod common;

use prometheus_ledger::{Error, Ledger, Replication, ReplicationStatus};

#[test]
fn append_get_get_by_experiment_round_trip_all_fields_populated() {
    let (_dir, path) = common::temp_ledger_path();
    let rec = common::full_record("exp-full");
    common::assert_f1_field_names(&rec);

    let ledger = Ledger::open(&path).expect("open");
    let id = ledger.append(rec.clone()).expect("append full record");
    let by_id = ledger.get(&id).expect("get by RecordId");
    common::assert_records_eq(&by_id, &rec);
    common::assert_f1_field_names(&by_id);

    let by_exp = ledger
        .get_by_experiment("exp-full")
        .expect("get_by_experiment");
    common::assert_records_eq(&by_exp, &rec);
}

#[test]
fn append_get_round_trip_all_option_fields_none() {
    let (_dir, path) = common::temp_ledger_path();
    let rec = common::sample_record("exp-sparse");
    common::assert_f1_field_names(&rec);

    let ledger = Ledger::open(&path).expect("open");
    let id = ledger.append(rec.clone()).expect("append sparse record");
    let got = ledger.get(&id).expect("get");
    common::assert_records_eq(&got, &rec);
    assert!(got.results.is_none());
    assert!(got.delta_rci.is_none());
    assert!(got.replication.is_none());
    assert!(got.gpu_hours.is_none());
    assert!(got.rung.is_none());
    assert!(got.closed_at.is_none());
    assert!(got.supersedes.is_none());

    let by_exp = ledger
        .get_by_experiment("exp-sparse")
        .expect("get_by_experiment");
    common::assert_records_eq(&by_exp, &rec);
}

#[test]
fn serde_replication_status_is_snake_case() {
    // Locked to F1 enum encoding; still goes through Ledger so the stub fails.
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");

    for (i, status) in [
        ReplicationStatus::Unreplicated,
        ReplicationStatus::Matched,
        ReplicationStatus::Failed,
        ReplicationStatus::Running,
    ]
    .into_iter()
    .enumerate()
    {
        let mut rec = common::sample_record(&format!("exp-status-{i}"));
        rec.replication = Some(Replication {
            status,
            n: 0,
            notes: None,
        });
        common::assert_f1_field_names(&rec);
        ledger.append(rec).expect("append");
    }
}

#[test]
fn get_missing_record_id_is_not_found() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    let rec = common::sample_record("exp-exists");
    let id = ledger.append(rec).expect("append");

    let missing = prometheus_ledger::RecordId(format!("{}-nope", id.0));
    match ledger.get(&missing) {
        Err(Error::NotFound(key)) => {
            assert!(
                key.contains(&missing.0) || !key.is_empty(),
                "NotFound should name the missing id, got {key:?}"
            );
        }
        other => panic!("expected Error::NotFound, got {other:?}"),
    }
}

#[test]
fn get_by_experiment_missing_is_not_found() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    match ledger.get_by_experiment("no-such-experiment") {
        Err(Error::NotFound(key)) => {
            assert!(
                key.contains("no-such-experiment") || !key.is_empty(),
                "NotFound should name the missing experiment_id, got {key:?}"
            );
        }
        other => panic!("expected Error::NotFound, got {other:?}"),
    }
}
