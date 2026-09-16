//! Group: corrections are a new row; bodies are never updated.

mod common;

use prometheus_ledger::{Error, Ledger};

#[test]
fn append_with_supersedes_existing_id_inserts_new_row_original_unchanged() {
    let (_dir, path) = common::temp_ledger_path();
    let original = common::full_record("exp-v1");
    let ledger = Ledger::open(&path).expect("open");
    let orig_id = ledger.append(original.clone()).expect("append original");

    let mut correction = common::sample_record("exp-v2");
    correction.hypothesis = "corrected hypothesis".to_string();
    correction.supersedes = Some("exp-v1".to_string());
    correction.author_role = "program_lead".to_string();

    let new_id = ledger
        .append(correction.clone())
        .expect("append correction with supersedes");
    assert_ne!(new_id, orig_id, "correction must be a distinct RecordId");
    assert_ne!(
        correction.experiment_id, original.experiment_id,
        "correction must use a NEW experiment_id"
    );

    assert_eq!(ledger.len().expect("len includes superseded"), 2);

    let still = ledger
        .get_by_experiment("exp-v1")
        .expect("original row still addressable");
    common::assert_records_eq(&still, &original);
    assert!(still.supersedes.is_none());

    let got_new = ledger.get_by_experiment("exp-v2").expect("correction row");
    common::assert_records_eq(&got_new, &correction);
    assert_eq!(got_new.supersedes.as_deref(), Some("exp-v1"));

    let by_new_id = ledger.get(&new_id).expect("get correction by RecordId");
    common::assert_records_eq(&by_new_id, &correction);
}

#[test]
fn append_with_supersedes_missing_id_is_not_found() {
    // Convention: missing supersedes target → Error::NotFound (not Other).
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    ledger
        .append(common::sample_record("exp-unrelated"))
        .expect("seed");

    let mut rec = common::sample_record("exp-orphan");
    rec.supersedes = Some("exp-does-not-exist".to_string());
    match ledger.append(rec) {
        Err(Error::NotFound(key)) => {
            assert!(
                key.contains("exp-does-not-exist"),
                "NotFound should name the missing supersedes target, got {key:?}"
            );
        }
        other => panic!("expected Error::NotFound for missing supersedes, got {other:?}"),
    }
    assert_eq!(
        ledger.len().expect("len"),
        1,
        "failed append must not insert"
    );
}
