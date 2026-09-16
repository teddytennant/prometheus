//! Group: len / is_empty. len counts every row, including superseded.

mod common;

use prometheus_ledger::Ledger;

#[test]
fn len_and_is_empty() {
    let (_dir, path) = common::temp_ledger_path();
    let ledger = Ledger::open(&path).expect("open");
    assert!(ledger.is_empty().expect("is_empty"));
    assert_eq!(ledger.len().expect("len"), 0);

    ledger.append(common::sample_record("exp-1")).unwrap();
    assert!(!ledger.is_empty().expect("is_empty after append"));
    assert_eq!(ledger.len().expect("len"), 1);

    ledger.append(common::sample_record("exp-2")).unwrap();
    assert_eq!(ledger.len().expect("len"), 2);

    let mut correction = common::sample_record("exp-1-fix");
    correction.supersedes = Some("exp-1".to_string());
    ledger.append(correction).unwrap();
    assert_eq!(
        ledger.len().expect("len includes superseded"),
        3,
        "superseded originals remain in the append-only log"
    );
    assert!(!ledger.is_empty().expect("still not empty"));
}
