//! Group: open + persistence across process-local reopen.

mod common;

use prometheus_ledger::Ledger;

#[test]
fn open_on_tempfile_creates_store_and_second_open_sees_appended_rows() {
    let (_dir, path) = common::temp_ledger_path();
    let rec = common::sample_record("exp-persist-1");

    {
        let ledger = Ledger::open(&path).expect("open creates a store");
        assert!(
            ledger.is_empty().expect("is_empty on a fresh store"),
            "fresh store must be empty"
        );
        ledger.append(rec.clone()).expect("append");
        assert_eq!(ledger.len().expect("len after append"), 1);
    }

    {
        let ledger = Ledger::open(&path).expect("second open of the same path");
        assert_eq!(
            ledger.len().expect("len after reopen"),
            1,
            "second open must see appended rows"
        );
        assert!(!ledger.is_empty().expect("is_empty after reopen"));
        let got = ledger
            .get_by_experiment("exp-persist-1")
            .expect("row survived reopen");
        common::assert_records_eq(&got, &rec);
    }
}
