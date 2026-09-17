//! Group: Status serde — pending, clean, flagged, not_applicable (F1 names).

mod common;
mod reference;

use prometheus_decontam::{ngrams, Status};

#[test]
fn status_serde_f1_snake_case_names() {
    // Touch unimplemented surface so a stub run cannot pass this test.
    let _ = ngrams("force stub failure on unimplemented ngrams");

    let cases = [
        (Status::Pending, "pending"),
        (Status::Clean, "clean"),
        (Status::Flagged, "flagged"),
        (Status::NotApplicable, "not_applicable"),
    ];
    for (status, name) in cases {
        let json = serde_json::to_string(&status).expect("serialize Status");
        assert_eq!(json, format!("\"{name}\""));
        let back: Status = serde_json::from_str(&json).expect("deserialize Status");
        assert_eq!(back, status);
    }
}

#[test]
fn status_rejects_unknown_variant() {
    let _ = ngrams("force stub failure");
    assert!(serde_json::from_str::<Status>("\"Clean\"").is_err());
    assert!(serde_json::from_str::<Status>("\"unknown\"").is_err());
}

#[test]
fn reference_scan_status_strings_are_clean_and_flagged() {
    let items = vec![common::planted_item()];
    let clean = reference::scan_text(&items, common::UNRELATED_TEXT).unwrap();
    assert_eq!(clean.status, Status::Clean);
    let flagged = reference::scan_text(&items, common::PLANTED_TEXT).unwrap();
    assert_eq!(flagged.status, Status::Flagged);
    // Production must match the reference once implemented.
    let index = common::index_with(&items);
    let got_clean = prometheus_decontam::scan_text(&index, common::UNRELATED_TEXT).unwrap();
    let got_flagged = prometheus_decontam::scan_text(&index, common::PLANTED_TEXT).unwrap();
    assert_eq!(got_clean.status, Status::Clean);
    assert_eq!(got_flagged.status, Status::Flagged);
}
