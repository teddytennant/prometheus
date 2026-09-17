//! Group: Report field names match F1 — status, against, method_hash (plus hits).

mod common;

use prometheus_decontam::{scan_text, Status};
use serde_json::Value;

#[test]
fn report_json_field_names_are_status_against_method_hash_hits() {
    let index = common::index_with(&[common::planted_item()]);
    let report = scan_text(&index, common::PLANTED_TEXT).unwrap();
    let value = serde_json::to_value(&report).expect("serialize Report");
    let obj = value.as_object().expect("Report is a JSON object");
    for key in ["status", "against", "method_hash", "hits"] {
        assert!(obj.contains_key(key), "Report JSON missing {key}: {obj:?}");
    }
    assert_eq!(obj["status"], "flagged");
    assert!(obj["against"].is_array());
    let hash = obj["method_hash"].as_str().expect("method_hash string");
    common::assert_sha256_hex(hash);
    assert!(obj["hits"].is_array());
    let hit = &obj["hits"][0];
    assert_eq!(hit["suite"], common::PLANTED_SUITE);
    assert_eq!(hit["item_id"], common::PLANTED_ID);
    assert_eq!(hit["method"], "ngram");
}

#[test]
fn clean_report_json_uses_status_clean_and_empty_hits() {
    let index = common::index_with(&[common::planted_item()]);
    let report = scan_text(&index, common::UNRELATED_TEXT).unwrap();
    assert_eq!(report.status, Status::Clean);
    let value = serde_json::to_value(&report).unwrap();
    assert_eq!(value["status"], "clean");
    assert_eq!(value["hits"], Value::Array(vec![]));
    assert!(value["method_hash"].as_str().is_some());
    assert_eq!(value["against"][0], common::PLANTED_SUITE);
}

#[test]
fn report_does_not_use_camel_case_field_names() {
    let index = common::index_with(&[common::planted_item()]);
    let report = scan_text(&index, common::PLANTED_TEXT).unwrap();
    let obj = serde_json::to_value(&report)
        .unwrap()
        .as_object()
        .cloned()
        .unwrap();
    for banned in ["methodHash", "itemId", "Status", "Against"] {
        assert!(
            !obj.contains_key(banned),
            "unexpected camelCase/PascalCase field {banned}"
        );
    }
}
