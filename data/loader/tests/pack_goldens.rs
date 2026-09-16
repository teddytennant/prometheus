//! Group: pack goldens — hand-computed JSON fixtures vs pack and the reference.

mod common;
mod reference;

use common::GoldenPack;
use prometheus_loader::pack;

fn load(raw: &str) -> GoldenPack {
    serde_json::from_str(raw).expect("golden JSON must deserialize")
}

fn check(raw: &str) {
    let golden = load(raw);
    let got = pack(&golden.docs, &golden.packing, golden.eos_id).expect("pack golden");
    let expected: Vec<_> = golden
        .expected
        .into_iter()
        .map(|s| s.into_packed())
        .collect();
    assert_eq!(got, expected, "public pack must match the hand golden");
    let reference = reference::pack(&golden.docs, &golden.packing, golden.eos_id)
        .expect("reference pack golden");
    assert_eq!(got, reference, "public pack must match the slow reference");
}

#[test]
fn golden_concat_eos() {
    check(include_str!("goldens/concat_eos.json"));
}

#[test]
fn golden_concat_no_eos() {
    check(include_str!("goldens/concat_no_eos.json"));
}

#[test]
fn golden_concat_split() {
    check(include_str!("goldens/concat_split.json"));
}

#[test]
fn golden_document_mask() {
    check(include_str!("goldens/document_mask.json"));
}

#[test]
fn golden_document_mask_pad() {
    check(include_str!("goldens/document_mask_pad.json"));
}

#[test]
fn golden_single_document() {
    check(include_str!("goldens/single_document.json"));
}
