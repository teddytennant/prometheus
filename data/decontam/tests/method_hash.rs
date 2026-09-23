//! Group: method_hash — stable SHA-256 hex; changes with contents or NGRAM_N.

mod common;
mod reference;

use prometheus_decontam::{method_hash, Index, NGRAM_N};

#[test]
fn method_hash_empty_index_is_missing_index() {
    common::assert_missing_index(method_hash(&Index::new()));
}

#[test]
fn method_hash_is_stable_lowercase_sha256_hex() {
    let items = vec![common::unrelated_item(), common::planted_item()];
    let index = common::index_with(&items);
    let got = method_hash(&index).expect("method_hash");
    common::assert_sha256_hex(&got);
    assert_eq!(got, method_hash(&index).expect("second call"));
    assert_eq!(got, reference::method_hash(&items).expect("reference"));
}

#[test]
fn method_hash_is_independent_of_insertion_order() {
    let a = common::planted_item();
    let b = common::unrelated_item();
    let index_ab = common::index_with(&[a.clone(), b.clone()]);
    let index_ba = common::index_with(&[b.clone(), a.clone()]);
    assert_eq!(
        method_hash(&index_ab).unwrap(),
        method_hash(&index_ba).unwrap()
    );
    assert_eq!(
        method_hash(&index_ab).unwrap(),
        reference::method_hash(&[a, b]).unwrap()
    );
}

#[test]
fn method_hash_changes_when_index_contents_change() {
    let planted = common::planted_item();
    let unrelated = common::unrelated_item();
    let one = common::index_with(std::slice::from_ref(&planted));
    let two = common::index_with(&[planted.clone(), unrelated.clone()]);
    let h1 = method_hash(&one).unwrap();
    let h2 = method_hash(&two).unwrap();
    assert_ne!(h1, h2);
    assert_eq!(
        h1,
        reference::method_hash(std::slice::from_ref(&planted)).unwrap()
    );
    assert_eq!(h2, reference::method_hash(&[planted, unrelated]).unwrap());
}

#[test]
fn method_hash_changes_if_ngram_n_would_change() {
    assert_eq!(NGRAM_N, 8);
    let items = vec![common::planted_item()];
    let h8 = reference::method_hash_n(&items, 8).unwrap();
    let h7 = reference::method_hash_n(&items, 7).unwrap();
    assert_ne!(
        h8, h7,
        "transcript must include n-gram width so NGRAM_N changes the hash"
    );
    let index = common::index_with(&items);
    assert_eq!(
        method_hash(&index).unwrap(),
        h8,
        "production method_hash must match the NGRAM_N=8 transcript"
    );
}

#[test]
fn method_hash_changes_when_item_text_changes() {
    let a = common::item("gpqa", "q1", &common::words(8));
    let b = common::item("gpqa", "q1", &common::words(9));
    let ha = method_hash(&common::index_with(std::slice::from_ref(&a))).unwrap();
    let hb = method_hash(&common::index_with(std::slice::from_ref(&b))).unwrap();
    assert_ne!(ha, hb);
    assert_eq!(ha, reference::method_hash(&[a]).unwrap());
    assert_eq!(hb, reference::method_hash(&[b]).unwrap());
}
