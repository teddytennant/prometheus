//! Group: exact dedup at Document and Paragraph grain.

mod common;
mod reference;

use common::doc;
use prometheus_dedup::{dedup_exact, Grain};

#[test]
fn document_grain_keeps_lowest_content_hash() {
    let docs = vec![
        doc("later", "Same TEXT", "bb"),
        doc("winner", "same   text", "aa"),
        doc("also", "same text", "cc"),
        doc("unique", "totally different", "99"),
    ];
    let got = dedup_exact(&docs, Grain::Document).expect("exact");
    assert_eq!(got.kept, vec!["unique", "winner"]);
    assert_eq!(got.dropped_exact, vec!["also", "later"]);
    assert!(got.dropped_near.is_empty());
    assert!(got.clusters.is_empty());
    common::assert_partition(&docs, &got);
    common::assert_report_canonical(&got);
    assert_eq!(got, reference::dedup_exact(&docs, Grain::Document).unwrap());
}

#[test]
fn document_grain_tie_breaks_on_id() {
    let docs = vec![doc("z", "dup", "aa"), doc("a", "DUP", "aa")];
    let got = dedup_exact(&docs, Grain::Document).expect("exact");
    assert_eq!(got.kept, vec!["a"]);
    assert_eq!(got.dropped_exact, vec!["z"]);
}

#[test]
fn document_grain_uses_content_hash_field_not_recomputed() {
    // Raw texts differ so a SHA of the raw bytes would reverse the keep order.
    let docs = vec![
        doc("raw-aaa", "hello world", "ff"),
        doc("raw-zzz", "HELLO   WORLD", "00"),
    ];
    let got = dedup_exact(&docs, Grain::Document).expect("exact");
    assert_eq!(got.kept, vec!["raw-zzz"]);
    assert_eq!(got.dropped_exact, vec!["raw-aaa"]);
}

#[test]
fn document_grain_does_not_drop_distinct_normalized_texts() {
    let docs = vec![doc("a", "cats sit", "01"), doc("b", "dogs run", "02")];
    let got = dedup_exact(&docs, Grain::Document).expect("exact");
    assert_eq!(got.kept, vec!["a", "b"]);
    assert!(got.dropped_exact.is_empty());
}

#[test]
fn paragraph_grain_drops_whole_document_if_any_paragraph_matches_a_kept_doc() {
    // Rule (also in tests/reference): walk docs by (content_hash, id). Keep a
    // document iff none of its paragraph hashes appear in a document already
    // kept; otherwise drop the *whole* document. Paragraph hashes are
    // exact_hash of each paragraphs(text) entry.
    let docs = vec![
        doc(
            "keep-src",
            "alpha paragraph lives here\n\nshared paragraph about cats",
            "01",
        ),
        doc(
            "drop-me",
            "shared paragraph about cats\n\nbrand new volcano paragraph",
            "02",
        ),
        doc("unrelated", "entirely different material about zinc", "03"),
    ];
    let got = dedup_exact(&docs, Grain::Paragraph).expect("exact");
    assert_eq!(got.kept, vec!["keep-src", "unrelated"]);
    assert_eq!(got.dropped_exact, vec!["drop-me"]);
    assert!(got.dropped_near.is_empty());
    assert!(got.clusters.is_empty());
    assert_eq!(
        got,
        reference::dedup_exact(&docs, Grain::Paragraph).unwrap()
    );
}

#[test]
fn paragraph_grain_lowest_hash_wins_even_if_it_is_the_shorter_doc() {
    let docs = vec![
        doc(
            "long",
            "shared paragraph about cats\n\nextra unique long tail",
            "10",
        ),
        doc("short", "shared paragraph about cats", "01"),
    ];
    let got = dedup_exact(&docs, Grain::Paragraph).expect("exact");
    assert_eq!(got.kept, vec!["short"]);
    assert_eq!(got.dropped_exact, vec!["long"]);
}

#[test]
fn paragraph_grain_keeps_docs_that_only_share_document_level_normalize() {
    // Same words, different paragraph breaks → different paragraph hashes.
    let docs = vec![
        doc("one-block", "hello world", "01"),
        doc("two-block", "hello\n\nworld", "02"),
    ];
    let para = dedup_exact(&docs, Grain::Paragraph).expect("para");
    assert_eq!(para.kept, vec!["one-block", "two-block"]);
    assert!(para.dropped_exact.is_empty());
    let docg = dedup_exact(&docs, Grain::Document).expect("doc");
    assert_eq!(docg.kept, vec!["one-block"]);
    assert_eq!(docg.dropped_exact, vec!["two-block"]);
}

#[test]
fn paragraph_grain_identical_docs_drop_the_higher_hash() {
    let docs = vec![doc("b", "same para", "02"), doc("a", "same para", "01")];
    let got = dedup_exact(&docs, Grain::Paragraph).expect("exact");
    assert_eq!(got.kept, vec!["a"]);
    assert_eq!(got.dropped_exact, vec!["b"]);
}

#[test]
fn input_order_does_not_change_who_is_kept() {
    let a = vec![
        doc("z", "dup text here", "02"),
        doc("a", "DUP   text here", "01"),
    ];
    let b = vec![a[1].clone(), a[0].clone()];
    let ra = dedup_exact(&a, Grain::Document).unwrap();
    let rb = dedup_exact(&b, Grain::Document).unwrap();
    assert_eq!(ra, rb);
    assert_eq!(ra.kept, vec!["a"]);
}

#[test]
fn matches_reference_on_fixture_both_grains() {
    let docs = common::fixture_corpus();
    for grain in [Grain::Document, Grain::Paragraph] {
        let got = dedup_exact(&docs, grain).expect("exact");
        let exp = reference::dedup_exact(&docs, grain).expect("ref");
        assert_eq!(got, exp, "mismatch at {grain:?}");
        common::assert_partition(&docs, &got);
        common::assert_report_canonical(&got);
    }
}
