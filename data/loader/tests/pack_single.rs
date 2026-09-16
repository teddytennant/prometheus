//! Group: pack single_document — one doc per sequence, pad or truncate,
//! drop docs shorter than drop_short_below.

mod common;
mod reference;

use common::{assert_err_dropped_short, assert_packed_shape, packing, packing_drop, EOS, PAD};
use prometheus_loader::{pack, PackedSequence, Strategy};

#[test]
fn pads_short_doc_and_truncates_long_doc() {
    let packing = packing(Strategy::SingleDocument, 4, false, PAD);
    let docs = vec![vec![1, 2], vec![4, 5, 6, 7, 8]];
    let got = pack(&docs, &packing, EOS).expect("pack single_document");
    assert_eq!(
        got,
        vec![
            PackedSequence {
                tokens: vec![1, 2, PAD, PAD],
                mask: vec![1, 1, 0, 0],
                document_ids: None,
            },
            PackedSequence {
                tokens: vec![4, 5, 6, 7],
                mask: vec![1, 1, 1, 1],
                document_ids: None,
            },
        ]
    );
}

#[test]
fn drops_docs_shorter_than_threshold_keeps_equal() {
    let packing = packing_drop(Strategy::SingleDocument, 4, false, PAD, 2);
    let docs = vec![vec![1, 2], vec![3], vec![4, 5, 6, 7, 8]];
    let got = pack(&docs, &packing, EOS).expect("pack");
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].tokens, vec![1, 2, PAD, PAD]);
    assert_eq!(got[1].tokens, vec![4, 5, 6, 7]);
    assert_eq!(got, reference::pack(&docs, &packing, EOS).unwrap());
}

#[test]
fn drop_short_below_none_keeps_empty_doc_as_all_pad() {
    let packing = packing(Strategy::SingleDocument, 3, false, PAD);
    let docs = vec![vec![]];
    let got = pack(&docs, &packing, EOS).expect("empty doc kept when no threshold");
    assert_eq!(
        got,
        vec![PackedSequence {
            tokens: vec![PAD, PAD, PAD],
            mask: vec![0, 0, 0],
            document_ids: None,
        }]
    );
}

#[test]
fn drop_short_below_zero_keeps_empty_doc() {
    let packing = packing_drop(Strategy::SingleDocument, 2, false, PAD, 0);
    let docs = vec![vec![]];
    let got = pack(&docs, &packing, EOS).expect("len 0 is not < 0");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].mask, vec![0, 0]);
}

#[test]
fn all_docs_dropped_is_dropped_short() {
    let packing = packing_drop(Strategy::SingleDocument, 8, false, PAD, 10);
    let docs = vec![vec![1, 2], vec![3]];
    assert_err_dropped_short(pack(&docs, &packing, EOS));
}

#[test]
fn mixed_keep_and_drop_preserves_order_of_kept() {
    let packing = packing_drop(Strategy::SingleDocument, 3, false, PAD, 2);
    let docs = vec![vec![1], vec![2, 3], vec![4], vec![5, 6, 7]];
    let got = pack(&docs, &packing, EOS).expect("pack");
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].tokens, vec![2, 3, PAD]);
    assert_eq!(got[1].tokens, vec![5, 6, 7]);
}

#[test]
fn eos_id_and_eos_between_docs_are_ignored() {
    let a = packing(Strategy::SingleDocument, 4, true, PAD);
    let b = packing(Strategy::SingleDocument, 4, false, PAD);
    let docs = vec![vec![9, 8, 7]];
    let left = pack(&docs, &a, 111).expect("pack a");
    let right = pack(&docs, &b, 222).expect("pack b");
    assert_eq!(left, right);
    assert!(!left[0].tokens.contains(&111));
    assert!(!left[0].tokens.contains(&222));
    assert_eq!(left[0].document_ids, None);
}

#[test]
fn exact_length_doc_is_kept_without_pad() {
    let packing = packing_drop(Strategy::SingleDocument, 3, false, PAD, 3);
    let docs = vec![vec![1, 2, 3]];
    let got = pack(&docs, &packing, EOS).expect("equal to threshold is kept");
    assert_eq!(got[0].tokens, vec![1, 2, 3]);
    assert_eq!(got[0].mask, vec![1, 1, 1]);
}

#[test]
fn vs_reference_and_shapes() {
    let packing = packing_drop(Strategy::SingleDocument, 5, true, PAD, 1);
    let docs = vec![vec![1], vec![], vec![2, 3, 4, 5, 6, 7], vec![8, 9]];
    let got = pack(&docs, &packing, EOS).expect("pack");
    assert_eq!(got, reference::pack(&docs, &packing, EOS).unwrap());
    assert_eq!(got.len(), 3, "empty doc dropped by threshold 1");
    for seq in &got {
        assert_packed_shape(seq, 5, false, PAD);
    }
}
