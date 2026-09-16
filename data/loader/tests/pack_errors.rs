//! Group: pack fault injection — empty docs and sequence_length 0.

mod common;

use common::{assert_err_bad_seq_len, assert_err_empty, packing, packing_drop, EOS, PAD};
use prometheus_loader::{pack, Strategy};

fn every_strategy() -> [Strategy; 3] {
    [
        Strategy::ConcatEos,
        Strategy::DocumentMask,
        Strategy::SingleDocument,
    ]
}

#[test]
fn empty_doc_list_is_empty_error_for_every_strategy() {
    for strategy in every_strategy() {
        let packing = packing(strategy, 8, true, PAD);
        assert_err_empty(pack(&[], &packing, EOS));
    }
}

#[test]
fn sequence_length_zero_is_bad_sequence_length_even_with_docs() {
    for strategy in every_strategy() {
        let packing = packing(strategy, 0, true, PAD);
        assert_err_bad_seq_len(pack(&[vec![1, 2, 3]], &packing, EOS));
    }
}

#[test]
fn sequence_length_zero_is_checked_before_empty() {
    for strategy in every_strategy() {
        let packing = packing(strategy, 0, false, PAD);
        assert_err_bad_seq_len(pack(&[], &packing, EOS));
    }
}

#[test]
fn empty_doc_list_with_drop_short_below_is_still_empty() {
    let packing = packing_drop(Strategy::SingleDocument, 8, false, PAD, 4);
    assert_err_empty(pack(&[], &packing, EOS));
}
