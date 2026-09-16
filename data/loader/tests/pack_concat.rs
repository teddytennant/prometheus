//! Group: pack concat_eos — concatenate, optional EOS between docs, fixed
//! length sequences, tail pad, mask 1=token 0=pad, no document_ids.

mod common;
mod reference;

use common::{assert_packed_shape, concat_stream, flatten_real_tokens, packing, EOS, PAD};
use prometheus_loader::{pack, PackedSequence, Strategy};

fn concat_pack(docs: &[Vec<u32>], seq_len: u32, eos_between: bool) -> Vec<PackedSequence> {
    let packing = packing(Strategy::ConcatEos, seq_len, eos_between, PAD);
    pack(docs, &packing, EOS).expect("pack concat_eos")
}

#[test]
fn two_docs_eos_between_pads_tail() {
    let docs = vec![vec![10, 11, 12], vec![20, 21]];
    let got = concat_pack(&docs, 4, true);
    let expected = vec![
        PackedSequence {
            tokens: vec![10, 11, 12, EOS],
            mask: vec![1, 1, 1, 1],
            document_ids: None,
        },
        PackedSequence {
            tokens: vec![20, 21, PAD, PAD],
            mask: vec![1, 1, 0, 0],
            document_ids: None,
        },
    ];
    assert_eq!(got, expected);
    assert_eq!(
        got,
        reference::pack(&docs, &packing(Strategy::ConcatEos, 4, true, PAD), EOS).unwrap()
    );
}

#[test]
fn two_docs_without_eos() {
    let docs = vec![vec![10, 11, 12], vec![20, 21]];
    let got = concat_pack(&docs, 4, false);
    assert_eq!(
        got,
        vec![
            PackedSequence {
                tokens: vec![10, 11, 12, 20],
                mask: vec![1, 1, 1, 1],
                document_ids: None,
            },
            PackedSequence {
                tokens: vec![21, PAD, PAD, PAD],
                mask: vec![1, 0, 0, 0],
                document_ids: None,
            },
        ]
    );
}

#[test]
fn single_doc_no_eos_inserted() {
    let docs = vec![vec![7, 8, 9]];
    let got = concat_pack(&docs, 8, true);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].tokens, vec![7, 8, 9, PAD, PAD, PAD, PAD, PAD]);
    assert_eq!(got[0].mask, vec![1, 1, 1, 0, 0, 0, 0, 0]);
    assert_eq!(got[0].document_ids, None);
}

#[test]
fn long_doc_splits_across_sequences() {
    let docs = vec![vec![1, 2, 3, 4, 5, 6, 7]];
    let got = concat_pack(&docs, 3, true);
    assert_eq!(
        got,
        vec![
            PackedSequence {
                tokens: vec![1, 2, 3],
                mask: vec![1, 1, 1],
                document_ids: None,
            },
            PackedSequence {
                tokens: vec![4, 5, 6],
                mask: vec![1, 1, 1],
                document_ids: None,
            },
            PackedSequence {
                tokens: vec![7, PAD, PAD],
                mask: vec![1, 0, 0],
                document_ids: None,
            },
        ]
    );
}

#[test]
fn exact_multiple_has_no_pad() {
    let docs = vec![vec![1, 2], vec![3, 4]];
    let got = concat_pack(&docs, 5, true);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].tokens, vec![1, 2, EOS, 3, 4]);
    assert_eq!(got[0].mask, vec![1, 1, 1, 1, 1]);
}

#[test]
fn seq_len_one_emits_one_sequence_per_token() {
    let docs = vec![vec![4, 5], vec![6]];
    let got = concat_pack(&docs, 1, true);
    let tokens: Vec<u32> = got.iter().map(|s| s.tokens[0]).collect();
    assert_eq!(tokens, vec![4, 5, EOS, 6]);
    assert!(got.iter().all(|s| s.mask == vec![1] && s.tokens.len() == 1));
}

#[test]
fn empty_docs_in_list_still_insert_eos_between() {
    let docs = vec![vec![1], vec![], vec![2]];
    let got = concat_pack(&docs, 8, true);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].tokens[..4], [1, EOS, EOS, 2]);
    assert_eq!(&got[0].mask[..4], &[1, 1, 1, 1]);
}

#[test]
fn all_empty_docs_without_eos_yield_no_sequences() {
    let docs = vec![vec![], vec![]];
    let got = concat_pack(&docs, 4, false);
    assert!(
        got.is_empty(),
        "empty stream must not invent a padded sequence"
    );
}

#[test]
fn all_empty_docs_with_eos_yield_eos_only_stream() {
    let docs = vec![vec![], vec![]];
    let got = concat_pack(&docs, 4, true);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].tokens, vec![EOS, PAD, PAD, PAD]);
    assert_eq!(got[0].mask, vec![1, 0, 0, 0]);
}

#[test]
fn pad_id_may_equal_a_real_token_mask_distinguishes() {
    let packing = packing(Strategy::ConcatEos, 4, false, 1);
    let docs = vec![vec![1, 2]];
    let got = pack(&docs, &packing, EOS).expect("pack");
    assert_eq!(got[0].tokens, vec![1, 2, 1, 1]);
    assert_eq!(got[0].mask, vec![1, 1, 0, 0]);
}

#[test]
fn drop_short_below_is_ignored_for_concat() {
    let packing = common::packing_drop(Strategy::ConcatEos, 4, false, PAD, 100);
    let docs = vec![vec![1, 2]];
    let got = pack(&docs, &packing, EOS).expect("short docs are not dropped in concat_eos");
    assert_eq!(got.len(), 1);
    assert_eq!(&got[0].tokens[..2], &[1, 2]);
}

#[test]
fn real_tokens_preserve_concatenated_order() {
    let docs = vec![vec![9, 8, 7], vec![1], vec![2, 3, 4, 5]];
    for eos_between in [true, false] {
        let packing = packing(Strategy::ConcatEos, 3, eos_between, PAD);
        let got = pack(&docs, &packing, EOS).expect("pack");
        for seq in &got {
            assert_packed_shape(seq, 3, false, PAD);
        }
        assert_eq!(
            flatten_real_tokens(&got),
            concat_stream(&docs, EOS, eos_between)
        );
        assert_eq!(got, reference::pack(&docs, &packing, EOS).unwrap());
    }
}

#[test]
fn vs_reference_random_docs() {
    let mut state: u64 = 0xC0FFEE;
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }
    for case in 0..12 {
        let n_docs = (next(&mut state) % 6 + 1) as usize;
        let mut docs = Vec::with_capacity(n_docs);
        for _ in 0..n_docs {
            let n = (next(&mut state) % 11) as usize;
            docs.push((0..n).map(|_| (next(&mut state) % 50) as u32).collect());
        }
        let seq_len = (next(&mut state) % 8 + 1) as u32;
        let eos_between = case % 2 == 0;
        let packing = packing(Strategy::ConcatEos, seq_len, eos_between, PAD);
        let got = pack(&docs, &packing, EOS).expect("pack");
        let expected = reference::pack(&docs, &packing, EOS).expect("reference pack");
        assert_eq!(got, expected, "concat_eos case {case} seq_len={seq_len}");
        for seq in &got {
            assert_packed_shape(seq, seq_len as usize, false, PAD);
        }
    }
}
