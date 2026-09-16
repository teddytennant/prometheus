//! Group: pack document_mask — concat_eos packing plus per-position document ids.

mod common;
mod reference;

use common::{assert_packed_shape, packing, EOS, PAD};
use prometheus_loader::{pack, PackedSequence, Strategy};

fn mask_pack(docs: &[Vec<u32>], seq_len: u32, eos_between: bool) -> Vec<PackedSequence> {
    let packing = packing(Strategy::DocumentMask, seq_len, eos_between, PAD);
    pack(docs, &packing, EOS).expect("pack document_mask")
}

#[test]
fn tokens_match_concat_eos_and_ids_track_documents() {
    let docs = vec![vec![10, 11], vec![20]];
    let got = mask_pack(&docs, 4, true);
    assert_eq!(
        got,
        vec![PackedSequence {
            tokens: vec![10, 11, EOS, 20],
            mask: vec![1, 1, 1, 1],
            document_ids: Some(vec![0, 0, 0, 1]),
        }]
    );
}

#[test]
fn eos_belongs_to_the_document_it_closes() {
    let docs = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
    let got = mask_pack(&docs, 4, true);
    // stream: 1 2 3 EOS  4 5 EOS 6
    // ids:    0 0 0  0   1 1  1  2
    assert_eq!(got[0].tokens, vec![1, 2, 3, EOS]);
    assert_eq!(got[0].document_ids.as_deref(), Some(&[0, 0, 0, 0][..]));
    assert_eq!(got[1].tokens, vec![4, 5, EOS, 6]);
    assert_eq!(got[1].document_ids.as_deref(), Some(&[1, 1, 1, 2][..]));
}

#[test]
fn pad_positions_use_document_id_zero() {
    let docs = vec![vec![10], vec![20, 21]];
    let got = mask_pack(&docs, 5, true);
    assert_eq!(
        got,
        vec![PackedSequence {
            tokens: vec![10, EOS, 20, 21, PAD],
            mask: vec![1, 1, 1, 1, 0],
            document_ids: Some(vec![0, 0, 1, 1, 0]),
        }]
    );
}

#[test]
fn without_eos_ids_are_still_per_token() {
    let docs = vec![vec![7, 8], vec![9]];
    let got = mask_pack(&docs, 8, false);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].tokens[..3], [7, 8, 9]);
    assert_eq!(got[0].document_ids.as_deref().unwrap()[..3], [0, 0, 1]);
    assert!(got[0].document_ids.as_deref().unwrap()[3..]
        .iter()
        .all(|&id| id == 0));
    assert_eq!(&got[0].mask[3..], &[0, 0, 0, 0, 0]);
}

#[test]
fn empty_middle_document_still_advances_ids() {
    let docs = vec![vec![1], vec![], vec![2]];
    let got = mask_pack(&docs, 8, true);
    // 1, EOS(doc0), EOS(doc1), 2(doc2)
    assert_eq!(got[0].tokens[..4], [1, EOS, EOS, 2]);
    assert_eq!(got[0].document_ids.as_deref().unwrap()[..4], [0, 0, 1, 2]);
}

#[test]
fn split_long_doc_keeps_the_same_id() {
    let docs = vec![vec![1, 2, 3, 4, 5]];
    let got = mask_pack(&docs, 3, true);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].document_ids.as_deref(), Some(&[0, 0, 0][..]));
    assert_eq!(got[1].tokens, vec![4, 5, PAD]);
    assert_eq!(got[1].document_ids.as_deref(), Some(&[0, 0, 0][..]));
}

#[test]
fn vs_concat_tokens_and_mask_identical() {
    let docs = vec![vec![3, 1, 4], vec![1, 5, 9, 2], vec![6]];
    let concat = packing(Strategy::ConcatEos, 5, true, PAD);
    let masked = packing(Strategy::DocumentMask, 5, true, PAD);
    let a = pack(&docs, &concat, EOS).expect("concat");
    let b = pack(&docs, &masked, EOS).expect("mask");
    assert_eq!(a.len(), b.len());
    for (left, right) in a.iter().zip(&b) {
        assert_eq!(left.tokens, right.tokens);
        assert_eq!(left.mask, right.mask);
        assert_eq!(left.document_ids, None);
        assert!(right.document_ids.is_some());
        assert_packed_shape(right, 5, true, PAD);
    }
}

#[test]
fn vs_reference_random_docs() {
    let mut state: u64 = 0xBEEF;
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *state
    }
    for _ in 0..10 {
        let n_docs = (next(&mut state) % 5 + 1) as usize;
        let mut docs = Vec::with_capacity(n_docs);
        for _ in 0..n_docs {
            let n = (next(&mut state) % 9) as usize;
            docs.push((0..n).map(|_| (next(&mut state) % 20) as u32).collect());
        }
        let seq_len = (next(&mut state) % 7 + 1) as u32;
        let packing = packing(Strategy::DocumentMask, seq_len, true, PAD);
        let got = pack(&docs, &packing, EOS).expect("pack");
        assert_eq!(got, reference::pack(&docs, &packing, EOS).unwrap());
        for seq in &got {
            assert_packed_shape(seq, seq_len as usize, true, PAD);
            if let Some(ids) = &seq.document_ids {
                for (i, &m) in seq.mask.iter().enumerate() {
                    if m == 0 {
                        assert_eq!(ids[i], 0, "pad document id must be 0");
                    }
                }
            }
        }
    }
}
