//! Group: `prometheus_obs::reconstruct_batch` is `prometheus_loader::reconstruct`
//! (spec 7.2 spike diagnosis). Allowed to pass the I10 stub (re-export).

use prometheus_loader::{reconstruct, Strategy};
use prometheus_obs::{reconstruct_batch, LoaderState, PackedSequence, Packing};

#[test]
fn reconstruct_batch_equals_loader_reconstruct_on_tiny_packed_set() {
    let sequences = vec![
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
            tokens: vec![7, 8, 9],
            mask: vec![1, 1, 1],
            document_ids: None,
        },
    ];
    let packing = Packing {
        packing_id: "tiny-i10".to_string(),
        sequence_length: 3,
        strategy: Strategy::ConcatEos,
        eos_between_docs: true,
        pad_id: 0,
        drop_short_below: None,
    };
    let state = LoaderState {
        epoch: 0,
        step: 0,
        shuffle_seed: 42,
        shard_index: 0,
        shard_offset: 0,
        consumed_token_count: 0,
        data_mix_hash: "ab".repeat(32),
        packing_remainder_hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_string(),
    };
    let via_obs = reconstruct_batch(&sequences, &packing, &state, 2).expect("obs reconstruct");
    let via_loader = reconstruct(&sequences, &packing, &state, 2).expect("loader reconstruct");
    assert_eq!(via_obs, via_loader);
    assert_eq!(via_obs.tokens.len(), 2);
    assert_eq!(via_obs.mask.len(), 2);
}
