//! Group: reconstruct — THE B5 GATE.
//!
//! Given seed + state, reconstruct yields the same batch next_batch produced
//! at that cursor.

mod common;

use common::{ident_corpus, packing, MIX_HASH, PAD};
use prometheus_loader::{reconstruct, shuffle_order, Loader, PackedSequence, Strategy};

const SEQ_LEN: usize = 4;

fn packing_cfg() -> prometheus_loader::Packing {
    packing(Strategy::ConcatEos, SEQ_LEN as u32, true, PAD)
}

fn make_loader(seqs: Vec<PackedSequence>, seed: u64) -> Loader {
    Loader::new(seqs, packing_cfg(), seed, MIX_HASH.to_string())
}

#[test]
fn reconstruct_matches_next_batch_at_the_origin() {
    let seqs = ident_corpus(6, SEQ_LEN);
    let mut loader = make_loader(seqs.clone(), 7);
    let origin = loader.state();
    let batch = loader.next_batch(4).expect("next_batch");
    let rebuilt = reconstruct(&seqs, &packing_cfg(), &origin, 4).expect("reconstruct");
    assert_eq!(rebuilt, batch);
}

#[test]
fn reconstruct_matches_every_step_for_several_batch_sizes() {
    let seqs = ident_corpus(10, SEQ_LEN);
    let packing = packing_cfg();
    for batch_size in [1usize, 2, 3, 7, 10] {
        let mut loader = Loader::new(seqs.clone(), packing.clone(), 13, MIX_HASH.to_string());
        for step in 0..12 {
            let cursor = loader.state();
            let batch = loader.next_batch(batch_size).expect("next_batch");
            let rebuilt = reconstruct(&seqs, &packing, &cursor, batch_size).expect("reconstruct");
            assert_eq!(
                rebuilt, batch,
                "gate failed at live step {step} batch_size={batch_size}"
            );
        }
    }
}

#[test]
fn reconstruct_matches_a_restored_loader() {
    let seqs = ident_corpus(8, SEQ_LEN);
    let packing = packing_cfg();
    let mut live = Loader::new(seqs.clone(), packing.clone(), 21, MIX_HASH.to_string());
    live.next_batch(3).expect("warmup");
    let cursor = live.state();
    let from_live = live.next_batch(4).expect("live");

    let mut restored =
        Loader::from_state(seqs.clone(), packing.clone(), cursor.clone()).expect("from_state");
    let from_restored = restored.next_batch(4).expect("restored");
    let rebuilt = reconstruct(&seqs, &packing, &cursor, 4).expect("reconstruct");
    assert_eq!(from_live, from_restored);
    assert_eq!(rebuilt, from_live);
}

#[test]
fn reconstruct_uses_shuffle_order_at_the_saved_epoch() {
    let seqs = ident_corpus(8, SEQ_LEN);
    let packing = packing_cfg();
    let seed = 42u64;
    let mut loader = Loader::new(seqs.clone(), packing.clone(), seed, MIX_HASH.to_string());
    loader.next_batch(5).expect("move cursor");
    let cursor = loader.state();
    let batch = reconstruct(&seqs, &packing, &cursor, 3).expect("reconstruct");
    let order = shuffle_order(seqs.len() as u64, seed, cursor.epoch);
    let start = cursor.shard_offset as usize;
    for i in 0..3 {
        let idx = order[start + i] as usize;
        assert_eq!(batch.tokens[i], seqs[idx].tokens);
    }
}

#[test]
fn reconstruct_after_epoch_wrap() {
    let seqs = ident_corpus(5, SEQ_LEN);
    let packing = packing_cfg();
    let mut loader = Loader::new(seqs.clone(), packing.clone(), 3, MIX_HASH.to_string());
    for _ in 0..5 {
        loader.next_batch(1).expect("drain epoch 0");
    }
    let cursor = loader.state();
    assert_eq!(cursor.epoch, 1);
    let batch = loader.next_batch(2).expect("epoch 1");
    let rebuilt = reconstruct(&seqs, &packing, &cursor, 2).expect("reconstruct");
    assert_eq!(rebuilt, batch);
}

#[test]
fn reconstruct_empty_corpus_is_exhausted() {
    match reconstruct(
        &[],
        &packing_cfg(),
        &prometheus_loader::LoaderState {
            epoch: 0,
            step: 0,
            shuffle_seed: 1,
            shard_index: 0,
            shard_offset: 0,
            consumed_token_count: 0,
            data_mix_hash: MIX_HASH.to_string(),
            packing_remainder_hash: common::EMPTY_SHA256.to_string(),
        },
        1,
    ) {
        Err(prometheus_loader::Error::Exhausted) => {}
        Ok(_) => panic!("expected Exhausted"),
        Err(other) => panic!("expected Exhausted, got {other:?}"),
    }
}
