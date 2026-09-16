//! Group: Loader — next_batch advances epoch/step/shard_offset/consumed_token_count
//! and draws sequences through shuffle_order.

mod common;

use common::{assert_err_exhausted, cursor_after, ident_corpus, is_sha256, packing, MIX_HASH, PAD};
use prometheus_loader::{shuffle_order, Loader, PackedSequence, Strategy};

const SEQ_LEN: usize = 4;
const N: u32 = 8;

fn packing_cfg() -> prometheus_loader::Packing {
    packing(Strategy::ConcatEos, SEQ_LEN as u32, true, PAD)
}

fn new_loader(seqs: Vec<PackedSequence>, seed: u64) -> Loader {
    Loader::new(seqs, packing_cfg(), seed, MIX_HASH.to_string())
}

#[test]
fn new_state_starts_at_origin() {
    let loader = new_loader(ident_corpus(N, SEQ_LEN), 7);
    let state = loader.state();
    assert_eq!(state.epoch, 0);
    assert_eq!(state.step, 0);
    assert_eq!(state.shuffle_seed, 7);
    assert_eq!(state.shard_index, 0);
    assert_eq!(state.shard_offset, 0);
    assert_eq!(state.consumed_token_count, 0);
    assert_eq!(state.data_mix_hash, MIX_HASH);
    assert!(
        is_sha256(&state.packing_remainder_hash),
        "packing_remainder_hash must be lowercase hex SHA-256, got {:?}",
        state.packing_remainder_hash
    );
}

#[test]
fn next_batch_shape_and_uses_shuffle_order() {
    let seqs = ident_corpus(N, SEQ_LEN);
    let seed = 7u64;
    let mut loader = new_loader(seqs.clone(), seed);
    let before = loader.state();
    let batch = loader.next_batch(3).expect("next_batch");
    assert_eq!(batch.tokens.len(), 3);
    assert_eq!(batch.mask.len(), 3);
    for (toks, mask) in batch.tokens.iter().zip(&batch.mask) {
        assert_eq!(toks.len(), SEQ_LEN);
        assert_eq!(mask.len(), SEQ_LEN);
    }

    let order = shuffle_order(u64::from(N), seed, before.epoch);
    for (i, toks) in batch.tokens.iter().enumerate() {
        let idx = order[before.shard_offset as usize + i] as usize;
        assert_eq!(
            toks,
            &seqs[idx].tokens,
            "batch slot {i} must be shuffle_order[{}]",
            before.shard_offset as usize + i
        );
        assert_eq!(batch.mask[i], seqs[idx].mask);
    }
}

#[test]
fn next_batch_advances_step_offset_and_consumed_tokens() {
    let mut loader = new_loader(ident_corpus(N, SEQ_LEN), 3);
    let before = loader.state();
    let batch_size = 3usize;
    let batch = loader.next_batch(batch_size).expect("next_batch");
    let after = loader.state();
    assert_eq!(after, batch.state, "loader.state() must equal Batch.state");
    assert_eq!(after.step, before.step + 1);
    assert_eq!(after.shuffle_seed, before.shuffle_seed);
    assert_eq!(after.shard_index, before.shard_index);
    assert_eq!(after.data_mix_hash, before.data_mix_hash);
    assert_eq!(after.packing_remainder_hash, before.packing_remainder_hash);

    let (epoch, offset) = cursor_after(u64::from(N), before.epoch, before.shard_offset, batch_size);
    assert_eq!(after.epoch, epoch);
    assert_eq!(after.shard_offset, offset);

    let emitted: u64 = batch.tokens.iter().map(|s| s.len() as u64).sum();
    assert_eq!(
        after.consumed_token_count,
        before.consumed_token_count + emitted
    );
    assert_eq!(emitted, (batch_size * SEQ_LEN) as u64);
}

#[test]
fn wrapping_an_epoch_advances_epoch_and_uses_the_next_shuffle() {
    let seqs = ident_corpus(N, SEQ_LEN);
    let seed = 11u64;
    let mut loader = new_loader(seqs.clone(), seed);
    let n = usize::try_from(N).unwrap();
    for _ in 0..n {
        loader.next_batch(1).expect("drain epoch 0");
    }
    let at_boundary = loader.state();
    assert_eq!(at_boundary.epoch, 1);
    assert_eq!(at_boundary.shard_offset, 0);
    assert_eq!(at_boundary.step, u64::from(N));

    let batch = loader.next_batch(1).expect("first batch of epoch 1");
    let order = shuffle_order(u64::from(N), seed, 1);
    assert_eq!(batch.tokens[0], seqs[order[0] as usize].tokens);
    assert_eq!(loader.state().epoch, 1);
    assert_eq!(loader.state().shard_offset, 1);
}

#[test]
fn wrap_inside_a_single_batch() {
    let seqs = ident_corpus(N, SEQ_LEN);
    let seed = 5u64;
    let mut loader = new_loader(seqs.clone(), seed);
    let take_first = usize::try_from(N).unwrap() - 2;
    loader.next_batch(take_first).expect("almost drain epoch 0");
    let before = loader.state();
    assert_eq!(before.epoch, 0);
    assert_eq!(before.shard_offset, take_first as u64);

    let batch = loader.next_batch(5).expect("batch that wraps");
    assert_eq!(batch.tokens.len(), 5);
    let order0 = shuffle_order(u64::from(N), seed, 0);
    let order1 = shuffle_order(u64::from(N), seed, 1);
    let expected_idx = [
        order0[take_first],
        order0[take_first + 1],
        order1[0],
        order1[1],
        order1[2],
    ];
    for (slot, idx) in expected_idx.iter().enumerate() {
        assert_eq!(batch.tokens[slot], seqs[*idx as usize].tokens);
    }
    let after = loader.state();
    assert_eq!(after.epoch, 1);
    assert_eq!(after.shard_offset, 3);
}

#[test]
fn from_state_resumes_where_the_saved_cursor_left_off() {
    let seqs = ident_corpus(N, SEQ_LEN);
    let seed = 19u64;
    let mut a = new_loader(seqs.clone(), seed);
    a.next_batch(2).expect("warmup");
    let saved = a.state();
    let follow = a.next_batch(3).expect("follow-on from live loader");

    let mut b = Loader::from_state(seqs, packing_cfg(), saved.clone()).expect("from_state");
    assert_eq!(b.state(), saved);
    let again = b.next_batch(3).expect("follow-on from restored loader");
    assert_eq!(again.tokens, follow.tokens);
    assert_eq!(again.mask, follow.mask);
    assert_eq!(again.state, follow.state);
}

#[test]
fn two_loaders_with_the_same_seed_agree() {
    let seqs = ident_corpus(N, SEQ_LEN);
    let mut a = new_loader(seqs.clone(), 4);
    let mut b = new_loader(seqs, 4);
    for _ in 0..5 {
        let left = a.next_batch(2).expect("a");
        let right = b.next_batch(2).expect("b");
        assert_eq!(left, right);
    }
}

#[test]
fn empty_corpus_next_batch_is_exhausted() {
    let mut loader = new_loader(Vec::new(), 1);
    assert_err_exhausted(loader.next_batch(1));
}

#[test]
fn from_state_empty_corpus_is_exhausted_on_next_batch() {
    let state = prometheus_loader::LoaderState {
        epoch: 0,
        step: 0,
        shuffle_seed: 1,
        shard_index: 0,
        shard_offset: 0,
        consumed_token_count: 0,
        data_mix_hash: MIX_HASH.to_string(),
        packing_remainder_hash: common::EMPTY_SHA256.to_string(),
    };
    let mut loader = Loader::from_state(Vec::new(), packing_cfg(), state).expect("from_state");
    assert_err_exhausted(loader.next_batch(2));
}
