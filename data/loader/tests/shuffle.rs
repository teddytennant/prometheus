//! Group: shuffle_order — permutation of 0..n, deterministic, (seed, epoch)
//! pairs differ. The PRNG is not pinned; Loader/reconstruct must use this
//! function.

use prometheus_loader::shuffle_order;

fn assert_permutation(order: &[u64], n: u64) {
    assert_eq!(order.len(), n as usize, "shuffle_order({n}) length");
    let mut sorted = order.to_vec();
    sorted.sort_unstable();
    let expected: Vec<u64> = (0..n).collect();
    assert_eq!(
        sorted, expected,
        "shuffle_order({n}) must be a permutation of 0..n"
    );
}

#[test]
fn empty_n_is_empty_permutation() {
    assert!(shuffle_order(0, 7, 0).is_empty());
    assert!(shuffle_order(0, 0, 3).is_empty());
}

#[test]
fn n_one_is_always_zero() {
    assert_eq!(shuffle_order(1, 0, 0), vec![0]);
    assert_eq!(shuffle_order(1, 99, 4), vec![0]);
}

#[test]
fn is_permutation_for_several_n() {
    for n in [2u64, 3, 8, 16, 32, 100, 1000] {
        for &(seed, epoch) in &[(0, 0), (1, 0), (0, 1), (7, 3), (u64::MAX, 9)] {
            assert_permutation(&shuffle_order(n, seed, epoch), n);
        }
    }
}

#[test]
fn deterministic_same_inputs_same_order() {
    for n in [8u64, 64, 257] {
        for &(seed, epoch) in &[(0, 0), (123456789, 2), (7, 11)] {
            let a = shuffle_order(n, seed, epoch);
            let b = shuffle_order(n, seed, epoch);
            assert_eq!(a, b, "shuffle_order must be deterministic");
        }
    }
}

#[test]
fn different_seed_or_epoch_differ() {
    let n = 32u64;
    let base = shuffle_order(n, 0, 0);
    assert_ne!(
        base,
        shuffle_order(n, 0, 1),
        "epoch must affect the permutation"
    );
    assert_ne!(
        base,
        shuffle_order(n, 1, 0),
        "seed must affect the permutation"
    );
    assert_ne!(
        shuffle_order(n, 1, 0),
        shuffle_order(n, 1, 1),
        "epoch must affect the permutation for a nonzero seed"
    );
    assert_ne!(
        shuffle_order(n, 7, 3),
        shuffle_order(n, 8, 3),
        "nearby seeds must not collapse"
    );
}

#[test]
fn not_the_identity_for_a_typical_seed() {
    let n = 64u64;
    let order = shuffle_order(n, 7, 0);
    let identity: Vec<u64> = (0..n).collect();
    assert_ne!(
        order, identity,
        "a real shuffle of 64 items is not the identity"
    );
}
