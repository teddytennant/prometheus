//! Exact and MinHash near-dedup (spec 7, 15.5 B2).

use sha2::{Digest, Sha256};

pub fn exact_hash(text: &str) -> String {
    let mut h = Sha256::new();
    h.update(text.as_bytes());
    format!("{:x}", h.finalize())
}

/// 5-gram MinHash with `n_perm` permutations (a*x+b mod large prime).
pub fn minhash(text: &str, n_perm: usize) -> Vec<u64> {
    let grams = shingles(text, 5);
    let mut sig = vec![u64::MAX; n_perm];
    if grams.is_empty() {
        return vec![0; n_perm];
    }
    for g in grams {
        let hv = hash64(&g);
        for i in 0..n_perm {
            let a = 0x9e37_79b9_7f4a_7c15u64.wrapping_add(i as u64 * 0x1003);
            let b = 0xbf58_476d_1ce4_e5b9u64.wrapping_add(i as u64);
            let v = a.wrapping_mul(hv).wrapping_add(b);
            if v < sig[i] {
                sig[i] = v;
            }
        }
    }
    sig
}

pub fn jaccard_est(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let eq = a.iter().zip(b).filter(|(x, y)| x == y).count();
    eq as f64 / a.len() as f64
}

pub fn is_near_dup(a: &str, b: &str, threshold: f64) -> bool {
    if exact_hash(a) == exact_hash(b) {
        return true;
    }
    jaccard_est(&minhash(a, 32), &minhash(b, 32)) >= threshold
}

fn shingles(text: &str, n: usize) -> Vec<String> {
    let c: Vec<char> = text.chars().collect();
    if c.len() < n {
        return if c.is_empty() { vec![] } else { vec![text.to_string()] };
    }
    (0..=c.len() - n).map(|i| c[i..i + n].iter().collect()).collect()
}

fn hash64(s: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let out = h.finalize();
    u64::from_be_bytes(out[0..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_same() {
        assert_eq!(exact_hash("abc"), exact_hash("abc"));
        assert_ne!(exact_hash("abc"), exact_hash("abd"));
    }

    #[test]
    fn near_dup_detects_overlap() {
        let a = "the quick brown fox jumps over the lazy dog";
        let b = "the quick brown fox jumps over the lazy cat";
        assert!(is_near_dup(a, a, 0.9));
        assert!(jaccard_est(&minhash(a, 64), &minhash(b, 64)) > 0.3);
    }
}
