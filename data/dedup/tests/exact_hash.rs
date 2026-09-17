//! Group: exact_hash — SHA-256 lowercase hex of normalize(text).

mod common;
mod reference;

use prometheus_dedup::{exact_hash, sha256_hex};

#[test]
fn hello_world_golden() {
    let want = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
    assert_eq!(
        sha256_hex(b"hello"),
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(sha256_hex(b"hello world"), want);
    assert_eq!(exact_hash("Hello World"), want);
    assert_eq!(exact_hash("  HELLO   world  "), want);
}

#[test]
fn is_sha256_of_normalized_utf8() {
    for s in ["Hello World", "A\u{200B}B", "", "café", common::alphabet()] {
        let got = exact_hash(s);
        let through_norm = sha256_hex(prometheus_dedup::normalize(s).as_bytes());
        assert_eq!(got, through_norm);
        assert_eq!(got, reference::exact_hash(s));
        assert_eq!(got.len(), 64);
        assert!(
            got.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
            "hash must be lowercase hex, got {got}"
        );
    }
}

#[test]
fn identical_normalized_texts_collide() {
    let a = exact_hash("The Quick BROWN fox");
    let b = exact_hash("the   quick brown  FOX");
    assert_eq!(a, b);
    assert_eq!(a, reference::exact_hash("The Quick BROWN fox"));
}

#[test]
fn different_texts_do_not_collide() {
    let a = exact_hash("hello world");
    let b = exact_hash("hello world!");
    assert_ne!(a, b);
}

#[test]
fn zero_width_does_not_change_hash_once_stripped() {
    assert_eq!(exact_hash("ab"), exact_hash("a\u{200B}b"));
}
