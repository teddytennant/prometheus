//! Slow, obvious B4 reference: ngrams, ngram_match, scan, method_hash.
#![allow(dead_code)]
//!
//! Rust tests call the public crate API and compare results to this module.
//! This code is not the production implementation.
//!
//! Rules (spec 07, 11, 15.5 B4, data/decontam/src/lib.rs):
//! - Tokenize by Unicode lowercase then `split_whitespace` (whitespace collapse).
//! - Emit overlapping word grams of width `NGRAM_N` (must stay 8) as
//!   space-joined strings, left to right. Fewer than `NGRAM_N` words → no grams.
//! - `ngram_match`: a shared 8-gram is one `Hit` per matching index item
//!   (`suite`, `item_id` = eval id, `method` = `"ngram"`). Empty index is
//!   `Error::MissingIndex`, never a clean/empty-hit success.
//! - `scan_text` / `scan_shard`: empty index → `MissingIndex`. No hits →
//!   `Status::Clean`; any hit → `Status::Flagged`. `against` is the unique
//!   suite list of the index; `method_hash` is always set on a successful scan.
//! - `scan_shard`: union hits across documents; any flagged document flags
//!   the shard. Hits are unique per `(suite, item_id)`.
//! - `method_hash`: lowercase SHA-256 hex of the canonical transcript below.
//!   It changes if index contents change or if `NGRAM_N` would change.
//!
//! Canonical `method_hash` transcript (UTF-8), empty index → `MissingIndex`:
//! ```text
//! prometheus-decontam/v1\n
//! ngram_n=<NGRAM_N>\n
//! embed_model=\n
//! ```
//! then for each item sorted by `(suite, id)`:
//! ```text
//! <suite>\0<id>\0<text>\n
//! ```
//! `embed_model` is empty when no embedder is configured.

use prometheus_decontam::{Error, EvalItem, Hit, Report, Status, NGRAM_N};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};

pub fn ngrams(text: &str) -> Vec<String> {
    ngrams_n(text, NGRAM_N)
}

pub fn ngrams_n(text: &str, n: usize) -> Vec<String> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();
    if n == 0 || words.len() < n {
        return Vec::new();
    }
    words.windows(n).map(|w| w.join(" ")).collect()
}

pub fn ngram_match(items: &[EvalItem], text: &str) -> Result<Vec<Hit>, Error> {
    if items.is_empty() {
        return Err(Error::MissingIndex);
    }
    let query: HashSet<String> = ngrams(text).into_iter().collect();
    let mut hits = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for item in items {
        let item_grams = ngrams(&item.text);
        if item_grams.iter().any(|g| query.contains(g))
            && seen.insert((item.suite.clone(), item.id.clone()))
        {
            hits.push(Hit {
                suite: item.suite.clone(),
                item_id: item.id.clone(),
                method: "ngram".to_string(),
            });
        }
    }
    Ok(hits)
}

pub fn method_hash(items: &[EvalItem]) -> Result<String, Error> {
    method_hash_n(items, NGRAM_N)
}

pub fn method_hash_n(items: &[EvalItem], ngram_n: usize) -> Result<String, Error> {
    if items.is_empty() {
        return Err(Error::MissingIndex);
    }
    let mut ordered = items.to_vec();
    ordered.sort_by(|a, b| (&a.suite, &a.id).cmp(&(&b.suite, &b.id)));
    let mut hasher = Sha256::new();
    hasher.update(b"prometheus-decontam/v1\n");
    hasher.update(format!("ngram_n={ngram_n}\n").as_bytes());
    hasher.update(b"embed_model=\n");
    for item in &ordered {
        hasher.update(item.suite.as_bytes());
        hasher.update([0u8]);
        hasher.update(item.id.as_bytes());
        hasher.update([0u8]);
        hasher.update(item.text.as_bytes());
        hasher.update([b'\n']);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn suites_against(items: &[EvalItem]) -> Vec<String> {
    let mut suites = BTreeSet::new();
    for item in items {
        suites.insert(item.suite.clone());
    }
    suites.into_iter().collect()
}

pub fn scan_text(items: &[EvalItem], text: &str) -> Result<Report, Error> {
    if items.is_empty() {
        return Err(Error::MissingIndex);
    }
    let hits = ngram_match(items, text)?;
    let hash = method_hash(items)?;
    Ok(report(hits, items, hash))
}

pub fn scan_shard(items: &[EvalItem], texts: &[String]) -> Result<Report, Error> {
    if items.is_empty() {
        return Err(Error::MissingIndex);
    }
    let mut hits = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for text in texts {
        for hit in ngram_match(items, text)? {
            if seen.insert((hit.suite.clone(), hit.item_id.clone())) {
                hits.push(hit);
            }
        }
    }
    let hash = method_hash(items)?;
    Ok(report(hits, items, hash))
}

fn report(hits: Vec<Hit>, items: &[EvalItem], method_hash: String) -> Report {
    let status = if hits.is_empty() {
        Status::Clean
    } else {
        Status::Flagged
    };
    Report {
        status,
        against: suites_against(items),
        method_hash,
        hits,
    }
}
