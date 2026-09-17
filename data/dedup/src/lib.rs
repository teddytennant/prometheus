//! Exact and MinHash near-duplicate detection (spec 7, 15.5 B2).
//!
//! Two passes over a corpus of extracted documents (B1):
//! - Exact: SHA-256 of normalized text, at document and paragraph grain.
//! - Near: MinHash + LSH so paraphrases and boilerplate collapse to one keep.
//!
//! Keep the lowest `content_hash` in each cluster. Golden outputs on a fixed
//! corpus are the B2 gate.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sha2::Digest;

/// Errors from config checks or an empty corpus.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dedup config: {0}")]
    Config(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One extracted document. `content_hash` is SHA-256 of the raw bytes B1 wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub content_hash: String,
}

/// MinHash / LSH knobs. `num_hashes == bands * rows`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupConfig {
    pub shingle_size: usize,
    pub num_hashes: usize,
    pub bands: usize,
    pub rows: usize,
}

impl DedupConfig {
    /// Default: 5-shingles, 128 hashes in 32 bands of 4.
    pub fn standard() -> Self {
        Self {
            shingle_size: 5,
            num_hashes: 128,
            bands: 32,
            rows: 4,
        }
    }
}

/// Grain of an exact-hash pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grain {
    Document,
    Paragraph,
}

/// One near-duplicate cluster. `kept` is the lowest `content_hash`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cluster {
    pub kept: String,
    pub dropped: Vec<String>,
}

/// Result of running both passes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupReport {
    pub kept: Vec<String>,
    pub dropped_exact: Vec<String>,
    pub dropped_near: Vec<String>,
    pub clusters: Vec<Cluster>,
}

const ZERO_WIDTH: &[char] = &[
    '\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}', '\u{FEFF}', '\u{180E}',
];

fn strip_zero_width(text: &str) -> String {
    text.chars().filter(|c| !ZERO_WIDTH.contains(c)).collect()
}

fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Lowercase, collapse whitespace, strip zero-width. Used before both hashes.
pub fn normalize(text: &str) -> String {
    collapse_ws(&strip_zero_width(text).to_lowercase())
}

/// Split on blank lines after `normalize`. Empty paragraphs are dropped.
pub fn paragraphs(text: &str) -> Vec<String> {
    let stripped = strip_zero_width(text)
        .to_lowercase()
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let mut out = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in stripped.split('\n') {
        if line.trim().is_empty() {
            flush_para(&mut current, &mut out);
        } else {
            current.push(line);
        }
    }
    flush_para(&mut current, &mut out);
    out
}

fn flush_para(current: &mut Vec<&str>, out: &mut Vec<String>) {
    if current.is_empty() {
        return;
    }
    let joined = current.join("\n");
    current.clear();
    let collapsed = collapse_ws(&joined);
    if !collapsed.is_empty() {
        out.push(collapsed);
    }
}

/// SHA-256 (lowercase hex) of `normalize(text)`.
pub fn exact_hash(text: &str) -> String {
    sha256_hex(normalize(text).as_bytes())
}

/// Word shingles of size `n`. Empty if the token list is shorter than `n`.
pub fn shingles(text: &str, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }
    let normalized = normalize(text);
    let words: Vec<&str> = normalized.split_whitespace().collect();
    if words.len() < n {
        return Vec::new();
    }
    words.windows(n).map(|w| w.join(" ")).collect()
}

fn u64_le_from_sha256(bytes: &[u8]) -> u64 {
    let d = sha2::Sha256::digest(bytes);
    u64::from_le_bytes(d[..8].try_into().expect("sha256 is 32 bytes"))
}

fn hash_shingle(shingle: &str, seed: u64) -> u64 {
    let mut buf = Vec::with_capacity(8 + shingle.len());
    buf.extend_from_slice(&seed.to_le_bytes());
    buf.extend_from_slice(shingle.as_bytes());
    u64_le_from_sha256(&buf)
}

/// `num_hashes` MinHash values of the shingle set.
pub fn minhash_signature(text: &str, config: DedupConfig) -> Vec<u64> {
    let unique: BTreeSet<String> = shingles(text, config.shingle_size).into_iter().collect();
    if unique.is_empty() {
        return vec![u64::MAX; config.num_hashes];
    }
    let mut sig = Vec::with_capacity(config.num_hashes);
    for i in 0..config.num_hashes {
        let mut best = u64::MAX;
        for s in &unique {
            let h = hash_shingle(s, i as u64);
            if h < best {
                best = h;
            }
        }
        sig.push(best);
    }
    sig
}

fn validate_config(config: DedupConfig) -> Result<()> {
    if config.shingle_size == 0 {
        return Err(Error::Config("shingle_size must be > 0".into()));
    }
    if config.bands == 0 || config.rows == 0 {
        return Err(Error::Config("bands and rows must be > 0".into()));
    }
    match config.bands.checked_mul(config.rows) {
        Some(n) if n == config.num_hashes => Ok(()),
        _ => Err(Error::Config("num_hashes must equal bands * rows".into())),
    }
}

/// Band keys: `bands` hashes, each over `rows` consecutive signature values.
pub fn lsh_band_keys(signature: &[u64], config: DedupConfig) -> Result<Vec<u64>> {
    validate_config(config)?;
    if signature.len() != config.num_hashes {
        return Err(Error::Config(
            "signature length must equal num_hashes (bands * rows)".into(),
        ));
    }
    let mut keys = Vec::with_capacity(config.bands);
    for b in 0..config.bands {
        let start = b * config.rows;
        let slice = &signature[start..start + config.rows];
        let mut buf = Vec::with_capacity(8 * config.rows);
        for x in slice {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        keys.push(u64_le_from_sha256(&buf));
    }
    Ok(keys)
}

fn empty_corpus<T>() -> Result<T> {
    Err(Error::Other("empty corpus".into()))
}

fn keeper_key(d: &Document) -> (&str, &str) {
    (d.content_hash.as_str(), d.id.as_str())
}

fn canonical(mut report: DedupReport) -> DedupReport {
    report.kept.sort();
    report.dropped_exact.sort();
    report.dropped_near.sort();
    for c in &mut report.clusters {
        c.dropped.sort();
    }
    report.clusters.sort_by(|a, b| a.kept.cmp(&b.kept));
    report
}

fn empty_report() -> DedupReport {
    DedupReport {
        kept: Vec::new(),
        dropped_exact: Vec::new(),
        dropped_near: Vec::new(),
        clusters: Vec::new(),
    }
}

/// Drop exact duplicates at `grain`. Keep the lowest `content_hash`.
pub fn dedup_exact(docs: &[Document], grain: Grain) -> Result<DedupReport> {
    if docs.is_empty() {
        return empty_corpus();
    }
    match grain {
        Grain::Document => exact_document(docs),
        Grain::Paragraph => exact_paragraph(docs),
    }
}

fn exact_document(docs: &[Document]) -> Result<DedupReport> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, d) in docs.iter().enumerate() {
        groups.entry(exact_hash(&d.text)).or_default().push(i);
    }
    let mut kept = Vec::new();
    let mut dropped_exact = Vec::new();
    for idxs in groups.values() {
        let mut idxs = idxs.clone();
        idxs.sort_by(|&a, &b| keeper_key(&docs[a]).cmp(&keeper_key(&docs[b])));
        kept.push(docs[idxs[0]].id.clone());
        for &i in &idxs[1..] {
            dropped_exact.push(docs[i].id.clone());
        }
    }
    Ok(canonical(DedupReport {
        kept,
        dropped_exact,
        dropped_near: Vec::new(),
        clusters: Vec::new(),
    }))
}

fn paragraph_hashes(text: &str) -> Vec<String> {
    paragraphs(text)
        .into_iter()
        .map(|p| exact_hash(&p))
        .collect()
}

/// Drop the whole document if any paragraph hash matches a kept document.
/// Walk in `(content_hash, id)` order so the kept doc is the lowest hash.
fn exact_paragraph(docs: &[Document]) -> Result<DedupReport> {
    let mut order: Vec<usize> = (0..docs.len()).collect();
    order.sort_by(|&a, &b| keeper_key(&docs[a]).cmp(&keeper_key(&docs[b])));
    let mut seen: HashSet<String> = HashSet::new();
    let mut kept = Vec::new();
    let mut dropped_exact = Vec::new();
    for i in order {
        let hashes = paragraph_hashes(&docs[i].text);
        if hashes.iter().any(|h| seen.contains(h)) {
            dropped_exact.push(docs[i].id.clone());
        } else {
            kept.push(docs[i].id.clone());
            seen.extend(hashes);
        }
    }
    Ok(canonical(DedupReport {
        kept,
        dropped_exact,
        dropped_near: Vec::new(),
        clusters: Vec::new(),
    }))
}

struct Uf {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Uf {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

/// MinHash LSH near-duplicate clustering. Union-find over shared band keys.
pub fn dedup_near(docs: &[Document], config: DedupConfig) -> Result<DedupReport> {
    if docs.is_empty() {
        return empty_corpus();
    }
    validate_config(config)?;
    let n = docs.len();
    let mut uf = Uf::new(n);
    let mut keys: Vec<Option<Vec<u64>>> = Vec::with_capacity(n);
    for d in docs {
        let sh = shingles(&d.text, config.shingle_size);
        if sh.is_empty() {
            keys.push(None);
            continue;
        }
        let sig = minhash_signature(&d.text, config);
        keys.push(Some(lsh_band_keys(&sig, config)?));
    }
    for b in 0..config.bands {
        let mut buckets: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (i, k) in keys.iter().enumerate() {
            if let Some(k) = k {
                buckets.entry(k[b]).or_default().push(i);
            }
        }
        for idxs in buckets.values() {
            for w in idxs.windows(2) {
                uf.union(w[0], w[1]);
            }
        }
    }
    let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        comps.entry(uf.find(i)).or_default().push(i);
    }
    let mut kept = Vec::new();
    let mut dropped_near = Vec::new();
    let mut clusters = Vec::new();
    for members in comps.values() {
        let mut members = members.clone();
        members.sort_by(|&a, &b| keeper_key(&docs[a]).cmp(&keeper_key(&docs[b])));
        let keep_id = docs[members[0]].id.clone();
        kept.push(keep_id.clone());
        if members.len() > 1 {
            let dropped: Vec<String> = members[1..].iter().map(|&i| docs[i].id.clone()).collect();
            dropped_near.extend(dropped.iter().cloned());
            clusters.push(Cluster {
                kept: keep_id,
                dropped,
            });
        }
    }
    Ok(canonical(DedupReport {
        kept,
        dropped_exact: Vec::new(),
        dropped_near,
        clusters,
    }))
}

fn select_kept(docs: &[Document], ids: &[String]) -> Vec<Document> {
    let want: HashSet<&str> = ids.iter().map(String::as_str).collect();
    docs.iter()
        .filter(|d| want.contains(d.id.as_str()))
        .cloned()
        .collect()
}

/// Exact document, exact paragraph, then near. Later passes see earlier keeps.
pub fn dedup_corpus(docs: &[Document], config: DedupConfig) -> Result<DedupReport> {
    if docs.is_empty() {
        return empty_corpus();
    }
    validate_config(config)?;
    let r_doc = dedup_exact(docs, Grain::Document)?;
    let after_doc = select_kept(docs, &r_doc.kept);
    let r_para = dedup_exact(&after_doc, Grain::Paragraph)?;
    let after_para = select_kept(&after_doc, &r_para.kept);
    let r_near = if after_para.is_empty() {
        canonical(empty_report())
    } else {
        dedup_near(&after_para, config)?
    };
    let mut dropped_exact = r_doc.dropped_exact;
    dropped_exact.extend(r_para.dropped_exact);
    Ok(canonical(DedupReport {
        kept: r_near.kept,
        dropped_exact,
        dropped_near: r_near.dropped_near,
        clusters: r_near.clusters,
    }))
}

/// SHA-256 of bytes as lowercase hex. Available to implementers; not the pass.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let d = sha2::Sha256::digest(bytes);
    hex_lower(&d)
}

fn hex_lower(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(H[(b >> 4) as usize] as char);
        out.push(H[(b & 0x0f) as usize] as char);
    }
    out
}
