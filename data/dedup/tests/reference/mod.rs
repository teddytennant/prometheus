//! Slow, obvious B2 reference: normalize, shingles, MinHash LSH, exact clustering.
#![allow(dead_code)]
//!
//! Compiled only as a submodule of the integration tests. Production
//! `src/lib.rs` must stay `unimplemented!` (except `sha256_hex`).
//!
//! Rules the public crate must match:
//!
//! **normalize**
//! 1. Strip zero-width format chars: U+200B, U+200C, U+200D, U+2060, U+FEFF, U+180E.
//! 2. Unicode lowercase (`str::to_lowercase`).
//! 3. Collapse every Unicode whitespace run (including newlines) to a single ASCII
//!    space and trim. Empty input or whitespace-only input becomes `""`.
//!
//! **paragraphs**
//! Blank-line split happens *after* the lowercase + zero-width strip, but *before*
//! collapsing newlines — otherwise there are no blank lines left. Concretely:
//! 1. Strip zero-width, lowercase, map `\r\n` / `\r` → `\n`.
//! 2. Split on runs of lines that are empty or Unicode-whitespace-only.
//! 3. Collapse whitespace inside each block (same as normalize).
//! 4. Drop empty strings.
//!
//! **exact_hash** — SHA-256 lowercase hex of `normalize(text)` as UTF-8.
//!
//! **shingles(text, n)** — word n-grams of `normalize(text)` split on whitespace,
//! joined by a single space, left-to-right. `n == 0` or fewer than `n` tokens →
//! empty vec. Duplicates are kept in the vec; MinHash uses the set.
//!
//! **minhash_signature** — `num_hashes` values. Hash function `i` of shingle `s` is
//! the first 8 bytes of SHA-256(`i` as little-endian u64 || UTF-8(`s`)), read as
//! little-endian `u64`. Signature slot `i` is the min of that over the unique
//! shingle set. Empty shingle set → `u64::MAX` repeated `num_hashes` times.
//! `bands`/`rows` are ignored here.
//!
//! **lsh_band_keys** — error (`Error::Config`) if `bands == 0` or `rows == 0` or
//! `num_hashes != bands * rows` (checked with `checked_mul`) or
//! `signature.len() != num_hashes`. Otherwise `bands` keys; key `b` is the first
//! 8 bytes of SHA-256(concatenation of `rows` little-endian u64s in
//! `signature[b*rows .. (b+1)*rows]`), read as little-endian `u64`.
//!
//! **Keep rule** — in any duplicate group / LSH component, keep the document with
//! the lexicographically lowest `(content_hash, id)`. `content_hash` is the field
//! on `Document`, never recomputed. Report `kept` / `dropped_*` / `Cluster.kept`
//! / `Cluster.dropped` are **document ids**. Lists are sorted lexicographically
//! by id; `clusters` sorted by `kept` id. Only components with at least one drop
//! appear in `clusters`.
//!
//! **dedup_exact Document** — group by `exact_hash(text)`; keep lowest
//! `(content_hash, id)` per group; others → `dropped_exact`. `dropped_near` and
//! `clusters` empty.
//!
//! **dedup_exact Paragraph** — drop the *whole* document if *any* of its
//! paragraph hashes (`exact_hash` of each `paragraphs(text)` entry, equivalently
//! SHA-256 of the already-normalized paragraph) matches a paragraph of a document
//! already kept. Walk documents in `(content_hash, id)` order so the keeper is
//! always the lowest hash. A document that shares no paragraph with any kept
//! document is kept. `clusters` empty.
//!
//! **dedup_near** — union-find over shared LSH band keys (transitivity applies).
//! Documents whose shingle set is empty do **not** participate (they do not join
//! each other either). `dropped_exact` empty.
//!
//! **dedup_corpus** — exact Document, then exact Paragraph on the remaining
//! keeps, then near on what is still kept. Later passes never see earlier drops.
//! `dropped_exact` concatenates both exact passes (then sorted).
//!
//! **Errors** — empty `docs` → `Error::Other`. Bad MinHash/LSH config →
//! `Error::Config`. Exact pass does not consult config.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use prometheus_dedup::{
    sha256_hex, Cluster, DedupConfig, DedupReport, Document, Error, Grain, Result,
};

const ZERO_WIDTH: &[char] = &[
    '\u{200B}', // ZERO WIDTH SPACE
    '\u{200C}', // ZERO WIDTH NON-JOINER
    '\u{200D}', // ZERO WIDTH JOINER
    '\u{2060}', // WORD JOINER
    '\u{FEFF}', // ZERO WIDTH NO-BREAK SPACE / BOM
    '\u{180E}', // MONGOLIAN VOWEL SEPARATOR
];

pub fn strip_zero_width(text: &str) -> String {
    text.chars().filter(|c| !ZERO_WIDTH.contains(c)).collect()
}

pub fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn normalize(text: &str) -> String {
    collapse_ws(&strip_zero_width(text).to_lowercase())
}

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

pub fn exact_hash(text: &str) -> String {
    sha256_hex(normalize(text).as_bytes())
}

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

/// First 8 digest bytes as little-endian u64.
pub fn u64_from_sha256(bytes: &[u8]) -> u64 {
    let hex = sha256_hex(bytes);
    let mut out = [0u8; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("sha256 hex");
    }
    u64::from_le_bytes(out)
}

fn hash_shingle(shingle: &str, seed: u64) -> u64 {
    let mut buf = Vec::with_capacity(8 + shingle.len());
    buf.extend_from_slice(&seed.to_le_bytes());
    buf.extend_from_slice(shingle.as_bytes());
    u64_from_sha256(&buf)
}

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

pub fn validate_config(config: DedupConfig) -> Result<()> {
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
        keys.push(u64_from_sha256(&buf));
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

pub fn jaccard(a: &str, b: &str, n: usize) -> f64 {
    let sa: HashSet<String> = shingles(a, n).into_iter().collect();
    let sb: HashSet<String> = shingles(b, n).into_iter().collect();
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

pub fn signature_matches(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x == y).count()
}
