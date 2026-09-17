//! Independent L2 eval-gate reference. **Not** `prometheus_eval_gate::EvalGate`.
//!
//! Production never imports `tests/`. Implementers should treat this module plus
//! `eval_gate.py` as the slow, obvious contract for hashing, grading, rotation
//! and the on-disk layout `EvalGate::open` must read.
//!
//! # On-disk layout (`GateConfig.suite_root`)
//!
//! ```text
//! <suite_root>/
//!   <slug>/
//!     current.json          # live generation pointer
//!     history.jsonl         # retired current.json objects, one per line
//!     generations/<hash>/
//!       meta.json
//!       items.jsonl         # one item object per line
//!     pending/
//!       meta.json
//!       items.jsonl         # candidate for rotate()
//!   subjects/
//!     genome/<rev>.json     # {"<item_id>": "<answer>", ...}
//!     checkpoint/<id>.json  # same shape
//! ```
//!
//! `current.json`:
//! `{"suite_hash","installed_ms","cutoff_ms","written_ms","compute_ms"}`
//!
//! Generation `meta.json`: `{"written_ms","compute_ms","cutoff_ms"}`
//!
//! Pending `meta.json`: `{"written_ms","compute_ms"}`
//!
//! Item line (extra keys forbidden):
//! `{"id","prompt","answer","weight_milli"}`
//!
//! # Math
//!
//! * `suite_hash` = SHA-256 (lowercase hex) of canonical JSON lines in **id**
//!   order. Each line is `{"answer","id","prompt","weight_milli"}` with sorted
//!   keys, compact, UTF-8 (raw unicode), then `\n`. Independent of file key
//!   order, harness_version, now, compute_ms.
//! * `rci_milli` = saturating sum of `weight_milli` for items whose subject
//!   answer equals the hidden answer (exact string match). Missing → 0.
//! * `n_items` = item count of the live generation.
//! * `compute_ms` on Scores is the generation's **fixed compute budget** from
//!   meta.json, not wall time.
//! * `harness_version` is copied from `GateConfig`; it does not affect RCI or
//!   suite_hash.
//! * `now` is injected. The reference does not sleep or read the wall clock.
//!
//! # Rotation
//!
//! `rotate(suite, cutoff_ms, now)` installs `pending/` iff:
//! 1. suite is held-out (else `UnknownSuite` / `NotHeldOut`);
//! 2. pending exists, is non-empty, ids unique, ids non-empty;
//! 3. `pending.written_ms > cutoff_ms` (strict) and `cutoff_ms <= now`;
//! 4. no live generation, **or** `now >= installed_ms + ROTATION_MS`.
//!
//! On success pending is consumed, the previous generation directory is left
//! in place (old hash stays on disk), and the new hash is returned.
//!
//! # Subjects
//!
//! Ids must be a single path component (`/`, `\`, NUL, `.`, `..` refused →
//! `SubjectNotFound`). Missing file → `SubjectNotFound`.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use prometheus_eval_gate::{Error, GateConfig, NowMs, Result, Scores, Subject, ROTATION_MS};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// F3 public slugs, duplicated here on purpose. Tests must not import `evals/`.
pub const PUBLIC_SUITES: &[&str] = &[
    "re_bench",
    "mle_bench",
    "paperbench",
    "speedrun",
    "metr_time_horizon",
    "swe_bench_verified",
    "swe_bench_pro",
    "terminal_bench",
    "competitive_programming",
    "aime",
    "hmmt",
    "frontiermath",
    "minif2f",
    "putnambench",
    "arc_agi_1",
    "arc_agi_2",
    "arc_agi_3",
    "hle",
    "gpqa",
    "long_context_128k",
    "long_context_1m",
    "forecasting",
];

pub const HELD_OUT_SUITES: &[&str] = &[
    "re_bench",
    "mle_bench",
    "paperbench",
    "speedrun",
    "metr_time_horizon",
];

pub const GOLDEN_A_HASH: &str = "b1f97a1b6089e43fd1eadfd1961819ce90696d9c677817cc08ff754083fc11b9";
pub const EMPTY_HASH: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Item {
    pub id: String,
    pub prompt: String,
    pub answer: String,
    pub weight_milli: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CurrentFile {
    suite_hash: String,
    installed_ms: u64,
    cutoff_ms: u64,
    written_ms: u64,
    compute_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenMeta {
    written_ms: u64,
    compute_ms: u64,
    cutoff_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingMeta {
    written_ms: u64,
    compute_ms: u64,
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub items: Vec<Item>,
    pub suite_hash: String,
    pub written_ms: u64,
    pub compute_ms: u64,
    pub cutoff_ms: u64,
    pub installed_ms: u64,
}

/// Independent gate. Does not construct or call `EvalGate`.
pub struct RefGate {
    root: PathBuf,
    harness_version: String,
}

pub fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

pub fn canonical_item_json(item: &Item) -> String {
    let mut map = serde_json::Map::new();
    map.insert(
        "answer".to_string(),
        serde_json::Value::String(item.answer.clone()),
    );
    map.insert("id".to_string(), serde_json::Value::String(item.id.clone()));
    map.insert(
        "prompt".to_string(),
        serde_json::Value::String(item.prompt.clone()),
    );
    map.insert(
        "weight_milli".to_string(),
        serde_json::Value::Number(item.weight_milli.into()),
    );
    serde_json::to_string(&serde_json::Value::Object(map)).expect("item json")
}

pub fn hash_items(items: &[Item]) -> String {
    let mut ordered: Vec<&Item> = items.iter().collect();
    ordered.sort_by(|a, b| a.id.cmp(&b.id));
    let mut h = Sha256::new();
    for item in ordered {
        h.update(canonical_item_json(item).as_bytes());
        h.update(b"\n");
    }
    hex_lower(&h.finalize())
}

pub fn grade_rci(items: &[Item], answers: &BTreeMap<String, String>) -> i64 {
    let mut total: i64 = 0;
    for item in items {
        if answers.get(&item.id).is_some_and(|p| p == &item.answer) {
            total = total.saturating_add(item.weight_milli);
        }
    }
    total
}

pub fn classify_suite(suite: &str) -> Result<()> {
    if HELD_OUT_SUITES.contains(&suite) {
        Ok(())
    } else if PUBLIC_SUITES.contains(&suite) {
        Err(Error::NotHeldOut(suite.to_string()))
    } else {
        Err(Error::UnknownSuite(suite.to_string()))
    }
}

pub fn is_safe_subject_id(id: &str) -> bool {
    !id.is_empty()
        && !id.contains('/')
        && !id.contains('\\')
        && !id.contains('\0')
        && id != "."
        && id != ".."
        && !id.contains("..")
}

fn suite_dir(root: &Path, suite: &str) -> PathBuf {
    root.join(suite)
}

fn current_path(root: &Path, suite: &str) -> PathBuf {
    suite_dir(root, suite).join("current.json")
}

fn history_path(root: &Path, suite: &str) -> PathBuf {
    suite_dir(root, suite).join("history.jsonl")
}

fn generation_dir(root: &Path, suite: &str, hash: &str) -> PathBuf {
    suite_dir(root, suite).join("generations").join(hash)
}

fn pending_dir(root: &Path, suite: &str) -> PathBuf {
    suite_dir(root, suite).join("pending")
}

fn subject_path(root: &Path, subject: &Subject) -> Result<PathBuf> {
    let (kind, id) = match subject {
        Subject::GenomeRev(s) => ("genome", s.as_str()),
        Subject::Checkpoint(s) => ("checkpoint", s.as_str()),
    };
    if !is_safe_subject_id(id) {
        return Err(Error::SubjectNotFound(id.to_string()));
    }
    Ok(root.join("subjects").join(kind).join(format!("{id}.json")))
}

fn msg(s: impl Into<String>) -> Error {
    Error::Message(s.into())
}

fn read_jsonl_items(path: &Path) -> Result<Vec<Item>> {
    let file = File::open(path).map_err(|e| msg(format!("open items: {e}")))?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| msg(format!("read items: {e}")))?;
        if line.trim().is_empty() {
            continue;
        }
        let item: Item = serde_json::from_str(&line)
            .map_err(|_| msg(format!("invalid item at line {}", i + 1)))?;
        if item.id.is_empty() {
            return Err(msg("empty item id"));
        }
        if !seen.insert(item.id.clone()) {
            return Err(msg("duplicate item id"));
        }
        items.push(item);
    }
    Ok(items)
}

fn write_jsonl_items(path: &Path, items: &[Item]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| msg(format!("mkdir: {e}")))?;
    }
    let mut f = File::create(path).map_err(|e| msg(format!("create items: {e}")))?;
    for item in items {
        let line = serde_json::to_string(item).map_err(|e| msg(format!("ser item: {e}")))?;
        writeln!(f, "{line}").map_err(|e| msg(format!("write items: {e}")))?;
    }
    f.flush().map_err(|e| msg(format!("flush items: {e}")))?;
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| msg(format!("mkdir: {e}")))?;
    }
    let bytes = serde_json::to_vec(value).map_err(|e| msg(format!("ser: {e}")))?;
    fs::write(path, bytes).map_err(|e| msg(format!("write: {e}")))?;
    Ok(())
}

fn load_generation(root: &Path, suite: &str) -> Result<Generation> {
    let cur_path = current_path(root, suite);
    let bytes = fs::read(&cur_path).map_err(|_| msg("no current generation"))?;
    let cur: CurrentFile =
        serde_json::from_slice(&bytes).map_err(|_| msg("invalid current.json"))?;
    let dir = generation_dir(root, suite, &cur.suite_hash);
    let items = read_jsonl_items(&dir.join("items.jsonl"))?;
    let hashed = hash_items(&items);
    if hashed != cur.suite_hash {
        return Err(msg("current suite_hash does not match items"));
    }
    Ok(Generation {
        items,
        suite_hash: cur.suite_hash,
        written_ms: cur.written_ms,
        compute_ms: cur.compute_ms,
        cutoff_ms: cur.cutoff_ms,
        installed_ms: cur.installed_ms,
    })
}

fn load_pending(root: &Path, suite: &str) -> Result<(Vec<Item>, PendingMeta)> {
    let dir = pending_dir(root, suite);
    if !dir.is_dir() {
        return Err(Error::RotationRefused("no pending suite".into()));
    }
    let meta_bytes = fs::read(dir.join("meta.json"))
        .map_err(|_| Error::RotationRefused("no pending meta".into()))?;
    let meta: PendingMeta = serde_json::from_slice(&meta_bytes)
        .map_err(|_| Error::RotationRefused("invalid pending meta".into()))?;
    let items = read_jsonl_items(&dir.join("items.jsonl"))
        .map_err(|e| Error::RotationRefused(format!("pending items: {e}")))?;
    if items.is_empty() {
        return Err(Error::RotationRefused("empty suite".into()));
    }
    Ok((items, meta))
}

fn load_answers(path: &Path, subject: &Subject) -> Result<BTreeMap<String, String>> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return Err(Error::SubjectNotFound(subject.as_str().to_string())),
    };
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| msg("invalid subject json"))?;
    let obj = value
        .as_object()
        .ok_or_else(|| msg("subject json must be an object"))?;
    let mut answers = BTreeMap::new();
    for (k, v) in obj {
        let s = v
            .as_str()
            .ok_or_else(|| msg("subject answers must be strings"))?;
        answers.insert(k.clone(), s.to_string());
    }
    Ok(answers)
}

/// Write a live generation the production `open` must be able to read.
pub fn install_current(
    root: &Path,
    suite: &str,
    items: Vec<Item>,
    written_ms: u64,
    cutoff_ms: u64,
    compute_ms: u64,
    installed_ms: u64,
) -> Result<String> {
    classify_suite(suite)?;
    let suite_hash = hash_items(&items);
    let dir = generation_dir(root, suite, &suite_hash);
    write_jsonl_items(&dir.join("items.jsonl"), &items)?;
    write_json(
        &dir.join("meta.json"),
        &GenMeta {
            written_ms,
            compute_ms,
            cutoff_ms,
        },
    )?;
    write_json(
        &current_path(root, suite),
        &CurrentFile {
            suite_hash: suite_hash.clone(),
            installed_ms,
            cutoff_ms,
            written_ms,
            compute_ms,
        },
    )?;
    Ok(suite_hash)
}

pub fn install_pending(
    root: &Path,
    suite: &str,
    items: Vec<Item>,
    written_ms: u64,
    compute_ms: u64,
) -> Result<()> {
    classify_suite(suite)?;
    let dir = pending_dir(root, suite);
    write_jsonl_items(&dir.join("items.jsonl"), &items)?;
    write_json(
        &dir.join("meta.json"),
        &PendingMeta {
            written_ms,
            compute_ms,
        },
    )?;
    Ok(())
}

pub fn install_subject(
    root: &Path,
    subject: &Subject,
    answers: &BTreeMap<String, String>,
) -> Result<()> {
    let path = subject_path(root, subject)?;
    write_json(&path, answers)?;
    Ok(())
}

impl RefGate {
    pub fn open(config: GateConfig) -> Result<Self> {
        if config.harness_version.is_empty() {
            return Err(msg("empty harness_version"));
        }
        if !config.suite_root.is_dir() {
            return Err(msg("suite_root is not a directory"));
        }
        Ok(Self {
            root: config.suite_root,
            harness_version: config.harness_version,
        })
    }

    pub fn request(&self, subject: Subject, suite: &str, _now: NowMs) -> Result<Scores> {
        classify_suite(suite)?;
        let gen = load_generation(&self.root, suite)?;
        let path = subject_path(&self.root, &subject)?;
        let answers = load_answers(&path, &subject)?;
        let rci_milli = grade_rci(&gen.items, &answers);
        let n_items = u32::try_from(gen.items.len()).map_err(|_| msg("n_items overflow"))?;
        Ok(Scores {
            suite: suite.to_string(),
            subject,
            rci_milli,
            n_items,
            compute_ms: gen.compute_ms,
            harness_version: self.harness_version.clone(),
            suite_hash: gen.suite_hash,
        })
    }

    pub fn rotate(&mut self, suite: &str, cutoff_ms: NowMs, now: NowMs) -> Result<String> {
        classify_suite(suite)?;
        if cutoff_ms > now {
            return Err(Error::RotationRefused("cutoff_ms is after now".into()));
        }
        let (items, meta) = load_pending(&self.root, suite)?;
        if meta.written_ms <= cutoff_ms {
            return Err(Error::RotationRefused(
                "items not written after cutoff".into(),
            ));
        }
        match load_generation(&self.root, suite) {
            Ok(cur) => {
                if now < cur.installed_ms.saturating_add(ROTATION_MS) {
                    return Err(Error::RotationRefused(
                        "rotation interval not elapsed".into(),
                    ));
                }
                let hist = history_path(&self.root, suite);
                if let Some(parent) = hist.parent() {
                    fs::create_dir_all(parent).map_err(|e| msg(format!("mkdir: {e}")))?;
                }
                let mut f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&hist)
                    .map_err(|e| msg(format!("history: {e}")))?;
                let retired = CurrentFile {
                    suite_hash: cur.suite_hash,
                    installed_ms: cur.installed_ms,
                    cutoff_ms: cur.cutoff_ms,
                    written_ms: cur.written_ms,
                    compute_ms: cur.compute_ms,
                };
                let line = serde_json::to_string(&retired).map_err(|e| msg(format!("ser: {e}")))?;
                writeln!(f, "{line}").map_err(|e| msg(format!("history write: {e}")))?;
            }
            Err(Error::Message(_)) => {}
            Err(e) => return Err(e),
        }
        let suite_hash = hash_items(&items);
        let dir = generation_dir(&self.root, suite, &suite_hash);
        write_jsonl_items(&dir.join("items.jsonl"), &items)?;
        write_json(
            &dir.join("meta.json"),
            &GenMeta {
                written_ms: meta.written_ms,
                compute_ms: meta.compute_ms,
                cutoff_ms,
            },
        )?;
        write_json(
            &current_path(&self.root, suite),
            &CurrentFile {
                suite_hash: suite_hash.clone(),
                installed_ms: now,
                cutoff_ms,
                written_ms: meta.written_ms,
                compute_ms: meta.compute_ms,
            },
        )?;
        let pending = pending_dir(&self.root, suite);
        let _ = fs::remove_dir_all(&pending);
        Ok(suite_hash)
    }

    pub fn suite_hash(&self, suite: &str) -> Result<String> {
        classify_suite(suite)?;
        Ok(load_generation(&self.root, suite)?.suite_hash)
    }
}

pub fn golden_item() -> Item {
    Item {
        id: "a".into(),
        prompt: "p".into(),
        answer: "s".into(),
        weight_milli: 1,
    }
}
