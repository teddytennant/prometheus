//! On-disk held-out suite store. Item bodies stay in this module; callers only
//! see hashes, scores, and leak-free error strings.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, GateConfig, NowMs, Result, Scores, Subject, HELD_OUT_SUITES, ROTATION_MS};

/// F3 public slugs that are not held-out. Checked after [`HELD_OUT_SUITES`].
const PUBLIC_SUITES: &[&str] = &[
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Item {
    id: String,
    prompt: String,
    answer: String,
    weight_milli: i64,
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

struct Generation {
    items: Vec<Item>,
    suite_hash: String,
    written_ms: u64,
    compute_ms: u64,
    cutoff_ms: u64,
    installed_ms: u64,
}

pub(crate) struct Store {
    root: PathBuf,
    harness_version: String,
}

fn msg(s: impl Into<String>) -> Error {
    Error::Message(s.into())
}

/// Never include file bytes, serde details, or item fields: those can carry
/// held-out prompts and answers into `Error` Display/Debug.
fn classify_suite(suite: &str) -> Result<()> {
    if HELD_OUT_SUITES.contains(&suite) {
        Ok(())
    } else if PUBLIC_SUITES.contains(&suite) {
        Err(Error::NotHeldOut(suite.to_string()))
    } else {
        Err(Error::UnknownSuite(suite.to_string()))
    }
}

fn is_safe_subject_id(id: &str) -> bool {
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

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn canonical_item_json(item: &Item) -> Result<String> {
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
    serde_json::to_string(&serde_json::Value::Object(map)).map_err(|_| msg("canonical json"))
}

fn hash_items(items: &[Item]) -> Result<String> {
    let mut ordered: Vec<&Item> = items.iter().collect();
    ordered.sort_by(|a, b| a.id.cmp(&b.id));
    let mut h = Sha256::new();
    for item in ordered {
        h.update(canonical_item_json(item)?.as_bytes());
        h.update(b"\n");
    }
    Ok(hex_lower(&h.finalize()))
}

fn grade_rci(items: &[Item], answers: &BTreeMap<String, String>) -> i64 {
    let mut total: i64 = 0;
    for item in items {
        if answers.get(&item.id).is_some_and(|p| p == &item.answer) {
            total = total.saturating_add(item.weight_milli);
        }
    }
    total
}

fn read_jsonl_items(path: &Path) -> Result<Vec<Item>> {
    let file = File::open(path).map_err(|_| msg("open items"))?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|_| msg("read items"))?;
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
        fs::create_dir_all(parent).map_err(|_| msg("mkdir"))?;
    }
    let mut f = File::create(path).map_err(|_| msg("create items"))?;
    for item in items {
        let line = serde_json::to_string(item).map_err(|_| msg("ser item"))?;
        writeln!(f, "{line}").map_err(|_| msg("write items"))?;
    }
    f.flush().map_err(|_| msg("flush items"))?;
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|_| msg("mkdir"))?;
    }
    let bytes = serde_json::to_vec(value).map_err(|_| msg("ser"))?;
    fs::write(path, bytes).map_err(|_| msg("write"))?;
    Ok(())
}

fn load_generation(root: &Path, suite: &str) -> Result<Generation> {
    let cur_path = current_path(root, suite);
    let bytes = fs::read(&cur_path).map_err(|_| msg("no current generation"))?;
    let cur: CurrentFile =
        serde_json::from_slice(&bytes).map_err(|_| msg("invalid current.json"))?;
    let dir = generation_dir(root, suite, &cur.suite_hash);
    let items = read_jsonl_items(&dir.join("items.jsonl"))?;
    let hashed = hash_items(&items)?;
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
        .map_err(|_| Error::RotationRefused("pending items".into()))?;
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

impl Store {
    pub(crate) fn open(config: GateConfig) -> Result<Self> {
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

    pub(crate) fn request(&self, subject: Subject, suite: &str) -> Result<Scores> {
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

    pub(crate) fn rotate(&mut self, suite: &str, cutoff_ms: NowMs, now: NowMs) -> Result<String> {
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
                    fs::create_dir_all(parent).map_err(|_| msg("mkdir"))?;
                }
                let mut f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&hist)
                    .map_err(|_| msg("history"))?;
                let retired = CurrentFile {
                    suite_hash: cur.suite_hash,
                    installed_ms: cur.installed_ms,
                    cutoff_ms: cur.cutoff_ms,
                    written_ms: cur.written_ms,
                    compute_ms: cur.compute_ms,
                };
                let line = serde_json::to_string(&retired).map_err(|_| msg("ser"))?;
                writeln!(f, "{line}").map_err(|_| msg("history write"))?;
            }
            Err(Error::Message(_)) => {}
            Err(e) => return Err(e),
        }
        let suite_hash = hash_items(&items)?;
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

    pub(crate) fn suite_hash(&self, suite: &str) -> Result<String> {
        classify_suite(suite)?;
        Ok(load_generation(&self.root, suite)?.suite_hash)
    }
}
