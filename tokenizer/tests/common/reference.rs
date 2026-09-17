//! Slow, obvious byte-fallback BPE reference (F6 oracle).
//!
//! Must match `tests/reference/tokenizer.py` token ids on the same docs+config.
//! Production `prometheus-tokenizer` must match this module; production must
//! never import `tests/`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use prometheus_tokenizer::{
    ArcGridTokenRange, Artifact, Error, SpecialTokens, TokenizerMeta, TrainConfig, VocabPointer,
    ALGORITHM, ARC_N_COLORS, SCHEMA_ID, SCHEMA_VERSION, VOCAB_SCHEMA_ID,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub const N_SPECIALS: u32 = 4;
pub const N_BYTES: u32 = 256;
pub const FIRST_BYTE: u32 = N_SPECIALS;
pub const FIRST_ARC: u32 = N_SPECIALS + N_BYTES;
pub const FIRST_MERGE: u32 = FIRST_ARC + ARC_N_COLORS as u32;
pub const MIN_VOCAB_SIZE: u32 = FIRST_MERGE + 1;
pub const VOCAB_FILENAME: &str = "vocab.jsonl";
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";
pub const VOCAB_MEDIA_TYPE: &str = "application/jsonl";
pub const VOCAB_FORMAT: &str = "jsonl";

const REQUIRED_SPECIALS: [&str; 4] = ["bos", "eos", "pad", "unk"];

pub fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn byte_id(byte: u8) -> u32 {
    FIRST_BYTE + u32::from(byte)
}

fn layout_specials() -> SpecialTokens {
    SpecialTokens {
        bos: 0,
        eos: 1,
        pad: 2,
        unk: 3,
        latent: None,
        latent_start: None,
        latent_end: None,
    }
}

fn arc_range() -> ArcGridTokenRange {
    ArcGridTokenRange {
        start: FIRST_ARC,
        end: FIRST_ARC + ARC_N_COLORS as u32,
    }
}

fn count_pairs(sequences: &[Vec<u32>]) -> HashMap<(u32, u32), u64> {
    let mut counts = HashMap::new();
    for seq in sequences {
        for pair in seq.windows(2) {
            *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
    }
    counts
}

fn best_pair(counts: &HashMap<(u32, u32), u64>) -> Option<(u32, u32)> {
    let mut best: Option<(u64, u32, u32)> = None;
    for (&(left, right), &count) in counts {
        let take = match best {
            None => true,
            Some((best_count, best_left, best_right)) => {
                count > best_count
                    || (count == best_count
                        && (left < best_left || (left == best_left && right < best_right)))
            }
        };
        if take {
            best = Some((count, left, right));
        }
    }
    best.map(|(_, left, right)| (left, right))
}

fn apply_merge(seq: &[u32], left: u32, right: u32, new_id: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
            out.push(new_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    out
}

fn bpe_encode(byte_ids: &[u32], merges: &[(u32, u32)]) -> Vec<u32> {
    if byte_ids.is_empty() || merges.is_empty() {
        return byte_ids.to_vec();
    }
    let rank: HashMap<(u32, u32), usize> = merges
        .iter()
        .enumerate()
        .map(|(i, pair)| (*pair, i))
        .collect();
    let mut ids = byte_ids.to_vec();
    loop {
        let mut best_rank: Option<usize> = None;
        let mut best_pos: Option<usize> = None;
        for i in 0..ids.len().saturating_sub(1) {
            if let Some(&r) = rank.get(&(ids[i], ids[i + 1])) {
                if best_rank.map(|br| r < br).unwrap_or(true) {
                    best_rank = Some(r);
                    best_pos = Some(i);
                }
            }
        }
        let Some(r) = best_rank else {
            break;
        };
        let pos = best_pos.expect("rank implies position");
        let new_id = FIRST_MERGE + r as u32;
        let mut next = Vec::with_capacity(ids.len() - 1);
        next.extend_from_slice(&ids[..pos]);
        next.push(new_id);
        next.extend_from_slice(&ids[pos + 2..]);
        ids = next;
    }
    ids
}

fn expand_token(token_id: u32, merges: &[(u32, u32)]) -> Result<Vec<u8>, Error> {
    if (FIRST_BYTE..FIRST_BYTE + N_BYTES).contains(&token_id) {
        return Ok(vec![(token_id - FIRST_BYTE) as u8]);
    }
    let Some(index) = (token_id.checked_sub(FIRST_MERGE)).and_then(|i| usize::try_from(i).ok())
    else {
        return Err(Error::UnknownId(token_id));
    };
    let Some(&(left, right)) = merges.get(index) else {
        return Err(Error::UnknownId(token_id));
    };
    let mut bytes = expand_token(left, merges)?;
    bytes.extend(expand_token(right, merges)?);
    Ok(bytes)
}

fn canonical_json(value: &Value) -> String {
    serde_json::to_string(value).expect("compact json")
}

fn artifact_bytes(
    specials: &SpecialTokens,
    merges: &[(u32, u32)],
    vocab_size: u32,
    tokenizer_id: &str,
    byte_fallback: bool,
) -> Vec<u8> {
    let mut lines = Vec::new();
    let ids = [specials.bos, specials.eos, specials.pad, specials.unk];
    for (name, id) in REQUIRED_SPECIALS.iter().zip(ids) {
        lines.push(canonical_json(
            &json!({"id": id, "kind": "special", "name": name}),
        ));
    }
    for byte in 0..=255u8 {
        lines.push(canonical_json(&json!({
            "byte": byte,
            "id": byte_id(byte),
            "kind": "byte"
        })));
    }
    for color in 0..ARC_N_COLORS as u32 {
        lines.push(canonical_json(&json!({
            "color": color,
            "id": FIRST_ARC + color,
            "kind": "arc"
        })));
    }
    for (rank, &(left, right)) in merges.iter().enumerate() {
        lines.push(canonical_json(&json!({
            "id": FIRST_MERGE + rank as u32,
            "kind": "merge",
            "left": left,
            "right": right
        })));
    }
    lines.push(canonical_json(&json!({
        "algorithm": ALGORITHM,
        "byte_fallback": byte_fallback,
        "kind": "header",
        "tokenizer_id": tokenizer_id,
        "vocab_size": vocab_size,
    })));
    let mut out = String::new();
    for line in lines {
        out.push_str(&line);
        out.push('\n');
    }
    out.into_bytes()
}

fn specials_map(specials: &SpecialTokens) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("bos".into(), json!(specials.bos));
    map.insert("eos".into(), json!(specials.eos));
    map.insert("pad".into(), json!(specials.pad));
    map.insert("unk".into(), json!(specials.unk));
    if let Some(v) = specials.latent {
        map.insert("latent".into(), json!(v));
    }
    if let Some(v) = specials.latent_start {
        map.insert("latent_start".into(), json!(v));
    }
    if let Some(v) = specials.latent_end {
        map.insert("latent_end".into(), json!(v));
    }
    map
}

fn corpus_hash(docs: &[String]) -> String {
    let mut blob = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        if i > 0 {
            blob.push(b'\n');
        }
        blob.extend_from_slice(doc.as_bytes());
    }
    sha256_hex(&blob)
}

pub struct ReferenceTokenizer {
    pub meta: TokenizerMeta,
    merges: Vec<(u32, u32)>,
    corpus_hash: Option<String>,
}

impl ReferenceTokenizer {
    pub fn train(docs: &[String], config: &TrainConfig) -> Result<Self, Error> {
        if !config.byte_fallback {
            return Err(Error::Message(
                "byte_fallback must be true for algorithm byte_fallback_bpe".into(),
            ));
        }
        if config.tokenizer_id.is_empty() {
            return Err(Error::Message("tokenizer_id must be non-empty".into()));
        }
        if config.vocab_size < MIN_VOCAB_SIZE {
            return Err(Error::VocabTooSmall(config.vocab_size));
        }

        let mut sequences: Vec<Vec<u32>> = docs
            .iter()
            .map(|doc| doc.as_bytes().iter().copied().map(byte_id).collect())
            .collect();
        let mut merges = Vec::new();
        let mut next_id = FIRST_MERGE;
        while next_id < config.vocab_size {
            let counts = count_pairs(&sequences);
            let Some((left, right)) = best_pair(&counts) else {
                break;
            };
            merges.push((left, right));
            sequences = sequences
                .iter()
                .map(|seq| apply_merge(seq, left, right, next_id))
                .collect();
            next_id += 1;
        }

        let specials = layout_specials();
        let blob = artifact_bytes(
            &specials,
            &merges,
            config.vocab_size,
            &config.tokenizer_id,
            config.byte_fallback,
        );
        let meta = TokenizerMeta {
            schema_id: SCHEMA_ID.into(),
            schema_version: SCHEMA_VERSION,
            tokenizer_id: config.tokenizer_id.clone(),
            algorithm: ALGORITHM.into(),
            vocab_size: config.vocab_size,
            byte_fallback: config.byte_fallback,
            special_token_ids: specials,
            arc_grid_token_range: arc_range(),
            artifact: Artifact {
                content_hash: sha256_hex(&blob),
                bytes: blob.len() as u64,
                path: None,
                media_type: Some(VOCAB_MEDIA_TYPE.into()),
            },
            frozen: false,
            frozen_at: None,
        };
        Ok(Self {
            meta,
            merges,
            corpus_hash: Some(corpus_hash(docs)),
        })
    }

    pub fn freeze(&mut self, frozen_at: &str) -> Result<(), Error> {
        if self.meta.frozen {
            if self.meta.frozen_at.as_deref() == Some(frozen_at) {
                return Ok(());
            }
            return Err(Error::Frozen);
        }
        self.meta.frozen = true;
        self.meta.frozen_at = Some(frozen_at.to_string());
        Ok(())
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        self.encode_bytes(text.as_bytes())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let bytes = self.decode_bytes(ids)?;
        String::from_utf8(bytes).map_err(|e| Error::Message(e.to_string()))
    }

    pub fn encode_bytes(&self, data: &[u8]) -> Result<Vec<u32>, Error> {
        let byte_ids: Vec<u32> = data.iter().copied().map(byte_id).collect();
        Ok(bpe_encode(&byte_ids, &self.merges))
    }

    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, Error> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend(expand_token(id, &self.merges)?);
        }
        Ok(out)
    }

    pub fn encode_grid(&self, cells: &[u8]) -> Result<Vec<u32>, Error> {
        let start = self.meta.arc_grid_token_range.start;
        let end = self.meta.arc_grid_token_range.end;
        let n_colors = end - start;
        let mut out = Vec::with_capacity(cells.len());
        for &cell in cells {
            if u32::from(cell) >= n_colors {
                return Err(Error::BadColor(cell));
            }
            out.push(start + u32::from(cell));
        }
        Ok(out)
    }

    pub fn tokens_per_word(&self, text: &str) -> Result<f64, Error> {
        let words = text.split_whitespace().count();
        if words == 0 {
            return Ok(0.0);
        }
        let n = self.encode(text)?.len();
        Ok(n as f64 / words as f64)
    }

    pub fn save(&mut self, dir: &Path) -> Result<VocabPointer, Error> {
        fs::create_dir_all(dir).map_err(|e| Error::Message(e.to_string()))?;
        let blob = artifact_bytes(
            &self.meta.special_token_ids,
            &self.merges,
            self.meta.vocab_size,
            &self.meta.tokenizer_id,
            self.meta.byte_fallback,
        );
        fs::write(dir.join(VOCAB_FILENAME), &blob).map_err(|e| Error::Message(e.to_string()))?;
        self.meta.artifact = Artifact {
            content_hash: sha256_hex(&blob),
            bytes: blob.len() as u64,
            path: Some(VOCAB_FILENAME.into()),
            media_type: Some(VOCAB_MEDIA_TYPE.into()),
        };
        let contract = self.to_contract();
        fs::write(
            dir.join(TOKENIZER_FILENAME),
            serde_json::to_string_pretty(&contract).map_err(|e| Error::Message(e.to_string()))?,
        )
        .map_err(|e| Error::Message(e.to_string()))?;
        Ok(VocabPointer {
            schema_id: VOCAB_SCHEMA_ID.into(),
            schema_version: SCHEMA_VERSION,
            tokenizer_id: self.meta.tokenizer_id.clone(),
            vocab_size: self.meta.vocab_size,
            special_token_ids: self.meta.special_token_ids.clone(),
            artifact: self.meta.artifact.clone(),
            format: VOCAB_FORMAT.into(),
        })
    }

    pub fn to_contract(&self) -> Value {
        let mut payload = json!({
            "schema_id": self.meta.schema_id,
            "schema_version": self.meta.schema_version,
            "tokenizer_id": self.meta.tokenizer_id,
            "algorithm": self.meta.algorithm,
            "vocab_size": self.meta.vocab_size,
            "byte_fallback": self.meta.byte_fallback,
            "special_tokens": Value::Object(specials_map(&self.meta.special_token_ids)),
            "arc_grid_token_range": {
                "start": self.meta.arc_grid_token_range.start,
                "end": self.meta.arc_grid_token_range.end,
            },
            "vocab_hash": self.meta.artifact.content_hash,
            "frozen": self.meta.frozen,
        });
        if let Some(ts) = &self.meta.frozen_at {
            payload
                .as_object_mut()
                .expect("object")
                .insert("frozen_at".into(), json!(ts));
        }
        if let Some(h) = &self.corpus_hash {
            payload
                .as_object_mut()
                .expect("object")
                .insert("corpus_hash".into(), json!(h));
        }
        payload
    }
}
