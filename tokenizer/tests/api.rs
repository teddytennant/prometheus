//! Implementer-facing tests for `prometheus-tokenizer`.
//!
//! Every test calls the production API (`Tokenizer::train` / encode / freeze /
//! save / load), which is `unimplemented!` today, so `cargo test` must be red.
//! After F6 is implemented these must match `common::reference`.

mod common;

use std::path::Path;

use prometheus_contracts::validate;
use prometheus_tokenizer::{
    baseline_byte_tokens_per_word, ALGORITHM, ARC_N_COLORS, PRODUCTION_VOCAB_SIZE, SCHEMA_ID,
    SCHEMA_VERSION, VOCAB_SCHEMA_ID, Error, Tokenizer, TrainConfig,
};
use serde_json::{json, Map, Value};

use common::reference::{
    ReferenceTokenizer, FIRST_ARC, FIRST_MERGE, MIN_VOCAB_SIZE, N_BYTES, VOCAB_FILENAME,
};

const FROZEN_AT: &str = "2026-09-16T12:00:00Z";
const GATE_SAMPLE: &str = "hello hello hello world world hello world";

const GOLDEN_HELLO: [u32; 1] = [274];
const GOLDEN_HELLO_HELLO_WORLD: [u32; 3] = [283, 282, 280];
const GOLDEN_THE_CAT: [u32; 3] = [278, 103, 271];
const GOLDEN_INVALID_UTF8: [u32; 4] = [259, 258, 4, 274];
const GOLDEN_GRID: [u32; 4] = [260, 261, 262, 269];

fn golden_corpus() -> Vec<String> {
    vec![
        "hello hello hello world".into(),
        "hello world hello".into(),
        "the cat sat on the mat".into(),
        "the cat sat".into(),
    ]
}

fn golden_config() -> TrainConfig {
    TrainConfig {
        tokenizer_id: "tiny-bpe-test".into(),
        vocab_size: 320,
        byte_fallback: true,
    }
}

fn v0_docs() -> Vec<String> {
    let mut docs = vec![
        include_str!("../../data/extract/tests/goldens/math.expected.txt").to_string(),
        include_str!("../../data/extract/tests/goldens/page.expected.txt").to_string(),
        include_str!("../../data/extract/tests/goldens/tiny.expected.txt").to_string(),
        include_str!("../../tests/fixtures/tokenizer/repeated_english.txt").to_string(),
        include_str!("../../tests/fixtures/tokenizer/utf8_mix.txt").to_string(),
    ];
    docs.extend(golden_corpus());
    docs
}

fn v0_config() -> TrainConfig {
    TrainConfig {
        tokenizer_id: "v0-sample".into(),
        vocab_size: 384,
        byte_fallback: true,
    }
}

fn specials_json(s: &prometheus_tokenizer::SpecialTokens) -> Value {
    let mut map = Map::new();
    map.insert("bos".into(), json!(s.bos));
    map.insert("eos".into(), json!(s.eos));
    map.insert("pad".into(), json!(s.pad));
    map.insert("unk".into(), json!(s.unk));
    if let Some(v) = s.latent {
        map.insert("latent".into(), json!(v));
    }
    if let Some(v) = s.latent_start {
        map.insert("latent_start".into(), json!(v));
    }
    if let Some(v) = s.latent_end {
        map.insert("latent_end".into(), json!(v));
    }
    Value::Object(map)
}

fn vocab_contract(pointer: &prometheus_tokenizer::VocabPointer) -> Value {
    let mut artifact = json!({
        "content_hash": pointer.artifact.content_hash,
        "bytes": pointer.artifact.bytes,
    });
    if let Some(path) = &pointer.artifact.path {
        artifact
            .as_object_mut()
            .unwrap()
            .insert("path".into(), json!(path));
    }
    if let Some(mt) = &pointer.artifact.media_type {
        artifact
            .as_object_mut()
            .unwrap()
            .insert("media_type".into(), json!(mt));
    }
    json!({
        "schema_id": pointer.schema_id,
        "schema_version": pointer.schema_version,
        "tokenizer_id": pointer.tokenizer_id,
        "vocab_size": pointer.vocab_size,
        "special_token_ids": specials_json(&pointer.special_token_ids),
        "artifact": artifact,
        "format": pointer.format,
    })
}

// ---------------------------------------------------------------------------
// Layout / VocabTooSmall
// ---------------------------------------------------------------------------

#[test]
fn train_rejects_vocab_too_small_to_fit_specials_bytes_and_arc() {
    let err = Tokenizer::train(["hello"], &TrainConfig {
        tokenizer_id: "tiny-bpe-test".into(),
        vocab_size: 10,
        byte_fallback: true,
    })
    .unwrap_err();
    assert!(matches!(err, Error::VocabTooSmall(10)));
}

#[test]
fn train_rejects_vocab_without_a_merge_slot() {
    assert_eq!(MIN_VOCAB_SIZE, 4u32 + N_BYTES + ARC_N_COLORS + 1);
    let err = Tokenizer::train(["hello"], &TrainConfig {
        tokenizer_id: "tiny-bpe-test".into(),
        vocab_size: FIRST_MERGE,
        byte_fallback: true,
    })
    .unwrap_err();
    assert!(matches!(err, Error::VocabTooSmall(_)));
}

#[test]
fn train_rejects_byte_fallback_false() {
    let err = Tokenizer::train(["hello"], &TrainConfig {
        tokenizer_id: "tiny-bpe-test".into(),
        vocab_size: 320,
        byte_fallback: false,
    });
    assert!(err.is_err());
}

#[test]
fn trained_layout_matches_reference_specials_bytes_and_arc_range() {
    let docs = golden_corpus();
    let cfg = golden_config();
    let prod = Tokenizer::train(&docs, &cfg).unwrap();
    let reference = ReferenceTokenizer::train(&docs, &cfg).unwrap();
    assert_eq!(prod.meta.algorithm, ALGORITHM);
    assert_eq!(ALGORITHM, "byte_fallback_bpe");
    assert_eq!(prod.meta.vocab_size, 320);
    assert!(prod.meta.byte_fallback);
    assert!(!prod.meta.frozen);
    assert_eq!(prod.meta.special_token_ids.bos, 0);
    assert_eq!(prod.meta.special_token_ids.eos, 1);
    assert_eq!(prod.meta.special_token_ids.pad, 2);
    assert_eq!(prod.meta.special_token_ids.unk, 3);
    assert_eq!(prod.meta.arc_grid_token_range.start, FIRST_ARC);
    assert_eq!(prod.meta.arc_grid_token_range.end, FIRST_ARC + ARC_N_COLORS);
    assert_eq!(
        prod.meta.special_token_ids.bos,
        reference.meta.special_token_ids.bos
    );
    assert_eq!(
        prod.meta.arc_grid_token_range.start,
        reference.meta.arc_grid_token_range.start
    );
    assert_eq!(PRODUCTION_VOCAB_SIZE, 256_000);
}

// ---------------------------------------------------------------------------
// Deterministic train
// ---------------------------------------------------------------------------

#[test]
fn two_trains_on_the_same_docs_and_config_match() {
    let docs = golden_corpus();
    let cfg = golden_config();
    let a = Tokenizer::train(&docs, &cfg).unwrap();
    let b = Tokenizer::train(&docs, &cfg).unwrap();
    let text = "hello hello world";
    assert_eq!(a.encode(text).unwrap(), b.encode(text).unwrap());
    assert_eq!(a.meta.artifact.content_hash, b.meta.artifact.content_hash);
    assert_eq!(a.meta.artifact.content_hash.len(), 64);
    assert_eq!(
        a.meta.artifact.content_hash,
        a.meta.artifact.content_hash.to_lowercase()
    );
}

// ---------------------------------------------------------------------------
// Golden id sequences
// ---------------------------------------------------------------------------

#[test]
fn golden_encode_sequences_on_fixed_tiny_corpus() {
    let tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    assert_eq!(tok.encode("hello").unwrap(), GOLDEN_HELLO);
    assert_eq!(
        tok.encode("hello hello world").unwrap(),
        GOLDEN_HELLO_HELLO_WORLD
    );
    assert_eq!(tok.encode("the cat").unwrap(), GOLDEN_THE_CAT);
    assert_eq!(tok.encode("").unwrap(), Vec::<u32>::new());
    assert_eq!(
        tok.encode_bytes(b"\xff\xfe\x00hello").unwrap(),
        GOLDEN_INVALID_UTF8
    );
    assert_eq!(tok.encode_grid(&[0, 1, 2, 9]).unwrap(), GOLDEN_GRID);
    assert_eq!(tok.tokens_per_word("hello hello world").unwrap(), 1.0);
}

#[test]
fn golden_sequences_match_independent_reference() {
    let docs = golden_corpus();
    let cfg = golden_config();
    let reference = ReferenceTokenizer::train(&docs, &cfg).unwrap();
    assert_eq!(reference.encode("hello").unwrap(), GOLDEN_HELLO);
    let tok = Tokenizer::train(&docs, &cfg).unwrap();
    assert_eq!(tok.encode("hello").unwrap(), reference.encode("hello").unwrap());
    assert_eq!(
        tok.encode("hello hello world").unwrap(),
        reference.encode("hello hello world").unwrap()
    );
    assert_eq!(
        tok.encode("the cat").unwrap(),
        reference.encode("the cat").unwrap()
    );
    assert_eq!(
        tok.encode_bytes(b"\xff\xfe\x00hello").unwrap(),
        reference.encode_bytes(b"\xff\xfe\x00hello").unwrap()
    );
}

// ---------------------------------------------------------------------------
// Encode / decode properties vs reference
// ---------------------------------------------------------------------------

#[test]
fn utf8_roundtrip_matches_reference() {
    let docs = v0_docs();
    let cfg = v0_config();
    let reference = ReferenceTokenizer::train(&docs, &cfg).unwrap();
    let prod = Tokenizer::train(&docs, &cfg).unwrap();
    let texts = [
        "",
        "hello",
        "hello hello world",
        "the cat sat on the mat",
        "café naïve résumé",
        "日本語テスト漢字かな",
        "emoji 🧊 snow",
        "spaces   and\ttabs\nnewlines",
        "Hello PDF",
        "Math stays as $x^2$.",
    ];
    for text in texts {
        let encoded = prod.encode(text).unwrap();
        assert_eq!(encoded, reference.encode(text).unwrap());
        assert_eq!(prod.decode(&encoded).unwrap(), text);
        for id in &encoded {
            assert!(*id < prod.meta.vocab_size);
        }
    }
}

#[test]
fn encode_bytes_roundtrip_invalid_utf8_matches_reference() {
    let docs = golden_corpus();
    let cfg = golden_config();
    let reference = ReferenceTokenizer::train(&docs, &cfg).unwrap();
    let prod = Tokenizer::train(&docs, &cfg).unwrap();
    let all: Vec<u8> = (0..=255).collect();
    let payloads: Vec<&[u8]> = vec![b"", b"hello", b"\xff\xfe\x00\x80\xbf", all.as_slice()];
    for data in payloads {
        let encoded = prod.encode_bytes(data).unwrap();
        assert_eq!(encoded, reference.encode_bytes(data).unwrap());
        assert_eq!(prod.decode_bytes(&encoded).unwrap(), data);
    }
}

#[test]
fn decode_unknown_id_and_specials_are_errors() {
    let tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    assert!(matches!(tok.decode(&[0]).unwrap_err(), Error::UnknownId(0)));
    let unknown = tok.meta.vocab_size + 5;
    assert!(matches!(
        tok.decode(&[unknown]).unwrap_err(),
        Error::UnknownId(_)
    ));
    assert!(matches!(
        tok.decode_bytes(&[FIRST_ARC]).unwrap_err(),
        Error::UnknownId(_)
    ));
}

// ---------------------------------------------------------------------------
// ARC grid
// ---------------------------------------------------------------------------

#[test]
fn encode_grid_one_distinct_id_per_color_inside_reserved_range() {
    let tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    let start = tok.meta.arc_grid_token_range.start;
    let end = tok.meta.arc_grid_token_range.end;
    assert_eq!(end - start, ARC_N_COLORS);
    let cells: Vec<u8> = (0..ARC_N_COLORS as u8).collect();
    let ids = tok.encode_grid(&cells).unwrap();
    let expected: Vec<u32> = (start..end).collect();
    assert_eq!(ids, expected);
    let unique: std::collections::BTreeSet<_> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ARC_N_COLORS as usize);
    assert!(tok.encode_grid(&[]).unwrap().is_empty());
    assert_eq!(
        tok.encode_grid(&[1, 2, 3, 0]).unwrap(),
        vec![start + 1, start + 2, start + 3, start]
    );
}

#[test]
fn encode_grid_out_of_range_color_errors() {
    let tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    assert!(matches!(
        tok.encode_grid(&[10]).unwrap_err(),
        Error::BadColor(10)
    ));
    assert!(matches!(
        tok.encode_grid(&[255]).unwrap_err(),
        Error::BadColor(255)
    ));
    assert!(matches!(
        tok.encode_grid(&[0, 9, 10]).unwrap_err(),
        Error::BadColor(10)
    ));
}

#[test]
fn encode_grid_matches_reference() {
    let docs = golden_corpus();
    let cfg = golden_config();
    let reference = ReferenceTokenizer::train(&docs, &cfg).unwrap();
    let prod = Tokenizer::train(&docs, &cfg).unwrap();
    let cells = [0u8, 4, 9, 1, 1, 8];
    let expected: Vec<u32> = cells.iter().map(|c| FIRST_ARC + u32::from(*c)).collect();
    assert_eq!(prod.encode_grid(&cells).unwrap(), expected);
    assert_eq!(
        prod.encode_grid(&cells).unwrap(),
        reference.encode_grid(&cells).unwrap()
    );
}

// ---------------------------------------------------------------------------
// F6 gate
// ---------------------------------------------------------------------------

#[test]
fn tokens_per_word_beats_byte_baseline_when_merges_exist() {
    let docs = v0_docs();
    let cfg = v0_config();
    let tok = Tokenizer::train(&docs, &cfg).unwrap();
    let ids = tok.encode(GATE_SAMPLE).unwrap();
    let has_merges = ids.iter().any(|id| *id >= FIRST_MERGE);
    assert!(has_merges, "repeated-word corpus must produce BPE merges");
    let tpw = tok.tokens_per_word(GATE_SAMPLE).unwrap();
    let baseline = baseline_byte_tokens_per_word(GATE_SAMPLE);
    assert!(tpw < baseline, "tpw={tpw} baseline={baseline}");
    let reference = ReferenceTokenizer::train(&docs, &cfg).unwrap();
    assert_eq!(
        tok.tokens_per_word(GATE_SAMPLE).unwrap(),
        reference.tokens_per_word(GATE_SAMPLE).unwrap()
    );
}

#[test]
fn tokens_per_word_empty_and_single_word() {
    let tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    assert_eq!(tok.tokens_per_word("").unwrap(), 0.0);
    assert_eq!(tok.tokens_per_word("   \n\t").unwrap(), 0.0);
    let hello = tok.tokens_per_word("hello").unwrap();
    assert_eq!(hello, tok.encode("hello").unwrap().len() as f64);
}

// ---------------------------------------------------------------------------
// Freeze
// ---------------------------------------------------------------------------

#[test]
fn freeze_sets_flag_keeps_hash_and_encode_works() {
    let mut tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    let before = tok.encode("hello").unwrap();
    let hash_before = tok.meta.artifact.content_hash.clone();
    tok.freeze(FROZEN_AT).unwrap();
    assert!(tok.meta.frozen);
    assert_eq!(tok.meta.frozen_at.as_deref(), Some(FROZEN_AT));
    assert_eq!(tok.meta.artifact.content_hash, hash_before);
    assert_eq!(tok.encode("hello").unwrap(), before);
    tok.freeze(FROZEN_AT).unwrap();
    assert_eq!(tok.meta.frozen_at.as_deref(), Some(FROZEN_AT));
}

#[test]
fn freeze_different_timestamp_is_error() {
    let mut tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    tok.freeze(FROZEN_AT).unwrap();
    assert!(matches!(
        tok.freeze("2026-09-16T13:00:00Z").unwrap_err(),
        Error::Frozen
    ));
    assert_eq!(tok.encode("hello").unwrap(), GOLDEN_HELLO);
}

// ---------------------------------------------------------------------------
// save / load / contracts
// ---------------------------------------------------------------------------

#[test]
fn to_contract_validates_as_prometheus_tokenizer() {
    let mut tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    let payload = tok.to_contract();
    assert_eq!(payload["schema_id"], SCHEMA_ID);
    assert_eq!(SCHEMA_ID, "prometheus.tokenizer");
    assert_eq!(payload["schema_version"], json!(SCHEMA_VERSION));
    assert_eq!(payload["algorithm"], ALGORITHM);
    assert_eq!(payload["vocab_size"], json!(320));
    assert_eq!(payload["byte_fallback"], json!(true));
    assert_eq!(payload["frozen"], json!(false));
    assert_eq!(payload["special_tokens"]["bos"], json!(0));
    assert_eq!(payload["special_tokens"]["eos"], json!(1));
    assert_eq!(payload["special_tokens"]["pad"], json!(2));
    assert_eq!(payload["special_tokens"]["unk"], json!(3));
    assert_eq!(payload["arc_grid_token_range"]["start"], json!(FIRST_ARC));
    assert_eq!(
        payload["arc_grid_token_range"]["end"],
        json!(FIRST_ARC + 10)
    );
    assert_eq!(
        payload["vocab_hash"],
        json!(tok.meta.artifact.content_hash.as_str())
    );
    validate(&payload).expect("unfrozen tokenizer contract");
    tok.freeze(FROZEN_AT).unwrap();
    let frozen = tok.to_contract();
    assert_eq!(frozen["frozen"], json!(true));
    assert_eq!(frozen["frozen_at"], json!(FROZEN_AT));
    validate(&frozen).expect("frozen tokenizer contract");
}

#[test]
fn save_load_roundtrip_and_artifact_hash() {
    let mut tok = Tokenizer::train(&golden_corpus(), &golden_config()).unwrap();
    tok.freeze(FROZEN_AT).unwrap();
    let before = tok.encode("hello hello world").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let pointer = tok.save(dir.path()).unwrap();

    let tokenizer_json = dir.path().join("tokenizer.json");
    assert!(tokenizer_json.is_file());
    let saved: Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer_json).unwrap()).unwrap();
    assert_eq!(saved["frozen"], json!(true));
    assert_eq!(saved["schema_id"], SCHEMA_ID);
    assert_eq!(saved["schema_version"], json!(SCHEMA_VERSION));
    validate(&saved).expect("saved tokenizer.json");

    let vocab_name = pointer
        .artifact
        .path
        .as_deref()
        .unwrap_or(VOCAB_FILENAME);
    let vocab_file = dir.path().join(vocab_name);
    assert!(vocab_file.is_file());
    let blob = std::fs::read(&vocab_file).unwrap();
    let digest = common::reference::sha256_hex(&blob);
    assert_eq!(pointer.artifact.content_hash, digest);
    assert_eq!(pointer.artifact.bytes, blob.len() as u64);
    assert_eq!(pointer.schema_id, VOCAB_SCHEMA_ID);
    assert!(
        pointer.format == "jsonl"
            || pointer.format == "sentencepiece"
            || pointer.format == "tiktoken"
    );
    validate(&vocab_contract(&pointer)).expect("vocab pointer");

    let loaded = Tokenizer::load(dir.path()).unwrap();
    assert!(loaded.meta.frozen);
    assert_eq!(loaded.meta.frozen_at.as_deref(), Some(FROZEN_AT));
    assert_eq!(loaded.encode("hello hello world").unwrap(), before);
    assert_eq!(
        loaded.decode(&loaded.encode("café naïve").unwrap()).unwrap(),
        "café naïve"
    );
    assert_eq!(
        loaded.encode_grid(&[0, 9]).unwrap(),
        vec![FIRST_ARC, FIRST_ARC + 9]
    );
}

#[test]
fn load_missing_directory_errors() {
    let err = Tokenizer::load(Path::new("/no/such/prometheus-tokenizer-dir")).unwrap_err();
    assert!(matches!(err, Error::Message(_)));
}
