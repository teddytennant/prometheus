"""Implementer-facing tests for the F6 tokenizer public API.

These import production ``tokenizer`` (train / encode / freeze / save / load
are ``NotImplementedError`` today) and must FAIL against that stub. They must
PASS once production matches ``tests.reference.tokenizer``.

Groups:
- Layout and VocabTooSmall: specials 0-3, 256 byte ids, ARC [260, 270),
  min vocab 271, byte_fallback required.
- Deterministic train: same docs+config => same encodings and vocab hash.
- Golden id sequences locked from the reference on a fixed tiny corpus.
- Property: production encode/decode matches the reference (UTF-8 round-trip,
  arbitrary bytes including invalid UTF-8).
- ARC grid: one distinct id per color 0-9 inside the reserved range; OOB error.
- F6 gate: tokens_per_word strictly below baseline_byte_tokens_per_word on a
  repeated-word sample after merges fire.
- Freeze: frozen flag, stable artifact hash, encode still works, re-freeze
  with a different timestamp errors.
- save/load: tokenizer.json + vocab artifact, content_hash is SHA-256 of the
  artifact bytes, contracts.validate on prometheus.tokenizer and prometheus.vocab.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import contracts
from tests.reference.tokenizer import (
    FIRST_ARC,
    FIRST_MERGE,
    MIN_VOCAB_SIZE,
    N_BYTES,
    VOCAB_FILENAME,
    ReferenceTokenizer,
)
from tokenizer import (
    ALGORITHM,
    ARC_N_COLORS,
    PRODUCTION_VOCAB_SIZE,
    SCHEMA_ID,
    SCHEMA_VERSION,
    VOCAB_SCHEMA_ID,
    Tokenizer,
    TokenizerError,
    TrainConfig,
    baseline_byte_tokens_per_word,
)

ROOT = Path(__file__).resolve().parents[1]
B1_GOLDEN_DIR = ROOT / "data" / "extract" / "tests" / "goldens"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "tokenizer"

GOLDEN_CORPUS = [
    "hello hello hello world",
    "hello world hello",
    "the cat sat on the mat",
    "the cat sat",
]
GOLDEN_CONFIG = TrainConfig(
    tokenizer_id="tiny-bpe-test", vocab_size=320, byte_fallback=True
)
FROZEN_AT = "2026-09-16T12:00:00Z"

# Locked from tests.reference.tokenizer on GOLDEN_CORPUS + GOLDEN_CONFIG.
GOLDEN_HELLO = [274]
GOLDEN_HELLO_HELLO_WORLD = [283, 282, 280]
GOLDEN_THE_CAT = [278, 103, 271]
GOLDEN_EMPTY: list[int] = []
GOLDEN_GRID = [260, 261, 262, 269]
GOLDEN_INVALID_UTF8 = [259, 258, 4, 274]
GOLDEN_TPW_HELLO_HELLO_WORLD = 1.0

GATE_SAMPLE = "hello hello hello world world hello world"


def _b1_docs() -> list[str]:
    docs = [
        path.read_text(encoding="utf-8")
        for path in sorted(B1_GOLDEN_DIR.glob("*.expected.txt"))
    ]
    assert docs, "B1 extract goldens missing"
    docs.append((FIXTURE_DIR / "repeated_english.txt").read_text(encoding="utf-8"))
    docs.append((FIXTURE_DIR / "utf8_mix.txt").read_text(encoding="utf-8"))
    docs.extend(GOLDEN_CORPUS)
    return docs


def _cfg(
    tokenizer_id: str = "tiny-bpe-test",
    vocab_size: int = 320,
    byte_fallback: bool = True,
) -> TrainConfig:
    return TrainConfig(
        tokenizer_id=tokenizer_id, vocab_size=vocab_size, byte_fallback=byte_fallback
    )


def _specials_payload(specials: object) -> dict[str, int]:
    payload: dict[str, int] = {}
    for name in (
        "bos",
        "eos",
        "pad",
        "unk",
        "latent",
        "latent_start",
        "latent_end",
    ):
        value = getattr(specials, name)
        if value is not None:
            payload[name] = int(value)
    return payload


def _vocab_contract(pointer: object) -> dict:
    artifact = pointer.artifact
    payload = {
        "schema_id": pointer.schema_id,
        "schema_version": pointer.schema_version,
        "tokenizer_id": pointer.tokenizer_id,
        "vocab_size": pointer.vocab_size,
        "special_token_ids": _specials_payload(pointer.special_token_ids),
        "artifact": {
            "content_hash": artifact.content_hash,
            "bytes": artifact.bytes,
        },
        "format": pointer.format,
    }
    if artifact.path is not None:
        payload["artifact"]["path"] = artifact.path
    if artifact.media_type is not None:
        payload["artifact"]["media_type"] = artifact.media_type
    return payload


# ---------------------------------------------------------------------------
# Layout / VocabTooSmall
# ---------------------------------------------------------------------------


def test_train_rejects_vocab_too_small_to_fit_specials_bytes_and_arc() -> None:
    with pytest.raises(TokenizerError):
        Tokenizer.train(["hello"], _cfg(vocab_size=10))


def test_train_rejects_vocab_without_a_merge_slot() -> None:
    assert MIN_VOCAB_SIZE == 4 + N_BYTES + ARC_N_COLORS + 1
    with pytest.raises(TokenizerError):
        Tokenizer.train(["hello"], _cfg(vocab_size=FIRST_MERGE))


def test_train_rejects_byte_fallback_false() -> None:
    with pytest.raises(TokenizerError):
        Tokenizer.train(["hello"], _cfg(byte_fallback=False))


def test_trained_layout_matches_reference_specials_bytes_and_arc_range() -> None:
    prod = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    ref = ReferenceTokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    assert prod.meta.algorithm == ALGORITHM == "byte_fallback_bpe"
    assert prod.meta.vocab_size == GOLDEN_CONFIG.vocab_size
    assert prod.meta.byte_fallback is True
    assert prod.meta.frozen is False
    assert prod.meta.special_token_ids.bos == 0
    assert prod.meta.special_token_ids.eos == 1
    assert prod.meta.special_token_ids.pad == 2
    assert prod.meta.special_token_ids.unk == 3
    assert prod.meta.arc_grid_token_range.start == FIRST_ARC == 260
    assert prod.meta.arc_grid_token_range.end == FIRST_ARC + ARC_N_COLORS == 270
    assert prod.meta.special_token_ids.bos == ref.meta.special_token_ids.bos
    assert prod.meta.arc_grid_token_range.start == ref.meta.arc_grid_token_range.start
    assert PRODUCTION_VOCAB_SIZE == 256_000
    assert 300 <= GOLDEN_CONFIG.vocab_size <= 512


# ---------------------------------------------------------------------------
# Deterministic train
# ---------------------------------------------------------------------------


def test_two_trains_on_the_same_docs_and_config_match() -> None:
    a = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    b = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    text = "hello hello world"
    assert a.encode(text) == b.encode(text)
    assert a.meta.artifact.content_hash == b.meta.artifact.content_hash
    assert len(a.meta.artifact.content_hash) == 64
    assert a.meta.artifact.content_hash == a.meta.artifact.content_hash.lower()


# ---------------------------------------------------------------------------
# Golden id sequences (locked from the reference)
# ---------------------------------------------------------------------------


def test_golden_encode_sequences_on_fixed_tiny_corpus() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    assert tok.encode("hello") == GOLDEN_HELLO
    assert tok.encode("hello hello world") == GOLDEN_HELLO_HELLO_WORLD
    assert tok.encode("the cat") == GOLDEN_THE_CAT
    assert tok.encode("") == GOLDEN_EMPTY
    assert tok.encode_bytes(b"\xff\xfe\x00hello") == GOLDEN_INVALID_UTF8
    assert tok.encode_grid([0, 1, 2, 9]) == GOLDEN_GRID
    assert tok.tokens_per_word("hello hello world") == GOLDEN_TPW_HELLO_HELLO_WORLD


def test_golden_sequences_match_independent_reference() -> None:
    ref = ReferenceTokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    assert ref.encode("hello") == GOLDEN_HELLO
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    assert tok.encode("hello") == ref.encode("hello")
    assert tok.encode("hello hello world") == ref.encode("hello hello world")
    assert tok.encode("the cat") == ref.encode("the cat")
    assert tok.encode_bytes(b"\xff\xfe\x00hello") == ref.encode_bytes(
        b"\xff\xfe\x00hello"
    )


# ---------------------------------------------------------------------------
# Encode / decode properties vs reference
# ---------------------------------------------------------------------------


ROUND_TRIP_TEXTS = [
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
]


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_utf8_roundtrip_matches_reference(text: str) -> None:
    docs = _b1_docs()
    cfg = _cfg(tokenizer_id="v0-sample", vocab_size=384)
    ref = ReferenceTokenizer.train(docs, cfg)
    prod = Tokenizer.train(docs, cfg)
    assert prod.encode(text) == ref.encode(text)
    assert prod.decode(prod.encode(text)) == text
    for token_id in prod.encode(text):
        assert 0 <= token_id < prod.meta.vocab_size


def test_encode_bytes_roundtrip_invalid_utf8_matches_reference() -> None:
    payloads = [
        b"",
        b"hello",
        b"\xff\xfe\x00\x80\xbf",
        bytes(range(256)),
        "café".encode(),
        "🧊".encode(),
    ]
    cfg = GOLDEN_CONFIG
    ref = ReferenceTokenizer.train(GOLDEN_CORPUS, cfg)
    prod = Tokenizer.train(GOLDEN_CORPUS, cfg)
    for data in payloads:
        assert prod.encode_bytes(data) == ref.encode_bytes(data)
        assert prod.decode_bytes(prod.encode_bytes(data)) == data


def test_decode_unknown_id_and_specials_are_errors() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    with pytest.raises(TokenizerError):
        tok.decode([0])
    with pytest.raises(TokenizerError):
        tok.decode([tok.meta.vocab_size + 5])
    with pytest.raises(TokenizerError):
        tok.decode_bytes([FIRST_ARC])


# ---------------------------------------------------------------------------
# ARC grid
# ---------------------------------------------------------------------------


def test_encode_grid_one_distinct_id_per_color_inside_reserved_range() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    start = tok.meta.arc_grid_token_range.start
    end = tok.meta.arc_grid_token_range.end
    assert end - start == ARC_N_COLORS == 10
    cells = list(range(ARC_N_COLORS))
    ids = tok.encode_grid(cells)
    assert ids == list(range(start, end))
    assert len(set(ids)) == ARC_N_COLORS
    assert tok.encode_grid([]) == []
    # Row-major flatten only: tokenizer does not emit 2D structure (spec 10).
    assert tok.encode_grid([1, 2, 3, 0]) == [start + 1, start + 2, start + 3, start]


def test_encode_grid_out_of_range_color_errors() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    with pytest.raises(TokenizerError):
        tok.encode_grid([10])
    with pytest.raises(TokenizerError):
        tok.encode_grid([255])
    with pytest.raises(TokenizerError):
        tok.encode_grid([-1])
    with pytest.raises(TokenizerError):
        tok.encode_grid([0, 9, 10])


def test_encode_grid_matches_reference() -> None:
    ref = ReferenceTokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    prod = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    cells = [0, 4, 9, 1, 1, 8]
    assert prod.encode_grid(cells) == ref.encode_grid(cells) == [
        FIRST_ARC + c for c in cells
    ]


# ---------------------------------------------------------------------------
# F6 gate: tokens/word vs byte baseline
# ---------------------------------------------------------------------------


def test_tokens_per_word_beats_byte_baseline_when_merges_exist() -> None:
    docs = _b1_docs()
    cfg = _cfg(tokenizer_id="v0-sample", vocab_size=384)
    tok = Tokenizer.train(docs, cfg)
    ids = tok.encode(GATE_SAMPLE)
    has_merges = any(token_id >= FIRST_MERGE for token_id in ids)
    assert has_merges, "repeated-word corpus must produce BPE merges"
    tpw = tok.tokens_per_word(GATE_SAMPLE)
    baseline = baseline_byte_tokens_per_word(GATE_SAMPLE)
    assert tpw < baseline
    ref = ReferenceTokenizer.train(docs, cfg)
    assert tok.tokens_per_word(GATE_SAMPLE) == ref.tokens_per_word(GATE_SAMPLE)


def test_tokens_per_word_empty_and_single_word() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    assert tok.tokens_per_word("") == 0.0
    assert tok.tokens_per_word("   \n\t") == 0.0
    hello_tpw = tok.tokens_per_word("hello")
    assert hello_tpw == float(len(tok.encode("hello")))


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def test_freeze_sets_flag_keeps_hash_and_encode_works() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    before = tok.encode("hello")
    hash_before = tok.meta.artifact.content_hash
    tok.freeze(FROZEN_AT)
    assert tok.meta.frozen is True
    assert tok.meta.frozen_at == FROZEN_AT
    assert tok.meta.artifact.content_hash == hash_before
    assert tok.encode("hello") == before
    tok.freeze(FROZEN_AT)  # idempotent at the same timestamp
    assert tok.meta.frozen_at == FROZEN_AT


def test_freeze_different_timestamp_is_error() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    tok.freeze(FROZEN_AT)
    with pytest.raises(TokenizerError):
        tok.freeze("2026-09-16T13:00:00Z")
    assert tok.encode("hello") == GOLDEN_HELLO


# ---------------------------------------------------------------------------
# save / load / contracts
# ---------------------------------------------------------------------------


def test_to_contract_validates_as_prometheus_tokenizer() -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    payload = tok.to_contract()
    assert payload["schema_id"] == SCHEMA_ID == "prometheus.tokenizer"
    assert payload["schema_version"] == SCHEMA_VERSION == 1
    assert payload["algorithm"] == ALGORITHM
    assert payload["vocab_size"] == GOLDEN_CONFIG.vocab_size
    assert payload["byte_fallback"] is True
    assert payload["frozen"] is False
    assert payload["special_tokens"]["bos"] == 0
    assert payload["special_tokens"]["eos"] == 1
    assert payload["special_tokens"]["pad"] == 2
    assert payload["special_tokens"]["unk"] == 3
    assert payload["arc_grid_token_range"] == {"start": FIRST_ARC, "end": FIRST_ARC + 10}
    assert payload["vocab_hash"] == tok.meta.artifact.content_hash
    contracts.validate(payload)
    tok.freeze(FROZEN_AT)
    frozen = tok.to_contract()
    assert frozen["frozen"] is True
    assert frozen["frozen_at"] == FROZEN_AT
    contracts.validate(frozen)


def test_save_load_roundtrip_and_artifact_hash(tmp_path: Path) -> None:
    tok = Tokenizer.train(GOLDEN_CORPUS, GOLDEN_CONFIG)
    tok.freeze(FROZEN_AT)
    before = tok.encode("hello hello world")
    pointer = tok.save(tmp_path)
    tokenizer_json = tmp_path / "tokenizer.json"
    assert tokenizer_json.is_file()
    saved = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    assert saved["frozen"] is True
    assert saved["schema_id"] == SCHEMA_ID
    assert saved["schema_version"] == SCHEMA_VERSION
    contracts.validate(saved)

    vocab_name = pointer.artifact.path or VOCAB_FILENAME
    vocab_file = tmp_path / vocab_name
    assert vocab_file.is_file()
    blob = vocab_file.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    assert pointer.artifact.content_hash == digest
    assert pointer.artifact.bytes == len(blob)
    assert pointer.schema_id == VOCAB_SCHEMA_ID
    assert pointer.format in {"jsonl", "sentencepiece", "tiktoken"}
    contracts.validate(_vocab_contract(pointer))

    loaded = Tokenizer.load(tmp_path)
    assert loaded.meta.frozen is True
    assert loaded.meta.frozen_at == FROZEN_AT
    assert loaded.encode("hello hello world") == before
    assert loaded.decode(loaded.encode("café naïve")) == "café naïve"
    assert loaded.encode_grid([0, 9]) == [FIRST_ARC, FIRST_ARC + 9]


def test_load_missing_directory_errors(tmp_path: Path) -> None:
    with pytest.raises(TokenizerError):
        Tokenizer.load(tmp_path / "does-not-exist")
