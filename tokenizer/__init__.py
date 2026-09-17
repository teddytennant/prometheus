"""Byte-fallback BPE tokenizer (spec 3.1, 7.1, 10, 15.5 F6).

Trained on a data v0 sample (B1 extract documents). The frozen artifact
matches F1 ``prometheus.tokenizer`` plus ``prometheus.vocab``. A vocab
change is a new model. Production vocab is 256_000 (spec 3.1); tests pass
a smaller ``TrainConfig.vocab_size``.

Algorithm is ``byte_fallback_bpe``. Specials bos, eos, pad, unk are
required; latent, latent_start, latent_end are optional. ARC grids use
one reserved token per cell color 0-9.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

SCHEMA_ID = "prometheus.tokenizer"
VOCAB_SCHEMA_ID = "prometheus.vocab"
SCHEMA_VERSION = 1
ALGORITHM = "byte_fallback_bpe"
PRODUCTION_VOCAB_SIZE = 256_000
ARC_N_COLORS = 10

_N_SPECIALS = 4
_N_BYTES = 256
_FIRST_BYTE = _N_SPECIALS
_FIRST_ARC = _N_SPECIALS + _N_BYTES
_FIRST_MERGE = _FIRST_ARC + ARC_N_COLORS
_MIN_VOCAB_SIZE = _FIRST_MERGE + 1

_VOCAB_FILENAME = "vocab.jsonl"
_TOKENIZER_FILENAME = "tokenizer.json"
_VOCAB_MEDIA_TYPE = "application/jsonl"
_VOCAB_FORMAT = "jsonl"
_REQUIRED_SPECIAL_NAMES = ("bos", "eos", "pad", "unk")


class TokenizerError(ValueError):
    """Raised when train, encode, decode, freeze, save, or load fails."""


@dataclass(frozen=True)
class TrainConfig:
    """Training knobs. ``vocab_size`` is required; production is 256_000."""

    tokenizer_id: str
    vocab_size: int
    byte_fallback: bool = True


@dataclass(frozen=True)
class SpecialTokens:
    """F1 special_token_ids. bos, eos, pad, unk are required."""

    bos: int
    eos: int
    pad: int
    unk: int
    latent: int | None = None
    latent_start: int | None = None
    latent_end: int | None = None


@dataclass(frozen=True)
class ArcGridTokenRange:
    """Reserved id span for ARC cell-color tokens. Half-open ``[start, end)``."""

    start: int
    end: int


@dataclass(frozen=True)
class Artifact:
    """F1 artifact pointer."""

    content_hash: str
    bytes: int
    path: str | None = None
    media_type: str | None = None


@dataclass(frozen=True)
class TokenizerMeta:
    """In-memory form of ``contracts/schemas/v1/tokenizer.schema.json``."""

    tokenizer_id: str
    algorithm: str
    vocab_size: int
    byte_fallback: bool
    special_token_ids: SpecialTokens
    arc_grid_token_range: ArcGridTokenRange
    frozen: bool
    artifact: Artifact
    frozen_at: str | None = None
    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class VocabPointer:
    """In-memory form of ``contracts/schemas/v1/vocab.schema.json``."""

    tokenizer_id: str
    vocab_size: int
    special_token_ids: SpecialTokens
    artifact: Artifact
    format: str
    schema_id: str = VOCAB_SCHEMA_ID
    schema_version: int = SCHEMA_VERSION


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _layout_specials() -> SpecialTokens:
    return SpecialTokens(bos=0, eos=1, pad=2, unk=3)


def _arc_range() -> ArcGridTokenRange:
    return ArcGridTokenRange(start=_FIRST_ARC, end=_FIRST_ARC + ARC_N_COLORS)


def _byte_id(byte: int) -> int:
    return _FIRST_BYTE + byte


def _count_pairs(sequences: list[list[int]]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for seq in sequences:
        for left, right in zip(seq, seq[1:], strict=False):
            pair = (left, right)
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _best_pair(counts: dict[tuple[int, int], int]) -> tuple[int, int] | None:
    if not counts:
        return None
    return max(counts, key=lambda pair: (counts[pair], -pair[0], -pair[1]))


def _apply_merge(seq: list[int], left: int, right: int, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == left and seq[i + 1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _bpe_encode(byte_ids: list[int], merges: list[tuple[int, int]]) -> list[int]:
    if not byte_ids or not merges:
        return list(byte_ids)
    rank = {pair: i for i, pair in enumerate(merges)}
    ids = list(byte_ids)
    while True:
        best_rank: int | None = None
        best_pos: int | None = None
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            r = rank.get(pair)
            if r is None:
                continue
            if best_rank is None or r < best_rank:
                best_rank = r
                best_pos = i
        if best_rank is None or best_pos is None:
            break
        new_id = _FIRST_MERGE + best_rank
        ids = ids[:best_pos] + [new_id] + ids[best_pos + 2 :]
    return ids


def _expand_token(token_id: int, merges: list[tuple[int, int]]) -> bytes:
    if _FIRST_BYTE <= token_id < _FIRST_BYTE + _N_BYTES:
        return bytes([token_id - _FIRST_BYTE])
    merge_index = token_id - _FIRST_MERGE
    if 0 <= merge_index < len(merges):
        left, right = merges[merge_index]
        return _expand_token(left, merges) + _expand_token(right, merges)
    raise TokenizerError(f"unknown token id {token_id}")


def _artifact_bytes(
    specials: SpecialTokens,
    merges: list[tuple[int, int]],
    *,
    vocab_size: int,
    tokenizer_id: str,
    byte_fallback: bool,
) -> bytes:
    lines: list[str] = []
    for name in _REQUIRED_SPECIAL_NAMES:
        lines.append(
            _canonical_json(
                {
                    "id": int(getattr(specials, name)),
                    "kind": "special",
                    "name": name,
                }
            )
        )
    for byte in range(_N_BYTES):
        lines.append(_canonical_json({"byte": byte, "id": _byte_id(byte), "kind": "byte"}))
    for color in range(ARC_N_COLORS):
        lines.append(_canonical_json({"color": color, "id": _FIRST_ARC + color, "kind": "arc"}))
    for rank, (left, right) in enumerate(merges):
        lines.append(
            _canonical_json(
                {
                    "id": _FIRST_MERGE + rank,
                    "kind": "merge",
                    "left": left,
                    "right": right,
                }
            )
        )
    lines.append(
        _canonical_json(
            {
                "algorithm": ALGORITHM,
                "byte_fallback": byte_fallback,
                "kind": "header",
                "tokenizer_id": tokenizer_id,
                "vocab_size": vocab_size,
            }
        )
    )
    return ("".join(line + "\n" for line in lines)).encode("utf-8")


def _parse_vocab_jsonl(data: bytes) -> list[tuple[int, int]]:
    merges: list[tuple[int, int]] = []
    for raw_line in data.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("kind") != "merge":
            continue
        merges.append((int(obj["left"]), int(obj["right"])))
    return merges


def _specials_dict(specials: SpecialTokens) -> dict[str, int]:
    out: dict[str, int] = {}
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
            out[name] = int(value)
    return out


def _corpus_hash(docs: list[str]) -> str:
    blob = b"\n".join(doc.encode("utf-8") for doc in docs)
    return _sha256_hex(blob)


class Tokenizer:
    """Trained tokenizer. Encode/decode work before and after freeze."""

    def __init__(
        self,
        meta: TokenizerMeta,
        merges: list[tuple[int, int]] | None = None,
        *,
        corpus_hash: str | None = None,
    ) -> None:
        self.meta = meta
        self._merges = list(merges or [])
        self._corpus_hash = corpus_hash

    @classmethod
    def train(cls, docs: Iterable[str], config: TrainConfig) -> Tokenizer:
        """Train byte-fallback BPE on UTF-8 documents."""
        if not config.byte_fallback:
            raise TokenizerError("byte_fallback must be true for algorithm byte_fallback_bpe")
        if not config.tokenizer_id:
            raise TokenizerError("tokenizer_id must be non-empty")
        if config.vocab_size < _MIN_VOCAB_SIZE:
            raise TokenizerError(
                f"vocab_size {config.vocab_size} cannot fit specials + "
                f"{_N_BYTES} byte tokens + {ARC_N_COLORS} ARC colors + one merge "
                f"(need at least {_MIN_VOCAB_SIZE})"
            )

        corpus = list(docs)
        sequences: list[list[int]] = [
            [_byte_id(b) for b in doc.encode("utf-8")] for doc in corpus
        ]
        merges: list[tuple[int, int]] = []
        next_id = _FIRST_MERGE
        while next_id < config.vocab_size:
            counts = _count_pairs(sequences)
            pair = _best_pair(counts)
            if pair is None:
                break
            left, right = pair
            merges.append((left, right))
            sequences = [_apply_merge(seq, left, right, next_id) for seq in sequences]
            next_id += 1

        specials = _layout_specials()
        artifact_blob = _artifact_bytes(
            specials,
            merges,
            vocab_size=config.vocab_size,
            tokenizer_id=config.tokenizer_id,
            byte_fallback=config.byte_fallback,
        )
        meta = TokenizerMeta(
            tokenizer_id=config.tokenizer_id,
            algorithm=ALGORITHM,
            vocab_size=config.vocab_size,
            byte_fallback=config.byte_fallback,
            special_token_ids=specials,
            arc_grid_token_range=_arc_range(),
            artifact=Artifact(
                content_hash=_sha256_hex(artifact_blob),
                bytes=len(artifact_blob),
                path=None,
                media_type=_VOCAB_MEDIA_TYPE,
            ),
            frozen=False,
            frozen_at=None,
            schema_id=SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
        )
        return cls(meta, merges, corpus_hash=_corpus_hash(corpus))

    def freeze(self, frozen_at: str) -> None:
        """Set frozen and frozen_at. Further train raises TokenizerError."""
        if self.meta.frozen:
            if self.meta.frozen_at == frozen_at:
                return
            raise TokenizerError("tokenizer is frozen")
        self.meta = replace(self.meta, frozen=True, frozen_at=frozen_at)

    def encode(self, text: str) -> list[int]:
        """Encode UTF-8 text to token ids."""
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of encode. Round-trip returns the original text."""
        return self.decode_bytes(ids).decode("utf-8")

    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode raw bytes. Used by byte fallback."""
        byte_ids = [_byte_id(b) for b in data]
        return _bpe_encode(byte_ids, self._merges)

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        """Inverse of encode_bytes."""
        parts = bytearray()
        for token_id in ids:
            parts.extend(_expand_token(int(token_id), self._merges))
        return bytes(parts)

    def encode_grid(self, cells: Sequence[int]) -> list[int]:
        """One token per ARC cell color. Values in ``range(ARC_N_COLORS)``."""
        start = self.meta.arc_grid_token_range.start
        end = self.meta.arc_grid_token_range.end
        n_colors = end - start
        out: list[int] = []
        for cell in cells:
            color = int(cell)
            if color < 0 or color >= n_colors:
                raise TokenizerError(f"ARC color {color} out of range 0..{n_colors - 1}")
            out.append(start + color)
        return out

    def tokens_per_word(self, text: str) -> float:
        """encode(text) length / whitespace-separated word count. Empty is 0.0."""
        words = text.split()
        if not words:
            return 0.0
        return len(self.encode(text)) / len(words)

    def save(self, directory: str | Path) -> VocabPointer:
        """Write tokenizer.json and the vocab artifact. Return the vocab pointer."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        blob = _artifact_bytes(
            self.meta.special_token_ids,
            self._merges,
            vocab_size=self.meta.vocab_size,
            tokenizer_id=self.meta.tokenizer_id,
            byte_fallback=self.meta.byte_fallback,
        )
        vocab_path = directory / _VOCAB_FILENAME
        vocab_path.write_bytes(blob)
        artifact = Artifact(
            content_hash=_sha256_hex(blob),
            bytes=len(blob),
            path=_VOCAB_FILENAME,
            media_type=_VOCAB_MEDIA_TYPE,
        )
        self.meta = replace(self.meta, artifact=artifact)
        contract = self.to_contract()
        (directory / _TOKENIZER_FILENAME).write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return VocabPointer(
            schema_id=VOCAB_SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            tokenizer_id=self.meta.tokenizer_id,
            vocab_size=self.meta.vocab_size,
            special_token_ids=self.meta.special_token_ids,
            artifact=artifact,
            format=_VOCAB_FORMAT,
        )

    @classmethod
    def load(cls, directory: str | Path) -> Tokenizer:
        """Load a directory written by save."""
        directory = Path(directory)
        try:
            contract = json.loads((directory / _TOKENIZER_FILENAME).read_text(encoding="utf-8"))
        except OSError as exc:
            raise TokenizerError(f"failed to load tokenizer.json: {exc}") from exc
        try:
            blob = (directory / _VOCAB_FILENAME).read_bytes()
        except OSError as exc:
            raise TokenizerError(f"failed to load {_VOCAB_FILENAME}: {exc}") from exc
        merges = _parse_vocab_jsonl(blob)
        specials_raw = contract.get("special_tokens") or {}
        specials = SpecialTokens(
            bos=int(specials_raw["bos"]),
            eos=int(specials_raw["eos"]),
            pad=int(specials_raw["pad"]),
            unk=int(specials_raw["unk"]),
            latent=specials_raw.get("latent"),
            latent_start=specials_raw.get("latent_start"),
            latent_end=specials_raw.get("latent_end"),
        )
        arc_raw = contract["arc_grid_token_range"]
        frozen = bool(contract.get("frozen", False))
        meta = TokenizerMeta(
            tokenizer_id=str(contract["tokenizer_id"]),
            algorithm=str(contract["algorithm"]),
            vocab_size=int(contract["vocab_size"]),
            byte_fallback=bool(contract["byte_fallback"]),
            special_token_ids=specials,
            arc_grid_token_range=ArcGridTokenRange(
                start=int(arc_raw["start"]),
                end=int(arc_raw["end"]),
            ),
            artifact=Artifact(
                content_hash=_sha256_hex(blob),
                bytes=len(blob),
                path=_VOCAB_FILENAME,
                media_type=_VOCAB_MEDIA_TYPE,
            ),
            frozen=frozen,
            frozen_at=contract.get("frozen_at"),
            schema_id=str(contract["schema_id"]),
            schema_version=int(contract["schema_version"]),
        )
        return cls(meta, merges, corpus_hash=contract.get("corpus_hash"))

    def to_contract(self) -> dict:
        """JSON object matching prometheus.tokenizer for contracts.validate."""
        payload: dict[str, Any] = {
            "schema_id": self.meta.schema_id,
            "schema_version": self.meta.schema_version,
            "tokenizer_id": self.meta.tokenizer_id,
            "algorithm": self.meta.algorithm,
            "vocab_size": self.meta.vocab_size,
            "byte_fallback": self.meta.byte_fallback,
            "special_tokens": _specials_dict(self.meta.special_token_ids),
            "arc_grid_token_range": {
                "start": self.meta.arc_grid_token_range.start,
                "end": self.meta.arc_grid_token_range.end,
            },
            "vocab_hash": self.meta.artifact.content_hash,
            "frozen": self.meta.frozen,
        }
        if self.meta.frozen_at is not None:
            payload["frozen_at"] = self.meta.frozen_at
        if self._corpus_hash is not None:
            payload["corpus_hash"] = self._corpus_hash
        return payload


def baseline_byte_tokens_per_word(text: str) -> float:
    """UTF-8 byte count / whitespace-separated words. Empty text is 0.0.

    F6 gate: trained tokens_per_word must beat this baseline on the same text.
    """
    words = len(text.split())
    if words == 0:
        return 0.0
    return len(text.encode("utf-8")) / words
