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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

SCHEMA_ID = "prometheus.tokenizer"
VOCAB_SCHEMA_ID = "prometheus.vocab"
SCHEMA_VERSION = 1
ALGORITHM = "byte_fallback_bpe"
PRODUCTION_VOCAB_SIZE = 256_000
ARC_N_COLORS = 10


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


class Tokenizer:
    """Trained tokenizer. Encode/decode work before and after freeze."""

    def __init__(self, meta: TokenizerMeta) -> None:
        self.meta = meta

    @classmethod
    def train(cls, docs: Iterable[str], config: TrainConfig) -> Tokenizer:
        """Train byte-fallback BPE on UTF-8 documents."""
        raise NotImplementedError("F6 train")

    def freeze(self, frozen_at: str) -> None:
        """Set frozen and frozen_at. Further train raises TokenizerError."""
        raise NotImplementedError("F6 freeze")

    def encode(self, text: str) -> list[int]:
        """Encode UTF-8 text to token ids."""
        raise NotImplementedError("F6 encode")

    def decode(self, ids: Sequence[int]) -> str:
        """Inverse of encode. Round-trip returns the original text."""
        raise NotImplementedError("F6 decode")

    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode raw bytes. Used by byte fallback."""
        raise NotImplementedError("F6 encode_bytes")

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        """Inverse of encode_bytes."""
        raise NotImplementedError("F6 decode_bytes")

    def encode_grid(self, cells: Sequence[int]) -> list[int]:
        """One token per ARC cell color. Values in ``range(ARC_N_COLORS)``."""
        raise NotImplementedError("F6 encode_grid")

    def tokens_per_word(self, text: str) -> float:
        """encode(text) length / whitespace-separated word count. Empty is 0.0."""
        raise NotImplementedError("F6 tokens_per_word")

    def save(self, directory: str) -> VocabPointer:
        """Write tokenizer.json and the vocab artifact. Return the vocab pointer."""
        raise NotImplementedError("F6 save")

    @classmethod
    def load(cls, directory: str) -> Tokenizer:
        """Load a directory written by save."""
        raise NotImplementedError("F6 load")

    def to_contract(self) -> dict:
        """JSON object matching prometheus.tokenizer for contracts.validate."""
        raise NotImplementedError("F6 to_contract")


def baseline_byte_tokens_per_word(text: str) -> float:
    """UTF-8 byte count / whitespace-separated words. Empty text is 0.0.

    F6 gate: trained tokens_per_word must beat this baseline on the same text.
    """
    words = len(text.split())
    if words == 0:
        return 0.0
    return len(text.encode("utf-8")) / words
