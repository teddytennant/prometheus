"""Slow, obvious byte-fallback BPE reference for F6 (spec 3, 7, 10, 15).

This is the oracle, not the production tokenizer. Production ``tokenizer`` /
``prometheus-tokenizer`` must match these token ids, grid ids, freeze rules,
and contract fields on the same inputs.

Do not import production ``Tokenizer.train`` / ``encode`` (those are stubs
until F6 is implemented). Duck-type the production dataclasses.

Algorithm
---------
Byte-level BPE with a reserved alphabet and no pre-tokenization (spaces are
just byte 0x20). Layout of the id space::

    [0, N_SPECIALS)                         required specials bos,eos,pad,unk
    [N_SPECIALS, N_SPECIALS+256)            raw bytes 0..=255   (byte_fallback)
    [FIRST_ARC, FIRST_ARC+ARC_N_COLORS)     ARC grid colors 0..=9
    [FIRST_MERGE, vocab_size)               BPE merges, in train order

``N_SPECIALS`` is 4. Optional latent specials are not allocated by this
reference (schema extraProperties); production may add them only by shifting
this whole layout, which would fail the locked goldens — so production must
use this layout too.

Training (slow recount every step)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each doc is UTF-8 bytes mapped to byte-token ids. Adjacent pair counts are
summed across docs (never across a doc boundary). The next merge is the pair
with the highest count; ties break by smaller left id, then smaller right id.
The chosen pair is replaced left-to-right, non-overlapping, in every sequence.
Stop when ``next_id == vocab_size`` or no adjacent pair remains.

Encoding
~~~~~~~~
Map bytes to byte-token ids, then repeatedly merge the lowest-rank learned
pair present in the sequence (leftmost occurrence if several share that rank).
Specials and ARC ids are never produced by ``encode`` / ``encode_bytes``.

Decoding
~~~~~~~~
Expand merge ids recursively into bytes. Specials, ARC colors, and unknown
ids are errors. ``decode`` UTF-8-decodes the bytes of ``decode_bytes``.

Grid
~~~~
``encode_grid`` is a 1-1 map ``color -> FIRST_ARC + color`` with no BPE and
no extra tokens. 2D RoPE is model-side (spec 10), not tokenizer-side.

Artifact
~~~~~~~~
``vocab.jsonl``: one canonical JSON object per line (sorted keys, compact
separators, trailing newline). Lines are specials, then 256 bytes, then 10
ARC colors, then merges in rank order. ``content_hash`` is SHA-256 of those
bytes, lowercase hex. ``to_contract()`` emits ``prometheus.tokenizer`` with
``vocab_hash`` equal to that content hash (not an embedded artifact object).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from tokenizer import (
    ALGORITHM,
    ARC_N_COLORS,
    SCHEMA_ID,
    SCHEMA_VERSION,
    VOCAB_SCHEMA_ID,
    ArcGridTokenRange,
    Artifact,
    SpecialTokens,
    TokenizerError,
    TokenizerMeta,
    TrainConfig,
    VocabPointer,
)

N_SPECIALS = 4
N_BYTES = 256
FIRST_BYTE = N_SPECIALS
FIRST_ARC = N_SPECIALS + N_BYTES
FIRST_MERGE = FIRST_ARC + ARC_N_COLORS
# specials + bytes + ARC colors + at least one merge slot
MIN_VOCAB_SIZE = FIRST_MERGE + 1

VOCAB_FILENAME = "vocab.jsonl"
TOKENIZER_FILENAME = "tokenizer.json"
VOCAB_MEDIA_TYPE = "application/jsonl"
VOCAB_FORMAT = "jsonl"

REQUIRED_SPECIAL_NAMES = ("bos", "eos", "pad", "unk")


def min_vocab_size(*, byte_fallback: bool) -> int:
    """Smallest ``vocab_size`` this algorithm will accept."""
    if not byte_fallback:
        # Character-level BPE is not specified; F6 is byte_fallback_bpe.
        return MIN_VOCAB_SIZE
    return MIN_VOCAB_SIZE


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _layout_specials() -> SpecialTokens:
    return SpecialTokens(bos=0, eos=1, pad=2, unk=3)


def _arc_range() -> ArcGridTokenRange:
    return ArcGridTokenRange(start=FIRST_ARC, end=FIRST_ARC + ARC_N_COLORS)


def _byte_id(byte: int) -> int:
    return FIRST_BYTE + byte


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
    # Highest count, then smaller left id, then smaller right id.
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
        new_id = FIRST_MERGE + best_rank
        ids = ids[:best_pos] + [new_id] + ids[best_pos + 2 :]
    return ids


def _expand_token(token_id: int, merges: list[tuple[int, int]]) -> bytes:
    if FIRST_BYTE <= token_id < FIRST_BYTE + N_BYTES:
        return bytes([token_id - FIRST_BYTE])
    merge_index = token_id - FIRST_MERGE
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
    for name in REQUIRED_SPECIAL_NAMES:
        lines.append(
            _canonical_json(
                {
                    "id": int(getattr(specials, name)),
                    "kind": "special",
                    "name": name,
                }
            )
        )
    for byte in range(N_BYTES):
        lines.append(
            _canonical_json({"byte": byte, "id": _byte_id(byte), "kind": "byte"})
        )
    for color in range(ARC_N_COLORS):
        lines.append(
            _canonical_json(
                {"color": color, "id": FIRST_ARC + color, "kind": "arc"}
            )
        )
    for rank, (left, right) in enumerate(merges):
        lines.append(
            _canonical_json(
                {
                    "id": FIRST_MERGE + rank,
                    "kind": "merge",
                    "left": left,
                    "right": right,
                }
            )
        )
    # Header trailer so a reader can recover config without tokenizer.json.
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
    return sha256_hex(blob)


class ReferenceTokenizer:
    """Independent tokenizer used as the F6 oracle."""

    def __init__(
        self,
        meta: TokenizerMeta,
        merges: list[tuple[int, int]],
        *,
        corpus_hash: str | None = None,
    ) -> None:
        self.meta = meta
        self._merges = list(merges)
        self._corpus_hash = corpus_hash

    @classmethod
    def train(cls, docs: Iterable[str], config: TrainConfig) -> ReferenceTokenizer:
        if not config.byte_fallback:
            raise TokenizerError(
                "byte_fallback must be true for algorithm byte_fallback_bpe"
            )
        if not config.tokenizer_id:
            raise TokenizerError("tokenizer_id must be non-empty")
        if config.vocab_size < MIN_VOCAB_SIZE:
            raise TokenizerError(
                f"vocab_size {config.vocab_size} cannot fit specials + "
                f"{N_BYTES} byte tokens + {ARC_N_COLORS} ARC colors + one merge "
                f"(need at least {MIN_VOCAB_SIZE})"
            )

        corpus = list(docs)
        sequences: list[list[int]] = [
            [_byte_id(b) for b in doc.encode("utf-8")] for doc in corpus
        ]
        merges: list[tuple[int, int]] = []
        next_id = FIRST_MERGE
        while next_id < config.vocab_size:
            counts = _count_pairs(sequences)
            pair = _best_pair(counts)
            if pair is None:
                break
            left, right = pair
            merges.append((left, right))
            sequences = [
                _apply_merge(seq, left, right, next_id) for seq in sequences
            ]
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
            schema_id=SCHEMA_ID,
            schema_version=SCHEMA_VERSION,
            tokenizer_id=config.tokenizer_id,
            algorithm=ALGORITHM,
            vocab_size=config.vocab_size,
            byte_fallback=config.byte_fallback,
            special_token_ids=specials,
            arc_grid_token_range=_arc_range(),
            artifact=Artifact(
                content_hash=sha256_hex(artifact_blob),
                bytes=len(artifact_blob),
                path=None,
                media_type=VOCAB_MEDIA_TYPE,
            ),
            frozen=False,
            frozen_at=None,
        )
        return cls(meta, merges, corpus_hash=_corpus_hash(corpus))

    def freeze(self, frozen_at: str) -> None:
        if self.meta.frozen:
            if self.meta.frozen_at == frozen_at:
                return
            raise TokenizerError("tokenizer is frozen")
        self.meta = replace(self.meta, frozen=True, frozen_at=frozen_at)

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8")

    def encode_bytes(self, data: bytes) -> list[int]:
        byte_ids = [_byte_id(b) for b in data]
        return _bpe_encode(byte_ids, self._merges)

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        parts = bytearray()
        for token_id in ids:
            parts.extend(_expand_token(int(token_id), self._merges))
        return bytes(parts)

    def encode_grid(self, cells: Sequence[int]) -> list[int]:
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
        words = text.split()
        if not words:
            return 0.0
        return len(self.encode(text)) / len(words)

    def save(self, directory: str | Path) -> VocabPointer:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        blob = _artifact_bytes(
            self.meta.special_token_ids,
            self._merges,
            vocab_size=self.meta.vocab_size,
            tokenizer_id=self.meta.tokenizer_id,
            byte_fallback=self.meta.byte_fallback,
        )
        vocab_path = directory / VOCAB_FILENAME
        vocab_path.write_bytes(blob)
        artifact = Artifact(
            content_hash=sha256_hex(blob),
            bytes=len(blob),
            path=VOCAB_FILENAME,
            media_type=VOCAB_MEDIA_TYPE,
        )
        self.meta = replace(self.meta, artifact=artifact)
        contract = self.to_contract()
        (directory / TOKENIZER_FILENAME).write_text(
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
            format=VOCAB_FORMAT,
        )

    @classmethod
    def load(cls, directory: str | Path) -> ReferenceTokenizer:
        directory = Path(directory)
        try:
            contract = json.loads(
                (directory / TOKENIZER_FILENAME).read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise TokenizerError(f"failed to load tokenizer.json: {exc}") from exc
        vocab_name = VOCAB_FILENAME
        try:
            blob = (directory / vocab_name).read_bytes()
        except OSError as exc:
            raise TokenizerError(f"failed to load {vocab_name}: {exc}") from exc
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
            schema_id=str(contract["schema_id"]),
            schema_version=int(contract["schema_version"]),
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
                content_hash=sha256_hex(blob),
                bytes=len(blob),
                path=vocab_name,
                media_type=VOCAB_MEDIA_TYPE,
            ),
            frozen=frozen,
            frozen_at=contract.get("frozen_at"),
        )
        return cls(meta, merges, corpus_hash=contract.get("corpus_hash"))

    def to_contract(self) -> dict[str, Any]:
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
