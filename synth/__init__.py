"""Synthetic rewrites: generation orchestration and fact-check (spec 7.1, 15.5 E1).

High-quality documents are rephrased in several styles (Kimi K2-style) and
fact-checked against the source. The checker is source-grounded and does not
call the generator. The generator is F5's batch API (prompts, max_tokens,
temperature) behind :class:`Generator`. Token counts use F6 encode length
when a :class:`TokenCounter` is provided.

Gate: supported-class precision of :func:`check_claim` on a labeled sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

SCHEMA_VERSION = 1
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_BATCH = 8


class SynthError(ValueError):
    """Raised when rephrase, fact-check, or generation orchestration fails."""


class Style(StrEnum):
    """Rewrite styles. Serde/JSON names are the values."""

    ENCYCLOPEDIA = "encyclopedia"
    TEXTBOOK = "textbook"
    CONVERSATIONAL = "conversational"
    NEWS = "news"
    TECHNICAL = "technical"
    SIMPLIFIED = "simplified"


STYLES: tuple[Style, ...] = (
    Style.ENCYCLOPEDIA,
    Style.TEXTBOOK,
    Style.CONVERSATIONAL,
    Style.NEWS,
    Style.TECHNICAL,
    Style.SIMPLIFIED,
)


class Verdict(StrEnum):
    """Fact-check of one claim against the source document."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_IN_SOURCE = "not_in_source"


@runtime_checkable
class Generator(Protocol):
    """F5 batch generation. One completion per prompt, same order."""

    def generate(
        self, prompts: Sequence[str], max_tokens: int, temperature: float
    ) -> list[str]:
        """Return one string per prompt.

        Empty ``prompts`` is an error. Length mismatch with the result is an
        error. ``max_tokens`` and ``temperature`` match F5 ``GenerateRequest``.
        """
        ...


@runtime_checkable
class TokenCounter(Protocol):
    """F6 encode length. :class:`tokenizer.Tokenizer` satisfies this."""

    def encode(self, text: str) -> Sequence[int]:
        """Encode UTF-8 text to token ids."""
        ...


@dataclass(frozen=True)
class SourceDocument:
    """One source document to rephrase."""

    source_id: str
    text: str


@dataclass(frozen=True)
class Claim:
    """One atomic claim extracted from a rewrite, with a source verdict."""

    text: str
    verdict: Verdict


@dataclass(frozen=True)
class FactCheck:
    """All claims from a rewrite. Counts are derived from ``claims``."""

    claims: tuple[Claim, ...]

    @property
    def n_supported(self) -> int:
        return sum(1 for c in self.claims if c.verdict == Verdict.SUPPORTED)

    @property
    def n_contradicted(self) -> int:
        return sum(1 for c in self.claims if c.verdict == Verdict.CONTRADICTED)

    @property
    def n_not_in_source(self) -> int:
        return sum(1 for c in self.claims if c.verdict == Verdict.NOT_IN_SOURCE)


@dataclass(frozen=True)
class Rewrite:
    """One style rewrite of a source document, after fact-check."""

    source_id: str
    style: Style
    text: str
    token_count: int
    fact_check: FactCheck
    accepted: bool


@dataclass(frozen=True)
class LabeledExample:
    """Gold verdict for :func:`check_claim`. Used by :func:`precision`."""

    source: str
    claim: str
    gold: Verdict


@dataclass(frozen=True)
class OrchestratorConfig:
    """Batch and decoding knobs. ``styles`` is the default Cartesian factor."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    max_batch: int = DEFAULT_MAX_BATCH
    styles: tuple[Style, ...] = STYLES


def generate_prompt(doc: SourceDocument, style: Style) -> str:
    """Prompt the generator to rephrase ``doc`` in ``style``.

    Empty ``doc.text`` raises :class:`SynthError`. The template is the
    contract between Python and Rust so both languages send the same bytes
    to F5.
    """
    if not doc.text:
        raise SynthError("empty source")
    return (
        f"Rephrase the document in {style.value} style. "
        "Preserve every fact. Do not add facts.\n\n"
        f"Document:\n{doc.text}"
    )


def extract_claims(text: str) -> list[str]:
    """Split a rewrite into atomic factual claims. Empty text is empty."""
    raise NotImplementedError("E1 extract_claims")


def check_claim(source: str, claim: str) -> Verdict:
    """Ground ``claim`` in ``source``. Empty source raises :class:`SynthError`."""
    raise NotImplementedError("E1 check_claim")


def fact_check(source: str, rewrite: str) -> FactCheck:
    """Extract claims from ``rewrite`` and ground each in ``source``.

    Empty source raises :class:`SynthError`. Empty rewrite is zero claims.
    """
    raise NotImplementedError("E1 fact_check")


def accept(check: FactCheck) -> bool:
    """Keep a rewrite iff no claim is contradicted. ``not_in_source`` is kept."""
    raise NotImplementedError("E1 accept")


def token_count(text: str, tokenizer: TokenCounter | None = None) -> int:
    """``len(tokenizer.encode(text))`` if given, else whitespace words. Empty is 0."""
    raise NotImplementedError("E1 token_count")


def precision(examples: Sequence[LabeledExample]) -> float:
    """Supported-class precision of :func:`check_claim` vs gold.

    TP = predicted supported and gold supported. FP = predicted supported
    and gold not supported. Empty examples, or no predicted supported, is
    0.0 (fail-closed).
    """
    raise NotImplementedError("E1 precision")


class Orchestrator:
    """Batch rephrase over documents × styles, then fact-check.

    ``generator`` is F5. ``tokenizer`` is F6 when token counts should match
    the frozen vocab; omit it to count whitespace words.
    """

    def __init__(
        self,
        generator: Generator,
        tokenizer: TokenCounter | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        raise NotImplementedError("E1 Orchestrator")

    def rephrase(self, doc: SourceDocument, style: Style) -> Rewrite:
        """One document, one style. Empty source raises :class:`SynthError`."""
        raise NotImplementedError("E1 rephrase")

    def rephrase_many(
        self,
        docs: Sequence[SourceDocument],
        styles: Sequence[Style] | None = None,
    ) -> list[Rewrite]:
        """Cartesian product of docs and styles, chunked by ``max_batch``.

        Empty ``docs`` raises :class:`SynthError`. Default styles are
        ``config.styles``. Result order is docs-major, then styles.
        """
        raise NotImplementedError("E1 rephrase_many")
