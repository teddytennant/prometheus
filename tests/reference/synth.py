"""Independent slow reference for E1 synthetic rewrites (spec 7.1, 15.5 E1).

This is the source of truth for extract_claims / check_claim / fact_check /
accept / token_count / precision / orchestration. Production ``synth`` must
match these results. Production must never import ``tests/``.

Pinned rules
------------
extract_claims
    Empty or whitespace-only text (no ``str.split()`` tokens) returns ``[]``.
    Otherwise split on ``(?<=[.!?])\\s+``: a sentence-ending ``.`` ``!`` or
    ``?`` followed by one or more Unicode whitespace characters. The
    punctuation stays on the preceding piece; the whitespace is discarded.
    Trim each piece and drop empties. Casing is preserved. Abbreviations are
    not special-cased (``Dr. Smith`` yields ``["Dr.", "Smith"]`` if a space
    follows the first period).

check_claim
    Empty source (zero-length string, not whitespace) raises ``SynthError``.
    Empty or whitespace-only claim returns ``not_in_source``.
    Matching uses a normalized form: lowercase, Unicode whitespace collapsed
    to single spaces, ends stripped (``" ".join(text.lower().split())``).
    Tokens are that string split on spaces (bag-of-words).
    Numbers are ASCII runs matching ``[0-9]+(?:\\.[0-9]+)?`` on the original
    text (the user-facing regex ``\\d+(?:\\.\\d+)?`` on ASCII).
    CONTRADICTED if the claim has a number that is not in the set of numbers
    extracted from the source, AND the source has at least one number.
    CONTRADICTED if the claim has a content word ``w`` and the source has
    adjacent tokens ``not w`` or ``no w`` (after normalize). Number conflict
    is checked before negation.
    A content word is a token of Unicode length >= 4 that is not in
    CONTENT_STOPLIST.
    Else SUPPORTED if the claim has content words and every one appears as a
    source token, OR the claim has no content words and the full normalized
    claim is a substring of the normalized source.
    Else NOT_IN_SOURCE.

fact_check
    Empty source raises. Empty rewrite is zero claims. Otherwise
    extract_claims(rewrite) then check_claim(source, c) for each, in order.
    Does not call a generator.

accept
    True iff ``n_contradicted == 0``. ``not_in_source`` does not reject.
    Zero claims accept.

token_count
    Empty string returns 0 without calling a tokenizer.
    If ``tokenizer`` is given: ``len(tokenizer.encode(text))``.
    Else Unicode whitespace words: ``len(text.split())``.

precision
    Supported-class precision of check_claim vs gold.
    TP = predicted supported AND gold supported.
    FP = predicted supported AND gold != supported.
    Empty examples or TP+FP == 0 returns 0.0 (fail-closed).
    Empty source in an example propagates SynthError.

Orchestrator
    Empty doc.text errors via generate_prompt before generate.
    rephrase_many: empty docs is SynthError before generate. Default styles
    are config.styles. Cartesian product is docs-major then styles. Prompts
    are chunked by config.max_batch (last chunk may be shorter). Generator
    length mismatch is SynthError. max_batch in tests is always >= 1.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from synth import (
    Claim,
    FactCheck,
    Generator,
    LabeledExample,
    OrchestratorConfig,
    Rewrite,
    SourceDocument,
    Style,
    SynthError,
    TokenCounter,
    Verdict,
    generate_prompt,
)

CONTENT_STOPLIST = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "they",
        "them",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "into",
        "over",
        "under",
        "than",
        "then",
        "when",
        "what",
        "which",
        "their",
    }
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _content_words(tokens: Sequence[str]) -> list[str]:
    return [t for t in tokens if len(t) >= 4 and t not in CONTENT_STOPLIST]


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(text)


def extract_claims(text: str) -> list[str]:
    """Split a rewrite into atomic factual claims. See module docstring."""
    if not text.split():
        return []
    return [piece.strip() for piece in _SENTENCE_SPLIT.split(text) if piece.strip()]


def check_claim(source: str, claim: str) -> Verdict:
    """Ground ``claim`` in ``source``. See module docstring."""
    if source == "":
        raise SynthError("empty source")
    if not claim.split():
        return Verdict.NOT_IN_SOURCE

    src_nums = _numbers(source)
    if src_nums:
        src_num_set = set(src_nums)
        if any(n not in src_num_set for n in _numbers(claim)):
            return Verdict.CONTRADICTED

    src_norm = _normalize(source)
    claim_norm = _normalize(claim)
    src_tokens = src_norm.split()
    claim_tokens = claim_norm.split()
    claim_content = _content_words(claim_tokens)

    content_set = set(claim_content)
    for i in range(len(src_tokens) - 1):
        if src_tokens[i] in ("not", "no") and src_tokens[i + 1] in content_set:
            return Verdict.CONTRADICTED

    if claim_content:
        src_token_set = set(src_tokens)
        if all(w in src_token_set for w in claim_content):
            return Verdict.SUPPORTED
        return Verdict.NOT_IN_SOURCE
    if claim_norm in src_norm:
        return Verdict.SUPPORTED
    return Verdict.NOT_IN_SOURCE


def fact_check(source: str, rewrite: str) -> FactCheck:
    """Extract claims from ``rewrite`` and ground each in ``source``."""
    if source == "":
        raise SynthError("empty source")
    if rewrite == "":
        return FactCheck(claims=())
    claims = tuple(Claim(text=c, verdict=check_claim(source, c)) for c in extract_claims(rewrite))
    return FactCheck(claims=claims)


def accept(check: FactCheck) -> bool:
    """Keep a rewrite iff no claim is contradicted."""
    return check.n_contradicted == 0


def token_count(text: str, tokenizer: TokenCounter | None = None) -> int:
    """Encode length if a tokenizer is given, else whitespace words."""
    if text == "":
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def precision(examples: Sequence[LabeledExample]) -> float:
    """Supported-class precision of check_claim vs gold. Fail-closed."""
    tp = 0
    fp = 0
    for ex in examples:
        pred = check_claim(ex.source, ex.claim)
        if pred == Verdict.SUPPORTED:
            if ex.gold == Verdict.SUPPORTED:
                tp += 1
            else:
                fp += 1
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def rewrite_from_completion(
    doc: SourceDocument,
    style: Style,
    completion: str,
    tokenizer: TokenCounter | None = None,
) -> Rewrite:
    """Build the Rewrite the orchestrator must return for ``completion``."""
    check = fact_check(doc.text, completion)
    return Rewrite(
        source_id=doc.source_id,
        style=style,
        text=completion,
        token_count=token_count(completion, tokenizer),
        fact_check=check,
        accepted=accept(check),
    )


class ReferenceOrchestrator:
    """Same contract as ``synth.Orchestrator``, implemented here."""

    def __init__(
        self,
        generator: Generator,
        tokenizer: TokenCounter | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer
        self.config = config if config is not None else OrchestratorConfig()

    def rephrase(self, doc: SourceDocument, style: Style) -> Rewrite:
        prompt = generate_prompt(doc, style)
        outs = self.generator.generate([prompt], self.config.max_tokens, self.config.temperature)
        if len(outs) != 1:
            raise SynthError(f"generator returned {len(outs)} completions for 1 prompts")
        return rewrite_from_completion(doc, style, outs[0], self.tokenizer)

    def rephrase_many(
        self,
        docs: Sequence[SourceDocument],
        styles: Sequence[Style] | None = None,
    ) -> list[Rewrite]:
        if len(docs) == 0:
            raise SynthError("empty batch")
        use_styles: Sequence[Style] = self.config.styles if styles is None else styles
        pairs = [(doc, style) for doc in docs for style in use_styles]
        prompts = [generate_prompt(doc, style) for doc, style in pairs]
        max_batch = self.config.max_batch
        completions: list[str] = []
        for i in range(0, len(prompts), max_batch):
            chunk = prompts[i : i + max_batch]
            outs = self.generator.generate(chunk, self.config.max_tokens, self.config.temperature)
            if len(outs) != len(chunk):
                raise SynthError(
                    f"generator returned {len(outs)} completions for {len(chunk)} prompts"
                )
            completions.extend(outs)
        return [
            rewrite_from_completion(doc, style, text, self.tokenizer)
            for (doc, style), text in zip(pairs, completions, strict=True)
        ]


__all__ = [
    "CONTENT_STOPLIST",
    "ReferenceOrchestrator",
    "accept",
    "check_claim",
    "extract_claims",
    "fact_check",
    "precision",
    "rewrite_from_completion",
    "token_count",
]
