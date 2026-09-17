"""Synthetic data: rewrites (E1), rejection sampling (E2), procedural ARC (E3).

E1: high-quality documents are rephrased in several styles (Kimi K2-style)
and fact-checked against the source. Gate: supported-class precision of
:func:`check_claim` on a labeled sample.

E2: reasoning traces sampled from F5 and kept only when a D2 verifier
sets ``passed``. Gate: :func:`verified_correct_rate`.

E3: re-arc-style grid families, 2D tokenization via F6 ``encode_grid``,
dihedral × color-perm augmentation. Gate: :func:`diversity_stats`.

The generator is F5's batch API (prompts, max_tokens, temperature) behind
:class:`Generator`. Token counts use F6 encode length when a
:class:`TokenCounter` is provided.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

SCHEMA_VERSION = 1
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_BATCH = 8
DEFAULT_N_SAMPLES = 8
N_COLORS = 10
MIN_GRID_SIZE = 1
MAX_GRID_SIZE = 30
N_DIHEDRAL = 8
DEFAULT_N_TRAIN = 3


class SynthError(ValueError):
    """Raised when rephrase, fact-check, generation, or rejection sampling fails."""


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


_CONTENT_STOPLIST = frozenset(
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
    return [t for t in tokens if len(t) >= 4 and t not in _CONTENT_STOPLIST]


def _numbers(text: str) -> list[str]:
    return _NUMBER.findall(text)


def extract_claims(text: str) -> list[str]:
    """Split a rewrite into atomic factual claims. Empty text is empty."""
    if not text.split():
        return []
    return [piece.strip() for piece in _SENTENCE_SPLIT.split(text) if piece.strip()]


def check_claim(source: str, claim: str) -> Verdict:
    """Ground ``claim`` in ``source``. Empty source raises :class:`SynthError`."""
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
    """Extract claims from ``rewrite`` and ground each in ``source``.

    Empty source raises :class:`SynthError`. Empty rewrite is zero claims.
    """
    if source == "":
        raise SynthError("empty source")
    if rewrite == "":
        return FactCheck(claims=())
    claims = tuple(Claim(text=c, verdict=check_claim(source, c)) for c in extract_claims(rewrite))
    return FactCheck(claims=claims)


def accept(check: FactCheck) -> bool:
    """Keep a rewrite iff no claim is contradicted. ``not_in_source`` is kept."""
    return check.n_contradicted == 0


def token_count(text: str, tokenizer: TokenCounter | None = None) -> int:
    """``len(tokenizer.encode(text))`` if given, else whitespace words. Empty is 0."""
    if text == "":
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def precision(examples: Sequence[LabeledExample]) -> float:
    """Supported-class precision of :func:`check_claim` vs gold.

    TP = predicted supported and gold supported. FP = predicted supported
    and gold not supported. Empty examples, or no predicted supported, is
    0.0 (fail-closed).
    """
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


def _rewrite_from_completion(
    doc: SourceDocument,
    style: Style,
    completion: str,
    tokenizer: TokenCounter | None = None,
) -> Rewrite:
    check = fact_check(doc.text, completion)
    return Rewrite(
        source_id=doc.source_id,
        style=style,
        text=completion,
        token_count=token_count(completion, tokenizer),
        fact_check=check,
        accepted=accept(check),
    )


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
        self.generator = generator
        self.tokenizer = tokenizer
        self.config = config if config is not None else OrchestratorConfig()

    def rephrase(self, doc: SourceDocument, style: Style) -> Rewrite:
        """One document, one style. Empty source raises :class:`SynthError`."""
        prompt = generate_prompt(doc, style)
        outs = self.generator.generate([prompt], self.config.max_tokens, self.config.temperature)
        if len(outs) != 1:
            raise SynthError(f"generator returned {len(outs)} completions for 1 prompts")
        return _rewrite_from_completion(doc, style, outs[0], self.tokenizer)

    def rephrase_many(
        self,
        docs: Sequence[SourceDocument],
        styles: Sequence[Style] | None = None,
    ) -> list[Rewrite]:
        """Cartesian product of docs and styles, chunked by ``max_batch``.

        Empty ``docs`` raises :class:`SynthError`. Default styles are
        ``config.styles``. Result order is docs-major, then styles.
        """
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
            _rewrite_from_completion(doc, style, text, self.tokenizer)
            for (doc, style), text in zip(pairs, completions, strict=True)
        ]


class Domain(StrEnum):
    """D2 domain. Values map onto VerifierKind (math → SymbolicMath, code →
    SandboxedTests, lean → LeanKernel, grid → GridMatch, market →
    MarketResolution)."""

    MATH = "math"
    CODE = "code"
    LEAN = "lean"
    GRID = "grid"
    MARKET = "market"


@dataclass(frozen=True)
class Problem:
    """One problem to sample. ``prompt`` is shown to F5; ``verifier_id``
    selects the D2 verifier. Empty ``prompt`` is :class:`SynthError`."""

    problem_id: str
    domain: Domain
    prompt: str
    verifier_id: str


@runtime_checkable
class Verifier(Protocol):
    """D2 check. True iff ``RewardResponse.passed`` is true."""

    def verify(self, problem: Problem, answer: str) -> bool:
        """Return True iff the answer passes the D2 verifier for ``problem``."""
        ...


@dataclass(frozen=True)
class Sample:
    """One completion. ``text`` is the raw generator output, ``answer`` is
    :func:`extract_answer` of that text, ``passed`` is the D2 result."""

    problem_id: str
    text: str
    answer: str
    passed: bool
    token_count: int


@dataclass(frozen=True)
class RejectionResult:
    """All ``n`` samples for one problem, accepted and rejected."""

    problem_id: str
    n_generated: int
    n_accepted: int
    samples: tuple[Sample, ...]

    def verified_correct_rate(self) -> float:
        """``n_accepted / n_generated``. Zero generated is 0.0."""
        if self.n_generated == 0:
            return 0.0
        return self.n_accepted / self.n_generated


@dataclass(frozen=True)
class SamplerConfig:
    """Batch and decoding knobs. ``n_samples`` is the default group size."""

    n_samples: int = DEFAULT_N_SAMPLES
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    max_batch: int = DEFAULT_MAX_BATCH


def generate_solve_prompt(problem: Problem) -> str:
    """Prompt F5 to solve ``problem``.

    Empty ``problem.prompt`` raises :class:`SynthError`. The template is the
    contract between Python and Rust so both languages send the same bytes
    to F5.
    """
    if problem.prompt == "":
        raise SynthError("empty prompt")
    return (
        "Solve the following "
        + problem.domain.value
        + " problem. Put the final answer after the reasoning.\n\nProblem:\n"
        + problem.prompt
    )


def extract_answer(text: str) -> str:
    """Extract the answer the D2 verifier should see.

    1. Strip trailing whitespace. Empty is empty.
    2. If a line matches ``(?i)^answer\\s*:\\s*(.*)$``, the capture of the
       last such line with a non-empty capture is the answer (trimmed).
    3. Else if the text contains a fenced block starting with three
       backticks, the body of the last fence is the answer (strip one
       leading newline after the optional language tag line).
    4. Else the last non-empty line.

    This is the only parse step. The verifier does not re-parse the reasoning.
    """
    raise NotImplementedError("E2 extract_answer")


def verified_correct_rate(results: Sequence[RejectionResult]) -> float:
    """Gate: ``sum n_accepted / sum n_generated`` over ``results``. Empty or
    zero generated is 0.0."""
    generated = sum(r.n_generated for r in results)
    accepted = sum(r.n_accepted for r in results)
    if generated == 0:
        return 0.0
    return accepted / generated


class RejectionSampler:
    """Batch rejection sampling. ``generator`` is F5. ``verifier`` is D2.
    ``tokenizer`` is F6 when token counts should match the frozen vocab;
    omit it to count whitespace words.
    """

    def __init__(
        self,
        generator: Generator,
        verifier: Verifier,
        config: SamplerConfig | None = None,
        tokenizer: TokenCounter | None = None,
    ) -> None:
        self.generator = generator
        self.verifier = verifier
        self.config = SamplerConfig() if config is None else config
        self.tokenizer = tokenizer

    def sample(self, problem: Problem, n: int | None = None) -> RejectionResult:
        """``n`` completions for one problem. ``n == 0`` raises
        :class:`SynthError`. Empty prompt raises :class:`SynthError`.
        Default ``n`` is ``config.n_samples``.
        """
        raise NotImplementedError("E2 RejectionSampler.sample")

    def sample_many(
        self, problems: Sequence[Problem], n: int | None = None
    ) -> list[RejectionResult]:
        """``n`` completions per problem, chunked by ``max_batch``. Empty
        ``problems`` raises :class:`SynthError`. Result order matches
        ``problems``. Default ``n`` is ``config.n_samples``.
        """
        raise NotImplementedError("E2 RejectionSampler.sample_many")


class Dihedral(StrEnum):
    """D4 action. Indices 0..8: identity, rot90 CW, rot180, rot270 CW,
    flip left-right, flip up-down, transpose, anti-transpose."""

    IDENTITY = "identity"
    ROT90 = "rot90"
    ROT180 = "rot180"
    ROT270 = "rot270"
    FLIP_H = "flip_h"
    FLIP_V = "flip_v"
    TRANSPOSE = "transpose"
    ANTI_TRANSPOSE = "anti_transpose"

    def index(self) -> int:
        return DIHEDRALS.index(self)

    @classmethod
    def from_index(cls, k: int) -> Dihedral:
        if k < 0 or k >= len(DIHEDRALS):
            raise SynthError(f"dihedral index {k} out of range")
        return DIHEDRALS[k]


DIHEDRALS: tuple[Dihedral, ...] = (
    Dihedral.IDENTITY,
    Dihedral.ROT90,
    Dihedral.ROT180,
    Dihedral.ROT270,
    Dihedral.FLIP_H,
    Dihedral.FLIP_V,
    Dihedral.TRANSPOSE,
    Dihedral.ANTI_TRANSPOSE,
)


class FamilyKind(StrEnum):
    """Primitive generator kind. Distinct ``(kind, params)`` pairs are
    distinct families.

    * translate: ``dx``, ``dy``, ``bg``
    * recolor: ``src``, ``dst``
    * crop: ``bg``
    * tile: ``nx``, ``ny``
    * gravity: ``dir`` (0 down, 1 up, 2 left, 3 right), ``bg``
    * mirror: ``dihedral`` (0..8)
    * scale: ``factor`` (2 or 3)
    * border: ``color``, ``width`` (1 or 2)
    """

    TRANSLATE = "translate"
    RECOLOR = "recolor"
    CROP = "crop"
    TILE = "tile"
    GRAVITY = "gravity"
    MIRROR = "mirror"
    SCALE = "scale"
    BORDER = "border"


FAMILY_KINDS: tuple[FamilyKind, ...] = tuple(FamilyKind)


def _validate_grid_cells(cells: list[list[int]]) -> None:
    if not cells or not cells[0]:
        raise SynthError("empty grid")
    rows = len(cells)
    cols = len(cells[0])
    if (
        rows < MIN_GRID_SIZE
        or rows > MAX_GRID_SIZE
        or cols < MIN_GRID_SIZE
        or cols > MAX_GRID_SIZE
    ):
        raise SynthError(f"grid size {rows}x{cols} out of range")
    for row in cells:
        if len(row) != cols:
            raise SynthError("jagged grid")
        for c in row:
            if c < 0 or c >= N_COLORS:
                raise SynthError(f"ARC color {c} out of range")


@dataclass
class Grid:
    """One ARC-style grid. Cell values are palette indices in ``0..N_COLORS``.
    Rectangular, rows and cols in ``MIN_GRID_SIZE..=MAX_GRID_SIZE``.
    Structurally the D2 grid (``list[list[int]]``)."""

    cells: list[list[int]]

    def __post_init__(self) -> None:
        _validate_grid_cells(self.cells)

    def rows(self) -> int:
        return len(self.cells)

    def cols(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    def flatten(self) -> list[int]:
        """Row-major flatten. This is the F6 ``encode_grid`` input."""
        return [c for row in self.cells for c in row]


@dataclass
class Pair:
    """One input/output pair."""

    input: Grid
    output: Grid


@dataclass
class Task:
    """One generated task. ``train`` are demonstrations; ``test`` is held out."""

    family_id: str
    seed: int
    train: list[Pair]
    test: Pair


@dataclass(frozen=True)
class FamilySpec:
    """Parameterized family. ``params`` keys are documented on
    :class:`FamilyKind`. ``family_id`` is ``"{kind}:{k}={v},..."`` with keys
    sorted."""

    kind: FamilyKind
    params: tuple[tuple[str, int], ...] = ()

    @staticmethod
    def make(kind: FamilyKind, params: dict[str, int] | None = None) -> FamilySpec:
        items = tuple(sorted((params or {}).items()))
        return FamilySpec(kind=kind, params=items)

    def family_id(self) -> str:
        if not self.params:
            return self.kind.value
        body = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.kind.value}:{body}"

    def params_dict(self) -> dict[str, int]:
        return dict(self.params)


@runtime_checkable
class Family(Protocol):
    """Generator for one family. ``generate`` is deterministic in ``seed``."""

    def id(self) -> str: ...

    def kind(self) -> FamilyKind: ...

    def generate(self, seed: int) -> Task: ...


@runtime_checkable
class GridTokenizer(Protocol):
    """F6 ``encode_grid``. Implementors wrap ``tokenizer.Tokenizer.encode_grid``."""

    def encode_grid(self, cells: Sequence[int]) -> list[int]: ...


@dataclass(frozen=True)
class DiversityStats:
    """Gate numbers. ``collision_rate`` is ``1 - unique_tasks / n_tasks``,
    or 0.0 when ``n_tasks == 0``. Hashes are in-process identity of the
    grid cells, not a stable digest."""

    n_tasks: int
    n_families: int
    unique_test_inputs: int
    unique_test_outputs: int
    unique_tasks: int
    color_histogram: tuple[int, ...]
    unique_shapes: int
    collision_rate: float


def apply_dihedral(grid: Grid, dihedral: Dihedral) -> Grid:
    """Apply a D4 action. ``rot90`` is clockwise. Output size swaps on rot90,
    rot270, transpose, anti-transpose."""
    raise NotImplementedError("E3 apply_dihedral")


def apply_color_perm(grid: Grid, perm: Sequence[int]) -> Grid:
    """``perm[c]`` is the new color of ``c``. Must be a permutation of
    ``0..N_COLORS``."""
    raise NotImplementedError("E3 apply_color_perm")


def augment_pair(pair: Pair, dihedral: Dihedral, perm: Sequence[int]) -> Pair:
    """Same dihedral and color perm on every grid in the pair."""
    raise NotImplementedError("E3 augment_pair")


def augment_task(task: Task, dihedral: Dihedral, perm: Sequence[int]) -> Task:
    """Same dihedral and color perm on every grid in the task. ``family_id``
    and ``seed`` are unchanged. Augmentation is a training-time view, not a
    new family."""
    raise NotImplementedError("E3 augment_task")


def tokenize_grid(tok: GridTokenizer, grid: Grid) -> list[int]:
    """F6 encode of ``grid.flatten()``."""
    raise NotImplementedError("E3 tokenize_grid")


def family_from_spec(spec: FamilySpec) -> Family:
    """Construct a family from a spec. Unknown or invalid params raise
    :class:`SynthError`."""
    raise NotImplementedError("E3 family_from_spec")


def sample_family_specs(n: int, seed: int) -> list[FamilySpec]:
    """``n`` distinct specs from the parameter grid, deterministic in
    ``seed``."""
    raise NotImplementedError("E3 sample_family_specs")


def diversity_stats(tasks: Sequence[Task]) -> DiversityStats:
    """Gate: uniqueness and color/shape coverage over ``tasks``. Empty is
    zeros and ``collision_rate == 0.0``."""
    raise NotImplementedError("E3 diversity_stats")


class ProceduralCorpus:
    """A bag of families to sample from."""

    def __init__(
        self,
        specs: Sequence[FamilySpec],
        seed: int,
        n_train: int = DEFAULT_N_TRAIN,
    ) -> None:
        self.specs = list(specs)
        self.seed = seed
        self.n_train = n_train

    def sample(self, n: int) -> list[Task]:
        """``n`` tasks, cycling specs, seeds ``seed + i``. Empty ``specs``
        raises :class:`SynthError`. ``n == 0`` raises :class:`SynthError`.
        """
        raise NotImplementedError("E3 ProceduralCorpus.sample")

