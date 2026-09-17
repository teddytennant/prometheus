"""Independent slow reference for E2 rejection sampling (spec 7.1, 15.5 E2).

This is the source of truth for extract_answer / sample / sample_many.
Production ``synth`` must match these results. Production must never import
``tests/``. Must match ``synth/tests/common/reject_ref.rs`` on the same inputs.

Pinned rules
------------
extract_answer
    1. Strip trailing Unicode whitespace (``str.rstrip`` / ``trim_end``). A
       zero-length result is ``""``. Leading whitespace is kept.
    2. Split into lines (universal newlines). If a line matches
       ``(?i)^answer\\s*:\\s*(.*)$``, take the last line whose capture is
       non-empty (zero-length after the colon-whitespace, not after trim).
       Return that capture stripped. ``Answer:`` with an empty capture does
       not count and does not stop the scan. Leading whitespace on the line
       means ``^answer`` fails, so ``"  Answer: 42"`` is not an Answer line.
       ``answered: 9`` is not an Answer line (need whitespace or colon after
       ``answer``). ASCII case folding only for the word ``answer``.
    3. Else scan for complete fenced blocks. An opening fence is three
       backticks at the start of the text or just after a newline. The rest
       of that opening line is the optional language tag and is discarded.
       There must be a newline after the opening backticks (the language tag
       line). Strip that one newline. The body runs until the next three
       backticks. No such closer means the opening is unclosed and is not a
       fence. The last complete fence body is the answer, including any
       interior newlines and a newline just before the closer. An empty body
       is a valid answer (``""``), not a fall-through. A single-line
       ````code```` with no newline after the opener is not a complete fence.
    4. Else the last line whose ``strip()`` / ``trim()`` is non-empty, returned
       as-is (leading whitespace kept). No such line is ``""``.

    Answer: wins over fences. Empty Answer: captures fall through to fences
    then to the last non-empty line. This is the only parse step.

token_count
    Empty string returns 0 without calling a tokenizer.
    If ``tokenizer`` is given: ``len(tokenizer.encode(text))``.
    Else Unicode whitespace words: ``len(text.split())``.
    Counted on the raw completion, not the extracted answer.

sample(problem, n)
    ``n is None`` uses ``config.n_samples``. ``n == 0`` is SynthError
    ``n_samples must be > 0`` before looking at the prompt or calling
    generate. Empty ``problem.prompt`` is SynthError ``empty prompt`` (via
    generate_solve_prompt) before generate. Build the solve prompt once, then
    ``n`` identical copies. Chunk by ``config.max_batch`` (last chunk may be
    shorter). ``max_batch`` in tests is always >= 1. Each chunk is one
    ``Generator.generate`` with ``config.max_tokens`` and
    ``config.temperature``. Length mismatch is SynthError. For each
    completion, in order: extract_answer, verifier.verify(problem, answer),
    token_count on the raw text. Return every sample (passed and failed).
    ``n_generated`` is ``n``. ``n_accepted`` is the number of ``passed``.

sample_many(problems, n)
    Empty ``problems`` is SynthError ``empty batch`` before n, generate, or
    verify. Then ``n is None`` defaults. ``n == 0`` is ZeroSamples. For each
    problem in order, n copies of its solve prompt (problem-major). Flatten,
    chunk by max_batch, generate, split back into per-problem results of
    length n. Empty prompt in any problem errors before generate. Result
    order matches ``problems``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from synth import (
    Generator,
    Problem,
    RejectionResult,
    Sample,
    SamplerConfig,
    SynthError,
    TokenCounter,
    Verifier,
    generate_solve_prompt,
)

_ANSWER_LINE = re.compile(r"(?i)^answer\s*:\s*(.*)$")


def token_count(text: str, tokenizer: TokenCounter | None = None) -> int:
    """Encode length if a tokenizer is given, else whitespace words."""
    if text == "":
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def _last_fence_body(text: str) -> str | None:
    """Body of the last complete ``` fence, or None if none / only unclosed."""
    last: str | None = None
    i = 0
    while True:
        idx = text.find("```", i)
        if idx < 0:
            break
        if idx != 0 and text[idx - 1] != "\n":
            i = idx + 3
            continue
        nl = text.find("\n", idx + 3)
        if nl < 0:
            break
        close = text.find("```", nl + 1)
        if close < 0:
            break
        last = text[nl + 1 : close]
        i = close + 3
    return last


def extract_answer(text: str) -> str:
    """Extract the answer the D2 verifier should see. See module docstring."""
    text = text.rstrip()
    if text == "":
        return ""

    last_ans: str | None = None
    for line in text.splitlines():
        m = _ANSWER_LINE.match(line)
        if m is None:
            continue
        cap = m.group(1)
        if cap:
            last_ans = cap.strip()
    if last_ans is not None:
        return last_ans

    body = _last_fence_body(text)
    if body is not None:
        return body

    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return ""


def _generate_chunked(
    generator: Generator,
    prompts: Sequence[str],
    max_tokens: int,
    temperature: float,
    max_batch: int,
) -> list[str]:
    completions: list[str] = []
    for i in range(0, len(prompts), max_batch):
        chunk = list(prompts[i : i + max_batch])
        outs = generator.generate(chunk, max_tokens, temperature)
        if len(outs) != len(chunk):
            raise SynthError(f"generator returned {len(outs)} completions for {len(chunk)} prompts")
        completions.extend(outs)
    return completions


def _one_sample(
    problem: Problem,
    text: str,
    verifier: Verifier,
    tokenizer: TokenCounter | None,
) -> Sample:
    answer = extract_answer(text)
    passed = verifier.verify(problem, answer)
    return Sample(
        problem_id=problem.problem_id,
        text=text,
        answer=answer,
        passed=passed,
        token_count=token_count(text, tokenizer),
    )


class ReferenceRejectionSampler:
    """Same contract as ``synth.RejectionSampler``, implemented here."""

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
        if n is None:
            n = self.config.n_samples
        if n == 0:
            raise SynthError("n_samples must be > 0")
        prompt = generate_solve_prompt(problem)
        prompts = [prompt] * n
        completions = _generate_chunked(
            self.generator,
            prompts,
            self.config.max_tokens,
            self.config.temperature,
            self.config.max_batch,
        )
        samples = [
            _one_sample(problem, text, self.verifier, self.tokenizer) for text in completions
        ]
        n_accepted = sum(1 for s in samples if s.passed)
        return RejectionResult(
            problem_id=problem.problem_id,
            n_generated=n,
            n_accepted=n_accepted,
            samples=tuple(samples),
        )

    def sample_many(
        self, problems: Sequence[Problem], n: int | None = None
    ) -> list[RejectionResult]:
        if len(problems) == 0:
            raise SynthError("empty batch")
        if n is None:
            n = self.config.n_samples
        if n == 0:
            raise SynthError("n_samples must be > 0")
        prompts: list[str] = []
        for problem in problems:
            prompt = generate_solve_prompt(problem)
            prompts.extend([prompt] * n)
        completions = _generate_chunked(
            self.generator,
            prompts,
            self.config.max_tokens,
            self.config.temperature,
            self.config.max_batch,
        )
        results: list[RejectionResult] = []
        idx = 0
        for problem in problems:
            texts = completions[idx : idx + n]
            idx += n
            samples = [_one_sample(problem, text, self.verifier, self.tokenizer) for text in texts]
            n_accepted = sum(1 for s in samples if s.passed)
            results.append(
                RejectionResult(
                    problem_id=problem.problem_id,
                    n_generated=n,
                    n_accepted=n_accepted,
                    samples=tuple(samples),
                )
            )
        return results


__all__ = [
    "ReferenceRejectionSampler",
    "extract_answer",
    "token_count",
]
