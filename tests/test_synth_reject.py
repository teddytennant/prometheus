"""E2 rejection-sampling oracle tests (spec 7.1, 15.5).

Production ``synth.extract_answer`` / ``RejectionSampler.sample`` /
``RejectionSampler.sample_many`` must match ``tests.reference.reject``.
Tests of already-implemented helpers (generate_solve_prompt,
verified_correct_rate, types) may pass on the stub. Every test that
calls a stub must fail until the implementer fills it in.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from synth import (
    DEFAULT_MAX_BATCH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_SAMPLES,
    DEFAULT_TEMPERATURE,
    Domain,
    Problem,
    RejectionResult,
    RejectionSampler,
    Sample,
    SamplerConfig,
    SynthError,
    extract_answer,
    generate_solve_prompt,
    verified_correct_rate,
)
from tests.reference import reject as ref

# 2 accepted / 8 generated. Completions 1-indexed in GOLDEN_COMPLETIONS:
# 2 and 4 extract to "42" and MatchVerifier accepts only "42".
GOLDEN_VERIFIED_CORRECT_RATE = 0.25
GOLDEN_N_GENERATED = 8
GOLDEN_N_ACCEPTED = 2
GOLDEN_EXPECTED_ANSWER = "42"

GOLDEN_COMPLETIONS = [
    "I think the result is 41.\nAnswer: 41",
    "Work shown.\nAnswer: 42",
    "```python\nprint(0)\n```",
    "Answer: 42",
    "nope",
    "Answer: 40",
    "the last line is 99",
    "Answer: 7",
]

# (text, expected). Shared with the Rust oracle; both references must match.
EXTRACT_GOLDENS: list[tuple[str, str]] = [
    ("", ""),
    ("   \t\n", ""),
    ("   ", ""),
    ("Answer: first\nAnswer: second", "second"),
    ("Answer: first\nReasoning\nAnswer: second", "second"),
    ("ANSWER: 42", "42"),
    ("answer: 42", "42"),
    ("AnSwEr: 42", "42"),
    ("answer : 42", "42"),
    ("answer:42", "42"),
    ("Answer:\n42", "42"),
    ("Answer:   \n42", "42"),
    ("Answer:\nAnswer: 7", "7"),
    ("Answer: first\nAnswer:", "first"),
    ("Answer: first\nAnswer:   ", "first"),
    ("```\nfirst\n```\n```\nsecond\n```", "second\n"),
    ("```python\nprint(1)\n```", "print(1)\n"),
    ("thinking\n```python\nprint(1)", "print(1)"),
    ("line1\nline2\nline3", "line3"),
    ("line1\n\nline2", "line2"),
    ("Answer: 42   \n\n", "42"),
    ("```\ncode\n```\nAnswer: 99", "99"),
    ("Answer:\n```\ncode\n```", "code\n"),
    ("  Answer: 42", "  Answer: 42"),
    ("foo\n  42", "  42"),
    ("```code```", "```code```"),
    ("```\n```", ""),
    ("hello\n```\n```", ""),
    ("```python\nprint(1)```", "print(1)"),
    ("Answer: 0", "0"),
    ("Answer: false", "false"),
    ("answered: 9\n9", "9"),
    ("Answer: Café", "Café"),
    ("the answer is 42", "the answer is 42"),
    ("42", "42"),
    ("  hello", "  hello"),
    ("Answer:  foo  \nbar", "foo"),
    ("answer\t:\t42", "42"),
    ("Answer :42", "42"),
    ("ANSWER:\n42", "42"),
    ("```rust\nfn main() {}\n```", "fn main() {}\n"),
    ("see ```code``` here", "see ```code``` here"),
    ("foo\n```\nbar\n```\nbaz", "bar\n"),
    ("Answer: 1\nAnswer:\nAnswer: 3", "3"),
    ("```\nfirst\n```\n```python\nsecond\n```", "second\n"),
    ("hello\n```\nunclosed", "unclosed"),
    ("Answer:   42", "42"),
    ("\n\nAnswer: 5\n", "5"),
    ("line1\n   \nline2", "line2"),
    ("```\nbody\n```", "body\n"),
    ("``` python\nx\n```", "x\n"),
    ("lots of words here\nAnswer: 42", "42"),
    ("Answer: 42 extra", "42 extra"),
    ("answer: \nlast", "last"),
    ("```\nfirst\n```\nnot a fence", "first\n"),
    ("foo\n\nbar\n\n", "bar"),
    ("Answer:\nAnswer:\nhello", "hello"),
    ("```\n\n```", "\n"),
    ("x\n```python\n\ny\n```", "\ny\n"),
]

PROPERTY_TEXTS = [text for text, _ in EXTRACT_GOLDENS] + [
    "What is 2+2?\nAnswer: 4",
    "```js\nconsole.log(1)\n```",
    "no answer here\nstill none",
    "Answer:\n```python\nx = 1\n```",
]


def _problem(
    pid: str = "p1",
    prompt: str = "What is 6*7?",
    domain: Domain = Domain.MATH,
    verifier_id: str = GOLDEN_EXPECTED_ANSWER,
) -> Problem:
    return Problem(problem_id=pid, domain=domain, prompt=prompt, verifier_id=verifier_id)


class QueueGenerator:
    """Yields the next completions in order, recording each generate call."""

    def __init__(self, completions: Sequence[str]) -> None:
        self._completions = list(completions)
        self._i = 0
        self.calls: list[tuple[list[str], int, float]] = []

    def generate(self, prompts: Sequence[str], max_tokens: int, temperature: float) -> list[str]:
        self.calls.append((list(prompts), int(max_tokens), float(temperature)))
        n = len(prompts)
        out = self._completions[self._i : self._i + n]
        self._i += n
        if len(out) != n:
            raise AssertionError("not enough scripted completions")
        return list(out)


class MismatchGenerator:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls: list[tuple[list[str], int, float]] = []

    def generate(self, prompts: Sequence[str], max_tokens: int, temperature: float) -> list[str]:
        self.calls.append((list(prompts), int(max_tokens), float(temperature)))
        return ["x"] * self.n


class BoomGenerator:
    def generate(self, prompts: Sequence[str], max_tokens: int, temperature: float) -> list[str]:
        raise AssertionError("generator must not be called")


class MatchVerifier:
    """Accept iff the extracted answer equals problem.verifier_id."""

    def verify(self, problem: Problem, answer: str) -> bool:
        return answer == problem.verifier_id


class ExactVerifier:
    def __init__(self, expected: str) -> None:
        self.expected = expected
        self.seen: list[tuple[str, str]] = []

    def verify(self, problem: Problem, answer: str) -> bool:
        self.seen.append((problem.problem_id, answer))
        return answer == self.expected


class BoomVerifier:
    def verify(self, problem: Problem, answer: str) -> bool:
        raise AssertionError("verifier must not be called")


class VerifierBoom(Exception):
    """Distinct from NotImplementedError so the stub still fails this test."""


class RaisingVerifier:
    def verify(self, problem: Problem, answer: str) -> bool:
        raise VerifierBoom("verifier failed")


class FakeTokenizer:
    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.seen: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.seen.append(text)
        return list(range(self.n))


class LenTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class BoomTokenizer:
    def encode(self, text: str) -> list[int]:
        raise AssertionError(f"encode must not be called for {text!r}")


def _cfg(**kwargs: object) -> SamplerConfig:
    return SamplerConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Already implemented: types, generate_solve_prompt, verified_correct_rate
# ---------------------------------------------------------------------------


def test_defaults_and_domains() -> None:
    assert DEFAULT_N_SAMPLES == 8
    assert DEFAULT_MAX_TOKENS == 512
    assert abs(DEFAULT_TEMPERATURE - 0.7) < 1e-6
    assert DEFAULT_MAX_BATCH == 8
    assert list(Domain) == [
        Domain.MATH,
        Domain.CODE,
        Domain.LEAN,
        Domain.GRID,
        Domain.MARKET,
    ]
    assert Domain.MATH == "math"
    assert Domain.CODE == "code"
    assert Domain.LEAN == "lean"
    assert Domain.GRID == "grid"
    assert Domain.MARKET == "market"
    cfg = SamplerConfig()
    assert cfg.n_samples == 8
    assert cfg.max_tokens == 512
    assert abs(cfg.temperature - 0.7) < 1e-6
    assert cfg.max_batch == 8


def test_generate_solve_prompt_template() -> None:
    p = _problem(prompt="What is 2+2?")
    prompt = generate_solve_prompt(p)
    assert prompt == (
        "Solve the following math problem. Put the final answer after the reasoning."
        "\n\nProblem:\nWhat is 2+2?"
    )


def test_generate_solve_prompt_embeds_every_domain() -> None:
    for domain in Domain:
        p = _problem(domain=domain, prompt="Body.")
        prompt = generate_solve_prompt(p)
        assert prompt.startswith(
            f"Solve the following {domain} problem. Put the final answer after the reasoning."
        )
        assert prompt.endswith("Problem:\nBody.")


def test_generate_solve_prompt_empty_raises() -> None:
    with pytest.raises(SynthError, match="empty prompt"):
        generate_solve_prompt(_problem(prompt=""))


def test_generate_solve_prompt_preserves_utf8() -> None:
    p = _problem(prompt="Café mass 3.14?")
    assert "Café mass 3.14?" in generate_solve_prompt(p)


def test_verified_correct_rate_empty_or_zero_is_zero() -> None:
    assert verified_correct_rate([]) == 0.0
    empty = RejectionResult("p", 0, 0, ())
    assert verified_correct_rate([empty]) == 0.0
    assert empty.verified_correct_rate() == 0.0


def test_verified_correct_rate_sums_accepted_over_generated() -> None:
    a = RejectionResult("a", 8, 2, ())
    b = RejectionResult("b", 8, 0, ())
    assert a.verified_correct_rate() == GOLDEN_VERIFIED_CORRECT_RATE
    assert verified_correct_rate([a, b]) == 0.125


# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", EXTRACT_GOLDENS)
def test_extract_answer_goldens(text: str, expected: str) -> None:
    assert extract_answer(text) == expected
    assert extract_answer(text) == ref.extract_answer(text)


def test_extract_answer_matches_reference_on_property_texts() -> None:
    for text in PROPERTY_TEXTS:
        assert extract_answer(text) == ref.extract_answer(text), repr(text)


def test_extract_answer_empty_and_whitespace() -> None:
    assert extract_answer("") == ""
    assert extract_answer("   \n\t") == ""


def test_extract_answer_last_answer_wins() -> None:
    assert extract_answer("Answer: first\nAnswer: second") == "second"


def test_extract_answer_case_insensitive() -> None:
    assert extract_answer("ANSWER: 42") == "42"
    assert extract_answer("AnSwEr: 42") == "42"


def test_extract_answer_empty_capture_falls_through() -> None:
    assert extract_answer("Answer:\n42") == "42"
    assert extract_answer("Answer: first\nAnswer:") == "first"
    assert extract_answer("Answer:\n```\ncode\n```") == "code\n"


def test_extract_answer_last_fence_wins_language_tag_stripped() -> None:
    assert extract_answer("```\nfirst\n```\n```python\nsecond\n```") == "second\n"


def test_extract_answer_unclosed_fence_falls_through() -> None:
    assert extract_answer("hello\n```\nunclosed") == "unclosed"
    assert extract_answer("```code```") == "```code```"


def test_extract_answer_last_nonempty_line() -> None:
    assert extract_answer("line1\nline2\nline3") == "line3"
    assert extract_answer("foo\n  42") == "  42"


def test_extract_answer_trailing_whitespace_stripped_before_parse() -> None:
    assert extract_answer("Answer: 42   \n\n") == "42"
    assert extract_answer("foo\n\nbar\n\n") == "bar"


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_zero_n_errors_before_generate() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier(), _cfg(n_samples=8))
    with pytest.raises(SynthError, match="n_samples must be > 0"):
        sampler.sample(_problem(), n=0)


def test_sample_none_uses_config_n_samples_zero() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier(), _cfg(n_samples=0))
    with pytest.raises(SynthError, match="n_samples must be > 0"):
        sampler.sample(_problem())


def test_sample_empty_prompt_errors_before_generate() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier())
    with pytest.raises(SynthError, match="empty prompt"):
        sampler.sample(_problem(prompt=""), n=3)


def test_sample_zero_n_wins_over_empty_prompt() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier())
    with pytest.raises(SynthError, match="n_samples must be > 0"):
        sampler.sample(_problem(prompt=""), n=0)


def test_sample_length_mismatch() -> None:
    sampler = RejectionSampler(MismatchGenerator(0), MatchVerifier(), _cfg(max_batch=8))
    with pytest.raises(SynthError, match="generator returned 0 completions for 2 prompts"):
        sampler.sample(_problem(), n=2)


def test_sample_matches_reference_golden_rate() -> None:
    problem = _problem()
    cfg = _cfg(n_samples=8, max_batch=3, max_tokens=64, temperature=0.2)
    prod = RejectionSampler(QueueGenerator(GOLDEN_COMPLETIONS), MatchVerifier(), cfg).sample(
        problem, n=GOLDEN_N_GENERATED
    )
    expected = ref.ReferenceRejectionSampler(
        QueueGenerator(GOLDEN_COMPLETIONS), MatchVerifier(), cfg
    ).sample(problem, n=GOLDEN_N_GENERATED)
    assert prod == expected
    assert prod.n_generated == GOLDEN_N_GENERATED
    assert prod.n_accepted == GOLDEN_N_ACCEPTED
    assert prod.verified_correct_rate() == GOLDEN_VERIFIED_CORRECT_RATE
    assert verified_correct_rate([prod]) == GOLDEN_VERIFIED_CORRECT_RATE
    assert [s.passed for s in prod.samples] == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert [s.answer for s in prod.samples] == [
        "41",
        "42",
        "print(0)\n",
        "42",
        "nope",
        "40",
        "the last line is 99",
        "7",
    ]
    assert all(s.problem_id == "p1" for s in prod.samples)
    assert all(s.text == c for s, c in zip(prod.samples, GOLDEN_COMPLETIONS, strict=True))


def test_sample_n_none_defaults_to_config() -> None:
    cfg = _cfg(n_samples=3, max_batch=8)
    completions = ["Answer: 1", "Answer: 2", "Answer: 3"]
    problem = _problem(verifier_id="2")
    prod = RejectionSampler(QueueGenerator(completions), MatchVerifier(), cfg).sample(problem)
    expected = ref.ReferenceRejectionSampler(
        QueueGenerator(completions), MatchVerifier(), cfg
    ).sample(problem)
    assert prod == expected
    assert prod.n_generated == 3
    assert prod.n_accepted == 1


def test_sample_chunks_by_max_batch() -> None:
    cfg = _cfg(max_batch=2, max_tokens=11, temperature=0.5)
    completions = ["Answer: a", "Answer: b", "Answer: c", "Answer: d", "Answer: e"]
    gen = QueueGenerator(completions)
    problem = _problem()
    prompt = generate_solve_prompt(problem)
    RejectionSampler(gen, MatchVerifier(), cfg).sample(problem, n=5)
    assert [len(c[0]) for c in gen.calls] == [2, 2, 1]
    for prompts, max_tokens, temperature in gen.calls:
        assert prompts == [prompt] * len(prompts)
        assert max_tokens == 11
        assert temperature == pytest.approx(0.5)


def test_sample_keeps_rejected_traces() -> None:
    cfg = _cfg(max_batch=8)
    completions = ["Answer: no", "Answer: 42"]
    prod = RejectionSampler(QueueGenerator(completions), MatchVerifier(), cfg).sample(
        _problem(), n=2
    )
    assert prod.n_generated == 2
    assert prod.n_accepted == 1
    assert len(prod.samples) == 2
    assert prod.samples[0].passed is False
    assert prod.samples[1].passed is True


def test_sample_whitespace_token_count_on_raw_text() -> None:
    cfg = _cfg(max_batch=8)
    text = "lots of words here\nAnswer: 42"
    prod = RejectionSampler(QueueGenerator([text]), MatchVerifier(), cfg).sample(_problem(), n=1)
    expected = ref.ReferenceRejectionSampler(QueueGenerator([text]), MatchVerifier(), cfg).sample(
        _problem(), n=1
    )
    assert prod == expected
    assert prod.samples[0].token_count == 6
    assert prod.samples[0].answer == "42"


def test_sample_tokenizer_encode_length() -> None:
    cfg = _cfg(max_batch=8)
    text = "Answer: 42"
    tok = FakeTokenizer(n=9)
    prod = RejectionSampler(QueueGenerator([text]), MatchVerifier(), cfg, tokenizer=tok).sample(
        _problem(), n=1
    )
    assert prod.samples[0].token_count == 9
    assert tok.seen == [text]


def test_sample_empty_completion_skips_tokenizer() -> None:
    cfg = _cfg(max_batch=8)
    tok = FakeTokenizer(n=3)
    prod = RejectionSampler(
        QueueGenerator(["", "Answer: 42"]),
        MatchVerifier(),
        cfg,
        tokenizer=tok,
    ).sample(_problem(), n=2)
    expected = ref.ReferenceRejectionSampler(
        QueueGenerator(["", "Answer: 42"]),
        MatchVerifier(),
        cfg,
        tokenizer=LenTokenizer(),
    ).sample(_problem(), n=2)
    assert prod.samples[0].token_count == 0
    assert prod.samples[1].token_count == 3
    assert tok.seen == ["Answer: 42"]
    assert prod.samples[0].answer == expected.samples[0].answer
    assert prod.samples[1].answer == expected.samples[1].answer


def test_sample_verifier_sees_extracted_answer_not_raw_text() -> None:
    cfg = _cfg(max_batch=8)
    v = ExactVerifier("42")
    text = "reasoning\nAnswer: 42"
    RejectionSampler(QueueGenerator([text]), v, cfg).sample(_problem(), n=1)
    assert v.seen == [("p1", "42")]


def test_sample_raising_verifier_propagates() -> None:
    sampler = RejectionSampler(QueueGenerator(["Answer: 42"]), RaisingVerifier(), _cfg(max_batch=8))
    with pytest.raises(VerifierBoom, match="verifier failed"):
        sampler.sample(_problem(), n=1)


# ---------------------------------------------------------------------------
# sample_many
# ---------------------------------------------------------------------------


def test_sample_many_empty_batch_errors_before_generate() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier())
    with pytest.raises(SynthError, match="empty batch"):
        sampler.sample_many([], n=3)


def test_sample_many_empty_batch_wins_over_zero_n() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier())
    with pytest.raises(SynthError, match="empty batch"):
        sampler.sample_many([], n=0)


def test_sample_many_zero_n_errors_before_generate() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier())
    with pytest.raises(SynthError, match="n_samples must be > 0"):
        sampler.sample_many([_problem()], n=0)


def test_sample_many_empty_prompt_errors_before_generate() -> None:
    sampler = RejectionSampler(BoomGenerator(), BoomVerifier())
    with pytest.raises(SynthError, match="empty prompt"):
        sampler.sample_many([_problem(), _problem(pid="p2", prompt="")], n=2)


def test_sample_many_length_mismatch() -> None:
    sampler = RejectionSampler(MismatchGenerator(1), MatchVerifier(), _cfg(max_batch=8))
    with pytest.raises(SynthError, match="generator returned 1 completions for 4 prompts"):
        sampler.sample_many([_problem(), _problem(pid="p2")], n=2)


def test_sample_many_matches_reference_and_preserves_order() -> None:
    problems = [
        _problem(pid="a", prompt="one", verifier_id="1"),
        _problem(pid="b", prompt="two", domain=Domain.CODE, verifier_id="2"),
    ]
    completions = ["Answer: 1", "Answer: x", "Answer: 2", "Answer: 2"]
    cfg = _cfg(max_batch=3, max_tokens=20, temperature=0.1)
    prod = RejectionSampler(QueueGenerator(completions), MatchVerifier(), cfg).sample_many(
        problems, n=2
    )
    expected = ref.ReferenceRejectionSampler(
        QueueGenerator(completions), MatchVerifier(), cfg
    ).sample_many(problems, n=2)
    assert prod == expected
    assert [r.problem_id for r in prod] == ["a", "b"]
    assert prod[0].n_generated == 2
    assert prod[0].n_accepted == 1
    assert prod[1].n_generated == 2
    assert prod[1].n_accepted == 2
    assert [s.text for s in prod[0].samples] == ["Answer: 1", "Answer: x"]
    assert [s.text for s in prod[1].samples] == ["Answer: 2", "Answer: 2"]


def test_sample_many_chunks_across_problems() -> None:
    problems = [_problem(pid="a", prompt="one"), _problem(pid="b", prompt="two")]
    completions = ["A", "B", "C", "D"]
    gen = QueueGenerator(completions)
    cfg = _cfg(max_batch=3, max_tokens=5, temperature=0.0)
    RejectionSampler(gen, MatchVerifier(), cfg).sample_many(problems, n=2)
    p1 = generate_solve_prompt(problems[0])
    p2 = generate_solve_prompt(problems[1])
    assert [c[0] for c in gen.calls] == [[p1, p1, p2], [p2]]
    assert all(c[1] == 5 and c[2] == pytest.approx(0.0) for c in gen.calls)


def test_sample_many_n_none_defaults_to_config() -> None:
    cfg = _cfg(n_samples=2, max_batch=8)
    problems = [_problem(pid="a"), _problem(pid="b", prompt="other")]
    completions = ["Answer: 42", "no", "Answer: 42", "Answer: 42"]
    prod = RejectionSampler(QueueGenerator(completions), MatchVerifier(), cfg).sample_many(problems)
    expected = ref.ReferenceRejectionSampler(
        QueueGenerator(completions), MatchVerifier(), cfg
    ).sample_many(problems)
    assert prod == expected
    assert [r.n_generated for r in prod] == [2, 2]


def test_sample_many_golden_aggregate_rate() -> None:
    problems = [_problem(pid="a"), _problem(pid="b", prompt="other")]
    # 2 accepted out of 8 across both problems.
    completions = list(GOLDEN_COMPLETIONS)
    cfg = _cfg(max_batch=5)
    prod = RejectionSampler(QueueGenerator(completions), MatchVerifier(), cfg).sample_many(
        problems, n=4
    )
    expected = ref.ReferenceRejectionSampler(
        QueueGenerator(completions), MatchVerifier(), cfg
    ).sample_many(problems, n=4)
    assert prod == expected
    assert sum(r.n_accepted for r in prod) == GOLDEN_N_ACCEPTED
    assert sum(r.n_generated for r in prod) == GOLDEN_N_GENERATED
    assert verified_correct_rate(prod) == GOLDEN_VERIFIED_CORRECT_RATE


# ---------------------------------------------------------------------------
# Property: several problems x n x max_batch, production == reference
# ---------------------------------------------------------------------------


def test_sample_property_grid_matches_reference() -> None:
    problem = _problem(pid="grid", prompt="sum?", verifier_id="yes")
    pool = [
        "Answer: yes",
        "Answer: no",
        "```\nyes\n```",
        "last line",
        "Answer:",
        "yes",
        "```python\nno\n```",
        "Answer: yes",
        "thinking\nAnswer: no",
        "```\n```",
    ]
    for n in (1, 2, 5):
        for max_batch in (1, 2, 3, 8):
            completions = pool[:n]
            cfg = _cfg(n_samples=n, max_batch=max_batch, max_tokens=32, temperature=0.3)
            prod = RejectionSampler(QueueGenerator(completions), MatchVerifier(), cfg).sample(
                problem, n=n
            )
            expected = ref.ReferenceRejectionSampler(
                QueueGenerator(completions), MatchVerifier(), cfg
            ).sample(problem, n=n)
            assert prod == expected, (n, max_batch)


def test_sample_many_property_grid_matches_reference() -> None:
    problems = [
        _problem(pid="m1", prompt="p1", verifier_id="A"),
        _problem(pid="m2", prompt="p2", domain=Domain.CODE, verifier_id="B"),
        _problem(pid="m3", prompt="p3", domain=Domain.LEAN, verifier_id="C"),
    ]
    pool = [
        "Answer: A",
        "Answer: B",
        "Answer: C",
        "Answer: A",
        "nope",
        "```\nB\n```",
        "Answer: C",
        "last",
        "Answer: A",
        "Answer: B",
        "```python\nC\n```",
        "Answer: Z",
        "Answer: A",
        "Answer: B",
        "Answer: C",
    ]
    for n in (1, 2, 5):
        for max_batch in (1, 2, 4, 8):
            completions = pool[: n * len(problems)]
            cfg = _cfg(max_batch=max_batch, max_tokens=16, temperature=1.0)
            prod = RejectionSampler(QueueGenerator(completions), MatchVerifier(), cfg).sample_many(
                problems, n=n
            )
            expected = ref.ReferenceRejectionSampler(
                QueueGenerator(completions), MatchVerifier(), cfg
            ).sample_many(problems, n=n)
            assert prod == expected, (n, max_batch)
            assert [r.problem_id for r in prod] == ["m1", "m2", "m3"]
            assert all(r.n_generated == n for r in prod)


def test_sample_result_types() -> None:
    prod = RejectionSampler(
        QueueGenerator(["Answer: 42"]), MatchVerifier(), _cfg(max_batch=1)
    ).sample(_problem(), n=1)
    assert isinstance(prod, RejectionResult)
    assert isinstance(prod.samples, tuple)
    assert isinstance(prod.samples[0], Sample)
    assert prod.samples[0].passed is True
    assert isinstance(prod.samples[0].token_count, int)
