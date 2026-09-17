"""Implementer-facing tests for E1 synthetic rewrites.

Every production call except generate_prompt / Style / Verdict / FactCheck
counts / constants hits unimplemented ``synth`` today, so this file must be
red. After E1 is implemented, production must match ``tests.reference.synth``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth import (
    DEFAULT_MAX_BATCH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    SCHEMA_VERSION,
    STYLES,
    Claim,
    FactCheck,
    LabeledExample,
    Orchestrator,
    OrchestratorConfig,
    SourceDocument,
    Style,
    SynthError,
    Verdict,
    accept,
    check_claim,
    extract_claims,
    fact_check,
    generate_prompt,
    precision,
    token_count,
)
from tests.reference import synth as ref

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "synth"
LABELED_PATH = FIXTURE_DIR / "labeled.jsonl"
WATER_TEXT = (FIXTURE_DIR / "water.txt").read_text(encoding="utf-8").strip()
PARIS_TEXT = (FIXTURE_DIR / "paris.txt").read_text(encoding="utf-8").strip()

GOLDEN_PRECISION = 0.8
LABELED_SAMPLE_SIZE = 12

EXTRACT_GOLDENS: list[tuple[str, list[str]]] = [
    ("", []),
    ("   \t\n", []),
    ("Hello. World!", ["Hello.", "World!"]),
    ("One sentence", ["One sentence"]),
    ("First. Second! Third?", ["First.", "Second!", "Third?"]),
    ("Hello.World!", ["Hello.World!"]),
    ("  Hello.   World!  ", ["Hello.", "World!"]),
    ("Dr. Smith went home.", ["Dr.", "Smith went home."]),
    ("What?Yes.", ["What?Yes."]),
    ("Done.", ["Done."]),
    ("A. B. C.", ["A.", "B.", "C."]),
    ("Hello.\nWorld!", ["Hello.", "World!"]),
    ("Café is open. Naïve approach.", ["Café is open.", "Naïve approach."]),
    ("What??", ["What??"]),
    ("Hello. ", ["Hello."]),
]

# (source, claim, verdict). Empty source is an error path, not a golden.
CHECK_GOLDENS: list[tuple[str, str, Verdict]] = [
    (
        "Water boils at 100 degrees Celsius at sea level",
        "Water boils at 100 degrees",
        Verdict.SUPPORTED,
    ),
    (
        "Water boils at 100 degrees Celsius at sea level",
        "Water boils at 90 degrees",
        Verdict.CONTRADICTED,
    ),
    (
        "The solution is not acidic and the mixture contains salt",
        "The solution is acidic",
        Verdict.CONTRADICTED,
    ),
    (
        "There are no cats in the library",
        "There are cats in the library",
        Verdict.CONTRADICTED,
    ),
    ("Paris is the capital of France", "Berlin is in Germany", Verdict.NOT_IN_SOURCE),
    (
        "Paris is the capital of France",
        "Paris is the capital of France",
        Verdict.SUPPORTED,
    ),
    ("hello world", "   ", Verdict.NOT_IN_SOURCE),
    ("hello world", "", Verdict.NOT_IN_SOURCE),
    ("Water is a liquid at room temperature", "is a", Verdict.SUPPORTED),
    ("Water is a liquid at room temperature", "is an", Verdict.NOT_IN_SOURCE),
    (
        "The mass is 3.14 kilograms",
        "The mass is 2.71 kilograms",
        Verdict.CONTRADICTED,
    ),
    (
        "Paris is the capital of France and the city has many museums",
        "Paris has 12 museums",
        Verdict.SUPPORTED,
    ),
    ("that this with from they", "that this", Verdict.SUPPORTED),
    ("hello world", "that this", Verdict.NOT_IN_SOURCE),
    ("The cats are here", "The cats are here", Verdict.SUPPORTED),
    ("The cats are here", "The dogs are here", Verdict.NOT_IN_SOURCE),
    ("Mass is 3.14", "Volume is 3.14", Verdict.NOT_IN_SOURCE),
    ("The value is 3.14", "The value is 14", Verdict.CONTRADICTED),
    ("page 14", "page 3.14", Verdict.CONTRADICTED),
    ("not available here", "available", Verdict.CONTRADICTED),
    ("note available here", "available", Verdict.SUPPORTED),
    ("no cats here", "cats", Verdict.CONTRADICTED),
    ("nobody cats here", "cats", Verdict.SUPPORTED),
    ("The mixture contains salt", "The mixture contains salt", Verdict.SUPPORTED),
    ("Café serves pastry", "Café serves pastry", Verdict.SUPPORTED),
    (
        "The solution is not acidic",
        "The solution is not acidic",
        Verdict.CONTRADICTED,
    ),
    ("catscradle sits here", "cats", Verdict.NOT_IN_SOURCE),
    ("WATER boils", "water BOILS", Verdict.SUPPORTED),
    ("water    boils", "water boils", Verdict.SUPPORTED),
    ("hello world", "hello.", Verdict.NOT_IN_SOURCE),
    ("The cats are here", "The cats are here.", Verdict.NOT_IN_SOURCE),
    ("The mass is 3.14 kilograms", "The mass is 3.14 kilograms", Verdict.SUPPORTED),
    ("hello", "3.14", Verdict.NOT_IN_SOURCE),
]

PROPERTY_TEXTS = [
    "",
    " ",
    "\t\n",
    "Hello.",
    "Hello. World!",
    "Dr. Smith went home.",
    "Water boils at 100 degrees.",
    "The solution is not acidic.",
    "Café is open. Naïve? Yes!",
    "No cats here.",
    "3.14 is pi.",
    "Hello.World!",
    "What? Yes. No!",
    "a",
    "that this with from",
    "The mass is 3.14 kilograms",
    "There are no cats in the library",
    "Paris is the capital of France",
    "note available here",
    "not available here",
    "The cats are here.",
    "page 14",
]


def load_labeled() -> list[LabeledExample]:
    rows: list[LabeledExample] = []
    for line in LABELED_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append(
            LabeledExample(
                source=obj["source"],
                claim=obj["claim"],
                gold=Verdict(obj["gold"]),
            )
        )
    return rows


class ScriptedGenerator:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[list[str], int, float]] = []

    def generate(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        self.calls.append((list(prompts), int(max_tokens), float(temperature)))
        return [self.mapping[p] for p in prompts]


class MismatchGenerator:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls: list[tuple[list[str], int, float]] = []

    def generate(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        self.calls.append((list(prompts), int(max_tokens), float(temperature)))
        return ["x"] * self.n


class BoomGenerator:
    def generate(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        raise AssertionError("generator must not be called")


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


# ---------------------------------------------------------------------------
# Constants / Style / Verdict / FactCheck counts / generate_prompt (may pass)
# ---------------------------------------------------------------------------


def test_schema_and_defaults() -> None:
    assert SCHEMA_VERSION == 1
    assert DEFAULT_MAX_TOKENS == 512
    assert DEFAULT_TEMPERATURE == pytest.approx(0.7)
    assert DEFAULT_MAX_BATCH == 8
    assert len(STYLES) == 6
    assert STYLES == (
        Style.ENCYCLOPEDIA,
        Style.TEXTBOOK,
        Style.CONVERSATIONAL,
        Style.NEWS,
        Style.TECHNICAL,
        Style.SIMPLIFIED,
    )


def test_style_and_verdict_values() -> None:
    assert Style.ENCYCLOPEDIA == "encyclopedia"
    assert Style.TEXTBOOK == "textbook"
    assert Style.CONVERSATIONAL == "conversational"
    assert Style.NEWS == "news"
    assert Style.TECHNICAL == "technical"
    assert Style.SIMPLIFIED == "simplified"
    assert Verdict.SUPPORTED == "supported"
    assert Verdict.CONTRADICTED == "contradicted"
    assert Verdict.NOT_IN_SOURCE == "not_in_source"


def test_factcheck_counts_are_derived() -> None:
    fc = FactCheck(
        claims=(
            Claim("a", Verdict.SUPPORTED),
            Claim("b", Verdict.CONTRADICTED),
            Claim("c", Verdict.NOT_IN_SOURCE),
            Claim("d", Verdict.SUPPORTED),
        )
    )
    assert fc.n_supported == 2
    assert fc.n_contradicted == 1
    assert fc.n_not_in_source == 1
    empty = FactCheck(claims=())
    assert empty.n_supported == 0
    assert empty.n_contradicted == 0
    assert empty.n_not_in_source == 0


def test_generate_prompt_matches_template() -> None:
    doc = SourceDocument("w", "Water boils.")
    prompt = generate_prompt(doc, Style.NEWS)
    assert prompt == (
        "Rephrase the document in news style. Preserve every fact. Do not add facts.\n"
        "\n"
        "Document:\n"
        "Water boils."
    )


@pytest.mark.parametrize("style", list(Style))
def test_generate_prompt_embeds_style_value(style: Style) -> None:
    doc = SourceDocument("id", "Body text.")
    prompt = generate_prompt(doc, style)
    assert prompt.startswith(
        f"Rephrase the document in {style.value} style. Preserve every fact. Do not add facts."
    )
    assert prompt.endswith("Document:\nBody text.")


def test_generate_prompt_preserves_utf8() -> None:
    doc = SourceDocument("cafe", "Café is open.")
    prompt = generate_prompt(doc, Style.ENCYCLOPEDIA)
    assert "Café is open." in prompt
    assert "encyclopedia" in prompt


def test_generate_prompt_empty_source_raises() -> None:
    with pytest.raises(SynthError, match="empty source"):
        generate_prompt(SourceDocument("w", ""), Style.NEWS)


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", EXTRACT_GOLDENS)
def test_extract_claims_goldens(text: str, expected: list[str]) -> None:
    assert extract_claims(text) == expected


def test_extract_claims_preserves_casing() -> None:
    assert extract_claims("Hello. World!") == ["Hello.", "World!"]
    assert extract_claims("Hello. World!") != ["hello.", "world!"]


# ---------------------------------------------------------------------------
# check_claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source,claim,expected", CHECK_GOLDENS)
def test_check_claim_goldens(source: str, claim: str, expected: Verdict) -> None:
    assert check_claim(source, claim) == expected


def test_check_claim_empty_source_raises() -> None:
    with pytest.raises(SynthError):
        check_claim("", "Water boils.")
    with pytest.raises(SynthError):
        check_claim("", "")


def test_check_claim_whitespace_source_is_not_empty_error() -> None:
    assert check_claim("   ", "hello") == Verdict.NOT_IN_SOURCE


# ---------------------------------------------------------------------------
# fact_check
# ---------------------------------------------------------------------------


def test_fact_check_empty_source_raises() -> None:
    with pytest.raises(SynthError):
        fact_check("", "Water is a liquid.")
    with pytest.raises(SynthError):
        fact_check("", "")


def test_fact_check_empty_rewrite_is_zero_claims() -> None:
    fc = fact_check("Water is a liquid.", "")
    assert fc.claims == ()
    assert fc.n_supported == 0
    assert fc.n_contradicted == 0
    assert fc.n_not_in_source == 0


def test_fact_check_whitespace_rewrite_is_zero_claims() -> None:
    fc = fact_check("Water is a liquid.", "   \n")
    assert fc.claims == ()


def test_fact_check_composes_extract_and_check() -> None:
    source = "Water boils at 100 degrees Celsius at sea level and water is a liquid"
    rewrite = "Water is a liquid. Helium is a liquid. Water boils at 90 degrees."
    fc = fact_check(source, rewrite)
    texts = extract_claims(rewrite)
    assert texts == [
        "Water is a liquid.",
        "Helium is a liquid.",
        "Water boils at 90 degrees.",
    ]
    assert [c.text for c in fc.claims] == texts
    assert [c.verdict for c in fc.claims] == [check_claim(source, t) for t in texts]
    # Trailing periods stay on the last token, so "liquid." does not match "liquid".
    assert [c.verdict for c in fc.claims] == [
        Verdict.NOT_IN_SOURCE,
        Verdict.NOT_IN_SOURCE,
        Verdict.CONTRADICTED,
    ]
    assert fc.n_supported == 0
    assert fc.n_not_in_source == 2
    assert fc.n_contradicted == 1
    assert fc == ref.fact_check(source, rewrite)


# ---------------------------------------------------------------------------
# accept
# ---------------------------------------------------------------------------


def test_accept_contradicted_rejects() -> None:
    fc = FactCheck(claims=(Claim("x", Verdict.CONTRADICTED),))
    assert accept(fc) is False
    mixed = FactCheck(
        claims=(
            Claim("ok", Verdict.SUPPORTED),
            Claim("bad", Verdict.CONTRADICTED),
            Claim("miss", Verdict.NOT_IN_SOURCE),
        )
    )
    assert accept(mixed) is False


def test_accept_not_in_source_alone_accepts() -> None:
    fc = FactCheck(claims=(Claim("x", Verdict.NOT_IN_SOURCE),))
    assert accept(fc) is True
    mixed = FactCheck(
        claims=(
            Claim("ok", Verdict.SUPPORTED),
            Claim("miss", Verdict.NOT_IN_SOURCE),
        )
    )
    assert accept(mixed) is True


def test_accept_empty_claims_accepts() -> None:
    assert accept(FactCheck(claims=())) is True


# ---------------------------------------------------------------------------
# token_count
# ---------------------------------------------------------------------------


def test_token_count_empty_is_zero_without_calling_tokenizer() -> None:
    assert token_count("") == 0
    assert token_count("", tokenizer=BoomTokenizer()) == 0


def test_token_count_whitespace_words() -> None:
    assert token_count("hello  world\t\nfoo") == 3
    assert token_count("  hello  ") == 1
    assert token_count("   \t") == 0
    assert token_count("hello\u00a0world") == 2
    assert token_count("hello.") == 1


def test_token_count_uses_tokenizer_encode_length() -> None:
    tok = FakeTokenizer(n=4)
    assert token_count("hello world", tokenizer=tok) == 4
    assert tok.seen == ["hello world"]
    lens = LenTokenizer()
    assert token_count("abcd", tokenizer=lens) == 4


# ---------------------------------------------------------------------------
# precision (E1 gate sample)
# ---------------------------------------------------------------------------


def test_precision_on_labeled_gate_sample() -> None:
    rows = load_labeled()
    assert len(rows) == LABELED_SAMPLE_SIZE
    golds = {row.gold for row in rows}
    assert golds == {Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.NOT_IN_SOURCE}
    got = precision(rows)
    assert got == pytest.approx(GOLDEN_PRECISION, abs=1e-12)
    assert got == pytest.approx(ref.precision(rows), abs=1e-12)


def test_precision_fail_closed_on_empty_or_no_supported_preds() -> None:
    assert precision([]) == 0.0
    only_neg = [LabeledExample("hello world", "zzzz missing claim", Verdict.NOT_IN_SOURCE)]
    assert precision(only_neg) == 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _water_doc() -> SourceDocument:
    return SourceDocument("water", WATER_TEXT)


def _paris_doc() -> SourceDocument:
    return SourceDocument("paris", PARIS_TEXT)


def test_rephrase_empty_source_errors_before_generate() -> None:
    with pytest.raises(SynthError):
        Orchestrator(BoomGenerator()).rephrase(SourceDocument("w", ""), Style.NEWS)


def test_rephrase_scripted_generator_matches_reference() -> None:
    doc = _water_doc()
    style = Style.NEWS
    prompt = generate_prompt(doc, style)
    completion = "Water is a liquid at room temperature."
    gen = ScriptedGenerator({prompt: completion})
    cfg = OrchestratorConfig(max_tokens=64, temperature=0.25, max_batch=2)
    orch = Orchestrator(gen, config=cfg)
    got = orch.rephrase(doc, style)
    expected = ref.rewrite_from_completion(doc, style, completion)
    assert got == expected
    assert gen.calls == [([prompt], 64, 0.25)]
    assert got.accepted is True
    assert got.source_id == "water"
    assert got.style == Style.NEWS
    assert got.text == completion


def test_rephrase_uses_tokenizer_encode_when_provided() -> None:
    doc = _water_doc()
    style = Style.TEXTBOOK
    prompt = generate_prompt(doc, style)
    completion = "Water is a liquid."
    gen = ScriptedGenerator({prompt: completion})
    tok = FakeTokenizer(n=7)
    orch = Orchestrator(gen, tokenizer=tok)
    got = orch.rephrase(doc, style)
    assert got.token_count == 7
    assert tok.seen == [completion]
    assert got.token_count == ref.token_count(completion, tokenizer=tok)


def test_rephrase_contradicted_completion_is_not_accepted() -> None:
    doc = _water_doc()
    style = Style.TECHNICAL
    prompt = generate_prompt(doc, style)
    completion = "Water boils at 90 degrees."
    gen = ScriptedGenerator({prompt: completion})
    got = Orchestrator(gen).rephrase(doc, style)
    assert got.accepted is False
    assert got.fact_check.n_contradicted >= 1
    assert got == ref.rewrite_from_completion(doc, style, completion)


def test_rephrase_generator_length_mismatch_errors() -> None:
    doc = _water_doc()
    gen = MismatchGenerator(0)
    with pytest.raises(SynthError):
        Orchestrator(gen).rephrase(doc, Style.NEWS)
    gen2 = MismatchGenerator(2)
    with pytest.raises(SynthError):
        Orchestrator(gen2).rephrase(doc, Style.NEWS)


def test_rephrase_many_empty_docs_errors_before_generate() -> None:
    with pytest.raises(SynthError):
        Orchestrator(BoomGenerator()).rephrase_many([])
    with pytest.raises(SynthError):
        Orchestrator(BoomGenerator()).rephrase_many([], styles=[Style.NEWS])


def test_rephrase_many_empty_source_in_batch_errors_before_generate() -> None:
    docs = [_water_doc(), SourceDocument("empty", "")]
    with pytest.raises(SynthError):
        Orchestrator(BoomGenerator()).rephrase_many(docs, styles=[Style.NEWS])


def test_rephrase_many_cartesian_docs_major_and_max_batch_chunking() -> None:
    docs = [_water_doc(), _paris_doc()]
    styles = [Style.NEWS, Style.TECHNICAL]
    cfg = OrchestratorConfig(max_tokens=32, temperature=0.5, max_batch=3, styles=styles)
    mapping: dict[str, str] = {}
    completions = {
        ("water", Style.NEWS): "Water is a liquid.",
        ("water", Style.TECHNICAL): "Water boils at 90 degrees.",
        ("paris", Style.NEWS): "Paris is the capital of France.",
        ("paris", Style.TECHNICAL): "Berlin is the capital of Germany.",
    }
    for doc in docs:
        for style in styles:
            mapping[generate_prompt(doc, style)] = completions[(doc.source_id, style)]
    gen = ScriptedGenerator(mapping)
    orch = Orchestrator(gen, config=cfg)
    got = orch.rephrase_many(docs, styles=styles)
    assert [r.source_id for r in got] == ["water", "water", "paris", "paris"]
    assert [r.style for r in got] == [
        Style.NEWS,
        Style.TECHNICAL,
        Style.NEWS,
        Style.TECHNICAL,
    ]
    assert [r.text for r in got] == [
        completions[("water", Style.NEWS)],
        completions[("water", Style.TECHNICAL)],
        completions[("paris", Style.NEWS)],
        completions[("paris", Style.TECHNICAL)],
    ]
    expected = [
        ref.rewrite_from_completion(doc, style, completions[(doc.source_id, style)])
        for doc in docs
        for style in styles
    ]
    assert got == expected
    assert [len(c[0]) for c in gen.calls] == [3, 1]
    assert all(c[1] == 32 and c[2] == 0.5 for c in gen.calls)
    assert all(len(c[0]) <= cfg.max_batch for c in gen.calls)
    concat = [p for c in gen.calls for p in c[0]]
    assert concat == [generate_prompt(doc, style) for doc in docs for style in styles]


def test_rephrase_many_default_styles_come_from_config() -> None:
    doc = _water_doc()
    styles = (Style.SIMPLIFIED,)
    cfg = OrchestratorConfig(styles=styles, max_tokens=16, temperature=0.1)
    prompt = generate_prompt(doc, Style.SIMPLIFIED)
    gen = ScriptedGenerator({prompt: "Water is a liquid."})
    got = Orchestrator(gen, config=cfg).rephrase_many([doc])
    assert len(got) == 1
    assert got[0].style == Style.SIMPLIFIED
    assert gen.calls[0][0] == [prompt]


def test_rephrase_many_generator_length_mismatch_errors() -> None:
    docs = [_water_doc()]
    styles = [Style.NEWS, Style.SIMPLIFIED]
    cfg = OrchestratorConfig(max_batch=8, styles=styles)
    gen = MismatchGenerator(1)
    with pytest.raises(SynthError):
        Orchestrator(gen, config=cfg).rephrase_many(docs, styles=styles)


def test_rephrase_many_defaults_max_tokens_and_temperature() -> None:
    doc = _water_doc()
    prompt = generate_prompt(doc, Style.NEWS)
    gen = ScriptedGenerator({prompt: "Water is a liquid."})
    Orchestrator(gen).rephrase(doc, Style.NEWS)
    assert gen.calls[0][1] == DEFAULT_MAX_TOKENS
    assert gen.calls[0][2] == pytest.approx(DEFAULT_TEMPERATURE)


# ---------------------------------------------------------------------------
# Property: production matches the independent reference
# ---------------------------------------------------------------------------


def test_production_matches_reference_on_hand_rolled_strings() -> None:
    for text in PROPERTY_TEXTS:
        assert extract_claims(text) == ref.extract_claims(text)
        assert token_count(text) == ref.token_count(text)
    tok = LenTokenizer()
    for text in PROPERTY_TEXTS:
        if text == "":
            assert token_count(text, tokenizer=tok) == 0
        else:
            assert token_count(text, tokenizer=tok) == ref.token_count(text, tokenizer=tok)
    sources = [t for t in PROPERTY_TEXTS if t != ""]
    for source in sources:
        for claim in PROPERTY_TEXTS:
            assert check_claim(source, claim) == ref.check_claim(source, claim)
        for rewrite in PROPERTY_TEXTS:
            prod_fc = fact_check(source, rewrite)
            ref_fc = ref.fact_check(source, rewrite)
            assert prod_fc == ref_fc
            assert accept(prod_fc) is ref.accept(ref_fc)
    rows = [
        LabeledExample(s, c, ref.check_claim(s, c))
        for s in sources[:6]
        for c in PROPERTY_TEXTS[3:9]
    ]
    assert precision(rows) == pytest.approx(ref.precision(rows), abs=1e-12)
