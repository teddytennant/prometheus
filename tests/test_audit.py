"""Implementer-facing tests for I9 ``audit/`` (spec 12, 4.3, 15.5).

Drive production ``audit`` AND ``tests.reference.audit`` on every behavior
assertion. Constants, enums, ``AuditError`` subclassing, and frozen dataclass
construction MAY pass against the stub. Every test that calls ``force_discrete``,
``compare_answers``, ``compare_suite``, ``decode_thoughts``, or
``find_planted_bug`` MUST FAIL on the stub with ``NotImplementedError``.

CPU analog only. No GPU marker, no JAX, no SGLang, no Firecracker, no WORM log.
Production must never import ``tests/``. The reference does not import
``model.latent`` or ``rl.loss``.

Groups
------
1. constants / enums / AuditError subclass / frozen dataclasses.
2. force_discrete: happy path, empty/whitespace error.
3. compare_answers: diverge, match, gold None vs set, empty problem_id,
   empty answers allowed.
4. compare_suite: order preserved, empty error, mixed diverge.
5. decode_thoughts: golden small tensor, tie takes smallest id, rank/window/
   empty/non-finite errors.
6. find_planted_bug ANSWER_SWAP: gold match discrete only; diverge without
   gold; missing pair error.
7. find_planted_bug DECODE_FLIP: sign flip found; all-zero not found;
   missing logits.
8. find_planted_bug HALT_UNPINNED: last != 1 found; last == 1.0 not found;
   missing/rank errors.
9. Property: lockstep prod vs ref on random small tensors
   (n_thoughts=1..8, vocab=2..16) at exact token_ids match.

Goldens are frozen from the hand-computed reference rules, not from production.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import audit
from tests.reference import audit as ref

# Hand-computed greedy ids for GOLDEN_LOGITS (ties take the smallest id).
GOLDEN_TOKEN_IDS = (1, 0, 0, 1, 3, 0, 3, 1)
GOLDEN_TOKEN_IDS_ZEROS = (0, 0, 0, 0, 0, 0, 0, 0)


def _golden_logits() -> np.ndarray:
    # shape (1, 8, 4). Each row is one of 8 thought-decode positions.
    return np.array(
        [
            [
                [0.1, 0.9, 0.2, 0.3],  # 1
                [0.5, 0.5, 0.1, 0.0],  # 0 (tie 0 and 1)
                [0.0, 0.0, 0.0, 0.0],  # 0 (all tie)
                [-1.0, -0.5, -2.0, -0.5],  # 1 (tie 1 and 3)
                [1.0, 2.0, 3.0, 4.0],  # 3
                [4.0, 3.0, 2.0, 1.0],  # 0
                [-0.1, -0.1, -0.1, 0.0],  # 3
                [0.0, 1.0, 1.0, 1.0],  # 1 (tie 1, 2, 3)
            ]
        ],
        dtype=np.float32,
    )


def _both_raise(prod_fn, ref_fn) -> None:
    with pytest.raises(audit.AuditError):
        prod_fn()
    with pytest.raises(ref.AuditError):
        ref_fn()


def _pair_prod(
    problem_id: str,
    discrete_answer: str,
    latent_answer: str,
    gold: str | None = None,
) -> audit.AnswerPair:
    return audit.AnswerPair(
        problem_id=problem_id,
        discrete_answer=discrete_answer,
        latent_answer=latent_answer,
        gold=gold,
    )


def _pair_ref(
    problem_id: str,
    discrete_answer: str,
    latent_answer: str,
    gold: str | None = None,
) -> ref.AnswerPair:
    return ref.AnswerPair(
        problem_id=problem_id,
        discrete_answer=discrete_answer,
        latent_answer=latent_answer,
        gold=gold,
    )


def _assert_divergence(got, exp, **fields: object) -> None:
    for name, value in fields.items():
        assert getattr(got, name) == value
        assert getattr(exp, name) == value


def _assert_steps(got, exp) -> None:
    assert isinstance(got, tuple)
    assert isinstance(exp, tuple)
    assert len(got) == len(exp)
    for g_step, e_step in zip(got, exp, strict=True):
        assert g_step.index == e_step.index
        assert g_step.token_ids == e_step.token_ids
        assert len(g_step.token_ids) == audit.TOKENS_PER_THOUGHT
        assert len(e_step.token_ids) == ref.TOKENS_PER_THOUGHT


def _assert_bug(*, got, exp, kind_value: str, found: bool) -> None:
    assert got.kind == kind_value
    assert exp.kind == kind_value
    assert got.found is found
    assert exp.found is found
    assert isinstance(got.evidence, str) and got.evidence
    assert isinstance(exp.evidence, str) and exp.evidence


# ---------------------------------------------------------------------------
# 1. constants / enums / AuditError subclass / frozen dataclasses
#    MAY pass the stub.
# ---------------------------------------------------------------------------


def test_tokens_per_thought():
    assert audit.TOKENS_PER_THOUGHT == 8
    assert ref.TOKENS_PER_THOUGHT == 8


def test_mode_enum_values():
    assert audit.Mode.LATENT == "latent"
    assert audit.Mode.DISCRETE == "discrete"
    assert ref.Mode.LATENT == "latent"
    assert ref.Mode.DISCRETE == "discrete"
    assert audit.Mode.LATENT != audit.Mode.DISCRETE


def test_bug_kind_enum_values():
    assert audit.BugKind.ANSWER_SWAP == "answer_swap"
    assert audit.BugKind.DECODE_FLIP == "decode_flip"
    assert audit.BugKind.HALT_UNPINNED == "halt_unpinned"
    assert ref.BugKind.ANSWER_SWAP == "answer_swap"
    assert ref.BugKind.DECODE_FLIP == "decode_flip"
    assert ref.BugKind.HALT_UNPINNED == "halt_unpinned"


def test_audit_error_is_value_error():
    assert issubclass(audit.AuditError, ValueError)
    assert issubclass(ref.AuditError, ValueError)
    assert not issubclass(audit.AuditError, TypeError)


def test_dataclasses_are_frozen():
    ckpt = audit.CheckpointMode(checkpoint_id="c", mode=audit.Mode.DISCRETE)
    pair = audit.AnswerPair(problem_id="p", discrete_answer="a", latent_answer="b")
    div = audit.Divergence(
        problem_id="p",
        discrete_answer="a",
        latent_answer="b",
        gold=None,
        diverge=True,
        discrete_correct=None,
        latent_correct=None,
    )
    step = audit.ThoughtStep(index=0, token_ids=(0,) * 8)
    bug = audit.PlantedBug(
        kind=audit.BugKind.ANSWER_SWAP, found=False, evidence="none"
    )
    with pytest.raises(FrozenInstanceError):
        ckpt.checkpoint_id = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pair.problem_id = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        div.diverge = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        step.index = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bug.found = True  # type: ignore[misc]

    ckpt_r = ref.CheckpointMode(checkpoint_id="c", mode=ref.Mode.DISCRETE)
    pair_r = ref.AnswerPair(problem_id="p", discrete_answer="a", latent_answer="b")
    div_r = ref.Divergence(
        problem_id="p",
        discrete_answer="a",
        latent_answer="b",
        gold=None,
        diverge=True,
        discrete_correct=None,
        latent_correct=None,
    )
    step_r = ref.ThoughtStep(index=0, token_ids=(0,) * 8)
    bug_r = ref.PlantedBug(kind=ref.BugKind.ANSWER_SWAP, found=False, evidence="none")
    with pytest.raises(FrozenInstanceError):
        ckpt_r.mode = ref.Mode.LATENT  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pair_r.gold = "g"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        div_r.gold = "g"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        step_r.token_ids = (1,) * 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bug_r.evidence = "x"  # type: ignore[misc]


def test_answer_pair_gold_defaults_to_none():
    pair = audit.AnswerPair(problem_id="p", discrete_answer="a", latent_answer="b")
    pair_r = ref.AnswerPair(problem_id="p", discrete_answer="a", latent_answer="b")
    assert pair.gold is None
    assert pair_r.gold is None


# ---------------------------------------------------------------------------
# 2. force_discrete
# ---------------------------------------------------------------------------


def test_force_discrete_happy_path():
    got = audit.force_discrete("r0-step-3")
    exp = ref.force_discrete("r0-step-3")
    assert got.checkpoint_id == "r0-step-3"
    assert exp.checkpoint_id == "r0-step-3"
    assert got.mode == audit.Mode.DISCRETE
    assert exp.mode == ref.Mode.DISCRETE
    assert got.mode == "discrete"


def test_force_discrete_preserves_surrounding_content_spaces():
    cid = " ckpt "
    got = audit.force_discrete(cid)
    exp = ref.force_discrete(cid)
    assert got.checkpoint_id == cid
    assert exp.checkpoint_id == cid
    assert got.mode == audit.Mode.DISCRETE
    assert exp.mode == ref.Mode.DISCRETE


def test_force_discrete_id_containing_latent_still_discrete():
    got = audit.force_discrete("latent-ckpt")
    exp = ref.force_discrete("latent-ckpt")
    assert got.mode == audit.Mode.DISCRETE
    assert exp.mode == ref.Mode.DISCRETE


@pytest.mark.parametrize("cid", ["", " ", "\t", "\n", "\r\n", "  \t\n  "])
def test_force_discrete_rejects_empty_or_whitespace(cid: str):
    _both_raise(lambda: audit.force_discrete(cid), lambda: ref.force_discrete(cid))


# ---------------------------------------------------------------------------
# 3. compare_answers
# ---------------------------------------------------------------------------


def test_compare_answers_diverge_gold_none():
    got = audit.compare_answers(_pair_prod("p1", "yes", "no"))
    exp = ref.compare_answers(_pair_ref("p1", "yes", "no"))
    _assert_divergence(
        got,
        exp,
        problem_id="p1",
        discrete_answer="yes",
        latent_answer="no",
        gold=None,
        diverge=True,
        discrete_correct=None,
        latent_correct=None,
    )


def test_compare_answers_match_gold_none():
    got = audit.compare_answers(_pair_prod("p1", "yes", "yes"))
    exp = ref.compare_answers(_pair_ref("p1", "yes", "yes"))
    _assert_divergence(
        got,
        exp,
        problem_id="p1",
        discrete_answer="yes",
        latent_answer="yes",
        gold=None,
        diverge=False,
        discrete_correct=None,
        latent_correct=None,
    )


def test_compare_answers_gold_set_discrete_correct_only():
    got = audit.compare_answers(_pair_prod("p2", "42", "7", gold="42"))
    exp = ref.compare_answers(_pair_ref("p2", "42", "7", gold="42"))
    _assert_divergence(
        got,
        exp,
        problem_id="p2",
        discrete_answer="42",
        latent_answer="7",
        gold="42",
        diverge=True,
        discrete_correct=True,
        latent_correct=False,
    )


def test_compare_answers_gold_set_latent_correct_only():
    got = audit.compare_answers(_pair_prod("p2", "7", "42", gold="42"))
    exp = ref.compare_answers(_pair_ref("p2", "7", "42", gold="42"))
    _assert_divergence(
        got,
        exp,
        diverge=True,
        discrete_correct=False,
        latent_correct=True,
    )


def test_compare_answers_gold_set_both_correct():
    got = audit.compare_answers(_pair_prod("p2", "42", "42", gold="42"))
    exp = ref.compare_answers(_pair_ref("p2", "42", "42", gold="42"))
    _assert_divergence(
        got,
        exp,
        diverge=False,
        discrete_correct=True,
        latent_correct=True,
    )


def test_compare_answers_gold_set_both_wrong_same_answer():
    got = audit.compare_answers(_pair_prod("p2", "7", "7", gold="42"))
    exp = ref.compare_answers(_pair_ref("p2", "7", "7", gold="42"))
    _assert_divergence(
        got,
        exp,
        diverge=False,
        discrete_correct=False,
        latent_correct=False,
    )


def test_compare_answers_gold_set_both_wrong_different_answers():
    got = audit.compare_answers(_pair_prod("p2", "7", "8", gold="42"))
    exp = ref.compare_answers(_pair_ref("p2", "7", "8", gold="42"))
    _assert_divergence(
        got,
        exp,
        diverge=True,
        discrete_correct=False,
        latent_correct=False,
    )


def test_compare_answers_empty_answers_allowed():
    got = audit.compare_answers(_pair_prod("p3", "", ""))
    exp = ref.compare_answers(_pair_ref("p3", "", ""))
    _assert_divergence(
        got,
        exp,
        discrete_answer="",
        latent_answer="",
        diverge=False,
        discrete_correct=None,
        latent_correct=None,
    )
    got2 = audit.compare_answers(_pair_prod("p3", "", "x", gold=""))
    exp2 = ref.compare_answers(_pair_ref("p3", "", "x", gold=""))
    _assert_divergence(
        got2,
        exp2,
        gold="",
        diverge=True,
        discrete_correct=True,
        latent_correct=False,
    )


def test_compare_answers_whitespace_problem_id_is_not_empty():
    got = audit.compare_answers(_pair_prod(" ", "a", "a"))
    exp = ref.compare_answers(_pair_ref(" ", "a", "a"))
    _assert_divergence(got, exp, problem_id=" ", diverge=False)


def test_compare_answers_case_sensitive():
    got = audit.compare_answers(_pair_prod("p", "A", "a"))
    exp = ref.compare_answers(_pair_ref("p", "A", "a"))
    _assert_divergence(got, exp, diverge=True)


def test_compare_answers_empty_problem_id():
    _both_raise(
        lambda: audit.compare_answers(_pair_prod("", "a", "b")),
        lambda: ref.compare_answers(_pair_ref("", "a", "b")),
    )


# ---------------------------------------------------------------------------
# 4. compare_suite
# ---------------------------------------------------------------------------


def test_compare_suite_empty_tuple():
    _both_raise(lambda: audit.compare_suite(()), lambda: ref.compare_suite(()))


def test_compare_suite_order_preserved_mixed_diverge():
    prod_pairs = (
        _pair_prod("a", "1", "1", gold="1"),
        _pair_prod("b", "1", "2", gold="1"),
        _pair_prod("c", "x", "y"),
        _pair_prod("d", "", "", gold="z"),
    )
    ref_pairs = (
        _pair_ref("a", "1", "1", gold="1"),
        _pair_ref("b", "1", "2", gold="1"),
        _pair_ref("c", "x", "y"),
        _pair_ref("d", "", "", gold="z"),
    )
    got = audit.compare_suite(prod_pairs)
    exp = ref.compare_suite(ref_pairs)
    assert len(got) == 4
    assert len(exp) == 4
    assert [row.problem_id for row in got] == ["a", "b", "c", "d"]
    assert [row.problem_id for row in exp] == ["a", "b", "c", "d"]
    assert [row.diverge for row in got] == [False, True, True, False]
    assert [row.diverge for row in exp] == [False, True, True, False]
    assert got[0].discrete_correct is True and got[0].latent_correct is True
    assert exp[0].discrete_correct is True and exp[0].latent_correct is True
    assert got[1].discrete_correct is True and got[1].latent_correct is False
    assert exp[1].discrete_correct is True and exp[1].latent_correct is False
    assert got[2].discrete_correct is None and got[2].latent_correct is None
    assert exp[2].discrete_correct is None and exp[2].latent_correct is None
    assert got[3].discrete_correct is False and got[3].latent_correct is False
    assert exp[3].discrete_correct is False and exp[3].latent_correct is False


def test_compare_suite_single_non_diverging():
    got = audit.compare_suite((_pair_prod("only", "ok", "ok"),))
    exp = ref.compare_suite((_pair_ref("only", "ok", "ok"),))
    assert len(got) == 1 and len(exp) == 1
    assert got[0].diverge is False
    assert exp[0].diverge is False


def test_compare_suite_propagates_empty_problem_id():
    _both_raise(
        lambda: audit.compare_suite((_pair_prod("", "a", "b"),)),
        lambda: ref.compare_suite((_pair_ref("", "a", "b"),)),
    )


# ---------------------------------------------------------------------------
# 5. decode_thoughts
# ---------------------------------------------------------------------------


def test_decode_thoughts_golden_small_tensor():
    logits = _golden_logits()
    got = audit.decode_thoughts(logits)
    exp = ref.decode_thoughts(logits)
    _assert_steps(got, exp)
    assert got[0].index == 0
    assert got[0].token_ids == GOLDEN_TOKEN_IDS
    assert exp[0].token_ids == GOLDEN_TOKEN_IDS


def test_decode_thoughts_two_thoughts_second_all_zero():
    first = _golden_logits()
    zeros = np.zeros((1, 8, 4), dtype=np.float32)
    logits = np.concatenate([first, zeros], axis=0)
    got = audit.decode_thoughts(logits)
    exp = ref.decode_thoughts(logits)
    _assert_steps(got, exp)
    assert [s.index for s in got] == [0, 1]
    assert [s.index for s in exp] == [0, 1]
    assert got[0].token_ids == GOLDEN_TOKEN_IDS
    assert got[1].token_ids == GOLDEN_TOKEN_IDS_ZEROS
    assert exp[1].token_ids == GOLDEN_TOKEN_IDS_ZEROS


def test_decode_thoughts_tie_takes_smallest_id():
    logits = np.zeros((1, 8, 5), dtype=np.float32)
    logits[0, 0] = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    logits[0, 1] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    logits[0, 2] = np.array([-3.0, -3.0, -1.0, -1.0, -1.0], dtype=np.float32)
    got = audit.decode_thoughts(logits)
    exp = ref.decode_thoughts(logits)
    _assert_steps(got, exp)
    assert got[0].token_ids[0] == 0
    assert got[0].token_ids[1] == 0
    assert got[0].token_ids[2] == 2
    assert exp[0].token_ids[0] == 0
    assert exp[0].token_ids[1] == 0
    assert exp[0].token_ids[2] == 2


def test_decode_thoughts_vocab_one():
    logits = np.array([[[0.3]] * 8], dtype=np.float32)
    got = audit.decode_thoughts(logits)
    exp = ref.decode_thoughts(logits)
    _assert_steps(got, exp)
    assert got[0].token_ids == GOLDEN_TOKEN_IDS_ZEROS
    assert exp[0].token_ids == GOLDEN_TOKEN_IDS_ZEROS


def test_decode_thoughts_rejects_rank_not_three():
    _both_raise(
        lambda: audit.decode_thoughts(np.array(1.0, dtype=np.float32)),
        lambda: ref.decode_thoughts(np.array(1.0, dtype=np.float32)),
    )
    _both_raise(
        lambda: audit.decode_thoughts(np.zeros((8,), dtype=np.float32)),
        lambda: ref.decode_thoughts(np.zeros((8,), dtype=np.float32)),
    )
    _both_raise(
        lambda: audit.decode_thoughts(np.zeros((8, 4), dtype=np.float32)),
        lambda: ref.decode_thoughts(np.zeros((8, 4), dtype=np.float32)),
    )
    _both_raise(
        lambda: audit.decode_thoughts(np.zeros((1, 8, 4, 1), dtype=np.float32)),
        lambda: ref.decode_thoughts(np.zeros((1, 8, 4, 1), dtype=np.float32)),
    )


def test_decode_thoughts_rejects_empty_and_zero_thoughts():
    _both_raise(
        lambda: audit.decode_thoughts(np.array([], dtype=np.float32)),
        lambda: ref.decode_thoughts(np.array([], dtype=np.float32)),
    )
    _both_raise(
        lambda: audit.decode_thoughts(np.zeros((0, 8, 4), dtype=np.float32)),
        lambda: ref.decode_thoughts(np.zeros((0, 8, 4), dtype=np.float32)),
    )


@pytest.mark.parametrize("window", [1, 7, 9, 16])
def test_decode_thoughts_rejects_wrong_window(window: int):
    logits = np.zeros((1, window, 4), dtype=np.float32)
    _both_raise(lambda: audit.decode_thoughts(logits), lambda: ref.decode_thoughts(logits))


def test_decode_thoughts_rejects_vocab_less_than_one():
    _both_raise(
        lambda: audit.decode_thoughts(np.zeros((1, 8, 0), dtype=np.float32)),
        lambda: ref.decode_thoughts(np.zeros((1, 8, 0), dtype=np.float32)),
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_decode_thoughts_rejects_non_finite(bad: float):
    logits = _golden_logits()
    logits = logits.copy()
    logits[0, 3, 1] = bad
    _both_raise(lambda: audit.decode_thoughts(logits), lambda: ref.decode_thoughts(logits))


def test_decode_thoughts_rejects_non_array():
    nested = [[[0.0] * 4 for _ in range(8)]]
    _both_raise(lambda: audit.decode_thoughts(nested), lambda: ref.decode_thoughts(nested))


# ---------------------------------------------------------------------------
# 6. find_planted_bug ANSWER_SWAP
# ---------------------------------------------------------------------------


def test_answer_swap_discrete_matches_gold_latent_does_not():
    got = audit.find_planted_bug(
        audit.BugKind.ANSWER_SWAP,
        _pair_prod("bug", "right", "wrong", gold="right"),
        None,
        None,
    )
    exp = ref.find_planted_bug(
        ref.BugKind.ANSWER_SWAP,
        _pair_ref("bug", "right", "wrong", gold="right"),
        None,
        None,
    )
    _assert_bug(got=got, exp=exp, kind_value="answer_swap", found=True)


def test_answer_swap_diverge_without_gold():
    got = audit.find_planted_bug(
        audit.BugKind.ANSWER_SWAP, _pair_prod("bug", "a", "b"), None, None
    )
    exp = ref.find_planted_bug(
        ref.BugKind.ANSWER_SWAP, _pair_ref("bug", "a", "b"), None, None
    )
    _assert_bug(got=got, exp=exp, kind_value="answer_swap", found=True)


def test_answer_swap_match_without_gold_not_found():
    got = audit.find_planted_bug(
        audit.BugKind.ANSWER_SWAP, _pair_prod("bug", "a", "a"), None, None
    )
    exp = ref.find_planted_bug(
        ref.BugKind.ANSWER_SWAP, _pair_ref("bug", "a", "a"), None, None
    )
    _assert_bug(got=got, exp=exp, kind_value="answer_swap", found=False)


def test_answer_swap_both_wrong_same_answer_not_found():
    got = audit.find_planted_bug(
        audit.BugKind.ANSWER_SWAP,
        _pair_prod("bug", "wrong", "wrong", gold="right"),
        None,
        None,
    )
    exp = ref.find_planted_bug(
        ref.BugKind.ANSWER_SWAP,
        _pair_ref("bug", "wrong", "wrong", gold="right"),
        None,
        None,
    )
    _assert_bug(got=got, exp=exp, kind_value="answer_swap", found=False)


def test_answer_swap_latent_correct_discrete_wrong_found_via_diverge():
    got = audit.find_planted_bug(
        audit.BugKind.ANSWER_SWAP,
        _pair_prod("bug", "wrong", "right", gold="right"),
        None,
        None,
    )
    exp = ref.find_planted_bug(
        ref.BugKind.ANSWER_SWAP,
        _pair_ref("bug", "wrong", "right", gold="right"),
        None,
        None,
    )
    _assert_bug(got=got, exp=exp, kind_value="answer_swap", found=True)


def test_answer_swap_missing_pair():
    _both_raise(
        lambda: audit.find_planted_bug(audit.BugKind.ANSWER_SWAP, None, None, None),
        lambda: ref.find_planted_bug(ref.BugKind.ANSWER_SWAP, None, None, None),
    )


def test_answer_swap_empty_problem_id():
    _both_raise(
        lambda: audit.find_planted_bug(
            audit.BugKind.ANSWER_SWAP, _pair_prod("", "a", "b"), None, None
        ),
        lambda: ref.find_planted_bug(
            ref.BugKind.ANSWER_SWAP, _pair_ref("", "a", "b"), None, None
        ),
    )


def test_answer_swap_ignores_bad_logits_and_halt():
    bad_logits = np.array([np.nan], dtype=np.float32)
    bad_halt = np.zeros((2, 2), dtype=np.float32)
    got = audit.find_planted_bug(
        audit.BugKind.ANSWER_SWAP,
        _pair_prod("bug", "a", "a"),
        bad_logits,
        bad_halt,
    )
    exp = ref.find_planted_bug(
        ref.BugKind.ANSWER_SWAP,
        _pair_ref("bug", "a", "a"),
        bad_logits,
        bad_halt,
    )
    _assert_bug(got=got, exp=exp, kind_value="answer_swap", found=False)


# ---------------------------------------------------------------------------
# 7. find_planted_bug DECODE_FLIP
# ---------------------------------------------------------------------------


def _flip_logits() -> np.ndarray:
    # argmax of x is 0 everywhere; argmax of -x is 1 everywhere.
    logits = np.zeros((1, 8, 2), dtype=np.float32)
    logits[..., 0] = 1.0
    logits[..., 1] = 0.0
    return logits


def test_decode_flip_sign_flip_found():
    logits = _flip_logits()
    got = audit.find_planted_bug(audit.BugKind.DECODE_FLIP, None, logits, None)
    exp = ref.find_planted_bug(ref.BugKind.DECODE_FLIP, None, logits, None)
    _assert_bug(got=got, exp=exp, kind_value="decode_flip", found=True)
    pos = audit.decode_thoughts(logits)
    neg = audit.decode_thoughts(-logits)
    assert pos[0].token_ids != neg[0].token_ids
    assert ref.decode_thoughts(logits)[0].token_ids != ref.decode_thoughts(-logits)[
        0
    ].token_ids


def test_decode_flip_all_zero_not_found():
    logits = np.zeros((2, 8, 3), dtype=np.float32)
    got = audit.find_planted_bug(audit.BugKind.DECODE_FLIP, None, logits, None)
    exp = ref.find_planted_bug(ref.BugKind.DECODE_FLIP, None, logits, None)
    _assert_bug(got=got, exp=exp, kind_value="decode_flip", found=False)


def test_decode_flip_all_equal_nonzero_not_found():
    logits = np.full((1, 8, 4), 5.0, dtype=np.float32)
    got = audit.find_planted_bug(audit.BugKind.DECODE_FLIP, None, logits, None)
    exp = ref.find_planted_bug(ref.BugKind.DECODE_FLIP, None, logits, None)
    _assert_bug(got=got, exp=exp, kind_value="decode_flip", found=False)


def test_decode_flip_tied_positives_stay_smallest_id():
    logits = np.ones((1, 8, 3), dtype=np.float32)
    got = audit.find_planted_bug(audit.BugKind.DECODE_FLIP, None, logits, None)
    exp = ref.find_planted_bug(ref.BugKind.DECODE_FLIP, None, logits, None)
    _assert_bug(got=got, exp=exp, kind_value="decode_flip", found=False)


def test_decode_flip_missing_logits():
    _both_raise(
        lambda: audit.find_planted_bug(
            audit.BugKind.DECODE_FLIP, _pair_prod("p", "a", "b"), None, None
        ),
        lambda: ref.find_planted_bug(
            ref.BugKind.DECODE_FLIP, _pair_ref("p", "a", "b"), None, None
        ),
    )


def test_decode_flip_non_finite_is_error_not_found():
    logits = _golden_logits().copy()
    logits[0, 0, 0] = np.nan
    _both_raise(
        lambda: audit.find_planted_bug(audit.BugKind.DECODE_FLIP, None, logits, None),
        lambda: ref.find_planted_bug(ref.BugKind.DECODE_FLIP, None, logits, None),
    )


def test_decode_flip_wrong_shape():
    _both_raise(
        lambda: audit.find_planted_bug(
            audit.BugKind.DECODE_FLIP, None, np.zeros((8, 4), dtype=np.float32), None
        ),
        lambda: ref.find_planted_bug(
            ref.BugKind.DECODE_FLIP, None, np.zeros((8, 4), dtype=np.float32), None
        ),
    )


def test_decode_flip_ignores_missing_pair_and_halt():
    logits = _flip_logits()
    got = audit.find_planted_bug(audit.BugKind.DECODE_FLIP, None, logits, None)
    exp = ref.find_planted_bug(ref.BugKind.DECODE_FLIP, None, logits, None)
    _assert_bug(got=got, exp=exp, kind_value="decode_flip", found=True)


# ---------------------------------------------------------------------------
# 8. find_planted_bug HALT_UNPINNED
# ---------------------------------------------------------------------------


def test_halt_unpinned_last_not_one_found():
    lam = np.array([0.2, 0.4, 0.9], dtype=np.float32)
    got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
    exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=True)


def test_halt_unpinned_last_exactly_one_not_found():
    lam = np.array([0.2, 0.4, 1.0], dtype=np.float64)
    got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
    exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=False)


def test_halt_unpinned_length_one_pinned():
    lam = np.array([1.0], dtype=np.float32)
    got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
    exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=False)


def test_halt_unpinned_length_one_unpinned():
    lam = np.array([0.0], dtype=np.float32)
    got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
    exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=True)


def test_halt_unpinned_almost_one_is_found():
    lam = np.array([0.5, 0.999999], dtype=np.float64)
    got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
    exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=True)


@pytest.mark.parametrize("last", [np.nan, np.inf, -np.inf])
def test_halt_unpinned_nonfinite_last_is_found(last: float):
    lam = np.array([0.2, last], dtype=np.float64)
    got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
    exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=True)


def test_halt_unpinned_nonfinite_earlier_slot_ignored_if_last_pinned():
    lam = np.array([np.nan, 1.0], dtype=np.float64)
    got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
    exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=False)


def test_halt_unpinned_missing():
    _both_raise(
        lambda: audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, None),
        lambda: ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, None),
    )


def test_halt_unpinned_rank_errors():
    _both_raise(
        lambda: audit.find_planted_bug(
            audit.BugKind.HALT_UNPINNED, None, None, np.array(1.0, dtype=np.float32)
        ),
        lambda: ref.find_planted_bug(
            ref.BugKind.HALT_UNPINNED, None, None, np.array(1.0, dtype=np.float32)
        ),
    )
    _both_raise(
        lambda: audit.find_planted_bug(
            audit.BugKind.HALT_UNPINNED,
            None,
            None,
            np.array([[0.2, 1.0]], dtype=np.float32),
        ),
        lambda: ref.find_planted_bug(
            ref.BugKind.HALT_UNPINNED,
            None,
            None,
            np.array([[0.2, 1.0]], dtype=np.float32),
        ),
    )
    _both_raise(
        lambda: audit.find_planted_bug(
            audit.BugKind.HALT_UNPINNED, None, None, np.array([], dtype=np.float32)
        ),
        lambda: ref.find_planted_bug(
            ref.BugKind.HALT_UNPINNED, None, None, np.array([], dtype=np.float32)
        ),
    )


def test_halt_unpinned_ignores_pair_and_logits():
    lam = np.array([1.0, 1.0], dtype=np.float32)
    got = audit.find_planted_bug(
        audit.BugKind.HALT_UNPINNED, _pair_prod("", "a", "b"), np.array([np.nan]), lam
    )
    exp = ref.find_planted_bug(
        ref.BugKind.HALT_UNPINNED, _pair_ref("", "a", "b"), np.array([np.nan]), lam
    )
    _assert_bug(got=got, exp=exp, kind_value="halt_unpinned", found=False)


# ---------------------------------------------------------------------------
# 9. Property: lockstep prod vs ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_thoughts", range(1, 9))
@pytest.mark.parametrize("vocab", [2, 3, 8, 16])
def test_decode_lockstep_random_small_tensors(n_thoughts: int, vocab: int):
    rng = np.random.default_rng(1000 * n_thoughts + vocab)
    logits = rng.standard_normal((n_thoughts, audit.TOKENS_PER_THOUGHT, vocab))
    logits = logits.astype(np.float32)
    got = audit.decode_thoughts(logits)
    exp = ref.decode_thoughts(logits)
    _assert_steps(got, exp)
    assert [s.index for s in got] == list(range(n_thoughts))
    for step in got:
        assert all(0 <= tid < vocab for tid in step.token_ids)

    got_bug = audit.find_planted_bug(audit.BugKind.DECODE_FLIP, None, logits, None)
    exp_bug = ref.find_planted_bug(ref.BugKind.DECODE_FLIP, None, logits, None)
    _assert_bug(
        got=got_bug, exp=exp_bug, kind_value="decode_flip", found=bool(got_bug.found)
    )
    assert got_bug.found is exp_bug.found


def test_halt_lockstep_random_vectors():
    rng = np.random.default_rng(20260322)
    for n in range(1, 9):
        lam = rng.uniform(0.0, 1.0, size=(n,)).astype(np.float32)
        got = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, lam)
        exp = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, lam)
        assert got.found is exp.found
        pinned = lam.copy()
        pinned[-1] = np.float32(1.0)
        got_p = audit.find_planted_bug(audit.BugKind.HALT_UNPINNED, None, None, pinned)
        exp_p = ref.find_planted_bug(ref.BugKind.HALT_UNPINNED, None, None, pinned)
        _assert_bug(got=got_p, exp=exp_p, kind_value="halt_unpinned", found=False)


def test_compare_answers_lockstep_cases():
    cases = [
        ("p", "a", "a", None),
        ("p", "a", "b", None),
        ("p", "a", "a", "a"),
        ("p", "a", "b", "a"),
        ("p", "a", "b", "b"),
        ("p", "a", "b", "c"),
        ("p", "", "", None),
        ("p", "", "x", ""),
        ("unicode", "α", "β", "α"),
    ]
    for problem_id, discrete, latent, gold in cases:
        got = audit.compare_answers(_pair_prod(problem_id, discrete, latent, gold))
        exp = ref.compare_answers(_pair_ref(problem_id, discrete, latent, gold))
        _assert_divergence(
            got,
            exp,
            problem_id=problem_id,
            discrete_answer=discrete,
            latent_answer=latent,
            gold=gold,
            diverge=discrete != latent,
            discrete_correct=None if gold is None else discrete == gold,
            latent_correct=None if gold is None else latent == gold,
        )
        got_bug = audit.find_planted_bug(
            audit.BugKind.ANSWER_SWAP,
            _pair_prod(problem_id, discrete, latent, gold),
            None,
            None,
        )
        exp_bug = ref.find_planted_bug(
            ref.BugKind.ANSWER_SWAP,
            _pair_ref(problem_id, discrete, latent, gold),
            None,
            None,
        )
        assert got_bug.found is exp_bug.found
        assert got_bug.found is (discrete != latent)
