"""Implementer-facing tests for I4 SGLang fork deltas (spec 13.2-13.8, 4.1-4.2, 9.2, 10, 15.5).

These import production ``sglang_fork.fork``. Constants, enums, and frozen
dataclass construction may pass against the stub. Every test that calls
``bucket_recurrence``, ``validate_latent_chunk``, ``decode_mode``,
``next_kv_tier``, ``swap_waiting_session``, ``capture_routing``,
``schedule_prefix_group``, or ``register_ttt`` MUST FAIL on the stub with
``NotImplementedError``.

No GPU marker. Discrete serving contract only (no tensors, no gradients,
no live SGLang engine).

Groups:
- constants / interface: RECURRENCE_BUCKETS == (1, 2, 4, 8, 16),
  RECURRENCE_MAX == 16, LATENT_CHUNK_MIN/MAX 4/64, SUBAGENT_DEPTH_MAX == 3,
  KV_TIER_ORDER HBM->GRACE->NVME->DISTRIBUTED, enum values, ForkError is
  ValueError, frozen dataclasses.
- bucket_recurrence: ceil r to next bucket, cap min(budget, 16); kv_shared
  True; ForkError on non-positive r or budget. Goldens: r=3 budget=16 -> 4;
  r=1 budget=16 -> 1; r=16 budget=8 -> 8; r=5 budget=4 -> 4.
- validate_latent_chunk: accepts 4 and 64, ForkError on 3 and 65.
- decode_mode: LATENT iff in_think and not halt; VERBAL otherwise.
- next_kv_tier: HBM->GRACE->NVME->DISTRIBUTED->None.
- swap_waiting_session: decoding True -> HBM; False -> GRACE.
- capture_routing: length==top_k, unique non-negative ids; ForkError on
  duplicates, negatives, length mismatch, top_k<=0.
- schedule_prefix_group: non-empty parent and children, no parent in
  children, depth in [1, 3]; ForkError on empty, overlap, depth 0 or 4.
- register_ttt: both ids non-empty; ForkError on empty.
- properties: brute-force vs the reference over r/budget, chunk sizes,
  routing records, prefix groups, TTT ids.
- edges / faults: r above every bucket, budget above RECURRENCE_MAX,
  expert id 0, depth 1 and 3, one-sided empty TTT ids.

Frozen goldens live in this file and were computed from
``tests.reference.sglang_fork``, not from production.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from sglang_fork import fork

from tests.reference import sglang_fork as ref

# Locked from tests.reference.sglang_fork.bucket_recurrence.
# (r, budget, expected plan.r)
GOLDEN_BUCKETS = (
    (3, 16, 4),
    (1, 16, 1),
    (16, 8, 8),
    (5, 4, 4),
)

GOLDEN_BUCKETS_EXTRA = (
    (2, 16, 2),
    (4, 16, 4),
    (7, 16, 8),
    (9, 16, 16),
    (8, 8, 8),
    (16, 16, 16),
    (16, 1, 1),
    (1, 1, 1),
    (6, 5, 5),
    (20, 16, 16),
    (4, 32, 4),
)


def _assert_plan(got: fork.RecurrencePlan, exp: ref.RecurrencePlan) -> None:
    assert got.r == exp.r
    assert got.budget == exp.budget
    assert got.kv_shared is True
    assert got.kv_shared == exp.kv_shared
    assert isinstance(got, fork.RecurrencePlan)
    assert isinstance(got.r, int)
    assert isinstance(got.budget, int)


def _assert_chunk(got: fork.LatentChunk, exp: ref.LatentChunk) -> None:
    assert got.n_thoughts == exp.n_thoughts
    assert got.mode == exp.mode
    assert got.mode is fork.DecodeMode.LATENT
    assert isinstance(got, fork.LatentChunk)


def _assert_routing(got: fork.RoutingRecord, exp: ref.RoutingRecord) -> None:
    assert got.token_index == exp.token_index
    assert got.layer_index == exp.layer_index
    assert got.expert_ids == exp.expert_ids
    assert isinstance(got.expert_ids, tuple)
    assert isinstance(got, fork.RoutingRecord)


def _assert_group(got: fork.PrefixGroup, exp: ref.PrefixGroup) -> None:
    assert got.parent_session == exp.parent_session
    assert got.child_sessions == exp.child_sessions
    assert got.depth == exp.depth
    assert isinstance(got.child_sessions, tuple)
    assert isinstance(got, fork.PrefixGroup)


def _assert_ttt(got: fork.TttHook, exp: ref.TttHook) -> None:
    assert got.task_id == exp.task_id
    assert got.lora_id == exp.lora_id
    assert isinstance(got, fork.TttHook)


# ---------------------------------------------------------------------------
# constants / interface (may pass on the stub)
# ---------------------------------------------------------------------------


def test_recurrence_buckets_and_max():
    assert fork.RECURRENCE_BUCKETS == (1, 2, 4, 8, 16)
    assert fork.RECURRENCE_MAX == 16
    assert fork.RECURRENCE_BUCKETS == ref.RECURRENCE_BUCKETS
    assert fork.RECURRENCE_MAX == ref.RECURRENCE_MAX
    assert fork.RECURRENCE_MAX == fork.RECURRENCE_BUCKETS[-1]


def test_latent_chunk_bounds():
    assert fork.LATENT_CHUNK_MIN == 4
    assert fork.LATENT_CHUNK_MAX == 64
    assert fork.LATENT_CHUNK_MIN == ref.LATENT_CHUNK_MIN
    assert fork.LATENT_CHUNK_MAX == ref.LATENT_CHUNK_MAX
    assert fork.LATENT_CHUNK_MIN < fork.LATENT_CHUNK_MAX


def test_subagent_depth_max():
    assert fork.SUBAGENT_DEPTH_MAX == 3
    assert fork.SUBAGENT_DEPTH_MAX == ref.SUBAGENT_DEPTH_MAX


def test_kv_tier_order_constant():
    assert fork.KV_TIER_ORDER == (
        fork.KvTier.HBM,
        fork.KvTier.GRACE,
        fork.KvTier.NVME,
        fork.KvTier.DISTRIBUTED,
    )
    assert tuple(t.value for t in fork.KV_TIER_ORDER) == tuple(
        t.value for t in ref.KV_TIER_ORDER
    )


def test_decode_mode_enum_values():
    assert fork.DecodeMode.VERBAL == "verbal"
    assert fork.DecodeMode.LATENT == "latent"
    assert fork.DecodeMode.VERBAL == ref.DecodeMode.VERBAL
    assert fork.DecodeMode.LATENT == ref.DecodeMode.LATENT


def test_kv_tier_enum_values():
    assert fork.KvTier.HBM == "hbm"
    assert fork.KvTier.GRACE == "grace"
    assert fork.KvTier.NVME == "nvme"
    assert fork.KvTier.DISTRIBUTED == "distributed"
    assert fork.KvTier.HBM == ref.KvTier.HBM
    assert fork.KvTier.GRACE == ref.KvTier.GRACE
    assert fork.KvTier.NVME == ref.KvTier.NVME
    assert fork.KvTier.DISTRIBUTED == ref.KvTier.DISTRIBUTED


def test_fork_error_is_value_error():
    assert issubclass(fork.ForkError, ValueError)


def test_dataclasses_are_frozen():
    plan = fork.RecurrencePlan(r=4, budget=16)
    chunk = fork.LatentChunk(n_thoughts=8)
    record = fork.RoutingRecord(token_index=0, layer_index=1, expert_ids=(0, 1))
    group = fork.PrefixGroup(
        parent_session="p", child_sessions=("c",), depth=1
    )
    hook = fork.TttHook(task_id="t", lora_id="l")
    with pytest.raises(FrozenInstanceError):
        plan.r = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        chunk.n_thoughts = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        record.token_index = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        group.depth = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        hook.task_id = "x"  # type: ignore[misc]
    assert plan.kv_shared is True
    assert chunk.mode is fork.DecodeMode.LATENT


# ---------------------------------------------------------------------------
# bucket_recurrence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r, budget, want_r", GOLDEN_BUCKETS)
def test_bucket_recurrence_goldens(r: int, budget: int, want_r: int):
    got = fork.bucket_recurrence(r, budget)
    exp = ref.bucket_recurrence(r, budget)
    _assert_plan(got, exp)
    assert got.r == want_r
    assert got.budget == budget
    assert got.kv_shared is True


@pytest.mark.parametrize("r, budget, want_r", GOLDEN_BUCKETS_EXTRA)
def test_bucket_recurrence_extra_goldens(r: int, budget: int, want_r: int):
    got = fork.bucket_recurrence(r, budget)
    exp = ref.bucket_recurrence(r, budget)
    _assert_plan(got, exp)
    assert got.r == want_r


@pytest.mark.parametrize("r", [0, -1, -16])
def test_bucket_recurrence_non_positive_r_raises(r: int):
    with pytest.raises(fork.ForkError):
        fork.bucket_recurrence(r, 16)
    with pytest.raises(ref.ForkError):
        ref.bucket_recurrence(r, 16)


@pytest.mark.parametrize("budget", [0, -1, -8])
def test_bucket_recurrence_non_positive_budget_raises(budget: int):
    with pytest.raises(fork.ForkError):
        fork.bucket_recurrence(4, budget)
    with pytest.raises(ref.ForkError):
        ref.bucket_recurrence(4, budget)


def test_bucket_recurrence_both_non_positive_raises():
    with pytest.raises(fork.ForkError):
        fork.bucket_recurrence(0, 0)
    with pytest.raises(fork.ForkError):
        fork.bucket_recurrence(-3, -3)


def test_bucket_recurrence_kv_shared_always_true():
    for r, budget in ((1, 16), (3, 16), (16, 8), (5, 4), (9, 32)):
        got = fork.bucket_recurrence(r, budget)
        assert got.kv_shared is True


def test_property_bucket_recurrence_matches_reference():
    for r in range(1, 33):
        for budget in range(1, 33):
            got = fork.bucket_recurrence(r, budget)
            exp = ref.bucket_recurrence(r, budget)
            _assert_plan(got, exp)
            assert got.r <= min(budget, fork.RECURRENCE_MAX)
            assert got.r >= 1


def test_bucket_recurrence_exact_buckets_stay():
    for bucket in fork.RECURRENCE_BUCKETS:
        got = fork.bucket_recurrence(bucket, 16)
        assert got.r == bucket
        assert got.r == ref.bucket_recurrence(bucket, 16).r


# ---------------------------------------------------------------------------
# validate_latent_chunk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 64])
def test_validate_latent_chunk_bounds_accepted(n: int):
    got = fork.validate_latent_chunk(n)
    exp = ref.validate_latent_chunk(n)
    _assert_chunk(got, exp)
    assert got.n_thoughts == n


@pytest.mark.parametrize("n", [3, 65])
def test_validate_latent_chunk_bounds_rejected(n: int):
    with pytest.raises(fork.ForkError):
        fork.validate_latent_chunk(n)
    with pytest.raises(ref.ForkError):
        ref.validate_latent_chunk(n)


@pytest.mark.parametrize("n", [5, 8, 16, 32, 63])
def test_validate_latent_chunk_interior(n: int):
    got = fork.validate_latent_chunk(n)
    exp = ref.validate_latent_chunk(n)
    _assert_chunk(got, exp)


@pytest.mark.parametrize("n", [-1, 0, 1, 2, 66, 100])
def test_validate_latent_chunk_outside_raises(n: int):
    with pytest.raises(fork.ForkError):
        fork.validate_latent_chunk(n)


def test_property_validate_latent_chunk_matches_reference():
    for n in range(-2, 70):
        if fork.LATENT_CHUNK_MIN <= n <= fork.LATENT_CHUNK_MAX:
            got = fork.validate_latent_chunk(n)
            exp = ref.validate_latent_chunk(n)
            _assert_chunk(got, exp)
        else:
            with pytest.raises(fork.ForkError):
                fork.validate_latent_chunk(n)
            with pytest.raises(ref.ForkError):
                ref.validate_latent_chunk(n)


# ---------------------------------------------------------------------------
# decode_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "in_think, halt, want",
    [
        (True, False, fork.DecodeMode.LATENT),
        (True, True, fork.DecodeMode.VERBAL),
        (False, False, fork.DecodeMode.VERBAL),
        (False, True, fork.DecodeMode.VERBAL),
    ],
)
def test_decode_mode_truth_table(in_think: bool, halt: bool, want: fork.DecodeMode):
    got = fork.decode_mode(in_think, halt)
    exp = ref.decode_mode(in_think, halt)
    assert got == exp
    assert got is want
    if in_think and not halt:
        assert got is fork.DecodeMode.LATENT
        assert got == "latent"
    else:
        assert got is fork.DecodeMode.VERBAL
        assert got == "verbal"


def test_decode_mode_latent_only_inside_think_and_not_halt():
    assert fork.decode_mode(True, False) is fork.DecodeMode.LATENT
    for in_think, halt in ((True, True), (False, False), (False, True)):
        assert fork.decode_mode(in_think, halt) is fork.DecodeMode.VERBAL


# ---------------------------------------------------------------------------
# next_kv_tier
# ---------------------------------------------------------------------------


def test_next_kv_tier_walks_colder():
    assert fork.next_kv_tier(fork.KvTier.HBM) is fork.KvTier.GRACE
    assert fork.next_kv_tier(fork.KvTier.GRACE) is fork.KvTier.NVME
    assert fork.next_kv_tier(fork.KvTier.NVME) is fork.KvTier.DISTRIBUTED
    assert fork.next_kv_tier(fork.KvTier.DISTRIBUTED) is None


def test_next_kv_tier_matches_reference_and_order():
    order = fork.KV_TIER_ORDER
    for index, tier in enumerate(order):
        got = fork.next_kv_tier(tier)
        exp = ref.next_kv_tier(ref.KV_TIER_ORDER[index])
        if index + 1 < len(order):
            assert got is order[index + 1]
            assert got == exp
        else:
            assert got is None
            assert exp is None


# ---------------------------------------------------------------------------
# swap_waiting_session
# ---------------------------------------------------------------------------


def test_swap_waiting_session_decoding_stays_hbm():
    got = fork.swap_waiting_session(True)
    exp = ref.swap_waiting_session(True)
    assert got is fork.KvTier.HBM
    assert got == exp == "hbm"


def test_swap_waiting_session_waiting_moves_to_grace():
    got = fork.swap_waiting_session(False)
    exp = ref.swap_waiting_session(False)
    assert got is fork.KvTier.GRACE
    assert got == exp == "grace"


def test_swap_waiting_session_never_nvme_or_distributed():
    assert fork.swap_waiting_session(True) is not fork.KvTier.NVME
    assert fork.swap_waiting_session(False) is not fork.KvTier.NVME
    assert fork.swap_waiting_session(True) is not fork.KvTier.DISTRIBUTED
    assert fork.swap_waiting_session(False) is not fork.KvTier.DISTRIBUTED


# ---------------------------------------------------------------------------
# capture_routing
# ---------------------------------------------------------------------------


def test_capture_routing_valid_record():
    got = fork.capture_routing(3, 5, (1, 4, 7), 3)
    exp = ref.capture_routing(3, 5, (1, 4, 7), 3)
    _assert_routing(got, exp)
    assert got.token_index == 3
    assert got.layer_index == 5
    assert got.expert_ids == (1, 4, 7)


def test_capture_routing_preserves_order_and_zero():
    got = fork.capture_routing(0, 0, (0, 2), 2)
    exp = ref.capture_routing(0, 0, (0, 2), 2)
    _assert_routing(got, exp)
    assert got.expert_ids == (0, 2)


def test_capture_routing_duplicates_raise():
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 0, (1, 1), 2)
    with pytest.raises(ref.ForkError):
        ref.capture_routing(0, 0, (1, 1), 2)
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 1, (4, 5, 4), 3)


def test_capture_routing_negatives_raise():
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 0, (0, -1), 2)
    with pytest.raises(ref.ForkError):
        ref.capture_routing(0, 0, (0, -1), 2)
    with pytest.raises(fork.ForkError):
        fork.capture_routing(1, 1, (-3,), 1)


def test_capture_routing_length_mismatch_raises():
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 0, (1, 2, 3), 2)
    with pytest.raises(ref.ForkError):
        ref.capture_routing(0, 0, (1, 2, 3), 2)
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 0, (1,), 2)
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 0, (), 1)


@pytest.mark.parametrize("top_k", [0, -1, -8])
def test_capture_routing_non_positive_top_k_raises(top_k: int):
    ids = tuple(range(max(top_k, 0)))
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 0, ids, top_k)
    with pytest.raises(ref.ForkError):
        ref.capture_routing(0, 0, ids, top_k)


def test_capture_routing_top_k_zero_empty_ids_raises():
    with pytest.raises(fork.ForkError):
        fork.capture_routing(0, 0, (), 0)


def test_property_capture_routing_matches_reference():
    cases = (
        (0, 0, (0,), 1),
        (2, 7, (1, 3, 5, 9), 4),
        (11, 2, (8, 0, 1), 3),
        (4, 4, tuple(range(8)), 8),
    )
    for token_index, layer_index, expert_ids, top_k in cases:
        got = fork.capture_routing(token_index, layer_index, expert_ids, top_k)
        exp = ref.capture_routing(token_index, layer_index, expert_ids, top_k)
        _assert_routing(got, exp)
        assert len(got.expert_ids) == top_k
        assert len(set(got.expert_ids)) == top_k
        assert min(got.expert_ids) >= 0


# ---------------------------------------------------------------------------
# schedule_prefix_group
# ---------------------------------------------------------------------------


def test_schedule_prefix_group_valid():
    got = fork.schedule_prefix_group("parent-a", ("child-1", "child-2"), 1)
    exp = ref.schedule_prefix_group("parent-a", ("child-1", "child-2"), 1)
    _assert_group(got, exp)
    assert got.parent_session == "parent-a"
    assert got.child_sessions == ("child-1", "child-2")
    assert got.depth == 1


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_schedule_prefix_group_depth_in_range(depth: int):
    got = fork.schedule_prefix_group("p", ("c",), depth)
    exp = ref.schedule_prefix_group("p", ("c",), depth)
    _assert_group(got, exp)
    assert got.depth == depth


def test_schedule_prefix_group_empty_parent_raises():
    with pytest.raises(fork.ForkError):
        fork.schedule_prefix_group("", ("c",), 1)
    with pytest.raises(ref.ForkError):
        ref.schedule_prefix_group("", ("c",), 1)


def test_schedule_prefix_group_empty_children_raises():
    with pytest.raises(fork.ForkError):
        fork.schedule_prefix_group("p", (), 1)
    with pytest.raises(ref.ForkError):
        ref.schedule_prefix_group("p", (), 1)


def test_schedule_prefix_group_overlap_raises():
    with pytest.raises(fork.ForkError):
        fork.schedule_prefix_group("p", ("p", "c"), 1)
    with pytest.raises(ref.ForkError):
        ref.schedule_prefix_group("p", ("p", "c"), 1)
    with pytest.raises(fork.ForkError):
        fork.schedule_prefix_group("sess", ("sess",), 2)


@pytest.mark.parametrize("depth", [0, 4])
def test_schedule_prefix_group_depth_out_of_range_raises(depth: int):
    with pytest.raises(fork.ForkError):
        fork.schedule_prefix_group("p", ("c",), depth)
    with pytest.raises(ref.ForkError):
        ref.schedule_prefix_group("p", ("c",), depth)


@pytest.mark.parametrize("depth", [-1, 5, 99])
def test_schedule_prefix_group_depth_far_out_of_range_raises(depth: int):
    with pytest.raises(fork.ForkError):
        fork.schedule_prefix_group("p", ("c",), depth)


def test_schedule_prefix_group_preserves_child_order():
    children = ("c3", "c1", "c2")
    got = fork.schedule_prefix_group("parent", children, 2)
    exp = ref.schedule_prefix_group("parent", children, 2)
    _assert_group(got, exp)
    assert got.child_sessions == children


# ---------------------------------------------------------------------------
# register_ttt
# ---------------------------------------------------------------------------


def test_register_ttt_valid():
    got = fork.register_ttt("arc-001", "lora-7")
    exp = ref.register_ttt("arc-001", "lora-7")
    _assert_ttt(got, exp)
    assert got.task_id == "arc-001"
    assert got.lora_id == "lora-7"


def test_register_ttt_empty_task_id_raises():
    with pytest.raises(fork.ForkError):
        fork.register_ttt("", "lora-7")
    with pytest.raises(ref.ForkError):
        ref.register_ttt("", "lora-7")


def test_register_ttt_empty_lora_id_raises():
    with pytest.raises(fork.ForkError):
        fork.register_ttt("arc-001", "")
    with pytest.raises(ref.ForkError):
        ref.register_ttt("arc-001", "")


def test_register_ttt_both_empty_raises():
    with pytest.raises(fork.ForkError):
        fork.register_ttt("", "")


def test_property_register_ttt_matches_reference():
    for task_id, lora_id in (
        ("task", "lora"),
        ("a", "b"),
        ("arc-agi-2", "adapter-0"),
    ):
        got = fork.register_ttt(task_id, lora_id)
        exp = ref.register_ttt(task_id, lora_id)
        _assert_ttt(got, exp)
