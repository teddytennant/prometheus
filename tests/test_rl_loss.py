"""Implementer-facing tests for I2 ``rl.loss`` (spec 9.1, 9.2, 4.4, 15.5).

These import production ``rl.loss``. Constants, ``LossError`` subclassing,
frozen dataclass construction, and reading ``GROUP_SIZE`` / ``MAX_STALENESS`` /
clip defaults MAY pass against the stub. Every test that calls ``make_config``,
``mean_center``, ``all_equal_reward``, ``drop_zero_advantage_groups``,
``sequence_ratio``, ``clip_higher``, ``staleness``, ``truncated_is``,
``overlong_soft_penalty``, ``gaussian_log_density``, ``latent_ratio``,
``routing_table``, or ``gspo_dapo_loss`` MUST FAIL on the stub with
``NotImplementedError``.

CPU analog of V7 math only. No GPU marker, no JAX, no model forward.

Groups
------
constants / interface: GROUP_SIZE 16, MAX_STALENESS 4, clip defaults 0.2/0.28,
  LossError is ValueError, dataclasses frozen.
make_config: accepts defaults; LossError on negative epsilons, non-finite or
  non-positive loss_normalizer, negative overlong_cache/penalty, max_staleness
  0 or 5, tis_clip 0.5 or non-finite, kl_coeff != 0.
mean_center: subtract group mean, no std; empty or non-finite -> LossError.
  Golden: (1, 3, 5) -> (-2, 0, 2).
all_equal_reward: True for (1,1,1), False for (1,1,2); empty -> LossError.
drop_zero_advantage_groups: drops all-equal groups; empty input -> empty;
  group with no samples -> LossError.
sequence_ratio: exp(sum(new-old)), NO divide by length. Length mismatch,
  empty, non-finite -> LossError. Goldens 1.0 and 2.0.
clip_higher: clip to [1-eps_low, 1+eps_high]. 1.5 -> 1.28; 0.5 -> 0.8.
staleness: trainer - policy; negative -> LossError.
truncated_is: k==0 -> 1.0 even if logps differ; k>0 -> min(ratio, tis_clip).
overlong_soft_penalty: 0 when n_tokens<=cache; else penalty*(n_tokens-cache).
gaussian_log_density: independent 1-D Gaussians; finite-difference d/dz
  matches -(z-mu)/sigma^2.
latent_ratio: exp(sum(new-old)); both empty -> 1.0.
routing_table: dense [token][layer] -> expert_ids of length top_k.
gspo_dapo_loss: all-equal -> zeros n_kept=0; otherwise mean-center, sequence-
  clip discrete, latent-clip latent separately, multiply by truncated IS,
  add overlong, divide by loss_normalizer. No reference KL. Totals vs
  reference at 1e-5.
properties: brute-force vs reference over small groups (2-4 samples, 1-4
  tokens), k in 0..4, with/without latents, with/without routing.
edges: GROUP_SIZE=16 group, expert id 0, cache=0, tis_clip=1.0, discrete-only.

Frozen goldens live in this file and were computed from
``tests.reference.rl_loss``, not from production.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

import rl.loss as loss
from tests.reference import rl_loss as ref

TOL = dict(rtol=1e-5, atol=1e-5)

# Locked from tests.reference.rl_loss (not production).
GOLDEN_MEAN_CENTER = (-2.0, 0.0, 2.0)
GOLDEN_SEQ_RATIO_ONES = 1.0
GOLDEN_SEQ_RATIO_TWO = 2.0
GOLDEN_SEQ_RATIO_NO_LEN_NORM = 4.0
GOLDEN_CLIP_HIGH = 1.28
GOLDEN_CLIP_LOW = 0.8
GOLDEN_GAUSS_STANDARD = -0.9189385332046727
GOLDEN_ON_POLICY_TOTAL = 0.28
GOLDEN_ON_POLICY_PG = 0.28
GOLDEN_OVERLONG_TOTAL = 4.28
GOLDEN_K2_TIS2_TOTAL = 1.56
GOLDEN_K2_TIS2_PG = 1.56
GOLDEN_K2_TIS2_TIS = 1.5
GOLDEN_K2_TIS1_TOTAL = 0.28
GOLDEN_LATENT_TOTAL = 0.56
GOLDEN_LATENT_PG = 0.28
GOLDEN_LATENT_LATENT = 0.28
GOLDEN_Z2_TOTAL = 0.14
GOLDEN_G16_TOTAL = 3.558673862627357
GOLDEN_OVERLONG_CACHE0 = 2.0
GOLDEN_OVERLONG_INTERIOR = 0.2


def _close(got: object, exp: object) -> None:
    np.testing.assert_allclose(got, exp, **TOL)


def _assert_breakdown(got: object, exp: object) -> None:
    assert got.n_kept == exp.n_kept
    for name in ("pg", "tis", "overlong", "latent", "total"):
        _close(getattr(got, name), getattr(exp, name))


def _assert_config(got: object, exp: object) -> None:
    for field in fields(exp):
        gv = getattr(got, field.name)
        ev = getattr(exp, field.name)
        if isinstance(ev, float):
            _close(gv, ev)
        else:
            assert gv == ev


def _trace(mod, token_index: int, layer_index: int, expert_ids: tuple[int, ...]):
    return mod.RoutingTrace(
        token_index=token_index,
        layer_index=layer_index,
        expert_ids=expert_ids,
    )


def _sample(mod, **kwargs):
    fields_ = dict(
        token_logp_trainer=(0.0,),
        token_logp_rollout=(0.0,),
        latent_logp_trainer=(),
        latent_logp_rollout=(),
        reward=1.0,
        n_tokens=1,
        policy_version=0,
        routing=(),
    )
    fields_.update(kwargs)
    return mod.Sample(**fields_)


def _group(mod, samples, prompt_id: str = "prompt"):
    return mod.Group(prompt_id=prompt_id, samples=tuple(samples))


def _to_ref_trace(trace) -> ref.RoutingTrace:
    return ref.RoutingTrace(
        token_index=trace.token_index,
        layer_index=trace.layer_index,
        expert_ids=tuple(trace.expert_ids),
    )


def _to_ref_sample(sample) -> ref.Sample:
    return ref.Sample(
        token_logp_trainer=tuple(sample.token_logp_trainer),
        token_logp_rollout=tuple(sample.token_logp_rollout),
        latent_logp_trainer=tuple(sample.latent_logp_trainer),
        latent_logp_rollout=tuple(sample.latent_logp_rollout),
        reward=float(sample.reward),
        n_tokens=int(sample.n_tokens),
        policy_version=int(sample.policy_version),
        routing=tuple(_to_ref_trace(t) for t in sample.routing),
    )


def _to_ref_group(group) -> ref.Group:
    return ref.Group(
        prompt_id=group.prompt_id,
        samples=tuple(_to_ref_sample(s) for s in group.samples),
    )


def _to_ref_config(config) -> ref.LossConfig:
    return ref.LossConfig(**{field.name: getattr(config, field.name) for field in fields(config)})


def _golden_pair(mod, *, latents: bool = False):
    """Two-sample discrete (optional latent) group used for frozen goldens."""
    log2 = math.log(2.0)
    lat0 = (0.0,) if latents else ()
    lat1 = (log2,) if latents else ()
    lat_old = (0.0,) if latents else ()
    return _group(
        mod,
        (
            _sample(
                mod,
                token_logp_trainer=(0.0, 0.0),
                token_logp_rollout=(0.0, 0.0),
                latent_logp_trainer=lat0,
                latent_logp_rollout=lat_old,
                reward=1.0,
                n_tokens=2,
            ),
            _sample(
                mod,
                token_logp_trainer=(0.0, log2),
                token_logp_rollout=(0.0, 0.0),
                latent_logp_trainer=lat1,
                latent_logp_rollout=lat_old,
                reward=3.0,
                n_tokens=2,
            ),
        ),
    )


def _g16(mod):
    samples = tuple(
        _sample(
            mod,
            token_logp_trainer=(0.1 * i,),
            token_logp_rollout=(0.0,),
            reward=float(i),
            n_tokens=1,
        )
        for i in range(loss.GROUP_SIZE)
    )
    return _group(mod, samples, prompt_id="prompt-16")


# ---------------------------------------------------------------------------
# constants / interface (may pass on the stub)
# ---------------------------------------------------------------------------


def test_group_size_is_16():
    assert loss.GROUP_SIZE == 16
    assert ref.GROUP_SIZE == 16


def test_max_staleness_is_4():
    assert loss.MAX_STALENESS == 4
    assert ref.MAX_STALENESS == 4


def test_clip_defaults():
    assert loss.DEFAULT_CLIP_EPS_LOW == 0.2
    assert loss.DEFAULT_CLIP_EPS_HIGH == 0.28
    assert ref.DEFAULT_CLIP_EPS_LOW == 0.2
    assert ref.DEFAULT_CLIP_EPS_HIGH == 0.28


def test_loss_error_is_value_error():
    assert issubclass(loss.LossError, ValueError)
    assert issubclass(ref.LossError, ValueError)


def test_dataclasses_are_frozen():
    trace = loss.RoutingTrace(token_index=0, layer_index=0, expert_ids=(0,))
    sample = _sample(loss)
    group = _group(loss, (sample,))
    config = loss.LossConfig()
    breakdown = loss.LossBreakdown(pg=0.0, tis=0.0, overlong=0.0, latent=0.0, n_kept=0, total=0.0)
    with pytest.raises(FrozenInstanceError):
        trace.token_index = 1
    with pytest.raises(FrozenInstanceError):
        sample.reward = 0.0
    with pytest.raises(FrozenInstanceError):
        group.prompt_id = "x"
    with pytest.raises(FrozenInstanceError):
        config.kl_coeff = 1.0
    with pytest.raises(FrozenInstanceError):
        breakdown.total = 1.0


def test_loss_config_dataclass_defaults():
    config = loss.LossConfig()
    assert config.clip_eps_low == loss.DEFAULT_CLIP_EPS_LOW
    assert config.clip_eps_high == loss.DEFAULT_CLIP_EPS_HIGH
    assert config.latent_clip_eps_low == loss.DEFAULT_CLIP_EPS_LOW
    assert config.latent_clip_eps_high == loss.DEFAULT_CLIP_EPS_HIGH
    assert config.loss_normalizer == 1.0
    assert config.overlong_cache == 0
    assert config.overlong_penalty == 0.0
    assert config.max_staleness == loss.MAX_STALENESS
    assert config.tis_clip == 1.0
    assert config.kl_coeff == 0.0


# ---------------------------------------------------------------------------
# make_config
# ---------------------------------------------------------------------------


def test_make_config_defaults_match_reference():
    got = loss.make_config()
    exp = ref.make_config()
    _assert_config(got, exp)
    assert got.clip_eps_low == loss.DEFAULT_CLIP_EPS_LOW
    assert got.clip_eps_high == loss.DEFAULT_CLIP_EPS_HIGH
    assert got.tis_clip == 1.0
    assert got.kl_coeff == 0.0
    assert got.max_staleness == loss.MAX_STALENESS


def test_make_config_accepts_custom_valid_values():
    got = loss.make_config(
        clip_eps_low=0.0,
        clip_eps_high=0.0,
        latent_clip_eps_low=0.1,
        latent_clip_eps_high=0.3,
        loss_normalizer=16.0,
        overlong_cache=4,
        overlong_penalty=0.01,
        max_staleness=1,
        tis_clip=3.0,
        kl_coeff=0.0,
    )
    exp = ref.make_config(
        clip_eps_low=0.0,
        clip_eps_high=0.0,
        latent_clip_eps_low=0.1,
        latent_clip_eps_high=0.3,
        loss_normalizer=16.0,
        overlong_cache=4,
        overlong_penalty=0.01,
        max_staleness=1,
        tis_clip=3.0,
        kl_coeff=0.0,
    )
    _assert_config(got, exp)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_eps_low": -0.01},
        {"clip_eps_high": -0.01},
        {"latent_clip_eps_low": -0.2},
        {"latent_clip_eps_high": -0.28},
        {"loss_normalizer": 0.0},
        {"loss_normalizer": -1.0},
        {"loss_normalizer": math.nan},
        {"loss_normalizer": math.inf},
        {"loss_normalizer": -math.inf},
        {"overlong_cache": -1},
        {"overlong_penalty": -0.1},
        {"max_staleness": 0},
        {"max_staleness": 5},
        {"max_staleness": -1},
        {"max_staleness": 6},
        {"tis_clip": 0.5},
        {"tis_clip": 0.0},
        {"tis_clip": math.nan},
        {"tis_clip": math.inf},
        {"tis_clip": -math.inf},
        {"kl_coeff": 1.0},
        {"kl_coeff": 0.1},
        {"kl_coeff": -1.0},
    ],
)
def test_make_config_rejects_invalid(kwargs: dict):
    with pytest.raises(loss.LossError):
        loss.make_config(**kwargs)
    with pytest.raises(ref.LossError):
        ref.make_config(**kwargs)


def test_make_config_max_staleness_one_through_four():
    for k in (1, 2, 3, 4):
        got = loss.make_config(max_staleness=k)
        exp = ref.make_config(max_staleness=k)
        assert got.max_staleness == k
        _assert_config(got, exp)


# ---------------------------------------------------------------------------
# mean_center
# ---------------------------------------------------------------------------


def test_mean_center_golden_1_3_5():
    got = loss.mean_center((1.0, 3.0, 5.0))
    exp = ref.mean_center((1.0, 3.0, 5.0))
    _close(got, GOLDEN_MEAN_CENTER)
    _close(exp, GOLDEN_MEAN_CENTER)
    _close(got, exp)
    assert len(got) == 3
    assert all(isinstance(v, float) for v in got)


def test_mean_center_no_std_divide():
    # std of (1,3,5) is 2; dividing by std would yield (-1, 0, 1).
    got = loss.mean_center((1.0, 3.0, 5.0))
    _close(got, (-2.0, 0.0, 2.0))
    assert got != pytest.approx((-1.0, 0.0, 1.0))


def test_mean_center_advantages_sum_to_zero():
    rewards = (1.0, 2.0, 4.0, 7.0)
    got = loss.mean_center(rewards)
    _close(sum(got), 0.0)
    _close(got, ref.mean_center(rewards))


def test_mean_center_single_is_zero():
    got = loss.mean_center((5.0,))
    _close(got, (0.0,))


def test_mean_center_empty_raises():
    with pytest.raises(loss.LossError):
        loss.mean_center(())
    with pytest.raises(ref.LossError):
        ref.mean_center(())


@pytest.mark.parametrize("bad", [(math.nan,), (math.inf,), (1.0, -math.inf)])
def test_mean_center_non_finite_raises(bad: tuple[float, ...]):
    with pytest.raises(loss.LossError):
        loss.mean_center(bad)
    with pytest.raises(ref.LossError):
        ref.mean_center(bad)


# ---------------------------------------------------------------------------
# all_equal_reward
# ---------------------------------------------------------------------------


def test_all_equal_reward_true_and_false():
    assert loss.all_equal_reward((1.0, 1.0, 1.0)) is True
    assert ref.all_equal_reward((1.0, 1.0, 1.0)) is True
    assert loss.all_equal_reward((1.0, 1.0, 2.0)) is False
    assert ref.all_equal_reward((1.0, 1.0, 2.0)) is False


def test_all_equal_reward_single_true():
    assert loss.all_equal_reward((4.0,)) is True
    assert ref.all_equal_reward((4.0,)) is True


def test_all_equal_reward_exact_not_close():
    assert loss.all_equal_reward((1.0, 1.0 + 1e-12)) is False


def test_all_equal_reward_empty_raises():
    with pytest.raises(loss.LossError):
        loss.all_equal_reward(())
    with pytest.raises(ref.LossError):
        ref.all_equal_reward(())


# ---------------------------------------------------------------------------
# drop_zero_advantage_groups
# ---------------------------------------------------------------------------


def test_drop_zero_advantage_groups_drops_all_equal():
    keep = _group(
        loss,
        (_sample(loss, reward=1.0), _sample(loss, reward=2.0)),
        prompt_id="keep",
    )
    drop = _group(
        loss,
        (_sample(loss, reward=3.0), _sample(loss, reward=3.0)),
        prompt_id="drop",
    )
    got = loss.drop_zero_advantage_groups((keep, drop, keep))
    exp = ref.drop_zero_advantage_groups(
        (_to_ref_group(keep), _to_ref_group(drop), _to_ref_group(keep))
    )
    assert tuple(g.prompt_id for g in got) == ("keep", "keep")
    assert tuple(g.prompt_id for g in exp) == ("keep", "keep")


def test_drop_zero_advantage_groups_empty_input():
    got = loss.drop_zero_advantage_groups(())
    exp = ref.drop_zero_advantage_groups(())
    assert got == ()
    assert exp == ()


def test_drop_zero_advantage_groups_empty_group_raises():
    empty = loss.Group(prompt_id="e", samples=())
    with pytest.raises(loss.LossError):
        loss.drop_zero_advantage_groups((empty,))
    with pytest.raises(ref.LossError):
        ref.drop_zero_advantage_groups((ref.Group(prompt_id="e", samples=()),))


def test_drop_zero_advantage_groups_empty_among_valid_raises():
    valid = _group(loss, (_sample(loss, reward=1.0), _sample(loss, reward=2.0)))
    empty = loss.Group(prompt_id="e", samples=())
    with pytest.raises(loss.LossError):
        loss.drop_zero_advantage_groups((valid, empty))


# ---------------------------------------------------------------------------
# sequence_ratio
# ---------------------------------------------------------------------------


def test_sequence_ratio_golden_ones():
    got = loss.sequence_ratio((0.0, 0.0), (0.0, 0.0))
    exp = ref.sequence_ratio((0.0, 0.0), (0.0, 0.0))
    _close(got, GOLDEN_SEQ_RATIO_ONES)
    _close(exp, GOLDEN_SEQ_RATIO_ONES)
    assert isinstance(got, float)


def test_sequence_ratio_golden_two():
    new = (0.0, math.log(2.0))
    old = (0.0, 0.0)
    got = loss.sequence_ratio(new, old)
    exp = ref.sequence_ratio(new, old)
    _close(got, GOLDEN_SEQ_RATIO_TWO)
    _close(exp, GOLDEN_SEQ_RATIO_TWO)


def test_sequence_ratio_no_length_normalization():
    new = (math.log(2.0), math.log(2.0))
    old = (0.0, 0.0)
    got = loss.sequence_ratio(new, old)
    exp = ref.sequence_ratio(new, old)
    _close(got, GOLDEN_SEQ_RATIO_NO_LEN_NORM)
    _close(exp, GOLDEN_SEQ_RATIO_NO_LEN_NORM)
    # Length-normalized GSPO would return 2.0, not 4.0.
    assert got != pytest.approx(2.0)


def test_sequence_ratio_empty_raises():
    with pytest.raises(loss.LossError):
        loss.sequence_ratio((), ())
    with pytest.raises(loss.LossError):
        loss.sequence_ratio((), (0.0,))
    with pytest.raises(loss.LossError):
        loss.sequence_ratio((0.0,), ())
    with pytest.raises(ref.LossError):
        ref.sequence_ratio((), ())


def test_sequence_ratio_length_mismatch_raises():
    with pytest.raises(loss.LossError):
        loss.sequence_ratio((0.0, 1.0), (0.0,))
    with pytest.raises(ref.LossError):
        ref.sequence_ratio((0.0, 1.0), (0.0,))


@pytest.mark.parametrize(
    "new, old",
    [
        ((math.nan,), (0.0,)),
        ((0.0,), (math.inf,)),
        ((-math.inf, 0.0), (0.0, 0.0)),
    ],
)
def test_sequence_ratio_non_finite_raises(new: tuple, old: tuple):
    with pytest.raises(loss.LossError):
        loss.sequence_ratio(new, old)
    with pytest.raises(ref.LossError):
        ref.sequence_ratio(new, old)


# ---------------------------------------------------------------------------
# clip_higher
# ---------------------------------------------------------------------------


def test_clip_higher_golden_high_side():
    got = loss.clip_higher(1.5, 0.2, 0.28)
    exp = ref.clip_higher(1.5, 0.2, 0.28)
    _close(got, GOLDEN_CLIP_HIGH)
    _close(exp, GOLDEN_CLIP_HIGH)
    # Symmetric clip at 0.2 would yield 1.2.
    assert got != pytest.approx(1.2)


def test_clip_higher_golden_low_side():
    got = loss.clip_higher(0.5, 0.2, 0.28)
    exp = ref.clip_higher(0.5, 0.2, 0.28)
    _close(got, GOLDEN_CLIP_LOW)
    _close(exp, GOLDEN_CLIP_LOW)


def test_clip_higher_interior_unchanged():
    got = loss.clip_higher(1.0, 0.2, 0.28)
    _close(got, 1.0)
    _close(got, ref.clip_higher(1.0, 0.2, 0.28))


def test_clip_higher_at_boundaries():
    _close(loss.clip_higher(0.8, 0.2, 0.28), 0.8)
    _close(loss.clip_higher(1.28, 0.2, 0.28), 1.28)


@pytest.mark.parametrize("ratio", [math.nan, math.inf, -math.inf])
def test_clip_higher_non_finite_ratio_raises(ratio: float):
    with pytest.raises(loss.LossError):
        loss.clip_higher(ratio, 0.2, 0.28)
    with pytest.raises(ref.LossError):
        ref.clip_higher(ratio, 0.2, 0.28)


@pytest.mark.parametrize("low, high", [(-0.1, 0.28), (0.2, -0.01), (-0.1, -0.1)])
def test_clip_higher_negative_eps_raises(low: float, high: float):
    with pytest.raises(loss.LossError):
        loss.clip_higher(1.0, low, high)
    with pytest.raises(ref.LossError):
        ref.clip_higher(1.0, low, high)


# ---------------------------------------------------------------------------
# staleness
# ---------------------------------------------------------------------------


def test_staleness_zero_and_positive():
    assert loss.staleness(4, 4) == 0
    assert loss.staleness(4, 0) == 4
    assert loss.staleness(7, 3) == 4
    assert ref.staleness(7, 3) == 4
    assert isinstance(loss.staleness(1, 0), int)


def test_staleness_negative_raises():
    with pytest.raises(loss.LossError):
        loss.staleness(3, 4)
    with pytest.raises(ref.LossError):
        ref.staleness(3, 4)


# ---------------------------------------------------------------------------
# truncated_is
# ---------------------------------------------------------------------------


def test_truncated_is_k0_is_one_even_if_logps_differ():
    trainer = (0.0, math.log(2.0))
    rollout = (0.0, 0.0)
    got = loss.truncated_is(trainer, rollout, 0, 2.0)
    exp = ref.truncated_is(trainer, rollout, 0, 2.0)
    _close(got, 1.0)
    _close(exp, 1.0)


def test_truncated_is_k0_ignores_empty_and_mismatch():
    _close(loss.truncated_is((), (), 0, 1.0), 1.0)
    _close(loss.truncated_is((0.0,), (0.0, 1.0), 0, 1.0), 1.0)
    _close(ref.truncated_is((), (), 0, 1.0), 1.0)


def test_truncated_is_k_positive_min_ratio_and_clip():
    trainer = (0.0, math.log(2.0))
    rollout = (0.0, 0.0)
    # ratio = 2.0
    _close(loss.truncated_is(trainer, rollout, 1, 3.0), 2.0)
    _close(loss.truncated_is(trainer, rollout, 1, 1.0), 1.0)
    _close(loss.truncated_is(trainer, rollout, 4, 1.5), 1.5)
    _close(ref.truncated_is(trainer, rollout, 1, 3.0), 2.0)
    _close(ref.truncated_is(trainer, rollout, 1, 1.0), 1.0)


def test_truncated_is_tis_clip_one_clips_every_rho_above_one():
    trainer = (math.log(4.0),)
    rollout = (0.0,)
    got = loss.truncated_is(trainer, rollout, 2, 1.0)
    _close(got, 1.0)
    _close(got, ref.truncated_is(trainer, rollout, 2, 1.0))


def test_truncated_is_ratio_below_one_not_raised():
    trainer = (0.0,)
    rollout = (math.log(2.0),)
    got = loss.truncated_is(trainer, rollout, 1, 2.0)
    _close(got, 0.5)
    _close(got, ref.truncated_is(trainer, rollout, 1, 2.0))


def test_truncated_is_negative_k_raises():
    with pytest.raises(loss.LossError):
        loss.truncated_is((0.0,), (0.0,), -1, 1.0)
    with pytest.raises(ref.LossError):
        ref.truncated_is((0.0,), (0.0,), -1, 1.0)


def test_truncated_is_tis_clip_below_one_raises():
    with pytest.raises(loss.LossError):
        loss.truncated_is((0.0,), (0.0,), 1, 0.5)
    with pytest.raises(ref.LossError):
        ref.truncated_is((0.0,), (0.0,), 1, 0.5)


# ---------------------------------------------------------------------------
# overlong_soft_penalty
# ---------------------------------------------------------------------------


def test_overlong_soft_penalty_zero_when_at_or_below_cache():
    _close(loss.overlong_soft_penalty(3, 3, 1.0), 0.0)
    _close(loss.overlong_soft_penalty(2, 3, 1.0), 0.0)
    _close(ref.overlong_soft_penalty(3, 3, 1.0), 0.0)


def test_overlong_soft_penalty_interior_golden():
    got = loss.overlong_soft_penalty(5, 3, 0.1)
    exp = ref.overlong_soft_penalty(5, 3, 0.1)
    _close(got, GOLDEN_OVERLONG_INTERIOR)
    _close(exp, GOLDEN_OVERLONG_INTERIOR)


def test_overlong_soft_penalty_cache_zero():
    got = loss.overlong_soft_penalty(4, 0, 0.5)
    exp = ref.overlong_soft_penalty(4, 0, 0.5)
    _close(got, GOLDEN_OVERLONG_CACHE0)
    _close(exp, GOLDEN_OVERLONG_CACHE0)


@pytest.mark.parametrize(
    "n_tokens, cache, penalty",
    [(-1, 0, 0.0), (1, -1, 0.0), (1, 0, -0.1), (-1, -1, -1.0)],
)
def test_overlong_soft_penalty_negatives_raise(n_tokens: int, cache: int, penalty: float):
    with pytest.raises(loss.LossError):
        loss.overlong_soft_penalty(n_tokens, cache, penalty)
    with pytest.raises(ref.LossError):
        ref.overlong_soft_penalty(n_tokens, cache, penalty)


# ---------------------------------------------------------------------------
# gaussian_log_density
# ---------------------------------------------------------------------------


def test_gaussian_log_density_standard_normal_at_mean():
    got = loss.gaussian_log_density((0.0,), (0.0,), (1.0,))
    exp = ref.gaussian_log_density((0.0,), (0.0,), (1.0,))
    _close(got, GOLDEN_GAUSS_STANDARD)
    _close(exp, GOLDEN_GAUSS_STANDARD)
    assert isinstance(got, float)


def test_gaussian_log_density_independent_sum():
    got = loss.gaussian_log_density((0.0, 1.0), (0.0, 1.0), (1.0, 1.0))
    _close(got, 2.0 * GOLDEN_GAUSS_STANDARD)
    _close(got, ref.gaussian_log_density((0.0, 1.0), (0.0, 1.0), (1.0, 1.0)))


def test_gaussian_log_density_finite_difference_matches_analytic():
    z = (0.3, -0.2, 1.1)
    mu = (0.1, 0.0, 0.5)
    sigma = (0.5, 1.25, 0.8)
    h = 1e-5
    for i in range(len(z)):
        zp = list(z)
        zm = list(z)
        zp[i] += h
        zm[i] -= h
        num = (
            loss.gaussian_log_density(tuple(zp), mu, sigma)
            - loss.gaussian_log_density(tuple(zm), mu, sigma)
        ) / (2.0 * h)
        analytic = -(z[i] - mu[i]) / (sigma[i] ** 2)
        np.testing.assert_allclose(num, analytic, **TOL)
        ref_num = (
            ref.gaussian_log_density(tuple(zp), mu, sigma)
            - ref.gaussian_log_density(tuple(zm), mu, sigma)
        ) / (2.0 * h)
        np.testing.assert_allclose(ref_num, analytic, **TOL)


def test_gaussian_log_density_empty_raises():
    with pytest.raises(loss.LossError):
        loss.gaussian_log_density((), (), ())
    with pytest.raises(ref.LossError):
        ref.gaussian_log_density((), (0.0,), (1.0,))


def test_gaussian_log_density_length_mismatch_raises():
    with pytest.raises(loss.LossError):
        loss.gaussian_log_density((0.0, 1.0), (0.0,), (1.0,))
    with pytest.raises(loss.LossError):
        loss.gaussian_log_density((0.0,), (0.0, 1.0), (1.0,))
    with pytest.raises(ref.LossError):
        ref.gaussian_log_density((0.0,), (0.0,), (1.0, 1.0))


@pytest.mark.parametrize(
    "z, mu, sigma",
    [
        ((math.nan,), (0.0,), (1.0,)),
        ((0.0,), (math.inf,), (1.0,)),
        ((0.0,), (0.0,), (math.nan,)),
    ],
)
def test_gaussian_log_density_non_finite_raises(z, mu, sigma):
    with pytest.raises(loss.LossError):
        loss.gaussian_log_density(z, mu, sigma)
    with pytest.raises(ref.LossError):
        ref.gaussian_log_density(z, mu, sigma)


@pytest.mark.parametrize("sigma", [(0.0,), (-1.0,), (-0.01, 1.0)])
def test_gaussian_log_density_non_positive_sigma_raises(sigma: tuple):
    z = tuple(0.0 for _ in sigma)
    mu = tuple(0.0 for _ in sigma)
    with pytest.raises(loss.LossError):
        loss.gaussian_log_density(z, mu, sigma)
    with pytest.raises(ref.LossError):
        ref.gaussian_log_density(z, mu, sigma)


# ---------------------------------------------------------------------------
# latent_ratio
# ---------------------------------------------------------------------------


def test_latent_ratio_both_empty_is_one():
    got = loss.latent_ratio((), ())
    exp = ref.latent_ratio((), ())
    _close(got, 1.0)
    _close(exp, 1.0)


def test_latent_ratio_matches_sequence_ratio():
    new = (0.0, math.log(2.0))
    old = (0.0, 0.0)
    got = loss.latent_ratio(new, old)
    _close(got, 2.0)
    _close(got, ref.latent_ratio(new, old))
    _close(got, loss.sequence_ratio(new, old))


def test_latent_ratio_one_empty_raises():
    with pytest.raises(loss.LossError):
        loss.latent_ratio((), (0.0,))
    with pytest.raises(loss.LossError):
        loss.latent_ratio((0.0,), ())
    with pytest.raises(ref.LossError):
        ref.latent_ratio((0.0,), ())


def test_latent_ratio_mismatch_and_non_finite_raise():
    with pytest.raises(loss.LossError):
        loss.latent_ratio((0.0, 1.0), (0.0,))
    with pytest.raises(loss.LossError):
        loss.latent_ratio((math.nan,), (0.0,))
    with pytest.raises(ref.LossError):
        ref.latent_ratio((math.inf,), (0.0,))


# ---------------------------------------------------------------------------
# routing_table
# ---------------------------------------------------------------------------


def test_routing_table_dense_and_expert_zero():
    traces = (
        _trace(loss, 0, 0, (0, 1)),
        _trace(loss, 0, 1, (2, 3)),
        _trace(loss, 1, 0, (0, 4)),
        _trace(loss, 1, 1, (5, 0)),
    )
    got = loss.routing_table(traces, 2, 2, 2)
    exp = ref.routing_table(tuple(_to_ref_trace(t) for t in traces), 2, 2, 2)
    assert got == exp
    assert got == (((0, 1), (2, 3)), ((0, 4), (5, 0)))
    assert got[0][0][0] == 0  # expert id 0 is legal


def test_routing_table_single_cell_top_k_one():
    traces = (_trace(loss, 0, 0, (0,)),)
    got = loss.routing_table(traces, 1, 1, 1)
    assert got == (((0,),),)
    assert got == ref.routing_table((_to_ref_trace(traces[0]),), 1, 1, 1)


@pytest.mark.parametrize("n_tokens, n_layers, top_k", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)])
def test_routing_table_non_positive_dims_raise(n_tokens: int, n_layers: int, top_k: int):
    traces = (_trace(loss, 0, 0, (0,)),)
    with pytest.raises(loss.LossError):
        loss.routing_table(traces, n_tokens, n_layers, top_k)
    with pytest.raises(ref.LossError):
        ref.routing_table((_to_ref_trace(traces[0]),), n_tokens, n_layers, top_k)


def test_routing_table_duplicate_cell_raises():
    traces = (
        _trace(loss, 0, 0, (1,)),
        _trace(loss, 0, 0, (2,)),
    )
    with pytest.raises(loss.LossError):
        loss.routing_table(traces, 1, 1, 1)
    with pytest.raises(ref.LossError):
        ref.routing_table(tuple(_to_ref_trace(t) for t in traces), 1, 1, 1)


def test_routing_table_missing_cell_raises():
    traces = (_trace(loss, 0, 0, (1,)),)
    with pytest.raises(loss.LossError):
        loss.routing_table(traces, 2, 1, 1)
    with pytest.raises(ref.LossError):
        ref.routing_table((_to_ref_trace(traces[0]),), 2, 1, 1)


def test_routing_table_wrong_expert_ids_length_raises():
    traces = (_trace(loss, 0, 0, (1, 2)),)
    with pytest.raises(loss.LossError):
        loss.routing_table(traces, 1, 1, 1)
    with pytest.raises(ref.LossError):
        ref.routing_table((_to_ref_trace(traces[0]),), 1, 1, 1)


def test_routing_table_negative_expert_id_raises():
    traces = (_trace(loss, 0, 0, (-1,)),)
    with pytest.raises(loss.LossError):
        loss.routing_table(traces, 1, 1, 1)
    with pytest.raises(ref.LossError):
        ref.routing_table((_to_ref_trace(traces[0]),), 1, 1, 1)


# ---------------------------------------------------------------------------
# gspo_dapo_loss
# ---------------------------------------------------------------------------


def test_gspo_dapo_loss_all_equal_rewards_zeros():
    group = _group(
        loss,
        (
            _sample(loss, reward=2.0, n_tokens=1),
            _sample(loss, reward=2.0, n_tokens=1),
        ),
    )
    cfg = loss.make_config()
    got = loss.gspo_dapo_loss(group, cfg, 0)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 0)
    _assert_breakdown(got, exp)
    assert got.n_kept == 0
    _close(got.total, 0.0)
    _close(got.pg, 0.0)
    _close(got.tis, 0.0)
    _close(got.overlong, 0.0)
    _close(got.latent, 0.0)


def test_gspo_dapo_loss_on_policy_discrete_golden():
    group = _golden_pair(loss)
    cfg = loss.make_config()
    got = loss.gspo_dapo_loss(group, cfg, 0)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 0)
    _assert_breakdown(got, exp)
    _close(got.total, GOLDEN_ON_POLICY_TOTAL)
    _close(got.pg, GOLDEN_ON_POLICY_PG)
    _close(got.latent, 0.0)
    assert got.n_kept == 2
    _close(got.total, got.pg + got.latent + got.overlong)


def test_gspo_dapo_loss_overlong_added():
    group = _golden_pair(loss)
    cfg = loss.make_config(overlong_penalty=1.0)
    got = loss.gspo_dapo_loss(group, cfg, 0)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 0)
    _assert_breakdown(got, exp)
    _close(got.total, GOLDEN_OVERLONG_TOTAL)
    _close(got.overlong, 4.0)


def test_gspo_dapo_loss_k2_tis_clip_two():
    group = _golden_pair(loss)
    cfg = loss.make_config(tis_clip=2.0)
    got = loss.gspo_dapo_loss(group, cfg, 2)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 2)
    _assert_breakdown(got, exp)
    _close(got.total, GOLDEN_K2_TIS2_TOTAL)
    _close(got.pg, GOLDEN_K2_TIS2_PG)
    _close(got.tis, GOLDEN_K2_TIS2_TIS)


def test_gspo_dapo_loss_tis_clip_one_clips_rho_above_one():
    group = _golden_pair(loss)
    cfg = loss.make_config(tis_clip=1.0)
    got = loss.gspo_dapo_loss(group, cfg, 2)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 2)
    _assert_breakdown(got, exp)
    _close(got.total, GOLDEN_K2_TIS1_TOTAL)


def test_gspo_dapo_loss_latent_clipped_separately():
    group = _golden_pair(loss, latents=True)
    cfg = loss.make_config()
    got = loss.gspo_dapo_loss(group, cfg, 0)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 0)
    _assert_breakdown(got, exp)
    _close(got.total, GOLDEN_LATENT_TOTAL)
    _close(got.pg, GOLDEN_LATENT_PG)
    _close(got.latent, GOLDEN_LATENT_LATENT)


def test_gspo_dapo_loss_discrete_only_empty_latent_tuples():
    group = _golden_pair(loss, latents=False)
    cfg = loss.make_config()
    got = loss.gspo_dapo_loss(group, cfg, 0)
    _close(got.latent, 0.0)
    for sample in group.samples:
        assert sample.latent_logp_trainer == ()
        assert sample.latent_logp_rollout == ()


def test_gspo_dapo_loss_divides_by_loss_normalizer():
    group = _golden_pair(loss)
    cfg = loss.make_config(loss_normalizer=2.0)
    got = loss.gspo_dapo_loss(group, cfg, 0)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 0)
    _assert_breakdown(got, exp)
    _close(got.total, GOLDEN_Z2_TOTAL)
    _close(got.pg, GOLDEN_Z2_TOTAL)


def test_gspo_dapo_loss_group_size_16():
    group = _g16(loss)
    assert len(group.samples) == loss.GROUP_SIZE
    cfg = loss.make_config()
    got = loss.gspo_dapo_loss(group, cfg, 0)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 0)
    _assert_breakdown(got, exp)
    _close(got.total, GOLDEN_G16_TOTAL)
    assert got.n_kept == loss.GROUP_SIZE


def test_gspo_dapo_loss_routing_does_not_change_total():
    traces = (_trace(loss, 0, 0, (0,)),)
    with_r = _group(
        loss,
        (
            _sample(loss, reward=1.0, routing=traces),
            _sample(loss, reward=2.0, routing=traces),
        ),
    )
    without = _group(
        loss,
        (
            _sample(loss, reward=1.0, routing=()),
            _sample(loss, reward=2.0, routing=()),
        ),
    )
    cfg = loss.make_config()
    got_r = loss.gspo_dapo_loss(with_r, cfg, 0)
    got = loss.gspo_dapo_loss(without, cfg, 0)
    _close(got_r.total, got.total)
    _assert_breakdown(got, ref.gspo_dapo_loss(_to_ref_group(without), _to_ref_config(cfg), 0))


def test_gspo_dapo_loss_does_not_apply_reference_kl():
    group = _golden_pair(loss)
    cfg = loss.LossConfig(kl_coeff=0.0)
    # make_config rejects kl_coeff != 0; a raw dataclass still must ignore it.
    sneaky = loss.LossConfig(kl_coeff=1.0)
    got = loss.gspo_dapo_loss(group, cfg, 0)
    got_s = loss.gspo_dapo_loss(group, sneaky, 0)
    _close(got.total, got_s.total)
    _close(got.total, GOLDEN_ON_POLICY_TOTAL)


def test_gspo_dapo_loss_empty_group_raises():
    group = loss.Group(prompt_id="p", samples=())
    cfg = loss.LossConfig()
    with pytest.raises(loss.LossError):
        loss.gspo_dapo_loss(group, cfg, 0)
    with pytest.raises(ref.LossError):
        ref.gspo_dapo_loss(ref.Group(prompt_id="p", samples=()), ref.LossConfig(), 0)


def test_gspo_dapo_loss_empty_prompt_id_raises():
    group = _group(loss, (_sample(loss, reward=1.0), _sample(loss, reward=2.0)), prompt_id="")
    cfg = loss.LossConfig()
    with pytest.raises(loss.LossError):
        loss.gspo_dapo_loss(group, cfg, 0)
    with pytest.raises(ref.LossError):
        ref.gspo_dapo_loss(_to_ref_group(group), ref.LossConfig(), 0)


def test_gspo_dapo_loss_empty_token_logps_raises():
    group = _group(
        loss,
        (
            _sample(loss, token_logp_trainer=(), token_logp_rollout=(), reward=1.0),
            _sample(loss, reward=2.0),
        ),
    )
    with pytest.raises(loss.LossError):
        loss.gspo_dapo_loss(group, loss.LossConfig(), 0)


def test_gspo_dapo_loss_mismatched_token_lengths_raises():
    group = _group(
        loss,
        (
            _sample(
                loss,
                token_logp_trainer=(0.0, 1.0),
                token_logp_rollout=(0.0,),
                reward=1.0,
            ),
            _sample(loss, reward=2.0),
        ),
    )
    with pytest.raises(loss.LossError):
        loss.gspo_dapo_loss(group, loss.LossConfig(), 0)


def test_gspo_dapo_loss_mismatched_latent_lengths_raises():
    group = _group(
        loss,
        (
            _sample(
                loss,
                latent_logp_trainer=(0.0, 1.0),
                latent_logp_rollout=(0.0,),
                reward=1.0,
            ),
            _sample(loss, reward=2.0),
        ),
    )
    with pytest.raises(loss.LossError):
        loss.gspo_dapo_loss(group, loss.LossConfig(), 0)


def test_gspo_dapo_loss_staleness_above_max_raises():
    group = _group(
        loss,
        (
            _sample(loss, reward=1.0, policy_version=0),
            _sample(loss, reward=2.0, policy_version=0),
        ),
    )
    cfg = loss.LossConfig(max_staleness=4)
    with pytest.raises(loss.LossError):
        loss.gspo_dapo_loss(group, cfg, 5)
    with pytest.raises(ref.LossError):
        ref.gspo_dapo_loss(_to_ref_group(group), ref.LossConfig(max_staleness=4), 5)


def test_gspo_dapo_loss_staleness_equal_max_ok():
    group = _golden_pair(loss)
    cfg = loss.make_config(max_staleness=4, tis_clip=2.0)
    got = loss.gspo_dapo_loss(group, cfg, 4)
    exp = ref.gspo_dapo_loss(_to_ref_group(group), _to_ref_config(cfg), 4)
    _assert_breakdown(got, exp)
    assert got.n_kept == 2


def test_gspo_dapo_loss_non_finite_reward_raises():
    group = _group(
        loss,
        (
            _sample(loss, reward=math.nan),
            _sample(loss, reward=1.0),
        ),
    )
    with pytest.raises(loss.LossError):
        loss.gspo_dapo_loss(group, loss.LossConfig(), 0)


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


def _small_group(mod, n_samples: int, n_tokens: int, k: int, with_lat: bool, with_route: bool):
    samples = []
    for i in range(n_samples):
        tok_tr = tuple(-0.1 * (i + 1) * (t + 1) for t in range(n_tokens))
        tok_ro = tuple(-0.2 * (i + 1) * (t + 1) for t in range(n_tokens))
        if with_lat:
            lat_tr = tuple(-0.05 * (i + 1) * (t + 1) for t in range(min(n_tokens, 2)))
            lat_ro = tuple(-0.03 * (i + 1) * (t + 1) for t in range(min(n_tokens, 2)))
        else:
            lat_tr = ()
            lat_ro = ()
        if with_route:
            routing = tuple(_trace(mod, t, 0, (0,)) for t in range(n_tokens))
        else:
            routing = ()
        samples.append(
            _sample(
                mod,
                token_logp_trainer=tok_tr,
                token_logp_rollout=tok_ro,
                latent_logp_trainer=lat_tr,
                latent_logp_rollout=lat_ro,
                reward=1.0 + 0.5 * i,
                n_tokens=n_tokens,
                policy_version=0,
                routing=routing,
            )
        )
    return _group(mod, samples, prompt_id=f"g-{n_samples}-{n_tokens}-{k}"), k


def test_property_gspo_dapo_loss_matches_reference_small_groups():
    for n_samples in (2, 3, 4):
        for n_tokens in (1, 2, 3, 4):
            for k in (0, 1, 2, 3, 4):
                for with_lat in (False, True):
                    for with_route in (False, True):
                        group, trainer_version = _small_group(
                            loss, n_samples, n_tokens, k, with_lat, with_route
                        )
                        cfg = loss.make_config(tis_clip=2.0, overlong_penalty=0.01)
                        got = loss.gspo_dapo_loss(group, cfg, trainer_version)
                        exp = ref.gspo_dapo_loss(
                            _to_ref_group(group), _to_ref_config(cfg), trainer_version
                        )
                        _assert_breakdown(got, exp)
                        _close(got.total, got.pg + got.latent + got.overlong)
                        assert got.n_kept == n_samples
                        if not with_lat:
                            _close(got.latent, 0.0)


def test_property_helpers_match_reference_grid():
    ratios = (0.5, 0.8, 1.0, 1.28, 1.5, 2.0)
    for ratio in ratios:
        _close(loss.clip_higher(ratio, 0.2, 0.28), ref.clip_higher(ratio, 0.2, 0.28))
    for k in range(0, 5):
        trainer = (0.0, 0.25 * k)
        rollout = (0.0, 0.0)
        _close(
            loss.truncated_is(trainer, rollout, k, 2.0),
            ref.truncated_is(trainer, rollout, k, 2.0),
        )
    for n in range(0, 8):
        for cache in range(0, 5):
            _close(
                loss.overlong_soft_penalty(n, cache, 0.25),
                ref.overlong_soft_penalty(n, cache, 0.25),
            )


def test_property_mean_center_and_sequence_ratio_grid():
    for n in range(1, 6):
        rewards = tuple(float(i) for i in range(n))
        _close(loss.mean_center(rewards), ref.mean_center(rewards))
    for length in range(1, 5):
        new = tuple(0.1 * i for i in range(length))
        old = tuple(-0.05 * i for i in range(length))
        _close(loss.sequence_ratio(new, old), ref.sequence_ratio(new, old))
        _close(loss.latent_ratio(new, old), ref.latent_ratio(new, old))
    _close(loss.latent_ratio((), ()), ref.latent_ratio((), ()))
