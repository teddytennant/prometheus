"""I5 oracle: production ``model.latent`` vs the independent NumPy reference.

Groups
------
1. constants / Default config / LatentError subclass / frozen dataclasses
   (may pass against the stub; values live on the interface).
2. validate_latent_config errors in the locked order.
3. ponder_distribution: pin last, clip, sum-1, empty/rank errors; golden.
4. geometric_prior + halt_from_logits expected_depth.
5. halt_loss clip of teacher_steps; halt_kl vs prior.
6. thought_decode_ce: mean, all-zero mask, OOB id, length mismatch.
7. answer_ce_at_depths.
8. stage_b_loss: four terms, weights, K=0 edge, config invalid first.
9. jacobi_sweeps: identity, contraction, shape change, truncated_sweeps.
10. clamp_sigma + noisy_latent: z formula, log_density finite-diff, empty/mismatch.
11. Property: lockstep prod vs ref on random small vectors at 1e-5.

Every test that calls a stubbed function must fail until production is filled in.
No GPU. No JAX. No ``rl.loss``. Drive both production and the reference on
every behavior assertion.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import model.latent as latent
from tests.reference import latent as ref

TOL = dict(rtol=1e-5, atol=1e-5)

# Short-vector golden: λ = [0.3, 0.4, 0.5, 0.6], last pinned to 1.0.
# p = [0.3, 0.4*0.7, 0.5*0.7*0.6, 1.0*0.7*0.6*0.5] = [0.3, 0.28, 0.21, 0.21]
GOLDEN_LAMBDAS = (0.3, 0.4, 0.5, 0.6)
GOLDEN_PONDER = (0.3, 0.28, 0.21, 0.21)
GOLDEN_DEPTH = 1.33  # 0*0.3 + 1*0.28 + 2*0.21 + 3*0.21


def _close(got: object, exp: object) -> None:
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), exp, **TOL)


def _prod_cfg(**kwargs: object) -> latent.LatentConfig:
    return latent.LatentConfig(**kwargs)  # type: ignore[arg-type]


def _ref_cfg(**kwargs: object) -> ref.LatentConfig:
    return ref.LatentConfig(**kwargs)  # type: ignore[arg-type]


def _assert_halt(got: object, exp: object) -> None:
    _close(got.lambdas, exp.lambdas)  # type: ignore[attr-defined]
    _close(got.p, exp.p)  # type: ignore[attr-defined]
    _close(got.expected_depth, exp.expected_depth)  # type: ignore[attr-defined]
    assert np.ndim(got.expected_depth) == 0  # type: ignore[attr-defined]


def _assert_stage(got: object, exp: object) -> None:
    for name in ("l_task", "l_traj", "l_halt", "l_kl", "expected_depth", "total"):
        _close(getattr(got, name), getattr(exp, name))
        assert np.ndim(getattr(got, name)) == 0


def _assert_noisy(got: object, exp: object) -> None:
    _close(got.z, exp.z)  # type: ignore[attr-defined]
    _close(got.log_density, exp.log_density)  # type: ignore[attr-defined]
    _close(got.sigma, exp.sigma)  # type: ignore[attr-defined]
    _close(got.mu, exp.mu)  # type: ignore[attr-defined]
    _close(got.eps, exp.eps)  # type: ignore[attr-defined]
    assert np.ndim(got.log_density) == 0  # type: ignore[attr-defined]


def _both_raise_latent(prod_fn, ref_fn) -> None:
    with pytest.raises(latent.LatentError):
        prod_fn()
    with pytest.raises(ref.LatentError):
        ref_fn()


# ---------------------------------------------------------------------------
# 1. constants / Default config / LatentError subclass / frozen dataclasses
#    These may pass against the stub.
# ---------------------------------------------------------------------------


def test_chunk_and_compression_constants():
    assert latent.CHUNK_MIN_THOUGHTS == 4
    assert latent.CHUNK_MAX_THOUGHTS == 64
    assert latent.COMPRESSION_TOKENS_PER_THOUGHT == 8


def test_jacobi_and_truncation_constants():
    assert latent.DEFAULT_JACOBI_SWEEPS == 4
    assert latent.TRUNCATED_SWEEPS == 2
    assert latent.TRUNCATED_RECURRENCE == 4


def test_halt_and_sigma_constants():
    assert latent.DEFAULT_MAX_THOUGHTS == 16
    assert latent.DEFAULT_ALPHA_TRAJ == 1.0
    assert latent.DEFAULT_GAMMA_HALT == 1.0
    assert latent.DEFAULT_BETA_KL == 0.01
    assert latent.DEFAULT_LAMBDA_PRIOR == 0.2
    assert latent.HALT_EPS == 1e-6
    assert latent.SIGMA_MIN == 1e-4
    assert latent.SIGMA_MAX == 0.1


def test_default_latent_config_equals_module_constants():
    cfg = latent.LatentConfig()
    assert cfg.max_thoughts == latent.DEFAULT_MAX_THOUGHTS
    assert cfg.chunk_min == latent.CHUNK_MIN_THOUGHTS
    assert cfg.chunk_max == latent.CHUNK_MAX_THOUGHTS
    assert cfg.alpha_traj == latent.DEFAULT_ALPHA_TRAJ
    assert cfg.gamma_halt == latent.DEFAULT_GAMMA_HALT
    assert cfg.beta_kl == latent.DEFAULT_BETA_KL
    assert cfg.lambda_prior == latent.DEFAULT_LAMBDA_PRIOR
    assert cfg.jacobi_sweeps == latent.DEFAULT_JACOBI_SWEEPS
    assert cfg.truncated_sweeps == latent.TRUNCATED_SWEEPS
    assert cfg.sigma_min == latent.SIGMA_MIN
    assert cfg.sigma_max == latent.SIGMA_MAX


def test_latent_error_is_value_error():
    assert issubclass(latent.LatentError, ValueError)
    assert issubclass(ref.LatentError, ValueError)


def test_dataclasses_are_frozen():
    cfg = latent.LatentConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.max_thoughts = 3  # type: ignore[misc]
    halt = latent.HaltOutput(
        lambdas=np.array([1.0]), p=np.array([1.0]), expected_depth=0.0
    )
    with pytest.raises(FrozenInstanceError):
        halt.expected_depth = 1.0  # type: ignore[misc]
    stage = latent.StageBLoss(
        l_task=0.0, l_traj=0.0, l_halt=0.0, l_kl=0.0, expected_depth=0.0, total=0.0
    )
    with pytest.raises(FrozenInstanceError):
        stage.total = 1.0  # type: ignore[misc]
    noisy = latent.NoisyLatent(
        mu=np.array([0.0]),
        sigma=np.array([0.1]),
        eps=np.array([0.0]),
        z=np.array([0.0]),
        log_density=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        noisy.log_density = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. validate_latent_config errors in the locked order
# ---------------------------------------------------------------------------


def test_validate_default_config_ok():
    latent.validate_latent_config(latent.LatentConfig())
    ref.validate_latent_config(ref.LatentConfig())


def test_validate_max_thoughts_zero_ok():
    latent.validate_latent_config(_prod_cfg(max_thoughts=0))
    ref.validate_latent_config(_ref_cfg(max_thoughts=0))


def test_validate_max_thoughts_negative():
    # Check 1. Other fields left at defaults (valid).
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(max_thoughts=-1)),
        lambda: ref.validate_latent_config(_ref_cfg(max_thoughts=-1)),
    )


def test_validate_chunk_min_below_floor():
    # Check 2a. chunk_min < CHUNK_MIN_THOUGHTS (4).
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(chunk_min=3)),
        lambda: ref.validate_latent_config(_ref_cfg(chunk_min=3)),
    )


def test_validate_chunk_max_above_ceiling():
    # Check 2b. chunk_max > CHUNK_MAX_THOUGHTS (64).
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(chunk_max=65)),
        lambda: ref.validate_latent_config(_ref_cfg(chunk_max=65)),
    )


def test_validate_chunk_min_greater_than_max():
    # Check 2c. chunk_min > chunk_max, both inside the legal window.
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(chunk_min=16, chunk_max=8)),
        lambda: ref.validate_latent_config(_ref_cfg(chunk_min=16, chunk_max=8)),
    )


def test_validate_alpha_traj_negative_then_nonfinite():
    # Check 3, alpha first.
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(alpha_traj=-1.0)),
        lambda: ref.validate_latent_config(_ref_cfg(alpha_traj=-1.0)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(alpha_traj=math.nan)),
        lambda: ref.validate_latent_config(_ref_cfg(alpha_traj=math.nan)),
    )


def test_validate_gamma_halt_negative_then_nonfinite():
    # Check 3, gamma after alpha (alpha left valid).
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(gamma_halt=-0.1)),
        lambda: ref.validate_latent_config(_ref_cfg(gamma_halt=-0.1)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(gamma_halt=math.inf)),
        lambda: ref.validate_latent_config(_ref_cfg(gamma_halt=math.inf)),
    )


def test_validate_beta_kl_negative_then_nonfinite():
    # Check 3, beta after alpha and gamma.
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(beta_kl=-1e-9)),
        lambda: ref.validate_latent_config(_ref_cfg(beta_kl=-1e-9)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(beta_kl=math.nan)),
        lambda: ref.validate_latent_config(_ref_cfg(beta_kl=math.nan)),
    )


def test_validate_zero_loss_weights_ok():
    latent.validate_latent_config(_prod_cfg(alpha_traj=0.0, gamma_halt=0.0, beta_kl=0.0))
    ref.validate_latent_config(_ref_cfg(alpha_traj=0.0, gamma_halt=0.0, beta_kl=0.0))


def test_validate_lambda_prior_not_exclusive_open_unit_interval():
    # Check 4. 0.0 and 1.0 fail; values outside fail.
    for bad in (0.0, 1.0, -0.1, 1.1, math.nan, math.inf):
        _both_raise_latent(
            lambda b=bad: latent.validate_latent_config(_prod_cfg(lambda_prior=b)),
            lambda b=bad: ref.validate_latent_config(_ref_cfg(lambda_prior=b)),
        )


def test_validate_jacobi_sweeps_below_one():
    # Check 5.
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(jacobi_sweeps=0)),
        lambda: ref.validate_latent_config(_ref_cfg(jacobi_sweeps=0)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(jacobi_sweeps=-3)),
        lambda: ref.validate_latent_config(_ref_cfg(jacobi_sweeps=-3)),
    )


def test_validate_truncated_sweeps_range():
    # Check 6. truncated_sweeps < 1 or > jacobi_sweeps.
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(truncated_sweeps=0)),
        lambda: ref.validate_latent_config(_ref_cfg(truncated_sweeps=0)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(
            _prod_cfg(jacobi_sweeps=4, truncated_sweeps=5)
        ),
        lambda: ref.validate_latent_config(_ref_cfg(jacobi_sweeps=4, truncated_sweeps=5)),
    )


def test_validate_sigma_bounds():
    # Check 7. sigma_min <= 0, sigma_max < sigma_min, non-finite.
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(sigma_min=0.0)),
        lambda: ref.validate_latent_config(_ref_cfg(sigma_min=0.0)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(sigma_min=-1e-3)),
        lambda: ref.validate_latent_config(_ref_cfg(sigma_min=-1e-3)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(
            _prod_cfg(sigma_min=0.05, sigma_max=0.01)
        ),
        lambda: ref.validate_latent_config(_ref_cfg(sigma_min=0.05, sigma_max=0.01)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(sigma_min=math.nan)),
        lambda: ref.validate_latent_config(_ref_cfg(sigma_min=math.nan)),
    )
    _both_raise_latent(
        lambda: latent.validate_latent_config(_prod_cfg(sigma_max=math.inf)),
        lambda: ref.validate_latent_config(_ref_cfg(sigma_max=math.inf)),
    )


def test_validate_equal_sigma_bounds_ok():
    latent.validate_latent_config(_prod_cfg(sigma_min=0.05, sigma_max=0.05))
    ref.validate_latent_config(_ref_cfg(sigma_min=0.05, sigma_max=0.05))


# ---------------------------------------------------------------------------
# 3. ponder_distribution
# ---------------------------------------------------------------------------


def test_ponder_golden_short_vector():
    lam = np.array(GOLDEN_LAMBDAS, dtype=np.float64)
    got = latent.ponder_distribution(lam)
    exp = ref.ponder_distribution(lam)
    _close(got, GOLDEN_PONDER)
    _close(exp, GOLDEN_PONDER)
    _close(got, exp)
    assert np.asarray(got).ndim == 1
    assert np.asarray(got).shape == (4,)
    assert np.issubdtype(np.asarray(got).dtype, np.floating)


def test_ponder_pins_last_slot_and_sums_to_one():
    lam = np.array([0.2, 0.2, 0.2], dtype=np.float64)
    got = np.asarray(latent.ponder_distribution(lam), dtype=np.float64)
    exp = np.asarray(ref.ponder_distribution(lam), dtype=np.float64)
    # clip no-op, pin last → [0.2, 0.2, 1.0]; p = [0.2, 0.16, 0.64]
    _close(got, (0.2, 0.16, 0.64))
    _close(exp, (0.2, 0.16, 0.64))
    assert abs(float(got.sum()) - 1.0) < 1e-6
    assert abs(float(exp.sum()) - 1.0) < 1e-6


def test_ponder_clips_before_pin():
    lam = np.array([0.0, 1.0, 2.0, -5.0], dtype=np.float64)
    got = np.asarray(latent.ponder_distribution(lam), dtype=np.float64)
    exp = np.asarray(ref.ponder_distribution(lam), dtype=np.float64)
    eps = float(latent.HALT_EPS)
    clipped = np.array([eps, 1.0 - eps, 1.0 - eps, eps], dtype=np.float64)
    clipped[-1] = 1.0
    survival = 1.0
    hand = np.empty(4, dtype=np.float64)
    for m in range(4):
        hand[m] = clipped[m] * survival
        survival *= 1.0 - clipped[m]
    _close(got, hand)
    _close(exp, hand)
    assert abs(float(got.sum()) - 1.0) < 1e-6


def test_ponder_single_slot_is_one():
    got = latent.ponder_distribution(np.array([0.3], dtype=np.float64))
    exp = ref.ponder_distribution(np.array([0.3], dtype=np.float64))
    _close(got, (1.0,))
    _close(exp, (1.0,))


def test_ponder_empty_and_rank_errors():
    _both_raise_latent(
        lambda: latent.ponder_distribution(np.array([], dtype=np.float64)),
        lambda: ref.ponder_distribution(np.array([], dtype=np.float64)),
    )
    _both_raise_latent(
        lambda: latent.ponder_distribution(np.array([[0.5, 0.5]], dtype=np.float64)),
        lambda: ref.ponder_distribution(np.array([[0.5, 0.5]], dtype=np.float64)),
    )
    _both_raise_latent(
        lambda: latent.ponder_distribution(np.array(0.5, dtype=np.float64)),
        lambda: ref.ponder_distribution(np.array(0.5, dtype=np.float64)),
    )
    _both_raise_latent(
        lambda: latent.ponder_distribution(np.array([0.5, math.nan], dtype=np.float64)),
        lambda: ref.ponder_distribution(np.array([0.5, math.nan], dtype=np.float64)),
    )
    _both_raise_latent(
        lambda: latent.ponder_distribution(np.array([0.5, math.inf], dtype=np.float64)),
        lambda: ref.ponder_distribution(np.array([0.5, math.inf], dtype=np.float64)),
    )


# ---------------------------------------------------------------------------
# 4. geometric_prior + halt_from_logits expected_depth
# ---------------------------------------------------------------------------


def test_geometric_prior_formula_and_renormalize():
    n = 4
    lam_p = 0.2
    got = np.asarray(latent.geometric_prior(n, lam_p), dtype=np.float64)
    exp = np.asarray(ref.geometric_prior(n, lam_p), dtype=np.float64)
    raw = np.array([lam_p * (1.0 - lam_p) ** m for m in range(n)], dtype=np.float64)
    hand = raw / raw.sum()
    _close(got, hand)
    _close(exp, hand)
    _close(got, exp)
    assert abs(float(got.sum()) - 1.0) < 1e-6


def test_geometric_prior_n_one():
    got = latent.geometric_prior(1, 0.5)
    exp = ref.geometric_prior(1, 0.5)
    _close(got, (1.0,))
    _close(exp, (1.0,))


def test_geometric_prior_errors():
    _both_raise_latent(
        lambda: latent.geometric_prior(0, 0.2),
        lambda: ref.geometric_prior(0, 0.2),
    )
    _both_raise_latent(
        lambda: latent.geometric_prior(-1, 0.2),
        lambda: ref.geometric_prior(-1, 0.2),
    )
    for bad in (0.0, 1.0, -0.2, 2.0):
        _both_raise_latent(
            lambda b=bad: latent.geometric_prior(3, b),
            lambda b=bad: ref.geometric_prior(3, b),
        )


def test_halt_from_logits_expected_depth_golden():
    logits = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    got = latent.halt_from_logits(logits)
    exp = ref.halt_from_logits(logits)
    # sigmoid(0)=0.5, pin last → [0.5, 0.5, 1.0]; p=[0.5, 0.25, 0.25]; E=0.75
    _close(got.lambdas, (0.5, 0.5, 1.0))
    _close(got.p, (0.5, 0.25, 0.25))
    _close(got.expected_depth, 0.75)
    _assert_halt(got, exp)
    assert np.asarray(got.lambdas)[-1] == 1.0
    assert abs(float(np.asarray(got.p).sum()) - 1.0) < 1e-6


def test_halt_from_logits_golden_lambdas_depth():
    # Invert sigmoid of GOLDEN_LAMBDAS so halt_from_logits recovers the golden p.
    # last is pinned regardless of the inverted logit.
    raw = np.array(GOLDEN_LAMBDAS, dtype=np.float64)
    logits = np.log(raw / (1.0 - raw))
    got = latent.halt_from_logits(logits)
    exp = ref.halt_from_logits(logits)
    _close(got.p, GOLDEN_PONDER)
    _close(got.expected_depth, GOLDEN_DEPTH)
    _assert_halt(got, exp)


def test_halt_from_logits_errors():
    _both_raise_latent(
        lambda: latent.halt_from_logits(np.array([], dtype=np.float64)),
        lambda: ref.halt_from_logits(np.array([], dtype=np.float64)),
    )
    _both_raise_latent(
        lambda: latent.halt_from_logits(np.array([[0.0]], dtype=np.float64)),
        lambda: ref.halt_from_logits(np.array([[0.0]], dtype=np.float64)),
    )
    _both_raise_latent(
        lambda: latent.halt_from_logits(np.array([0.0, math.nan], dtype=np.float64)),
        lambda: ref.halt_from_logits(np.array([0.0, math.nan], dtype=np.float64)),
    )


# ---------------------------------------------------------------------------
# 5. halt_loss + halt_kl
# ---------------------------------------------------------------------------


def test_halt_loss_clip_teacher_steps():
    p = np.array(GOLDEN_PONDER, dtype=np.float64)
    eps = float(latent.HALT_EPS)
    cases = {
        -5: 0,
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        99: 3,
    }
    for teacher, k in cases.items():
        got = latent.halt_loss(p, teacher)
        exp = ref.halt_loss(p, teacher)
        hand = -math.log(float(p[k]) + eps)
        _close(got, hand)
        _close(exp, hand)
        _close(got, exp)


def test_halt_kl_zero_when_p_matches_prior():
    g = np.asarray(ref.geometric_prior(5, 0.2), dtype=np.float64)
    got = latent.halt_kl(g, 0.2)
    exp = ref.halt_kl(g, 0.2)
    _close(got, 0.0)
    _close(exp, 0.0)


def test_halt_kl_positive_away_from_prior():
    p = np.array(GOLDEN_PONDER, dtype=np.float64)
    got = latent.halt_kl(p, 0.2)
    exp = ref.halt_kl(p, 0.2)
    g = np.asarray(ref.geometric_prior(4, 0.2), dtype=np.float64)
    eps = float(latent.HALT_EPS)
    hand = float(np.sum(p * (np.log(p + eps) - np.log(g + eps))))
    _close(got, hand)
    _close(exp, hand)
    assert float(got) > 0.0


def test_halt_loss_and_kl_errors():
    _both_raise_latent(
        lambda: latent.halt_loss(np.array([], dtype=np.float64), 0),
        lambda: ref.halt_loss(np.array([], dtype=np.float64), 0),
    )
    _both_raise_latent(
        lambda: latent.halt_loss(np.array([[0.5]], dtype=np.float64), 0),
        lambda: ref.halt_loss(np.array([[0.5]], dtype=np.float64), 0),
    )
    _both_raise_latent(
        lambda: latent.halt_loss(np.array([0.5, math.nan], dtype=np.float64), 0),
        lambda: ref.halt_loss(np.array([0.5, math.nan], dtype=np.float64), 0),
    )
    p = np.array(GOLDEN_PONDER, dtype=np.float64)
    _both_raise_latent(
        lambda: latent.halt_kl(p, 0.0),
        lambda: ref.halt_kl(p, 0.0),
    )
    _both_raise_latent(
        lambda: latent.halt_kl(p, 1.0),
        lambda: ref.halt_kl(p, 1.0),
    )


# ---------------------------------------------------------------------------
# 6. thought_decode_ce
# ---------------------------------------------------------------------------


def test_thought_decode_ce_mean_over_mask():
    logits = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    ids = np.array([0, 1, 0], dtype=np.int64)
    mask = np.array([1, 1, 0], dtype=np.int64)
    got = latent.thought_decode_ce(logits, ids, mask)
    exp = ref.thought_decode_ce(logits, ids, mask)

    def _ce_row(row: np.ndarray, label: int) -> float:
        shifted = row - np.max(row)
        logp = shifted - math.log(float(np.sum(np.exp(shifted))))
        return float(-logp[label])

    hand = 0.5 * (_ce_row(logits[0], 0) + _ce_row(logits[1], 1))
    _close(got, hand)
    _close(exp, hand)
    _close(got, exp)


def test_thought_decode_ce_all_zero_mask_is_zero():
    logits = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)
    ids = np.array([1, 0], dtype=np.int64)
    mask_int = np.array([0, 0], dtype=np.int64)
    mask_bool = np.array([False, False])
    for mask in (mask_int, mask_bool):
        got = latent.thought_decode_ce(logits, ids, mask)
        exp = ref.thought_decode_ce(logits, ids, mask)
        _close(got, 0.0)
        _close(exp, 0.0)


def test_thought_decode_ce_bool_mask_mean():
    logits = np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int64)
    mask = np.array([True, False])
    got = latent.thought_decode_ce(logits, ids, mask)
    exp = ref.thought_decode_ce(logits, ids, mask)
    _close(got, exp)
    # only row 0 contributes
    shifted = logits[0] - np.max(logits[0])
    logp = shifted - math.log(float(np.sum(np.exp(shifted))))
    _close(got, float(-logp[0]))


def test_thought_decode_ce_oob_and_mismatch():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    _both_raise_latent(
        lambda: latent.thought_decode_ce(
            logits, np.array([2, 0]), np.array([1, 1])
        ),
        lambda: ref.thought_decode_ce(logits, np.array([2, 0]), np.array([1, 1])),
    )
    _both_raise_latent(
        lambda: latent.thought_decode_ce(
            logits, np.array([-1, 0]), np.array([1, 1])
        ),
        lambda: ref.thought_decode_ce(logits, np.array([-1, 0]), np.array([1, 1])),
    )
    # OOB on a masked-out row still raises.
    _both_raise_latent(
        lambda: latent.thought_decode_ce(
            logits, np.array([9, 0]), np.array([0, 1])
        ),
        lambda: ref.thought_decode_ce(logits, np.array([9, 0]), np.array([0, 1])),
    )
    _both_raise_latent(
        lambda: latent.thought_decode_ce(
            logits, np.array([0, 1, 0]), np.array([1, 1])
        ),
        lambda: ref.thought_decode_ce(
            logits, np.array([0, 1, 0]), np.array([1, 1])
        ),
    )
    _both_raise_latent(
        lambda: latent.thought_decode_ce(
            logits, np.array([0, 1]), np.array([1])
        ),
        lambda: ref.thought_decode_ce(logits, np.array([0, 1]), np.array([1])),
    )
    _both_raise_latent(
        lambda: latent.thought_decode_ce(
            np.zeros((0, 4)), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        ),
        lambda: ref.thought_decode_ce(
            np.zeros((0, 4)), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        ),
    )


def test_thought_decode_ce_finite_diff_vs_log_softmax():
    logits = np.array([[0.0, 0.0]], dtype=np.float64)
    ids = np.array([0], dtype=np.int64)
    mask = np.array([1], dtype=np.int64)
    h = 1e-5
    # d CE / d logit_j = softmax_j - 1[j == y]
    for j, analytic in ((0, -0.5), (1, 0.5)):
        plus = logits.copy()
        minus = logits.copy()
        plus[0, j] += h
        minus[0, j] -= h
        d_prod = (
            float(latent.thought_decode_ce(plus, ids, mask))
            - float(latent.thought_decode_ce(minus, ids, mask))
        ) / (2.0 * h)
        d_ref = (
            float(ref.thought_decode_ce(plus, ids, mask))
            - float(ref.thought_decode_ce(minus, ids, mask))
        ) / (2.0 * h)
        _close(d_prod, analytic)
        _close(d_ref, analytic)


# ---------------------------------------------------------------------------
# 7. answer_ce_at_depths
# ---------------------------------------------------------------------------


def test_answer_ce_at_depths_matches_per_row_ce():
    logits = np.array(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    got = np.asarray(latent.answer_ce_at_depths(logits, 0), dtype=np.float64)
    exp = np.asarray(ref.answer_ce_at_depths(logits, 0), dtype=np.float64)
    hand = np.empty(3, dtype=np.float64)
    for i, row in enumerate(logits):
        shifted = row - np.max(row)
        logp = shifted - math.log(float(np.sum(np.exp(shifted))))
        hand[i] = float(-logp[0])
    _close(got, hand)
    _close(exp, hand)
    assert got.shape == (3,)
    assert np.issubdtype(got.dtype, np.floating)


def test_answer_ce_at_depths_errors():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    _both_raise_latent(
        lambda: latent.answer_ce_at_depths(logits, 2),
        lambda: ref.answer_ce_at_depths(logits, 2),
    )
    _both_raise_latent(
        lambda: latent.answer_ce_at_depths(logits, -1),
        lambda: ref.answer_ce_at_depths(logits, -1),
    )
    _both_raise_latent(
        lambda: latent.answer_ce_at_depths(np.zeros((0, 3)), 0),
        lambda: ref.answer_ce_at_depths(np.zeros((0, 3)), 0),
    )
    _both_raise_latent(
        lambda: latent.answer_ce_at_depths(np.array([1.0, 0.0]), 0),
        lambda: ref.answer_ce_at_depths(np.array([1.0, 0.0]), 0),
    )


# ---------------------------------------------------------------------------
# 8. stage_b_loss
# ---------------------------------------------------------------------------


def _stage_inputs(k: int, vocab: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    halt_logits = rng.normal(size=(k + 1,)).astype(np.float64)
    answer_logits = rng.normal(size=(k + 1, vocab)).astype(np.float64)
    if k == 0:
        thought_logits = np.zeros((0, vocab), dtype=np.float64)
        teacher_ids = np.zeros((0,), dtype=np.int64)
        thought_mask = np.zeros((0,), dtype=np.int64)
    else:
        thought_logits = rng.normal(size=(k, vocab)).astype(np.float64)
        teacher_ids = rng.integers(0, vocab, size=(k,)).astype(np.int64)
        thought_mask = rng.integers(0, 2, size=(k,)).astype(np.int64)
        thought_mask[0] = 1
    answer_id = int(rng.integers(0, vocab))
    teacher_steps = int(rng.integers(0, k + 1))
    return (
        halt_logits,
        answer_logits,
        thought_logits,
        teacher_ids,
        thought_mask,
        answer_id,
        teacher_steps,
    )


def test_stage_b_loss_four_terms_and_weights():
    halt_logits = np.array([0.0, 0.0], dtype=np.float64)
    answer_logits = np.array([[8.0, 0.0], [0.0, 8.0]], dtype=np.float64)
    thought_logits = np.array([[8.0, 0.0]], dtype=np.float64)
    teacher_ids = np.array([0], dtype=np.int64)
    thought_mask = np.array([1], dtype=np.int64)
    prod_cfg = _prod_cfg(alpha_traj=2.0, gamma_halt=3.0, beta_kl=4.0)
    ref_cfg = _ref_cfg(alpha_traj=2.0, gamma_halt=3.0, beta_kl=4.0)
    got = latent.stage_b_loss(
        halt_logits,
        answer_logits,
        thought_logits,
        teacher_ids,
        thought_mask,
        0,
        0,
        prod_cfg,
    )
    exp = ref.stage_b_loss(
        halt_logits,
        answer_logits,
        thought_logits,
        teacher_ids,
        thought_mask,
        0,
        0,
        ref_cfg,
    )
    _assert_stage(got, exp)
    halt = ref.halt_from_logits(halt_logits)
    answer_ce = ref.answer_ce_at_depths(answer_logits, 0)
    l_task = float(np.sum(np.asarray(halt.p) * np.asarray(answer_ce)))
    l_traj = float(ref.thought_decode_ce(thought_logits, teacher_ids, thought_mask))
    l_halt = float(ref.halt_loss(halt.p, 0))
    l_kl = float(ref.halt_kl(halt.p, 0.2))
    _close(got.l_task, l_task)
    _close(got.l_traj, l_traj)
    _close(got.l_halt, l_halt)
    _close(got.l_kl, l_kl)
    _close(got.expected_depth, halt.expected_depth)
    _close(got.total, l_task + 2.0 * l_traj + 3.0 * l_halt + 4.0 * l_kl)


def test_stage_b_loss_k_zero_traj_is_zero():
    halt_logits = np.array([0.0], dtype=np.float64)
    answer_logits = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    thought_logits = np.zeros((0, 3), dtype=np.float64)
    teacher_ids = np.zeros((0,), dtype=np.int64)
    thought_mask = np.zeros((0,), dtype=np.int64)
    got = latent.stage_b_loss(
        halt_logits,
        answer_logits,
        thought_logits,
        teacher_ids,
        thought_mask,
        0,
        0,
        latent.LatentConfig(),
    )
    exp = ref.stage_b_loss(
        halt_logits,
        answer_logits,
        thought_logits,
        teacher_ids,
        thought_mask,
        0,
        0,
        ref.LatentConfig(),
    )
    _assert_stage(got, exp)
    _close(got.l_traj, 0.0)
    _close(got.expected_depth, 0.0)


def test_stage_b_loss_invalid_config_first():
    # Garbage tensors would also fail, but invalid config is checked first.
    bad_prod = _prod_cfg(max_thoughts=-1)
    bad_ref = _ref_cfg(max_thoughts=-1)
    _both_raise_latent(
        lambda: latent.stage_b_loss(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            0,
            0,
            bad_prod,
        ),
        lambda: ref.stage_b_loss(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            0,
            0,
            bad_ref,
        ),
    )


def test_stage_b_loss_length_mismatch():
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    halt = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # K+1 = 3, K = 2
    answer_wrong = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    thought = np.zeros((2, 2), dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int64)
    mask = np.array([1, 1], dtype=np.int64)
    _both_raise_latent(
        lambda: latent.stage_b_loss(halt, answer_wrong, thought, ids, mask, 0, 0, cfg_p),
        lambda: ref.stage_b_loss(halt, answer_wrong, thought, ids, mask, 0, 0, cfg_r),
    )
    answer_ok = np.zeros((3, 2), dtype=np.float64)
    thought_wrong = np.zeros((1, 2), dtype=np.float64)
    _both_raise_latent(
        lambda: latent.stage_b_loss(
            halt, answer_ok, thought_wrong, ids[:1], mask[:1], 0, 0, cfg_p
        ),
        lambda: ref.stage_b_loss(
            halt, answer_ok, thought_wrong, ids[:1], mask[:1], 0, 0, cfg_r
        ),
    )


# ---------------------------------------------------------------------------
# 9. jacobi_sweeps
# ---------------------------------------------------------------------------


def test_jacobi_identity_leaves_thoughts_unchanged():
    thoughts = np.array([[1.0, -2.0, 3.0], [0.5, 0.25, 0.0]], dtype=np.float64)

    def identity(x):
        return x

    got = latent.jacobi_sweeps(thoughts, identity, 5, 2)
    exp = ref.jacobi_sweeps(thoughts, identity, 5, 2)
    _close(got, thoughts)
    _close(exp, thoughts)
    _close(got, exp)


def test_jacobi_contraction_converges():
    target = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    start = np.zeros_like(target)

    def contract(x):
        return 0.5 * np.asarray(x) + 0.5 * target

    got = np.asarray(latent.jacobi_sweeps(start, contract, 20, 2), dtype=np.float64)
    exp = np.asarray(ref.jacobi_sweeps(start, contract, 20, 2), dtype=np.float64)
    _close(got, target)
    _close(exp, target)


def test_jacobi_truncated_sweeps_does_not_change_cpu_forward():
    thoughts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    def scale(x):
        return 0.5 * np.asarray(x)

    a = latent.jacobi_sweeps(thoughts, scale, 4, 1)
    b = latent.jacobi_sweeps(thoughts, scale, 4, 4)
    ar = ref.jacobi_sweeps(thoughts, scale, 4, 1)
    br = ref.jacobi_sweeps(thoughts, scale, 4, 4)
    _close(a, b)
    _close(ar, br)
    _close(a, (0.5**4) * thoughts)


def test_jacobi_shape_change_and_constraint_errors():
    thoughts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    def shrink(x):
        return np.asarray(x)[:-1]

    _both_raise_latent(
        lambda: latent.jacobi_sweeps(thoughts, shrink, 2, 1),
        lambda: ref.jacobi_sweeps(thoughts, shrink, 2, 1),
    )
    _both_raise_latent(
        lambda: latent.jacobi_sweeps(thoughts, lambda x: x, 0, 1),
        lambda: ref.jacobi_sweeps(thoughts, lambda x: x, 0, 1),
    )
    _both_raise_latent(
        lambda: latent.jacobi_sweeps(thoughts, lambda x: x, 3, 0),
        lambda: ref.jacobi_sweeps(thoughts, lambda x: x, 3, 0),
    )
    _both_raise_latent(
        lambda: latent.jacobi_sweeps(thoughts, lambda x: x, 3, 4),
        lambda: ref.jacobi_sweeps(thoughts, lambda x: x, 3, 4),
    )
    _both_raise_latent(
        lambda: latent.jacobi_sweeps(np.array([1.0, 2.0]), lambda x: x, 2, 1),
        lambda: ref.jacobi_sweeps(np.array([1.0, 2.0]), lambda x: x, 2, 1),
    )
    _both_raise_latent(
        lambda: latent.jacobi_sweeps(np.zeros((0, 3)), lambda x: x, 2, 1),
        lambda: ref.jacobi_sweeps(np.zeros((0, 3)), lambda x: x, 2, 1),
    )
    _both_raise_latent(
        lambda: latent.jacobi_sweeps(np.zeros((3, 0)), lambda x: x, 2, 1),
        lambda: ref.jacobi_sweeps(np.zeros((3, 0)), lambda x: x, 2, 1),
    )


# ---------------------------------------------------------------------------
# 10. clamp_sigma + noisy_latent
# ---------------------------------------------------------------------------


def test_clamp_sigma_elementwise():
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    sigma = np.array([0.0, 1e-6, 0.05, 0.2, 5.0], dtype=np.float64)
    got = np.asarray(latent.clamp_sigma(sigma, cfg_p), dtype=np.float64)
    exp = np.asarray(ref.clamp_sigma(sigma, cfg_r), dtype=np.float64)
    lo, hi = float(latent.SIGMA_MIN), float(latent.SIGMA_MAX)
    hand = np.clip(sigma, lo, hi)
    _close(got, hand)
    _close(exp, hand)
    assert got.shape == sigma.shape


def test_clamp_sigma_empty_nonfinite_invalid_config():
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    _both_raise_latent(
        lambda: latent.clamp_sigma(np.array([], dtype=np.float64), cfg_p),
        lambda: ref.clamp_sigma(np.array([], dtype=np.float64), cfg_r),
    )
    _both_raise_latent(
        lambda: latent.clamp_sigma(np.array([0.05, math.nan], dtype=np.float64), cfg_p),
        lambda: ref.clamp_sigma(np.array([0.05, math.nan], dtype=np.float64), cfg_r),
    )
    _both_raise_latent(
        lambda: latent.clamp_sigma(np.array([0.05], dtype=np.float64), _prod_cfg(sigma_min=0.0)),
        lambda: ref.clamp_sigma(np.array([0.05], dtype=np.float64), _ref_cfg(sigma_min=0.0)),
    )


def test_noisy_latent_z_formula_and_log_density():
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    mu = np.array([0.2, -0.5, 1.0], dtype=np.float64)
    sigma = np.array([1e-6, 0.05, 5.0], dtype=np.float64)
    eps = np.array([0.3, -1.2, 0.7], dtype=np.float64)
    got = latent.noisy_latent(mu, sigma, eps, cfg_p)
    exp = ref.noisy_latent(mu, sigma, eps, cfg_r)
    _assert_noisy(got, exp)
    s = np.asarray(ref.clamp_sigma(sigma, cfg_r), dtype=np.float64)
    z_hand = mu + s * eps
    _close(got.z, z_hand)
    _close(got.sigma, s)
    log_two_pi = math.log(2.0 * math.pi)
    total = 0.0
    for i in range(3):
        total += ((z_hand[i] - mu[i]) / s[i]) ** 2 + 2.0 * math.log(s[i]) + log_two_pi
    _close(got.log_density, -0.5 * total)


def test_noisy_latent_log_density_finite_diff_dz():
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    mu = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    sigma = np.array([0.05, 0.02, 0.08], dtype=np.float64)
    eps = np.array([0.4, -0.5, 1.1], dtype=np.float64)
    h = 1e-6
    s = np.asarray(ref.clamp_sigma(sigma, cfg_r), dtype=np.float64)
    base_p = latent.noisy_latent(mu, sigma, eps, cfg_p)
    base_r = ref.noisy_latent(mu, sigma, eps, cfg_r)
    z = np.asarray(base_p.z, dtype=np.float64)
    for i in range(3):
        eps_plus = eps.copy()
        eps_minus = eps.copy()
        eps_plus[i] += h / s[i]
        eps_minus[i] -= h / s[i]
        d_prod = (
            float(latent.noisy_latent(mu, sigma, eps_plus, cfg_p).log_density)
            - float(latent.noisy_latent(mu, sigma, eps_minus, cfg_p).log_density)
        ) / (2.0 * h)
        d_ref = (
            float(ref.noisy_latent(mu, sigma, eps_plus, cfg_r).log_density)
            - float(ref.noisy_latent(mu, sigma, eps_minus, cfg_r).log_density)
        ) / (2.0 * h)
        analytic = -(z[i] - mu[i]) / (s[i] ** 2)
        _close(d_prod, analytic)
        _close(d_ref, analytic)
        _close(base_p.log_density, base_r.log_density)


def test_noisy_latent_empty_mismatch_order():
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    ok = np.array([0.1, 0.2], dtype=np.float64)
    # mu first
    _both_raise_latent(
        lambda: latent.noisy_latent(np.array([], dtype=np.float64), ok, ok, cfg_p),
        lambda: ref.noisy_latent(np.array([], dtype=np.float64), ok, ok, cfg_r),
    )
    _both_raise_latent(
        lambda: latent.noisy_latent(
            np.array([[0.1, 0.2]], dtype=np.float64), ok, ok, cfg_p
        ),
        lambda: ref.noisy_latent(
            np.array([[0.1, 0.2]], dtype=np.float64), ok, ok, cfg_r
        ),
    )
    _both_raise_latent(
        lambda: latent.noisy_latent(
            np.array([math.nan, 0.2], dtype=np.float64), ok, ok, cfg_p
        ),
        lambda: ref.noisy_latent(
            np.array([math.nan, 0.2], dtype=np.float64), ok, ok, cfg_r
        ),
    )
    # sigma next
    _both_raise_latent(
        lambda: latent.noisy_latent(ok, np.array([], dtype=np.float64), ok, cfg_p),
        lambda: ref.noisy_latent(ok, np.array([], dtype=np.float64), ok, cfg_r),
    )
    _both_raise_latent(
        lambda: latent.noisy_latent(
            ok, np.array([0.05, math.inf], dtype=np.float64), ok, cfg_p
        ),
        lambda: ref.noisy_latent(
            ok, np.array([0.05, math.inf], dtype=np.float64), ok, cfg_r
        ),
    )
    # eps next
    _both_raise_latent(
        lambda: latent.noisy_latent(ok, ok, np.array([0.0], dtype=np.float64)[:0], cfg_p),
        lambda: ref.noisy_latent(ok, ok, np.array([0.0], dtype=np.float64)[:0], cfg_r),
    )
    _both_raise_latent(
        lambda: latent.noisy_latent(
            ok, ok, np.array([0.0, math.nan], dtype=np.float64), cfg_p
        ),
        lambda: ref.noisy_latent(
            ok, ok, np.array([0.0, math.nan], dtype=np.float64), cfg_r
        ),
    )
    # length match last
    _both_raise_latent(
        lambda: latent.noisy_latent(
            ok, np.array([0.05, 0.05, 0.05], dtype=np.float64), ok, cfg_p
        ),
        lambda: ref.noisy_latent(
            ok, np.array([0.05, 0.05, 0.05], dtype=np.float64), ok, cfg_r
        ),
    )


# ---------------------------------------------------------------------------
# 11. Property: lockstep prod vs ref on random small vectors
# ---------------------------------------------------------------------------


def test_lockstep_prod_vs_ref_random_small_vectors():
    rng = np.random.default_rng(20260322)
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    for n in range(1, 9):
        lam = rng.uniform(0.05, 0.95, size=(n,))
        _close(latent.ponder_distribution(lam), ref.ponder_distribution(lam))
        logits = rng.normal(size=(n,))
        _assert_halt(latent.halt_from_logits(logits), ref.halt_from_logits(logits))
        p = np.asarray(ref.ponder_distribution(lam), dtype=np.float64)
        teacher = int(rng.integers(-2, n + 3))
        _close(latent.halt_loss(p, teacher), ref.halt_loss(p, teacher))
        _close(latent.halt_kl(p, 0.2), ref.halt_kl(p, 0.2))
        _close(latent.geometric_prior(n, 0.35), ref.geometric_prior(n, 0.35))
        mu = rng.normal(size=(n,))
        sigma = rng.uniform(1e-5, 0.5, size=(n,))
        eps = rng.normal(size=(n,))
        _assert_noisy(
            latent.noisy_latent(mu, sigma, eps, cfg_p),
            ref.noisy_latent(mu, sigma, eps, cfg_r),
        )
        _close(latent.clamp_sigma(sigma, cfg_p), ref.clamp_sigma(sigma, cfg_r))
        vocab = 5
        thought_logits = rng.normal(size=(n, vocab))
        ids = rng.integers(0, vocab, size=(n,))
        mask = rng.integers(0, 2, size=(n,))
        if int(mask.sum()) == 0:
            mask[0] = 1
        _close(
            latent.thought_decode_ce(thought_logits, ids, mask),
            ref.thought_decode_ce(thought_logits, ids, mask),
        )
        answer_logits = rng.normal(size=(n, vocab))
        aid = int(rng.integers(0, vocab))
        _close(
            latent.answer_ce_at_depths(answer_logits, aid),
            ref.answer_ce_at_depths(answer_logits, aid),
        )
        thoughts = rng.normal(size=(n, max(n, 1)))
        _close(
            latent.jacobi_sweeps(thoughts, lambda x: 0.7 * np.asarray(x), 3, 2),
            ref.jacobi_sweeps(thoughts, lambda x: 0.7 * np.asarray(x), 3, 2),
        )
    for k in range(0, 9):
        (
            halt_logits,
            answer_logits,
            thought_logits,
            teacher_ids,
            thought_mask,
            answer_id,
            teacher_steps,
        ) = _stage_inputs(k, vocab=6, seed=1000 + k)
        got = latent.stage_b_loss(
            halt_logits,
            answer_logits,
            thought_logits,
            teacher_ids,
            thought_mask,
            answer_id,
            teacher_steps,
            cfg_p,
        )
        exp = ref.stage_b_loss(
            halt_logits,
            answer_logits,
            thought_logits,
            teacher_ids,
            thought_mask,
            answer_id,
            teacher_steps,
            cfg_r,
        )
        _assert_stage(got, exp)
