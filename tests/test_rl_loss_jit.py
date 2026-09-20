"""I2-loss-jit oracle: jax.jit of the six public loss kernels must match eager at 1e-5.

These tests call the public ``rl.loss`` signatures with 1-D ``jnp.float32`` arrays
and host-side Python scalars (``k``, ``tis_clip``, ``eps_low``, ``eps_high``
closed over or ``static_argnums``). They must fail on the current
``numpy.asarray`` / Python ``float()`` / ``math.log`` / ``math.exp`` /
``math.isfinite`` implementations (``TracerArrayConversionError`` or
``ConcretizationTypeError``) and pass once the six functions stay in JAX.

Coverage
--------
1. jit vs eager at 1e-5 for gaussian_log_density, sequence_ratio, latent_ratio,
   truncated_is, clip_higher, mean_center (1-D jnp arrays; host Python scalars).
2. jit vs tests.reference.rl_loss at 1e-5, independent of production math.
3. jax.grad through jitted gaussian_log_density vs eager jax.grad at 1e-5;
   cheap finite-difference vs the NumPy/reference at FD_TOL 1e-3.
4. jax.grad through jitted sequence_ratio and latent_ratio vs eager at 1e-5.
5. clip_higher: ratio below 1-eps_low, inside the band, above 1+eps_high;
   unused autodiff branch stays finite (grad wrt ratio).
6. truncated_is: k=0 -> 1.0; k=1 and k=4 with tis_clip=2; ratio above clip is
   clipped. k and tis_clip are static. Do not Python-branch on a traced ratio.
7. mean_center: (1, 2, 3) -> (-1, 0, 1); jit vs eager vs reference. Subtract
   group mean, no std divide.
8. latent_ratio empty both sides -> 1.0 (eager Python tuples and jit of
   length-0 1-D arrays).
9. shape/dtype: jitted outputs are float32 scalars (0-d arrays ok) except
   mean_center, a float32 sequence of the same length as rewards.
10. Do not assert exact Python-float equality on jitted scalars; use _close
    at 1e-5.

Do not mark gpu. Do not jit gspo_dapo_loss, Group, Sample, make_config,
drop_zero_advantage_groups, all_equal_reward, staleness, routing_table, or
overlong_soft_penalty as entry points. Tests import rl.loss; rl.loss must not
import tests.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

import rl.loss as loss
from tests.reference import rl_loss as ref

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)

LOG2 = float(math.log(2.0))
LOG4 = float(math.log(4.0))

# Independent 1-D Gaussians (spec 4.4 / 9.1).
Z = np.array([0.3, -0.2, 1.1], dtype=np.float32)
MU = np.array([0.1, 0.0, 0.5], dtype=np.float32)
SIGMA = np.array([0.5, 1.25, 0.8], dtype=np.float32)
Z_V = np.array([0.2, -0.4, 0.1], dtype=np.float32)

# sequence_ratio / latent_ratio: exp(sum(new-old)), no length divide.
# (0, log 2) vs zeros -> 2.0; (log 2, log 2) vs zeros -> 4.0 not 2.0.
LOGP_NEW = np.array([0.0, LOG2], dtype=np.float32)
LOGP_OLD = np.array([0.0, 0.0], dtype=np.float32)
LOGP_NEW_NOLEN = np.array([LOG2, LOG2], dtype=np.float32)
LOGP_NEW_V = np.array([0.3, -0.2], dtype=np.float32)

# truncated_is: ratio 4.0 (above tis_clip=2) and 0.5 (below clip).
TIS_TRAINER_HI = np.array([0.0, LOG4], dtype=np.float32)
TIS_ROLLOUT = np.array([0.0, 0.0], dtype=np.float32)
TIS_TRAINER_LO = np.array([0.0], dtype=np.float32)
TIS_ROLLOUT_LO = np.array([LOG2], dtype=np.float32)

EPS_LOW = 0.2
EPS_HIGH = 0.28
CLIP_BELOW = 0.5
CLIP_INSIDE = 1.0
CLIP_ABOVE = 1.5

REWARDS_123 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
REWARDS_123_CENTERED = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
# std of (1, 3, 5) is not 1, so std-divide would not match (-2, 0, 2).
REWARDS_135 = np.array([1.0, 3.0, 5.0], dtype=np.float32)


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _j32(x: object) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _tup(x: object) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(x).reshape(-1))


def _close(got: object, exp: object, **kwargs: float) -> None:
    tol = dict(TOL)
    tol.update(kwargs)
    np.testing.assert_allclose(_np(got), _np(exp), **tol)


def _assert_f32_scalar(got: object) -> None:
    arr = _np(got)
    assert arr.dtype == np.float32
    assert arr.shape == ()


def _assert_f32_vec(got: object, n: int) -> None:
    arr = _np(got)
    assert arr.dtype == np.float32
    assert arr.shape == (n,)


# ---------------------------------------------------------------------------
# gaussian_log_density
# ---------------------------------------------------------------------------


def test_gaussian_log_density_jit_matches_eager_and_reference() -> None:
    z, mu, sigma = _j32(Z), _j32(MU), _j32(SIGMA)
    jitted = jax.jit(loss.gaussian_log_density)
    got = jitted(z, mu, sigma)
    eager = loss.gaussian_log_density(z, mu, sigma)
    gold = ref.gaussian_log_density(_tup(Z), _tup(MU), _tup(SIGMA))
    _close(got, eager)
    _close(got, gold)
    _assert_f32_scalar(got)


def test_gaussian_log_density_jitted_grad_matches_eager_grad() -> None:
    z, mu, sigma = _j32(Z), _j32(MU), _j32(SIGMA)

    def gld(z_):
        return loss.gaussian_log_density(z_, mu, sigma)

    jit_g = jax.grad(jax.jit(gld))(z)
    eager_g = jax.grad(gld)(z)
    _close(jit_g, eager_g)
    _assert_f32_vec(jit_g, Z.shape[0])


def test_gaussian_log_density_jitted_grad_finite_difference_vs_reference() -> None:
    z, mu, sigma = _j32(Z), _j32(MU), _j32(SIGMA)

    def gld(z_):
        return loss.gaussian_log_density(z_, mu, sigma)

    jit_g = jax.grad(jax.jit(gld))(z)
    directional = float(np.sum(_np(jit_g) * Z_V))

    eps = np.float32(1e-3)

    def ref_gld(z_np: np.ndarray) -> float:
        return float(ref.gaussian_log_density(_tup(z_np), _tup(MU), _tup(SIGMA)))

    fd = (ref_gld(Z + eps * Z_V) - ref_gld(Z - eps * Z_V)) / (2.0 * float(eps))
    _close(directional, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# sequence_ratio
# ---------------------------------------------------------------------------


def test_sequence_ratio_jit_matches_eager_and_reference() -> None:
    new, old = _j32(LOGP_NEW), _j32(LOGP_OLD)
    jitted = jax.jit(loss.sequence_ratio)
    got = jitted(new, old)
    eager = loss.sequence_ratio(new, old)
    gold = ref.sequence_ratio(_tup(LOGP_NEW), _tup(LOGP_OLD))
    _close(got, eager)
    _close(got, gold)

    # No length divide: exp(log 2 + log 2) = 4, not 2.
    new4 = _j32(LOGP_NEW_NOLEN)
    got4 = jitted(new4, old)
    gold4 = ref.sequence_ratio(_tup(LOGP_NEW_NOLEN), _tup(LOGP_OLD))
    _close(got4, gold4)
    _close(got4, loss.sequence_ratio(new4, old))
    _assert_f32_scalar(got)
    _assert_f32_scalar(got4)


def test_sequence_ratio_jitted_grad_matches_eager_grad() -> None:
    new, old = _j32(LOGP_NEW), _j32(LOGP_OLD)
    jitted = jax.jit(loss.sequence_ratio)
    jit_g = jax.grad(jitted)(new, old)
    eager_g = jax.grad(loss.sequence_ratio)(new, old)
    _close(jit_g, eager_g)
    _assert_f32_vec(jit_g, LOGP_NEW.shape[0])


# ---------------------------------------------------------------------------
# latent_ratio
# ---------------------------------------------------------------------------


def test_latent_ratio_jit_matches_eager_and_reference() -> None:
    new, old = _j32(LOGP_NEW), _j32(LOGP_OLD)
    jitted = jax.jit(loss.latent_ratio)
    got = jitted(new, old)
    eager = loss.latent_ratio(new, old)
    gold = ref.latent_ratio(_tup(LOGP_NEW), _tup(LOGP_OLD))
    _close(got, eager)
    _close(got, gold)
    _assert_f32_scalar(got)


def test_latent_ratio_jit_empty_both_sides_is_one() -> None:
    # Empty-on-both-sides is a Python (static) length check. Eager tuples and
    # jit of length-0 1-D arrays (static shape) both return 1.0.
    _close(loss.latent_ratio((), ()), 1.0)
    _close(ref.latent_ratio((), ()), 1.0)

    empty = jnp.zeros((0,), dtype=jnp.float32)
    got = jax.jit(loss.latent_ratio)(empty, empty)
    _close(got, np.float32(1.0))
    _close(got, ref.latent_ratio((), ()))
    _assert_f32_scalar(got)

    # Non-empty 1-D path must also stay on tracers (empty-only would not).
    new, old = _j32(LOGP_NEW), _j32(LOGP_OLD)
    got_ne = jax.jit(loss.latent_ratio)(new, old)
    _close(got_ne, loss.latent_ratio(new, old))
    _close(got_ne, ref.latent_ratio(_tup(LOGP_NEW), _tup(LOGP_OLD)))


def test_latent_ratio_jitted_grad_matches_eager_grad() -> None:
    new, old = _j32(LOGP_NEW), _j32(LOGP_OLD)
    jitted = jax.jit(loss.latent_ratio)
    jit_g = jax.grad(jitted)(new, old)
    eager_g = jax.grad(loss.latent_ratio)(new, old)
    _close(jit_g, eager_g)
    _assert_f32_vec(jit_g, LOGP_NEW.shape[0])


# ---------------------------------------------------------------------------
# truncated_is (k, tis_clip static)
# ---------------------------------------------------------------------------


def test_truncated_is_jit_k_and_clip() -> None:
    trainer_hi, rollout = _j32(TIS_TRAINER_HI), _j32(TIS_ROLLOUT)
    trainer_lo, rollout_lo = _j32(TIS_TRAINER_LO), _j32(TIS_ROLLOUT_LO)
    hi_t, roll_t = _tup(TIS_TRAINER_HI), _tup(TIS_ROLLOUT)
    lo_t, roll_lo_t = _tup(TIS_TRAINER_LO), _tup(TIS_ROLLOUT_LO)
    tis_clip = 2.0

    def tis_k0(t, r):
        return loss.truncated_is(t, r, 0, tis_clip)

    def tis_k1(t, r):
        return loss.truncated_is(t, r, 1, tis_clip)

    def tis_k4(t, r):
        return loss.truncated_is(t, r, 4, tis_clip)

    # k==0 returns 1.0 without using rollout log-probs (closed-over k).
    got0 = jax.jit(tis_k0)(trainer_hi, rollout)
    _close(got0, np.float32(1.0))
    _close(got0, loss.truncated_is(trainer_hi, rollout, 0, tis_clip))
    _close(got0, ref.truncated_is(hi_t, roll_t, 0, tis_clip))

    # k>0 is min(sequence_ratio, tis_clip). Ratio 4 is clipped to 2.
    # Do not Python-branch on the traced ratio when applying the clip.
    got1 = jax.jit(tis_k1)(trainer_hi, rollout)
    gold1 = ref.truncated_is(hi_t, roll_t, 1, tis_clip)
    _close(got1, loss.truncated_is(trainer_hi, rollout, 1, tis_clip))
    _close(got1, gold1)

    got4 = jax.jit(tis_k4)(trainer_hi, rollout)
    gold4 = ref.truncated_is(hi_t, roll_t, 4, tis_clip)
    _close(got4, loss.truncated_is(trainer_hi, rollout, 4, tis_clip))
    _close(got4, gold4)
    _close(got1, got4)

    # Ratio 0.5 is below tis_clip=2 and is not raised.
    got_lo = jax.jit(tis_k1)(trainer_lo, rollout_lo)
    _close(got_lo, loss.truncated_is(trainer_lo, rollout_lo, 1, tis_clip))
    _close(got_lo, ref.truncated_is(lo_t, roll_lo_t, 1, tis_clip))
    _assert_f32_scalar(got0)
    _assert_f32_scalar(got1)
    _assert_f32_scalar(got4)
    _assert_f32_scalar(got_lo)


def test_truncated_is_jit_static_argnums() -> None:
    trainer, rollout = _j32(TIS_TRAINER_HI), _j32(TIS_ROLLOUT)
    jitted = jax.jit(loss.truncated_is, static_argnums=(2, 3))
    got = jitted(trainer, rollout, 1, 2.0)
    gold = ref.truncated_is(_tup(TIS_TRAINER_HI), _tup(TIS_ROLLOUT), 1, 2.0)
    _close(got, gold)
    _close(got, loss.truncated_is(trainer, rollout, 1, 2.0))
    got4 = jitted(trainer, rollout, 4, 2.0)
    _close(got4, gold)
    _assert_f32_scalar(got)


# ---------------------------------------------------------------------------
# clip_higher (eps_low, eps_high static)
# ---------------------------------------------------------------------------


def test_clip_higher_jit_below_inside_above() -> None:
    def ch(ratio):
        return loss.clip_higher(ratio, EPS_LOW, EPS_HIGH)

    jitted = jax.jit(ch)
    for raw in (CLIP_BELOW, CLIP_INSIDE, CLIP_ABOVE):
        ratio = jnp.asarray(np.float32(raw))
        got = jitted(ratio)
        eager = loss.clip_higher(ratio, EPS_LOW, EPS_HIGH)
        gold = ref.clip_higher(float(raw), EPS_LOW, EPS_HIGH)
        _close(got, eager)
        _close(got, gold)
        _assert_f32_scalar(got)


def test_clip_higher_jit_static_argnums_epsilons() -> None:
    jitted = jax.jit(loss.clip_higher, static_argnums=(1, 2))
    for raw in (CLIP_BELOW, CLIP_INSIDE, CLIP_ABOVE):
        ratio = jnp.asarray(np.float32(raw))
        got = jitted(ratio, EPS_LOW, EPS_HIGH)
        gold = ref.clip_higher(float(raw), EPS_LOW, EPS_HIGH)
        _close(got, gold)
        _close(got, loss.clip_higher(ratio, EPS_LOW, EPS_HIGH))
        _assert_f32_scalar(got)


def test_clip_higher_jitted_grad_unused_branch_finite() -> None:
    def ch(ratio):
        return loss.clip_higher(ratio, EPS_LOW, EPS_HIGH)

    jitted = jax.jit(ch)
    below = jnp.asarray(np.float32(CLIP_BELOW))
    inside = jnp.asarray(np.float32(CLIP_INSIDE))
    above = jnp.asarray(np.float32(CLIP_ABOVE))

    g_below = jax.grad(jitted)(below)
    g_inside = jax.grad(jitted)(inside)
    g_above = jax.grad(jitted)(above)
    assert np.isfinite(_np(g_below)).all()
    assert np.isfinite(_np(g_inside)).all()
    assert np.isfinite(_np(g_above)).all()
    _close(g_below, jax.grad(ch)(below))
    _close(g_inside, jax.grad(ch)(inside))
    _close(g_above, jax.grad(ch)(above))
    # Clip is locally constant outside the band; identity inside.
    _close(g_below, np.float32(0.0))
    _close(g_inside, np.float32(1.0))
    _close(g_above, np.float32(0.0))


# ---------------------------------------------------------------------------
# mean_center
# ---------------------------------------------------------------------------


def test_mean_center_jit_known_rewards() -> None:
    rewards = _j32(REWARDS_123)
    jitted = jax.jit(loss.mean_center)
    got = jitted(rewards)
    eager = loss.mean_center(rewards)
    gold = ref.mean_center(_tup(REWARDS_123))
    _close(got, eager)
    _close(got, gold)
    _close(got, REWARDS_123_CENTERED)
    _assert_f32_vec(got, 3)

    rewards135 = _j32(REWARDS_135)
    got135 = jitted(rewards135)
    _close(got135, loss.mean_center(rewards135))
    _close(got135, ref.mean_center(_tup(REWARDS_135)))
    _assert_f32_vec(got135, 3)


# ---------------------------------------------------------------------------
# shape / dtype
# ---------------------------------------------------------------------------


def test_loss_jit_shape_dtype_float32() -> None:
    z, mu, sigma = _j32(Z), _j32(MU), _j32(SIGMA)
    new, old = _j32(LOGP_NEW), _j32(LOGP_OLD)
    rewards = _j32(REWARDS_123)
    ratio = jnp.asarray(np.float32(CLIP_ABOVE))

    gld = jax.jit(loss.gaussian_log_density)(z, mu, sigma)
    seq = jax.jit(loss.sequence_ratio)(new, old)
    lat = jax.jit(loss.latent_ratio)(new, old)

    def tis(t, r):
        return loss.truncated_is(t, r, 1, 2.0)

    tis_v = jax.jit(tis)(new, old)

    def ch(r):
        return loss.clip_higher(r, EPS_LOW, EPS_HIGH)

    clip_v = jax.jit(ch)(ratio)
    mc = jax.jit(loss.mean_center)(rewards)

    _assert_f32_scalar(gld)
    _assert_f32_scalar(seq)
    _assert_f32_scalar(lat)
    _assert_f32_scalar(tis_v)
    _assert_f32_scalar(clip_v)
    _assert_f32_vec(mc, REWARDS_123.shape[0])
