"""I5-jacobi-jit oracle: ``jax.jit`` of Jacobi / noisy-latent ops must match eager at 1e-5.

These tests call the public ``model.latent`` signatures with ``jnp`` arrays and
Python scalars / ``LatentConfig`` held static (closed over or
``static_argnums``). They must fail on the current ``numpy.asarray`` /
Python ``float()`` / Python-loop implementations (``TracerArrayConversionError``
or ``ConcretizationTypeError``) and pass once ``jacobi_sweeps``,
``clamp_sigma``, and ``noisy_latent`` stay in JAX.

Coverage
--------
1. jit vs eager (1e-5): clamp_sigma; noisy_latent (z, sigma, log_density);
   jacobi_sweeps (identity update; contraction ``0.5 * x``).
2. jit vs tests/reference/latent.py (1e-5), independent of production math.
3. jax.grad through jitted clamp_sigma / noisy_latent.log_density vs eager
   jax.grad at 1e-5; cheap finite-diff vs the NumPy reference.
4. clamp: below sigma_min, above sigma_max, interior unchanged. Unused clamp
   branch stays finite (no nan on the unselected side that would poison
   jax.grad).
5. noisy_latent length mismatch / empty still raise LatentError on host
   inputs (eager). Under jit, host-side config. log_density formula on
   clamped s.
6. jacobi: identity leaves thoughts; CPU forward identical for
   truncated_sweeps=1 vs n_sweeps (same n_sweeps); n_sweeps / truncated
   host ints. Do not pass n_sweeps as a traced positional without
   static_argnums.
7. shape / dtype: jitted log_density is float32 0-d; vectors keep input
   shape. Do not mark gpu.

Do not mark gpu. Python scalars (``n_sweeps``, ``truncated_sweeps``) and
``LatentConfig`` stay host-side. Expected values come from
tests/reference/, not from copying production. Do not jit
``validate_latent_config`` as an entry point.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import model.latent as latent
from tests.reference import latent as ref

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)

# Mixed clamp vector: two below min, one interior, two above max.
CLAMP_SIGMA = np.array([0.0, 1e-6, 0.05, 0.2, 5.0], dtype=np.float32)
# Interior-only, away from the clip boundaries, for finite differences.
CLAMP_INTERIOR = np.array([0.02, 0.04, 0.06], dtype=np.float32)
CLAMP_INTERIOR_V = np.array([0.3, -0.2, 0.1], dtype=np.float32)

# noisy_latent golden: sigma hits below / interior / above.
NOISY_MU = np.array([0.2, -0.5, 1.0], dtype=np.float32)
NOISY_SIGMA = np.array([1e-6, 0.05, 5.0], dtype=np.float32)
NOISY_EPS = np.array([0.3, -1.2, 0.7], dtype=np.float32)
NOISY_INTERIOR_SIGMA = np.array([0.05, 0.02, 0.08], dtype=np.float32)
NOISY_INTERIOR_MU = np.array([0.1, -0.2, 0.3], dtype=np.float32)
NOISY_INTERIOR_EPS = np.array([0.4, -0.5, 1.1], dtype=np.float32)
NOISY_INTERIOR_V = np.array([0.2, -0.3, 0.1], dtype=np.float32)

JACOBI_THOUGHTS = np.array(
    [[1.0, -2.0, 3.0], [0.5, 0.25, 0.0]],
    dtype=np.float32,
)
JACOBI_SCALE = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _j32(x: object) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _close(got: object, exp: object, **kwargs: float) -> None:
    tol = dict(TOL)
    tol.update(kwargs)
    np.testing.assert_allclose(_np(got), _np(exp), **tol)


def _assert_f32_vector(got: object, shape: tuple[int, ...]) -> None:
    arr = _np(got)
    assert arr.shape == shape
    assert _np(got).dtype == np.float32 or jnp.asarray(got).dtype == jnp.float32
    assert jnp.asarray(got).dtype == jnp.float32


def _assert_f32_scalar(got: object) -> None:
    arr = jnp.asarray(got)
    assert arr.dtype == jnp.float32
    assert arr.shape == ()
    assert np.ndim(got) == 0


def _identity(x: jax.Array) -> jax.Array:
    return x


def _half(x: jax.Array) -> jax.Array:
    return 0.5 * x


def _noisy_fields(
    mu: jax.Array,
    sigma: jax.Array,
    eps: jax.Array,
    config: latent.LatentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    out = latent.noisy_latent(mu, sigma, eps, config)
    return out.mu, out.sigma, out.eps, out.z, out.log_density


def _clamp_sum(sigma: jax.Array, config: latent.LatentConfig) -> jax.Array:
    return jnp.sum(latent.clamp_sigma(sigma, config))


def _noisy_log_density(
    mu: jax.Array,
    sigma: jax.Array,
    eps: jax.Array,
    config: latent.LatentConfig,
) -> jax.Array:
    return latent.noisy_latent(mu, sigma, eps, config).log_density


def _hand_noisy(
    mu: np.ndarray,
    sigma: np.ndarray,
    eps: np.ndarray,
    sigma_min: float,
    sigma_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Spec 4.4 Gaussian log-density on clamped s, independent of production."""
    mu64 = np.asarray(mu, dtype=np.float64)
    s = np.clip(np.asarray(sigma, dtype=np.float64), sigma_min, sigma_max)
    e = np.asarray(eps, dtype=np.float64)
    z = mu64 + s * e
    log_two_pi = math.log(2.0 * math.pi)
    total = np.sum(((z - mu64) / s) ** 2 + 2.0 * np.log(s) + log_two_pi)
    return z, s, float(-0.5 * total)


# ---------------------------------------------------------------------------
# 1–2, 4, 7. clamp_sigma: jit vs eager vs reference
# ---------------------------------------------------------------------------


def test_clamp_sigma_jit_matches_eager_and_reference() -> None:
    """jax.jit(clamp_sigma) with config closed over matches eager and the NumPy ref."""
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    sigma = _j32(CLAMP_SIGMA)
    lo, hi = float(latent.SIGMA_MIN), float(latent.SIGMA_MAX)

    def cs(s):
        return latent.clamp_sigma(s, cfg_p)

    got = jax.jit(cs)(sigma)
    eager = cs(sigma)
    exp = ref.clamp_sigma(_np(CLAMP_SIGMA), cfg_r)
    hand = np.clip(np.asarray(CLAMP_SIGMA, dtype=np.float64), lo, hi)

    _close(got, eager)
    _close(got, exp)
    _close(got, hand)
    _assert_f32_vector(got, CLAMP_SIGMA.shape)
    # below min / above max / interior unchanged
    assert float(got[0]) == pytest.approx(lo, **TOL)
    assert float(got[1]) == pytest.approx(lo, **TOL)
    assert float(got[2]) == pytest.approx(0.05, **TOL)
    assert float(got[3]) == pytest.approx(hi, **TOL)
    assert float(got[4]) == pytest.approx(hi, **TOL)


def test_clamp_sigma_jit_static_argnums_config() -> None:
    """Config is host-side via static_argnums; jit matches eager at 1e-5."""
    cfg = latent.LatentConfig()
    sigma = _j32(CLAMP_SIGMA)
    jitted = jax.jit(latent.clamp_sigma, static_argnums=(1,))
    got = jitted(sigma, cfg)
    eager = latent.clamp_sigma(sigma, cfg)
    _close(got, eager)
    _close(got, ref.clamp_sigma(_np(CLAMP_SIGMA), ref.LatentConfig()))
    _assert_f32_vector(got, CLAMP_SIGMA.shape)


def test_clamp_sigma_unused_branch_grad_stays_finite() -> None:
    """Values below min / above max clamp; unused branch must not poison jax.grad."""
    cfg = latent.LatentConfig()
    lo, hi = float(latent.SIGMA_MIN), float(latent.SIGMA_MAX)
    # 0 and negative would nan a log/div unused branch; 5.0 is above max.
    sigma_np = np.array([0.0, -1.0, 0.05, 5.0], dtype=np.float32)
    sigma = _j32(sigma_np)

    def cs(s):
        return latent.clamp_sigma(s, cfg)

    def clamped_sum(s):
        return jnp.sum(cs(s))

    def log_of_clamped(s):
        # log of the unused (unclamped) side would be nan at 0 / negative.
        return jnp.sum(jnp.log(cs(s)))

    got = jax.jit(cs)(sigma)
    _close(got, np.array([lo, lo, 0.05, hi], dtype=np.float64))
    assert np.all(np.isfinite(_np(got)))

    jit_g = jax.grad(jax.jit(clamped_sum))(sigma)
    eager_g = jax.grad(clamped_sum)(sigma)
    _close(jit_g, eager_g)
    assert np.all(np.isfinite(_np(jit_g)))
    _close(jit_g, np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64))

    log_g = jax.grad(jax.jit(log_of_clamped))(sigma)
    assert np.all(np.isfinite(_np(log_g)))
    _close(log_g, np.array([0.0, 0.0, 1.0 / 0.05, 0.0], dtype=np.float64))


# ---------------------------------------------------------------------------
# 3. jax.grad through jitted clamp_sigma
# ---------------------------------------------------------------------------


def test_clamp_sigma_jitted_grad_matches_eager_grad() -> None:
    cfg = latent.LatentConfig()
    sigma = _j32(CLAMP_INTERIOR)

    def loss(s):
        return _clamp_sum(s, cfg)

    jit_g = jax.grad(jax.jit(loss))(sigma)
    eager_g = jax.grad(loss)(sigma)
    _close(jit_g, eager_g)
    # Interior: d/ds clip(s) = 1.
    _close(jit_g, np.ones_like(CLAMP_INTERIOR, dtype=np.float64))
    _assert_f32_vector(jit_g, CLAMP_INTERIOR.shape)


def test_clamp_sigma_jitted_grad_finite_difference_vs_reference() -> None:
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    sigma_np = np.asarray(CLAMP_INTERIOR, dtype=np.float64)
    v_np = np.asarray(CLAMP_INTERIOR_V, dtype=np.float64)
    sigma = _j32(sigma_np)
    v = _j32(v_np)

    def loss(s):
        return _clamp_sum(s, cfg_p)

    jit_g = jax.grad(jax.jit(loss))(sigma)
    directional = float(jnp.sum(jit_g * v))

    eps = 1e-4
    plus = ref.clamp_sigma(sigma_np + eps * v_np, cfg_r)
    minus = ref.clamp_sigma(sigma_np - eps * v_np, cfg_r)
    fd = (float(np.sum(plus)) - float(np.sum(minus))) / (2.0 * eps)
    _close(directional, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# 1–2, 5, 7. noisy_latent: jit vs eager vs reference
# ---------------------------------------------------------------------------


def test_noisy_latent_jit_matches_eager_and_reference() -> None:
    """jit vs eager vs NumPy ref for z, clamped sigma, and log_density."""
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)

    def fields(m, s, e):
        return _noisy_fields(m, s, e, cfg_p)

    got_mu, got_s, got_e, got_z, got_ld = jax.jit(fields)(mu, sigma, eps)
    eager_mu, eager_s, eager_e, eager_z, eager_ld = fields(mu, sigma, eps)
    exp = ref.noisy_latent(_np(NOISY_MU), _np(NOISY_SIGMA), _np(NOISY_EPS), cfg_r)
    z_hand, s_hand, ld_hand = _hand_noisy(
        NOISY_MU,
        NOISY_SIGMA,
        NOISY_EPS,
        float(latent.SIGMA_MIN),
        float(latent.SIGMA_MAX),
    )

    _close(got_z, eager_z)
    _close(got_s, eager_s)
    _close(got_ld, eager_ld)
    _close(got_z, exp.z)
    _close(got_s, exp.sigma)
    _close(got_ld, exp.log_density)
    _close(got_z, z_hand)
    _close(got_s, s_hand)
    _close(got_ld, ld_hand)
    _close(got_mu, eager_mu)
    _close(got_e, eager_e)

    _assert_f32_vector(got_z, NOISY_MU.shape)
    _assert_f32_vector(got_s, NOISY_SIGMA.shape)
    _assert_f32_scalar(got_ld)


def test_noisy_latent_jit_static_argnums_config() -> None:
    """Under jit, LatentConfig stays host-side via static_argnums."""
    cfg = latent.LatentConfig()
    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)
    jitted = jax.jit(_noisy_fields, static_argnums=(3,))
    got_mu, got_s, got_e, got_z, got_ld = jitted(mu, sigma, eps, cfg)
    eager = latent.noisy_latent(mu, sigma, eps, cfg)
    _close(got_z, eager.z)
    _close(got_s, eager.sigma)
    _close(got_ld, eager.log_density)
    _close(got_mu, eager.mu)
    _close(got_e, eager.eps)
    _assert_f32_scalar(got_ld)


def test_noisy_latent_log_density_formula_on_clamped_s() -> None:
    """log_density = -0.5 * sum[((z-mu)/s)^2 + 2 log s + log(2π)] on clamped s."""
    cfg = latent.LatentConfig()
    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)

    def fields(m, s, e):
        return _noisy_fields(m, s, e, cfg)

    _got_mu, got_s, _got_e, got_z, got_ld = jax.jit(fields)(mu, sigma, eps)
    s = _np(got_s).astype(np.float64)
    z = _np(got_z).astype(np.float64)
    mu64 = _np(NOISY_MU).astype(np.float64)
    log_two_pi = math.log(2.0 * math.pi)
    total = np.sum(((z - mu64) / s) ** 2 + 2.0 * np.log(s) + log_two_pi)
    _close(got_ld, -0.5 * total)
    _close(got_z, mu64 + s * _np(NOISY_EPS).astype(np.float64))
    _assert_f32_scalar(got_ld)


def test_noisy_latent_eager_empty_mismatch_raise_and_jit_uses_host_config() -> None:
    """Empty / length mismatch raise LatentError on host inputs; jit uses host config."""
    cfg = latent.LatentConfig()
    ok = np.array([0.1, 0.2], dtype=np.float32)
    with pytest.raises(latent.LatentError):
        latent.noisy_latent(np.array([], dtype=np.float32), ok, ok, cfg)
    with pytest.raises(latent.LatentError):
        latent.noisy_latent(ok, np.array([], dtype=np.float32), ok, cfg)
    with pytest.raises(latent.LatentError):
        latent.noisy_latent(ok, ok, np.array([], dtype=np.float32), cfg)
    with pytest.raises(latent.LatentError):
        latent.noisy_latent(
            ok,
            np.array([0.05, 0.05, 0.05], dtype=np.float32),
            ok,
            cfg,
        )

    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)

    def fields(m, s, e):
        return _noisy_fields(m, s, e, cfg)

    _got_mu, got_s, _got_e, got_z, got_ld = jax.jit(fields)(mu, sigma, eps)
    exp = ref.noisy_latent(_np(NOISY_MU), _np(NOISY_SIGMA), _np(NOISY_EPS), ref.LatentConfig())
    _close(got_z, exp.z)
    _close(got_s, exp.sigma)
    _close(got_ld, exp.log_density)
    _assert_f32_scalar(got_ld)


# ---------------------------------------------------------------------------
# 3. jax.grad through jitted noisy_latent.log_density
# ---------------------------------------------------------------------------


def test_noisy_latent_jitted_grad_matches_eager_grad() -> None:
    cfg = latent.LatentConfig()
    mu = _j32(NOISY_INTERIOR_MU)
    sigma = _j32(NOISY_INTERIOR_SIGMA)
    eps = _j32(NOISY_INTERIOR_EPS)

    def ld_eps(e):
        return _noisy_log_density(mu, sigma, e, cfg)

    def ld_sigma(s):
        return _noisy_log_density(mu, s, eps, cfg)

    def ld_mu(m):
        return _noisy_log_density(m, sigma, eps, cfg)

    jit_ge = jax.grad(jax.jit(ld_eps))(eps)
    eager_ge = jax.grad(ld_eps)(eps)
    _close(jit_ge, eager_ge)
    # log_density = -0.5 * sum(eps^2) - sum(log s) - const, so d/d(eps) = -eps.
    _close(jit_ge, -_np(NOISY_INTERIOR_EPS).astype(np.float64))

    jit_gs = jax.grad(jax.jit(ld_sigma))(sigma)
    eager_gs = jax.grad(ld_sigma)(sigma)
    _close(jit_gs, eager_gs)
    _close(jit_gs, -1.0 / _np(NOISY_INTERIOR_SIGMA).astype(np.float64))

    jit_gm = jax.grad(jax.jit(ld_mu))(mu)
    eager_gm = jax.grad(ld_mu)(mu)
    _close(jit_gm, eager_gm)
    _close(jit_gm, np.zeros_like(NOISY_INTERIOR_MU, dtype=np.float64))


def test_noisy_latent_jitted_grad_finite_difference_vs_reference() -> None:
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    mu_np = np.asarray(NOISY_INTERIOR_MU, dtype=np.float64)
    sigma_np = np.asarray(NOISY_INTERIOR_SIGMA, dtype=np.float64)
    eps_np = np.asarray(NOISY_INTERIOR_EPS, dtype=np.float64)
    v_np = np.asarray(NOISY_INTERIOR_V, dtype=np.float64)
    mu = _j32(mu_np)
    sigma = _j32(sigma_np)
    eps = _j32(eps_np)
    v = _j32(v_np)

    def ld_eps(e):
        return _noisy_log_density(mu, sigma, e, cfg_p)

    jit_g = jax.grad(jax.jit(ld_eps))(eps)
    directional = float(jnp.sum(jit_g * v))

    h = 1e-4

    def ref_ld(e: np.ndarray) -> float:
        return float(ref.noisy_latent(mu_np, sigma_np, e, cfg_r).log_density)

    fd = (ref_ld(eps_np + h * v_np) - ref_ld(eps_np - h * v_np)) / (2.0 * h)
    _close(directional, fd, **FD_TOL)


def test_noisy_latent_grad_through_clamp_stays_finite() -> None:
    """Zero / negative sigma clamp first; log_density grad must stay finite."""
    cfg = latent.LatentConfig()
    mu = _j32(np.array([0.1, -0.2, 0.3], dtype=np.float32))
    sigma = _j32(np.array([0.0, -1.0, 0.05], dtype=np.float32))
    eps = _j32(np.array([0.4, -0.5, 1.1], dtype=np.float32))

    def ld_sigma(s):
        return _noisy_log_density(mu, s, eps, cfg)

    jit_g = jax.grad(jax.jit(ld_sigma))(sigma)
    eager_g = jax.grad(ld_sigma)(sigma)
    _close(jit_g, eager_g)
    assert np.all(np.isfinite(_np(jit_g)))
    # Below min: clamped s is constant, grad 0. Interior: -1/s.
    _close(jit_g, np.array([0.0, 0.0, -1.0 / 0.05], dtype=np.float64))


# ---------------------------------------------------------------------------
# 1–2, 6, 7. jacobi_sweeps: identity, contraction, truncated, host ints
# ---------------------------------------------------------------------------


def test_jacobi_sweeps_jit_identity_matches_eager_and_reference() -> None:
    """Identity update leaves thoughts; jit matches eager and the NumPy ref."""
    thoughts = _j32(JACOBI_THOUGHTS)
    n_sweeps = 5
    truncated = 2

    def js(t):
        return latent.jacobi_sweeps(t, _identity, n_sweeps, truncated)

    got = jax.jit(js)(thoughts)
    eager = js(thoughts)
    exp = ref.jacobi_sweeps(_np(JACOBI_THOUGHTS), lambda x: x, n_sweeps, truncated)
    _close(got, eager)
    _close(got, exp)
    _close(got, JACOBI_THOUGHTS)
    _assert_f32_vector(got, JACOBI_THOUGHTS.shape)


def test_jacobi_sweeps_jit_contraction_matches_eager_and_reference() -> None:
    """Contraction update ``0.5 * x`` under jit matches eager and the NumPy ref."""
    thoughts = _j32(JACOBI_SCALE)
    n_sweeps = 4
    truncated = 2

    def js(t):
        return latent.jacobi_sweeps(t, _half, n_sweeps, truncated)

    got = jax.jit(js)(thoughts)
    eager = js(thoughts)
    exp = ref.jacobi_sweeps(
        _np(JACOBI_SCALE),
        lambda x: 0.5 * np.asarray(x),
        n_sweeps,
        truncated,
    )
    _close(got, eager)
    _close(got, exp)
    _close(got, (0.5**n_sweeps) * JACOBI_SCALE)
    _assert_f32_vector(got, JACOBI_SCALE.shape)


def test_jacobi_sweeps_jit_truncated_same_cpu_forward() -> None:
    """CPU forward is identical for truncated_sweeps=1 vs n_sweeps (same n_sweeps)."""
    thoughts = _j32(JACOBI_SCALE)
    n_sweeps = 4

    def js_one(t):
        return latent.jacobi_sweeps(t, _half, n_sweeps, 1)

    def js_all(t):
        return latent.jacobi_sweeps(t, _half, n_sweeps, n_sweeps)

    a = jax.jit(js_one)(thoughts)
    b = jax.jit(js_all)(thoughts)
    eager_a = js_one(thoughts)
    eager_b = js_all(thoughts)
    ar = ref.jacobi_sweeps(_np(JACOBI_SCALE), lambda x: 0.5 * np.asarray(x), n_sweeps, 1)
    br = ref.jacobi_sweeps(
        _np(JACOBI_SCALE),
        lambda x: 0.5 * np.asarray(x),
        n_sweeps,
        n_sweeps,
    )
    _close(a, b)
    _close(a, eager_a)
    _close(b, eager_b)
    _close(ar, br)
    _close(a, ar)
    _close(a, (0.5**n_sweeps) * JACOBI_SCALE)


def test_jacobi_sweeps_jit_static_argnums_host_ints() -> None:
    """n_sweeps and truncated_sweeps are host ints via static_argnums, not tracers."""
    thoughts = _j32(JACOBI_THOUGHTS)
    jitted = jax.jit(latent.jacobi_sweeps, static_argnums=(1, 2, 3))
    got = jitted(thoughts, _identity, 5, 2)
    eager = latent.jacobi_sweeps(thoughts, _identity, 5, 2)
    exp = ref.jacobi_sweeps(_np(JACOBI_THOUGHTS), lambda x: x, 5, 2)
    _close(got, eager)
    _close(got, exp)
    _close(got, JACOBI_THOUGHTS)
    _assert_f32_vector(got, JACOBI_THOUGHTS.shape)


# ---------------------------------------------------------------------------
# 7. shape / dtype across the three jitted entry points
# ---------------------------------------------------------------------------


def test_jacobi_clamp_noisy_jit_shape_dtype_float32() -> None:
    """Jitted log_density is float32 0-d; vectors keep input shape."""
    cfg = latent.LatentConfig()
    sigma = _j32(CLAMP_SIGMA)
    mu = _j32(NOISY_MU)
    nsigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)
    thoughts = _j32(JACOBI_THOUGHTS)

    def cs(s):
        return latent.clamp_sigma(s, cfg)

    def fields(m, s, e):
        return _noisy_fields(m, s, e, cfg)

    def js(t):
        return latent.jacobi_sweeps(t, _half, 3, 1)

    c = jax.jit(cs)(sigma)
    _got_mu, got_s, got_e, got_z, got_ld = jax.jit(fields)(mu, nsigma, eps)
    j = jax.jit(js)(thoughts)

    _assert_f32_vector(c, CLAMP_SIGMA.shape)
    _assert_f32_vector(got_z, NOISY_MU.shape)
    _assert_f32_vector(got_s, NOISY_SIGMA.shape)
    _assert_f32_vector(got_e, NOISY_EPS.shape)
    _assert_f32_scalar(got_ld)
    _assert_f32_vector(j, JACOBI_THOUGHTS.shape)


def test_jit_lockstep_vs_reference_random_small_vectors() -> None:
    """Property: jitted prod vs independent NumPy ref on random small vectors."""
    rng = np.random.default_rng(20260323)
    cfg_p = latent.LatentConfig()
    cfg_r = ref.LatentConfig()
    j_clamp = jax.jit(latent.clamp_sigma, static_argnums=(1,))
    j_noisy = jax.jit(_noisy_fields, static_argnums=(3,))

    def js(t):
        return latent.jacobi_sweeps(t, _half, 3, 2)

    j_jacobi = jax.jit(js)

    for n in range(1, 6):
        mu_np = rng.normal(size=(n,)).astype(np.float32)
        sigma_np = rng.uniform(1e-5, 0.5, size=(n,)).astype(np.float32)
        eps_np = rng.normal(size=(n,)).astype(np.float32)
        thoughts_np = rng.normal(size=(n, max(n, 2))).astype(np.float32)

        got_c = j_clamp(_j32(sigma_np), cfg_p)
        _close(got_c, ref.clamp_sigma(sigma_np, cfg_r))
        _assert_f32_vector(got_c, sigma_np.shape)

        _gm, got_s, _ge, got_z, got_ld = j_noisy(
            _j32(mu_np), _j32(sigma_np), _j32(eps_np), cfg_p
        )
        exp = ref.noisy_latent(mu_np, sigma_np, eps_np, cfg_r)
        _close(got_z, exp.z)
        _close(got_s, exp.sigma)
        _close(got_ld, exp.log_density)
        _assert_f32_scalar(got_ld)

        got_j = j_jacobi(_j32(thoughts_np))
        exp_j = ref.jacobi_sweeps(
            thoughts_np, lambda x: 0.5 * np.asarray(x), 3, 2
        )
        _close(got_j, exp_j)
        _assert_f32_vector(got_j, thoughts_np.shape)
