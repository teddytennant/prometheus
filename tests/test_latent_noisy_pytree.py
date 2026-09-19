"""I5-noisy-pytree oracle: ``jax.jit(noisy_latent)`` must return a pytree ``NoisyLatent``.

These tests call the public ``model.latent.noisy_latent`` under ``jax.jit``
directly (no field-tuple wrapper) and compare the returned ``NoisyLatent`` to
eager JAX and ``tests.reference.latent`` at 1e-5. They must fail on the current
unregistered dataclass (TypeError / "not a valid JAX type", or flatten treating
the whole object as one leaf) and pass once ``NoisyLatent`` is a
``jax.tree_util`` registered dataclass with every field as data and no meta
fields.

Coverage
--------
1. ``NoisyLatent`` is a registered pytree: flatten leaves are ``mu`` / ``sigma``
   / ``eps`` / ``z`` / ``log_density`` as array or scalar values (not the whole
   object as one leaf, not arrays stuffed into aux). ``tree_map`` on floats
   doubles those fields. ``jax.jit(lambda n: n)(noisy)`` roundtrips.
2. Constructing ``NoisyLatent`` from traced array args is legal under jit
   (the jitted function returns the dataclass, not a pulled-out field).
3. ``jax.jit(noisy_latent)(mu, sigma, eps)`` vs eager at 1e-5 on ``z``,
   ``sigma``, ``log_density``. Direct call; ``LatentConfig`` stays host-side.
   Numpy eager ``noisy_latent`` still returns ``NoisyLatent``.
4. Jitted vs ``tests.reference.latent.noisy_latent`` at 1e-5 (no production
   math copied into this file).
5. ``jax.grad`` through jitted ``noisy_latent(...).log_density`` vs eager
   ``jax.grad`` at 1e-5.

Do not mark gpu. Do not jit ``validate_latent_config``. Tests import
``model.latent``; production must not import ``tests``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import model.latent as latent
from tests.reference import latent as ref

TOL = dict(rtol=1e-5, atol=1e-5)

DATA_FIELDS = ("mu", "sigma", "eps", "z", "log_density")

# Same short vector as the jacobi jit oracle (clamp hits min and max).
NOISY_MU = (0.2, -0.5, 1.0)
NOISY_SIGMA = (1e-6, 0.05, 5.0)
NOISY_EPS = (0.3, -1.2, 0.7)


def _close(got: object, exp: object) -> None:
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64), exp, **TOL)


def _j32(xs: object):
    return jnp.asarray(np.asarray(xs, dtype=np.float32))


def _np32(xs: object) -> np.ndarray:
    return np.asarray(xs, dtype=np.float32)


def _leaf_matches(leaf: object, value: object) -> bool:
    """True if ``value`` appears as a numeric pytree leaf, not a NoisyLatent object."""
    if type(leaf) is latent.NoisyLatent:
        return False
    got = np.asarray(leaf)
    target = np.asarray(value)
    if got.shape != target.shape:
        return False
    if not np.issubdtype(got.dtype, np.number) or not np.issubdtype(target.dtype, np.number):
        return False
    return np.allclose(got.astype(np.float64), target.astype(np.float64), **TOL)


def _double_floats(x: object) -> object:
    if hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.floating):
        return x * 2
    return x


def _assert_noisy_fields(got: object, exp: object) -> None:
    _close(got.z, exp.z)  # type: ignore[attr-defined]
    _close(got.sigma, exp.sigma)  # type: ignore[attr-defined]
    _close(got.log_density, exp.log_density)  # type: ignore[attr-defined]
    _close(got.mu, exp.mu)  # type: ignore[attr-defined]
    _close(got.eps, exp.eps)  # type: ignore[attr-defined]
    assert np.ndim(got.log_density) == 0  # type: ignore[attr-defined]


def _closed_noisy(cfg: latent.LatentConfig):
    """Three array args; config stays host-side. Returns the dataclass."""

    def noisy_latent(mu, sigma, eps):
        return latent.noisy_latent(mu, sigma, eps, cfg)

    return noisy_latent


# ---------------------------------------------------------------------------
# 1. NoisyLatent is a registered pytree
# ---------------------------------------------------------------------------


def test_noisy_latent_is_registered_jax_pytree() -> None:
    """Flatten leaves are the five data fields; tree_map doubles; identity jit.

    A dummy register that treats the whole object as one leaf, or stuffs arrays
    into aux, fails. ``jax.jit(lambda n: n)(noisy)`` must roundtrip.
    """
    cfg = latent.LatentConfig()
    noisy = latent.noisy_latent(_np32(NOISY_MU), _np32(NOISY_SIGMA), _np32(NOISY_EPS), cfg)
    assert type(noisy) is latent.NoisyLatent

    leaves, treedef = jax.tree_util.tree_flatten(noisy)
    assert not any(type(leaf) is latent.NoisyLatent for leaf in leaves), (
        "NoisyLatent must not be a pytree leaf; jax.jit would treat it as an "
        "abstract array. Register it so mu/sigma/eps/z/log_density are data leaves."
    )
    assert len(leaves) == len(DATA_FIELDS), (
        "NoisyLatent must flatten to one leaf per data field "
        f"(got {len(leaves)} leaves, expected {len(DATA_FIELDS)})"
    )
    for name in DATA_FIELDS:
        assert any(_leaf_matches(leaf, getattr(noisy, name)) for leaf in leaves), (
            f"{name} must be a pytree data-field leaf (value match); a dummy "
            "register or aux-only flatten is not enough"
        )

    node = treedef.node_data()
    assert node is not None, (
        "NoisyLatent must be a registered pytree node, not a generic leaf"
    )
    cls, meta = node
    assert cls is latent.NoisyLatent
    assert meta == (), (
        "NoisyLatent has no meta fields; arrays must not be stuffed into aux"
    )

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is latent.NoisyLatent
    _assert_noisy_fields(rebuilt, noisy)

    mapped = jax.tree_util.tree_map(_double_floats, noisy)
    assert type(mapped) is latent.NoisyLatent
    for name in DATA_FIELDS:
        _close(getattr(mapped, name), np.asarray(getattr(noisy, name), dtype=np.float64) * 2.0)

    roundtrip = jax.jit(lambda n: n)(noisy)
    assert type(roundtrip) is latent.NoisyLatent
    _assert_noisy_fields(roundtrip, noisy)


# ---------------------------------------------------------------------------
# 2. Construct from traced args under jit
# ---------------------------------------------------------------------------


def test_constructing_noisy_latent_from_traced_args_is_legal_under_jit() -> None:
    """Jitted rebuild must return a NoisyLatent, not a pulled-out field."""
    cfg = latent.LatentConfig()
    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)
    eager = latent.noisy_latent(mu, sigma, eps, cfg)

    def rebuild(mu, sigma, eps, z, log_density):
        return latent.NoisyLatent(mu=mu, sigma=sigma, eps=eps, z=z, log_density=log_density)

    got = jax.jit(rebuild)(eager.mu, eager.sigma, eager.eps, eager.z, eager.log_density)
    assert type(got) is latent.NoisyLatent
    _assert_noisy_fields(got, eager)


# ---------------------------------------------------------------------------
# 3. jax.jit(noisy_latent) vs eager
# ---------------------------------------------------------------------------


def test_jax_jit_noisy_latent_matches_eager() -> None:
    """Direct ``jax.jit(noisy_latent)(mu, sigma, eps)`` matches eager at 1e-5.

    Numpy eager ``noisy_latent`` still returns ``NoisyLatent``. Config is
    host-side (closed over, or ``static_argnums=(3,)``). No field-tuple wrapper.
    """
    cfg = latent.LatentConfig()
    mu_np = np.asarray(NOISY_MU, dtype=np.float64)
    sigma_np = np.asarray(NOISY_SIGMA, dtype=np.float64)
    eps_np = np.asarray(NOISY_EPS, dtype=np.float64)
    np_out = latent.noisy_latent(mu_np, sigma_np, eps_np, cfg)
    assert type(np_out) is latent.NoisyLatent
    assert np.ndim(np_out.log_density) == 0

    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)
    eager = latent.noisy_latent(mu, sigma, eps, cfg)

    noisy_latent = _closed_noisy(cfg)
    got = jax.jit(noisy_latent)(mu, sigma, eps)
    assert type(got) is latent.NoisyLatent
    _close(got.z, eager.z)
    _close(got.sigma, eager.sigma)
    _close(got.log_density, eager.log_density)
    assert got.z.shape == eager.z.shape
    assert got.sigma.shape == eager.sigma.shape
    assert np.ndim(got.log_density) == 0

    got_static = jax.jit(latent.noisy_latent, static_argnums=(3,))(mu, sigma, eps, cfg)
    assert type(got_static) is latent.NoisyLatent
    _close(got_static.z, eager.z)
    _close(got_static.sigma, eager.sigma)
    _close(got_static.log_density, eager.log_density)


# ---------------------------------------------------------------------------
# 4. jitted vs independent NumPy reference
# ---------------------------------------------------------------------------


def test_jax_jit_noisy_latent_matches_reference() -> None:
    """Jitted production ``noisy_latent`` matches ``tests.reference.latent`` at 1e-5."""
    cfg = latent.LatentConfig()
    ref_cfg = ref.LatentConfig()
    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)
    expected = ref.noisy_latent(
        np.asarray(NOISY_MU, dtype=np.float64),
        np.asarray(NOISY_SIGMA, dtype=np.float64),
        np.asarray(NOISY_EPS, dtype=np.float64),
        ref_cfg,
    )

    got = jax.jit(_closed_noisy(cfg))(mu, sigma, eps)
    assert type(got) is latent.NoisyLatent
    _close(got.z, expected.z)
    _close(got.sigma, expected.sigma)
    _close(got.log_density, expected.log_density)

    got_static = jax.jit(latent.noisy_latent, static_argnums=(3,))(mu, sigma, eps, cfg)
    _close(got_static.z, expected.z)
    _close(got_static.sigma, expected.sigma)
    _close(got_static.log_density, expected.log_density)


def test_jax_jit_noisy_latent_lockstep_vs_eager_and_reference() -> None:
    """Random 1-D vectors: jitted NoisyLatent matches eager JAX and the NumPy ref."""
    rng = np.random.default_rng(20260328)
    cfg = latent.LatentConfig()
    ref_cfg = ref.LatentConfig()
    jitted = jax.jit(_closed_noisy(cfg))
    for n in (1, 2, 3, 5, 8):
        mu_np = rng.normal(size=(n,)).astype(np.float32)
        sigma_np = rng.uniform(1e-5, 2.0, size=(n,)).astype(np.float32)
        eps_np = rng.normal(size=(n,)).astype(np.float32)
        mu, sigma, eps = jnp.asarray(mu_np), jnp.asarray(sigma_np), jnp.asarray(eps_np)
        eager = latent.noisy_latent(mu, sigma, eps, cfg)
        got = jitted(mu, sigma, eps)
        expected = ref.noisy_latent(
            mu_np.astype(np.float64),
            sigma_np.astype(np.float64),
            eps_np.astype(np.float64),
            ref_cfg,
        )
        assert type(got) is latent.NoisyLatent
        _close(got.z, eager.z)
        _close(got.sigma, eager.sigma)
        _close(got.log_density, eager.log_density)
        _close(got.z, expected.z)
        _close(got.sigma, expected.sigma)
        _close(got.log_density, expected.log_density)
        assert got.z.shape == (n,)
        assert got.sigma.shape == (n,)
        assert np.ndim(got.log_density) == 0


# ---------------------------------------------------------------------------
# 5. jax.grad through jitted noisy_latent(...).log_density
# ---------------------------------------------------------------------------


def test_jax_grad_through_jitted_noisy_latent_log_density() -> None:
    """``jax.grad`` through jitted ``noisy_latent(...).log_density`` vs eager at 1e-5.

    Grad walks the returned pytree (``.log_density`` after ``jax.jit(noisy_latent)``),
    not a wrapper that pulls the scalar out inside the jitted function.
    """
    cfg = latent.LatentConfig()
    mu = _j32(NOISY_MU)
    sigma = _j32(NOISY_SIGMA)
    eps = _j32(NOISY_EPS)
    jitted = jax.jit(latent.noisy_latent, static_argnums=(3,))

    def jitted_log_density(mu, sigma, eps):
        return jitted(mu, sigma, eps, cfg).log_density

    def eager_log_density(mu, sigma, eps):
        return latent.noisy_latent(mu, sigma, eps, cfg).log_density

    for argnums in (0, 1, 2):
        jit_g = jax.grad(jitted_log_density, argnums=argnums)(mu, sigma, eps)
        eager_g = jax.grad(eager_log_density, argnums=argnums)(mu, sigma, eps)
        _close(jit_g, eager_g)
        assert np.asarray(jit_g).shape == np.asarray(eager_g).shape
