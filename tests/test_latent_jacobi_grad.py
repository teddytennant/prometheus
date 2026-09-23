"""Truncated-backprop lock for ``model.latent.jacobi_sweeps`` (spec 4.3).

Forward value is the state after all ``n_sweeps`` updates, for every legal
``truncated_sweeps``. Changing ``truncated_sweeps`` must not change that value.

Backward flows only through the last ``truncated_sweeps`` updates. The state
entering that window (the input to update index ``n_sweeps - truncated_sweeps``,
0-based) is a constant: stop-gradient. Earlier updates do not receive gradient.
``truncated_sweeps == n_sweeps`` is the full unroll.

``n_sweeps`` and ``truncated_sweeps`` are Python ints, closed over by the
differentiated function (and, in the jit check, also passed as
``static_argnums``). The update used here is JAX-traceable and does not call
``numpy.asarray``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import model.latent as latent
from tests.reference import latent as ref

TOL = {"rtol": 1e-5, "atol": 1e-5}

# (2, 3), float32, finite, nonzero. Values are dyadic so 0.5**4 * thoughts is exact.
THOUGHTS = np.array(
    [[1.0, -2.0, 0.5], [0.25, 3.0, -1.5]],
    dtype=np.float32,
)
N_SWEEPS = 4
TRUNCATED = 2


def _half(x):
    """Contraction. The only path from the sweep input; does not close over thoughts."""
    return 0.5 * x


def _close(got: object, exp: object) -> None:
    np.testing.assert_allclose(np.asarray(got), np.asarray(exp), **TOL)


def _loss(thoughts, n_sweeps: int, truncated_sweeps: int):
    return jnp.sum(latent.jacobi_sweeps(thoughts, _half, n_sweeps, truncated_sweeps))


def _ref_grad(n_sweeps: int, truncated_sweeps: int) -> np.ndarray:
    return ref.jacobi_sweeps_truncated_grad(THOUGHTS, _half, n_sweeps, truncated_sweeps)


def _closed_form_full_grad(n_sweeps: int) -> np.ndarray:
    """d(sum((1/2)**n * thoughts))/d(thoughts) = (1/2)**n, everywhere."""
    scale = np.float64(0.5) ** int(n_sweeps)
    return np.full(THOUGHTS.shape, scale, dtype=np.float64)


def test_truncated_window_gradient_is_zero() -> None:
    """Last 2 of 4 sweeps: entrance is constant, so d(sum)/d(thoughts) is 0.

    ``update`` does not close over ``thoughts``. A backward pass that still
    walks the earlier sweeps (full unroll) yields ``0.5**n_sweeps``, not zero.
    """
    assert THOUGHTS.shape == (2, 3)
    assert THOUGHTS.dtype == np.float32
    assert np.all(np.isfinite(THOUGHTS))
    assert np.all(THOUGHTS != 0.0)

    thoughts = jnp.asarray(THOUGHTS)
    assert thoughts.dtype == jnp.float32

    exp = _ref_grad(N_SWEEPS, TRUNCATED)
    np.testing.assert_array_equal(exp, np.zeros(THOUGHTS.shape, dtype=np.float64))

    def loss(t):
        return _loss(t, N_SWEEPS, TRUNCATED)

    got = jax.grad(loss)(thoughts)
    got_np = np.asarray(got)
    # Defect lock. Do not relax this to a nonzero full-unroll gradient.
    np.testing.assert_array_equal(
        got_np,
        np.zeros(got_np.shape, dtype=got_np.dtype),
        err_msg=(
            "stop-gradient the state entering the last truncated_sweeps "
            "updates; d(loss)/d(thoughts) must be all zeros"
        ),
    )
    _close(got_np, exp)

    jit_got = jax.jit(jax.grad(loss))(thoughts)
    _close(jit_got, got)

    jitted = jax.jit(latent.jacobi_sweeps, static_argnums=(1, 2, 3))

    def loss_static(t):
        return jnp.sum(jitted(t, _half, N_SWEEPS, TRUNCATED))

    static_got = jax.grad(loss_static)(thoughts)
    _close(static_got, got)
    _close(jax.jit(jax.grad(loss_static))(thoughts), got)


def test_full_unroll_gradient_matches_reference() -> None:
    """truncated_sweeps == n_sweeps is a full unroll: nonzero and exact."""
    thoughts = jnp.asarray(THOUGHTS)
    for n_sweeps in (1, N_SWEEPS):
        def loss(t, n=n_sweeps):
            return _loss(t, n, n)

        got = jax.grad(loss)(thoughts)
        exp = _ref_grad(n_sweeps, n_sweeps)
        closed = _closed_form_full_grad(n_sweeps)
        _close(exp, closed)
        _close(got, closed)
        _close(got, exp)
        assert got.shape == thoughts.shape
        assert got.dtype == jnp.float32
        assert np.any(np.asarray(got) != 0.0)

        jit_got = jax.jit(jax.grad(loss))(thoughts)
        _close(jit_got, got)
        _close(jax.jit(jax.grad(loss))(thoughts), closed)


def test_forward_independent_of_truncated_sweeps() -> None:
    """truncated_sweeps=1 and truncated_sweeps=n_sweeps return the same array."""
    thoughts = jnp.asarray(THOUGHTS)
    one = latent.jacobi_sweeps(thoughts, _half, N_SWEEPS, 1)
    full = latent.jacobi_sweeps(thoughts, _half, N_SWEEPS, N_SWEEPS)
    _close(one, full)

    scale = np.float32(0.5) ** N_SWEEPS
    expected = scale * THOUGHTS
    _close(one, expected)
    _close(full, expected)
    _close(one, ref.jacobi_sweeps(THOUGHTS, _half, N_SWEEPS, 1))
    _close(full, ref.jacobi_sweeps(THOUGHTS, _half, N_SWEEPS, N_SWEEPS))

    for truncated in range(1, N_SWEEPS + 1):
        got = latent.jacobi_sweeps(thoughts, _half, N_SWEEPS, truncated)
        _close(got, expected)
        assert np.asarray(got).shape == THOUGHTS.shape
        assert jnp.asarray(got).dtype == jnp.float32

    def js_one(t):
        return latent.jacobi_sweeps(t, _half, N_SWEEPS, 1)

    def js_full(t):
        return latent.jacobi_sweeps(t, _half, N_SWEEPS, N_SWEEPS)

    jit_one = jax.jit(js_one)(thoughts)
    jit_full = jax.jit(js_full)(thoughts)
    _close(jit_one, one)
    _close(jit_full, full)
    _close(jit_one, jit_full)
    _close(jit_one, expected)
