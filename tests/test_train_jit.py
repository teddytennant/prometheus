"""A2-opt-jit oracle: ``jax.jit`` of MuonClip / AdamW ops must match eager at 1e-5.

These tests call the public ``train.*`` signatures with ``jnp`` arrays and
Python scalars held static (closed over or ``static_argnums`` /
``static_argnames``). They must fail on the current numpy.asarray /
Python ``float()`` implementations (``TracerArrayConversionError`` or
``ConcretizationTypeError``) and pass once the four functions stay in JAX.

Coverage
--------
jit vs eager (1e-5): newton_schulz, muon_update, adamw_update, qk_clip.
jit vs tests/reference/train.py (1e-5): newton_schulz and muon_update on
    a 3x2 and a 4x4 matrix (independent of production math).
jax.grad through jitted newton_schulz vs eager jax.grad at 1e-5; cheap
    finite-difference check vs the NumPy reference.
qk_clip under jit: max |q k^T| <= max_logit; identity when already under cap.
adamw_update under jit: decoupled wd; 1-based bias correction; match eager.
shape / dtype: float32, same shape as the inputs.

Do not mark gpu. Python scalars (steps, lr, ns_steps, step, max_logit, …)
stay host-side. Expected values come from tests/reference/, not from
copying production.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import train
from tests.reference import train as ref

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)

# Deterministic matrices. 3x2 is the Newton–Schulz golden; 4x4 is square.
NS_3X2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
NS_4X4 = np.array(
    [
        [1.0, 0.2, -0.3, 0.4],
        [0.5, 1.1, 0.0, -0.2],
        [-0.4, 0.3, 0.9, 0.1],
        [0.2, -0.5, 0.4, 1.2],
    ],
    dtype=np.float32,
)
MUON_3X2 = np.array([[0.5, -0.25], [0.1, 0.8], [-0.3, 0.4]], dtype=np.float32)
QK_Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
QK_K = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
QK_MAX_LOGIT = 1.5
ADAMW_KW = dict(lr=2e-4, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1)
MUON_KW = dict(lr=0.02, momentum_coeff=0.95, ns_steps=5)


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _j32(x: object) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _close(got: object, exp: object, **kwargs: float) -> None:
    tol = dict(TOL)
    tol.update(kwargs)
    np.testing.assert_allclose(_np(got), _np(exp), **tol)


# ---------------------------------------------------------------------------
# newton_schulz
# ---------------------------------------------------------------------------
def test_newton_schulz_jit_matches_eager() -> None:
    """jax.jit(newton_schulz) with steps closed over matches eager at 1e-5."""
    matrix = _j32(NS_3X2)
    steps = 5

    def ns(m):
        return train.newton_schulz(m, steps)

    eager = train.newton_schulz(matrix, steps)
    got = jax.jit(ns)(matrix)
    _close(got, eager)
    assert tuple(got.shape) == tuple(matrix.shape) == (3, 2)
    assert _np(got).dtype == np.float32


def test_newton_schulz_jit_static_argnums_matches_eager() -> None:
    """jax.jit(..., static_argnums=(1,)) matches eager on the same jnp array."""
    matrix = _j32(NS_4X4)
    jitted = jax.jit(train.newton_schulz, static_argnums=(1,))
    got = jitted(matrix, 5)
    eager = train.newton_schulz(matrix, 5)
    _close(got, eager)
    assert tuple(got.shape) == (4, 4)
    assert _np(got).dtype == np.float32


def test_newton_schulz_jit_matches_reference_3x2_and_4x4() -> None:
    """Jitted Newton–Schulz matches the NumPy oracle (not production math)."""
    jitted = jax.jit(train.newton_schulz, static_argnums=(1,))
    for g in (NS_3X2, NS_4X4):
        got = jitted(_j32(g), 5)
        _close(got, ref.newton_schulz(g, steps=5))
        assert tuple(got.shape) == g.shape
        assert _np(got).dtype == np.float32


def test_newton_schulz_jit_grad_matches_eager() -> None:
    """jax.grad of sum(outputs) through jitted newton_schulz matches eager at 1e-5."""
    matrix = _j32(NS_3X2)
    steps = 5
    jitted_ns = jax.jit(lambda m: train.newton_schulz(m, steps))

    def jitted_loss(m):
        return jnp.sum(jitted_ns(m))

    def eager_loss(m):
        return jnp.sum(train.newton_schulz(m, steps))

    jit_g = jax.grad(jitted_loss)(matrix)
    eager_g = jax.grad(eager_loss)(matrix)
    _close(jit_g, eager_g)
    assert tuple(jit_g.shape) == tuple(matrix.shape)
    assert _np(jit_g).dtype == np.float32


def test_newton_schulz_jit_grad_fd_vs_reference() -> None:
    """Directional finite difference of the NumPy oracle vs jitted jax.grad."""
    g_np = NS_3X2
    v_np = np.array([[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1]], dtype=np.float32)
    eps = np.float32(1e-3)
    steps = 1

    def ref_sum(z: np.ndarray) -> float:
        return float(np.sum(ref.newton_schulz(z, steps=steps)))

    fd = (ref_sum(g_np + eps * v_np) - ref_sum(g_np - eps * v_np)) / (2.0 * float(eps))

    jitted_ns = jax.jit(lambda m: train.newton_schulz(m, steps))
    jit_g = jax.grad(lambda m: jnp.sum(jitted_ns(m)))(_j32(g_np))
    analytic = float(np.sum(_np(jit_g) * v_np))
    np.testing.assert_allclose(analytic, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# muon_update
# ---------------------------------------------------------------------------
def test_muon_update_jit_matches_eager() -> None:
    """jax.jit of muon_update with lr / momentum / ns_steps closed over."""
    grad = _j32(MUON_3X2)
    momentum = jnp.zeros_like(grad)

    def mu(g, m):
        return train.muon_update(g, m, **MUON_KW)

    eager_d, eager_m = train.muon_update(grad, momentum, **MUON_KW)
    jit_d, jit_m = jax.jit(mu)(grad, momentum)
    _close(jit_d, eager_d)
    _close(jit_m, eager_m)
    assert tuple(jit_d.shape) == tuple(grad.shape) == (3, 2)
    assert tuple(jit_m.shape) == (3, 2)
    assert _np(jit_d).dtype == np.float32
    assert _np(jit_m).dtype == np.float32


def test_muon_update_jit_static_argnames_matches_eager() -> None:
    """static_argnames for the keyword-only Python scalars matches eager."""
    grad = _j32(NS_4X4)
    momentum = jnp.zeros_like(grad)
    jitted = jax.jit(
        train.muon_update,
        static_argnames=("lr", "momentum_coeff", "ns_steps"),
    )
    jit_d, jit_m = jitted(grad, momentum, **MUON_KW)
    eager_d, eager_m = train.muon_update(grad, momentum, **MUON_KW)
    _close(jit_d, eager_d)
    _close(jit_m, eager_m)


def test_muon_update_jit_matches_reference_3x2_and_4x4() -> None:
    """Jitted muon_update matches the NumPy oracle on 3x2 and 4x4."""
    jitted = jax.jit(
        train.muon_update,
        static_argnames=("lr", "momentum_coeff", "ns_steps"),
    )
    for g_np in (MUON_3X2, NS_4X4):
        grad = _j32(g_np)
        momentum = jnp.zeros_like(grad)
        jit_d, jit_m = jitted(grad, momentum, **MUON_KW)
        exp_d, exp_m = ref.muon_update(g_np, np.zeros_like(g_np), **MUON_KW)
        _close(jit_d, exp_d)
        _close(jit_m, exp_m)
        assert tuple(jit_d.shape) == g_np.shape
        assert _np(jit_d).dtype == np.float32
        assert _np(jit_m).dtype == np.float32


# ---------------------------------------------------------------------------
# qk_clip
# ---------------------------------------------------------------------------
def test_qk_clip_jit_matches_eager() -> None:
    """jax.jit(qk_clip) with max_logit static matches eager at 1e-5."""
    q = _j32(QK_Q)
    k = _j32(QK_K)
    jitted = jax.jit(train.qk_clip, static_argnums=(2,))
    jq, jk = jitted(q, k, QK_MAX_LOGIT)
    eq, ek = train.qk_clip(q, k, QK_MAX_LOGIT)
    _close(jq, eq)
    _close(jk, ek)
    assert tuple(jq.shape) == tuple(q.shape)
    assert tuple(jk.shape) == tuple(k.shape)
    assert _np(jq).dtype == np.float32
    assert _np(jk).dtype == np.float32


def test_qk_clip_jit_caps_max_logit() -> None:
    """After a jitted clip, max |q k^T| <= max_logit (same property as eager)."""
    q = _j32(QK_Q)
    k = _j32(QK_K)
    assert float(np.max(np.abs(QK_Q @ QK_K.T))) > QK_MAX_LOGIT

    def clip(qv, kv):
        return train.qk_clip(qv, kv, QK_MAX_LOGIT)

    jq, jk = jax.jit(clip)(q, k)
    scores = _np(jq) @ _np(jk).T
    assert float(np.max(np.abs(scores))) <= QK_MAX_LOGIT + 1e-5
    eq, ek = ref.qk_clip(QK_Q, QK_K, QK_MAX_LOGIT)
    _close(jq, eq)
    _close(jk, ek)


def test_qk_clip_jit_identity_when_already_under_cap() -> None:
    """When already under the cap, jitted qk_clip leaves q and k unchanged."""
    q = _j32(QK_Q) * np.float32(0.01)
    k = _j32(QK_K) * np.float32(0.01)
    scores = _np(q) @ _np(k).T
    assert float(np.max(np.abs(scores))) < QK_MAX_LOGIT

    def clip(qv, kv):
        return train.qk_clip(qv, kv, QK_MAX_LOGIT)

    jq, jk = jax.jit(clip)(q, k)
    _close(jq, q)
    _close(jk, k)


def test_qk_clip_jit_batched_caps_max_logit() -> None:
    """Batched (..., n, d) q/k under jit still satisfy the logit cap."""
    rng = np.random.default_rng(2)
    q_np = rng.standard_normal((2, 6, 4)).astype(np.float32)
    k_np = rng.standard_normal((2, 6, 4)).astype(np.float32)
    tau = 2.0

    def clip(qv, kv):
        return train.qk_clip(qv, kv, tau)

    jq, jk = jax.jit(clip)(_j32(q_np), _j32(k_np))
    scores = np.einsum("...id,...jd->...ij", _np(jq), _np(jk))
    assert float(np.max(np.abs(scores))) <= tau + 1e-5
    eq, ek = ref.qk_clip(q_np, k_np, tau)
    _close(jq, eq)
    _close(jk, ek)


# ---------------------------------------------------------------------------
# adamw_update
# ---------------------------------------------------------------------------
def test_adamw_update_jit_matches_eager() -> None:
    """jax.jit of adamw_update with Python scalars closed over matches eager."""
    p = _j32([1.0])
    g = _j32([0.5])
    m = jnp.zeros(1, dtype=jnp.float32)
    v = jnp.zeros(1, dtype=jnp.float32)

    def aw(param, grad, mm, vv):
        return train.adamw_update(param, grad, mm, vv, step=1, **ADAMW_KW)

    ep, em, ev = train.adamw_update(p, g, m, v, step=1, **ADAMW_KW)
    jp, jm, jv = jax.jit(aw)(p, g, m, v)
    _close(jp, ep)
    _close(jm, em)
    _close(jv, ev)
    assert tuple(jp.shape) == (1,)
    assert _np(jp).dtype == np.float32
    assert _np(jm).dtype == np.float32
    assert _np(jv).dtype == np.float32


def test_adamw_update_jit_static_argnames_matches_eager_and_reference() -> None:
    """static_argnames path matches eager and the NumPy oracle on a 4x4."""
    rng = np.random.default_rng(3)
    p_np = rng.standard_normal((4, 4)).astype(np.float32)
    g_np = rng.standard_normal((4, 4)).astype(np.float32)
    z_np = np.zeros_like(p_np)
    jitted = jax.jit(
        train.adamw_update,
        static_argnames=("lr", "beta1", "beta2", "eps", "wd", "step"),
    )
    jp, jm, jv = jitted(_j32(p_np), _j32(g_np), _j32(z_np), _j32(z_np), step=1, **ADAMW_KW)
    ep, em, ev = train.adamw_update(
        _j32(p_np), _j32(g_np), _j32(z_np), _j32(z_np), step=1, **ADAMW_KW
    )
    rp, rm, rv = ref.adamw_update(p_np, g_np, z_np, z_np, step=1, **ADAMW_KW)
    _close(jp, ep)
    _close(jm, em)
    _close(jv, ev)
    _close(jp, rp)
    _close(jm, rm)
    _close(jv, rv)
    assert tuple(jp.shape) == (4, 4)
    assert _np(jp).dtype == np.float32


def test_adamw_update_jit_decoupled_wd() -> None:
    """Under jit, decoupled wd still shrinks p when g = m = v = 0."""
    p0 = _j32([1.0, -2.0])
    z = jnp.zeros_like(p0)

    def aw(param, grad, mm, vv):
        return train.adamw_update(
            param, grad, mm, vv, lr=0.1, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1, step=1
        )

    jp, jm, jv = jax.jit(aw)(p0, z, z, z)
    _close(jp, _np(p0) * np.float32(1.0 - 0.1 * 0.1))
    _close(jm, 0.0)
    _close(jv, 0.0)
    ep, _, _ = train.adamw_update(
        p0, z, z, z, lr=0.1, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1, step=1
    )
    _close(jp, ep)


def test_adamw_update_jit_one_based_bias_correction() -> None:
    """1-based bias correction: jitted step=1 differs from step=2 and matches eager."""
    rng = np.random.default_rng(3)
    p_np = rng.standard_normal((4, 4)).astype(np.float32)
    g_np = rng.standard_normal((4, 4)).astype(np.float32)
    z = jnp.zeros((4, 4), dtype=jnp.float32)
    p = _j32(p_np)
    g = _j32(g_np)

    def aw_at(step: int):
        def aw(param, grad, mm, vv):
            return train.adamw_update(param, grad, mm, vv, step=step, **ADAMW_KW)

        return jax.jit(aw)

    p1, m1, v1 = aw_at(1)(p, g, z, z)
    p2, m2, v2 = aw_at(2)(p, g, z, z)
    e1 = train.adamw_update(p, g, z, z, step=1, **ADAMW_KW)
    e2 = train.adamw_update(p, g, z, z, step=2, **ADAMW_KW)
    r1 = ref.adamw_update(p_np, g_np, np.zeros_like(p_np), np.zeros_like(p_np), step=1, **ADAMW_KW)
    r2 = ref.adamw_update(p_np, g_np, np.zeros_like(p_np), np.zeros_like(p_np), step=2, **ADAMW_KW)
    _close(p1, e1[0])
    _close(m1, e1[1])
    _close(v1, e1[2])
    _close(p2, e2[0])
    _close(m2, e2[1])
    _close(v2, e2[2])
    _close(p1, r1[0])
    _close(p2, r2[0])
    # m, v do not depend on t; only the bias-corrected update (hence p) does.
    _close(m1, m2)
    _close(v1, v2)
    assert not np.allclose(_np(p1), _np(p2), atol=1e-8)
