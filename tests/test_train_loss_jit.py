"""A2-loss-jit oracle: ``jax.jit`` of CE / MTP / z-loss / soft-cap must match eager at 1e-5.

These tests call the public ``train.*`` signatures with ``jnp`` arrays and
Python scalars held static (``cap`` closed over or ``static_argnums``;
``TrainConfig`` closed over or static). They must fail on the current
``numpy.asarray`` / Python ``float()`` / ``np.tanh`` implementations
(``TracerArrayConversionError`` or ``ConcretizationTypeError``) and pass
once the five functions stay in JAX.

Coverage
--------
jit vs eager (1e-5): soft_cap, cross_entropy, mtp_loss, z_loss, total_loss.
jit vs tests/reference/train.py (1e-5): independent of production math.
jax.grad through jitted cross_entropy vs eager jax.grad at 1e-5; cheap
    finite-difference check vs the NumPy reference.
empty mask: CE returns 0 under jit (no Python branch on a traced denom).
MTP: two heads, offset targets; mean of per-head CE.
z_loss: mean of squared log-sum-exp over experts.
total_loss: ce + mtp + z_loss_weight * z computed *inside* the jitted
    function; config host-side.
shape / dtype: CE / MTP / z / total are float32 scalars (0-d arrays ok);
    soft_cap is float32, same shape as logits.

Do not mark gpu. Do not pass ``cap`` as a traced positional into jit
without ``static_argnums``. Tests import ``train``; ``train`` must not
import ``tests``.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import train
from tests.reference import train as ref

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)

CAP = 30.0

# Same arrays as tests/test_train.py goldens (NumPy reference, not production).
SOFTCAP_X = np.array([0.0, 30.0, 60.0, -15.0], dtype=np.float32)

CE_LOGITS = np.array([[[2.0, 0.0, -1.0], [0.0, 3.0, 0.0]]], dtype=np.float32)
CE_TARGETS = np.array([[0, 1]], dtype=np.int32)
CE_MASK = np.array([[1.0, 1.0]], dtype=np.float32)
CE_MASK_PARTIAL = np.array([[1.0, 0.0]], dtype=np.float32)

MTP_TOKENS = np.array([[1, 2, 0, 1]], dtype=np.int32)
MTP_MASK = np.ones((1, 4), dtype=np.float32)
MTP_H0 = np.zeros((1, 4, 3), dtype=np.float32)
MTP_H0[0, 0, 2] = 4.0
MTP_H0[0, 1, 0] = 4.0
MTP_H0[0, 2, 1] = 4.0
MTP_H1 = np.zeros((1, 4, 3), dtype=np.float32)
MTP_H1[0, 0, 0] = 4.0
MTP_H1[0, 1, 1] = 4.0

Z_PROBS = np.array([[[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]], dtype=np.float32)

Z_WEIGHT = 0.5


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _j32(x: object) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.float32)


def _ji32(x: object) -> jax.Array:
    return jnp.asarray(x, dtype=jnp.int32)


def _close(got: object, exp: object, **kwargs: float) -> None:
    tol = dict(TOL)
    tol.update(kwargs)
    np.testing.assert_allclose(_np(got), _np(exp), **tol)


def _assert_f32_scalar(got: object) -> None:
    arr = _np(got)
    assert arr.dtype == np.float32
    assert arr.shape == ()


def _cfg(*, z_loss_weight: float = Z_WEIGHT) -> train.TrainConfig:
    return replace(train.tiny_train_config(), z_loss_weight=z_loss_weight)


def _mean_sq_logsumexp(probs: np.ndarray) -> np.float32:
    """Mean over tokens of (logsumexp over experts)^2. Independent of production."""
    p = np.asarray(probs, dtype=np.float32)
    m = np.max(p, axis=-1, keepdims=True)
    lse = np.squeeze(
        m + np.log(np.maximum(np.exp(p - m).sum(axis=-1, keepdims=True), 1e-12)),
        axis=-1,
    )
    return np.float32(np.mean(lse**2))


def _mean_of_two_head_ce(
    h0: np.ndarray,
    h1: np.ndarray,
    tokens: np.ndarray,
    mask: np.ndarray,
) -> np.float32:
    """Head i predicts token at offset i+1; mean of the two masked CEs."""
    losses = []
    for i, logits in enumerate((h0, h1)):
        offset = i + 1
        head_logits = logits[..., :-offset, :]
        tgt = tokens[..., offset:]
        head_mask = mask[..., :-offset] * mask[..., offset:]
        losses.append(ref.cross_entropy(head_logits, tgt, head_mask))
    return np.float32(np.mean(np.stack(losses)))


def _loss_pipeline(logits, targets, mask, h0, h1, tokens, mtp_mask, probs, config):
    """Public losses composed the way a training step would compose them."""
    ce = train.cross_entropy(logits, targets, mask)
    mtp = train.mtp_loss((h0, h1), tokens, mtp_mask)
    z = train.z_loss(probs)
    return train.total_loss(ce, mtp, z, config)


# ---------------------------------------------------------------------------
# soft_cap
# ---------------------------------------------------------------------------
def test_soft_cap_jit_matches_eager_and_reference() -> None:
    """jax.jit(soft_cap) with cap closed over matches eager and the NumPy oracle."""
    logits = _j32(SOFTCAP_X)
    cap = float(CAP)

    def sc(z):
        return train.soft_cap(z, cap)

    eager = train.soft_cap(logits, cap)
    got = jax.jit(sc)(logits)
    _close(got, eager)
    _close(got, ref.soft_cap(SOFTCAP_X, cap))
    assert tuple(got.shape) == tuple(logits.shape) == (4,)
    assert _np(got).dtype == np.float32


def test_soft_cap_jit_static_argnums_cap() -> None:
    """jax.jit(..., static_argnums=(1,)) — cap stays a host Python float."""
    logits = _j32(SOFTCAP_X)
    jitted = jax.jit(train.soft_cap, static_argnums=(1,))
    got = jitted(logits, float(CAP))
    eager = train.soft_cap(logits, float(CAP))
    _close(got, eager)
    _close(got, ref.soft_cap(SOFTCAP_X, CAP))
    assert tuple(got.shape) == (4,)
    assert _np(got).dtype == np.float32


# ---------------------------------------------------------------------------
# cross_entropy
# ---------------------------------------------------------------------------
def test_cross_entropy_jit_matches_eager_and_reference() -> None:
    """jax.jit(cross_entropy) matches eager and the NumPy oracle at 1e-5."""
    logits = _j32(CE_LOGITS)
    targets = _ji32(CE_TARGETS)
    mask = _j32(CE_MASK)
    jitted = jax.jit(train.cross_entropy)

    got = jitted(logits, targets, mask)
    eager = train.cross_entropy(logits, targets, mask)
    _close(got, eager)
    _close(got, ref.cross_entropy(CE_LOGITS, CE_TARGETS, CE_MASK))
    _assert_f32_scalar(got)

    partial = jitted(logits, targets, _j32(CE_MASK_PARTIAL))
    _close(partial, train.cross_entropy(logits, targets, CE_MASK_PARTIAL))
    _close(partial, ref.cross_entropy(CE_LOGITS, CE_TARGETS, CE_MASK_PARTIAL))
    _assert_f32_scalar(partial)


def test_cross_entropy_jit_empty_mask_returns_zero() -> None:
    """Empty mask returns 0 under jit (no Python branch on a traced denom)."""
    logits = _j32(CE_LOGITS)
    targets = _ji32(CE_TARGETS)
    mask = jnp.zeros_like(_j32(CE_MASK))
    got = jax.jit(train.cross_entropy)(logits, targets, mask)
    _close(got, np.float32(0.0))
    _close(got, ref.cross_entropy(CE_LOGITS, CE_TARGETS, np.zeros_like(CE_MASK)))
    _assert_f32_scalar(got)


def test_cross_entropy_jitted_grad_matches_eager_grad() -> None:
    """jax.grad through jitted cross_entropy matches eager jax.grad at 1e-5."""
    logits = _j32(CE_LOGITS)
    targets = _ji32(CE_TARGETS)
    mask = _j32(CE_MASK)

    jitted_ce = jax.jit(lambda lg: train.cross_entropy(lg, targets, mask))

    def eager_ce(lg):
        return train.cross_entropy(lg, targets, mask)

    jit_g = jax.grad(jitted_ce)(logits)
    eager_g = jax.grad(eager_ce)(logits)
    _close(jit_g, eager_g)
    assert tuple(jit_g.shape) == tuple(logits.shape)
    assert _np(jit_g).dtype == np.float32


def test_cross_entropy_jitted_grad_finite_difference_vs_reference() -> None:
    """Directional finite difference of the NumPy oracle vs jitted jax.grad."""
    g_np = CE_LOGITS
    v_np = np.array([[[0.3, -0.2, 0.1], [-0.1, 0.4, -0.3]]], dtype=np.float32)
    eps = np.float32(1e-3)

    def ref_ce(z: np.ndarray) -> float:
        return float(ref.cross_entropy(z, CE_TARGETS, CE_MASK))

    fd = (ref_ce(g_np + eps * v_np) - ref_ce(g_np - eps * v_np)) / (2.0 * float(eps))

    targets = _ji32(CE_TARGETS)
    mask = _j32(CE_MASK)
    jitted_ce = jax.jit(lambda lg: train.cross_entropy(lg, targets, mask))
    jit_g = jax.grad(jitted_ce)(_j32(g_np))
    analytic = float(np.sum(_np(jit_g) * v_np))
    np.testing.assert_allclose(analytic, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# mtp_loss
# ---------------------------------------------------------------------------
def test_mtp_loss_jit_two_heads_offset_targets() -> None:
    """Two MTP heads under jit: head i predicts token t+i+1; mean of per-head CE."""
    h0 = _j32(MTP_H0)
    h1 = _j32(MTP_H1)
    tokens = _ji32(MTP_TOKENS)
    mask = _j32(MTP_MASK)

    def mtp(a, b, t, m):
        return train.mtp_loss((a, b), t, m)

    got = jax.jit(mtp)(h0, h1, tokens, mask)
    eager = train.mtp_loss((h0, h1), tokens, mask)
    expected_ref = ref.mtp_loss((MTP_H0, MTP_H1), MTP_TOKENS, MTP_MASK)
    expected_mean = _mean_of_two_head_ce(MTP_H0, MTP_H1, MTP_TOKENS, MTP_MASK)
    _close(got, eager)
    _close(got, expected_ref)
    _close(got, expected_mean)
    _assert_f32_scalar(got)


def test_mtp_loss_jit_matches_eager_and_reference() -> None:
    """jax.jit(mtp_loss) on a pytree of heads matches eager and the NumPy oracle."""
    heads = (_j32(MTP_H0), _j32(MTP_H1))
    tokens = _ji32(MTP_TOKENS)
    mask = _j32(MTP_MASK)
    got = jax.jit(train.mtp_loss)(heads, tokens, mask)
    _close(got, train.mtp_loss(heads, tokens, mask))
    _close(got, ref.mtp_loss((MTP_H0, MTP_H1), MTP_TOKENS, MTP_MASK))
    _assert_f32_scalar(got)


# ---------------------------------------------------------------------------
# z_loss
# ---------------------------------------------------------------------------
def test_z_loss_jit_mean_squared_logsumexp() -> None:
    """z_loss under jit is mean of (logsumexp over experts)^2."""
    probs = _j32(Z_PROBS)
    got = jax.jit(train.z_loss)(probs)
    _close(got, train.z_loss(probs))
    _close(got, ref.z_loss(Z_PROBS))
    _close(got, _mean_sq_logsumexp(Z_PROBS))
    _assert_f32_scalar(got)


def test_z_loss_jit_matches_eager_and_reference() -> None:
    """jax.jit(z_loss) matches eager and the NumPy oracle on a larger router tensor."""
    rng = np.random.default_rng(0)
    probs_np = rng.random((2, 3, 4), dtype=np.float32)
    probs_np = probs_np / probs_np.sum(axis=-1, keepdims=True)
    probs = _j32(probs_np)
    got = jax.jit(train.z_loss)(probs)
    _close(got, train.z_loss(probs))
    _close(got, ref.z_loss(probs_np))
    _close(got, _mean_sq_logsumexp(probs_np))
    _assert_f32_scalar(got)


# ---------------------------------------------------------------------------
# total_loss (composed inside jit so traced CE / MTP / z stay in JAX)
# ---------------------------------------------------------------------------
def test_total_loss_jit_matches_eager_and_reference() -> None:
    """Jitted ce + mtp + z_loss_weight * z; TrainConfig closed over (host-side)."""
    cfg = _cfg()
    logits = _j32(CE_LOGITS)
    targets = _ji32(CE_TARGETS)
    mask = _j32(CE_MASK)
    h0 = _j32(MTP_H0)
    h1 = _j32(MTP_H1)
    tokens = _ji32(MTP_TOKENS)
    mtp_mask = _j32(MTP_MASK)
    probs = _j32(Z_PROBS)

    def pipeline(lg, tgt, msk, a, b, tok, mm, rp):
        return _loss_pipeline(lg, tgt, msk, a, b, tok, mm, rp, cfg)

    got = jax.jit(pipeline)(logits, targets, mask, h0, h1, tokens, mtp_mask, probs)
    eager = _loss_pipeline(logits, targets, mask, h0, h1, tokens, mtp_mask, probs, cfg)
    ce_r = ref.cross_entropy(CE_LOGITS, CE_TARGETS, CE_MASK)
    mtp_r = ref.mtp_loss((MTP_H0, MTP_H1), MTP_TOKENS, MTP_MASK)
    z_r = ref.z_loss(Z_PROBS)
    ref_total = ref.total_loss(ce_r, mtp_r, z_r, cfg)
    _close(got, eager)
    _close(got, ref_total)
    _close(got, np.float32(float(ce_r) + float(mtp_r) + Z_WEIGHT * float(z_r)))
    _assert_f32_scalar(got)


def test_total_loss_jit_static_argnums_config() -> None:
    """TrainConfig via static_argnums; combination matches eager and reference."""
    cfg = _cfg(z_loss_weight=0.25)
    logits = _j32(CE_LOGITS)
    targets = _ji32(CE_TARGETS)
    mask = _j32(CE_MASK)
    h0 = _j32(MTP_H0)
    h1 = _j32(MTP_H1)
    tokens = _ji32(MTP_TOKENS)
    mtp_mask = _j32(MTP_MASK)
    probs = _j32(Z_PROBS)

    def pipeline(lg, tgt, msk, a, b, tok, mm, rp, config):
        return _loss_pipeline(lg, tgt, msk, a, b, tok, mm, rp, config)

    jitted = jax.jit(pipeline, static_argnums=(8,))
    got = jitted(logits, targets, mask, h0, h1, tokens, mtp_mask, probs, cfg)
    eager = _loss_pipeline(logits, targets, mask, h0, h1, tokens, mtp_mask, probs, cfg)
    ce_r = ref.cross_entropy(CE_LOGITS, CE_TARGETS, CE_MASK)
    mtp_r = ref.mtp_loss((MTP_H0, MTP_H1), MTP_TOKENS, MTP_MASK)
    z_r = ref.z_loss(Z_PROBS)
    ref_total = ref.total_loss(ce_r, mtp_r, z_r, cfg)
    _close(got, eager)
    _close(got, ref_total)
    _assert_f32_scalar(got)


# ---------------------------------------------------------------------------
# shape / dtype
# ---------------------------------------------------------------------------
def test_loss_jit_shape_dtype_float32_scalars() -> None:
    """Jitted CE, MTP, z-loss, and total_loss are float32 scalars (0-d ok)."""
    logits = _j32(CE_LOGITS)
    targets = _ji32(CE_TARGETS)
    mask = _j32(CE_MASK)
    heads = (_j32(MTP_H0), _j32(MTP_H1))
    tokens = _ji32(MTP_TOKENS)
    mtp_mask = _j32(MTP_MASK)
    probs = _j32(Z_PROBS)
    cfg = _cfg()

    ce = jax.jit(train.cross_entropy)(logits, targets, mask)
    mtp = jax.jit(train.mtp_loss)(heads, tokens, mtp_mask)
    z = jax.jit(train.z_loss)(probs)

    def pipeline(lg, tgt, msk, a, b, tok, mm, rp):
        return _loss_pipeline(lg, tgt, msk, a, b, tok, mm, rp, cfg)

    tot = jax.jit(pipeline)(logits, targets, mask, heads[0], heads[1], tokens, mtp_mask, probs)
    _assert_f32_scalar(ce)
    _assert_f32_scalar(mtp)
    _assert_f32_scalar(z)
    _assert_f32_scalar(tot)
