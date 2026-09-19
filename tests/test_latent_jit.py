"""I5-latent-jit oracle: ``jax.jit`` of Stage-B losses must match eager at 1e-5.

These tests call the public ``model.latent`` signatures with ``jnp`` arrays and
Python scalars held static (``teacher_steps``, ``answer_id``, ``lambda_prior``
closed over or ``static_argnums``; ``LatentConfig`` closed over or static).
They must fail on the current ``numpy.asarray`` / Python ``float()`` /
``math.log`` / Python-loop implementations (``TracerArrayConversionError`` or
``ConcretizationTypeError``) and pass once the seven functions stay in JAX.

``geometric_prior`` is host-side (Python ``n``). Do not jit it. ``halt_kl`` may
call it with a Python int length.

Coverage
--------
jit vs eager (1e-5): ponder_distribution, halt_from_logits, halt_loss,
    halt_kl, thought_decode_ce, answer_ce_at_depths, stage_b_loss.
jit vs tests/reference/latent.py (1e-5): independent of production math.
jax.grad through jitted halt_loss / thought_decode_ce vs eager jax.grad at
    1e-5; cheap finite-difference check vs the NumPy reference.
empty thought mask: thought_decode_ce returns 0 under jit (no Python branch
    on a traced denom).
halt_loss clips teacher_steps to [0, K] with the int host-side.
halt_kl vs geometric prior (prior itself is host-side).
stage_b_loss: total = l_task + alpha*l_traj + gamma*l_halt + beta*l_kl;
    config host-side.
shape / dtype: jitted scalars are float32 0-d; vectors keep input shape.

Do not mark gpu. Do not pass ``teacher_steps`` / ``answer_id`` /
``lambda_prior`` as traced positionals into jit without ``static_argnums``.
Tests import ``model.latent``; production must not import ``tests``.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import model.latent as latent
from tests.reference import latent as ref

TOL = dict(rtol=1e-5, atol=1e-5)
FD_TOL = dict(rtol=1e-3, atol=1e-3)

HALT_EPS = float(latent.HALT_EPS)

# Short-vector golden: λ = [0.3, 0.4, 0.5, 0.6], last pinned to 1.0.
# p = [0.3, 0.4*0.7, 0.5*0.7*0.6, 1.0*0.7*0.6*0.5] = [0.3, 0.28, 0.21, 0.21]
GOLDEN_LAMBDAS = np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32)
GOLDEN_PONDER = np.array([0.3, 0.28, 0.21, 0.21], dtype=np.float32)

CLIP_LAMBDAS = np.array([0.0, 1.0, 2.0, -5.0], dtype=np.float32)

HALT_LOGITS = np.array([0.0, 0.5, -0.5], dtype=np.float32)

CE_LOGITS = np.array([[2.0, 0.0, -1.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
CE_IDS = np.array([0, 1, 0], dtype=np.int32)
CE_MASK = np.array([1.0, 1.0, 0.0], dtype=np.float32)
CE_MASK_EMPTY = np.array([0.0, 0.0, 0.0], dtype=np.float32)

ANSWER_LOGITS = np.array(
    [[2.0, 0.0, -1.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)
ANSWER_ID = 0

# stage_b: K+1 = 3 halt / answer rows; K = 2 thought rows.
STAGE_HALT_LOGITS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
STAGE_ANSWER_LOGITS = np.array(
    [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
    dtype=np.float32,
)
STAGE_THOUGHT_LOGITS = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]], dtype=np.float32)
STAGE_TEACHER_IDS = np.array([0, 1], dtype=np.int32)
STAGE_THOUGHT_MASK = np.array([1.0, 1.0], dtype=np.float32)
STAGE_ANSWER_ID = 0
STAGE_TEACHER_STEPS = 1
STAGE_ALPHA = 2.0
STAGE_GAMMA = 3.0
STAGE_BETA = 4.0
LAMBDA_PRIOR = 0.2


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


def _assert_f32_vec(got: object, shape: tuple[int, ...]) -> None:
    arr = _np(got)
    assert arr.dtype == np.float32
    assert tuple(arr.shape) == shape


def _ponder_np(lambdas: np.ndarray) -> np.ndarray:
    """Vectorized PonderNet mass. Independent of production loops."""
    lam = np.clip(np.asarray(lambdas, dtype=np.float64), HALT_EPS, 1.0 - HALT_EPS).copy()
    lam[-1] = 1.0
    stay = 1.0 - lam
    survival = np.empty_like(lam)
    survival[0] = 1.0
    if lam.size > 1:
        survival[1:] = np.cumprod(stay[:-1])
    return lam * survival


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    em = np.exp(x[~pos])
    out[~pos] = em / (1.0 + em)
    return out


def _ce_rows_np(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-row CE. Independent of production row loops."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    shifted = z - z.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_p = shifted - log_z
    return -log_p[np.arange(z.shape[0]), y]


def _thought_ce_np(logits: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> np.float64:
    ces = _ce_rows_np(logits, ids)
    kept = np.asarray(mask) != 0
    if not np.any(kept):
        return np.float64(0.0)
    return np.float64(ces[kept].mean())


def _geometric_np(n: int, lambda_prior: float) -> np.ndarray:
    """Host-side truncated Geometric. Not jitted."""
    m = np.arange(int(n), dtype=np.float64)
    g = float(lambda_prior) * np.power(1.0 - float(lambda_prior), m)
    return g / g.sum()


def _halt_kl_np(p: np.ndarray, lambda_prior: float) -> np.float64:
    arr = np.asarray(p, dtype=np.float64)
    g = _geometric_np(arr.shape[0], lambda_prior)
    return np.float64(np.sum(arr * (np.log(arr + HALT_EPS) - np.log(g + HALT_EPS))))


def _halt_fields(halt_logits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    out = latent.halt_from_logits(halt_logits)
    return out.lambdas, out.p, out.expected_depth


def _stage_fields(
    halt_logits: jax.Array,
    answer_logits: jax.Array,
    thought_logits: jax.Array,
    teacher_ids: jax.Array,
    thought_mask: jax.Array,
    answer_id: int,
    teacher_steps: int,
    config: latent.LatentConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    out = latent.stage_b_loss(
        halt_logits,
        answer_logits,
        thought_logits,
        teacher_ids,
        thought_mask,
        answer_id,
        teacher_steps,
        config,
    )
    return out.l_task, out.l_traj, out.l_halt, out.l_kl, out.expected_depth, out.total


def _prod_cfg(**kwargs: object) -> latent.LatentConfig:
    return replace(latent.LatentConfig(), **kwargs)  # type: ignore[arg-type]


def _ref_cfg(**kwargs: object) -> ref.LatentConfig:
    return replace(ref.LatentConfig(), **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ponder_distribution
# ---------------------------------------------------------------------------
def test_ponder_distribution_jit_matches_eager_and_reference() -> None:
    """jax.jit(ponder_distribution) matches eager and the NumPy oracle at 1e-5."""
    lam = _j32(GOLDEN_LAMBDAS)
    got = jax.jit(latent.ponder_distribution)(lam)
    eager = latent.ponder_distribution(lam)
    _close(got, eager)
    _close(got, ref.ponder_distribution(GOLDEN_LAMBDAS))
    _close(got, GOLDEN_PONDER)
    _close(got, _ponder_np(GOLDEN_LAMBDAS))
    _assert_f32_vec(got, (4,))


def test_ponder_distribution_jit_clip_and_pin() -> None:
    """Under jit, λ is clipped then the last slot is pinned to 1 so p sums to 1."""
    lam = _j32(CLIP_LAMBDAS)
    got = jax.jit(latent.ponder_distribution)(lam)
    _close(got, latent.ponder_distribution(lam))
    _close(got, ref.ponder_distribution(CLIP_LAMBDAS))
    _close(got, _ponder_np(CLIP_LAMBDAS))
    _close(_np(got).sum(), np.float32(1.0))
    _assert_f32_vec(got, (4,))


# ---------------------------------------------------------------------------
# halt_from_logits
# ---------------------------------------------------------------------------
def test_halt_from_logits_jit_matches_eager_and_reference() -> None:
    """jax.jit of halt_from_logits fields matches eager, reference, and E[depth]."""
    logits = _j32(HALT_LOGITS)
    got_lam, got_p, got_d = jax.jit(_halt_fields)(logits)
    eager = latent.halt_from_logits(logits)
    exp = ref.halt_from_logits(HALT_LOGITS)
    _close(got_lam, eager.lambdas)
    _close(got_p, eager.p)
    _close(got_d, eager.expected_depth)
    _close(got_lam, exp.lambdas)
    _close(got_p, exp.p)
    _close(got_d, exp.expected_depth)
    sig = _sigmoid_np(HALT_LOGITS)
    hand_p = _ponder_np(sig)
    _close(got_p, hand_p)
    depths = np.arange(hand_p.shape[0], dtype=np.float64)
    _close(got_d, np.sum(hand_p * depths))
    _assert_f32_vec(got_lam, (3,))
    _assert_f32_vec(got_p, (3,))
    _assert_f32_scalar(got_d)


# ---------------------------------------------------------------------------
# halt_loss
# ---------------------------------------------------------------------------
def test_halt_loss_jit_matches_eager_and_reference() -> None:
    """jax.jit(halt_loss) with teacher_steps closed over matches eager and ref."""
    p = _j32(GOLDEN_PONDER)
    k = 1

    def hl(prob):
        return latent.halt_loss(prob, k)

    got = jax.jit(hl)(p)
    _close(got, latent.halt_loss(p, k))
    _close(got, ref.halt_loss(GOLDEN_PONDER, k))
    _close(got, -np.log(float(GOLDEN_PONDER[k]) + HALT_EPS))
    _assert_f32_scalar(got)


def test_halt_loss_jit_clips_teacher_steps() -> None:
    """teacher_steps is a host Python int; values outside 0..K clip, not reject."""
    p = _j32(GOLDEN_PONDER)
    k_max = int(GOLDEN_PONDER.shape[0]) - 1
    cases = ((-5, 0), (0, 0), (1, 1), (k_max, k_max), (99, k_max))
    for teacher, k in cases:

        def hl(prob, t=teacher):
            return latent.halt_loss(prob, t)

        got = jax.jit(hl)(p)
        _close(got, latent.halt_loss(p, teacher))
        _close(got, ref.halt_loss(GOLDEN_PONDER, teacher))
        _close(got, -np.log(float(GOLDEN_PONDER[k]) + HALT_EPS))
        _assert_f32_scalar(got)


def test_halt_loss_jit_static_argnums_teacher_steps() -> None:
    """jax.jit(..., static_argnums=(1,)) — teacher_steps stays a host Python int."""
    p = _j32(GOLDEN_PONDER)
    jitted = jax.jit(latent.halt_loss, static_argnums=(1,))
    got = jitted(p, 2)
    _close(got, latent.halt_loss(p, 2))
    _close(got, ref.halt_loss(GOLDEN_PONDER, 2))
    _assert_f32_scalar(got)


def test_halt_loss_jitted_grad_matches_eager_grad() -> None:
    """jax.grad through jitted halt_loss matches eager jax.grad at 1e-5."""
    p = _j32(GOLDEN_PONDER)
    k = 1
    jitted = jax.jit(lambda prob: latent.halt_loss(prob, k))

    def eager_hl(prob):
        return latent.halt_loss(prob, k)

    jit_g = jax.grad(jitted)(p)
    eager_g = jax.grad(eager_hl)(p)
    _close(jit_g, eager_g)
    _assert_f32_vec(jit_g, (4,))


def test_halt_loss_jitted_grad_finite_difference_vs_reference() -> None:
    """Directional finite difference of the NumPy oracle vs jitted jax.grad."""
    p_np = GOLDEN_PONDER
    v_np = np.array([0.05, -0.02, 0.04, -0.07], dtype=np.float32)
    k = 2
    eps = np.float32(1e-3)

    def ref_hl(prob: np.ndarray) -> float:
        return float(ref.halt_loss(prob, k))

    fd = (ref_hl(p_np + eps * v_np) - ref_hl(p_np - eps * v_np)) / (2.0 * float(eps))
    jitted = jax.jit(lambda prob: latent.halt_loss(prob, k))
    jit_g = jax.grad(jitted)(_j32(p_np))
    analytic = float(np.sum(_np(jit_g) * v_np))
    np.testing.assert_allclose(analytic, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# halt_kl
# ---------------------------------------------------------------------------
def test_halt_kl_jit_matches_eager_and_reference() -> None:
    """jax.jit(halt_kl) with lambda_prior closed over matches eager and ref."""
    p = _j32(GOLDEN_PONDER)
    lam_p = float(LAMBDA_PRIOR)

    def hkl(prob):
        return latent.halt_kl(prob, lam_p)

    got = jax.jit(hkl)(p)
    _close(got, latent.halt_kl(p, lam_p))
    _close(got, ref.halt_kl(GOLDEN_PONDER, lam_p))
    _close(got, _halt_kl_np(GOLDEN_PONDER, lam_p))
    _assert_f32_scalar(got)


def test_halt_kl_jit_vs_geometric_prior() -> None:
    """Jitted KL(p || Geometric(λ)) matches the host-side prior; KL(g||g) is 0."""
    lam_p = float(LAMBDA_PRIOR)
    g_np = _geometric_np(5, lam_p)
    _close(g_np, ref.geometric_prior(5, lam_p))

    def hkl(prob):
        return latent.halt_kl(prob, lam_p)

    jitted = jax.jit(hkl)
    zero = jitted(_j32(g_np))
    _close(zero, 0.0)
    _close(zero, ref.halt_kl(g_np, lam_p))
    _assert_f32_scalar(zero)

    p = _j32(GOLDEN_PONDER)
    got = jitted(p)
    g4 = _geometric_np(4, lam_p)
    hand = np.sum(GOLDEN_PONDER.astype(np.float64) * (
        np.log(GOLDEN_PONDER.astype(np.float64) + HALT_EPS) - np.log(g4 + HALT_EPS)
    ))
    _close(got, hand)
    _close(got, ref.halt_kl(GOLDEN_PONDER, lam_p))
    assert float(_np(got)) > 0.0


def test_halt_kl_jit_static_argnums_lambda_prior() -> None:
    """jax.jit(..., static_argnums=(1,)) — lambda_prior stays a host Python float."""
    p = _j32(GOLDEN_PONDER)
    jitted = jax.jit(latent.halt_kl, static_argnums=(1,))
    got = jitted(p, float(LAMBDA_PRIOR))
    _close(got, latent.halt_kl(p, LAMBDA_PRIOR))
    _close(got, ref.halt_kl(GOLDEN_PONDER, LAMBDA_PRIOR))
    _assert_f32_scalar(got)


# ---------------------------------------------------------------------------
# thought_decode_ce
# ---------------------------------------------------------------------------
def test_thought_decode_ce_jit_matches_eager_and_reference() -> None:
    """jax.jit(thought_decode_ce) matches eager, reference, and masked-mean CE."""
    logits = _j32(CE_LOGITS)
    ids = _ji32(CE_IDS)
    mask = _j32(CE_MASK)
    jitted = jax.jit(latent.thought_decode_ce)
    got = jitted(logits, ids, mask)
    _close(got, latent.thought_decode_ce(logits, ids, mask))
    _close(got, ref.thought_decode_ce(CE_LOGITS, CE_IDS, CE_MASK))
    _close(got, _thought_ce_np(CE_LOGITS, CE_IDS, CE_MASK))
    _assert_f32_scalar(got)


def test_thought_decode_ce_jit_empty_mask_returns_zero() -> None:
    """Empty mask returns 0 under jit (no Python branch on a traced denom)."""
    logits = _j32(CE_LOGITS)
    ids = _ji32(CE_IDS)
    mask = _j32(CE_MASK_EMPTY)
    got = jax.jit(latent.thought_decode_ce)(logits, ids, mask)
    _close(got, np.float32(0.0))
    _close(got, latent.thought_decode_ce(logits, ids, mask))
    _close(got, ref.thought_decode_ce(CE_LOGITS, CE_IDS, CE_MASK_EMPTY))
    _close(got, _thought_ce_np(CE_LOGITS, CE_IDS, CE_MASK_EMPTY))
    _assert_f32_scalar(got)


def test_thought_decode_ce_jitted_grad_matches_eager_grad() -> None:
    """jax.grad through jitted thought_decode_ce matches eager jax.grad at 1e-5."""
    logits = _j32(CE_LOGITS)
    ids = _ji32(CE_IDS)
    mask = _j32(CE_MASK)
    jitted = jax.jit(lambda lg: latent.thought_decode_ce(lg, ids, mask))

    def eager_ce(lg):
        return latent.thought_decode_ce(lg, ids, mask)

    jit_g = jax.grad(jitted)(logits)
    eager_g = jax.grad(eager_ce)(logits)
    _close(jit_g, eager_g)
    _assert_f32_vec(jit_g, tuple(CE_LOGITS.shape))


def test_thought_decode_ce_jitted_grad_finite_difference_vs_reference() -> None:
    """Directional finite difference of the NumPy oracle vs jitted jax.grad."""
    g_np = CE_LOGITS
    v_np = np.array(
        [[0.3, -0.2, 0.1], [-0.1, 0.4, -0.3], [0.2, -0.05, 0.15]],
        dtype=np.float32,
    )
    eps = np.float32(1e-3)

    def ref_ce(z: np.ndarray) -> float:
        return float(ref.thought_decode_ce(z, CE_IDS, CE_MASK))

    fd = (ref_ce(g_np + eps * v_np) - ref_ce(g_np - eps * v_np)) / (2.0 * float(eps))
    ids = _ji32(CE_IDS)
    mask = _j32(CE_MASK)
    jitted = jax.jit(lambda lg: latent.thought_decode_ce(lg, ids, mask))
    jit_g = jax.grad(jitted)(_j32(g_np))
    analytic = float(np.sum(_np(jit_g) * v_np))
    np.testing.assert_allclose(analytic, fd, **FD_TOL)


# ---------------------------------------------------------------------------
# answer_ce_at_depths
# ---------------------------------------------------------------------------
def test_answer_ce_at_depths_jit_matches_eager_and_reference() -> None:
    """jax.jit(answer_ce_at_depths) with answer_id closed over matches eager/ref."""
    logits = _j32(ANSWER_LOGITS)
    aid = int(ANSWER_ID)

    def ace(lg):
        return latent.answer_ce_at_depths(lg, aid)

    got = jax.jit(ace)(logits)
    _close(got, latent.answer_ce_at_depths(logits, aid))
    _close(got, ref.answer_ce_at_depths(ANSWER_LOGITS, aid))
    _close(got, _ce_rows_np(ANSWER_LOGITS, np.full(ANSWER_LOGITS.shape[0], aid)))
    _assert_f32_vec(got, (3,))


def test_answer_ce_at_depths_jit_static_argnums_answer_id() -> None:
    """jax.jit(..., static_argnums=(1,)) — answer_id stays a host Python int."""
    logits = _j32(ANSWER_LOGITS)
    jitted = jax.jit(latent.answer_ce_at_depths, static_argnums=(1,))
    got = jitted(logits, int(ANSWER_ID))
    _close(got, latent.answer_ce_at_depths(logits, ANSWER_ID))
    _close(got, ref.answer_ce_at_depths(ANSWER_LOGITS, ANSWER_ID))
    _assert_f32_vec(got, (3,))


# ---------------------------------------------------------------------------
# stage_b_loss (composed inside jit so traced halt / CE / KL stay in JAX)
# ---------------------------------------------------------------------------
def test_stage_b_loss_jit_matches_eager_and_reference() -> None:
    """Jitted Stage-B fields match eager and the NumPy oracle; config closed over."""
    cfg = _prod_cfg(alpha_traj=STAGE_ALPHA, gamma_halt=STAGE_GAMMA, beta_kl=STAGE_BETA)
    ref_cfg = _ref_cfg(alpha_traj=STAGE_ALPHA, gamma_halt=STAGE_GAMMA, beta_kl=STAGE_BETA)
    halt = _j32(STAGE_HALT_LOGITS)
    answer = _j32(STAGE_ANSWER_LOGITS)
    thought = _j32(STAGE_THOUGHT_LOGITS)
    ids = _ji32(STAGE_TEACHER_IDS)
    mask = _j32(STAGE_THOUGHT_MASK)
    aid = int(STAGE_ANSWER_ID)
    steps = int(STAGE_TEACHER_STEPS)

    def pipeline(h, a, th, i, m):
        return _stage_fields(h, a, th, i, m, aid, steps, cfg)

    got = jax.jit(pipeline)(halt, answer, thought, ids, mask)
    eager = latent.stage_b_loss(
        halt, answer, thought, ids, mask, aid, steps, cfg
    )
    exp = ref.stage_b_loss(
        STAGE_HALT_LOGITS,
        STAGE_ANSWER_LOGITS,
        STAGE_THOUGHT_LOGITS,
        STAGE_TEACHER_IDS,
        STAGE_THOUGHT_MASK,
        aid,
        steps,
        ref_cfg,
    )
    names = ("l_task", "l_traj", "l_halt", "l_kl", "expected_depth", "total")
    for value, name in zip(got, names, strict=True):
        _close(value, getattr(eager, name))
        _close(value, getattr(exp, name))
        _assert_f32_scalar(value)


def test_stage_b_loss_jit_weighted_total() -> None:
    """total = l_task + alpha*l_traj + gamma*l_halt + beta*l_kl inside the jit."""
    cfg = _prod_cfg(alpha_traj=STAGE_ALPHA, gamma_halt=STAGE_GAMMA, beta_kl=STAGE_BETA)
    halt = _j32(STAGE_HALT_LOGITS)
    answer = _j32(STAGE_ANSWER_LOGITS)
    thought = _j32(STAGE_THOUGHT_LOGITS)
    ids = _ji32(STAGE_TEACHER_IDS)
    mask = _j32(STAGE_THOUGHT_MASK)
    aid = int(STAGE_ANSWER_ID)
    steps = int(STAGE_TEACHER_STEPS)

    def pipeline(h, a, th, i, m):
        return _stage_fields(h, a, th, i, m, aid, steps, cfg)

    l_task, l_traj, l_halt, l_kl, expected_depth, total = jax.jit(pipeline)(
        halt, answer, thought, ids, mask
    )
    _close(total, l_task + STAGE_ALPHA * l_traj + STAGE_GAMMA * l_halt + STAGE_BETA * l_kl)

    halt_out = ref.halt_from_logits(STAGE_HALT_LOGITS)
    answer_ce = ref.answer_ce_at_depths(STAGE_ANSWER_LOGITS, aid)
    l_task_r = float(np.sum(np.asarray(halt_out.p) * np.asarray(answer_ce)))
    l_traj_r = float(
        ref.thought_decode_ce(STAGE_THOUGHT_LOGITS, STAGE_TEACHER_IDS, STAGE_THOUGHT_MASK)
    )
    l_halt_r = float(ref.halt_loss(halt_out.p, steps))
    l_kl_r = float(ref.halt_kl(halt_out.p, LAMBDA_PRIOR))
    _close(l_task, l_task_r)
    _close(l_traj, l_traj_r)
    _close(l_halt, l_halt_r)
    _close(l_kl, l_kl_r)
    _close(expected_depth, halt_out.expected_depth)
    _close(total, l_task_r + STAGE_ALPHA * l_traj_r + STAGE_GAMMA * l_halt_r + STAGE_BETA * l_kl_r)
    _assert_f32_scalar(total)


def test_stage_b_loss_jit_static_argnums_config() -> None:
    """answer_id, teacher_steps, LatentConfig via static_argnums; host-side."""
    cfg = _prod_cfg(alpha_traj=STAGE_ALPHA, gamma_halt=STAGE_GAMMA, beta_kl=STAGE_BETA)
    ref_cfg = _ref_cfg(alpha_traj=STAGE_ALPHA, gamma_halt=STAGE_GAMMA, beta_kl=STAGE_BETA)
    halt = _j32(STAGE_HALT_LOGITS)
    answer = _j32(STAGE_ANSWER_LOGITS)
    thought = _j32(STAGE_THOUGHT_LOGITS)
    ids = _ji32(STAGE_TEACHER_IDS)
    mask = _j32(STAGE_THOUGHT_MASK)

    jitted = jax.jit(_stage_fields, static_argnums=(5, 6, 7))
    got = jitted(
        halt,
        answer,
        thought,
        ids,
        mask,
        int(STAGE_ANSWER_ID),
        int(STAGE_TEACHER_STEPS),
        cfg,
    )
    eager = latent.stage_b_loss(
        halt,
        answer,
        thought,
        ids,
        mask,
        STAGE_ANSWER_ID,
        STAGE_TEACHER_STEPS,
        cfg,
    )
    exp = ref.stage_b_loss(
        STAGE_HALT_LOGITS,
        STAGE_ANSWER_LOGITS,
        STAGE_THOUGHT_LOGITS,
        STAGE_TEACHER_IDS,
        STAGE_THOUGHT_MASK,
        STAGE_ANSWER_ID,
        STAGE_TEACHER_STEPS,
        ref_cfg,
    )
    names = ("l_task", "l_traj", "l_halt", "l_kl", "expected_depth", "total")
    for value, name in zip(got, names, strict=True):
        _close(value, getattr(eager, name))
        _close(value, getattr(exp, name))
        _assert_f32_scalar(value)


# ---------------------------------------------------------------------------
# shape / dtype
# ---------------------------------------------------------------------------
def test_latent_jit_shape_dtype_float32() -> None:
    """Jitted Stage-B outputs are float32; vectors keep the input length."""
    lam = _j32(GOLDEN_LAMBDAS)
    p = jax.jit(latent.ponder_distribution)(lam)
    _assert_f32_vec(p, (4,))

    got_lam, got_p, got_d = jax.jit(_halt_fields)(_j32(HALT_LOGITS))
    _assert_f32_vec(got_lam, (3,))
    _assert_f32_vec(got_p, (3,))
    _assert_f32_scalar(got_d)

    hl = jax.jit(lambda prob: latent.halt_loss(prob, 1))(lam * 0 + _j32(GOLDEN_PONDER))
    _assert_f32_scalar(hl)
    hkl = jax.jit(lambda prob: latent.halt_kl(prob, LAMBDA_PRIOR))(_j32(GOLDEN_PONDER))
    _assert_f32_scalar(hkl)

    ce = jax.jit(latent.thought_decode_ce)(_j32(CE_LOGITS), _ji32(CE_IDS), _j32(CE_MASK))
    _assert_f32_scalar(ce)
    ace = jax.jit(lambda lg: latent.answer_ce_at_depths(lg, ANSWER_ID))(_j32(ANSWER_LOGITS))
    _assert_f32_vec(ace, (3,))

    cfg = _prod_cfg()

    def pipeline(h, a, th, i, m):
        return _stage_fields(h, a, th, i, m, STAGE_ANSWER_ID, STAGE_TEACHER_STEPS, cfg)

    fields = jax.jit(pipeline)(
        _j32(STAGE_HALT_LOGITS),
        _j32(STAGE_ANSWER_LOGITS),
        _j32(STAGE_THOUGHT_LOGITS),
        _ji32(STAGE_TEACHER_IDS),
        _j32(STAGE_THOUGHT_MASK),
    )
    for value in fields:
        _assert_f32_scalar(value)
