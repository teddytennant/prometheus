"""Oracle tests for μP-for-Muon transfer (spec 5.4, Moonlight arXiv 2502.16982).

Every test calls a stubbed function and must fail on the NotImplementedError
stub. ``MOONLIGHT_UPDATE_RMS`` is already defined; it is checked only inside
tests that also call a stub.

Coverage
--------
muon_rms_scale: Lemma 1 identity ``scale * 1/sqrt(max(A, B)) == 0.2`` on
    square, wide, and tall shapes, including rows > cols. Exact semi-orthogonal
    matrices (QR), not the Newton–Schulz approximation. Rejects non-integers
    and integers < 1.
transferred_muon_lr: returns ``base_lr`` for every width >= 1. The shape
    factor is not folded in (not Adam μP, not ``sqrt(width/base_width)``).
    Rejects non-finite or negative ``base_lr`` and non-integer widths.
muon_transfer_step: equation 4 vs the NumPy reference and vs
    ``newton_schulz`` of the same Nesterov momentum as ``muon_update``,
    scaled by ``lr * muon_rms_scale`` and subtracted from ``param``.
    Weight decay is decoupled (difference is ``lr * weight_decay * param``,
    momentum unchanged, zero-grad update is pure decay). Not Jordan's scale.
    Shape, float32 dtype, no input mutation. Finite differences and
    ``jax.grad`` for the differentiable paths. ``jax.jit`` closed-over and
    static kwargs match eager at 1e-5.
faults: bad lr / weight decay / rank / shape, even Newton–Schulz steps.
gpu (V1): same step vs reference; skipped without a GPU.

Frozen goldens were computed from ``tests.reference.train``, not production.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import train
from tests.reference import train as ref

TOL = dict(rtol=1e-5, atol=1e-5)

# Square, wide, and tall (rows > cols). 25x25 is omitted: there Jordan's
# sqrt(max(1, rows/cols)) equals the Moonlight scale, so it cannot separate them.
RMS_SHAPES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (3, 5),
    (8, 3),
    (3, 8),
    (4, 4),
    (7, 13),
    (16, 9),
    (9, 16),
    (32, 1),
    (1, 32),
    (2, 100),
    (100, 2),
    (6, 6),
)

STEP_SHAPES = ((1, 1), (3, 2), (2, 5), (4, 4), (8, 1), (1, 7))

# Frozen step. Inputs are explicit so the golden does not depend on Generator.
GOLDEN_PARAM = np.array([[0.5, -0.25], [0.1, 0.8], [-0.3, 0.4]], dtype=np.float32)
GOLDEN_GRAD = np.array([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.05]], dtype=np.float32)
GOLDEN_MOMENTUM = np.array([[0.1, 0.2], [-0.3, 0.0], [0.4, -0.2]], dtype=np.float32)
GOLDEN_LR = 0.02
GOLDEN_BETA = 0.95
GOLDEN_STEPS = 5
GOLDEN_WD = 0.1
GOLDEN_P00 = 0.4941366910934448
GOLDEN_PSUM = 1.2451510429382324
GOLDEN_M00 = 0.29500001668930054
GOLDEN_MSUM = 0.24
GOLDEN_BARE_P00 = 0.49513667821884155
GOLDEN_BARE_PSUM = 1.2476511001586914


def _np(x: object) -> np.ndarray:
    return np.asarray(x)


def _close(got: object, expected: object, **tol: float) -> None:
    np.testing.assert_allclose(_np(got), _np(expected), **(tol or TOL))


def _assert_real_scalar(got: object) -> float:
    if isinstance(got, bool) or not isinstance(got, (int, float, np.floating)):
        raise AssertionError(f"expected a real scalar, got {type(got).__name__}: {got!r}")
    return float(got)


def _mats(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    param, grad, momentum = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    return param, grad, momentum


def _semi_orthogonal(rows: int, cols: int, seed: int) -> np.ndarray:
    """Full-rank semi-orthogonal matrix. Columns orthonormal if tall or square."""
    draw = np.random.default_rng(seed).standard_normal((rows, cols))
    if rows >= cols:
        q, _ = np.linalg.qr(draw)
        return q[:, :cols]
    q, _ = np.linalg.qr(draw.T)
    return q[:, :rows].T


def _require_gpu() -> None:
    try:
        gpu = jax.devices("gpu")
    except RuntimeError:
        gpu = []
    if not gpu:
        pytest.skip("no GPU")


def _step(
    param: np.ndarray,
    grad: np.ndarray,
    momentum: np.ndarray,
    *,
    lr: float = GOLDEN_LR,
    momentum_coeff: float = GOLDEN_BETA,
    ns_steps: int = GOLDEN_STEPS,
    weight_decay: float = 0.0,
) -> tuple[object, object]:
    return train.muon_transfer_step(
        param,
        grad,
        momentum,
        lr=lr,
        momentum_coeff=momentum_coeff,
        ns_steps=ns_steps,
        weight_decay=weight_decay,
    )


def _ref_step(
    param: np.ndarray,
    grad: np.ndarray,
    momentum: np.ndarray,
    *,
    lr: float = GOLDEN_LR,
    momentum_coeff: float = GOLDEN_BETA,
    ns_steps: int = GOLDEN_STEPS,
    weight_decay: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    return ref.muon_transfer_step(
        param,
        grad,
        momentum,
        lr=lr,
        momentum_coeff=momentum_coeff,
        ns_steps=ns_steps,
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# muon_rms_scale — Lemma 1 and equation 4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows,cols", RMS_SHAPES)
def test_muon_rms_scale_lemma1_identity(rows: int, cols: int) -> None:
    """scale * RMS(orthogonal) == 0.2, including tall and non-square shapes."""
    assert train.MOONLIGHT_UPDATE_RMS == 0.2
    scale = _assert_real_scalar(train.muon_rms_scale(rows, cols))
    root = math.sqrt(max(rows, cols))
    assert scale == pytest.approx(0.2 * root, rel=1e-5, abs=1e-6)
    assert scale == pytest.approx(ref.muon_rms_scale(rows, cols), rel=1e-5, abs=1e-6)
    # Multiply by the Lemma 1 RMS, not by a separately rounded reciprocal only.
    assert scale * (1.0 / root) == pytest.approx(0.2, abs=1e-6)
    orth = _semi_orthogonal(rows, cols, seed=1000 + rows * 17 + cols)
    orth_rms = float(np.sqrt(np.mean(np.square(orth))))
    assert orth_rms == pytest.approx(1.0 / root, abs=1e-6)
    scaled_rms = float(np.sqrt(np.mean(np.square(scale * orth))))
    assert scaled_rms == pytest.approx(0.2, abs=1e-5)


def test_muon_rms_scale_depends_on_max_side_only() -> None:
    """Swapping sides does not change the scale. The smaller side does not either."""
    tall = train.muon_rms_scale(8, 3)
    wide = train.muon_rms_scale(3, 8)
    assert tall == pytest.approx(wide, rel=1e-6, abs=1e-6)
    left = train.muon_rms_scale(10, 1)
    assert left == pytest.approx(train.muon_rms_scale(10, 10), rel=1e-6, abs=1e-6)
    assert left == pytest.approx(train.muon_rms_scale(2, 10), rel=1e-6, abs=1e-6)
    # Strictly increasing in max(rows, cols).
    prev = _assert_real_scalar(train.muon_rms_scale(1, 1))
    for n in (2, 3, 4, 8, 9):
        got = _assert_real_scalar(train.muon_rms_scale(n, 1))
        assert got > prev
        prev = got


@pytest.mark.parametrize(
    "rows,cols",
    [
        (0, 1),
        (1, 0),
        (-1, 4),
        (4, -3),
        (1.0, 4),
        (4, 2.0),
        (1.5, 2),
        (2, 2.5),
        (None, 2),
        (2, None),
        ("4", 4),
        (4, "4"),
    ],
)
def test_muon_rms_scale_rejects_bad_dimensions(rows: object, cols: object) -> None:
    with pytest.raises(ValueError):
        train.muon_rms_scale(rows, cols)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# transferred_muon_lr — width-independent under the RMS match
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width,base_width",
    [(1, 1), (1, 128), (128, 1), (128, 128), (2048, 128), (128, 2048), (7, 13), (10**6, 3)],
)
@pytest.mark.parametrize("base_lr", [0.0, 0.02, 1.0, 1e-12, 100.0])
def test_transferred_muon_lr_is_base_lr(base_lr: float, width: int, base_width: int) -> None:
    """Shape scale stays in muon_rms_scale. Width does not rescale the rate."""
    got = train.transferred_muon_lr(base_lr, width=width, base_width=base_width)
    _assert_real_scalar(got)
    assert got == base_lr
    assert got == ref.transferred_muon_lr(base_lr, width=width, base_width=base_width)
    # Adam μP (base/width) and a folded Moonlight factor are both the wrong transfer.
    if width != base_width and base_lr > 0.0:
        adam = base_lr * (base_width / width)
        folded = base_lr * math.sqrt(width / base_width)
        assert got != pytest.approx(adam, rel=1e-3, abs=0.0)
        assert got != pytest.approx(folded, rel=1e-3, abs=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_lr": -1e-8, "width": 4, "base_width": 4},
        {"base_lr": -1.0, "width": 8, "base_width": 2},
        {"base_lr": float("nan"), "width": 4, "base_width": 4},
        {"base_lr": float("inf"), "width": 4, "base_width": 4},
        {"base_lr": float("-inf"), "width": 4, "base_width": 4},
        {"base_lr": 0.02, "width": 0, "base_width": 4},
        {"base_lr": 0.02, "width": -3, "base_width": 4},
        {"base_lr": 0.02, "width": 1.0, "base_width": 4},
        {"base_lr": 0.02, "width": 4, "base_width": 0},
        {"base_lr": 0.02, "width": 4, "base_width": -1},
        {"base_lr": 0.02, "width": 4, "base_width": 2.0},
        {"base_lr": 0.02, "width": 4.5, "base_width": 4},
        {"base_lr": "0.02", "width": 4, "base_width": 4},
        {"base_lr": None, "width": 4, "base_width": 4},
    ],
)
def test_transferred_muon_lr_rejects_bad_args(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        train.transferred_muon_lr(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# muon_transfer_step — equation 4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", STEP_SHAPES)
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
@pytest.mark.parametrize("ns_steps", [1, 5])
def test_transfer_step_matches_reference_and_newton_schulz(
    shape: tuple[int, int], weight_decay: float, ns_steps: int
) -> None:
    """Equation 4: subtract lr * (scale * O + wd * param). O is production NS."""
    param, grad, momentum = _mats(shape, seed=50 + shape[0] * 20 + shape[1] + ns_steps)
    lr = 0.02
    beta = 0.95
    kw = dict(lr=lr, momentum_coeff=beta, ns_steps=ns_steps, weight_decay=weight_decay)
    new_p, new_m = _step(param, grad, momentum, **kw)
    ref_p, ref_m = _ref_step(param, grad, momentum, **kw)
    _close(new_p, ref_p)
    _close(new_m, ref_m)

    beta32 = np.float32(beta)
    expect_m = beta32 * momentum + grad
    _close(new_m, expect_m)
    nesterov = grad + beta32 * expect_m
    orth = train.newton_schulz(nesterov, ns_steps)
    scale = np.float32(_assert_real_scalar(train.muon_rms_scale(*shape)))
    expect_p = param - np.float32(lr) * (scale * _np(orth) + np.float32(weight_decay) * param)
    _close(new_p, expect_p)

    _delta, muon_m = train.muon_update(
        grad, momentum, lr=lr, momentum_coeff=beta, ns_steps=ns_steps
    )
    _close(new_m, muon_m)
    assert _np(new_p).shape == shape
    assert _np(new_m).shape == shape
    assert _np(new_p).dtype == np.float32
    assert _np(new_m).dtype == np.float32


def test_transfer_step_golden_scalars() -> None:
    """Frozen NumPy-reference scalars, not fitted to production."""
    new_p, new_m = _step(
        GOLDEN_PARAM,
        GOLDEN_GRAD,
        GOLDEN_MOMENTUM,
        lr=GOLDEN_LR,
        momentum_coeff=GOLDEN_BETA,
        ns_steps=GOLDEN_STEPS,
        weight_decay=GOLDEN_WD,
    )
    assert float(_np(new_p)[0, 0]) == pytest.approx(GOLDEN_P00, rel=1e-5, abs=1e-5)
    assert float(_np(new_p).sum()) == pytest.approx(GOLDEN_PSUM, rel=1e-5, abs=1e-5)
    assert float(_np(new_m)[0, 0]) == pytest.approx(GOLDEN_M00, rel=1e-5, abs=1e-5)
    assert float(_np(new_m).sum()) == pytest.approx(GOLDEN_MSUM, rel=1e-5, abs=1e-5)
    bare_p, bare_m = _step(
        GOLDEN_PARAM,
        GOLDEN_GRAD,
        GOLDEN_MOMENTUM,
        lr=GOLDEN_LR,
        momentum_coeff=GOLDEN_BETA,
        ns_steps=GOLDEN_STEPS,
        weight_decay=0.0,
    )
    assert float(_np(bare_p)[0, 0]) == pytest.approx(GOLDEN_BARE_P00, rel=1e-5, abs=1e-5)
    assert float(_np(bare_p).sum()) == pytest.approx(GOLDEN_BARE_PSUM, rel=1e-5, abs=1e-5)
    _close(bare_m, new_m)


def test_weight_decay_is_decoupled_from_newton_schulz() -> None:
    """Non-zero wd adds lr * wd * param to the subtracted term and does not touch O."""
    lr = 0.2
    wd = 0.5
    beta = 0.9
    for shape in ((3, 2), (2, 5), (4, 4), (8, 1), (1, 8)):
        param, grad, momentum = _mats(shape, seed=80 + shape[0])
        bare_p, bare_m = _step(
            param, grad, momentum, lr=lr, momentum_coeff=beta, ns_steps=5, weight_decay=0.0
        )
        wd_p, wd_m = _step(
            param, grad, momentum, lr=lr, momentum_coeff=beta, ns_steps=5, weight_decay=wd
        )
        _close(wd_m, bare_m)
        _close(_np(bare_p) - _np(wd_p), np.float32(lr) * np.float32(wd) * param)
        # Same identity written as the subtracted term.
        subtracted = param - _np(wd_p)
        bare_subtracted = param - _np(bare_p)
        _close(subtracted - bare_subtracted, np.float32(lr) * np.float32(wd) * param)


def test_zero_weight_decay_direction_is_not_jordan_scale() -> None:
    """Moonlight scale is not muon_update's sqrt(max(1, rows/cols))."""
    lr = 0.02
    beta = 0.95
    steps = 5
    for shape in ((3, 2), (2, 5), (8, 3), (4, 4)):
        moon = 0.2 * math.sqrt(max(shape))
        jordan = math.sqrt(max(1.0, shape[0] / shape[1]))
        assert abs(moon - jordan) > 1e-2
        param, grad, momentum = _mats(shape, seed=90)
        new_p, new_m = _step(
            param, grad, momentum, lr=lr, momentum_coeff=beta, ns_steps=steps, weight_decay=0.0
        )
        delta, muon_m = train.muon_update(
            grad, momentum, lr=lr, momentum_coeff=beta, ns_steps=steps
        )
        _close(new_m, muon_m)
        jordan_param = param - _np(delta)
        gap = float(np.max(np.abs(_np(new_p) - jordan_param)))
        assert gap > 1e-3, shape


def test_zero_grad_update_is_pure_weight_decay() -> None:
    """NS of a zero Nesterov momentum is zero, so only the decay term remains."""
    lr = 0.02
    wd = 0.1
    for shape in ((1, 1), (3, 2), (5, 4)):
        param, _grad, _momentum = _mats(shape, seed=11)
        zeros = np.zeros(shape, dtype=np.float32)
        new_p, new_m = _step(
            param, zeros, zeros, lr=lr, momentum_coeff=0.95, ns_steps=5, weight_decay=wd
        )
        _close(new_m, zeros)
        _close(new_p, param * np.float32(1.0 - lr * wd))
        ref_p, ref_m = _ref_step(
            param, zeros, zeros, lr=lr, momentum_coeff=0.95, ns_steps=5, weight_decay=wd
        )
        _close(new_p, ref_p)
        _close(new_m, ref_m)


def test_zero_lr_holds_param_and_still_updates_momentum() -> None:
    param, grad, momentum = _mats((3, 2), seed=12)
    beta = 0.95
    new_p, new_m = _step(
        param, grad, momentum, lr=0.0, momentum_coeff=beta, ns_steps=5, weight_decay=0.1
    )
    _close(new_p, param)
    _close(new_m, np.float32(beta) * momentum + grad)


def test_momentum_coeff_zero_nesterov_is_the_grad() -> None:
    """β = 0 → momentum buffer is g and NS sees g, not a mix with the old buffer."""
    param, grad, momentum = _mats((3, 2), seed=13)
    lr = 0.05
    new_p, new_m = _step(
        param, grad, momentum, lr=lr, momentum_coeff=0.0, ns_steps=1, weight_decay=0.0
    )
    _close(new_m, grad)
    orth = train.newton_schulz(grad, 1)
    scale = np.float32(train.muon_rms_scale(3, 2))
    _close(new_p, param - np.float32(lr) * scale * _np(orth))


def test_transfer_step_does_not_mutate_inputs() -> None:
    param, grad, momentum = _mats((4, 3), seed=14)
    p, g, m = param.copy(), grad.copy(), momentum.copy()
    _step(p, g, m, weight_decay=0.1)
    assert np.array_equal(p, param)
    assert np.array_equal(g, grad)
    assert np.array_equal(m, momentum)


def test_transfer_step_float64_input_matches_float32_reference() -> None:
    param, grad, momentum = _mats((3, 2), seed=16)
    new_p, new_m = _step(
        param.astype(np.float64),
        grad.astype(np.float64),
        momentum.astype(np.float64),
        weight_decay=0.1,
    )
    assert _np(new_p).dtype == np.float32
    assert _np(new_m).dtype == np.float32
    ref_p, ref_m = _ref_step(param, grad, momentum, weight_decay=0.1)
    _close(new_p, ref_p)
    _close(new_m, ref_m)


def test_two_step_momentum_carry_matches_reference() -> None:
    shape = (3, 2)
    param, grad, momentum = _mats(shape, seed=17)
    grad2 = np.random.default_rng(18).standard_normal(shape).astype(np.float32)
    kw = dict(lr=0.02, momentum_coeff=0.95, ns_steps=5, weight_decay=0.1)
    p1, m1 = _step(param, grad, momentum, **kw)
    p2, m2 = _step(_np(p1), grad2, _np(m1), **kw)
    rp1, rm1 = _ref_step(param, grad, momentum, **kw)
    rp2, rm2 = _ref_step(rp1, grad2, rm1, **kw)
    _close(p1, rp1)
    _close(m1, rm1)
    _close(p2, rp2)
    _close(m2, rm2)


@pytest.mark.parametrize(
    "fault",
    [
        "wd_negative",
        "wd_nan",
        "wd_string",
        "lr_nan",
        "lr_inf",
        "lr_neg_inf",
        "lr_string",
        "rank1",
        "rank3",
        "rank0",
        "shape_mismatch",
        "momentum_mismatch",
        "ns_even",
        "ns_zero",
        "ns_negative",
    ],
)
def test_transfer_step_rejects_faults(fault: str) -> None:
    param = np.ones((2, 3), dtype=np.float32)
    grad = np.ones((2, 3), dtype=np.float32)
    momentum = np.ones((2, 3), dtype=np.float32)
    lr: object = 0.02
    wd: object = 0.0
    steps = 1
    if fault == "wd_negative":
        wd = -1e-8
    elif fault == "wd_nan":
        wd = float("nan")
    elif fault == "wd_string":
        wd = "0.1"
    elif fault == "lr_nan":
        lr = float("nan")
    elif fault == "lr_inf":
        lr = float("inf")
    elif fault == "lr_neg_inf":
        lr = float("-inf")
    elif fault == "lr_string":
        lr = "0.02"
    elif fault == "rank1":
        param = grad = momentum = np.ones((6,), dtype=np.float32)
    elif fault == "rank3":
        param = grad = momentum = np.ones((2, 3, 1), dtype=np.float32)
    elif fault == "rank0":
        param = grad = momentum = np.ones((), dtype=np.float32)
    elif fault == "shape_mismatch":
        grad = np.ones((3, 2), dtype=np.float32)
    elif fault == "momentum_mismatch":
        momentum = np.ones((2, 1), dtype=np.float32)
    elif fault == "ns_even":
        steps = 2
    elif fault == "ns_zero":
        steps = 0
    elif fault == "ns_negative":
        steps = -1
    with pytest.raises(ValueError):
        train.muon_transfer_step(
            param,
            grad,
            momentum,
            lr=lr,  # type: ignore[arg-type]
            momentum_coeff=0.95,
            ns_steps=steps,
            weight_decay=wd,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Gradients. The decay path is exactly linear; the NS path is checked by FD.
# ---------------------------------------------------------------------------


def _central(fn, x: np.ndarray, direction: np.ndarray, eps: float = 1e-3) -> float:
    return (fn(x + eps * direction) - fn(x - eps * direction)) / (2.0 * eps)


def test_param_finite_difference_is_decay_shrink() -> None:
    """d(new_param)/d(param) = (1 - lr * weight_decay) I. O does not depend on param."""
    param, grad, momentum = _mats((3, 2), seed=21)
    lr = 0.2
    wd = 0.25
    direction = np.random.default_rng(22).standard_normal(param.shape).astype(np.float32)

    def loss(p: np.ndarray) -> float:
        new_p, _new_m = _step(
            p, grad, momentum, lr=lr, momentum_coeff=0.9, ns_steps=5, weight_decay=wd
        )
        return float(np.sum(_np(new_p)))

    fd = _central(loss, param, direction)
    expect = (1.0 - lr * wd) * float(np.sum(direction))
    assert fd == pytest.approx(expect, rel=1e-4, abs=1e-4)
    # Same closed form against the reference, so a shared bug cannot hide.
    def ref_loss(p: np.ndarray) -> float:
        new_p, _new_m = _ref_step(
            p, grad, momentum, lr=lr, momentum_coeff=0.9, ns_steps=5, weight_decay=wd
        )
        return float(np.sum(new_p))

    assert fd == pytest.approx(_central(ref_loss, param, direction), rel=1e-4, abs=1e-4)


def test_momentum_finite_difference() -> None:
    param, grad, momentum = _mats((3, 2), seed=23)
    beta = 0.95
    direction = np.random.default_rng(24).standard_normal(momentum.shape).astype(np.float32)

    def loss_m(m: np.ndarray) -> float:
        _new_p, new_m = _step(param, grad, m, momentum_coeff=beta, weight_decay=0.1)
        return float(np.sum(_np(new_m)))

    fd_m = _central(loss_m, momentum, direction)
    assert fd_m == pytest.approx(beta * float(np.sum(direction)), rel=1e-4, abs=1e-4)

    def loss_g(g: np.ndarray) -> float:
        _new_p, new_m = _step(param, g, momentum, momentum_coeff=beta, weight_decay=0.1)
        return float(np.sum(_np(new_m)))

    g_dir = np.random.default_rng(25).standard_normal(grad.shape).astype(np.float32)
    fd_g = _central(loss_g, grad, g_dir)
    assert fd_g == pytest.approx(float(np.sum(g_dir)), rel=1e-4, abs=1e-4)


def test_grad_finite_difference_matches_reference() -> None:
    """Directional derivative through Newton–Schulz matches the NumPy oracle."""
    param, grad, momentum = _mats((3, 2), seed=26)
    direction = np.random.default_rng(27).standard_normal(grad.shape).astype(np.float32)
    kw = dict(lr=0.02, momentum_coeff=0.8, ns_steps=1, weight_decay=0.0)

    def loss(g: np.ndarray) -> float:
        new_p, new_m = _step(param, g, momentum, **kw)
        return float(np.sum(_np(new_p)) + 0.3 * np.sum(_np(new_m)))

    def ref_loss(g: np.ndarray) -> float:
        new_p, new_m = _ref_step(param, g, momentum, **kw)
        return float(np.sum(new_p) + 0.3 * np.sum(new_m))

    fd = _central(loss, grad, direction, eps=1e-2)
    ref_fd = _central(ref_loss, grad, direction, eps=1e-2)
    assert fd == pytest.approx(ref_fd, rel=1e-2, abs=1e-3)


def test_jax_grad_matches_finite_difference_and_decay_closed_form() -> None:
    param, grad, momentum = _mats((3, 2), seed=28)
    lr = 0.2
    wd = 0.25
    beta = 0.9
    p_j = jnp.asarray(param)
    g_j = jnp.asarray(grad)
    m_j = jnp.asarray(momentum)

    def loss_p(p):
        new_p, _new_m = train.muon_transfer_step(
            p, g_j, m_j, lr=lr, momentum_coeff=beta, ns_steps=1, weight_decay=wd
        )
        return jnp.sum(new_p)

    analytic = jax.grad(loss_p)(p_j)
    expect = jnp.full(param.shape, np.float32(1.0 - lr * wd))
    _close(analytic, expect, rtol=1e-5, atol=1e-5)

    direction = np.random.default_rng(29).standard_normal(grad.shape).astype(np.float32)
    direction = jnp.asarray(direction)

    def loss_g(g):
        new_p, new_m = train.muon_transfer_step(
            p_j, g, m_j, lr=0.02, momentum_coeff=0.8, ns_steps=1, weight_decay=0.0
        )
        return jnp.sum(new_p) + 0.3 * jnp.sum(new_m)

    analytic_g = jax.grad(loss_g)(g_j)
    eps = 1e-3
    fd = (loss_g(g_j + eps * direction) - loss_g(g_j - eps * direction)) / (2.0 * eps)
    directional = jnp.sum(analytic_g * direction)
    assert float(directional) == pytest.approx(float(fd), rel=1e-3, abs=1e-4)


# ---------------------------------------------------------------------------
# jit. Scalar kwargs closed over, or static. Must not trace-convert arrays.
# ---------------------------------------------------------------------------


def _jit_inputs(shape: tuple[int, int], seed: int):
    param, grad, momentum = _mats(shape, seed)
    return jnp.asarray(param), jnp.asarray(grad), jnp.asarray(momentum)


@pytest.mark.parametrize("shape", [(3, 2), (2, 5), (4, 4)])
def test_jit_closed_over_matches_eager(shape: tuple[int, int]) -> None:
    """V1-compatible CPU jit: closed-over scalars match eager and the reference at 1e-5."""
    param, grad, momentum = _jit_inputs(shape, seed=30 + shape[0])
    lr, beta, steps, wd = 0.02, 0.95, 5, 0.1

    def body(p, g, m):
        return train.muon_transfer_step(
            p, g, m, lr=lr, momentum_coeff=beta, ns_steps=steps, weight_decay=wd
        )

    eager_p, eager_m = body(param, grad, momentum)
    jit_p, jit_m = jax.jit(body)(param, grad, momentum)
    _close(jit_p, eager_p)
    _close(jit_m, eager_m)
    ref_p, ref_m = _ref_step(
        _np(param),
        _np(grad),
        _np(momentum),
        lr=lr,
        momentum_coeff=beta,
        ns_steps=steps,
        weight_decay=wd,
    )
    _close(jit_p, ref_p)
    _close(jit_m, ref_m)


@pytest.mark.parametrize("shape", [(3, 2), (2, 5)])
def test_jit_static_kwargs_matches_eager(shape: tuple[int, int]) -> None:
    param, grad, momentum = _jit_inputs(shape, seed=40 + shape[1])
    compiled = jax.jit(
        train.muon_transfer_step,
        static_argnames=("lr", "momentum_coeff", "ns_steps", "weight_decay"),
    )
    kw = dict(lr=0.02, momentum_coeff=0.95, ns_steps=5, weight_decay=0.0)
    eager_p, eager_m = train.muon_transfer_step(param, grad, momentum, **kw)
    jit_p, jit_m = compiled(param, grad, momentum, **kw)
    _close(jit_p, eager_p)
    _close(jit_m, eager_m)
    ref_p, ref_m = _ref_step(_np(param), _np(grad), _np(momentum), **kw)
    _close(jit_p, ref_p)
    _close(jit_m, ref_m)


@pytest.mark.gpu
def test_v1_gpu_transfer_step_matches_reference() -> None:
    """V1: Moonlight Muon step on GPU matches the NumPy reference at 1e-5."""
    _require_gpu()
    param, grad, momentum = _mats((4, 3), seed=41)
    new_p, new_m = _step(param, grad, momentum, weight_decay=0.1)
    ref_p, ref_m = _ref_step(param, grad, momentum, weight_decay=0.1)
    _close(new_p, ref_p)
    _close(new_m, ref_m)
