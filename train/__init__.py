"""Train step: MuonClip, WSD, mixed precision, minimal loader (spec 5.3, 5.4, 15.5 A2).

Optimizer split (5.3):
- Muon + QK-clip on 2D matrices (hidden weights). Per-expert Newton-Schulz
  runs locally because expert matrices already live on one GPU.
- AdamW on embeddings, norms, router biases, and latent sigma.

Precision (5.3):
- Master weights stay FP32.
- Gradient all-reduce is BF16.
- Linears are FP8 with per-block scaling (NVFP4 is a later swap, V3).
- Router, norms, embeddings, softmax, the last two layers, and the latent
  adapter stay BF16.

Schedule is WSD (warmup, stable, decay). A stable-phase checkpoint is a
valid starting point for a later decay. Loss is next-token CE plus MTP
heads plus router z-loss, with logit soft-capping (3.1).

Aux-loss-free router bias updates live here, not in the A1 forward.

``jax.jit`` of ``newton_schulz``, ``muon_update``, ``adamw_update``,
``qk_clip``, ``soft_cap``, ``cross_entropy``, ``mtp_loss``, ``z_loss``,
``total_loss``, and ``train_step`` must match the eager call at 1e-5.
Traced arrays must not be converted with ``numpy.asarray`` or Python
``float()`` / ``int()`` on values. Python scalars (``steps``, ``lr``,
``ns_steps``, ``step``, ``max_logit``, ``cap``) and ``TrainConfig`` /
``ModelConfig`` stay host-side. After ``value_and_grad``, traced grads
and losses must stay in JAX: no Python ``float()`` on the loss leaves,
no ``numpy.asarray`` on QK-clip / router-bias updates, no Python
branch on a traced ``grad_norm``.

``Batch``, ``LossBreakdown``, and ``StepOutput`` must be jax.tree_util
registered dataclasses so ``jax.jit`` can take a ``Batch`` and return a
``StepOutput`` without unpacking fields to tuples. Analog of ``Fp8Meta``
/ ``NoisyLatent`` / ``DispatchMeta``. ``Batch`` fields ``tokens``,
``loss_mask``, ``positions`` are data (arrays). ``LossBreakdown`` fields
``ce``, ``mtp``, ``z``, ``total`` are data (arrays). ``StepOutput``
fields ``params``, ``opt_state``, ``loss``, ``grad_norm`` are data
(dicts of arrays, nested ``LossBreakdown``, array). ``lr`` (Python
float) and ``step`` (Python int) are meta. Do not register
``TrainConfig``. Do not jit ``apply_precision`` / ``init_opt_state`` /
``wsd_lr`` as entry points. Analog math unchanged.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import model
from model import FLAGSHIP_N_LAYERS, ModelConfig

Array = Any  # numpy.ndarray or jax.Array


class ParamKind(StrEnum):
    """Which optimizer a parameter takes. 2D hidden weights are Muon."""

    MUON_2D = "muon_2d"
    ADAMW = "adamw"


class PrecisionKind(StrEnum):
    """Cast used in the forward, not the master-weight dtype."""

    FP8_LINEAR = "fp8_linear"
    BF16_STABLE = "bf16_stable"
    FP32_MASTER = "fp32_master"


class TrainConfigError(ValueError):
    """TrainConfig violates a 5.3 / 5.4 invariant."""


# spec 5.3 / 5.4 flagship-scale numbers. Tiny() is the V1 stand-in.
FLAGSHIP_PEAK_LR = 2.0e-4
FLAGSHIP_MUON_LR = 2.0e-2
FLAGSHIP_ADAMW_LR = 2.0e-4
FLAGSHIP_ADAMW_BETA1 = 0.9
FLAGSHIP_ADAMW_BETA2 = 0.95
FLAGSHIP_ADAMW_EPS = 1.0e-8
FLAGSHIP_ADAMW_WD = 0.1
FLAGSHIP_MUON_MOMENTUM = 0.95
FLAGSHIP_MUON_NS_STEPS = 5
FLAGSHIP_WARMUP_STEPS = 2_000
FLAGSHIP_STABLE_STEPS = 500_000
FLAGSHIP_DECAY_STEPS = 100_000
FLAGSHIP_Z_LOSS_WEIGHT = 1.0e-3
FLAGSHIP_SOFTCAP = 30.0
FLAGSHIP_QK_CLIP = 100.0
FLAGSHIP_FP8_BLOCK = 128
FLAGSHIP_GRAD_CLIP = 1.0
# Last two unique layers stay BF16 even when the rest of the stack is FP8.
FLAGSHIP_BF16_TAIL_LAYERS = 2

_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315
_NS_EPS = 1e-7
_ADAMW_NAME_TOKS = ("embed", "norm", "bias", "sigma")
_BF16_NAME_TOKS = ("router", "norm", "embed", "adapter", "softmax", "sigma")


@dataclass(frozen=True)
class TrainConfig:
    """Discrete training choices. Flagship numbers are 5.3; tiny() is V1."""

    peak_lr: float
    muon_lr: float
    adamw_lr: float
    adamw_beta1: float
    adamw_beta2: float
    adamw_eps: float
    adamw_wd: float
    muon_momentum: float
    muon_ns_steps: int
    warmup_steps: int
    stable_steps: int
    decay_steps: int
    z_loss_weight: float
    softcap: float
    qk_clip: float
    fp8_block: int
    grad_clip: float
    bf16_tail_layers: int
    n_layers: int


def flagship_train_config() -> TrainConfig:
    return TrainConfig(
        peak_lr=FLAGSHIP_PEAK_LR,
        muon_lr=FLAGSHIP_MUON_LR,
        adamw_lr=FLAGSHIP_ADAMW_LR,
        adamw_beta1=FLAGSHIP_ADAMW_BETA1,
        adamw_beta2=FLAGSHIP_ADAMW_BETA2,
        adamw_eps=FLAGSHIP_ADAMW_EPS,
        adamw_wd=FLAGSHIP_ADAMW_WD,
        muon_momentum=FLAGSHIP_MUON_MOMENTUM,
        muon_ns_steps=FLAGSHIP_MUON_NS_STEPS,
        warmup_steps=FLAGSHIP_WARMUP_STEPS,
        stable_steps=FLAGSHIP_STABLE_STEPS,
        decay_steps=FLAGSHIP_DECAY_STEPS,
        z_loss_weight=FLAGSHIP_Z_LOSS_WEIGHT,
        softcap=FLAGSHIP_SOFTCAP,
        qk_clip=FLAGSHIP_QK_CLIP,
        fp8_block=FLAGSHIP_FP8_BLOCK,
        grad_clip=FLAGSHIP_GRAD_CLIP,
        bf16_tail_layers=FLAGSHIP_BF16_TAIL_LAYERS,
        n_layers=FLAGSHIP_N_LAYERS,
    )


def tiny_train_config() -> TrainConfig:
    """V1 CPU stand-in. Same discrete choices, short WSD, tiny clip/block."""
    return TrainConfig(
        peak_lr=FLAGSHIP_PEAK_LR,
        muon_lr=FLAGSHIP_MUON_LR,
        adamw_lr=FLAGSHIP_ADAMW_LR,
        adamw_beta1=FLAGSHIP_ADAMW_BETA1,
        adamw_beta2=FLAGSHIP_ADAMW_BETA2,
        adamw_eps=FLAGSHIP_ADAMW_EPS,
        adamw_wd=FLAGSHIP_ADAMW_WD,
        muon_momentum=FLAGSHIP_MUON_MOMENTUM,
        muon_ns_steps=FLAGSHIP_MUON_NS_STEPS,
        warmup_steps=2,
        stable_steps=8,
        decay_steps=4,
        z_loss_weight=FLAGSHIP_Z_LOSS_WEIGHT,
        softcap=FLAGSHIP_SOFTCAP,
        qk_clip=FLAGSHIP_QK_CLIP,
        fp8_block=16,
        grad_clip=FLAGSHIP_GRAD_CLIP,
        bf16_tail_layers=FLAGSHIP_BF16_TAIL_LAYERS,
        n_layers=8,
    )


@dataclass(frozen=True)
class Batch:
    """One packed training batch. Tokens are int32; loss_mask is 0/1 float.

    Must be a jax.tree_util registered dataclass so jax.jit can take it.
    tokens, loss_mask, positions are data fields (arrays). No meta fields.
    Analog of Fp8Meta (arrays as data).
    """

    tokens: Array
    loss_mask: Array
    positions: Array


@dataclass(frozen=True)
class LossBreakdown:
    """CE on the next token, MTP heads, router z-loss, and the weighted sum.

    Must be a jax.tree_util registered dataclass so jax.jit(train_step) can
    return it nested in StepOutput. ce, mtp, z, total are data fields
    (arrays). No meta fields. Analog of NoisyLatent.
    """

    ce: Array
    mtp: Array
    z: Array
    total: Array


@dataclass(frozen=True)
class StepOutput:
    """One optimizer step. `params` are the new FP32 master weights.

    Must be a jax.tree_util registered dataclass so jax.jit(train_step) can
    return it without unpacking. params, opt_state, loss, grad_norm are data
    fields (dicts of arrays, nested LossBreakdown, array). lr (Python float)
    and step (Python int) are meta. Analog of DispatchMeta (arrays data,
    Python scalars meta).
    """

    params: dict[str, Any]
    opt_state: dict[str, Any]
    loss: LossBreakdown
    lr: float
    grad_norm: Array
    step: int


jax.tree_util.register_dataclass(
    Batch,
    data_fields=("tokens", "loss_mask", "positions"),
    meta_fields=(),
)
jax.tree_util.register_dataclass(
    LossBreakdown,
    data_fields=("ce", "mtp", "z", "total"),
    meta_fields=(),
)
jax.tree_util.register_dataclass(
    StepOutput,
    data_fields=("params", "opt_state", "loss", "grad_norm"),
    meta_fields=("lr", "step"),
)


def validate_train_config(config: TrainConfig) -> None:
    """Raise TrainConfigError if WSD lengths, lrs, or tail layers do not hold."""
    if min(float(config.peak_lr), float(config.muon_lr), float(config.adamw_lr)) <= 0:
        raise TrainConfigError("peak_lr, muon_lr, and adamw_lr must be > 0")
    total = int(config.warmup_steps) + int(config.stable_steps) + int(config.decay_steps)
    if total <= 0:
        raise TrainConfigError("warmup + stable + decay must be > 0")
    if int(config.bf16_tail_layers) > int(config.n_layers):
        raise TrainConfigError("bf16_tail_layers must be <= n_layers")
    return None


def classify_param(name: str, value: Array) -> ParamKind:
    """2D hidden weights are Muon; embeddings, norms, biases, sigma are AdamW."""
    low = name.lower()
    if any(tok in low for tok in _ADAMW_NAME_TOKS):
        return ParamKind.ADAMW
    if int(jnp.asarray(value).ndim) == 2:
        return ParamKind.MUON_2D
    return ParamKind.ADAMW


def precision_for(name: str, config: TrainConfig) -> PrecisionKind:
    """FP8 on hidden linears; BF16 on router/norm/embed/softmax/tail/adapter."""
    low = name.lower()
    match = re.match(r"layers\.(\d+)\.", low)
    if match is not None:
        idx = int(match.group(1))
        if idx >= int(config.n_layers) - int(config.bf16_tail_layers):
            return PrecisionKind.BF16_STABLE
    if any(tok in low for tok in _BF16_NAME_TOKS):
        return PrecisionKind.BF16_STABLE
    return PrecisionKind.FP8_LINEAR


def _as_f32(x: Array) -> Array:
    """Cast to float32 without ``numpy.asarray`` (tracer-safe under ``jax.jit``)."""
    return jnp.asarray(x, dtype=jnp.float32)


def _positive_int(name: str, value: object) -> int:
    """Host-side integer >= 1. Rejects bool (an ``int`` subclass) and floats."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return value


def _finite_real(name: str, value: object) -> float:
    """Host-side finite real. Not for traced arrays (no ``float()`` on tracers)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite real number")
    return out


def _logsumexp(x: Array, axis: int = -1) -> Array:
    x = _as_f32(x)
    m = jnp.max(x, axis=axis, keepdims=True)
    s = jnp.log(jnp.maximum(jnp.exp(x - m).sum(axis=axis, keepdims=True), 1e-12))
    return jnp.squeeze(m + s, axis=axis)


def newton_schulz(matrix: Array, steps: int) -> Array:
    """Newton-Schulz orthogonalization of a 2D gradient. Odd `steps` >= 1.

    ``jax.jit`` of this function, with ``steps`` a Python int (closed over
    or ``static_argnums``), must match the eager result at 1e-5. Must not
    call ``numpy.asarray`` or Python ``float()`` on traced ``matrix``.
    """
    if int(steps) < 1 or int(steps) % 2 == 0:
        raise ValueError("newton_schulz steps must be odd and >= 1")
    g = _as_f32(matrix)
    if g.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2D matrix, got shape {g.shape}")
    transposed = g.shape[0] > g.shape[1]
    x = g.T if transposed else g
    fro = jnp.linalg.norm(x)
    # Both branches stay in JAX: tiny / zero inputs map to zeros (no Python float()).
    x = jnp.where(
        fro > jnp.float32(_NS_EPS),
        x / jnp.maximum(fro, jnp.float32(_NS_EPS)),
        jnp.zeros_like(x),
    )
    a = jnp.float32(_NS_A)
    b = jnp.float32(_NS_B)
    c = jnp.float32(_NS_C)
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.astype(jnp.float32)


def muon_update(
    grad: Array,
    momentum: Array,
    *,
    lr: float,
    momentum_coeff: float,
    ns_steps: int,
) -> tuple[Array, Array]:
    """Nesterov momentum then Newton-Schulz. Returns (delta, new_momentum).

    ``jax.jit`` of this function, with ``lr``, ``momentum_coeff``, and
    ``ns_steps`` Python scalars (closed over or static), must match the
    eager pair at 1e-5. Must not convert traced ``grad`` or ``momentum``
    with ``numpy.asarray``.
    """
    g = _as_f32(grad)
    m = _as_f32(momentum)
    if g.shape != m.shape:
        raise ValueError(f"grad shape {g.shape} != momentum shape {m.shape}")
    if g.ndim != 2:
        raise ValueError(f"muon_update expects a 2D grad, got shape {g.shape}")
    beta = jnp.float32(momentum_coeff)
    new_m = beta * m + g
    nesterov = g + beta * new_m
    orth = newton_schulz(nesterov, int(ns_steps))
    rows, cols = int(orth.shape[0]), int(orth.shape[1])
    rms = jnp.float32(np.sqrt(max(1.0, rows / max(cols, 1))))
    delta = jnp.float32(lr) * rms * orth
    return delta.astype(jnp.float32), new_m.astype(jnp.float32)


# Moonlight adjusted-LR constant (arXiv 2502.16982 eq. 4). AdamW update RMS
# they matched. Not Jordan's sqrt(max(1, rows/cols)) scale.
MOONLIGHT_UPDATE_RMS = 0.2


def muon_rms_scale(rows: int, cols: int) -> float:
    """Shape scale that makes a Muon update RMS match AdamW (spec 5.4).

    arXiv 2502.16982 Lemma 1: a full-rank orthogonal update of shape
    ``(rows, cols)`` has RMS ``1/sqrt(max(rows, cols))``. Equation 4
    multiplies that update by ``0.2 * sqrt(max(rows, cols))``, so the
    RMS is ``MOONLIGHT_UPDATE_RMS`` for every shape. The same learning
    rate then transfers across widths (μP-for-Muon). ``rows`` and
    ``cols`` must be integers >= 1.

    Does not change ``muon_update``. That function stays on Jordan's
    ``sqrt(max(1, rows/cols))`` scale.
    """
    rows_i = _positive_int("rows", rows)
    cols_i = _positive_int("cols", cols)
    # Lemma 1: orthogonal RMS is 1/sqrt(max(A, B)); eq. 4 cancels that factor.
    return MOONLIGHT_UPDATE_RMS * math.sqrt(max(rows_i, cols_i))


def muon_transfer_step(
    param: Array,
    grad: Array,
    momentum: Array,
    *,
    lr: float,
    momentum_coeff: float,
    ns_steps: int,
    weight_decay: float,
) -> tuple[Array, Array]:
    """One Muon step with Moonlight weight decay and RMS match.

    Same Nesterov momentum and Newton-Schulz as ``muon_update``. The
    parameter update is equation 4 of arXiv 2502.16982:

        new = param - lr * (muon_rms_scale(A, B) * O + weight_decay * param)

    ``O`` is the orthogonalized Nesterov momentum. ``(A, B)`` is
    ``param.shape``. Returns ``(new_param, new_momentum)``.

    ``weight_decay`` must be >= 0. ``lr`` must be finite. ``param``,
    ``grad``, and ``momentum`` must be 2D and the same shape.
    ``jax.jit`` with the scalar kwargs closed over or static must match
    the eager pair at 1e-5. Must not call ``numpy.asarray`` or Python
    ``float()`` on traced arrays.

    ``muon_update`` and ``train_step`` stay on the old scale. This
    function is the transfer rule; it is not wired into the step yet.
    """
    lr_f = _finite_real("lr", lr)
    wd_f = _finite_real("weight_decay", weight_decay)
    if wd_f < 0.0:
        raise ValueError("weight_decay must be >= 0")
    odd = isinstance(ns_steps, int) and not isinstance(ns_steps, bool)
    if not odd or ns_steps < 1 or ns_steps % 2 == 0:
        raise ValueError("ns_steps must be a positive odd integer")

    p = _as_f32(param)
    g = _as_f32(grad)
    m = _as_f32(momentum)
    same = p.ndim == 2 and g.ndim == 2 and m.ndim == 2 and p.shape == g.shape == m.shape
    if not same:
        raise ValueError(
            f"muon_transfer_step expects matching 2D matrices, got {p.shape}, {g.shape}, {m.shape}"
        )
    # Shape is static under jit; the scale is a host float, not a traced reduction.
    rows, cols = int(p.shape[0]), int(p.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"muon_transfer_step expects a non-empty matrix, got {p.shape}")

    beta = jnp.float32(momentum_coeff)
    new_m = beta * m + g
    nesterov = g + beta * new_m
    orth = newton_schulz(nesterov, ns_steps)
    scale = jnp.float32(muon_rms_scale(rows, cols))
    # Eq. 4: decay is decoupled (applied to param, outside Newton-Schulz).
    new_p = p - jnp.float32(lr_f) * (scale * orth + jnp.float32(wd_f) * p)
    return new_p.astype(jnp.float32), new_m.astype(jnp.float32)


def transferred_muon_lr(base_lr: float, *, width: int, base_width: int) -> float:
    """Learning rate to reuse at ``width`` after tuning at ``base_width``.

    Spec 5.4 μP-for-Muon, under the Moonlight RMS match: update RMS does
    not depend on width, so the transferred rate is ``base_lr`` itself.
    The shape scale lives in ``muon_rms_scale``, not here. ``width`` and
    ``base_width`` must be integers >= 1. ``base_lr`` must be finite
    and >= 0.
    """
    lr = _finite_real("base_lr", base_lr)
    if lr < 0.0:
        raise ValueError("base_lr must be finite and >= 0")
    _positive_int("width", width)
    _positive_int("base_width", base_width)
    return lr


def qk_clip(q: Array, k: Array, max_logit: float) -> tuple[Array, Array]:
    """Scale Q or K so max |q k^T| does not exceed `max_logit` (Kimi K2).

    ``jax.jit`` of this function, with ``max_logit`` a Python float, must
    match the eager pair at 1e-5. Must not convert traced ``q`` or ``k``
    with ``numpy.asarray`` or Python ``float()`` on traced maxima.
    """
    if max_logit <= 0:
        raise ValueError("qk_clip max_logit must be > 0")
    q32 = _as_f32(q)
    k32 = _as_f32(k)
    if q32.shape[-1] != k32.shape[-1]:
        raise ValueError("q and k last dim (head dim) must match")
    scores = jnp.einsum("...id,...jd->...ij", q32, k32)
    mu = jnp.max(jnp.abs(scores))
    tau = jnp.float32(max_logit)
    do_scale = (mu > tau) & (mu != jnp.float32(0.0))
    scale = jnp.where(
        do_scale,
        jnp.sqrt(tau / jnp.maximum(mu, jnp.float32(_NS_EPS))),
        jnp.float32(1.0),
    )
    return (q32 * scale).astype(jnp.float32), (k32 * scale).astype(jnp.float32)


def adamw_update(
    param: Array,
    grad: Array,
    m: Array,
    v: Array,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
    step: int,
) -> tuple[Array, Array, Array]:
    """Decoupled AdamW. `step` is 1-based for bias correction. Returns (p, m, v).

    ``jax.jit`` of this function, with ``lr``, ``beta1``, ``beta2``,
    ``eps``, ``wd``, and ``step`` Python scalars (closed over or static),
    must match the eager triple at 1e-5. Must not convert traced
    ``param``, ``grad``, ``m``, or ``v`` with ``numpy.asarray``.
    """
    if int(step) < 1:
        raise ValueError("adamw_update step must be 1-based (>= 1)")
    p = _as_f32(param)
    g = _as_f32(grad)
    m32 = _as_f32(m)
    v32 = _as_f32(v)
    b1 = jnp.float32(beta1)
    b2 = jnp.float32(beta2)
    one = jnp.float32(1.0)
    m_t = b1 * m32 + (one - b1) * g
    v_t = b2 * v32 + (one - b2) * (g * g)
    t = int(step)
    m_hat = m_t / (one - b1**t)
    v_hat = v_t / (one - b2**t)
    adam_step = m_hat / (jnp.sqrt(v_hat) + jnp.float32(eps))
    p_t = p * (one - jnp.float32(lr) * jnp.float32(wd)) - jnp.float32(lr) * adam_step
    return p_t.astype(jnp.float32), m_t.astype(jnp.float32), v_t.astype(jnp.float32)


def wsd_lr(step: int, config: TrainConfig) -> float:
    """Warmup (linear), stable (peak), decay (linear to 0). `step` is 1-based."""
    if int(step) < 1:
        raise ValueError("wsd_lr step must be 1-based (>= 1)")
    peak = float(config.peak_lr)
    warmup = int(config.warmup_steps)
    stable = int(config.stable_steps)
    decay = int(config.decay_steps)
    t = int(step)
    if warmup > 0 and t <= warmup:
        return peak * (t / warmup)
    if t <= warmup + stable:
        return peak
    if decay <= 0 or t >= warmup + stable + decay:
        return 0.0
    decayed = t - warmup - stable
    return peak * (1.0 - decayed / decay)


def soft_cap(logits: Array, cap: float) -> Array:
    """Logit soft-capping: cap * tanh(logits / cap).

    ``jax.jit`` of this function, with ``cap`` a Python float (closed over
    or ``static_argnums``), must match the eager result at 1e-5. Must not
    convert traced ``logits`` with ``numpy.asarray``.
    """
    if cap <= 0:
        raise ValueError("soft_cap cap must be > 0")
    x = _as_f32(logits)
    c = jnp.float32(cap)
    return (c * jnp.tanh(x / c)).astype(jnp.float32)


def cross_entropy(logits: Array, targets: Array, loss_mask: Array) -> Array:
    """Masked mean CE. `logits` are already soft-capped.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``logits``, ``targets``, or ``loss_mask`` with
    ``numpy.asarray`` or Python ``float()`` on the mask sum. Empty mask
    returns 0 without a Python branch on a traced denom.
    """
    logits_f = _as_f32(logits)
    targets_i = jnp.asarray(targets)
    mask = _as_f32(loss_mask)
    if logits_f.shape[:-1] != tuple(targets_i.shape) or logits_f.shape[:-1] != tuple(mask.shape):
        raise ValueError(
            f"shape mismatch: logits {logits_f.shape}, targets {targets_i.shape}, "
            f"mask {mask.shape}"
        )
    vocab = int(logits_f.shape[-1])
    flat = logits_f.reshape(-1, vocab)
    tgt = targets_i.reshape(-1).astype(jnp.int32)
    m = mask.reshape(-1)
    shifted = flat - jnp.max(flat, axis=-1, keepdims=True)
    log_z = jnp.log(jnp.maximum(jnp.exp(shifted).sum(axis=-1), 1e-12))
    nll = -(shifted[jnp.arange(flat.shape[0]), tgt] - log_z)
    denom = m.sum()
    eps = jnp.float32(1e-12)
    return jnp.where(
        denom > 0,
        (nll * m).sum() / jnp.maximum(denom, eps),
        jnp.float32(0.0),
    ).astype(jnp.float32)


def mtp_loss(mtp_logits: tuple[Array, ...], tokens: Array, loss_mask: Array) -> Array:
    """Mean CE of each MTP head on the token `head_index + 1` steps ahead.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``mtp_logits``, ``tokens``, or ``loss_mask``
    with ``numpy.asarray``. Head count is a Python int (tuple length).
    """
    tokens_i = jnp.asarray(tokens)
    mask = _as_f32(loss_mask)
    seq = int(tokens_i.shape[-1])
    losses: list[Array] = []
    for i, logits in enumerate(mtp_logits):
        offset = i + 1
        if offset >= seq:
            continue
        logits_f = _as_f32(logits)
        src = logits_f[..., : seq - offset, :]
        tgt = tokens_i[..., offset:]
        m = mask[..., offset:] * mask[..., : seq - offset]
        losses.append(cross_entropy(src, tgt, m))
    if not losses:
        return jnp.float32(0.0)
    stacked = jnp.stack([_as_f32(x) for x in losses])
    return jnp.mean(stacked).astype(jnp.float32)


def z_loss(router_probs: Array) -> Array:
    """Mean squared log-sum-exp of router probabilities, per token then mean.

    ``jax.jit`` of this function must match the eager result at 1e-5.
    Must not convert traced ``router_probs`` with ``numpy.asarray`` or
    Python ``float()`` on the mean.
    """
    p = _as_f32(router_probs)
    lse = _logsumexp(p, axis=-1)
    return jnp.mean(lse**2).astype(jnp.float32)


def total_loss(
    ce: Array,
    mtp: Array,
    z: Array,
    config: TrainConfig,
) -> Array:
    """ce + mtp + z_loss_weight * z.

    ``jax.jit`` of this function, with ``config`` host-side (closed over
    or static), must match the eager result at 1e-5. Must not convert
    traced ``ce``, ``mtp``, or ``z`` with ``numpy.asarray``.
    """
    w = jnp.float32(float(config.z_loss_weight))
    return (_as_f32(ce) + _as_f32(mtp) + w * _as_f32(z)).astype(jnp.float32)


def _map_tree(obj: Any, fn: Any, prefix: str = "") -> Any:
    if isinstance(obj, dict):
        return {
            k: _map_tree(v, fn, f"{prefix}.{k}" if prefix else str(k)) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_map_tree(v, fn, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(
            _map_tree(v, fn, f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(obj)
        )
    return fn(prefix, obj)


def init_opt_state(params: dict[str, Any], config: TrainConfig) -> dict[str, Any]:
    """FP32 optimizer state matching `classify_param` for every leaf."""
    del config

    def leaf(name: str, p: Any) -> dict[str, np.ndarray]:
        shape = tuple(int(s) for s in np.asarray(p).shape)
        if classify_param(name, p) is ParamKind.MUON_2D:
            return {"momentum": np.zeros(shape, dtype=np.float32)}
        return {
            "m": np.zeros(shape, dtype=np.float32),
            "v": np.zeros(shape, dtype=np.float32),
        }

    return _map_tree(params, leaf)


def apply_precision(
    params: dict[str, Any],
    config: TrainConfig,
) -> dict[str, Any]:
    """Cast a master-weight tree for the forward. Masters themselves stay FP32.

    CPU/V1 keeps values in float32; FP8/BF16 materialization is a GPU concern.
    Each leaf is still routed through `precision_for` so the copy is independent.
    """

    def leaf(name: str, p: Any) -> np.ndarray:
        kind = precision_for(name, config)
        if kind is PrecisionKind.FP32_MASTER:
            raise TrainConfigError("precision_for must not return FP32_MASTER")
        return np.array(p, dtype=np.float32, copy=True)

    return _map_tree(params, leaf)


def _tree_sum_sq(obj: Any) -> Array:
    if isinstance(obj, dict):
        return sum((_tree_sum_sq(v) for v in obj.values()), start=jnp.float32(0.0))
    if isinstance(obj, (list, tuple)):
        return sum((_tree_sum_sq(v) for v in obj), start=jnp.float32(0.0))
    g = _as_f32(obj)
    return jnp.sum(g * g)


def _scale_tree(obj: Any, scale: float) -> Any:
    if isinstance(obj, dict):
        return {k: _scale_tree(v, scale) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scale_tree(v, scale) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_scale_tree(v, scale) for v in obj)
    return _as_f32(obj) * jnp.asarray(scale, dtype=jnp.float32)


def _map_opt(p: Any, g: Any, s: Any, fn: Any, prefix: str = "") -> tuple[Any, Any]:
    if isinstance(p, dict):
        out_p: dict[str, Any] = {}
        out_s: dict[str, Any] = {}
        for k in p:
            name = f"{prefix}.{k}" if prefix else str(k)
            out_p[k], out_s[k] = _map_opt(p[k], g[k], s[k], fn, name)
        return out_p, out_s
    if isinstance(p, (list, tuple)):
        ps = []
        ss = []
        for i, (pi, gi, si) in enumerate(zip(p, g, s, strict=True)):
            name = f"{prefix}.{i}" if prefix else str(i)
            ni, nsi = _map_opt(pi, gi, si, fn, name)
            ps.append(ni)
            ss.append(nsi)
        ctor: Any = list if isinstance(p, list) else tuple
        return ctor(ps), ctor(ss)
    return fn(prefix, p, g, s)


def _soft_cap_j(logits: Array, cap: float) -> Array:
    c = jnp.float32(cap)
    return c * jnp.tanh(logits.astype(jnp.float32) / c)


def _cross_entropy_j(logits: Array, targets: Array, loss_mask: Array) -> Array:
    x = logits.astype(jnp.float32)
    mask = loss_mask.astype(jnp.float32)
    log_z = jax.nn.logsumexp(x, axis=-1)
    gathered = jnp.take_along_axis(x, targets.astype(jnp.int32)[..., None], axis=-1)[..., 0]
    nll = (log_z - gathered) * mask
    denom = jnp.sum(mask)
    return jnp.where(denom > 0, jnp.sum(nll) / denom, jnp.float32(0.0))


def _mtp_loss_j(
    mtp_logits: tuple[Array, ...], tokens: Array, loss_mask: Array, cap: float
) -> Array:
    seq = int(tokens.shape[-1])
    acc = jnp.float32(0.0)
    n_heads = 0
    for i, head in enumerate(mtp_logits):
        offset = i + 1
        if offset >= seq:
            continue
        src = _soft_cap_j(head[..., : seq - offset, :], cap)
        tgt = tokens[..., offset:]
        m = loss_mask[..., offset:] * loss_mask[..., : seq - offset]
        acc = acc + _cross_entropy_j(src, tgt, m)
        n_heads += 1
    if n_heads == 0:
        return jnp.float32(0.0)
    return acc / jnp.float32(n_heads)


def _z_loss_j(router_probs: Array) -> Array:
    p = router_probs.astype(jnp.float32)
    lse = jax.nn.logsumexp(p, axis=-1)
    return jnp.mean(jnp.square(lse))


def _clip_qk_weights(params: Any, max_logit: float) -> Any:
    if isinstance(params, dict):
        out = {k: _clip_qk_weights(v, max_logit) for k, v in params.items()}
        if "W_q" in out and "W_k" in out:
            q, k = out["W_q"], out["W_k"]
            if getattr(q, "shape", None) == getattr(k, "shape", None) and q.ndim >= 2:
                qn = _as_f32(q).reshape(-1, int(q.shape[-1]))
                kn = _as_f32(k).reshape(-1, int(k.shape[-1]))
                qc, kc = qk_clip(qn, kn, max_logit)
                out["W_q"] = jnp.asarray(qc, dtype=jnp.float32).reshape(q.shape)
                out["W_k"] = jnp.asarray(kc, dtype=jnp.float32).reshape(k.shape)
        return out
    if isinstance(params, list):
        return [_clip_qk_weights(v, max_logit) for v in params]
    if isinstance(params, tuple):
        return tuple(_clip_qk_weights(v, max_logit) for v in params)
    return params


def _update_router_bias(params: Any, router_probs: Array, u: float = 1e-3) -> Any:
    """Aux-loss-free: b ← b − u · sign(load − mean). No-op without router_bias."""
    probs = _as_f32(router_probs)
    if probs.size == 0:
        return params
    load = jnp.mean(probs.reshape(-1, probs.shape[-1]), axis=0).astype(jnp.float32)
    delta = jnp.sign(load - jnp.mean(load)).astype(jnp.float32) * jnp.float32(u)

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {k: walk(v) for k, v in obj.items()}
            if "router_bias" in out:
                bias = _as_f32(out["router_bias"])
                n = min(int(bias.shape[-1]), int(delta.shape[0]))
                out["router_bias"] = bias.at[..., :n].add(-delta[:n]).astype(jnp.float32)
            return out
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(walk(v) for v in obj)
        return obj

    return walk(params)


def train_step(
    params: dict[str, Any],
    opt_state: dict[str, Any],
    batch: Batch,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    step: int,
) -> StepOutput:
    """One FP32-master step: forward, loss, MuonClip / AdamW, QK-clip.

    `step` is 1-based. Router bias balancing (aux-loss-free) updates inside
    `opt_state` from the forward's `router_probs`.

    ``jax.jit(train_step)`` (with ``step``, ``model_config``, and
    ``train_config`` closed over or ``static_argnums``) must match the eager
    call at 1e-5 and return a pytree ``StepOutput`` (``Batch`` in, no field
    unpack). Traced ``params`` / ``opt_state`` / batch arrays / grads
    must not be converted with ``numpy.asarray`` or Python ``float()``.
    Grad clip uses a finite ``jnp.where`` (no Python ``if`` on traced
    ``grad_norm``). ``LossBreakdown`` / ``StepOutput`` array fields stay
    JAX types under jit.
    """
    validate_train_config(train_config)
    t = int(step)
    if t < 1:
        raise ValueError("train_step step must be 1-based (>= 1)")
    sched = float(wsd_lr(t, train_config))
    peak = float(train_config.peak_lr)
    mult = sched / peak if peak > 0 else 0.0
    muon_lr_t = float(train_config.muon_lr) * mult
    adamw_lr_t = float(train_config.adamw_lr) * mult

    tokens = jnp.asarray(batch.tokens)
    mask = jnp.asarray(batch.loss_mask, dtype=jnp.float32)
    cap = float(train_config.softcap)
    z_weight = jnp.float32(train_config.z_loss_weight)

    def packed(p: dict[str, Any]) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        out = model.forward(tokens, p, model_config, r=1)
        ce_p = _cross_entropy_j(
            _soft_cap_j(out.logits[:, :-1, :], cap),
            tokens[:, 1:],
            mask[:, 1:],
        )
        mtp_p = _mtp_loss_j(out.mtp_logits, tokens, mask, cap)
        z_p = _z_loss_j(out.router_probs)
        total_p = ce_p + mtp_p + z_weight * z_p
        return total_p, (ce_p, mtp_p, z_p, out.router_probs)

    (total, (ce, mtp, z, router_probs)), grads = jax.value_and_grad(packed, has_aux=True)(
        params
    )

    gnorm = jnp.sqrt(jnp.maximum(_tree_sum_sq(grads), jnp.float32(0.0)))
    clip = float(train_config.grad_clip)
    if clip > 0.0:
        clip_a = jnp.float32(clip)
        scale = jnp.where(
            gnorm > clip_a,
            clip_a / (gnorm + jnp.float32(1e-6)),
            jnp.float32(1.0),
        )
    else:
        scale = jnp.float32(1.0)
    grads = _scale_tree(grads, scale)

    def apply_one(name: str, p: Any, g: Any, s: Any) -> tuple[Any, Any]:
        if classify_param(name, p) is ParamKind.MUON_2D:
            delta, new_m = muon_update(
                g,
                s["momentum"],
                lr=muon_lr_t,
                momentum_coeff=float(train_config.muon_momentum),
                ns_steps=int(train_config.muon_ns_steps),
            )
            new_p = _as_f32(p) - _as_f32(delta)
            return new_p.astype(np.float32), {"momentum": _as_f32(new_m)}
        new_p, new_m, new_v = adamw_update(
            p,
            g,
            s["m"],
            s["v"],
            lr=adamw_lr_t,
            beta1=float(train_config.adamw_beta1),
            beta2=float(train_config.adamw_beta2),
            eps=float(train_config.adamw_eps),
            wd=float(train_config.adamw_wd),
            step=t,
        )
        return _as_f32(new_p), {"m": _as_f32(new_m), "v": _as_f32(new_v)}

    new_params, new_opt = _map_opt(params, grads, opt_state, apply_one)
    if float(train_config.qk_clip) > 0:
        new_params = _clip_qk_weights(new_params, float(train_config.qk_clip))

    new_params = _update_router_bias(new_params, router_probs)

    return StepOutput(
        params=new_params,
        opt_state=new_opt,
        loss=LossBreakdown(
            ce=_as_f32(ce),
            mtp=_as_f32(mtp),
            z=_as_f32(z),
            total=_as_f32(total),
        ),
        lr=sched,
        grad_norm=_as_f32(gnorm),
        step=t,
    )
