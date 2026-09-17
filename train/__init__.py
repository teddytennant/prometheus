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
"""

from __future__ import annotations

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
    """One packed training batch. Tokens are int32; loss_mask is 0/1 float."""

    tokens: Array
    loss_mask: Array
    positions: Array


@dataclass(frozen=True)
class LossBreakdown:
    """CE on the next token, MTP heads, router z-loss, and the weighted sum."""

    ce: Array
    mtp: Array
    z: Array
    total: Array


@dataclass(frozen=True)
class StepOutput:
    """One optimizer step. `params` are the new FP32 master weights."""

    params: dict[str, Any]
    opt_state: dict[str, Any]
    loss: LossBreakdown
    lr: float
    grad_norm: Array
    step: int


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
    if int(np.asarray(value).ndim) == 2:
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


def _as_f32(x: Array) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = _as_f32(x)
    m = np.max(x, axis=axis, keepdims=True)
    s = np.log(np.maximum(np.exp(x - m).sum(axis=axis, keepdims=True), 1e-12))
    return np.squeeze(m + s, axis=axis)


def newton_schulz(matrix: Array, steps: int) -> Array:
    """Newton-Schulz orthogonalization of a 2D gradient. Odd `steps` >= 1."""
    if int(steps) < 1 or int(steps) % 2 == 0:
        raise ValueError("newton_schulz steps must be odd and >= 1")
    g = _as_f32(matrix)
    if g.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2D matrix, got shape {g.shape}")
    transposed = g.shape[0] > g.shape[1]
    x = g.T if transposed else g
    fro = float(np.linalg.norm(x))
    if fro <= _NS_EPS:
        return np.zeros_like(g)
    x = x / np.float32(fro)
    a = np.float32(_NS_A)
    b = np.float32(_NS_B)
    c = np.float32(_NS_C)
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return np.asarray(x, dtype=np.float32)


def muon_update(
    grad: Array,
    momentum: Array,
    *,
    lr: float,
    momentum_coeff: float,
    ns_steps: int,
) -> tuple[Array, Array]:
    """Nesterov momentum then Newton-Schulz. Returns (delta, new_momentum)."""
    g = _as_f32(grad)
    m = _as_f32(momentum)
    if g.shape != m.shape:
        raise ValueError(f"grad shape {g.shape} != momentum shape {m.shape}")
    if g.ndim != 2:
        raise ValueError(f"muon_update expects a 2D grad, got shape {g.shape}")
    beta = np.float32(momentum_coeff)
    new_m = beta * m + g
    nesterov = g + beta * new_m
    orth = newton_schulz(nesterov, int(ns_steps))
    rows, cols = int(orth.shape[0]), int(orth.shape[1])
    rms = np.float32(np.sqrt(max(1.0, rows / max(cols, 1))))
    delta = np.float32(lr) * rms * orth
    return delta.astype(np.float32), new_m.astype(np.float32)


def qk_clip(q: Array, k: Array, max_logit: float) -> tuple[Array, Array]:
    """Scale Q or K so max |q k^T| does not exceed `max_logit` (Kimi K2)."""
    if max_logit <= 0:
        raise ValueError("qk_clip max_logit must be > 0")
    q32 = _as_f32(q)
    k32 = _as_f32(k)
    if q32.shape[-1] != k32.shape[-1]:
        raise ValueError("q and k last dim (head dim) must match")
    scores = np.einsum("...id,...jd->...ij", q32, k32)
    mu = float(np.max(np.abs(scores)))
    tau = float(max_logit)
    if mu <= tau or mu == 0.0:
        return q32.copy(), k32.copy()
    scale = np.float32(np.sqrt(tau / mu))
    return (q32 * scale).astype(np.float32), (k32 * scale).astype(np.float32)


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
    """Decoupled AdamW. `step` is 1-based for bias correction. Returns (p, m, v)."""
    if int(step) < 1:
        raise ValueError("adamw_update step must be 1-based (>= 1)")
    p = _as_f32(param)
    g = _as_f32(grad)
    m32 = _as_f32(m)
    v32 = _as_f32(v)
    b1 = np.float32(beta1)
    b2 = np.float32(beta2)
    m_t = b1 * m32 + (np.float32(1.0) - b1) * g
    v_t = b2 * v32 + (np.float32(1.0) - b2) * (g * g)
    t = int(step)
    m_hat = m_t / (np.float32(1.0) - b1**t)
    v_hat = v_t / (np.float32(1.0) - b2**t)
    adam_step = m_hat / (np.sqrt(v_hat) + np.float32(eps))
    p_t = p * (np.float32(1.0) - np.float32(lr) * np.float32(wd)) - np.float32(lr) * adam_step
    return p_t.astype(np.float32), m_t.astype(np.float32), v_t.astype(np.float32)


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
    """Logit soft-capping: cap * tanh(logits / cap)."""
    if cap <= 0:
        raise ValueError("soft_cap cap must be > 0")
    x = _as_f32(logits)
    c = np.float32(cap)
    return (c * np.tanh(x / c)).astype(np.float32)


def cross_entropy(logits: Array, targets: Array, loss_mask: Array) -> Array:
    """Masked mean CE. `logits` are already soft-capped."""
    logits_f = _as_f32(logits)
    targets_i = np.asarray(targets)
    mask = _as_f32(loss_mask)
    if logits_f.shape[:-1] != tuple(targets_i.shape) or logits_f.shape[:-1] != tuple(mask.shape):
        raise ValueError(
            f"shape mismatch: logits {logits_f.shape}, targets {targets_i.shape}, "
            f"mask {mask.shape}"
        )
    vocab = int(logits_f.shape[-1])
    flat = logits_f.reshape(-1, vocab)
    tgt = targets_i.reshape(-1).astype(np.int64)
    m = mask.reshape(-1)
    shifted = flat - np.max(flat, axis=-1, keepdims=True)
    log_z = np.log(np.maximum(np.exp(shifted).sum(axis=-1), 1e-12))
    nll = -(shifted[np.arange(flat.shape[0]), tgt] - log_z)
    denom = float(m.sum())
    if denom <= 0.0:
        return np.float32(0.0)
    return np.float32(float((nll * m).sum() / denom))


def mtp_loss(mtp_logits: tuple[Array, ...], tokens: Array, loss_mask: Array) -> Array:
    """Mean CE of each MTP head on the token `head_index + 1` steps ahead."""
    tokens_i = np.asarray(tokens)
    mask = _as_f32(loss_mask)
    seq = int(tokens_i.shape[-1])
    losses: list[np.ndarray] = []
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
        return np.float32(0.0)
    stacked = np.stack([np.asarray(x, dtype=np.float32) for x in losses])
    return np.float32(float(np.mean(stacked)))


def z_loss(router_probs: Array) -> Array:
    """Mean squared log-sum-exp of router probabilities, per token then mean."""
    p = _as_f32(router_probs)
    lse = _logsumexp(p, axis=-1)
    return np.float32(float(np.mean(lse**2)))


def total_loss(
    ce: Array,
    mtp: Array,
    z: Array,
    config: TrainConfig,
) -> Array:
    """ce + mtp + z_loss_weight * z."""
    w = np.float32(float(config.z_loss_weight))
    return (_as_f32(ce) + _as_f32(mtp) + w * _as_f32(z)).astype(np.float32)


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


def _tree_sum_sq(obj: Any) -> float:
    if isinstance(obj, dict):
        return sum(_tree_sum_sq(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tree_sum_sq(v) for v in obj)
    g = _as_f32(obj)
    return float(np.sum(g * g))


def _scale_tree(obj: Any, scale: float) -> Any:
    if isinstance(obj, dict):
        return {k: _scale_tree(v, scale) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scale_tree(v, scale) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_scale_tree(v, scale) for v in obj)
    return _as_f32(obj) * np.float32(scale)


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
                out["W_q"] = np.asarray(qc, dtype=np.float32).reshape(q.shape)
                out["W_k"] = np.asarray(kc, dtype=np.float32).reshape(k.shape)
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
    load = np.mean(probs.reshape(-1, probs.shape[-1]), axis=0).astype(np.float32)
    delta = np.sign(load - np.mean(load)).astype(np.float32) * np.float32(u)

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {k: walk(v) for k, v in obj.items()}
            if "router_bias" in out:
                bias = _as_f32(out["router_bias"])
                n = min(bias.shape[-1], delta.shape[0])
                bias = np.array(bias, copy=True)
                bias[..., :n] = bias[..., :n] - delta[:n]
                out["router_bias"] = bias.astype(np.float32)
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

    gnorm = float(np.sqrt(max(_tree_sum_sq(grads), 0.0)))
    clip = float(train_config.grad_clip)
    scale = 1.0
    if clip > 0.0 and gnorm > clip:
        scale = clip / (gnorm + 1e-6)
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
            ce=np.float32(float(ce)),
            mtp=np.float32(float(mtp)),
            z=np.float32(float(z)),
            total=np.float32(float(total)),
        ),
        lr=sched,
        grad_norm=np.float32(gnorm),
        step=t,
    )
