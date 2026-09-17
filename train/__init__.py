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

Nothing here runs. Types and flagship constants are real; math raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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
    raise NotImplementedError("A2 validate_train_config")


def classify_param(name: str, value: Array) -> ParamKind:
    """2D hidden weights are Muon; embeddings, norms, biases, sigma are AdamW."""
    raise NotImplementedError("A2 classify_param")


def precision_for(name: str, config: TrainConfig) -> PrecisionKind:
    """FP8 on hidden linears; BF16 on router/norm/embed/softmax/tail/adapter."""
    raise NotImplementedError("A2 precision_for")


def newton_schulz(matrix: Array, steps: int) -> Array:
    """Newton-Schulz orthogonalization of a 2D gradient. Odd `steps` >= 1."""
    raise NotImplementedError("A2 newton_schulz")


def muon_update(
    grad: Array,
    momentum: Array,
    *,
    lr: float,
    momentum_coeff: float,
    ns_steps: int,
) -> tuple[Array, Array]:
    """Nesterov momentum then Newton-Schulz. Returns (delta, new_momentum)."""
    raise NotImplementedError("A2 muon_update")


def qk_clip(q: Array, k: Array, max_logit: float) -> tuple[Array, Array]:
    """Scale Q or K so max |q k^T| does not exceed `max_logit` (Kimi K2)."""
    raise NotImplementedError("A2 qk_clip")


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
    raise NotImplementedError("A2 adamw_update")


def wsd_lr(step: int, config: TrainConfig) -> float:
    """Warmup (linear), stable (peak), decay (linear to 0). `step` is 1-based."""
    raise NotImplementedError("A2 wsd_lr")


def soft_cap(logits: Array, cap: float) -> Array:
    """Logit soft-capping: cap * tanh(logits / cap)."""
    raise NotImplementedError("A2 soft_cap")


def cross_entropy(logits: Array, targets: Array, loss_mask: Array) -> Array:
    """Masked mean CE. `logits` are already soft-capped."""
    raise NotImplementedError("A2 cross_entropy")


def mtp_loss(mtp_logits: tuple[Array, ...], tokens: Array, loss_mask: Array) -> Array:
    """Mean CE of each MTP head on the token `head_index + 1` steps ahead."""
    raise NotImplementedError("A2 mtp_loss")


def z_loss(router_probs: Array) -> Array:
    """Mean squared log-sum-exp of router probabilities, per token then mean."""
    raise NotImplementedError("A2 z_loss")


def total_loss(
    ce: Array,
    mtp: Array,
    z: Array,
    config: TrainConfig,
) -> Array:
    """ce + mtp + z_loss_weight * z."""
    raise NotImplementedError("A2 total_loss")


def init_opt_state(params: dict[str, Any], config: TrainConfig) -> dict[str, Any]:
    """FP32 optimizer state matching `classify_param` for every leaf."""
    raise NotImplementedError("A2 init_opt_state")


def apply_precision(
    params: dict[str, Any],
    config: TrainConfig,
) -> dict[str, Any]:
    """Cast a master-weight tree for the forward. Masters themselves stay FP32."""
    raise NotImplementedError("A2 apply_precision")


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
    raise NotImplementedError("A2 train_step")
