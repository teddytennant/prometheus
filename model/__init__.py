"""FP32 reference model (spec 3, 4.1, 15.5 A1).

Architecture is the flagship shape in 3.1: hybrid 3:1 linear vs MLA, DeepSeek-V3
MLA, Gated DeltaNet / KDA linear attention, sigmoid-gated MoE, 8-layer core
iterated r times, MTP (2 extra heads), pre-norm RMSNorm, QK-norm on MLA, logit
soft-capping, router z-loss, and the 2-layer MLP + norm latent adapter.

This package is the CPU/FP32 reference. Kernels (A3) and Stage-B latent
training (I5: halt, thought-decode loss, Jacobi, noisy latents) come later.
The adapter module is part of the V1 shape and lives here.

Nothing here runs. Types and flagship constants are real; forward and the
layer helpers raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

Array = Any  # numpy.ndarray or jax.Array; FP32 reference dtype


class AttentionKind(StrEnum):
    LINEAR = "linear"
    MLA = "mla"


class FfnKind(StrEnum):
    DENSE = "dense"
    MOE = "moe"


class AttentionKindError(ValueError):
    """Layer index is outside the hybrid 3:1 pattern."""


class ConfigError(ValueError):
    """Config violates a 3.1 invariant (counts, ratios, required fields)."""


@dataclass(frozen=True)
class ModelConfig:
    """Discrete architecture. Flagship numbers are 3.1; tiny() is the V1 stand-in."""

    d_model: int
    n_layers: int
    n_dense: int
    n_moe: int
    n_linear_attn: int
    n_mla: int
    n_routed_experts: int
    n_shared_experts: int
    top_k: int
    expert_hidden: int
    mtp_heads: int
    core_block_layers: int
    recurrence_train_mean: float
    recurrence_max: int
    vocab_size: int
    max_context: int
    max_racks: int = 4
    prelude_layers: int | None = None
    coda_layers: int | None = None
    adapter_hidden: int | None = None


# spec 3.1 flagship. Vocab here is 256k; the F1 tokenizer schema is 128k BPE.
# The model takes vocab_size from config, not from the tokenizer artifact.
FLAGSHIP_D_MODEL = 12288
FLAGSHIP_N_LAYERS = 96
FLAGSHIP_N_DENSE = 3
FLAGSHIP_N_MOE = 93
FLAGSHIP_N_LINEAR_ATTN = 72
FLAGSHIP_N_MLA = 24
FLAGSHIP_N_ROUTED_EXPERTS = 512
FLAGSHIP_N_SHARED_EXPERTS = 2
FLAGSHIP_TOP_K = 20
FLAGSHIP_EXPERT_HIDDEN = 4096
FLAGSHIP_MTP_HEADS = 2
FLAGSHIP_CORE_BLOCK_LAYERS = 8
FLAGSHIP_RECURRENCE_TRAIN_MEAN = 3.0
FLAGSHIP_RECURRENCE_MAX = 16
FLAGSHIP_VOCAB_SIZE = 256_000
FLAGSHIP_MAX_CONTEXT = 16_384
FLAGSHIP_MAX_RACKS = 4
FLAGSHIP_TOTAL_PARAMS = 7_400_000_000_000
FLAGSHIP_UNIQUE_ACTIVE_PARAMS = 400_000_000_000
FLAGSHIP_COMPUTE_ACTIVE_PARAMS = 450_000_000_000

# Hybrid pattern is 3 linear : 1 MLA, repeating. Index 0 is linear.
LINEAR_TO_MLA = 3


def flagship_config() -> ModelConfig:
    return ModelConfig(
        d_model=FLAGSHIP_D_MODEL,
        n_layers=FLAGSHIP_N_LAYERS,
        n_dense=FLAGSHIP_N_DENSE,
        n_moe=FLAGSHIP_N_MOE,
        n_linear_attn=FLAGSHIP_N_LINEAR_ATTN,
        n_mla=FLAGSHIP_N_MLA,
        n_routed_experts=FLAGSHIP_N_ROUTED_EXPERTS,
        n_shared_experts=FLAGSHIP_N_SHARED_EXPERTS,
        top_k=FLAGSHIP_TOP_K,
        expert_hidden=FLAGSHIP_EXPERT_HIDDEN,
        mtp_heads=FLAGSHIP_MTP_HEADS,
        core_block_layers=FLAGSHIP_CORE_BLOCK_LAYERS,
        recurrence_train_mean=FLAGSHIP_RECURRENCE_TRAIN_MEAN,
        recurrence_max=FLAGSHIP_RECURRENCE_MAX,
        vocab_size=FLAGSHIP_VOCAB_SIZE,
        max_context=FLAGSHIP_MAX_CONTEXT,
        max_racks=FLAGSHIP_MAX_RACKS,
    )


def tiny_config() -> ModelConfig:
    """V1 CPU stand-in (~10M). Same discrete choices as flagship, smaller sizes.

    8 unique layers, 3:1 linear/MLA, 1 dense + 7 MoE, 8 routed + 2 shared,
    top-2, MTP 2, core block 2 iterated r times, vocab 256, context 128.
    """
    return ModelConfig(
        d_model=128,
        n_layers=8,
        n_dense=1,
        n_moe=7,
        n_linear_attn=6,
        n_mla=2,
        n_routed_experts=8,
        n_shared_experts=2,
        top_k=2,
        expert_hidden=256,
        mtp_heads=2,
        core_block_layers=2,
        recurrence_train_mean=FLAGSHIP_RECURRENCE_TRAIN_MEAN,
        recurrence_max=FLAGSHIP_RECURRENCE_MAX,
        vocab_size=256,
        max_context=128,
        max_racks=FLAGSHIP_MAX_RACKS,
        prelude_layers=3,
        coda_layers=3,
        adapter_hidden=128,
    )


@dataclass(frozen=True)
class ForwardOutput:
    """Next-token logits plus MTP heads and the router aux used in the loss."""

    logits: Array
    mtp_logits: tuple[Array, ...]
    z_loss: Array
    router_probs: Array
    expert_ids: Array
    hidden: Array
    r_used: int


def validate_config(config: ModelConfig) -> None:
    """Raise ConfigError if counts or the 3:1 hybrid ratio do not hold."""
    raise NotImplementedError("A1 validate_config")


def attention_kind(layer_index: int, config: ModelConfig) -> AttentionKind:
    """3:1 linear vs MLA, starting at linear. Raises AttentionKindError if OOB."""
    raise NotImplementedError("A1 attention_kind")


def ffn_kind(layer_index: int, config: ModelConfig) -> FfnKind:
    """First n_dense layers are dense; the rest are MoE."""
    raise NotImplementedError("A1 ffn_kind")


def rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    raise NotImplementedError("A1 rms_norm")


def rope(q: Array, k: Array, positions: Array, *, partial: bool = True) -> tuple[Array, Array]:
    """RoPE on MLA. `partial` is the 3.1 partial-dim variant. 2D RoPE is separate."""
    raise NotImplementedError("A1 rope")


def rope_2d(q: Array, k: Array, row: Array, col: Array) -> tuple[Array, Array]:
    """2D RoPE on ARC grid spans."""
    raise NotImplementedError("A1 rope_2d")


def linear_attention(
    q: Array, k: Array, v: Array, state: Array | None = None
) -> tuple[Array, Array]:
    """Gated DeltaNet / KDA family. Returns (output, next_state)."""
    raise NotImplementedError("A1 linear_attention")


def mla_attention(q: Array, compressed_kv: Array, rope_k: Array, *, qk_norm: bool = True) -> Array:
    """DeepSeek-V3-style MLA with QK-norm on by default."""
    raise NotImplementedError("A1 mla_attention")


def moe(
    x: Array,
    *,
    router_weight: Array,
    routed_weights: Array,
    shared_weights: Array,
    top_k: int,
    max_racks: int = FLAGSHIP_MAX_RACKS,
) -> tuple[Array, Array, Array]:
    """Sigmoid gates, top-k routed + shared experts, node-limited routing.

    Returns (output, router_probs, expert_ids). Aux-loss-free bias balancing is
    a training concern (A2); the reference forward still returns router_probs
    so z-loss can be computed.
    """
    raise NotImplementedError("A1 moe")


def latent_adapter(x: Array, *, hidden: int) -> Array:
    """2-layer MLP + norm. Maps a continuous thought vector into residual stream."""
    raise NotImplementedError("A1 latent_adapter")


def init_params(config: ModelConfig, rng: Any) -> dict[str, Any]:
    """FP32 parameter tree. Embedding, per-layer attn/ffn, MTP heads, adapter."""
    raise NotImplementedError("A1 init_params")


def param_count(params: dict[str, Any]) -> int:
    raise NotImplementedError("A1 param_count")


def forward(
    tokens: Array,
    params: dict[str, Any],
    config: ModelConfig,
    *,
    r: int | None = None,
    thoughts: Array | None = None,
) -> ForwardOutput:
    """FP32 forward.

    `r` is the core-block iteration count. None means sample from the training
    heavy-tailed distribution with mean `config.recurrence_train_mean` and cap
    `config.recurrence_max`. KV is shared across iterations (Huginn-style).

    `thoughts` is an optional (batch, n_thoughts, d_model) tensor inserted via
    the latent adapter before the discrete tokens. Empty/None is discrete-only.
    """
    raise NotImplementedError("A1 forward")
