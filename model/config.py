"""Flagship / tiny configs and hybrid-ratio checks (spec 3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

LINEAR_TO_MLA = 3
ROPE_BASE = 10_000.0
DTYPE = "float32"

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


def _attn_geometry(d_model: int) -> tuple[int, int, int, int]:
    """(n_heads, d_head, d_nope, d_rope) derived from d_model."""
    d_head = 1
    for cand in (64, 32, 16, 8, 4, 2, 1):
        if d_model % cand == 0:
            d_head = cand
            break
    n_heads = d_model // d_head
    d_nope = d_head
    d_rope = (d_head // 2) // 2 * 2
    if d_rope < 2:
        d_rope = 2 if d_head >= 2 else d_head
    return n_heads, d_head, d_nope, d_rope


def _recurrent_split(config: ModelConfig) -> tuple[int, int, int]:
    """Prelude / core / coda unique-layer counts. Sum equals n_layers."""
    core = int(config.core_block_layers)
    if core < 1 or core > config.n_layers:
        raise ConfigError("core_block_layers out of range")
    if config.prelude_layers is not None and config.coda_layers is not None:
        prelude = int(config.prelude_layers)
        coda = int(config.coda_layers)
        if prelude + core + coda != config.n_layers:
            raise ConfigError("prelude + core + coda must equal n_layers")
        if prelude < 0 or coda < 0:
            raise ConfigError("prelude_layers and coda_layers must be >= 0")
        return prelude, core, coda
    rest = config.n_layers - core
    prelude = rest // 2
    coda = rest - prelude
    return prelude, core, coda


def validate_config(config: ModelConfig) -> None:
    """Raise ConfigError if counts or the 3:1 hybrid ratio do not hold."""
    if config.n_layers < 1:
        raise ConfigError("n_layers must be >= 1")
    if config.n_dense + config.n_moe != config.n_layers:
        raise ConfigError("n_dense + n_moe must equal n_layers")
    if config.n_linear_attn + config.n_mla != config.n_layers:
        raise ConfigError("n_linear_attn + n_mla must equal n_layers")
    period = LINEAR_TO_MLA + 1
    if config.n_layers % period != 0:
        raise ConfigError("n_layers must be a multiple of the 3:1 hybrid period")
    if config.n_mla * LINEAR_TO_MLA != config.n_linear_attn:
        raise ConfigError("hybrid attention must be 3 linear : 1 MLA")
    if config.n_dense < 0 or config.n_moe < 0:
        raise ConfigError("n_dense and n_moe must be >= 0")
    for name in (
        "d_model",
        "vocab_size",
        "max_context",
        "n_routed_experts",
        "top_k",
        "expert_hidden",
        "core_block_layers",
        "max_racks",
    ):
        if getattr(config, name) < 1:
            raise ConfigError(f"{name} must be >= 1")
    if config.n_shared_experts < 0:
        raise ConfigError("n_shared_experts must be >= 0")
    if config.top_k > config.n_routed_experts:
        raise ConfigError("top_k cannot exceed n_routed_experts")
    if config.mtp_heads < 0:
        raise ConfigError("mtp_heads must be >= 0")
    if config.recurrence_max < 1:
        raise ConfigError("recurrence_max must be >= 1")
    if config.recurrence_train_mean <= 0:
        raise ConfigError("recurrence_train_mean must be > 0")
    if config.adapter_hidden is not None and config.adapter_hidden < 1:
        raise ConfigError("adapter_hidden must be >= 1")
    _recurrent_split(config)


def attention_kind(layer_index: int, config: ModelConfig) -> AttentionKind:
    """3:1 linear vs MLA, starting at linear. Raises AttentionKindError if OOB."""
    if layer_index < 0 or layer_index >= config.n_layers:
        raise AttentionKindError(
            f"layer_index {layer_index} outside [0, {config.n_layers})"
        )
    period = LINEAR_TO_MLA + 1
    if layer_index % period == LINEAR_TO_MLA:
        return AttentionKind.MLA
    return AttentionKind.LINEAR


def ffn_kind(layer_index: int, config: ModelConfig) -> FfnKind:
    """First n_dense layers are dense; the rest are MoE."""
    if layer_index < 0 or layer_index >= config.n_layers:
        raise ConfigError(f"layer_index {layer_index} outside [0, {config.n_layers})")
    if layer_index < config.n_dense:
        return FfnKind.DENSE
    return FfnKind.MOE

