"""Model package (spec 3): config, layers, forward."""

from __future__ import annotations

from model.config import (
    DTYPE,
    LINEAR_TO_MLA,
    ROPE_BASE,
    AttentionKind,
    AttentionKindError,
    ConfigError,
    FfnKind,
    ModelConfig,
    attention_kind,
    cpu_config,
    ffn_kind,
    flagship_config,
    tiny_config,
    validate_config,
)
from model.forward import ForwardOutput, forward, init_params, param_count
from model.layers import (
    latent_adapter,
    linear_attention,
    mla_attention,
    moe,
    rms_norm,
    rope,
    rope_2d,
)

__all__ = [
    "LINEAR_TO_MLA",
    "ROPE_BASE",
    "DTYPE",
    "AttentionKind",
    "AttentionKindError",
    "ConfigError",
    "FfnKind",
    "ModelConfig",
    "ForwardOutput",
    "attention_kind",
    "ffn_kind",
    "cpu_config",
    "flagship_config",
    "tiny_config",
    "validate_config",
    "forward",
    "init_params",
    "param_count",
    "latent_adapter",
    "linear_attention",
    "mla_attention",
    "moe",
    "rms_norm",
    "rope",
    "rope_2d",
]
