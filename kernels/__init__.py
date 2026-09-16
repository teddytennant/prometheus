"""Kernels package (spec 3/4): quant, attention, expert-parallel."""

from __future__ import annotations

from kernels.attn import flash_attention_probe, linear_attention, mla_attention
from kernels.ep import combine, dispatch, ep_dispatch, ep_moe_match
from kernels.quant import (
    fake_quant_fp8,
    fake_quant_nvfp4,
    fp8_cast,
    fp8_linear,
    nvfp4_roundtrip,
    precision_probe,
    two_precision_train,
)

__all__ = [
    "linear_attention",
    "mla_attention",
    "flash_attention_probe",
    "precision_probe",
    "fp8_cast",
    "fp8_linear",
    "nvfp4_roundtrip",
    "fake_quant_fp8",
    "fake_quant_nvfp4",
    "dispatch",
    "ep_dispatch",
    "ep_moe_match",
    "combine",
    "two_precision_train",
]
