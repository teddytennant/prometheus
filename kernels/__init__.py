"""Kernels package (spec 3/4): quant, attention, expert-parallel."""

from __future__ import annotations

from kernels.attn import apply_rope, flash_attention_probe, linear_attention, mla_attention
from kernels.ep import all_to_all, combine, dispatch, ep_dispatch, ep_moe_match, partition_by_rank
from kernels.quant import (
    fake_quant_fp8,
    fake_quant_nvfp4,
    fp8_block_scales,
    fp8_cast,
    fp8_linear,
    nvfp4_pack,
    nvfp4_roundtrip,
    nvfp4_unpack,
    precision_probe,
    two_precision_train,
)

__all__ = [
    "linear_attention",
    "mla_attention",
    "flash_attention_probe",
    "apply_rope",
    "precision_probe",
    "fp8_cast",
    "fp8_linear",
    "fp8_block_scales",
    "nvfp4_roundtrip",
    "nvfp4_pack",
    "nvfp4_unpack",
    "fake_quant_fp8",
    "fake_quant_nvfp4",
    "dispatch",
    "ep_dispatch",
    "ep_moe_match",
    "combine",
    "all_to_all",
    "partition_by_rank",
    "two_precision_train",
]
