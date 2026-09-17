"""JAX to SGLang weight converter and numeric parity (spec 13.1, 15.5 C1).

Reads an A1 JAX parameter tree, writes safetensors-compatible shards at
bf16 / fp8 / nvfp4, and reports the payload that matches F1
``prometheus.weight_export``. Parity is max-abs error of dequantized
tensors against the JAX source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SCHEMA_ID = "prometheus.weight_export"
SCHEMA_VERSION = 1
TARGET = "sglang"

DTYPE_BF16 = "bf16"
DTYPE_FP8 = "fp8"
DTYPE_NVFP4 = "nvfp4"
DTYPES = (DTYPE_BF16, DTYPE_FP8, DTYPE_NVFP4)


class ExportError(ValueError):
    """Raised when a parameter tree or dtype cannot be exported."""


@dataclass(frozen=True)
class Shard:
    """One shard listed in a ``prometheus.weight_export`` payload."""

    name: str
    dtype: str
    content_hash: str
    scale_hash: str | None = None
    path: str | None = None


@dataclass(frozen=True)
class WeightExport:
    """In-memory form of ``contracts/schemas/v1/weight_export.schema.json``."""

    export_id: str
    source_checkpoint_id: str
    tokenizer_id: str
    target: str
    dtype: str
    shard_count: int
    shards: tuple[Shard, ...]
    routing_table_hash: str | None
    created_at: str
    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION


def flatten_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Walk a nested JAX pytree into dotted-name arrays.

    Names are ``block.0.attn.w_qkv`` style, matching A1 module paths.
    """
    raise NotImplementedError("C1 flatten_params")


def quantize(array: Any, dtype: str) -> tuple[Any, Any | None]:
    """Quantize a dense array to ``dtype``.

    Returns ``(quantized, scale)``. ``scale`` is None for bf16 and required
    for fp8 and nvfp4.
    """
    raise NotImplementedError("C1 quantize")


def dequantize(quantized: Any, scale: Any | None, dtype: str) -> Any:
    """Inverse of :func:`quantize`, used by the parity suite."""
    raise NotImplementedError("C1 dequantize")


def export_weights(
    params: Mapping[str, Any],
    *,
    export_id: str,
    source_checkpoint_id: str,
    tokenizer_id: str,
    dtype: str,
    created_at: str,
    out_dir: str,
    routing_table_hash: str | None = None,
) -> WeightExport:
    """Write shards under ``out_dir`` and return the F1 weight_export payload.

    ``dtype`` is one of :data:`DTYPES`. Shards are safetensors-compatible.
    """
    raise NotImplementedError("C1 export_weights")


def parity_max_abs_err(
    params: Mapping[str, Any],
    exported: WeightExport,
    out_dir: str,
) -> float:
    """Max absolute error of dequantized shards vs the JAX source arrays."""
    raise NotImplementedError("C1 parity_max_abs_err")


def to_contract(export: WeightExport) -> dict[str, Any]:
    """Serialize to a dict that validates against ``prometheus.weight_export``."""
    raise NotImplementedError("C1 to_contract")
