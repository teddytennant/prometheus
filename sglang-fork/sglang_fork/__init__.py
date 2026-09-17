"""JAX to SGLang weight converter and numeric parity (spec 13.1, 15.5 C1).

Reads an A1 JAX parameter tree, writes safetensors-compatible shards at
bf16 / fp8 / nvfp4, and reports the payload that matches F1
``prometheus.weight_export``. Parity is max-abs error of dequantized
tensors against the JAX source.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import contracts

SCHEMA_ID = "prometheus.weight_export"
SCHEMA_VERSION = 1
TARGET = "sglang"

DTYPE_BF16 = "bf16"
DTYPE_FP8 = "fp8"
DTYPE_NVFP4 = "nvfp4"
DTYPES = (DTYPE_BF16, DTYPE_FP8, DTYPE_NVFP4)

FP8_E4M3_MAX = np.float32(448.0)
FP8_CODE_ZERO = 128
FP8_CODE_FLOAT_SCALE = np.float32(float(FP8_E4M3_MAX) / 127.0)

NVFP4_BLOCK = 16
NVFP4_E2M1_MAX = np.float32(6.0)
E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

SIDECAR_NAME = "shards.json"
SIDECAR_FORMAT = "prometheus.c1.npy-shards.v1"

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


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


def _as_f32(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _is_leaf(node: Any) -> bool:
    if isinstance(node, Mapping | list | tuple | str | bytes | bytearray):
        return False
    return hasattr(node, "shape") and hasattr(node, "dtype")


def _raw_le_bytes(arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr)
    endian = a.dtype.byteorder
    if endian == ">" or (endian == "=" and sys.byteorder != "little"):
        a = a.byteswap().view(a.dtype.newbyteorder("<"))
    return a.tobytes()


def _sha256_le(arr: np.ndarray) -> str:
    return hashlib.sha256(_raw_le_bytes(np.asarray(arr))).hexdigest()


def _model_config_hash(tokenizer_id: str, shard_names: list[str] | tuple[str, ...]) -> str:
    lines = [tokenizer_id, *sorted(shard_names)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def flatten_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Walk a nested JAX pytree into dotted-name arrays.

    Names are ``block.0.attn.w_qkv`` style, matching A1 module paths.
    """
    if not isinstance(params, Mapping):
        raise ExportError("params must be a mapping")
    out: dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                walk(name, child)
            return
        if isinstance(node, list | tuple):
            for index, child in enumerate(node):
                name = f"{prefix}.{index}" if prefix else str(index)
                walk(name, child)
            return
        if _is_leaf(node):
            if not prefix:
                raise ExportError("unnamed array leaf")
            out[prefix] = np.asarray(node)
            return
        loc = prefix or "<root>"
        raise ExportError(f"non-array leaf at {loc}: {type(node)!r}")

    walk("", params)
    return {key: out[key] for key in sorted(out)}


def _fp8_code_as_float(code: np.ndarray) -> np.ndarray:
    return (code.astype(np.float32) - np.float32(FP8_CODE_ZERO)) * FP8_CODE_FLOAT_SCALE


def _quantize_fp8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    amax = np.max(np.abs(x)) if x.size else np.float32(0.0)
    if not np.isfinite(amax):
        raise ExportError("fp8 quantize requires finite values")
    scale_val = np.float32(1.0) if float(amax) == 0.0 else np.float32(amax / FP8_E4M3_MAX)
    scale = np.asarray(scale_val, dtype=np.float32)
    y = x / scale
    q = np.rint(y / FP8_CODE_FLOAT_SCALE) + FP8_CODE_ZERO
    return np.clip(q, 0, 255).astype(np.uint8), scale


def _dequantize_fp8(quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (_fp8_code_as_float(np.asarray(quantized)) * _as_f32(scale)).astype(np.float32)


def _to_e2m1_nibble(scaled: np.ndarray) -> np.ndarray:
    """Round to nearest E2M1; ties (equal distance) go to smaller magnitude."""
    sign = (np.asarray(scaled) < 0).astype(np.uint8)
    mag = np.abs(scaled).astype(np.float32)
    dist = np.abs(mag[..., None] - E2M1_POS)
    idx = np.argmin(dist, axis=-1).astype(np.uint8)
    return (idx | (sign << 3)).astype(np.uint8)


def _from_e2m1_nibble(code: np.ndarray) -> np.ndarray:
    bits = np.asarray(code, dtype=np.uint8)
    mag = E2M1_POS[bits & np.uint8(7)]
    sign = np.where((bits >> 3) & 1, np.float32(-1.0), np.float32(1.0))
    return (sign * mag).astype(np.float32)


def _pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    even = nibbles[..., 0::2] & np.uint8(0x0F)
    odd = nibbles[..., 1::2] & np.uint8(0x0F)
    return (even | (odd << 4)).astype(np.uint8)


def _unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    even = packed & np.uint8(0x0F)
    odd = (packed >> 4) & np.uint8(0x0F)
    stacked = np.stack((even, odd), axis=-1)
    return stacked.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def _quantize_nvfp4(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim < 1:
        x = x.reshape(1)
    prefix = x.shape[:-1]
    length = int(x.shape[-1])
    n_blocks = (length + NVFP4_BLOCK - 1) // NVFP4_BLOCK
    pad_len = n_blocks * NVFP4_BLOCK
    if pad_len != length:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad_len - length)]
        padded = np.pad(x, pad_width)
    else:
        padded = x
    blocked = padded.reshape(*prefix, n_blocks, NVFP4_BLOCK)
    amax = np.max(np.abs(blocked), axis=-1)
    if not np.all(np.isfinite(amax)):
        raise ExportError("nvfp4 quantize requires finite values")
    scale = np.empty_like(amax, dtype=np.float32)
    zero = amax == 0
    scale[zero] = np.float32(1.0)
    scale[~zero] = (amax[~zero] / NVFP4_E2M1_MAX).astype(np.float32)
    scaled = blocked / scale[..., None]
    nibbles = _to_e2m1_nibble(scaled).reshape(*prefix, pad_len)
    return _pack_nibbles(nibbles), scale.astype(np.float32)


def _dequantize_nvfp4(quantized: np.ndarray, scale: np.ndarray) -> np.ndarray:
    packed = np.asarray(quantized, dtype=np.uint8)
    scale_f = _as_f32(scale)
    nibbles = _unpack_nibbles(packed)
    decoded = _from_e2m1_nibble(nibbles)
    prefix = decoded.shape[:-1]
    pad_len = int(decoded.shape[-1])
    if pad_len % NVFP4_BLOCK != 0:
        raise ExportError("nvfp4 packed last axis is not a multiple of 8 bytes")
    n_blocks = pad_len // NVFP4_BLOCK
    blocked = decoded.reshape(*prefix, n_blocks, NVFP4_BLOCK)
    if scale_f.shape != (*prefix, n_blocks):
        raise ExportError(f"nvfp4 scale shape {scale_f.shape} != {(*prefix, n_blocks)}")
    restored = blocked * scale_f[..., None]
    return restored.reshape(*prefix, pad_len).astype(np.float32)


def quantize(array: Any, dtype: str) -> tuple[Any, Any | None]:
    """Quantize a dense array to ``dtype``.

    Returns ``(quantized, scale)``. ``scale`` is None for bf16 and required
    for fp8 and nvfp4.
    """
    if dtype not in DTYPES:
        raise ExportError(f"unsupported dtype {dtype!r}")
    x = _as_f32(array)
    if x.size and not np.all(np.isfinite(x)):
        raise ExportError("quantize requires finite values")
    if dtype == DTYPE_BF16:
        return x.copy(), None
    if dtype == DTYPE_FP8:
        return _quantize_fp8(x)
    if dtype == DTYPE_NVFP4:
        return _quantize_nvfp4(x)
    raise ExportError(f"unsupported dtype {dtype!r}")


def dequantize(quantized: Any, scale: Any | None, dtype: str) -> Any:
    """Inverse of :func:`quantize`, used by the parity suite."""
    if dtype not in DTYPES:
        raise ExportError(f"unsupported dtype {dtype!r}")
    if dtype == DTYPE_BF16:
        if scale is not None:
            raise ExportError("bf16 scale must be None")
        return _as_f32(quantized).copy()
    if scale is None:
        raise ExportError(f"{dtype} scale is required")
    if dtype == DTYPE_FP8:
        return _dequantize_fp8(np.asarray(quantized), np.asarray(scale))
    return _dequantize_nvfp4(np.asarray(quantized), np.asarray(scale))


def _crop_to_source(dequantized: np.ndarray, source: np.ndarray) -> np.ndarray:
    src = np.asarray(source)
    dq = np.asarray(dequantized, dtype=np.float32)
    if dq.shape == src.shape:
        return dq
    if dq.ndim == src.ndim and dq.shape[:-1] == src.shape[:-1] and dq.shape[-1] >= src.shape[-1]:
        return dq[..., : src.shape[-1]]
    raise ExportError(f"dequantized shape {dq.shape} does not match source {src.shape}")


def _validate_weight_export(payload: Mapping[str, Any]) -> None:
    named = getattr(contracts, "validate_named", None)
    if callable(named):
        named("prometheus.weight_export", payload)
        return
    contracts.validate(payload)


def to_contract(export: WeightExport) -> dict[str, Any]:
    """Serialize to a dict that validates against ``prometheus.weight_export``."""
    shards: list[dict[str, Any]] = []
    for shard in export.shards:
        item: dict[str, Any] = {
            "name": shard.name,
            "dtype": shard.dtype,
            "content_hash": shard.content_hash,
        }
        if shard.scale_hash is not None:
            item["scale_hash"] = shard.scale_hash
        if shard.path is not None:
            item["path"] = shard.path
        shards.append(item)
    payload: dict[str, Any] = {
        "schema_id": export.schema_id,
        "schema_version": export.schema_version,
        "export_id": export.export_id,
        "source_checkpoint_id": export.source_checkpoint_id,
        "target": export.target,
        "dtype": export.dtype,
        "tokenizer_id": export.tokenizer_id,
        "model_config_hash": _model_config_hash(
            export.tokenizer_id, [s.name for s in export.shards]
        ),
        "shards": shards,
        "created_at": export.created_at,
    }
    if export.routing_table_hash is not None:
        payload["routing_table_hash"] = export.routing_table_hash
    try:
        _validate_weight_export(payload)
    except contracts.ValidationError as exc:
        raise ExportError(str(exc)) from exc
    return payload


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
    if dtype not in DTYPES:
        raise ExportError(f"unsupported dtype {dtype!r}")
    if not export_id or not source_checkpoint_id or not tokenizer_id:
        raise ExportError("export_id, source_checkpoint_id, tokenizer_id must be non-empty")
    if routing_table_hash is not None and _SHA256_HEX.fullmatch(routing_table_hash) is None:
        raise ExportError("routing_table_hash must be lowercase hex SHA-256")
    flat = flatten_params(params)
    if not flat:
        raise ExportError("no array leaves to export")
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    shards: list[Shard] = []
    sidecar_shards: list[dict[str, str]] = []
    for name, array in flat.items():
        quantized, scale = quantize(array, dtype)
        rel = f"{name}.npy"
        np.save(dest / rel, quantized)
        content_hash = _sha256_le(np.asarray(quantized))
        scale_hash = None
        scale_rel = None
        if scale is not None:
            scale_rel = f"{name}.scale.npy"
            np.save(dest / scale_rel, np.asarray(scale))
            scale_hash = _sha256_le(np.asarray(scale))
        shards.append(
            Shard(
                name=name,
                dtype=dtype,
                content_hash=content_hash,
                scale_hash=scale_hash,
                path=rel,
            )
        )
        entry = {"name": name, "path": rel}
        if scale_rel is not None:
            entry["scale_path"] = scale_rel
        sidecar_shards.append(entry)
    sidecar = {
        "format": SIDECAR_FORMAT,
        "export_id": export_id,
        "dtype": dtype,
        "shards": sidecar_shards,
    }
    (dest / SIDECAR_NAME).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return WeightExport(
        export_id=export_id,
        source_checkpoint_id=source_checkpoint_id,
        tokenizer_id=tokenizer_id,
        target=TARGET,
        dtype=dtype,
        shard_count=len(shards),
        shards=tuple(shards),
        routing_table_hash=routing_table_hash,
        created_at=created_at,
        schema_id=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
    )


def parity_max_abs_err(
    params: Mapping[str, Any],
    exported: WeightExport,
    out_dir: str,
) -> float:
    """Max absolute error of dequantized shards vs the JAX source arrays."""
    flat = flatten_params(params)
    names = [shard.name for shard in exported.shards]
    if set(names) != set(flat):
        raise ExportError("exported shard names do not match flattened params")
    dest = Path(out_dir)
    err = 0.0
    for shard in exported.shards:
        source = flat[shard.name]
        if shard.path is None:
            raise ExportError(f"shard {shard.name!r} has no path")
        path = Path(shard.path)
        q_path = path if path.is_absolute() else dest / path
        quantized = np.load(q_path, allow_pickle=False)
        scale = None
        if shard.scale_hash is not None:
            if q_path.name.endswith(".npy"):
                scale_path = q_path.with_name(q_path.name[:-4] + ".scale.npy")
            else:
                scale_path = q_path.with_name(q_path.name + ".scale.npy")
            scale = np.load(scale_path, allow_pickle=False)
        restored = dequantize(quantized, scale, shard.dtype)
        cropped = _crop_to_source(restored, source)
        diff = np.max(np.abs(cropped - _as_f32(source)))
        err = max(err, float(diff))
    return float(err)
