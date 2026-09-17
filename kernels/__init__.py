"""Custom kernels: linear attention, FP8 linears, EP dispatch (spec 5.1, 15.5 A3).

These are the XLA custom ops the A1 reference calls through once A3 is
verified. Until then the A1 numpy/JAX path stays the CPU source of truth.

- Linear attention is the chunked delta-rule (Gated DeltaNet / KDA family)
  with a fused forward and backward. Constant-size state across the
  sequence; chunking is an implementation detail, not part of the math.
- FP8 linears use per-block scaling (`fp8_block` from A2). NVFP4 is a
  later dtype swap behind the same primitive (V3).
- EP dispatch / combine is the all-to-all that sends tokens to experts and
  weighted-sums them back. Node-limited routing (max 4 racks) is a
  constraint on the dispatch metadata, not a separate kernel.

Each primitive is a `jax.custom_vjp`. The CPU tests compare against a slow
reference the oracle writes; the GPU path is the V1 / V3 gate.

Nothing here runs. Types are real; every op raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

Array = Any  # numpy.ndarray or jax.Array


class KernelError(ValueError):
    """Shape, dtype, or routing invariant failed before the kernel ran."""


class DType(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"
    NVFP4 = "nvfp4"


# Chunk width for the delta-rule tiling. Independent of sequence length.
DEFAULT_CHUNK = 64
# FP8 scale block along the contracting dim. Matches train.FLAGSHIP_FP8_BLOCK.
DEFAULT_FP8_BLOCK = 128
# Node-limited routing cap from 3.1.
MAX_RACKS = 4


@dataclass(frozen=True)
class LinearAttnConfig:
    """Chunked delta-rule knobs. `chunk` does not change the mathematical result."""

    chunk: int = DEFAULT_CHUNK
    eps: float = 1e-6


@dataclass(frozen=True)
class Fp8Meta:
    """Per-block scales for an FP8 packed tensor.

    `q` is the quantized payload (int8-view of fp8). `scale` has one value
    per block of `block` elements along the contracting axis.
    """

    q: Array
    scale: Array
    block: int = DEFAULT_FP8_BLOCK
    dtype: DType = DType.FP8


@dataclass(frozen=True)
class DispatchMeta:
    """Who goes where for one MoE all-to-all.

    `expert_ids` is (tokens, top_k). `probs` is the same shape. `racks` is
    the rack id of each expert, length n_routed_experts. Dispatch must
    refuse a token whose chosen experts span more than `max_racks` racks.
    """

    expert_ids: Array
    probs: Array
    racks: Array
    n_experts: int
    max_racks: int = MAX_RACKS


def chunked_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    state: Array | None = None,
    *,
    config: LinearAttnConfig | None = None,
) -> tuple[Array, Array]:
    """Gated delta-rule linear attention.

    q, k, v: (batch, seq, heads, dim). beta: (batch, seq, heads) gate in (0, 1].
    state: (batch, heads, dim, dim) recurrent state, zeros if None.
    Returns (output, next_state) with output shaped like v.

    Must be jax.custom_vjp: the backward is the fused reverse-state kernel,
    not JAX's default loop autodiff.
    """
    raise NotImplementedError("A3 chunked_delta_rule")


def chunked_delta_rule_fwd(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    state: Array | None,
    config: LinearAttnConfig,
) -> tuple[tuple[Array, Array], Any]:
    """Custom VJP forward. Residual is whatever the backward needs."""
    raise NotImplementedError("A3 chunked_delta_rule_fwd")


def chunked_delta_rule_bwd(residual: Any, grads: tuple[Array, Array]) -> tuple[Array, ...]:
    """Custom VJP backward. Returns grads for (q, k, v, beta, state)."""
    raise NotImplementedError("A3 chunked_delta_rule_bwd")


def fp8_quantize(x: Array, *, block: int = DEFAULT_FP8_BLOCK) -> Fp8Meta:
    """Per-block abs-max scale, quantize to FP8. `x` is FP32 or BF16."""
    raise NotImplementedError("A3 fp8_quantize")


def fp8_dequantize(meta: Fp8Meta) -> Array:
    """Unpack FP8 + scales to FP32. Inverse of `fp8_quantize` up to rounding."""
    raise NotImplementedError("A3 fp8_dequantize")


def fp8_linear(x: Array, weight: Array, *, block: int = DEFAULT_FP8_BLOCK) -> Array:
    """y = x @ w^T with both sides quantized per-block to FP8.

    x: (..., in), weight: (out, in). Accumulates in FP32.
    Must be jax.custom_vjp so the backward uses the same scales as the forward.
    """
    raise NotImplementedError("A3 fp8_linear")


def fp8_linear_fwd(
    x: Array, weight: Array, block: int
) -> tuple[Array, Any]:
    raise NotImplementedError("A3 fp8_linear_fwd")


def fp8_linear_bwd(residual: Any, g: Array) -> tuple[Array, Array]:
    """Returns (grad_x, grad_weight). Scales are not differentiated."""
    raise NotImplementedError("A3 fp8_linear_bwd")


def ep_dispatch(tokens: Array, meta: DispatchMeta) -> tuple[Array, Any]:
    """All-to-all tokens to experts.

    tokens: (n_tokens, d_model). Returns (dispatched, residual) where
    dispatched is (n_experts, max_per_expert, d_model) padded, and residual
    is the inverse permutation `ep_combine` needs.

    Raises KernelError if a token's experts span more than meta.max_racks.
    """
    raise NotImplementedError("A3 ep_dispatch")


def ep_combine(expert_out: Array, meta: DispatchMeta, residual: Any) -> Array:
    """Weighted sum of expert outputs back to token order.

    expert_out: (n_experts, max_per_expert, d_model). Weights are meta.probs.
    Returns (n_tokens, d_model).
    """
    raise NotImplementedError("A3 ep_combine")
