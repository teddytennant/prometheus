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

Each primitive is a `jax.custom_vjp`. `jax.jit` of each public primitive
must match the eager result. `Fp8Meta` must be a jax.tree_util registered
dataclass so `jax.jit(fp8_quantize)` and `jax.jit(fp8_dequantize)` can
return and take it. `q` and `scale` are data fields (arrays). `block` and
`dtype` are meta fields (int, DType). Analog of `DispatchMeta`.

`_DeltaResidual` is the same analog for the explicit delta-rule VJP pair:
`q`, `k`, `v`, `beta`, `state0` are data fields (arrays). `chunk` is meta
(Python int). `jax.jit(chunked_delta_rule_fwd, static_argnames=('config',))`
must match eager at 1e-5 and return a pytree residual.
`jax.jit(chunked_delta_rule_bwd)(residual, grads)` must match eager at 1e-5.
`config` is a host `LinearAttnConfig` (static). Traced arrays must not be
converted with `numpy.asarray`.

The explicit FP8 VJP pair is the same analog: residual is `(x_meta, w_meta)`,
two pytree `Fp8Meta` values. `jax.jit(fp8_linear_fwd, static_argnames=('block',))`
must match eager at 1e-5 and return that pytree residual.
`jax.jit(fp8_linear_bwd)(residual, g)` must match eager at 1e-5. `block` is a
Python int (static). Traced arrays must not be converted with `numpy.asarray`.
Public `fp8_linear` custom_vjp stays.

The explicit EP VJP pair is the same analog. Residual for dispatch is a
pytree `_DispatchResidual` (`token_index` and `k_index` data; `max_per_expert`
meta, Python int). `jax.jit(ep_dispatch_fwd)(tokens, meta)` must match eager
at 1e-5 and return that pytree residual.
`jax.jit(ep_dispatch_bwd)(residual, g)` must match eager at 1e-5.
`jax.jit(ep_combine_fwd)(expert_out, meta, residual)` must match eager at 1e-5
and return a pytree residual.
`jax.jit(ep_combine_bwd)(residual, g)` must match eager at 1e-5.
DispatchMeta is a registered dataclass (static meta fields). Traced arrays
must not be converted with `numpy.asarray`. Public `ep_dispatch` /
`ep_combine` custom_vjp stay. The CPU tests compare against a slow
reference the oracle writes; the GPU path is the V1 / V3 gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from kernels.compile_cache import CacheError as CacheError
from kernels.compile_cache import CompileCache as CompileCache
from kernels.compile_cache import program_hash as program_hash

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
# E4M3FN finite max (exp=15, mantissa=6); exp=15 mantissa=7 is NaN.
FP8_E4M3_MAX = 448.0
_FP8_E4M3_MIN_SUB = 2.0**-9


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

    Must be a jax.tree_util registered dataclass so jax.jit(fp8_quantize)
    and jax.jit(fp8_dequantize) can return and take it. q and scale are data
    fields (arrays). block and dtype are meta fields (int, DType). Analog of
    DispatchMeta.
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

    Must be a jax.tree_util registered dataclass so jax.jit(ep_dispatch)
    and jax.jit(ep_combine) can take it. expert_ids, probs, racks are data
    fields (arrays). n_experts and max_racks are meta fields (ints).
    `_static_max_per` is the pad width baked into the jit treedef so the
    dispatched tensor can keep a data-dependent shape without host-copying
    traced routing arrays.
    """

    expert_ids: Array
    probs: Array
    racks: Array
    n_experts: int
    max_racks: int = MAX_RACKS
    _static_max_per: int | None = None

    def __post_init__(self) -> None:
        if self._static_max_per is not None:
            return
        object.__setattr__(
            self, "_static_max_per", _try_max_per(self.expert_ids, int(self.n_experts))
        )


@dataclass(frozen=True)
class _DeltaResidual:
    """VJP residual for `chunked_delta_rule_fwd` / `chunked_delta_rule_bwd`.

    Must be a jax.tree_util registered dataclass so jax.jit of the explicit
    fwd/bwd pair can return and take it. q, k, v, beta, state0 are data
    fields (arrays). chunk is a meta field (Python int). Analog of Fp8Meta
    and DispatchMeta.
    """

    q: Array
    k: Array
    v: Array
    beta: Array
    state0: Array
    chunk: int


@dataclass(frozen=True)
class _DispatchResidual:
    token_index: Array  # (n_experts, max_per_expert), -1 padded
    k_index: Array
    max_per_expert: int
    # Python-int meta so ep_dispatch_bwd can size grad_tokens under jit.
    n_tokens: int | None = None


jax.tree_util.register_dataclass(
    Fp8Meta,
    data_fields=("q", "scale"),
    meta_fields=("block", "dtype"),
)
jax.tree_util.register_dataclass(
    DispatchMeta,
    data_fields=("expert_ids", "probs", "racks"),
    meta_fields=("n_experts", "max_racks", "_static_max_per"),
)
jax.tree_util.register_dataclass(
    _DispatchResidual,
    data_fields=("token_index", "k_index"),
    meta_fields=("max_per_expert", "n_tokens"),
)
jax.tree_util.register_dataclass(
    _DeltaResidual,
    data_fields=("q", "k", "v", "beta", "state0"),
    meta_fields=("chunk",),
)


def _f32(x: Array) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _is_jax(*xs: Array) -> bool:
    return any(isinstance(x, jax.Array) for x in xs if x is not None)


def _is_tracer(*xs: Array) -> bool:
    return any(isinstance(x, jax.core.Tracer) for x in xs if x is not None)


def _try_max_per(expert_ids: Array, n_experts: int) -> int | None:
    """Concrete pad width, or None if `expert_ids` is a tracer."""
    if isinstance(expert_ids, jax.core.Tracer):
        return None
    try:
        flat = np.asarray(expert_ids).reshape(-1)
    except Exception:
        return None
    if flat.size == 0:
        return 0
    counts = np.bincount(flat.astype(np.int64, copy=False), minlength=int(n_experts))
    return int(counts.max()) if int(n_experts) else 0


def _static_max_per_of(meta: DispatchMeta, expert_ids: Array, n_experts: int) -> int:
    cached = getattr(meta, "_static_max_per", None)
    if cached is not None:
        return int(cached)
    got = _try_max_per(expert_ids, n_experts)
    if got is not None:
        return got
    return int(expert_ids.shape[0]) * int(expert_ids.shape[1])


# ---------------------------------------------------------------------------
# E4M3FN tables (vectorized encode / decode)
# ---------------------------------------------------------------------------


def _e4m3_positive_table() -> tuple[np.ndarray, np.ndarray]:
    vals: list[float] = []
    codes: list[int] = []
    for m in range(8):
        vals.append(m * _FP8_E4M3_MIN_SUB)
        codes.append(m)
    for e in range(1, 15):
        for m in range(8):
            vals.append((2.0 ** (e - 7)) * (1.0 + m / 8.0))
            codes.append((e << 3) | m)
    for m in range(7):
        vals.append((2.0 ** (15 - 7)) * (1.0 + m / 8.0))
        codes.append((15 << 3) | m)
    return np.asarray(vals, dtype=np.float32), np.asarray(codes, dtype=np.uint8)


_E4M3_POS, _E4M3_POS_CODES = _e4m3_positive_table()
_E4M3_POS_J = jnp.asarray(_E4M3_POS)
_E4M3_POS_CODES_J = jnp.asarray(_E4M3_POS_CODES)


def _from_e4m3_bits(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    sign = np.where((bits & np.uint8(0x80)) != 0, np.float32(-1.0), np.float32(1.0))
    exp = ((bits >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    man = (bits & np.uint8(0x07)).astype(np.int32)
    sub = exp == 0
    nan = (exp == 15) & (man == 7)
    val_sub = man.astype(np.float32) * np.float32(_FP8_E4M3_MIN_SUB)
    val_norm = (np.float32(2.0) ** (exp.astype(np.float32) - np.float32(7.0))) * (
        np.float32(1.0) + man.astype(np.float32) / np.float32(8.0)
    )
    val = np.where(sub, val_sub, val_norm)
    val = np.where(nan, np.float32(np.nan), val)
    return sign * val


_E4M3_DECODE = _from_e4m3_bits(np.arange(256, dtype=np.uint8))
_E4M3_DECODE_J = jnp.asarray(_E4M3_DECODE)


def _to_e4m3_bits(x: np.ndarray) -> np.ndarray:
    """Round to nearest finite E4M3FN; ties to the smaller-magnitude code."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x).astype(np.uint8)
    ax = np.abs(x)
    ax = np.nan_to_num(ax, nan=FP8_E4M3_MAX, posinf=FP8_E4M3_MAX, neginf=FP8_E4M3_MAX)
    ax = np.minimum(ax, np.float32(FP8_E4M3_MAX))
    diffs = np.abs(ax[..., None] - _E4M3_POS)
    idx = np.argmin(diffs, axis=-1)
    codes = _E4M3_POS_CODES[idx]
    return np.asarray(codes | (sign << np.uint8(7)), dtype=np.uint8)


def _to_e4m3_bits_jax(x: jax.Array) -> jax.Array:
    sign = jnp.signbit(x).astype(jnp.uint8)
    ax = jnp.abs(x)
    ax = jnp.nan_to_num(ax, nan=FP8_E4M3_MAX, posinf=FP8_E4M3_MAX, neginf=FP8_E4M3_MAX)
    ax = jnp.minimum(ax, jnp.float32(FP8_E4M3_MAX))
    diffs = jnp.abs(ax[..., None] - _E4M3_POS_J)
    idx = jnp.argmin(diffs, axis=-1)
    codes = _E4M3_POS_CODES_J[idx]
    return (codes | (sign << jnp.uint8(7))).astype(jnp.uint8)


# ---------------------------------------------------------------------------
# Gated delta-rule: rank-1 update, scan over chunks
# ---------------------------------------------------------------------------


def _validate_delta(
    q: Array, k: Array, v: Array, beta: Array, state: Array | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_np, k_np, v_np, beta_np = _f32(q), _f32(k), _f32(v), _f32(beta)
    if q_np.ndim != 4 or k_np.shape != q_np.shape or v_np.shape != q_np.shape:
        raise KernelError(f"q, k, v must all have shape (B, S, H, D); got {q_np.shape}")
    batch, seq, heads, dim = q_np.shape
    if beta_np.shape != (batch, seq, heads):
        raise KernelError(f"beta shape {beta_np.shape} != {(batch, seq, heads)}")
    if np.any(beta_np < 0):
        raise KernelError("beta must be in (0, 1]")
    if state is None:
        state_np = np.zeros((batch, heads, dim, dim), dtype=np.float32)
    else:
        state_np = _f32(state)
        if state_np.shape != (batch, heads, dim, dim):
            raise KernelError(f"state shape {state_np.shape} != {(batch, heads, dim, dim)}")
    return q_np, k_np, v_np, beta_np, state_np


def _validate_delta_jax(
    q: jax.Array, k: jax.Array, v: jax.Array, beta: jax.Array, state: Array | None
) -> None:
    """Shape checks on JAX arrays / tracers without host roundtrips."""
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise KernelError(f"q, k, v must all have shape (B, S, H, D); got {q.shape}")
    batch, seq, heads, dim = q.shape
    if beta.shape != (batch, seq, heads):
        raise KernelError(f"beta shape {beta.shape} != {(batch, seq, heads)}")
    if state is not None and state.shape != (batch, heads, dim, dim):
        raise KernelError(f"state shape {state.shape} != {(batch, heads, dim, dim)}")


def _delta_jax_inputs(
    q: Array, k: Array, v: Array, beta: Array, state: Array | None
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    qj = jnp.asarray(q, dtype=jnp.float32)
    kj = jnp.asarray(k, dtype=jnp.float32)
    vj = jnp.asarray(v, dtype=jnp.float32)
    bj = jnp.asarray(beta, dtype=jnp.float32)
    _validate_delta_jax(qj, kj, vj, bj, state)
    if state is None:
        batch, _, heads, dim = qj.shape
        sj = jnp.zeros((batch, heads, dim, dim), dtype=jnp.float32)
    else:
        sj = jnp.asarray(state, dtype=jnp.float32)
    return qj, kj, vj, bj, sj


def _delta_step_np(
    s: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One timestep. Rank-1 form of (I - β k kᵀ) S + β k vᵀ; o = q S."""
    bt = beta[..., None, None]
    # kᵀ S : (B, H, D)
    k_s = np.einsum("bhd,bhde->bhe", k, s)
    s_new = s - bt * np.einsum("bhd,bhe->bhde", k, k_s) + bt * np.einsum("bhd,bhe->bhde", k, v)
    o = np.einsum("bhd,bhde->bhe", q, s_new)
    return s_new, o


def _delta_step_jax(
    s: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, beta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    bt = beta[..., None, None]
    k_s = jnp.einsum("bhd,bhde->bhe", k, s)
    s_new = s - bt * jnp.einsum("bhd,bhe->bhde", k, k_s) + bt * jnp.einsum("bhd,bhe->bhde", k, v)
    o = jnp.einsum("bhd,bhde->bhe", q, s_new)
    return s_new, o


def _chunked_fwd_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch, seq, heads, dim = q.shape
    out = np.empty((batch, seq, heads, dim), dtype=np.float32)
    s = state
    for start in range(0, seq, chunk):
        end = min(start + chunk, seq)
        for t in range(start, end):
            s, o_t = _delta_step_np(s, q[:, t], k[:, t], v[:, t], beta[:, t])
            out[:, t] = o_t
    return out, s


def _pack_chunks_jax(
    q: jax.Array, k: jax.Array, v: jax.Array, beta: jax.Array, n_chunks: int, chunk: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    batch = q.shape[0]
    q = q.reshape(batch, n_chunks, chunk, q.shape[2], q.shape[3])
    k = k.reshape(batch, n_chunks, chunk, k.shape[2], k.shape[3])
    v = v.reshape(batch, n_chunks, chunk, v.shape[2], v.shape[3])
    beta = beta.reshape(batch, n_chunks, chunk, beta.shape[2])
    # scan over chunks, then over timesteps inside a chunk
    q = jnp.transpose(q, (1, 2, 0, 3, 4))
    k = jnp.transpose(k, (1, 2, 0, 3, 4))
    v = jnp.transpose(v, (1, 2, 0, 3, 4))
    beta = jnp.transpose(beta, (1, 2, 0, 3))
    return q, k, v, beta


def _unpack_chunks_jax(out: jax.Array, batch: int, heads: int, dim: int) -> jax.Array:
    # (n_chunks, chunk, B, H, D) -> (B, n_chunks * chunk, H, D)
    n_chunks, chunk = int(out.shape[0]), int(out.shape[1])
    out = jnp.transpose(out, (2, 0, 1, 3, 4))
    return out.reshape(batch, n_chunks * chunk, heads, dim)


def _chunked_fwd_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    chunk: int,
) -> tuple[jax.Array, jax.Array]:
    batch, seq, heads, dim = q.shape
    n_full, rem = seq // chunk, seq % chunk

    def step(s: jax.Array, ins: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
        qt, kt, vt, bt = ins
        return _delta_step_jax(s, qt, kt, vt, bt)

    def chunk_fn(s: jax.Array, ins: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
        return jax.lax.scan(step, s, ins)

    s = state
    pieces: list[jax.Array] = []
    if n_full:
        packed = _pack_chunks_jax(q[:, : n_full * chunk], k[:, : n_full * chunk],
                                  v[:, : n_full * chunk], beta[:, : n_full * chunk],
                                  n_full, chunk)
        s, out_full = jax.lax.scan(chunk_fn, s, packed)
        pieces.append(_unpack_chunks_jax(out_full, batch, heads, dim))
    if rem:
        qr = jnp.swapaxes(q[:, n_full * chunk :], 0, 1)
        kr = jnp.swapaxes(k[:, n_full * chunk :], 0, 1)
        vr = jnp.swapaxes(v[:, n_full * chunk :], 0, 1)
        br = jnp.swapaxes(beta[:, n_full * chunk :], 0, 1)
        s, out_rem = jax.lax.scan(step, s, (qr, kr, vr, br))
        pieces.append(jnp.swapaxes(out_rem, 0, 1))
    if pieces:
        out = jnp.concatenate(pieces, axis=1)
    else:
        out = jnp.zeros((batch, 0, heads, dim), dtype=q.dtype)
    return out, s


def _delta_vjp_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    go: np.ndarray,
    gs: np.ndarray,
    state0: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reverse-mode of the gated delta-rule. Chunking is only a traversal order."""
    batch, seq, heads, dim = q.shape
    eye = np.eye(dim, dtype=np.float32)
    s = state0
    s_hist = np.empty((seq + 1, batch, heads, dim, dim), dtype=np.float32)
    s_hist[0] = s
    for start in range(0, seq, chunk):
        end = min(start + chunk, seq)
        for t in range(start, end):
            kt = k[:, t]
            vt = v[:, t]
            bt = beta[:, t, :, None, None]
            kk = np.einsum("bhd,bhe->bhde", kt, kt)
            decay = eye - bt * kk
            s = np.einsum("bhij,bhjk->bhik", decay, s)
            s = s + bt * np.einsum("bhd,bhe->bhde", kt, vt)
            s_hist[t + 1] = s

    gq = np.empty_like(q)
    gk = np.empty_like(k)
    gv = np.empty_like(v)
    gbeta = np.empty_like(beta)
    g_s = gs.copy()
    starts = list(range(0, seq, chunk))
    for start in reversed(starts):
        end = min(start + chunk, seq)
        for t in range(end - 1, start - 1, -1):
            qt = q[:, t]
            kt = k[:, t]
            vt = v[:, t]
            bt = beta[:, t]
            s_t = s_hist[t + 1]
            s_prev = s_hist[t]
            a_t = eye - bt[:, :, None, None] * np.einsum("bhd,bhe->bhde", kt, kt)
            g_o = go[:, t]
            gq[:, t] = np.einsum("bhe,bhde->bhd", g_o, s_t)
            g_s = g_s + np.einsum("bhd,bhe->bhde", qt, g_o)
            g_s_prev = np.einsum("bhij,bhik->bhjk", a_t, g_s)
            g_a = np.einsum("bhik,bhjk->bhij", g_s, s_prev)
            gv[:, t] = np.einsum("bhde,bhd->bhe", g_s, kt) * bt[:, :, None]
            gk_v = np.einsum("bhde,bhe->bhd", g_s, vt) * bt[:, :, None]
            gbeta_v = np.einsum("bhde,bhd,bhe->bh", g_s, kt, vt)
            gbeta_a = -np.einsum("bhde,bhd,bhe->bh", g_a, kt, kt)
            gk_a = -bt[:, :, None] * (
                np.einsum("bhde,bhe->bhd", g_a, kt) + np.einsum("bhed,bhe->bhd", g_a, kt)
            )
            gk[:, t] = gk_v + gk_a
            gbeta[:, t] = gbeta_v + gbeta_a
            g_s = g_s_prev
    return (
        gq.astype(np.float32, copy=False),
        gk.astype(np.float32, copy=False),
        gv.astype(np.float32, copy=False),
        gbeta.astype(np.float32, copy=False),
        g_s.astype(np.float32, copy=False),
    )


def _delta_vjp_jax(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta: jax.Array,
    go: jax.Array,
    gs: jax.Array,
    state0: jax.Array,
    chunk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused reverse-state VJP on device. Chunking is only a traversal order."""
    del chunk
    _, seq, _, dim = q.shape
    eye = jnp.eye(dim, dtype=jnp.float32)
    q_t = jnp.swapaxes(q, 0, 1)
    k_t = jnp.swapaxes(k, 0, 1)
    v_t = jnp.swapaxes(v, 0, 1)
    beta_t = jnp.swapaxes(beta, 0, 1)
    go_t = jnp.swapaxes(go, 0, 1)

    def collect(s: jax.Array, ins: tuple[jax.Array, jax.Array, jax.Array]):
        kt, vt, bt = ins
        bt_e = bt[:, :, None, None]
        kk = jnp.einsum("bhd,bhe->bhde", kt, kt)
        decay = eye - bt_e * kk
        s_new = jnp.einsum("bhij,bhjk->bhik", decay, s)
        s_new = s_new + bt_e * jnp.einsum("bhd,bhe->bhde", kt, vt)
        return s_new, (s_new, s, decay)

    _, (s_t_all, s_prev_all, a_all) = jax.lax.scan(collect, state0, (k_t, v_t, beta_t))

    def bwd_step(g_s: jax.Array, ins: tuple):
        qt, kt, vt, bt, g_o, s_t, s_prev, a_t = ins
        gq_t = jnp.einsum("bhe,bhde->bhd", g_o, s_t)
        g_s = g_s + jnp.einsum("bhd,bhe->bhde", qt, g_o)
        g_s_prev = jnp.einsum("bhij,bhik->bhjk", a_t, g_s)
        g_a = jnp.einsum("bhik,bhjk->bhij", g_s, s_prev)
        gv_t = jnp.einsum("bhde,bhd->bhe", g_s, kt) * bt[:, :, None]
        gk_v = jnp.einsum("bhde,bhe->bhd", g_s, vt) * bt[:, :, None]
        gbeta_v = jnp.einsum("bhde,bhd,bhe->bh", g_s, kt, vt)
        gbeta_a = -jnp.einsum("bhde,bhd,bhe->bh", g_a, kt, kt)
        gk_a = -bt[:, :, None] * (
            jnp.einsum("bhde,bhe->bhd", g_a, kt) + jnp.einsum("bhed,bhe->bhd", g_a, kt)
        )
        return g_s_prev, (gq_t, gk_v + gk_a, gv_t, gbeta_v + gbeta_a)

    g_s0, (gq_t, gk_t, gv_t, gbeta_t) = jax.lax.scan(
        bwd_step,
        gs,
        (q_t, k_t, v_t, beta_t, go_t, s_t_all, s_prev_all, a_all),
        reverse=True,
        length=seq,
    )
    return (
        jnp.swapaxes(gq_t, 0, 1),
        jnp.swapaxes(gk_t, 0, 1),
        jnp.swapaxes(gv_t, 0, 1),
        jnp.swapaxes(gbeta_t, 0, 1),
        g_s0,
    )


@partial(jax.custom_vjp, nondiff_argnames=("config",))
def chunked_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    state: Array | None = None,
    config: LinearAttnConfig | None = None,
) -> tuple[Array, Array]:
    """Gated delta-rule linear attention.

    q, k, v: (batch, seq, heads, dim). beta: (batch, seq, heads) gate in (0, 1].
    state: (batch, heads, dim, dim) recurrent state, zeros if None.
    Returns (output, next_state) with output shaped like v.

    Must be the jax.custom_vjp object itself (not a wrapper around one).
    The backward is the fused reverse-state kernel, not JAX's default loop
    autodiff. jax.grad through the outputs equals
    chunked_delta_rule_bwd(residual, grads) with residual from
    chunked_delta_rule_fwd. config is not differentiated.
    """
    cfg = LinearAttnConfig() if config is None else config
    if cfg.chunk < 1:
        raise KernelError(f"chunk must be >= 1, got {cfg.chunk}")
    if _is_jax(q, k, v, beta, state):
        qj, kj, vj, bj, sj = _delta_jax_inputs(q, k, v, beta, state)
        return _chunked_fwd_jax(qj, kj, vj, bj, sj, int(cfg.chunk))
    q_np, k_np, v_np, beta_np, state_np = _validate_delta(q, k, v, beta, state)
    return _chunked_fwd_np(q_np, k_np, v_np, beta_np, state_np, int(cfg.chunk))


def _chunked_delta_rule_fwd(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    state: Array | None = None,
    config: LinearAttnConfig | None = None,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, ...]]:
    cfg = LinearAttnConfig() if config is None else config
    if cfg.chunk < 1:
        raise KernelError(f"chunk must be >= 1, got {cfg.chunk}")
    qj, kj, vj, bj, sj = _delta_jax_inputs(q, k, v, beta, state)
    out, ns = _chunked_fwd_jax(qj, kj, vj, bj, sj, int(cfg.chunk))
    return (out, ns), (qj, kj, vj, bj, sj)


def _chunked_delta_rule_bwd(
    config: LinearAttnConfig | None,
    residual: tuple[jax.Array, ...],
    grads: tuple[jax.Array | None, jax.Array | None],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    q, k, v, beta, state0 = residual
    go, gs = grads
    cfg = LinearAttnConfig() if config is None else config
    if go is None:
        go = jnp.zeros_like(q)
    else:
        go = jnp.asarray(go, dtype=jnp.float32)
    if gs is None:
        batch, _, heads, dim = q.shape
        gs = jnp.zeros((batch, heads, dim, dim), dtype=jnp.float32)
    else:
        gs = jnp.asarray(gs, dtype=jnp.float32)
    return _delta_vjp_jax(q, k, v, beta, go, gs, state0, int(cfg.chunk))


chunked_delta_rule.defvjp(_chunked_delta_rule_fwd, _chunked_delta_rule_bwd)


def chunked_delta_rule_fwd(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    state: Array | None,
    config: LinearAttnConfig,
) -> tuple[tuple[Array, Array], _DeltaResidual]:
    """Custom VJP forward. Residual is a pytree `_DeltaResidual`.

    `jax.jit(chunked_delta_rule_fwd, static_argnames=('config',))` must match
    eager at 1e-5 and return a pytree residual. `config` is a host
    `LinearAttnConfig` (static). Traced arrays must not be converted with
    `numpy.asarray`.
    """
    if config.chunk < 1:
        raise KernelError(f"chunk must be >= 1, got {config.chunk}")
    chunk = int(config.chunk)
    if _is_jax(q, k, v, beta, state):
        qj, kj, vj, bj, sj = _delta_jax_inputs(q, k, v, beta, state)
        out, ns = _chunked_fwd_jax(qj, kj, vj, bj, sj, chunk)
        residual = _DeltaResidual(q=qj, k=kj, v=vj, beta=bj, state0=sj, chunk=chunk)
        return (out, ns), residual
    q_np, k_np, v_np, beta_np, state_np = _validate_delta(q, k, v, beta, state)
    out, ns = _chunked_fwd_np(q_np, k_np, v_np, beta_np, state_np, chunk)
    residual = _DeltaResidual(
        q=q_np, k=k_np, v=v_np, beta=beta_np, state0=state_np, chunk=chunk
    )
    return (out, ns), residual


def chunked_delta_rule_bwd(
    residual: _DeltaResidual, grads: tuple[Array, Array]
) -> tuple[Array, ...]:
    """Custom VJP backward. Returns grads for (q, k, v, beta, state).

    `jax.jit(chunked_delta_rule_bwd)(residual, grads)` must match eager at
    1e-5. `residual` is a pytree `_DeltaResidual`. Traced arrays must not be
    converted with `numpy.asarray`.
    """
    go, gs = grads
    chunk = int(residual.chunk)
    if _is_jax(residual.q, residual.k, residual.v, residual.beta, residual.state0, go, gs):
        goj = go if isinstance(go, jax.Array) else jnp.asarray(go, dtype=jnp.float32)
        gsj = gs if isinstance(gs, jax.Array) else jnp.asarray(gs, dtype=jnp.float32)
        return _delta_vjp_jax(
            residual.q, residual.k, residual.v, residual.beta, goj, gsj, residual.state0, chunk
        )
    go_np, gs_np = _f32(go), _f32(gs)
    return _delta_vjp_np(
        residual.q,
        residual.k,
        residual.v,
        residual.beta,
        go_np,
        gs_np,
        residual.state0,
        chunk,
    )


# ---------------------------------------------------------------------------
# FP8 per-block fake-quant + linear
# ---------------------------------------------------------------------------


def _pad_blocks_np(x: np.ndarray, block: int) -> tuple[np.ndarray, int, int]:
    n = int(x.shape[-1])
    n_blocks = (n + block - 1) // block
    pad = n_blocks * block - n
    if pad:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        x = np.pad(x, pad_width)
    return x.reshape(*x.shape[:-1], n_blocks, block), n, n_blocks


def _pad_blocks_jax(x: jax.Array, block: int) -> tuple[jax.Array, int, int]:
    n = int(x.shape[-1])
    n_blocks = (n + block - 1) // block
    pad = n_blocks * block - n
    if pad:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        x = jnp.pad(x, pad_width)
    return x.reshape(*x.shape[:-1], n_blocks, block), n, n_blocks


def fp8_quantize(x: Array, *, block: int = DEFAULT_FP8_BLOCK) -> Fp8Meta:
    """Per-block abs-max scale, quantize to FP8. `x` is FP32 or BF16.

    `jax.jit(fp8_quantize, static_argnames=('block',))` must match eager at
    1e-5 and return a pytree `Fp8Meta`. `block` is a Python int (static).
    """
    if block < 1:
        raise KernelError(f"fp8 block must be >= 1, got {block}")
    if _is_jax(x):
        x_j = jnp.asarray(x, dtype=jnp.float32)
        if x_j.ndim < 1:
            raise KernelError("fp8_quantize expects at least a 1-D tensor")
        blocked, n, n_blocks = _pad_blocks_jax(x_j, block)
        amax = jnp.max(jnp.abs(blocked), axis=-1)
        scale = jnp.where(amax == 0, jnp.float32(1.0), amax / jnp.float32(FP8_E4M3_MAX))
        scaled = blocked / scale[..., None]
        bits = _to_e4m3_bits_jax(scaled)
        bits = bits.reshape(*x_j.shape[:-1], n_blocks * block)[..., :n]
        q = bits.astype(jnp.int8)
        return Fp8Meta(q=q, scale=scale.astype(jnp.float32), block=int(block), dtype=DType.FP8)
    x_np = _f32(x)
    if x_np.ndim < 1:
        raise KernelError("fp8_quantize expects at least a 1-D tensor")
    blocked, n, n_blocks = _pad_blocks_np(x_np, block)
    amax = np.max(np.abs(blocked), axis=-1)
    scale = np.where(amax == 0, np.float32(1.0), amax / np.float32(FP8_E4M3_MAX)).astype(np.float32)
    scaled = blocked / scale[..., None]
    bits = _to_e4m3_bits(scaled)
    bits = bits.reshape(*x_np.shape[:-1], n_blocks * block)[..., :n]
    q = bits.astype(np.int8, copy=False)
    return Fp8Meta(q=q, scale=scale, block=int(block), dtype=DType.FP8)


def fp8_dequantize(meta: Fp8Meta) -> Array:
    """Unpack FP8 + scales to FP32. Inverse of `fp8_quantize` up to rounding.

    `jax.jit(fp8_dequantize)(meta)` must match eager at 1e-5. `meta` is a
    pytree `Fp8Meta`. `block` / `dtype` stay meta (not traced arrays).
    """
    block = int(meta.block)
    if block < 1:
        raise KernelError(f"fp8 block must be >= 1, got {block}")
    if _is_jax(meta.q, meta.scale):
        bits = jnp.asarray(meta.q).astype(jnp.uint8)
        scale = jnp.asarray(meta.scale, dtype=jnp.float32)
        blocked, n, n_blocks = _pad_blocks_jax(bits, block)
        decoded = _E4M3_DECODE_J[blocked.astype(jnp.int32)]
        if scale.shape != tuple(decoded.shape[:-1]):
            raise KernelError(
                f"scale shape {tuple(scale.shape)} does not match "
                f"blocked payload {decoded.shape[:-1]}"
            )
        restored = decoded * scale[..., None]
        restored = restored.reshape(*restored.shape[:-2], n_blocks * block)[..., :n]
        return restored.astype(jnp.float32)
    bits = np.asarray(meta.q).astype(np.uint8, copy=False)
    scale = _f32(meta.scale)
    blocked, n, n_blocks = _pad_blocks_np(bits, block)
    decoded = _E4M3_DECODE[blocked]
    if scale.shape != tuple(decoded.shape[:-1]):
        raise KernelError(
            f"scale shape {scale.shape} does not match blocked payload {decoded.shape[:-1]}"
        )
    restored = decoded * scale[..., None]
    restored = restored.reshape(*restored.shape[:-2], n_blocks * block)[..., :n]
    return restored.astype(np.float32, copy=False)


def _fp8_linear_hats_jax(
    x: jax.Array, weight: jax.Array, block: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize both sides on-device and return ``(y, x_hat, w_hat)``."""
    if block < 1:
        raise KernelError(f"fp8 block must be >= 1, got {block}")
    x_j = jnp.asarray(x, dtype=jnp.float32)
    w_j = jnp.asarray(weight, dtype=jnp.float32)
    if w_j.ndim != 2:
        raise KernelError(f"weight must be 2-D (out, in), got {w_j.shape}")
    if x_j.shape[-1] != w_j.shape[-1]:
        raise KernelError(
            f"contracting dim mismatch: x[..., {x_j.shape[-1]}] vs weight[..., {w_j.shape[-1]}]"
        )
    x_hat = fp8_dequantize(fp8_quantize(x_j, block=block))
    w_hat = fp8_dequantize(fp8_quantize(w_j, block=block))
    y = jnp.matmul(x_hat, jnp.swapaxes(w_hat, -1, -2)).astype(jnp.float32)
    return y, x_hat, w_hat


def _fp8_linear_ste_bwd_jax(
    x_hat: jax.Array, w_hat: jax.Array, g: jax.Array
) -> tuple[jax.Array, jax.Array]:
    g_j = jnp.asarray(g, dtype=jnp.float32)
    in_f = x_hat.shape[-1]
    out_f = w_hat.shape[0]
    g_f = g_j.reshape(-1, out_f)
    x_f = x_hat.reshape(-1, in_f)
    grad_x = jnp.matmul(g_f, w_hat).reshape(x_hat.shape).astype(jnp.float32)
    grad_w = jnp.matmul(g_f.T, x_f).astype(jnp.float32)
    return grad_x, grad_w


@partial(jax.custom_vjp, nondiff_argnames=("block",))
def fp8_linear(x: Array, weight: Array, block: int = DEFAULT_FP8_BLOCK) -> Array:
    """y = x @ w^T with both sides quantized per-block to FP8.

    x: (..., in), weight: (out, in). Accumulates in FP32.
    Must be jax.custom_vjp so the backward uses the same scales as the
    forward. ``jax.grad(fp8_linear)(x, weight)`` equals
    ``fp8_linear_bwd(residual, g)`` with ``residual`` from
    ``fp8_linear_fwd``; scales are not differentiated (STE).
    """
    if _is_jax(x, weight):
        y, _, _ = _fp8_linear_hats_jax(x, weight, int(block))
        return y
    y, _ = fp8_linear_fwd(x, weight, block)
    return y


def _fp8_linear_fwd(x, weight, block):
    # Residual is the STE dequantized tensors, kept as JAX arrays (no host numpy).
    y, x_hat, w_hat = _fp8_linear_hats_jax(x, weight, int(block))
    return y, (x_hat, w_hat)


def _fp8_linear_bwd(_block, residual, g):
    x_hat, w_hat = residual
    return _fp8_linear_ste_bwd_jax(x_hat, w_hat, g)


fp8_linear.defvjp(_fp8_linear_fwd, _fp8_linear_bwd)


def fp8_linear_fwd(x: Array, weight: Array, block: int) -> tuple[Array, Any]:
    """Custom VJP forward. Residual is `(x_meta, w_meta)`, two pytree `Fp8Meta`.

    `jax.jit(fp8_linear_fwd, static_argnames=('block',))` must match eager at
    1e-5 and return a pytree residual. `block` is a Python int (static).
    Traced arrays must not be converted with `numpy.asarray`.
    """
    if block < 1:
        raise KernelError(f"fp8 block must be >= 1, got {block}")
    if _is_jax(x, weight) or _is_tracer(x, weight):
        x_j = jnp.asarray(x, dtype=jnp.float32)
        w_j = jnp.asarray(weight, dtype=jnp.float32)
        if w_j.ndim != 2:
            raise KernelError(f"weight must be 2-D (out, in), got {w_j.shape}")
        if x_j.shape[-1] != w_j.shape[-1]:
            raise KernelError(
                f"contracting dim mismatch: x[..., {x_j.shape[-1]}] vs weight[..., {w_j.shape[-1]}]"
            )
        x_meta = fp8_quantize(x_j, block=block)
        w_meta = fp8_quantize(w_j, block=block)
        x_hat = fp8_dequantize(x_meta)
        w_hat = fp8_dequantize(w_meta)
        y = jnp.matmul(x_hat, jnp.swapaxes(w_hat, -1, -2)).astype(jnp.float32)
        return y, (x_meta, w_meta)
    x_np = _f32(x)
    w_np = _f32(weight)
    if w_np.ndim != 2:
        raise KernelError(f"weight must be 2-D (out, in), got {w_np.shape}")
    if x_np.shape[-1] != w_np.shape[-1]:
        raise KernelError(
            f"contracting dim mismatch: x[..., {x_np.shape[-1]}] vs weight[..., {w_np.shape[-1]}]"
        )
    x_meta = fp8_quantize(x_np, block=block)
    w_meta = fp8_quantize(w_np, block=block)
    x_hat = fp8_dequantize(x_meta)
    w_hat = fp8_dequantize(w_meta)
    y = np.matmul(x_hat, np.swapaxes(w_hat, -1, -2)).astype(np.float32, copy=False)
    return y, (x_meta, w_meta)


def fp8_linear_bwd(residual: Any, g: Array) -> tuple[Array, Array]:
    """Returns (grad_x, grad_weight). Scales are not differentiated (STE).

    `jax.jit(fp8_linear_bwd)(residual, g)` must match eager at 1e-5.
    `residual` is the pytree pair from `fp8_linear_fwd`. Traced arrays must
    not be converted with `numpy.asarray`.
    """
    x_meta, w_meta = residual
    if _is_jax(x_meta.q, x_meta.scale, w_meta.q, w_meta.scale, g) or _is_tracer(
        x_meta.q, x_meta.scale, w_meta.q, w_meta.scale, g
    ):
        x_hat = fp8_dequantize(x_meta)
        w_hat = fp8_dequantize(w_meta)
        return _fp8_linear_ste_bwd_jax(x_hat, w_hat, g)
    x_hat = fp8_dequantize(x_meta)
    w_hat = fp8_dequantize(w_meta)
    x_hat = np.asarray(x_hat, dtype=np.float32)
    w_hat = np.asarray(w_hat, dtype=np.float32)
    g_np = _f32(g)
    in_f = int(x_hat.shape[-1])
    out_f = int(w_hat.shape[0])
    g_f = g_np.reshape(-1, out_f)
    x_f = x_hat.reshape(-1, in_f)
    grad_x = np.matmul(g_f, w_hat).reshape(x_hat.shape).astype(np.float32, copy=False)
    grad_w = np.matmul(g_f.T, x_f).astype(np.float32, copy=False)
    return grad_x, grad_w


# ---------------------------------------------------------------------------
# Expert-parallel dispatch / combine
# ---------------------------------------------------------------------------


def _validate_dispatch(tokens: Array, meta: DispatchMeta) -> tuple[
    Array, Array, Array, int, int
]:
    if tokens.ndim != 2:
        raise KernelError(f"tokens must be (n_tokens, d_model), got {tokens.shape}")
    n_tokens = int(tokens.shape[0])
    expert_ids = meta.expert_ids
    probs = meta.probs
    racks = meta.racks
    n_experts = int(meta.n_experts)
    max_racks = int(meta.max_racks)
    if expert_ids.ndim != 2 or int(expert_ids.shape[0]) != n_tokens:
        raise KernelError("expert_ids must be (n_tokens, top_k)")
    if tuple(probs.shape) != tuple(expert_ids.shape):
        raise KernelError(f"probs shape {probs.shape} != expert_ids shape {expert_ids.shape}")
    if tuple(racks.shape) != (n_experts,):
        raise KernelError(f"racks length {racks.shape} != n_experts={n_experts}")
    if n_experts < 0:
        raise KernelError(f"n_experts must be >= 0, got {n_experts}")
    if not _is_tracer(expert_ids) and np.asarray(expert_ids).size:
        ids_np = np.asarray(expert_ids)
        if np.any((ids_np < 0) | (ids_np >= n_experts)):
            raise KernelError(f"expert id out of range [0, {n_experts})")
    return expert_ids, probs, racks, n_experts, max_racks


def _check_rack_span(expert_ids: Array, racks: Array, max_racks: int) -> None:
    """KernelError if any token's chosen experts span more than max_racks racks."""
    if _is_tracer(expert_ids, racks):
        return
    expert_ids_np = np.asarray(expert_ids)
    racks_np = np.asarray(racks)
    if expert_ids_np.size == 0:
        return
    chosen = racks_np[expert_ids_np]
    if chosen.ndim == 1:
        chosen = chosen[:, None]
    ordered = np.sort(chosen, axis=1)
    n_unique = np.ones(ordered.shape[0], dtype=np.int32)
    if ordered.shape[1] > 1:
        n_unique = 1 + np.sum(ordered[:, 1:] != ordered[:, :-1], axis=1)
    if np.any(n_unique > max_racks):
        raise KernelError(
            f"token experts span more than max_racks={max_racks} distinct racks"
        )


def _dispatch_tables(
    expert_ids: np.ndarray, n_experts: int
) -> tuple[np.ndarray, np.ndarray, int]:
    n_tokens, top_k = expert_ids.shape
    n_assign = n_tokens * top_k
    if n_assign == 0:
        empty = np.full((n_experts, 0), -1, dtype=np.int32)
        return empty, empty.copy(), 0
    flat_e = np.ravel(expert_ids).astype(np.int32, copy=False)
    flat_t = np.repeat(np.arange(n_tokens, dtype=np.int32), top_k)
    flat_k = np.tile(np.arange(top_k, dtype=np.int32), n_tokens)
    counts = np.bincount(flat_e, minlength=n_experts)
    max_per = int(counts.max()) if n_experts else 0
    # Stable by (token, k) because ravel is C-order; slot = rank within expert.
    order = np.argsort(flat_e, kind="stable")
    sorted_e = flat_e[order]
    offsets = np.zeros(n_experts + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    slots = np.empty(n_assign, dtype=np.int32)
    slots[order] = np.arange(n_assign, dtype=np.int32) - offsets[sorted_e]
    token_index = np.full((n_experts, max_per), -1, dtype=np.int32)
    k_index = np.full((n_experts, max_per), -1, dtype=np.int32)
    token_index[flat_e, slots] = flat_t
    k_index[flat_e, slots] = flat_k
    return token_index, k_index, max_per


def _dispatch_tables_jax(
    expert_ids: jax.Array, n_experts: int, max_per: int
) -> tuple[jax.Array, jax.Array, int]:
    n_tokens = int(expert_ids.shape[0])
    top_k = int(expert_ids.shape[1])
    n_assign = n_tokens * top_k
    n_experts = int(n_experts)
    max_per = int(max_per)
    if n_assign == 0 or max_per == 0:
        empty = jnp.full((n_experts, max_per), -1, dtype=jnp.int32)
        return empty, empty, max_per
    flat_e = jnp.ravel(expert_ids).astype(jnp.int32)
    flat_t = jnp.repeat(jnp.arange(n_tokens, dtype=jnp.int32), top_k)
    flat_k = jnp.tile(jnp.arange(top_k, dtype=jnp.int32), n_tokens)
    counts = jnp.bincount(flat_e, length=n_experts)
    order = jnp.argsort(flat_e, stable=True)
    sorted_e = flat_e[order]
    offsets = jnp.zeros((n_experts + 1,), dtype=jnp.int32)
    offsets = offsets.at[1:].set(jnp.cumsum(counts).astype(jnp.int32))
    slots = jnp.zeros((n_assign,), dtype=jnp.int32)
    slots = slots.at[order].set(
        jnp.arange(n_assign, dtype=jnp.int32) - offsets[sorted_e]
    )
    token_index = jnp.full((n_experts, max_per), -1, dtype=jnp.int32)
    k_index = jnp.full((n_experts, max_per), -1, dtype=jnp.int32)
    token_index = token_index.at[flat_e, slots].set(flat_t)
    k_index = k_index.at[flat_e, slots].set(flat_k)
    return token_index, k_index, max_per


def _ep_dispatch_numpy(tokens: np.ndarray, token_index: np.ndarray) -> np.ndarray:
    n_experts, max_per = token_index.shape
    d_model = int(tokens.shape[-1])
    dispatched = np.zeros((n_experts, max_per, d_model), dtype=np.float32)
    valid = token_index >= 0
    if np.any(valid):
        dispatched[valid] = tokens[token_index[valid]]
    return dispatched


def _ep_dispatch_jax(tokens: jax.Array, token_index: Array) -> jax.Array:
    tokens = jnp.asarray(tokens, dtype=jnp.float32)
    idx = jnp.asarray(token_index, dtype=jnp.int32)
    valid = idx >= 0
    gathered = tokens[jnp.where(valid, idx, jnp.int32(0))]
    return jnp.where(valid[..., None], gathered, jnp.zeros_like(gathered))


def _ep_dispatch_vjp_jax(
    g_dispatched: jax.Array, token_index: Array, n_tokens: int
) -> jax.Array:
    g_dispatched = jnp.asarray(g_dispatched, dtype=jnp.float32)
    idx = jnp.asarray(token_index, dtype=jnp.int32)
    valid = idx >= 0
    contrib = jnp.where(valid[..., None], g_dispatched, jnp.zeros_like(g_dispatched))
    grad = jnp.zeros((n_tokens, g_dispatched.shape[-1]), dtype=jnp.float32)
    return grad.at[jnp.where(valid, idx, jnp.int32(0))].add(contrib)


def _ep_combine_numpy(
    expert_out: np.ndarray,
    probs: np.ndarray,
    token_index: np.ndarray,
    k_index: np.ndarray,
    n_tokens: int,
) -> np.ndarray:
    d_model = int(expert_out.shape[-1])
    combined = np.zeros((n_tokens, d_model), dtype=np.float32)
    valid = token_index >= 0
    if np.any(valid):
        t = token_index[valid]
        k = k_index[valid]
        weights = probs[t, k][:, None]
        np.add.at(combined, t, weights * expert_out[valid])
    return combined


def _ep_combine_jax(
    expert_out: jax.Array,
    probs: Array,
    token_index: Array,
    k_index: Array,
    n_tokens: int,
) -> jax.Array:
    expert_out = jnp.asarray(expert_out, dtype=jnp.float32)
    probs_j = jnp.asarray(probs, dtype=jnp.float32)
    idx_t = jnp.asarray(token_index, dtype=jnp.int32)
    idx_k = jnp.asarray(k_index, dtype=jnp.int32)
    valid = idx_t >= 0
    safe_t = jnp.where(valid, idx_t, jnp.int32(0))
    safe_k = jnp.where(valid, idx_k, jnp.int32(0))
    weights = jnp.where(valid, probs_j[safe_t, safe_k], jnp.float32(0))
    contrib = expert_out * weights[..., None]
    combined = jnp.zeros((n_tokens, expert_out.shape[-1]), dtype=jnp.float32)
    return combined.at[safe_t].add(contrib)


def _ep_combine_vjp_jax(
    g_combined: jax.Array, probs: Array, token_index: Array, k_index: Array
) -> jax.Array:
    g_combined = jnp.asarray(g_combined, dtype=jnp.float32)
    probs_j = jnp.asarray(probs, dtype=jnp.float32)
    idx_t = jnp.asarray(token_index, dtype=jnp.int32)
    idx_k = jnp.asarray(k_index, dtype=jnp.int32)
    valid = idx_t >= 0
    safe_t = jnp.where(valid, idx_t, jnp.int32(0))
    safe_k = jnp.where(valid, idx_k, jnp.int32(0))
    weights = jnp.where(valid, probs_j[safe_t, safe_k], jnp.float32(0))
    return g_combined[safe_t] * weights[..., None]


def _make_dispatch_residual(
    token_index: np.ndarray, k_index: np.ndarray, max_per: int, as_jax: bool
) -> _DispatchResidual:
    if as_jax:
        return _DispatchResidual(
            token_index=jnp.asarray(token_index),
            k_index=jnp.asarray(k_index),
            max_per_expert=max_per,
        )
    return _DispatchResidual(
        token_index=token_index, k_index=k_index, max_per_expert=max_per
    )


def _ep_dispatch_impl(
    tokens: Array, meta: DispatchMeta, *, as_jax: bool
) -> tuple[Array, _DispatchResidual]:
    expert_ids, _probs, racks, n_experts, max_racks = _validate_dispatch(tokens, meta)
    _check_rack_span(expert_ids, racks, max_racks)
    if as_jax:
        max_per = _static_max_per_of(meta, expert_ids, n_experts)
        token_index, k_index, max_per = _dispatch_tables_jax(
            jnp.asarray(expert_ids, dtype=jnp.int32), n_experts, max_per
        )
        residual = _make_dispatch_residual(token_index, k_index, max_per, as_jax=True)
        return _ep_dispatch_jax(tokens, token_index), residual
    token_index, k_index, max_per = _dispatch_tables(np.asarray(expert_ids), n_experts)
    residual = _make_dispatch_residual(token_index, k_index, max_per, as_jax=False)
    return _ep_dispatch_numpy(_f32(tokens), token_index), residual


@jax.custom_vjp
def ep_dispatch(tokens: Array, meta: DispatchMeta) -> tuple[Array, Any]:
    """All-to-all tokens to experts.

    tokens: (n_tokens, d_model). Returns (dispatched, residual) where
    dispatched is (n_experts, max_per_expert, d_model) padded, and residual
    is the inverse permutation `ep_combine` needs.

    Raises KernelError if a token's experts span more than meta.max_racks.

    Must be the jax.custom_vjp object itself (not a wrapper around one).
    jax.grad through dispatched equals the scatter of cotangents onto tokens
    via residual. meta is not differentiated. residual is the inverse
    permutation ep_combine needs and is not differentiated.
    jax.jit(ep_dispatch)(tokens, meta) must match the eager result (1e-5).
    The traced path must not require DispatchMeta to be an abstract array.
    """
    return _ep_dispatch_impl(
        tokens, meta, as_jax=_is_jax(tokens, meta.expert_ids)
    )


def _ep_dispatch_fwd(tokens: Array, meta: DispatchMeta):
    dispatched, residual = _ep_dispatch_impl(tokens, meta, as_jax=True)
    n_tokens = int(tokens.shape[0])
    return (dispatched, residual), (
        jnp.asarray(residual.token_index, dtype=jnp.int32),
        n_tokens,
    )


def _ep_dispatch_bwd(res: tuple[Any, ...], g):
    token_index, n_tokens = res
    g_disp, _g_residual = g
    if g_disp is None:
        return (jnp.zeros((n_tokens, 0), dtype=jnp.float32), None)
    return (_ep_dispatch_vjp_jax(g_disp, token_index, n_tokens), None)


ep_dispatch.defvjp(_ep_dispatch_fwd, _ep_dispatch_bwd)


def _require_int_index(name: str, x: Array) -> None:
    dtype = getattr(x, "dtype", None)
    if dtype is not None and not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError(f"{name} must be integer")


def _ep_combine_impl(
    expert_out: Array, meta: DispatchMeta, residual: Any, *, as_jax: bool
) -> Array:
    _require_int_index("residual.token_index", residual.token_index)
    n_experts = int(meta.n_experts)
    n_tokens = int(meta.expert_ids.shape[0])
    if expert_out.ndim != 3 or int(expert_out.shape[0]) != n_experts:
        raise KernelError("expert_out must be (n_experts, max_per_expert, d_model)")
    if as_jax:
        return _ep_combine_jax(
            expert_out, meta.probs, residual.token_index, residual.k_index, n_tokens
        )
    return _ep_combine_numpy(
        _f32(expert_out),
        _f32(meta.probs),
        np.asarray(residual.token_index),
        np.asarray(residual.k_index),
        n_tokens,
    )


@jax.custom_vjp
def ep_combine(expert_out: Array, meta: DispatchMeta, residual: Any) -> Array:
    """Weighted sum of expert outputs back to token order.

    expert_out: (n_experts, max_per_expert, d_model). Weights are meta.probs.
    Returns (n_tokens, d_model).

    Must be the jax.custom_vjp object itself (not a wrapper around one).
    jax.grad through the combined tokens equals the weighted scatter of
    cotangents onto expert slots using meta.probs and residual.
    residual is not differentiated. meta routing ids are not differentiated.
    jax.jit(ep_combine)(expert_out, meta, residual) must match eager (1e-5).
    jax.jit of dispatch-then-combine must compose.
    """
    return _ep_combine_impl(
        expert_out,
        meta,
        residual,
        as_jax=_is_jax(expert_out, meta.probs, residual.token_index),
    )


def _ep_combine_fwd(expert_out: Array, meta: DispatchMeta, residual: Any):
    out = _ep_combine_impl(expert_out, meta, residual, as_jax=True)
    return out, (
        jnp.asarray(meta.probs, dtype=jnp.float32),
        jnp.asarray(residual.token_index, dtype=jnp.int32),
        jnp.asarray(residual.k_index, dtype=jnp.int32),
    )


def _ep_combine_bwd(res: tuple[jax.Array, ...], g: jax.Array):
    probs, token_index, k_index = res
    return (_ep_combine_vjp_jax(g, probs, token_index, k_index), None, None)


ep_combine.defvjp(_ep_combine_fwd, _ep_combine_bwd)


def ep_dispatch_fwd(tokens: Array, meta: DispatchMeta) -> tuple[Array, _DispatchResidual]:
    """Custom VJP forward. Residual is a pytree `_DispatchResidual`.

    `jax.jit(ep_dispatch_fwd)(tokens, meta)` must match eager at 1e-5 and
    return a pytree residual. DispatchMeta is a registered dataclass.
    Traced arrays must not be converted with `numpy.asarray`.
    Public `ep_dispatch` custom_vjp stays.
    """
    (dispatched, residual), (_, n_tokens) = _ep_dispatch_fwd(tokens, meta)
    return dispatched, replace(residual, n_tokens=int(n_tokens))


def ep_dispatch_bwd(residual: _DispatchResidual, g: Array) -> Array:
    """Returns grad_tokens. residual is not differentiated.

    `jax.jit(ep_dispatch_bwd)(residual, g)` must match eager at 1e-5.
    `residual` is the pytree `_DispatchResidual` from `ep_dispatch_fwd`.
    Traced arrays must not be converted with `numpy.asarray`.
    """
    n_tokens = residual.n_tokens
    if n_tokens is None:
        raise KernelError("ep_dispatch_bwd residual missing n_tokens")
    grad_tokens, _meta_grad = _ep_dispatch_bwd((residual.token_index, int(n_tokens)), (g, None))
    return grad_tokens


def ep_combine_fwd(
    expert_out: Array, meta: DispatchMeta, residual: _DispatchResidual
) -> tuple[Array, Any]:
    """Custom VJP forward. Residual is a pytree for `ep_combine_bwd`.

    `jax.jit(ep_combine_fwd)(expert_out, meta, residual)` must match eager
    at 1e-5 and return a pytree residual. DispatchMeta is a registered
    dataclass. Traced arrays must not be converted with `numpy.asarray`.
    Public `ep_combine` custom_vjp stays.
    """
    combined, bwd_residual = _ep_combine_fwd(expert_out, meta, residual)
    return combined, bwd_residual


def ep_combine_bwd(residual: Any, g: Array) -> Array:
    """Returns grad_expert_out. residual is not differentiated.

    `jax.jit(ep_combine_bwd)(residual, g)` must match eager at 1e-5.
    `residual` is the pytree from `ep_combine_fwd`. Traced arrays must
    not be converted with `numpy.asarray`.
    """
    grad_expert_out, _meta_grad, _res_grad = _ep_combine_bwd(residual, g)
    return grad_expert_out
