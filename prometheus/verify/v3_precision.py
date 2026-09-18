"""V3 precision runner: BF16 vs FP8 vs NVFP4 fake-quant (spec 5.3, 16.2).

CPU analog is a tiny toy (not 0.1–0.5B). Passing it is not V3 verified;
the 8×H200 0.1–0.5B run is the real gate. NVFP4 is numerics-only (no HW
on H200).
"""

from __future__ import annotations

from functools import partial
from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import kernels
import train

FP8_REL_MAX = 0.005
DEFAULT_STEPS = 2000

# Tiny analog: contracting dim is a multiple of the flagship FP8 block (128)
# and of the NVFP4 microblock (16). Not 0.1–0.5B; not V3 verified.
_TOY_DIM = 128
_TOY_FFN = 128
_TOY_VOCAB = 16
_TOY_EXPERTS = 2
_TOY_BATCH = 2
_TOY_SEQ = 4
_TOY_LAYERS = 3
_N_TAIL = train.FLAGSHIP_BF16_TAIL_LAYERS
_N_BODY = _TOY_LAYERS - _N_TAIL
_FP8_BLOCK = train.FLAGSHIP_FP8_BLOCK
_NVFP4_BLOCK = 16
_NVFP4_MAX = 6.0
_REL_FLOOR = 1e-12
_LR = 1.0e-2
_INIT_STD = 0.02
_NORM_EPS = 1e-6

_E2M1_POS = jnp.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=jnp.float32,
)


class V3Result(TypedDict):
    fp8_loss_rel_diff: float
    nvfp4_numerics_ok: bool


class V3Error(Exception):
    """Raised when V3 cannot produce a finite precision report."""


def _to_bf16(x: Any) -> Any:
    """Forward BF16 truncation with STE back to the FP32 master."""
    x32 = jnp.asarray(x, dtype=jnp.float32)
    y = x32.astype(jnp.bfloat16).astype(jnp.float32)
    return x32 + jax.lax.stop_gradient(y - x32)


def _ste_fp8(x: Any, block: int = _FP8_BLOCK) -> Any:
    """E4M3 per-block fake-quant via `kernels.fp8_quantize`, STE to FP32."""
    x32 = jnp.asarray(x, dtype=jnp.float32)
    y = kernels.fp8_dequantize(kernels.fp8_quantize(x32, block=block))
    y = jnp.asarray(y, dtype=jnp.float32)
    return x32 + jax.lax.stop_gradient(y - x32)


def _nvfp4_round(x: Any, block: int = _NVFP4_BLOCK) -> Any:
    """E2M1 per-microblock fake-quant of `x` along the contracting dim."""
    x32 = jnp.asarray(x, dtype=jnp.float32)
    n = int(x32.shape[-1])
    n_blocks = (n + block - 1) // block
    pad = n_blocks * block - n
    if pad:
        x32 = jnp.pad(x32, [(0, 0)] * (x32.ndim - 1) + [(0, pad)])
    blocked = x32.reshape(*x32.shape[:-1], n_blocks, block)
    amax = jnp.max(jnp.abs(blocked), axis=-1)
    scale = jnp.where(amax == 0, jnp.float32(1.0), amax / jnp.float32(_NVFP4_MAX))
    scaled = blocked / scale[..., None]
    sign = jnp.where(scaled < 0, jnp.float32(-1.0), jnp.float32(1.0))
    mag = jnp.minimum(jnp.abs(scaled), jnp.float32(_NVFP4_MAX))
    idx = jnp.argmin(jnp.abs(mag[..., None] - _E2M1_POS), axis=-1)
    q = sign * _E2M1_POS[idx]
    out = (q * scale[..., None]).reshape(*x32.shape[:-1], n_blocks * block)
    return out[..., :n]


def _ste_nvfp4(x: Any) -> Any:
    x32 = jnp.asarray(x, dtype=jnp.float32)
    y = _nvfp4_round(x32)
    return x32 + jax.lax.stop_gradient(y - x32)


def _linear(x: Any, weight: Any, kind: str) -> Any:
    """y = x @ w^T. `kind` is bf16, fp8 (both sides), or nvfp4 (weights)."""
    x32 = jnp.asarray(x, dtype=jnp.float32)
    w32 = jnp.asarray(weight, dtype=jnp.float32)
    if kind == "fp8":
        return _ste_fp8(x32) @ _ste_fp8(w32).T
    if kind == "nvfp4":
        return _to_bf16(x32) @ _ste_nvfp4(w32).T
    return _to_bf16(x32) @ _to_bf16(w32).T


def _rms_norm(x: Any, weight: Any) -> Any:
    x32 = jnp.asarray(x, dtype=jnp.float32)
    ms = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    y = x32 * jax.lax.rsqrt(ms + jnp.float32(_NORM_EPS))
    return _to_bf16(y * _to_bf16(weight))


def _softmax_bf16(x: Any) -> Any:
    return jax.nn.softmax(_to_bf16(x), axis=-1)


def _init_params(rng: np.random.Generator) -> dict[str, Any]:
    def w(*shape: int) -> Any:
        return jnp.asarray(rng.normal(0.0, _INIT_STD, size=shape), dtype=jnp.float32)

    def ones(*shape: int) -> Any:
        return jnp.ones(shape, dtype=jnp.float32)

    return {
        "embed": w(_TOY_VOCAB, _TOY_DIM),
        "latent": w(_TOY_DIM, _TOY_DIM),
        "body_norm": ones(_N_BODY, _TOY_DIM),
        "w_in": w(_N_BODY, _TOY_FFN, _TOY_DIM),
        "w_out": w(_N_BODY, _TOY_DIM, _TOY_FFN),
        "router": w(_N_BODY, _TOY_EXPERTS, _TOY_DIM),
        "expert_w_in": w(_N_BODY, _TOY_EXPERTS, _TOY_FFN, _TOY_DIM),
        "expert_w_out": w(_N_BODY, _TOY_EXPERTS, _TOY_DIM, _TOY_FFN),
        "tail_norm": ones(_N_TAIL, _TOY_DIM),
        "tail_w": w(_N_TAIL, _TOY_DIM, _TOY_DIM),
        "unembed": w(_TOY_VOCAB, _TOY_DIM),
    }


def _clone(params: dict[str, Any]) -> dict[str, Any]:
    return {k: jnp.array(v, dtype=jnp.float32) for k, v in params.items()}


def _forward(params: dict[str, Any], tokens: Any, precision: str) -> tuple[Any, dict[str, Any]]:
    """Spec 5.3 roles: master is FP32; router/norm/embed/softmax/tail/latent BF16.

    Body linears are BF16 or FP8. Routed expert weights are NVFP4 on that path.
    """
    body_kind = "bf16" if precision == "bf16" else "fp8"
    expert_kind = "nvfp4" if precision == "nvfp4" else body_kind

    tok = jnp.asarray(tokens, dtype=jnp.int32)
    h = _to_bf16(params["embed"])[tok]
    h = h + _linear(h, params["latent"], "bf16")

    for i in range(_N_BODY):
        n = _rms_norm(h, params["body_norm"][i])
        mid = jax.nn.silu(_linear(n, params["w_in"][i], body_kind))
        h = h + _linear(mid, params["w_out"][i], body_kind)
        gates = _softmax_bf16(_linear(n, params["router"][i], "bf16"))
        mix = jnp.zeros_like(h)
        for e in range(_TOY_EXPERTS):
            eh = jax.nn.silu(_linear(n, params["expert_w_in"][i, e], expert_kind))
            out = _linear(eh, params["expert_w_out"][i, e], expert_kind)
            mix = mix + gates[..., e : e + 1] * out
        h = h + mix

    for t in range(_N_TAIL):
        n = _rms_norm(h, params["tail_norm"][t])
        h = h + _linear(n, params["tail_w"][t], "bf16")

    logits = _linear(h, params["unembed"], "bf16")
    labels = tok[:, 1:]
    logit_lm = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logit_lm, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    loss = jnp.mean(nll)
    return loss, {"hidden": h, "logits": logits}


@partial(jax.jit, static_argnames=("precision",))
def _sgd_step(
    params: dict[str, Any], tokens: Any, precision: str, lr: float
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    def loss_fn(p: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return _forward(p, tokens, precision)

    (loss, acts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updated = {k: params[k] - jnp.float32(lr) * grads[k] for k in params}
    return updated, loss, acts


def _train(
    params: dict[str, Any], tokens: Any, precision: str, steps: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    losses: list[float] = []
    acts_np: dict[str, np.ndarray] | None = None
    p = params
    for _ in range(steps):
        p, loss, acts = _sgd_step(p, tokens, precision, _LR)
        loss_np = np.asarray(loss, dtype=np.float64)
        losses.append(float(loss_np))
        acts_np = {k: np.asarray(v, dtype=np.float32) for k, v in acts.items()}
    if acts_np is None:
        raise V3Error("training produced no activations")
    return np.asarray(losses, dtype=np.float64), acts_np


def _fp8_loss_rel_diff(fp8: np.ndarray, bf16: np.ndarray) -> float:
    a = np.asarray(fp8, dtype=np.float64).reshape(-1)
    b = np.asarray(bf16, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise V3Error("fp8 and bf16 loss trajectories must have equal length")
    denom = np.maximum(np.abs(b), _REL_FLOOR)
    rel = np.abs(a - b) / denom
    value = float(np.max(rel))
    if not np.isfinite(value) or value < 0.0:
        raise V3Error("fp8_loss_rel_diff must be finite and >= 0")
    return value


def _finite(*arrays: np.ndarray) -> bool:
    return all(np.isfinite(np.asarray(a)).all() for a in arrays)


def run_v3(*, gpus: int, steps: int = DEFAULT_STEPS) -> V3Result:
    """Train a tiny analog at BF16, FP8, and NVFP4 and report the 16.2 numbers.

    `gpus` is the intended V3 world size (8). The CPU analog ignores it after
    validating `gpus >= 1`. Raises V3Error when gpus or steps is invalid, or
    when the backend cannot produce a finite report.
    """
    if gpus < 1:
        raise V3Error("gpus must be >= 1")
    if steps < 1:
        raise V3Error("steps must be >= 1")

    rng = np.random.default_rng(0)
    master = _init_params(rng)
    tokens = rng.integers(0, _TOY_VOCAB, size=(_TOY_BATCH, _TOY_SEQ), dtype=np.int32)
    tokens_j = jnp.asarray(tokens, dtype=jnp.int32)

    bf16_losses, _ = _train(_clone(master), tokens_j, "bf16", steps)
    fp8_losses, _ = _train(_clone(master), tokens_j, "fp8", steps)

    ran_nvfp4 = False
    nv_losses, nv_acts = _train(_clone(master), tokens_j, "nvfp4", steps)
    ran_nvfp4 = True

    rel = _fp8_loss_rel_diff(fp8_losses, bf16_losses)
    nv_ok = bool(ran_nvfp4 and _finite(nv_losses, nv_acts["hidden"], nv_acts["logits"]))
    if not nv_ok:
        raise V3Error("NVFP4 fake-quant path did not produce finite numerics")
    if not _finite(bf16_losses, fp8_losses):
        raise V3Error("BF16/FP8 trajectories were not finite")

    return {
        "fp8_loss_rel_diff": rel,
        "nvfp4_numerics_ok": nv_ok,
    }
