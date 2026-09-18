"""Independent NumPy V3 precision protocol (spec 16.2 / 5.3).

Slow and obvious. Does **not** import JAX, torch, ``prometheus.verify.v3_precision``,
``model/``, ``train/``, or ``kernels/``. Production ``run_v3`` must not import this
module; tests import both.

V3 meaning
----------
``fp8_loss_rel_diff``
    Relative gap between an **FP8-linear** train trajectory and a **BF16**
    trajectory over ``steps`` steps::

        max_t |L_fp8[t] - L_bf16[t]| / max(|L_bf16[t]|, floor)

    Scalars (final or mean) are allowed as length-1. Gate: finite, ``>= 0``,
    ``<= FP8_REL_MAX`` (0.005). Must not be obtained by comparing a run to itself.
``nvfp4_numerics_ok``
    True iff the NVFP4 fake-quant path actually ran and produced finite losses
    and activations (no NaN/Inf). Numerics only, never speed. Must not be True
    because NVFP4 was skipped.

Precision (spec 5.3)
--------------------
Master weights stay FP32. Linears fake-quant to FP8 E4M3 with per-block scaling
along the contracting dim (``FLAGSHIP_FP8_BLOCK`` = 128). Router, RMSNorm,
embeddings, softmax, last two layers, and the latent adapter stay BF16 (high 16
bits of FP32). NVFP4 is E2M1 fake-quant of routed expert weights with microblock
16; H200 has no NVFP4 HW.

The toy net below is the protocol's reference math, not a 0.1–0.5B flagship run.
Spec V3 is 8 H200; this analog is not V3 verified.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

Array = np.ndarray

# F4 ``check_exit(V3)`` / stub ``prometheus.verify.v3_precision``.
FP8_REL_MAX = 0.005
DEFAULT_STEPS = 2000
REL_FLOOR = 1e-12

# Production FP8 block (``train.FLAGSHIP_FP8_BLOCK``). Along the contracting dim.
FLAGSHIP_FP8_BLOCK = 128
# NVIDIA NVFP4 E2M1 microblock.
NVFP4_MICROBLOCK = 16
FP8_E4M3_MAX = 448.0
NVFP4_E2M1_MAX = 6.0
# Positive E2M1 codebook; sign is applied separately.
E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# Spec 16.2 row V3.
SPEC_GPUS = 8

# Toy net: contracting dims are multiples of both 128 and 16.
TOY_VOCAB = 8
TOY_DIM = 128
TOY_FFN = 128
TOY_LAYERS_TOTAL = 3  # 1 mixed-precision block + last two BF16 layers
TOY_EXPERTS = 2
TOY_BATCH = 2
TOY_SEQ = 4
TOY_SEED = 0
TOY_LR = 0.05
TOY_FP8_BLOCK = FLAGSHIP_FP8_BLOCK
TOY_NVFP4_BLOCK = NVFP4_MICROBLOCK
TOY_FD_EPS = 1e-3
TOY_GRAD_RTOL = 5e-2
TOY_GRAD_ATOL = 5e-3

# Well above the 0.5% gate so float32 rounding cannot flip the fail side.
GOLDEN_GAP_ABOVE_GATE = 1e-2

assert TOY_DIM % TOY_FP8_BLOCK == 0
assert TOY_DIM % TOY_NVFP4_BLOCK == 0
assert TOY_FFN % TOY_FP8_BLOCK == 0
assert TOY_FFN % TOY_NVFP4_BLOCK == 0


# ---------------------------------------------------------------------------
# Casts and fake-quant (obvious math, no packed storage)
# ---------------------------------------------------------------------------


def to_bf16(x: Any) -> Array:
    """FP32 → BF16 → FP32 by keeping the high 16 bits (truncate)."""
    x32 = np.asarray(x, dtype=np.float32)
    bits = x32.view(np.uint32) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def _e4m3_positive_table() -> Array:
    """Finite non-negative E4M3FN values (exp=15, mantissa=7 is NaN, omitted)."""
    vals: list[float] = []
    min_sub = 2.0**-9
    for man in range(8):
        vals.append(man * min_sub)
    for exp in range(1, 15):
        for man in range(8):
            vals.append((2.0 ** (exp - 7)) * (1.0 + man / 8.0))
    for man in range(7):
        vals.append((2.0 ** (15 - 7)) * (1.0 + man / 8.0))
    return np.asarray(vals, dtype=np.float32)


_E4M3_POS = _e4m3_positive_table()


def _round_to_codebook(x: Array, codebook: Array, max_finite: float) -> Array:
    x = np.asarray(x, dtype=np.float32)
    sign = np.where(np.signbit(x), np.float32(-1.0), np.float32(1.0))
    mag = np.abs(x)
    mag = np.nan_to_num(mag, nan=max_finite, posinf=max_finite, neginf=max_finite)
    mag = np.minimum(mag, np.float32(max_finite))
    idx = np.argmin(np.abs(mag[..., None] - codebook), axis=-1)
    return (sign * codebook[idx]).astype(np.float32)


def _pad_blocks(x: Array, block: int) -> tuple[Array, int, int]:
    x = np.asarray(x)
    if block < 1:
        raise ValueError("block must be >= 1")
    n = int(x.shape[-1])
    n_blocks = (n + block - 1) // block
    pad = n_blocks * block - n
    if pad:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
        x = np.pad(x, pad_width)
    return x.reshape(*x.shape[:-1], n_blocks, block), n, n_blocks


def _fakequant_per_block(
    x: Array, *, block: int, max_finite: float, codebook: Array
) -> tuple[Array, Array]:
    """Per-block abs-max scale, round to ``codebook``, dequant. Last axis contracts."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim < 1:
        raise ValueError("fake-quant expects at least a 1-D tensor")
    blocked, n, n_blocks = _pad_blocks(x, block)
    amax = np.max(np.abs(blocked), axis=-1)
    scale = np.empty_like(amax, dtype=np.float32)
    zero = amax == 0
    scale[zero] = np.float32(1.0)
    scale[~zero] = (amax[~zero] / np.float32(max_finite)).astype(np.float32)
    q = _round_to_codebook(blocked / scale[..., None], codebook, max_finite)
    restored = q * scale[..., None]
    restored = restored.reshape(*x.shape[:-1], n_blocks * block)[..., :n]
    return restored.astype(np.float32), scale.astype(np.float32)


def fakequant_fp8(x: Array, *, block: int = TOY_FP8_BLOCK) -> Array:
    """E4M3FN per-block fake-quant along the last (contracting) axis."""
    y, _ = _fakequant_per_block(
        x, block=block, max_finite=FP8_E4M3_MAX, codebook=_E4M3_POS
    )
    return y


def fp8_block_scales(x: Array, *, block: int = TOY_FP8_BLOCK) -> Array:
    """Per-block FP8 scales of ``x``; shape is ``x.shape[:-1] + (n_blocks,)``."""
    _, scale = _fakequant_per_block(
        x, block=block, max_finite=FP8_E4M3_MAX, codebook=_E4M3_POS
    )
    return scale


def fakequant_nvfp4(x: Array, *, block: int = TOY_NVFP4_BLOCK) -> Array:
    """NVFP4 E2M1 per-microblock fake-quant along the last axis."""
    y, _ = _fakequant_per_block(
        x, block=block, max_finite=NVFP4_E2M1_MAX, codebook=E2M1_POS
    )
    return y


def nvfp4_block_scales(x: Array, *, block: int = TOY_NVFP4_BLOCK) -> Array:
    """Per-microblock NVFP4 scales of ``x``."""
    _, scale = _fakequant_per_block(
        x, block=block, max_finite=NVFP4_E2M1_MAX, codebook=E2M1_POS
    )
    return scale


def fp8_linear(x: Array, weight: Array, *, block: int = TOY_FP8_BLOCK) -> Array:
    """``y = x_hat @ w_hat^T`` with both sides E4M3 per-block fake-quant."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D (out, in), got {weight.shape}")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"contracting dim mismatch: x[..., {x.shape[-1]}] vs weight[..., {weight.shape[-1]}]"
        )
    x_hat = fakequant_fp8(x, block=block)
    w_hat = fakequant_fp8(weight, block=block)
    return np.matmul(x_hat, np.swapaxes(w_hat, -1, -2)).astype(np.float32)


def nvfp4_linear(x: Array, weight: Array, *, block: int = TOY_NVFP4_BLOCK) -> Array:
    """``y = x_hat @ w_hat^T`` with both sides NVFP4 E2M1 fake-quant."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D (out, in), got {weight.shape}")
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"contracting dim mismatch: x[..., {x.shape[-1]}] vs weight[..., {weight.shape[-1]}]"
        )
    x_hat = fakequant_nvfp4(x, block=block)
    w_hat = fakequant_nvfp4(weight, block=block)
    return np.matmul(x_hat, np.swapaxes(w_hat, -1, -2)).astype(np.float32)


def bf16_linear(x: Array, weight: Array) -> Array:
    """BF16 matmul ``x @ w^T`` with both sides truncated to BF16."""
    x_b = to_bf16(x)
    w_b = to_bf16(weight)
    return np.matmul(x_b, np.swapaxes(w_b, -1, -2)).astype(np.float32)


def _linear(x: Array, weight: Array, mode: str) -> Array:
    if mode == "fp8":
        return fp8_linear(x, weight, block=TOY_FP8_BLOCK)
    if mode == "nvfp4":
        return nvfp4_linear(x, weight, block=TOY_NVFP4_BLOCK)
    if mode == "bf16":
        return bf16_linear(x, weight)
    raise ValueError(f"unknown linear mode {mode!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def fp8_loss_rel_diff(
    fp8_losses: Any, bf16_losses: Any, *, floor: float = REL_FLOOR
) -> float:
    """``max_t |L_fp8 - L_bf16| / max(|L_bf16|, floor)``. Length-1 scalars OK."""
    a = np.asarray(fp8_losses, dtype=np.float64).reshape(-1)
    b = np.asarray(bf16_losses, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"loss trajectory length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(b), float(floor))
    rel = np.abs(a - b) / denom
    return float(np.max(rel))


def meets_fp8_gate(diff: float, *, limit: float = FP8_REL_MAX) -> bool:
    """True iff ``diff`` is finite, ``>= 0``, and ``<= limit``."""
    d = float(diff)
    return bool(np.isfinite(d) and d >= 0.0 and d <= float(limit))


def nvfp4_numerics_ok(*arrays: Any, ran_nvfp4: bool) -> bool:
    """True iff the NVFP4 path ran and every array is non-empty and finite."""
    if not ran_nvfp4:
        return False
    if not arrays:
        return False
    for item in arrays:
        x = np.asarray(item)
        if x.size == 0:
            return False
        if not np.isfinite(x).all():
            return False
    return True


def perturb_fp8_losses(
    bf16_losses: Any, *, rel_gap: float = GOLDEN_GAP_ABOVE_GATE
) -> Array:
    """Build an FP8 trajectory with a uniform relative gap well above 0.5%."""
    b = np.asarray(bf16_losses, dtype=np.float64)
    return (b + float(rel_gap) * np.maximum(np.abs(b), REL_FLOOR)).astype(np.float64)


def inject_nonfinite(arr: Any, *, which: str = "nan") -> Array:
    """Copy ``arr`` and plant NaN or Inf at the first element (fault injection)."""
    out = np.array(arr, dtype=np.float64, copy=True)
    if out.size == 0:
        raise ValueError("cannot inject into an empty array")
    out.reshape(-1)[0] = np.nan if which == "nan" else np.inf
    return out


# ---------------------------------------------------------------------------
# Toy net (spec 5.3 roles, tiny width)
# ---------------------------------------------------------------------------


def _silu(x: Array) -> Array:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, np.float32(-80.0), np.float32(80.0))
    return (x * (np.float32(1.0) / (np.float32(1.0) + np.exp(-x)))).astype(np.float32)


def _rms_norm(x: Array, scale: Array, *, eps: float = 1e-6) -> Array:
    x = np.asarray(x, dtype=np.float32)
    scale_b = to_bf16(scale)
    mean_sq = np.mean(x * x, axis=-1, keepdims=True)
    y = x / np.sqrt(mean_sq + np.float32(eps))
    return to_bf16(y * scale_b)


def _softmax_last(x: Array) -> Array:
    """Softmax stays BF16: cast, then FP32 softmax, then BF16."""
    z = to_bf16(x).astype(np.float32)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(np.clip(z, np.float32(-80.0), np.float32(80.0)))
    p = e / np.sum(e, axis=-1, keepdims=True)
    return to_bf16(p)


def mean_cross_entropy(logits: Array, targets: Array) -> float:
    """Mean token CE. ``logits`` (..., V), ``targets`` (...) int ids in ``[0, V)``."""
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets)
    if z.ndim < 1:
        raise ValueError("logits must have a vocab axis")
    if z.shape[:-1] != t.shape:
        raise ValueError(f"CE shape mismatch: logits {z.shape} vs targets {t.shape}")
    if z.size == 0:
        return 0.0
    vocab = int(z.shape[-1])
    flat = z.reshape(-1, vocab)
    idx = t.reshape(-1).astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= vocab:
        raise ValueError("target id out of vocab")
    m = np.max(flat, axis=-1, keepdims=True)
    log_z = np.log(np.exp(flat - m).sum(axis=-1)) + m.reshape(-1)
    nll = log_z - flat[np.arange(flat.shape[0]), idx]
    return float(np.mean(nll))


def ce_dlogits(logits: Array, targets: Array) -> Array:
    """``d mean_CE / d logits``, FP32, same shape as ``logits``."""
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets)
    vocab = int(z.shape[-1])
    flat = z.reshape(-1, vocab)
    m = np.max(flat, axis=-1, keepdims=True)
    exp = np.exp(flat - m)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    idx = t.reshape(-1).astype(np.int64)
    probs[np.arange(probs.shape[0]), idx] -= 1.0
    probs /= float(probs.shape[0])
    return probs.reshape(z.shape).astype(np.float32)


def toy_init(seed: int = TOY_SEED) -> dict[str, Array]:
    """FP32 master weights. N(0, 0.02) except RMSNorm scales = 1."""
    rng = np.random.default_rng(int(seed))

    def w(shape: tuple[int, ...]) -> Array:
        return rng.normal(0.0, 0.02, size=shape).astype(np.float32)

    return {
        "embed": w((TOY_VOCAB, TOY_DIM)),
        "latent_w": w((TOY_DIM, TOY_DIM)),
        "ln1": np.ones((TOY_DIM,), dtype=np.float32),
        "w_in": w((TOY_FFN, TOY_DIM)),
        "w_out": w((TOY_DIM, TOY_FFN)),
        "router": w((TOY_EXPERTS, TOY_DIM)),
        "expert_w_in": w((TOY_EXPERTS, TOY_FFN, TOY_DIM)),
        "expert_w_out": w((TOY_EXPERTS, TOY_DIM, TOY_FFN)),
        "ln2": np.ones((TOY_DIM,), dtype=np.float32),
        "w_last": w((TOY_DIM, TOY_DIM)),
        "ln3": np.ones((TOY_DIM,), dtype=np.float32),
        "w_last2": w((TOY_DIM, TOY_DIM)),
        "unembed": w((TOY_VOCAB, TOY_DIM)),
    }


def toy_tokens(seed: int = TOY_SEED) -> Array:
    """``(TOY_BATCH, TOY_SEQ)`` int32 ids in ``[0, TOY_VOCAB)``."""
    rng = np.random.default_rng(int(seed) + 1)
    return rng.integers(0, TOY_VOCAB, size=(TOY_BATCH, TOY_SEQ), dtype=np.int32)


def clone_params(params: Mapping[str, Array]) -> dict[str, Array]:
    return {key: np.array(val, copy=True) for key, val in params.items()}


def toy_forward(
    params: Mapping[str, Array], tokens: Array, precision: str
) -> tuple[float, dict[str, Array]]:
    """One forward. ``precision`` is ``bf16``, ``fp8``, or ``nvfp4``.

    Body linears (not last two) are FP8 on the FP8 and NVFP4 paths. Routed
    expert weights are NVFP4 only on the NVFP4 path. Everything listed in
    spec 5.3 as BF16 stays BF16 on every path.
    """
    if precision not in ("bf16", "fp8", "nvfp4"):
        raise ValueError(f"unknown precision {precision!r}")
    tok = np.asarray(tokens)
    body_linear = "fp8" if precision in ("fp8", "nvfp4") else "bf16"
    expert_linear = "nvfp4" if precision == "nvfp4" else body_linear

    h = to_bf16(params["embed"][tok])
    h = to_bf16(bf16_linear(h, params["latent_w"]))  # latent adapter: BF16

    h = _rms_norm(h, params["ln1"])
    mlp_mid = _silu(_linear(h, params["w_in"], body_linear))
    mlp = _linear(mlp_mid, params["w_out"], body_linear)

    gates = _softmax_last(bf16_linear(h, params["router"]))  # router + softmax: BF16
    expert_out = np.zeros_like(h, dtype=np.float32)
    expert_acts: list[Array] = []
    for expert in range(TOY_EXPERTS):
        mid = _silu(_linear(h, params["expert_w_in"][expert], expert_linear))
        ye = _linear(mid, params["expert_w_out"][expert], expert_linear)
        expert_acts.append(ye)
        expert_out = expert_out + gates[..., expert : expert + 1] * ye
    h = to_bf16(h + mlp + expert_out)

    # Last two layers stay BF16 even in the FP8 / NVFP4 runs.
    h = _rms_norm(h, params["ln2"])
    h = to_bf16(h + bf16_linear(h, params["w_last"]))
    h = _rms_norm(h, params["ln3"])
    h = to_bf16(h + bf16_linear(h, params["w_last2"]))

    logits = to_bf16(bf16_linear(h, params["unembed"]))  # embeddings: BF16
    loss = mean_cross_entropy(logits, tok)
    activations = {
        "h": h.astype(np.float32),
        "logits": logits.astype(np.float32),
        "mlp": mlp.astype(np.float32),
        "expert_out": expert_out.astype(np.float32),
        "gates": gates.astype(np.float32),
        "expert_acts": np.stack(expert_acts, axis=0).astype(np.float32),
    }
    return loss, activations


def toy_sgd_step(
    params: Mapping[str, Array],
    tokens: Array,
    precision: str,
    *,
    lr: float = TOY_LR,
) -> tuple[dict[str, Array], float, dict[str, Array]]:
    """One SGD step on the FP32 unembed (always BF16 in forward; STE)."""
    loss, acts = toy_forward(params, tokens, precision)
    g_logits = ce_dlogits(acts["logits"], tokens)
    h_b = to_bf16(acts["h"])
    g_unembed = np.einsum("bsv,bsd->vd", g_logits, h_b).astype(np.float32)
    out = clone_params(params)
    out["unembed"] = (params["unembed"] - np.float32(lr) * g_unembed).astype(np.float32)
    return out, loss, acts


def train_trajectory(
    params: Mapping[str, Array],
    tokens: Array,
    *,
    steps: int,
    precision: str,
    lr: float = TOY_LR,
) -> tuple[Array, dict[str, Array]]:
    """``steps`` SGD steps. Returns ``(losses [steps], last activations)``."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    p = clone_params(params)
    losses: list[float] = []
    acts: dict[str, Array] | None = None
    for _ in range(int(steps)):
        p, loss, acts = toy_sgd_step(p, tokens, precision, lr=lr)
        losses.append(loss)
    assert acts is not None
    return np.asarray(losses, dtype=np.float64), acts


def evaluate_toy_protocol(
    *, steps: int = 4, seed: int = TOY_SEED
) -> dict[str, Any]:
    """Independent BF16, FP8, and NVFP4 trajectories from the same FP32 master.

    Three **distinct** paths (cloned masters). NVFP4 is always run, never skipped.
    """
    params = toy_init(seed)
    tokens = toy_tokens(seed)
    bf16_losses, _ = train_trajectory(params, tokens, steps=steps, precision="bf16")
    fp8_losses, _ = train_trajectory(params, tokens, steps=steps, precision="fp8")
    nvfp4_losses, nvfp4_acts = train_trajectory(
        params, tokens, steps=steps, precision="nvfp4"
    )
    rel = fp8_loss_rel_diff(fp8_losses, bf16_losses)
    ok = nvfp4_numerics_ok(
        nvfp4_losses,
        nvfp4_acts["h"],
        nvfp4_acts["logits"],
        nvfp4_acts["mlp"],
        nvfp4_acts["expert_out"],
        nvfp4_acts["expert_acts"],
        ran_nvfp4=True,
    )
    return {
        "bf16_losses": bf16_losses,
        "fp8_losses": fp8_losses,
        "nvfp4_losses": nvfp4_losses,
        "fp8_loss_rel_diff": rel,
        "nvfp4_numerics_ok": ok,
        "ran_nvfp4": True,
        "tokens": tokens,
        "w_in": params["w_in"],
        "expert_w_in": params["expert_w_in"],
        "activations": nvfp4_acts,
        "w_in_fp8": fakequant_fp8(params["w_in"], block=TOY_FP8_BLOCK),
        "w_in_fp8_scale": fp8_block_scales(params["w_in"], block=TOY_FP8_BLOCK),
        "expert_w_in_nvfp4": fakequant_nvfp4(
            params["expert_w_in"][0], block=TOY_NVFP4_BLOCK
        ),
        "expert_w_in_nvfp4_scale": nvfp4_block_scales(
            params["expert_w_in"][0], block=TOY_NVFP4_BLOCK
        ),
    }


# ---------------------------------------------------------------------------
# Differentiable linear MSE (finite-difference checks)
# ---------------------------------------------------------------------------


def linear_mse(x: Array, weight: Array, target: Array) -> float:
    """Mean squared error of ``x @ w^T`` vs ``target``. FP32 master, no quant."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    pred = np.matmul(x, np.swapaxes(weight, -1, -2))
    err = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.mean(err * err))


def linear_mse_grad_w(x: Array, weight: Array, target: Array) -> Array:
    """Analytic ``d linear_mse / d weight``. ``weight`` is ``(out, in)``."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    pred = np.matmul(x, np.swapaxes(weight, -1, -2))
    err = pred - target
    n = float(err.size)
    # dL/dW = (2/n) * err^T @ x
    grad = (np.float32(2.0) / np.float32(n)) * np.matmul(
        np.swapaxes(err, -1, -2), x
    )
    return np.asarray(grad, dtype=np.float32)


def linear_mse_grad_w_finite_diff(
    x: Array,
    weight: Array,
    target: Array,
    *,
    eps: float = TOY_FD_EPS,
    sl: tuple[slice, slice] = (slice(0, 2), slice(0, 2)),
) -> tuple[Array, Array]:
    """Central differences vs analytic on a small slice of ``weight``."""
    analytic = linear_mse_grad_w(x, weight, target)[sl]
    numeric = np.zeros_like(analytic, dtype=np.float32)
    w = np.array(weight, dtype=np.float32, copy=True)
    base = w[sl]
    delta = np.float32(eps)
    for index in np.ndindex(base.shape):
        full = (sl[0], sl[1])
        # Translate local index into the slice. Slices here start at 0.
        i0 = (full[0].start or 0) + index[0]
        i1 = (full[1].start or 0) + index[1]
        orig = w[i0, i1]
        w[i0, i1] = orig + delta
        plus = linear_mse(x, w, target)
        w[i0, i1] = orig - delta
        minus = linear_mse(x, w, target)
        w[i0, i1] = orig
        numeric[index] = np.float32((plus - minus) / (2.0 * float(eps)))
    return analytic.astype(np.float32), numeric.astype(np.float32)
