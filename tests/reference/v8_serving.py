"""Independent V8 serving protocol (spec 16.2 / 13).

Slow and obvious host NumPy. Does **not** import JAX, torch, the V8 serving
runner, ``model/``, ``train/``, ``kernels/``, or the Rust crates. Production
``run_v8`` must not import this module; tests import both.

CPU analog: tiny checkpoint, host stand-in for SGLang. Passing these tests is
**not** V8 verified. Spec V8 is 1 to 4 H200.

``logprob_within_threshold``
    SGLang-analog vs JAX-analog log-probs within ``LOGPROB_THRESHOLD`` (1e-5).
    Must not be True because a run was compared to itself.

``tiered_restore_matches``
    Session KV swapped HBM → host RAM (Grace stand-in) → NVMe (bytes stand-in)
    and restored matches an unswapped twin. Must not be True because restore
    was skipped or the swapped session was compared to itself.

Glue of C2 serving. This file is an independent Python stand-in of that
protocol, not a second kernel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray

# Analog constants the production CPU stand-in must match (spec 16.2 V8).
DEFAULT_GPUS = 4
DEFAULT_GPUS_MIN = 1
DEFAULT_GPUS_MAX = 4
LOGPROB_THRESHOLD = 1e-5  # FP32 parity (same scale as V1 logits)

RECURRENCE_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)
RECURRENCE_MAX = 16
MTP_HEADS = 2
LATENT_CHUNK_MIN = 4
LATENT_CHUNK_MAX = 64

TOY_VOCAB = 8
TOY_DIM = 4
TOY_BATCH = 2
TOY_SEQ = 4
TOY_PREFIX = 2
TOY_SEED = 8
TOY_R = 3  # buckets to 4
TOY_BUDGET = 16
TOY_WEIGHT_SCALE = 0.02

TOY_FD_EPS = 1e-3
TOY_GRAD_RTOL = 5e-2
TOY_GRAD_ATOL = 5e-3

# Uniform log-prob / logit shift that must fail the 1e-5 gate.
GOLDEN_MATCH_DIFF = 0.0
GOLDEN_SHIFT_ABOVE_GATE = 2e-5

KV_TIER_HBM = "hbm"
KV_TIER_GRACE = "grace"  # host RAM stand-in
KV_TIER_NVME = "nvme"
KV_TIER_ORDER: tuple[str, ...] = (KV_TIER_HBM, KV_TIER_GRACE, KV_TIER_NVME)

V8_RESULT_KEYS: tuple[str, ...] = (
    "logprob_within_threshold",
    "tiered_restore_matches",
)

_NEG_INF = np.float32(-1.0e9)
_SCALE = np.float32(1.0 / np.sqrt(np.float32(TOY_DIM)))


@dataclass
class TinyCkpt:
    """Tiny JAX-checkpoint stand-in (embedding + recurrent core + unembed)."""

    embed: Array  # (vocab, dim)
    w_core: Array  # (dim, dim)
    unembed: Array  # (dim, vocab)
    w_adapter1: Array  # (dim, dim)
    w_adapter2: Array  # (dim, dim)
    w_mtp: Array  # (mtp_heads, dim, vocab)


def bucket_r(r: int, budget: int = RECURRENCE_MAX) -> int:
    """Ceil ``r`` to the next recurrence bucket, then cap by budget / max."""
    if int(r) <= 0 or int(budget) <= 0:
        raise ValueError("r and budget must be positive")
    cap = min(int(budget), RECURRENCE_MAX)
    bucketed = RECURRENCE_MAX
    for item in RECURRENCE_BUCKETS:
        if item >= int(r):
            bucketed = item
            break
    return min(bucketed, cap)


def toy_ckpt(seed: int = TOY_SEED) -> TinyCkpt:
    """Deterministic tiny checkpoint (float32)."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    scale = np.float32(TOY_WEIGHT_SCALE)

    def _mat(*shape: int) -> Array:
        return (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

    return TinyCkpt(
        embed=_mat(TOY_VOCAB, TOY_DIM),
        w_core=_mat(TOY_DIM, TOY_DIM),
        unembed=_mat(TOY_DIM, TOY_VOCAB),
        w_adapter1=_mat(TOY_DIM, TOY_DIM),
        w_adapter2=_mat(TOY_DIM, TOY_DIM),
        w_mtp=_mat(MTP_HEADS, TOY_DIM, TOY_VOCAB),
    )


def toy_tokens(seed: int = TOY_SEED) -> Array:
    """Deterministic token ids in ``[0, TOY_VOCAB)``."""
    rng = np.random.Generator(np.random.PCG64(int(seed) + 1))
    return rng.integers(0, TOY_VOCAB, size=(TOY_BATCH, TOY_SEQ), dtype=np.int32)


def log_softmax(logits: Array, axis: int = -1) -> Array:
    """Stable log-softmax in float64."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def logprob_max_abs_diff(a: Array, b: Array) -> float:
    """Max ``|log p - log q|`` over two logit tensors, compared as float32 inputs.

    Empty arrays yield 0.0. Shape mismatch raises. Non-finite inputs yield a
    non-finite result (the V8 gate requires a finite value).
    """
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"logit shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    diff = np.abs(log_softmax(x) - log_softmax(y))
    return float(np.max(diff))


def meets_logprob_gate(diff: float, *, limit: float = LOGPROB_THRESHOLD) -> bool:
    """True iff ``diff`` is finite and ``diff <= limit``."""
    d = float(diff)
    return bool(np.isfinite(d) and d <= float(limit))


def logprobs_compared_ok(
    diff: float,
    *,
    distinct_backends: bool,
    same_object: bool,
    limit: float = LOGPROB_THRESHOLD,
) -> bool:
    """Numeric gate plus the 'not a self-compare' path check."""
    return (
        bool(distinct_backends)
        and (not bool(same_object))
        and meets_logprob_gate(diff, limit=limit)
    )


def meets_v8_gates(result: Mapping[str, Any]) -> bool:
    """F4 ``check_exit(V8)``: both bools true."""
    return bool(result["logprob_within_threshold"]) and bool(
        result["tiered_restore_matches"]
    )


def _recur(x: Array, w_core: Array, r: int) -> Array:
    """Input-injected recurrence; KV is shared across iterations (spec 13.3)."""
    h = np.asarray(x, dtype=np.float32)
    core = np.asarray(w_core, dtype=np.float32)
    inject = h
    steps = int(r)
    for _ in range(steps):
        h = h @ core + inject
    return h.astype(np.float32)


def latent_adapter(h: Array, ckpt: TinyCkpt) -> Array:
    """Latent decode adapter: ReLU MLP instead of a token embedding (spec 13.2)."""
    z = np.asarray(h, dtype=np.float32) @ ckpt.w_adapter1
    z = np.maximum(z, np.float32(0.0))
    return (z @ ckpt.w_adapter2).astype(np.float32)


def mtp_logits(ctx: Array, ckpt: TinyCkpt) -> Array:
    """Two MTP heads on attention context: ``(heads, batch, seq, vocab)``."""
    return np.einsum("bsd,hdv->hbsv", ctx.astype(np.float32), ckpt.w_mtp).astype(
        np.float32
    )


def _softmax_last(scores: Array) -> Array:
    x = np.asarray(scores, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


def jax_verbal_forward(
    ckpt: TinyCkpt, tokens: Array, r: int
) -> tuple[Array, Array, Array]:
    """Batched JAX-analog: embed → recur → causal attention → unembed.

    Distinct from the SGLang token loop: one einsum + tril mask over the
    whole sequence.
    """
    ids = np.asarray(tokens, dtype=np.int32)
    r_eff = bucket_r(int(r), TOY_BUDGET)
    x = ckpt.embed[ids].astype(np.float32)
    h = _recur(x, ckpt.w_core, r_eff)
    batch, seq, dim = h.shape
    scores = np.einsum("bqd,bkd->bqk", h, h).astype(np.float32) * _SCALE
    q_idx = np.arange(seq, dtype=np.int32)[:, None]
    k_idx = np.arange(seq, dtype=np.int32)[None, :]
    causal = k_idx <= q_idx
    scores = np.where(causal[None, :, :], scores, _NEG_INF)
    attn = _softmax_last(scores)
    ctx = np.einsum("bqk,bkd->bqd", attn, h).astype(np.float32)
    logits = (ctx @ ckpt.unembed).astype(np.float32)
    assert logits.shape == (batch, seq, ckpt.unembed.shape[1])
    assert h.shape[-1] == dim
    return logits, h, ctx


def sglang_verbal_forward(
    ckpt: TinyCkpt,
    tokens: Array,
    r: int,
    kv_h: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Sequential SGLang-analog: per-token recur, append KV, attend to cache.

    ``kv_h`` is optional prefix cache ``(batch, t_past, dim)``.
    Returns ``(logits, h_new, ctx_new, kv_h_out)``.
    """
    ids = np.asarray(tokens, dtype=np.int32)
    r_eff = bucket_r(int(r), TOY_BUDGET)
    batch, seq = ids.shape
    dim = int(ckpt.embed.shape[1])
    vocab = int(ckpt.unembed.shape[1])
    past = (
        np.zeros((batch, 0, dim), dtype=np.float32)
        if kv_h is None
        else np.asarray(kv_h, dtype=np.float32)
    )
    logits = np.empty((batch, seq, vocab), dtype=np.float32)
    h_new = np.empty((batch, seq, dim), dtype=np.float32)
    ctx_new = np.empty((batch, seq, dim), dtype=np.float32)
    kv_out = np.empty((batch, past.shape[1] + seq, dim), dtype=np.float32)
    kv_out[:, : past.shape[1], :] = past
    for b in range(batch):
        for t in range(seq):
            xt = ckpt.embed[int(ids[b, t])]
            ht = _recur(xt, ckpt.w_core, r_eff)
            h_new[b, t] = ht
            pos = past.shape[1] + t
            kv_out[b, pos] = ht
            cache = kv_out[b, : pos + 1]
            scores = (cache @ ht) * _SCALE
            weights = _softmax_last(scores)
            ctx = weights @ cache
            ctx_new[b, t] = ctx
            logits[b, t] = ctx @ ckpt.unembed
    return logits, h_new, ctx_new, kv_out


def latent_followup_logits(ckpt: TinyCkpt, kv_h: Array, ctx_last: Array, r: int) -> Array:
    """One latent step: adapter(ctx) instead of embed(token), then attend."""
    r_eff = bucket_r(int(r), TOY_BUDGET)
    x_lat = latent_adapter(ctx_last, ckpt)
    h_lat = _recur(x_lat, ckpt.w_core, r_eff)
    kv = np.concatenate([kv_h.astype(np.float32), h_lat[:, None, :]], axis=1)
    scores = np.einsum("bd,bkd->bk", h_lat, kv).astype(np.float32) * _SCALE
    weights = _softmax_last(scores)
    ctx = np.einsum("bk,bkd->bd", weights, kv).astype(np.float32)
    return (ctx @ ckpt.unembed).astype(np.float32)


def _pack_logprobs(main: Array, latent: Array, mtp: Array) -> Array:
    return np.concatenate(
        [
            log_softmax(main).ravel(),
            log_softmax(latent).ravel(),
            log_softmax(mtp).ravel(),
        ]
    )


def kv_to_nvme(kv: Array) -> tuple[bytes, tuple[int, ...]]:
    """NVMe stand-in: raw float32 bytes plus shape. Does not write ``v8.json``."""
    x = np.ascontiguousarray(np.asarray(kv, dtype=np.float32))
    return x.tobytes(), tuple(int(s) for s in x.shape)


def kv_from_nvme(blob: bytes, shape: tuple[int, ...]) -> Array:
    """Restore KV from the NVMe bytes stand-in."""
    return np.frombuffer(blob, dtype=np.float32).reshape(shape).copy()


def next_kv_tier(tier: str) -> str | None:
    """HBM → Grace (host RAM) → NVMe → None."""
    for index, current in enumerate(KV_TIER_ORDER):
        if current == tier:
            if index + 1 < len(KV_TIER_ORDER):
                return KV_TIER_ORDER[index + 1]
            return None
    raise ValueError(f"unknown kv tier {tier!r}")


def run_logprob_experiment(
    *,
    self_compare: bool = False,
    shift_sglang: float = 0.0,
    r: int = TOY_R,
) -> dict[str, Any]:
    """SGLang-analog vs JAX-analog log-probs on the tiny checkpoint.

    ``self_compare=True`` runs the JAX path twice (forbidden gate). A
    log-prob shift on the SGLang pack must fail the 1e-5 threshold.
    """
    ckpt = toy_ckpt()
    tokens = toy_tokens()
    r_eff = bucket_r(int(r), TOY_BUDGET)

    jax_logits, jax_h, jax_ctx = jax_verbal_forward(ckpt, tokens, r)
    jax_latent = latent_followup_logits(ckpt, jax_h, jax_ctx[:, -1, :], r)
    jax_mtp = mtp_logits(jax_ctx, ckpt)

    if self_compare:
        sgl_logits, sgl_h, sgl_ctx, _kv = (
            jax_logits.copy(),
            jax_h.copy(),
            jax_ctx.copy(),
            None,
        )
        sgl_latent = jax_latent.copy()
        sgl_mtp = jax_mtp.copy()
        backend_b = "jax"
        distinct = False
    else:
        sgl_logits, sgl_h, sgl_ctx, _kv = sglang_verbal_forward(ckpt, tokens, r)
        sgl_latent = latent_followup_logits(ckpt, sgl_h, sgl_ctx[:, -1, :], r)
        sgl_mtp = mtp_logits(sgl_ctx, ckpt)
        backend_b = "sglang"
        distinct = True

    jax_pack = _pack_logprobs(jax_logits, jax_latent, jax_mtp)
    sgl_pack = _pack_logprobs(sgl_logits, sgl_latent, sgl_mtp)
    # Uniform logit shifts are softmax-invariant; inject on the log-probs
    # themselves so the 1e-5 gate can fail (not a self-compare).
    if float(shift_sglang) != 0.0:
        sgl_pack = np.asarray(sgl_pack, dtype=np.float64) + float(shift_sglang)
    same_object = jax_pack is sgl_pack
    main_diff = logprob_max_abs_diff(jax_logits, sgl_logits)
    pack_diff = float(np.max(np.abs(jax_pack - sgl_pack))) if jax_pack.size else 0.0
    diff = max(main_diff, pack_diff)
    within = logprobs_compared_ok(
        diff, distinct_backends=distinct, same_object=same_object
    )
    return {
        "logprob_within_threshold": within,
        "distinct_backends": distinct,
        "self_compared": bool(self_compare),
        "same_object": same_object,
        "backend_a": "jax",
        "backend_b": backend_b,
        "diff": diff,
        "main_diff": main_diff,
        "r_requested": int(r),
        "r_bucketed": r_eff,
        "mtp_heads": int(ckpt.w_mtp.shape[0]),
        "used_latent_decode": True,
        "used_mtp": True,
        "jax_logits_shape": tuple(int(x) for x in jax_logits.shape),
        "sgl_logits_shape": tuple(int(x) for x in sgl_logits.shape),
        "jax_logits_dtype": str(jax_logits.dtype),
        "sgl_logits_dtype": str(sgl_logits.dtype),
        "tokens_shape": tuple(int(x) for x in tokens.shape),
        "tokens_dtype": str(tokens.dtype),
        "shift_sglang": float(shift_sglang),
        "threshold": float(LOGPROB_THRESHOLD),
        "jax_pack": jax_pack,
        "sgl_pack": sgl_pack,
    }


def run_tiered_restore_experiment(
    *,
    do_restore: bool = True,
    restore_corrupt: bool = False,
    skip_swap: bool = False,
    skip_nvme: bool = False,
    compare_to_self: bool = False,
    r: int = TOY_R,
) -> dict[str, Any]:
    """Swap session KV through host RAM and NVMe; restore vs unswapped twin.

    The unswapped twin keeps KV on HBM and decodes the suffix without a
    round-trip. The swapped path must actually restore and must be compared
    to that twin, not to itself.
    """
    ckpt = toy_ckpt()
    tokens = toy_tokens()
    prefix = tokens[:, :TOY_PREFIX]
    suffix = tokens[:, TOY_PREFIX:]

    twin_logits, _twin_h, _twin_ctx, _twin_kv = sglang_verbal_forward(ckpt, tokens, r)
    twin_suffix = twin_logits[:, TOY_PREFIX:, :]

    _pref_logits, _pref_h, _pref_ctx, kv = sglang_verbal_forward(ckpt, prefix, r)
    did_swap = False
    went_through_grace = False
    went_through_nvme = False
    did_restore = False
    tier = KV_TIER_HBM

    if not skip_swap:
        did_swap = True
        kv = kv.copy()
        tier = KV_TIER_GRACE
        went_through_grace = True
        if not skip_nvme:
            blob, shape = kv_to_nvme(kv)
            kv = None
            tier = KV_TIER_NVME
            went_through_nvme = True
            if do_restore:
                kv = kv_from_nvme(blob, shape)
                if restore_corrupt:
                    # LSB flips are too small for float32 attention; shift the
                    # restored cache so the suffix actually diverges.
                    kv = (kv + np.float32(1.0)).astype(np.float32)
                tier = KV_TIER_HBM
                did_restore = True
        elif do_restore:
            # Grace-only restore skips NVMe: not a full tiered round-trip.
            kv = kv.copy()
            tier = KV_TIER_HBM
            did_restore = True

    restored_suffix: Array | None
    if did_restore and kv is not None:
        restored_suffix, _h, _c, _k = sglang_verbal_forward(ckpt, suffix, r, kv_h=kv)
    else:
        restored_suffix = None

    if compare_to_self:
        compared_to_twin = False
        if restored_suffix is None:
            raw_equal = False
            restore_diff = float("inf")
        else:
            restore_diff = logprob_max_abs_diff(restored_suffix, restored_suffix)
            raw_equal = meets_logprob_gate(restore_diff)
    else:
        compared_to_twin = True
        if restored_suffix is None:
            raw_equal = False
            restore_diff = float("inf")
        else:
            restore_diff = logprob_max_abs_diff(restored_suffix, twin_suffix)
            raw_equal = meets_logprob_gate(restore_diff)

    matches = (
        bool(did_swap)
        and bool(went_through_grace)
        and bool(went_through_nvme)
        and bool(did_restore)
        and (not bool(restore_corrupt))
        and bool(compared_to_twin)
        and bool(raw_equal)
    )
    return {
        "tiered_restore_matches": matches,
        "did_swap": did_swap,
        "went_through_grace": went_through_grace,
        "went_through_nvme": went_through_nvme,
        "did_restore": did_restore,
        "restore_corrupt": bool(restore_corrupt),
        "skip_swap": bool(skip_swap),
        "skip_nvme": bool(skip_nvme),
        "compared_to_twin": compared_to_twin,
        "compared_to_self": bool(compare_to_self),
        "raw_equal": raw_equal,
        "restore_diff": restore_diff,
        "final_tier": tier,
        "prefix_len": int(TOY_PREFIX),
        "suffix_len": int(TOY_SEQ - TOY_PREFIX),
        "twin_suffix_shape": tuple(int(x) for x in twin_suffix.shape),
        "restored_suffix_shape": (
            None
            if restored_suffix is None
            else tuple(int(x) for x in restored_suffix.shape)
        ),
        "kv_tiers": KV_TIER_ORDER,
    }


def _mse_loss(x: Array, w: Array, y: Array) -> float:
    pred = x @ w
    err = pred - y
    return float(np.mean(err * err))


def toy_grad_ok(
    *,
    eps: float = TOY_FD_EPS,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    """Linear MSE: analytic ``dL/dW`` vs central finite differences."""
    rng = np.random.Generator(np.random.PCG64(TOY_SEED + 2))
    x = rng.standard_normal((TOY_BATCH, TOY_DIM)).astype(np.float64)
    w = rng.standard_normal((TOY_DIM, TOY_VOCAB)).astype(np.float64)
    y = rng.standard_normal((TOY_BATCH, TOY_VOCAB)).astype(np.float64)
    pred = x @ w
    err = pred - y
    n = float(err.size)
    analytic = (2.0 / n) * (x.T @ err)
    fd = np.empty_like(w)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            up = w.copy()
            dn = w.copy()
            up[i, j] += eps
            dn[i, j] -= eps
            fd[i, j] = (_mse_loss(x, up, y) - _mse_loss(x, dn, y)) / (2.0 * eps)
    return bool(np.allclose(analytic, fd, rtol=rtol, atol=atol))


def evaluate_v8_protocol(
    *,
    self_compare: bool = False,
    shift_sglang: float = 0.0,
    do_restore: bool = True,
    restore_corrupt: bool = False,
    skip_swap: bool = False,
    skip_nvme: bool = False,
    compare_restore_to_self: bool = False,
    r: int = TOY_R,
) -> dict[str, Any]:
    """Run both V8 analog experiments and report the F4 gates."""
    logprob = run_logprob_experiment(
        self_compare=self_compare, shift_sglang=shift_sglang, r=r
    )
    restore = run_tiered_restore_experiment(
        do_restore=do_restore,
        restore_corrupt=restore_corrupt,
        skip_swap=skip_swap,
        skip_nvme=skip_nvme,
        compare_to_self=compare_restore_to_self,
        r=r,
    )
    result: dict[str, Any] = {
        "logprob_within_threshold": bool(logprob["logprob_within_threshold"]),
        "tiered_restore_matches": bool(restore["tiered_restore_matches"]),
        "logprob": logprob,
        "restore": restore,
    }
    result["meets_gates"] = meets_v8_gates(result)
    result["grad_ok"] = toy_grad_ok()
    return result
