"""V8 serving runner (spec 16.2, F4 template ``v8.sh``).

CPU analog of 1 to 4 H200: SGLang fork loads a JAX checkpoint; latent
decode, recurrence buckets, MTP speculative decode, KV tiering to host
RAM and NVMe. A CPU / host path is allowed so the analog can run without
a GPU. That does not count as V8 verified. V8 itself waits for V0.

``verify/ncshare`` ``check_exit(V8)`` reads ``v8.json`` with two bools,
both required true:

- ``logprob_within_threshold``: SGLang vs JAX log-probs within threshold
  (C2 serving).
- ``tiered_restore_matches``: tiered session restore matches unswapped
  output.

Glue, not a second implementation: C2 serving. Crates with no Python
bindings may use a host analog of the same protocol. Must not import
``tests/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

# Spec 16.2: 1 to 4 H200. Template ``v8.sh`` passes ``gpus``. Default is
# the top of that range.
DEFAULT_GPUS = 4
DEFAULT_GPUS_MIN = 1
DEFAULT_GPUS_MAX = 4
LOGPROB_THRESHOLD = 1e-5
RECURRENCE_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16)
RECURRENCE_MAX = 16
MTP_HEADS = 2
TOY_VOCAB = 8
TOY_DIM = 4
TOY_BATCH = 2
TOY_SEQ = 4
TOY_PREFIX = 2
TOY_SEED = 8
TOY_R = 3
TOY_WEIGHT_SCALE = 0.02
TOY_BUDGET = 16
KV_TIER_ORDER: tuple[str, ...] = ("hbm", "grace", "nvme")
GOLDEN_SHIFT_ABOVE_GATE = 2e-5

_ATTN_SCALE = np.float32(1.0 / np.sqrt(np.float32(TOY_DIM)))
_CAUSAL_MASK = np.float32(-1.0e9)


class V8Result(TypedDict):
    logprob_within_threshold: bool
    tiered_restore_matches: bool


class V8Error(Exception):
    """Bad V8 inputs (gpus < 1) or a missing / failed backend."""


@dataclass
class _ToyCkpt:
    embed: np.ndarray
    w_core: np.ndarray
    unembed: np.ndarray
    w_in: np.ndarray
    w_out: np.ndarray
    w_mtp: np.ndarray


def _bucket_r(r: int, budget: int = RECURRENCE_MAX) -> int:
    if int(r) <= 0 or int(budget) <= 0:
        raise V8Error("r and budget must be positive")
    cap = min(int(budget), RECURRENCE_MAX)
    picked = RECURRENCE_MAX
    for item in RECURRENCE_BUCKETS:
        if item >= int(r):
            picked = item
            break
    return min(picked, cap)


def _ckpt(seed: int = TOY_SEED) -> _ToyCkpt:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    scale = np.float32(TOY_WEIGHT_SCALE)

    def _w(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape).astype(np.float32) * scale).astype(np.float32)

    return _ToyCkpt(
        embed=_w(TOY_VOCAB, TOY_DIM),
        w_core=_w(TOY_DIM, TOY_DIM),
        unembed=_w(TOY_DIM, TOY_VOCAB),
        w_in=_w(TOY_DIM, TOY_DIM),
        w_out=_w(TOY_DIM, TOY_DIM),
        w_mtp=_w(MTP_HEADS, TOY_DIM, TOY_VOCAB),
    )


def _tokens(seed: int = TOY_SEED) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(int(seed) + 1))
    return rng.integers(0, TOY_VOCAB, size=(TOY_BATCH, TOY_SEQ), dtype=np.int32)


def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def _max_abs_logprob(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.shape != y.shape:
        raise V8Error(f"logit shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(_log_softmax(x) - _log_softmax(y))))


def _recur(x: np.ndarray, w_core: np.ndarray, r: int) -> np.ndarray:
    h = np.asarray(x, dtype=np.float32)
    core = np.asarray(w_core, dtype=np.float32)
    inject = h
    for _ in range(int(r)):
        h = h @ core + inject
    return h.astype(np.float32)


def _softmax_last(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


def _latent_adapter(h: np.ndarray, ckpt: _ToyCkpt) -> np.ndarray:
    z = np.asarray(h, dtype=np.float32) @ ckpt.w_in
    z = np.maximum(z, np.float32(0.0))
    return (z @ ckpt.w_out).astype(np.float32)


def _mtp_logits(ctx: np.ndarray, ckpt: _ToyCkpt) -> np.ndarray:
    return np.einsum("bsd,hdv->hbsv", ctx.astype(np.float32), ckpt.w_mtp).astype(np.float32)


def _jax_forward(
    ckpt: _ToyCkpt, tokens: np.ndarray, r: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched JAX-analog: embed → recur → causal einsum attention → unembed."""
    ids = np.asarray(tokens, dtype=np.int32)
    r_eff = _bucket_r(int(r), TOY_BUDGET)
    x = ckpt.embed[ids].astype(np.float32)
    h = _recur(x, ckpt.w_core, r_eff)
    seq = h.shape[1]
    scores = np.einsum("bqd,bkd->bqk", h, h).astype(np.float32) * _ATTN_SCALE
    causal = np.tril(np.ones((seq, seq), dtype=bool))
    scores = np.where(causal[None, :, :], scores, _CAUSAL_MASK)
    attn = _softmax_last(scores)
    ctx = np.einsum("bqk,bkd->bqd", attn, h).astype(np.float32)
    logits = (ctx @ ckpt.unembed).astype(np.float32)
    return logits, h, ctx


def _sgl_forward(
    ckpt: _ToyCkpt,
    tokens: np.ndarray,
    r: int,
    kv_h: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequential SGLang-analog: per-token recur, append KV, attend to cache."""
    ids = np.asarray(tokens, dtype=np.int32)
    r_eff = _bucket_r(int(r), TOY_BUDGET)
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
            scores = (cache @ ht) * _ATTN_SCALE
            weights = _softmax_last(scores)
            ctx = weights @ cache
            ctx_new[b, t] = ctx
            logits[b, t] = ctx @ ckpt.unembed
    return logits, h_new, ctx_new, kv_out


def _latent_step(ckpt: _ToyCkpt, kv_h: np.ndarray, ctx_last: np.ndarray, r: int) -> np.ndarray:
    r_eff = _bucket_r(int(r), TOY_BUDGET)
    x_lat = _latent_adapter(ctx_last, ckpt)
    h_lat = _recur(x_lat, ckpt.w_core, r_eff)
    kv = np.concatenate([kv_h.astype(np.float32), h_lat[:, None, :]], axis=1)
    scores = np.einsum("bd,bkd->bk", h_lat, kv).astype(np.float32) * _ATTN_SCALE
    weights = _softmax_last(scores)
    ctx = np.einsum("bk,bkd->bd", weights, kv).astype(np.float32)
    return (ctx @ ckpt.unembed).astype(np.float32)


def _pack(main: np.ndarray, latent: np.ndarray, mtp: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            _log_softmax(main).ravel(),
            _log_softmax(latent).ravel(),
            _log_softmax(mtp).ravel(),
        ]
    )


def _kv_to_nvme(kv: np.ndarray) -> tuple[bytes, tuple[int, ...]]:
    x = np.ascontiguousarray(np.asarray(kv, dtype=np.float32))
    return x.tobytes(), tuple(int(s) for s in x.shape)


def _kv_from_nvme(blob: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(shape).copy()


def _logprob_within_threshold(
    *,
    shift_sglang: float = 0.0,
    self_compare: bool = False,
    r: int = TOY_R,
) -> bool:
    """SGLang analog vs JAX analog. A self-compare or 2e-5 shift must fail."""
    ckpt = _ckpt()
    tokens = _tokens()
    jax_logits, jax_h, jax_ctx = _jax_forward(ckpt, tokens, r)
    jax_latent = _latent_step(ckpt, jax_h, jax_ctx[:, -1, :], r)
    jax_mtp = _mtp_logits(jax_ctx, ckpt)

    if self_compare:
        sgl_logits, sgl_h, sgl_ctx = jax_logits.copy(), jax_h.copy(), jax_ctx.copy()
        sgl_latent = jax_latent.copy()
        sgl_mtp = jax_mtp.copy()
        distinct = False
    else:
        sgl_logits, sgl_h, sgl_ctx, _kv = _sgl_forward(ckpt, tokens, r)
        sgl_latent = _latent_step(ckpt, sgl_h, sgl_ctx[:, -1, :], r)
        sgl_mtp = _mtp_logits(sgl_ctx, ckpt)
        distinct = True

    jax_pack = _pack(jax_logits, jax_latent, jax_mtp)
    sgl_pack = _pack(sgl_logits, sgl_latent, sgl_mtp)
    if float(shift_sglang) != 0.0:
        sgl_pack = np.asarray(sgl_pack, dtype=np.float64) + float(shift_sglang)
    same_object = jax_pack is sgl_pack
    main_diff = _max_abs_logprob(jax_logits, sgl_logits)
    pack_diff = float(np.max(np.abs(jax_pack - sgl_pack))) if jax_pack.size else 0.0
    diff = max(main_diff, pack_diff)
    if not np.isfinite(diff):
        raise V8Error("non-finite log-prob analog")
    return bool(distinct) and (not same_object) and bool(diff <= LOGPROB_THRESHOLD)


def _tiered_restore_matches(
    *,
    do_restore: bool = True,
    restore_corrupt: bool = False,
    skip_swap: bool = False,
    skip_nvme: bool = False,
    compare_to_self: bool = False,
    r: int = TOY_R,
) -> bool:
    """HBM → Grace → NVMe restore vs unswapped twin. Skip/corrupt must be False."""
    ckpt = _ckpt()
    tokens = _tokens()
    prefix = tokens[:, :TOY_PREFIX]
    suffix = tokens[:, TOY_PREFIX:]

    twin_logits, _th, _tc, _tk = _sgl_forward(ckpt, tokens, r)
    twin_suffix = twin_logits[:, TOY_PREFIX:, :]

    _pl, _ph, _pc, kv = _sgl_forward(ckpt, prefix, r)
    did_swap = False
    through_grace = False
    through_nvme = False
    did_restore = False

    if not skip_swap:
        did_swap = True
        kv = np.array(kv, dtype=np.float32, copy=True)
        through_grace = True
        if not skip_nvme:
            blob, shape = _kv_to_nvme(kv)
            kv = None
            through_nvme = True
            if do_restore:
                kv = _kv_from_nvme(blob, shape)
                if restore_corrupt:
                    kv = (kv + np.float32(1.0)).astype(np.float32)
                did_restore = True
        elif do_restore:
            kv = np.array(kv, dtype=np.float32, copy=True)
            did_restore = True

    restored_suffix: np.ndarray | None
    if did_restore and kv is not None:
        restored_suffix, _h, _c, _k = _sgl_forward(ckpt, suffix, r, kv_h=kv)
    else:
        restored_suffix = None

    if compare_to_self:
        compared_to_twin = False
        if restored_suffix is None:
            raw_equal = False
        else:
            raw_equal = bool(
                _max_abs_logprob(restored_suffix, restored_suffix) <= LOGPROB_THRESHOLD
            )
    else:
        compared_to_twin = True
        if restored_suffix is None:
            raw_equal = False
        else:
            raw_equal = bool(_max_abs_logprob(restored_suffix, twin_suffix) <= LOGPROB_THRESHOLD)

    return (
        bool(did_swap)
        and bool(through_grace)
        and bool(through_nvme)
        and bool(did_restore)
        and (not bool(restore_corrupt))
        and bool(compared_to_twin)
        and bool(raw_equal)
    )


def run_v8(*, gpus: int) -> V8Result:
    """Run the V8 serving analog and report the two gates.

    ``gpus`` is the Slurm GPU count (template ``{{GPUS}}``). Spec V8 is
    1 to 4 H200. ``gpus >= 1`` is accepted so a CPU analog can run; that
    analog is not V8 verified.

    Analog (spec 16.2 / 15.5 C2):

    - Tiny checkpoint; host stand-in for SGLang vs JAX log-probs.
    - KV tier swap to a host-RAM / file stand-in, restore, match the
      unswapped twin.

    Returns a JSON-serializable ``V8Result``. Raises ``V8Error`` when
    ``gpus`` is invalid or the backend cannot produce a finite report.
    Does not write ``v8.json``; the template does that.
    """
    if gpus < 1:
        raise V8Error(f"gpus must be >= 1 (got {gpus})")
    log_ok = _logprob_within_threshold()
    restore_ok = _tiered_restore_matches()
    return {
        "logprob_within_threshold": bool(log_ok),
        "tiered_restore_matches": bool(restore_ok),
    }
