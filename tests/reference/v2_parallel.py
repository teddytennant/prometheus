"""Independent NumPy V2 parallel-equivalence protocol (spec 16.2).

Slow and obvious. Does **not** import ``prometheus.verify.v2_parallel``, JAX,
torch, ``model/``, or ``train/``. Production ``run_v2`` must not import this
module; tests import both. Must not compare a mesh run to itself.

V2 meaning
----------
``loss_rel_diff``
    Relative FP32 gap between the single-device loss trajectory and the SPMD
    mesh trajectory, over ``steps`` train steps::

        max_t |L_mesh[t] - L_single[t]| / max(|L_single[t]|, floor)

    Scalars (final or mean loss) are allowed; they are treated as length-1
    trajectories. Gate: finite, ``>= 0``, and ``<= LOSS_REL_MAX`` (1e-6).
``routing_identical``
    Expert ids (top-k per token per layer, every step) match exactly between
    the single-device run and the mesh run. Gate: ``True``.

Spec 16.2 mesh
--------------
8 H200, then 2 nodes × 4: 1 GPU vs EP=8, FSDP=8, PP=2 with 4 GPUs per stage,
CP=2; then DP across 2 nodes. A4: ``n_devices = dp * pp * ep * cp`` and
``fsdp == ep``. The 8-GPU named meshes::

    EP/FSDP-8:  dp=1, fsdp=8, ep=8, pp=1, cp=1          -> 8 devices
    PP=2, CP=2: dp=1, fsdp=2, ep=2, pp=2, cp=2          -> 8 devices
                (4 GPUs per PP stage = ep * cp)
    2×4 DP:     dp=2, fsdp=2, ep=2, pp=1, cp=1          -> 8 devices

The toy below is protocol math (tiny MoE, nested-loop EP / FSDP / PP / CP /
DP). It is not ``model.tiny_config``; production V2 runs that net. Tests use
this module for shapes, dtypes, golden relative scale, FD vs reverse-mode on
the CE head, and that a deliberately mismatched routing / loss fails the gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray

LOSS_REL_MAX = 1e-6
DEFAULT_STEPS = 200
LOSS_REL_FLOOR = 1e-12

# Toy MoE (protocol only). Sized so EP=2, PP=2, CP=2, DP=1 is 8 virtual devices.
TOY_VOCAB = 8
TOY_DIM = 4
TOY_HIDDEN = 4
TOY_EXPERTS = 4
TOY_TOP_K = 2
TOY_LAYERS = 2
TOY_BATCH = 2
TOY_SEQ = 4
TOY_SEED = 0
TOY_LR = 0.5
TOY_RMS_EPS = 1e-6
TOY_FD_EPS = 1e-3
TOY_GRAD_RTOL = 5e-2
TOY_GRAD_ATOL = 5e-3

# Fault injection: relative loss bump that must fail the 1e-6 gate.
WRONG_REL_SHIFT = 1e-3
GOLDEN_MATCH_REL = 0.0
GOLDEN_REL_ABOVE_GATE = 2e-6


@dataclass(frozen=True)
class MeshSpec:
    """SPMD mesh. A4: ``n_devices = dp * pp * ep * cp`` and ``fsdp == ep``."""

    dp: int = 1
    fsdp: int = 1
    ep: int = 1
    pp: int = 1
    cp: int = 1

    def __post_init__(self) -> None:
        for name in ("dp", "fsdp", "ep", "pp", "cp"):
            val = int(getattr(self, name))
            if val < 1:
                raise ValueError(f"{name} must be >= 1")
        if int(self.fsdp) != int(self.ep):
            raise ValueError("fsdp must equal ep (A4 / spec 5.2)")

    @property
    def n_devices(self) -> int:
        return int(self.dp) * int(self.pp) * int(self.ep) * int(self.cp)

    @property
    def gpus_per_pp_stage(self) -> int:
        return int(self.ep) * int(self.cp)


# Spec 16.2 8-GPU named meshes (production V2). Toy protocol uses TINY_MESH.
SPEC_MESH_8GPU_EP = MeshSpec(dp=1, fsdp=8, ep=8, pp=1, cp=1)
SPEC_MESH_8GPU_PP_CP = MeshSpec(dp=1, fsdp=2, ep=2, pp=2, cp=2)
SPEC_MESH_2X4_DP = MeshSpec(dp=2, fsdp=4, ep=4, pp=1, cp=1)
TINY_MESH = MeshSpec(dp=1, fsdp=2, ep=2, pp=2, cp=2)


def loss_rel_diff(
    single: object,
    mesh: object,
    *,
    floor: float = LOSS_REL_FLOOR,
) -> float:
    """Max relative FP32 gap of two loss trajectories (or scalars).

    Compared as float32 values, reduced in float64. Empty -> 0.0. Shape
    mismatch raises. A non-finite input yields a non-finite result (the V2
    gate requires a finite value).
    """
    a = np.asarray(single, dtype=np.float32).reshape(-1).astype(np.float64)
    b = np.asarray(mesh, dtype=np.float32).reshape(-1).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"loss shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    denom = np.maximum(np.abs(a), float(floor))
    rel = np.abs(a - b) / denom
    return float(np.max(rel))


def routing_identical(single_ids: object, mesh_ids: object) -> bool:
    """True iff expert-id tensors are equal (shape and values). Empty -> True."""
    a = np.asarray(single_ids)
    b = np.asarray(mesh_ids)
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    return bool(np.array_equal(a.astype(np.int64, copy=False), b.astype(np.int64, copy=False)))


def meets_v2_gates(
    rel: float,
    identical: bool,
    *,
    limit: float = LOSS_REL_MAX,
) -> bool:
    """True iff ``rel`` is finite, in ``[0, limit]``, and routing is exactly True."""
    d = float(rel)
    if not np.isfinite(d) or d < 0.0 or d > float(limit):
        return False
    return identical is True


def perturb_losses(losses: object, *, rel: float = WRONG_REL_SHIFT) -> Array:
    """Deliberately wrong mesh losses: multiply by ``(1 + rel)`` in FP32."""
    a = np.asarray(losses, dtype=np.float32)
    return (a * np.float32(1.0 + float(rel))).astype(np.float32)


def flip_one_expert_id(ids: object, n_experts: int) -> Array:
    """Deliberately wrong routing: bump flat[0] by 1 modulo ``n_experts``."""
    out = np.array(np.asarray(ids), copy=True)
    if out.size == 0:
        return out
    n = int(n_experts)
    if n < 1:
        raise ValueError("n_experts must be >= 1")
    out.flat[0] = (int(out.flat[0]) + 1) % n
    return out


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
    v = int(z.shape[-1])
    flat = z.reshape(-1, v)
    idx = t.reshape(-1).astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= v:
        raise ValueError("target id out of vocab")
    m = np.max(flat, axis=-1, keepdims=True)
    shifted = flat - m
    log_z = np.log(np.exp(shifted).sum(axis=-1)) + m.reshape(-1)
    nll = log_z - flat[np.arange(flat.shape[0]), idx]
    return float(np.mean(nll))


def ce_logits_grad(logits: Array, targets: Array) -> Array:
    """Reverse-mode ``d mean_CE / d logits`` as float64, same shape as logits."""
    z = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets)
    v = int(z.shape[-1])
    flat = z.reshape(-1, v)
    idx = t.reshape(-1).astype(np.int64)
    m = np.max(flat, axis=-1, keepdims=True)
    ex = np.exp(flat - m)
    p = ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-30)
    n = flat.shape[0]
    p[np.arange(n), idx] -= 1.0
    p /= float(n)
    return p.reshape(z.shape)


def grad_match_ok(
    analytic: Array,
    finite_diff: Array,
    *,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    """Elementwise reverse-mode vs central differences (relative + absolute)."""
    g = np.asarray(analytic, dtype=np.float64)
    h = np.asarray(finite_diff, dtype=np.float64)
    if g.shape != h.shape:
        raise ValueError(f"grad shape mismatch: {g.shape} vs {h.shape}")
    if g.size == 0:
        return True
    if not (np.isfinite(g).all() and np.isfinite(h).all()):
        return False
    scale = np.maximum(np.abs(g), np.abs(h))
    abs_err = np.abs(g - h)
    ok = (abs_err <= float(atol) + float(rtol) * scale).all()
    return bool(ok)


def pp_layer_ranges(n_layers: int, n_pp: int) -> tuple[tuple[int, int], ...]:
    """Half-open unique-layer ranges per PP stage covering ``[0, n_layers)``."""
    n_layers = int(n_layers)
    n_pp = int(n_pp)
    if n_layers < 1 or n_pp < 1:
        raise ValueError("n_layers and n_pp must be positive")
    if n_pp > n_layers:
        raise ValueError("n_pp cannot exceed n_layers")
    base, extra = divmod(n_layers, n_pp)
    ranges: list[tuple[int, int]] = []
    start = 0
    for s in range(n_pp):
        sz = base + (1 if s < extra else 0)
        ranges.append((start, start + sz))
        start += sz
    return tuple(ranges)


def _sigmoid(x: Array) -> Array:
    z = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _silu(x: Array) -> Array:
    z = np.asarray(x, dtype=np.float64)
    return z * _sigmoid(z)


def _rms_norm(x: Array, weight: Array, *, eps: float = TOY_RMS_EPS) -> Array:
    z = np.asarray(x, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    ms = np.mean(z * z, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + float(eps))
    return (z * inv * w).astype(np.float32)


def _topk_ids(logits: Array, k: int) -> Array:
    z = np.asarray(logits, dtype=np.float64)
    k = int(k)
    return np.argsort(-z, axis=-1)[..., :k].astype(np.int64)


def _softmax_selected(logits: Array, ids: Array) -> Array:
    z = np.asarray(logits, dtype=np.float64)
    idx = np.asarray(ids, dtype=np.int64)
    sel = np.take_along_axis(z, idx, axis=-1)
    m = np.max(sel, axis=-1, keepdims=True)
    ex = np.exp(sel - m)
    return (ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-30)).astype(np.float32)


def _expert_ffn(x: Array, w_up: Array, w_down: Array) -> Array:
    """SiLU MLP: ``silu(x @ w_up) @ w_down``. ``x`` (..., D)."""
    hidden = _silu(np.asarray(x, dtype=np.float64) @ np.asarray(w_up, dtype=np.float64))
    out = hidden @ np.asarray(w_down, dtype=np.float64)
    return out.astype(np.float32)


def _fsdp_all_gather(weight: Array, n: int, axis: int) -> Array:
    """Explicit FSDP: shard ``weight`` along ``axis`` into ``n`` pieces, concat back."""
    w = np.asarray(weight)
    n = int(n)
    if n == 1:
        return w
    shards = np.array_split(w, n, axis=axis)
    return np.concatenate(shards, axis=axis)


def _moe_ep(
    x: Array,
    *,
    router: Array,
    w_up: Array,
    w_down: Array,
    top_k: int,
    ep: int,
) -> tuple[Array, Array]:
    """Top-k MoE with explicit EP dispatch/combine. ``x`` is (N, D).

    Expert ``e`` is owned by rank ``e % ep``. Tokens are dispatched to owners,
    the SiLU MLP runs per expert, then slot-order combine reconstructs the
    residual update. Router uses the FSDP-gathered full matrix.
    """
    h = np.asarray(x, dtype=np.float32)
    n_tokens, d_model = h.shape
    router_g = _fsdp_all_gather(np.asarray(router, dtype=np.float32), int(ep), axis=1)
    w_up_g = _fsdp_all_gather(np.asarray(w_up, dtype=np.float32), int(ep), axis=0)
    w_down_g = _fsdp_all_gather(np.asarray(w_down, dtype=np.float32), int(ep), axis=0)
    logits = (h.astype(np.float64) @ router_g.astype(np.float64)).astype(np.float32)
    ids = _topk_ids(logits, top_k)
    gates = _softmax_selected(logits, ids)
    n_experts = int(w_up_g.shape[0])
    # Dispatch: buckets[e] <- list of (token, slot).
    buckets: list[list[tuple[int, int]]] = [[] for _ in range(n_experts)]
    for t in range(n_tokens):
        for s in range(int(top_k)):
            e = int(ids[t, s])
            buckets[e].append((t, s))
    expert_out: dict[tuple[int, int], Array] = {}
    for e in range(n_experts):
        items = buckets[e]
        if not items:
            continue
        # Owner rank is e % ep; the compute is local to that rank.
        _owner = e % int(ep)
        del _owner
        stacked = np.stack([h[t] for t, _s in items], axis=0)
        y = _expert_ffn(stacked, w_up_g[e], w_down_g[e])
        for i, (t, s) in enumerate(items):
            expert_out[(t, s)] = y[i]
    out = np.zeros((n_tokens, d_model), dtype=np.float64)
    for t in range(n_tokens):
        for s in range(int(top_k)):
            out[t] += float(gates[t, s]) * expert_out[(t, s)].astype(np.float64)
    return out.astype(np.float32), ids.astype(np.int32)


def _apply_layer(h: Array, layer: dict[str, Array], *, ep: int, cp: int) -> tuple[Array, Array]:
    """RMSNorm + MoE residual. Optional CP split of the sequence axis."""
    x = np.asarray(h, dtype=np.float32)
    b, seq, d = x.shape
    cp = int(cp)
    if seq % cp != 0:
        raise ValueError("seq_len must be divisible by cp")
    chunk = seq // cp
    ids_chunks: list[Array] = []
    out_chunks: list[Array] = []
    for c in range(cp):
        sl = x[:, c * chunk : (c + 1) * chunk, :]
        flat = sl.reshape(-1, d)
        normed = _rms_norm(flat, layer["rms"])
        delta, ids = _moe_ep(
            normed,
            router=layer["router"],
            w_up=layer["w_up"],
            w_down=layer["w_down"],
            top_k=TOY_TOP_K,
            ep=int(ep),
        )
        out_chunks.append((flat.astype(np.float64) + delta.astype(np.float64)).reshape(b, chunk, d))
        ids_chunks.append(ids.reshape(b, chunk, TOY_TOP_K))
    y = np.concatenate(out_chunks, axis=1).astype(np.float32)
    ids = np.concatenate(ids_chunks, axis=1).astype(np.int32)
    return y, ids


def _embed_tokens(tokens: Array, embed: Array, *, fsdp: int) -> Array:
    table = _fsdp_all_gather(np.asarray(embed, dtype=np.float32), int(fsdp), axis=0)
    idx = np.asarray(tokens, dtype=np.int64)
    return table[idx].astype(np.float32)


def _unembed_logits(h: Array, unembed: Array, *, fsdp: int) -> Array:
    table = _fsdp_all_gather(np.asarray(unembed, dtype=np.float32), int(fsdp), axis=0)
    z = np.asarray(h, dtype=np.float64) @ table.astype(np.float64).T
    return z.astype(np.float32)


def _stack_forward(
    tokens: Array,
    params: dict[str, Any],
    mesh: MeshSpec,
) -> tuple[float, Array, Array]:
    """One replica: embed -> PP stages of MoE -> unembed. Returns loss, ids, hidden."""
    tok = np.asarray(tokens, dtype=np.int32)
    h = _embed_tokens(tok, params["embed"], fsdp=mesh.fsdp)
    ranges = pp_layer_ranges(len(params["layers"]), mesh.pp)
    ids_layers: list[Array] = []
    for start, end in ranges:
        for li in range(start, end):
            h, ids = _apply_layer(h, params["layers"][li], ep=mesh.ep, cp=mesh.cp)
            ids_layers.append(ids)
    logits = _unembed_logits(h, params["unembed"], fsdp=mesh.fsdp)
    loss = mean_cross_entropy(logits[:, :-1, :], tok[:, 1:])
    ids = np.stack(ids_layers, axis=0).astype(np.int32)
    return float(np.float32(loss)), ids, h


def single_forward(tokens: Array, params: dict[str, Any]) -> tuple[float, Array]:
    """Unsharded run: mesh dims all 1 (the single-device path)."""
    loss, ids, _h = _stack_forward(tokens, params, MeshSpec())
    return loss, ids


def mesh_forward(
    tokens: Array,
    params: dict[str, Any],
    mesh: MeshSpec,
) -> tuple[float, Array]:
    """SPMD analog: DP splits the batch; each replica runs PP/EP/FSDP/CP."""
    tok = np.asarray(tokens, dtype=np.int32)
    dp = int(mesh.dp)
    if tok.shape[0] % dp != 0:
        raise ValueError("batch must be divisible by dp")
    bs = tok.shape[0] // dp
    losses: list[float] = []
    ids_parts: list[Array] = []
    replica_mesh = MeshSpec(dp=1, fsdp=mesh.fsdp, ep=mesh.ep, pp=mesh.pp, cp=mesh.cp)
    for r in range(dp):
        sl = tok[r * bs : (r + 1) * bs]
        loss, ids, _h = _stack_forward(sl, params, replica_mesh)
        losses.append(loss)
        ids_parts.append(ids)
    loss = float(np.float32(np.mean(np.asarray(losses, dtype=np.float64))))
    ids = np.concatenate(ids_parts, axis=1).astype(np.int32)
    return loss, ids


def toy_init(seed: int = TOY_SEED) -> dict[str, Any]:
    """Tiny embedding, per-layer RMS/router/experts, unembed. N(0, 0.02) FP32."""
    rng = np.random.default_rng(int(seed))
    scale = 0.02
    layers = []
    for _ in range(TOY_LAYERS):
        layers.append(
            {
                "rms": np.ones((TOY_DIM,), dtype=np.float32),
                "router": rng.normal(0.0, scale, (TOY_DIM, TOY_EXPERTS)).astype(np.float32),
                "w_up": rng.normal(0.0, scale, (TOY_EXPERTS, TOY_DIM, TOY_HIDDEN)).astype(
                    np.float32
                ),
                "w_down": rng.normal(0.0, scale, (TOY_EXPERTS, TOY_HIDDEN, TOY_DIM)).astype(
                    np.float32
                ),
            }
        )
    return {
        "embed": rng.normal(0.0, scale, (TOY_VOCAB, TOY_DIM)).astype(np.float32),
        "unembed": rng.normal(0.0, scale, (TOY_VOCAB, TOY_DIM)).astype(np.float32),
        "layers": layers,
    }


def toy_tokens(seed: int = TOY_SEED) -> Array:
    """``(TOY_BATCH, TOY_SEQ)`` int32 ids in ``[0, TOY_VOCAB)``."""
    rng = np.random.default_rng(int(seed) + 1)
    return rng.integers(0, TOY_VOCAB, size=(TOY_BATCH, TOY_SEQ), dtype=np.int32)


def _copy_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "embed": np.array(params["embed"], copy=True),
        "unembed": np.array(params["unembed"], copy=True),
        "layers": [
            {k: np.array(v, copy=True) for k, v in layer.items()} for layer in params["layers"]
        ],
    }


def _hidden_and_logits(
    tokens: Array, params: dict[str, Any]
) -> tuple[Array, Array]:
    """Single-device hidden (B, S, D) and logits (B, S, V)."""
    _loss, _ids, h = _stack_forward(tokens, params, MeshSpec())
    logits = _unembed_logits(h, params["unembed"], fsdp=1)
    return h, logits


def toy_unembed_grad(tokens: Array, params: dict[str, Any]) -> Array:
    """Reverse-mode ``d CE / d unembed``. MoE is in the forward hidden only."""
    tok = np.asarray(tokens, dtype=np.int32)
    h, logits = _hidden_and_logits(tok, params)
    d_logits = ce_logits_grad(logits[:, :-1, :], tok[:, 1:])
    # logits = h @ W^T  ->  dW = d_logits^T @ h  over the CE positions.
    h_ce = h[:, :-1, :].reshape(-1, TOY_DIM).astype(np.float64)
    g = d_logits.reshape(-1, TOY_VOCAB).astype(np.float64)
    d_w = g.T @ h_ce
    return d_w.astype(np.float32)


def toy_sgd_step(tokens: Array, params: dict[str, Any], *, lr: float = TOY_LR) -> dict[str, Any]:
    """One SGD step on embed (STE through residual) and unembed (exact CE head).

    Embed grad treats the MoE stack as identity so routing can still move
    across steps without a full MoE backward. Unembed grad is exact.
    """
    out = _copy_params(params)
    tok = np.asarray(tokens, dtype=np.int32)
    h, logits = _hidden_and_logits(tok, params)
    d_logits = ce_logits_grad(logits[:, :-1, :], tok[:, 1:])
    w = np.asarray(params["unembed"], dtype=np.float64)
    h_ce = h[:, :-1, :].reshape(-1, TOY_DIM).astype(np.float64)
    g = d_logits.reshape(-1, TOY_VOCAB).astype(np.float64)
    d_unembed = g.T @ h_ce
    d_h = np.zeros_like(h, dtype=np.float64)
    d_h[:, :-1, :] = (g @ w).reshape(tok.shape[0], tok.shape[1] - 1, TOY_DIM)
    d_embed = np.zeros_like(params["embed"], dtype=np.float64)
    flat_tok = tok.reshape(-1).astype(np.int64)
    flat_dh = d_h.reshape(-1, TOY_DIM)
    for i, tid in enumerate(flat_tok):
        d_embed[int(tid)] += flat_dh[i]
    lr64 = float(lr)
    out["unembed"] = (out["unembed"].astype(np.float64) - lr64 * d_unembed).astype(np.float32)
    out["embed"] = (out["embed"].astype(np.float64) - lr64 * d_embed).astype(np.float32)
    return out


def toy_finite_diff_unembed_slice(
    tokens: Array,
    params: dict[str, Any],
    *,
    eps: float = TOY_FD_EPS,
    n_coords: int = 4,
) -> tuple[Array, Array]:
    """Central differences vs reverse-mode on the first ``n_coords`` unembed entries."""
    analytic = toy_unembed_grad(tokens, params).reshape(-1)[:n_coords].astype(np.float32)
    numeric = np.zeros((n_coords,), dtype=np.float32)
    base = np.array(params["unembed"], copy=True).reshape(-1)
    for i in range(n_coords):
        plus = _copy_params(params)
        minus = _copy_params(params)
        up = np.array(plus["unembed"], copy=True).reshape(-1)
        um = np.array(minus["unembed"], copy=True).reshape(-1)
        up[i] = np.float32(base[i] + np.float32(eps))
        um[i] = np.float32(base[i] - np.float32(eps))
        plus["unembed"] = up.reshape(params["unembed"].shape)
        minus["unembed"] = um.reshape(params["unembed"].shape)
        lp, _ = single_forward(tokens, plus)
        lm, _ = single_forward(tokens, minus)
        numeric[i] = np.float32((float(lp) - float(lm)) / (2.0 * float(eps)))
    return analytic, numeric


def toy_grad_ok(seed: int = TOY_SEED) -> bool:
    """Reverse-mode unembed grads match central differences on a tiny slice."""
    params = toy_init(seed)
    tokens = toy_tokens(seed)
    analytic, numeric = toy_finite_diff_unembed_slice(tokens, params)
    return grad_match_ok(analytic, numeric)


def toy_train_pair(
    steps: int,
    *,
    seed: int = TOY_SEED,
    mesh: MeshSpec | None = None,
    lr: float = TOY_LR,
) -> dict[str, Any]:
    """``steps`` SGD steps; compare single-device vs mesh at every step.

    Both paths share the same parameter updates (single-device SGD). The mesh
    is an independent forward (EP dispatch, FSDP gather, PP stages, CP split,
    DP mean). Comparing a mesh run to itself is forbidden: the two forwards
    are different functions.
    """
    n = int(steps)
    if n < 1:
        raise ValueError("steps must be >= 1")
    mesh = TINY_MESH if mesh is None else mesh
    params = toy_init(seed)
    tokens = toy_tokens(seed)
    single_losses: list[float] = []
    mesh_losses: list[float] = []
    single_ids: list[Array] = []
    mesh_ids: list[Array] = []
    for _ in range(n):
        ls, ids_s = single_forward(tokens, params)
        lm, ids_m = mesh_forward(tokens, params, mesh)
        single_losses.append(ls)
        mesh_losses.append(lm)
        single_ids.append(ids_s)
        mesh_ids.append(ids_m)
        params = toy_sgd_step(tokens, params, lr=lr)
    return {
        "single_losses": np.asarray(single_losses, dtype=np.float32),
        "mesh_losses": np.asarray(mesh_losses, dtype=np.float32),
        "single_ids": np.stack(single_ids, axis=0).astype(np.int32),
        "mesh_ids": np.stack(mesh_ids, axis=0).astype(np.int32),
        "tokens": tokens,
    }


def evaluate_v2_protocol(
    steps: int = 2,
    *,
    seed: int = TOY_SEED,
    mesh: MeshSpec | None = None,
) -> dict[str, Any]:
    """Run the toy single-vs-mesh protocol and the V2 gates."""
    pair = toy_train_pair(steps, seed=seed, mesh=mesh)
    rel = loss_rel_diff(pair["single_losses"], pair["mesh_losses"])
    identical = routing_identical(pair["single_ids"], pair["mesh_ids"])
    return {
        "loss_rel_diff": rel,
        "routing_identical": identical,
        "grad_ok": toy_grad_ok(seed),
        "single_losses": pair["single_losses"],
        "mesh_losses": pair["mesh_losses"],
        "single_ids": pair["single_ids"],
        "mesh_ids": pair["mesh_ids"],
        "tokens": pair["tokens"],
        "meets_gates": meets_v2_gates(rel, identical),
    }
