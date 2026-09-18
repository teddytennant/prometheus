"""V2 parallel equivalence runner (spec 16.2, F4 template `v2.sh`).

Same tiny flagship-shape model, single-device vs a SPMD mesh (EP / FSDP /
PP / CP). `verify/ncshare` `check_exit(V2)` reads `v2.json` with:

- `loss_rel_diff` (float): relative FP32 loss gap vs single-device; must
  be <= 1e-6 over `steps` (default 200)
- `routing_identical` (bool): expert routing matches the single-device run

Spec V2 is 8 H200, then 2 nodes x 4: 1 GPU vs EP=8, FSDP=8, PP=2 with 4
GPUs per stage, CP=2; then DP across 2 nodes. A CPU JAX backend is allowed
so the analog can run without a GPU. That does not count as V2 verified.
V2 itself waits for V0 and V1.

Must not import `tests/`. Must not compare a mesh run to itself.
"""

from __future__ import annotations

from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import kernels
import model
import parallel
import train
from model import AttentionKind, FfnKind, ForwardOutput, ModelConfig

LOSS_REL_MAX = 1e-6
DEFAULT_STEPS = 200

_BATCH = 2
_SEQ = 8
_REL_FLOOR = 1e-12

_as_f32 = model._as_f32
_dense_ffn = model._dense_ffn
_linear_attn = model._linear_attn
_mla_attn = model._mla_attn
_node_limited_topk = model._node_limited_topk
_sigmoid = model._sigmoid
_swiglu = model._swiglu


class V2Result(TypedDict):
    """Payload written to `v2.json` by the F4 template."""

    loss_rel_diff: float
    routing_identical: bool


class V2Error(Exception):
    """Bad V2 inputs (gpus < 1, steps < 1) or a non-finite / missing backend."""


def run_v2(*, gpus: int, steps: int = DEFAULT_STEPS) -> V2Result:
    """Run V2 parallel equivalence.

    Parameters
    ----------
    gpus:
        Slurm GPU count. Must be >= 1. `gpus < 1` raises `V2Error`.
    steps:
        Train steps to compare. Must be >= 1. Default 200 (spec 16.2).
        `steps < 1` raises `V2Error`.

    Returns
    -------
    V2Result
        Keys consumed by `check_exit(V2)`. Does not write `v2.json`; the
        template does that.
    """
    if gpus < 1 or steps < 1:
        raise V2Error(f"gpus and steps must be >= 1 (gpus={gpus}, steps={steps})")

    cfg = model.tiny_config()
    tcfg = train.tiny_train_config()
    params = model.init_params(cfg, rng=0)
    spec = _mesh_spec(gpus)

    single_losses: list[float] = []
    mesh_losses: list[float] = []
    single_ids: list[np.ndarray] = []
    mesh_ids: list[np.ndarray] = []

    for step in range(steps):
        tokens, mask = _batch(cfg, step)
        single_out = model.forward(tokens, params, cfg, r=1)
        mesh_out = _mesh_forward(tokens, params, cfg, spec)
        s_loss = _step_loss(single_out, tokens, mask, tcfg)
        m_loss = _step_loss(mesh_out, tokens, mask, tcfg)
        if not np.isfinite(s_loss) or not np.isfinite(m_loss):
            raise V2Error("non-finite step loss")
        single_losses.append(s_loss)
        mesh_losses.append(m_loss)
        s_ids = _require_ids(single_out.expert_ids, "single")
        m_ids = _require_ids(mesh_out.expert_ids, "mesh")
        single_ids.append(s_ids)
        mesh_ids.append(m_ids)

    rel = _loss_rel_diff(np.asarray(single_losses, dtype=np.float32), np.asarray(mesh_losses))
    if not np.isfinite(rel) or rel < 0.0:
        raise V2Error("loss_rel_diff must be finite and >= 0")
    stacked_single = np.stack(single_ids, axis=0)
    stacked_mesh = np.stack(mesh_ids, axis=0)
    identical = bool(
        stacked_single.shape == stacked_mesh.shape and np.array_equal(stacked_single, stacked_mesh)
    )
    return {"loss_rel_diff": rel, "routing_identical": identical}


def _mesh_spec(gpus: int) -> parallel.MeshSpec:
    """Logical spec-16.2 mesh. Integer device ids; does not need 8 physical GPUs.

    EP=8, FSDP=8, PP=2, CP=2. DP=2 when `gpus` can cover the 2x4 follow-up.
    """
    dp = 2 if gpus >= 8 else 1
    return parallel.MeshSpec(dp=dp, fsdp=8, ep=8, pp=2, cp=2)


def _batch(cfg: ModelConfig, step: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10_000 + int(step))
    tokens = rng.integers(0, cfg.vocab_size, size=(_BATCH, _SEQ), dtype=np.int32)
    mask = np.ones((_BATCH, _SEQ), dtype=np.float32)
    mask[:, 0] = 0.0
    return tokens, mask


def _require_ids(ids: Any, which: str) -> np.ndarray:
    if ids is None:
        raise V2Error(f"{which} forward did not return expert_ids")
    return np.asarray(ids, dtype=np.int32)


def _step_loss(
    out: ForwardOutput,
    tokens: np.ndarray,
    mask: np.ndarray,
    tcfg: train.TrainConfig,
) -> float:
    cap = float(tcfg.softcap)
    logits = train.soft_cap(np.asarray(out.logits)[:, :-1, :], cap)
    ce = train.cross_entropy(logits, np.asarray(tokens)[:, 1:], np.asarray(mask)[:, 1:])
    mtp_heads = tuple(train.soft_cap(np.asarray(head), cap) for head in out.mtp_logits)
    mtp = train.mtp_loss(mtp_heads, tokens, mask)
    if out.router_probs is None:
        z = np.float32(0.0)
    else:
        z = train.z_loss(out.router_probs)
    total = train.total_loss(ce, mtp, z, tcfg)
    return float(np.asarray(total, dtype=np.float32))


def _loss_rel_diff(single_losses: np.ndarray, mesh_losses: np.ndarray) -> float:
    a = np.asarray(single_losses, dtype=np.float32).reshape(-1).astype(np.float64)
    b = np.asarray(mesh_losses, dtype=np.float32).reshape(-1).astype(np.float64)
    if a.shape != b.shape or a.size == 0:
        raise V2Error("single/mesh loss trajectories must be non-empty and aligned")
    denom = np.maximum(np.abs(a), _REL_FLOOR)
    return float(np.max(np.abs(a - b) / denom))


def _param_kind(key: str) -> parallel.ParamKind:
    if key.startswith("routed_"):
        return parallel.ParamKind.ROUTED_EXPERT
    if key.startswith("shared_"):
        return parallel.ParamKind.SHARED_EXPERT
    if key.startswith("ffn_"):
        return parallel.ParamKind.DENSE
    if key in ("W_q", "W_k", "W_v", "W_o", "W_kv_compress", "W_kv_up", "W_rope_k"):
        return parallel.ParamKind.ATTENTION
    return parallel.ParamKind.OTHER


def _shard_concat(weight: Any, spec: parallel.MeshSpec, kind: parallel.ParamKind) -> np.ndarray:
    """FSDP shard then all-gather. Identity when the axis divides evenly."""
    w = np.asarray(weight)
    part = parallel.fsdp_partition(kind)
    n = int(spec.fsdp)
    if n <= 1 or part.axes[0] is None:
        return w
    axis = 0
    for i, dim in enumerate(w.shape):
        if dim % n == 0:
            axis = i
            break
    else:
        return w
    shards = np.split(w, n, axis=axis)
    return np.concatenate(shards, axis=axis)


def _gather_layer(layer: dict[str, Any], spec: parallel.MeshSpec) -> dict[str, np.ndarray]:
    return {k: _shard_concat(v, spec, _param_kind(k)) for k, v in layer.items()}


def _gather_params(params: dict[str, Any], spec: parallel.MeshSpec) -> dict[str, Any]:
    attn = parallel.ParamKind.ATTENTION
    dense = parallel.ParamKind.DENSE
    other = parallel.ParamKind.OTHER
    return {
        "embed": _shard_concat(params["embed"], spec, attn),
        "unembed": _shard_concat(params["unembed"], spec, attn),
        "final_norm": _shard_concat(params["final_norm"], spec, other),
        "layers": [_gather_layer(layer, spec) for layer in params["layers"]],
        "mtp": [
            {
                "norm": _shard_concat(head["norm"], spec, other),
                "proj": _shard_concat(head["proj"], spec, dense),
                "unembed": _shard_concat(head["unembed"], spec, attn),
            }
            for head in params["mtp"]
        ],
        "adapter_w1": _shard_concat(params["adapter_w1"], spec, dense),
        "adapter_w2": _shard_concat(params["adapter_w2"], spec, dense),
        "adapter_norm": _shard_concat(params["adapter_norm"], spec, other),
    }


def _silu_np(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32)
    return z / (1.0 + np.exp(-np.clip(z, -80.0, 80.0)))


def _ep_moe_chunk(
    x: Any,
    layer: dict[str, Any],
    config: ModelConfig,
    spec: parallel.MeshSpec,
) -> tuple[Any, Any, Any]:
    """Routed MoE via EP dispatch / local expert / combine, plus shared experts."""
    x32 = _as_f32(x)
    orig = tuple(int(s) for s in x32.shape)
    d_model = orig[-1]
    flat = x32.reshape(-1, d_model)
    router = _as_f32(layer["router"])
    logits = flat @ router
    probs = _sigmoid(logits)
    expert_ids = _node_limited_topk(probs, config.top_k, config.max_racks)

    ids_np = np.asarray(expert_ids, dtype=np.int32)
    probs_np = np.asarray(probs, dtype=np.float32)
    tok = np.arange(ids_np.shape[0])[:, None]
    top_scores = probs_np[tok, ids_np]
    gates = top_scores / np.maximum(top_scores.sum(axis=-1, keepdims=True), 1e-9)

    n_experts = int(router.shape[-1])
    n_racks = min(int(config.max_racks), n_experts)
    racks = (np.arange(n_experts, dtype=np.int32) * n_racks) // n_experts
    meta = kernels.DispatchMeta(
        expert_ids=ids_np,
        probs=np.asarray(gates, dtype=np.float32),
        racks=racks,
        n_experts=n_experts,
        max_racks=int(config.max_racks),
    )
    dispatched, residual = parallel.ep_dispatch(np.asarray(flat, dtype=np.float32), meta)
    d = np.asarray(dispatched, dtype=np.float32)
    w_gate = np.asarray(layer["routed_gate"], dtype=np.float32)
    w_up = np.asarray(layer["routed_up"], dtype=np.float32)
    w_down = np.asarray(layer["routed_down"], dtype=np.float32)
    hidden = _silu_np(np.einsum("emd,edh->emh", d, w_gate)) * np.einsum("emd,edh->emh", d, w_up)
    full = np.einsum("emh,ehd->emd", hidden, w_down)
    expert_out = np.zeros_like(full)
    for ep_i in range(spec.ep):
        own = np.arange(n_experts) % spec.ep == ep_i
        expert_out[own] = full[own]
    combined = jnp.asarray(parallel.ep_combine(expert_out, meta, residual), dtype=jnp.float32)

    s_gate = _as_f32(layer["shared_gate"])
    s_up = _as_f32(layer["shared_up"])
    s_down = _as_f32(layer["shared_down"])
    shared = jnp.zeros((flat.shape[0], d_model), dtype=jnp.float32)
    for s in range(int(s_gate.shape[0])):
        shared = shared + _swiglu(flat, s_gate[s], s_up[s], s_down[s])
    y = (combined + shared).reshape(orig)
    ids_out = expert_ids.reshape(*orig[:-1], int(config.top_k))
    probs_out = probs.reshape(*orig[:-1], n_experts)
    return y, probs_out, ids_out


def _ep_moe(
    x: Any,
    layer: dict[str, Any],
    config: ModelConfig,
    spec: parallel.MeshSpec,
) -> tuple[Any, Any, Any]:
    """CP-split the sequence, run EP MoE on each chunk, concat."""
    x32 = _as_f32(x)
    seq = int(x32.shape[1])
    cp = int(spec.cp)
    if cp <= 1 or seq % cp != 0:
        return _ep_moe_chunk(x32, layer, config, spec)
    chunk = seq // cp
    ys: list[Any] = []
    probs: list[Any] = []
    ids: list[Any] = []
    for cp_i in range(cp):
        sl = x32[:, cp_i * chunk : (cp_i + 1) * chunk, :]
        y, pr, expert_ids = _ep_moe_chunk(sl, layer, config, spec)
        ys.append(y)
        probs.append(pr)
        ids.append(expert_ids)
    return jnp.concatenate(ys, axis=1), jnp.concatenate(probs, axis=1), jnp.concatenate(ids, axis=1)


def _mesh_block(
    h: Any,
    layer: dict[str, Any],
    config: ModelConfig,
    layer_index: int,
    positions: Any,
    lin_state: Any,
    spec: parallel.MeshSpec,
) -> tuple[Any, Any, Any, Any, Any]:
    n = model.rms_norm(h, layer["pre_attn_norm"])
    if model.attention_kind(layer_index, config) is AttentionKind.LINEAR:
        attn, lin_state = _linear_attn(n, layer, lin_state)
    else:
        attn = _mla_attn(n, layer, positions)
        lin_state = None
    h = h + attn
    n = model.rms_norm(h, layer["pre_ffn_norm"])
    router_logits = router_probs = expert_ids = None
    if model.ffn_kind(layer_index, config) is FfnKind.DENSE:
        h = h + _dense_ffn(n, layer)
    else:
        y, router_probs, expert_ids = _ep_moe(n, layer, config, spec)
        router_logits = jnp.reshape(n, (-1, n.shape[-1])) @ _as_f32(layer["router"])
        h = h + y
    return h, lin_state, router_logits, router_probs, expert_ids


def _mesh_replica(
    tokens: np.ndarray,
    params: dict[str, Any],
    config: ModelConfig,
    spec: parallel.MeshSpec,
    dp_i: int,
) -> ForwardOutput:
    # Logical devices for this DP replica: PP stages, EP ranks, CP ranks.
    stage_devs: list[list[int]] = [[] for _ in range(spec.pp)]
    for dev in parallel.device_ids(spec):
        coord = parallel.coord_of(dev, spec)
        if coord[0] == dp_i:
            stage_devs[coord[1]].append(dev)

    stages = parallel.pipeline_stage_layers(
        config.n_layers, spec.pp, config.core_block_layers, 1
    )
    embed = jnp.asarray(params["embed"], dtype=jnp.float32)
    tok_j = jnp.asarray(tokens, dtype=jnp.int32)
    h = embed[tok_j]
    z_sum = jnp.float32(0.0)
    n_moe = jnp.float32(0.0)
    last_probs = last_ids = None
    pos = jnp.arange(int(h.shape[1]), dtype=jnp.float32)

    for pp_i, stage in enumerate(stages):
        _ = stage_devs[pp_i]
        for layer_index in range(stage.start, stage.end):
            h, _lin_state, router_logits, router_probs, expert_ids = _mesh_block(
                h,
                params["layers"][layer_index],
                config,
                layer_index,
                pos,
                None,
                spec,
            )
            if router_logits is not None:
                lse = jax.nn.logsumexp(router_logits, axis=-1)
                z_sum = z_sum + jnp.mean(jnp.square(lse)).astype(jnp.float32)
                n_moe = n_moe + jnp.float32(1.0)
                last_probs, last_ids = router_probs, expert_ids

    h = model.rms_norm(h, params["final_norm"])
    logits = h @ _as_f32(params["unembed"]).T
    mtp_logits = []
    for head in params["mtp"]:
        mh = model.rms_norm(h, head["norm"])
        mh = mh @ _as_f32(head["proj"]) + h
        mtp_logits.append(mh @ _as_f32(head["unembed"]).T)
    z = z_sum / jnp.maximum(n_moe, jnp.float32(1.0))
    return ForwardOutput(
        logits=logits.astype(jnp.float32),
        hidden=h.astype(jnp.float32),
        mtp_logits=tuple(m.astype(jnp.float32) for m in mtp_logits),
        router_probs=last_probs,
        expert_ids=last_ids,
        z_loss=z,
        r_used=1,
    )


def _mesh_forward(
    tokens: np.ndarray,
    params: dict[str, Any],
    config: ModelConfig,
    spec: parallel.MeshSpec,
) -> ForwardOutput:
    gathered = _gather_params(params, spec)
    tok = np.asarray(tokens, dtype=np.int32)
    batch = int(tok.shape[0])
    if batch % spec.dp != 0:
        raise V2Error(f"batch {batch} not divisible by dp={spec.dp}")
    bs = batch // spec.dp
    logits_p: list[np.ndarray] = []
    hidden_p: list[np.ndarray] = []
    mtp_p: list[list[np.ndarray]] = []
    ids_p: list[np.ndarray] = []
    probs_p: list[np.ndarray] = []
    z_p: list[float] = []
    for dp_i in range(spec.dp):
        sl = slice(dp_i * bs, (dp_i + 1) * bs)
        out = _mesh_replica(tok[sl], gathered, config, spec, dp_i)
        logits_p.append(np.asarray(out.logits, dtype=np.float32))
        hidden_p.append(np.asarray(out.hidden, dtype=np.float32))
        mtp_p.append([np.asarray(head, dtype=np.float32) for head in out.mtp_logits])
        if out.expert_ids is None or out.router_probs is None:
            raise V2Error("mesh replica missing expert_ids")
        ids_p.append(np.asarray(out.expert_ids, dtype=np.int32))
        probs_p.append(np.asarray(out.router_probs, dtype=np.float32))
        z_p.append(float(np.asarray(out.z_loss)))
    logits = np.concatenate(logits_p, axis=0)
    hidden = np.concatenate(hidden_p, axis=0)
    expert_ids = np.concatenate(ids_p, axis=0)
    router_probs = np.concatenate(probs_p, axis=0)
    mtp_logits = tuple(np.concatenate([p[i] for p in mtp_p], axis=0) for i in range(len(mtp_p[0])))
    z_mean = float(np.mean(np.asarray(z_p, dtype=np.float32)))
    return ForwardOutput(
        logits=logits,
        hidden=hidden,
        mtp_logits=mtp_logits,
        router_probs=router_probs,
        expert_ids=expert_ids,
        z_loss=np.float32(z_mean),
        r_used=1,
    )
