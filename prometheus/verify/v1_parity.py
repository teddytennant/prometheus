"""V1 reference parity runner (spec 16.2)."""

from __future__ import annotations

from typing import Any, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import model
import train
from prometheus.verify._numpy_forward import forward as numpy_forward

LOGITS_MAX_ABS = 1e-5

_FD_EPS = 1e-3
_GRAD_RTOL = 5e-2
_GRAD_ATOL = 5e-3
_UNEMBED_COORDS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0))


class V1Result(TypedDict):
    logits_max_diff: float
    grad_ok: bool
    overfit_ok: bool


class V1Error(Exception):
    """Raised when V1 cannot run (e.g. gpus < 1)."""


def _numpy_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _numpy_tree(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        converted = [_numpy_tree(v) for v in tree]
        return list(converted) if isinstance(tree, list) else tuple(converted)
    return np.asarray(tree)


def _mean_logits(params: Any, tokens: Any, cfg: model.ModelConfig) -> Any:
    out = model.forward(tokens, params, cfg, r=1)
    return jnp.mean(out.logits)


def _packed_total(
    params: Any,
    batch: train.Batch,
    mcfg: model.ModelConfig,
    tcfg: train.TrainConfig,
) -> float:
    out = model.forward(batch.tokens, params, mcfg, r=1)
    cap = float(tcfg.softcap)
    tokens = np.asarray(batch.tokens)
    mask = np.asarray(batch.loss_mask, dtype=np.float32)
    ce = train.cross_entropy(
        train.soft_cap(np.asarray(out.logits)[:, :-1, :], cap),
        tokens[:, 1:],
        mask[:, 1:],
    )
    mtp_heads = tuple(train.soft_cap(np.asarray(head), cap) for head in out.mtp_logits)
    mtp = train.mtp_loss(mtp_heads, tokens, mask)
    z = (
        train.z_loss(out.router_probs)
        if out.router_probs is not None
        else np.float32(0.0)
    )
    return float(np.asarray(train.total_loss(ce, mtp, z, tcfg)))


def _grad_ok(params: Any, tokens: Any, cfg: model.ModelConfig) -> bool:
    tokens_j = jnp.asarray(tokens)
    unembed = params["unembed"]

    def loss_unembed(weight: Any) -> Any:
        return _mean_logits({**params, "unembed": weight}, tokens_j, cfg)

    reverse = jax.grad(loss_unembed)(unembed)
    reverse_np = np.asarray(reverse, dtype=np.float32)
    fd = np.empty(len(_UNEMBED_COORDS), dtype=np.float32)
    reverse_slice = np.empty(len(_UNEMBED_COORDS), dtype=np.float32)
    for n, (i, j) in enumerate(_UNEMBED_COORDS):
        plus = {**params, "unembed": unembed.at[i, j].add(np.float32(_FD_EPS))}
        minus = {**params, "unembed": unembed.at[i, j].add(np.float32(-_FD_EPS))}
        lp = float(np.asarray(_mean_logits(plus, tokens_j, cfg)))
        lm = float(np.asarray(_mean_logits(minus, tokens_j, cfg)))
        fd[n] = np.float32((lp - lm) / (2.0 * _FD_EPS))
        reverse_slice[n] = reverse_np[i, j]
    if not (np.isfinite(fd).all() and np.isfinite(reverse_slice).all()):
        return False
    return bool(np.allclose(reverse_slice, fd, rtol=_GRAD_RTOL, atol=_GRAD_ATOL))


def _overfit_ok(params: Any, mcfg: model.ModelConfig) -> bool:
    tcfg = train.tiny_train_config()
    opt = train.init_opt_state(params, tcfg)
    rng = np.random.default_rng(1)
    batch_n, seq = 2, 8
    tokens = rng.integers(0, mcfg.vocab_size, size=(batch_n, seq), dtype=np.int32)
    loss_mask = np.ones((batch_n, seq), dtype=np.float32)
    loss_mask[:, 0] = 0.0
    positions = np.broadcast_to(np.arange(seq, dtype=np.int32), (batch_n, seq)).copy()
    batch = train.Batch(tokens=tokens, loss_mask=loss_mask, positions=positions)
    stepped = train.train_step(params, opt, batch, mcfg, tcfg, step=1)
    loss_before = float(np.asarray(stepped.loss.total))
    loss_after = _packed_total(stepped.params, batch, mcfg, tcfg)
    if not (np.isfinite(loss_before) and np.isfinite(loss_after)):
        return False
    return bool(loss_after < loss_before)


def run_v1(*, gpus: int) -> V1Result:
    if gpus < 1:
        raise V1Error("gpus must be >= 1")

    cfg = model.tiny_config()
    params = model.init_params(cfg, rng=0)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, 4), dtype=np.int32)

    jax_out = model.forward(tokens, params, cfg, r=1)
    jax_logits = np.asarray(jax_out.logits, dtype=np.float32)
    np_logits = np.asarray(
        numpy_forward(tokens, _numpy_tree(params), cfg, r=1), dtype=np.float32
    )
    logits_max_diff = float(np.max(np.abs(jax_logits - np_logits)))

    grad_ok = _grad_ok(params, tokens, cfg)
    overfit_ok = _overfit_ok(params, cfg)

    return {
        "logits_max_diff": logits_max_diff,
        "grad_ok": grad_ok,
        "overfit_ok": overfit_ok,
    }
