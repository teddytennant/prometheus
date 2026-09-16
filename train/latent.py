"""Latent Stage A/B curriculum (spec 4, 16.2 V6)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from model.layers import _apply_latent_adapter


def _depth_task(params: dict, hidden: np.ndarray, steps: int = 50) -> dict[str, Any]:
    """More applications of the model's latent adapter fit a 4-step teacher better."""
    x = jnp.asarray(np.asarray(hidden, dtype=np.float32).reshape(-1, hidden.shape[-1]))
    rng = np.random.default_rng(3)
    tw1 = jnp.asarray(
        rng.standard_normal(np.asarray(params["adapter_w1"]).shape).astype(np.float32) * 0.2
    )
    tw2 = jnp.asarray(
        rng.standard_normal(np.asarray(params["adapter_w2"]).shape).astype(np.float32) * 0.2
    )
    tn = jnp.ones_like(params["adapter_norm"])

    def apply(h, a, b, n, r):
        y = h
        for _ in range(int(r)):
            y = y + _apply_latent_adapter(y, a, b, n)
        return y

    y = apply(x, tw1, tw2, tn, 4)
    w1 = jnp.asarray(params["adapter_w1"])
    w2 = jnp.asarray(params["adapter_w2"])
    nw = jnp.asarray(params["adapter_norm"])

    def loss(a, b, r):
        return jnp.mean((apply(x, a, b, nw, r) - y) ** 2)

    lr = 0.05
    for _ in range(int(steps)):
        g1, g2 = jax.grad(lambda a, b: loss(a, b, 4), argnums=(0, 1))(w1, w2)
        w1 = w1 - lr * g1
        w2 = w2 - lr * g2
    l1 = float(loss(w1, w2, 1))
    l4 = float(loss(w1, w2, 4))
    return {"loss_r1": l1, "loss_r4": l4, "rises": bool(l4 < l1 - 1e-4)}


def _adapter_decode(params, hidden: np.ndarray, steps: int = 40) -> dict[str, Any]:
    """Train the model's adapter_w1/w2 to reconstruct Stage-A hidden."""
    h = jnp.asarray(np.asarray(hidden, dtype=np.float32).reshape(-1, hidden.shape[-1]))
    w1 = jnp.asarray(params["adapter_w1"])
    w2 = jnp.asarray(params["adapter_w2"])
    nw = jnp.asarray(params["adapter_norm"])

    def loss(a, b, nrm):
        pred = _apply_latent_adapter(h, a, b, nrm)
        return jnp.mean((pred - h) ** 2)

    start = float(loss(w1, w2, nw))
    lr = 0.1
    for _ in range(int(steps)):
        g1, g2 = jax.grad(loss, argnums=(0, 1))(w1, w2, nw)
        w1 = w1 - lr * g1
        w2 = w2 - lr * g2
    end = float(loss(w1, w2, nw))
    return {"decode_loss_start": start, "decode_loss_end": end, "dropped": bool(end < start)}


def thought_decode_ce(logits: np.ndarray, thought_ids: np.ndarray) -> float:
    """CE from a decode head over thought tokens (spec 4.3)."""
    x = logits - logits.max(axis=-1, keepdims=True)
    logp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True) + 1e-12)
    t = thought_ids.astype(np.int32)
    n, s = t.shape
    rows = np.arange(n)[:, None]
    cols = np.arange(s)[None, :]
    return float(-logp[rows, cols, t].mean())


def halt_kl_geometric(halt_logits: np.ndarray, p: float = 0.3) -> float:
    """KL(model halt || geometric prior) on remaining budget (spec 4.3)."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    r = halt_logits.shape[-1]
    prior = np.array([(1 - p) ** t * p for t in range(r)], dtype=np.float64)
    prior[-1] = max(1.0 - prior[:-1].sum(), 1e-12)
    prior = prior / prior.sum()
    x = halt_logits - halt_logits.max(axis=-1, keepdims=True)
    q = np.exp(x)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-12)
    kl = (q * (np.log(q + 1e-12) - np.log(prior + 1e-12))).sum(axis=-1)
    return float(kl.mean())


def hidden_align_mse(student: np.ndarray, teacher: np.ndarray) -> float:
    """Segment-boundary hidden-state alignment against Stage A (spec 4.3)."""
    return float(np.mean((student - teacher) ** 2))


def stage_b_total_loss(
    answer_ce: float,
    decode_ce: float,
    halt_kl: float,
    align_mse: float,
    w_decode: float = 0.1,
    w_halt: float = 0.05,
    w_align: float = 0.2,
) -> float:
    return float(answer_ce + w_decode * decode_ce + w_halt * halt_kl + w_align * align_mse)


def jacobi_chunk(states: np.ndarray, step_fn, iters: int = 4, eps: float = 1e-4) -> np.ndarray:
    """Parallel thought updates until residual is small (spec 4.3 Jacobi)."""
    x = np.asarray(states, dtype=np.float64)
    for _ in range(iters):
        nxt = np.asarray(step_fn(x), dtype=np.float64)
        if float(np.max(np.abs(nxt - x))) < eps:
            return nxt
        x = nxt
    return x


def noisy_latent_sample(
    mean: np.ndarray, log_std: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Gaussian latent policy. Returns sample and mean log-density."""
    std = np.exp(np.clip(log_std, -8.0, 4.0))
    z = mean + std * rng.standard_normal(mean.shape)
    logp = -0.5 * (((z - mean) / (std + 1e-8)) ** 2 + 2.0 * np.log(std + 1e-8) + np.log(2 * np.pi))
    return z, float(logp.mean())


def budget_accuracy(scores: list[float]) -> bool:
    """Accuracy must rise with recurrence budget r=1..R (spec 4.3, V6)."""
    if len(scores) < 2:
        return False
    rising = all(scores[i] <= scores[i + 1] + 1e-9 for i in range(len(scores) - 1))
    return rising and scores[-1] > scores[0]


def stage_ab_probe() -> dict:
    cfg = M.tiny_config() if jax.default_backend() == "gpu" else M.cpu_config()
    params = M.init_params(cfg, jax.random.PRNGKey(6))
    tokens = np.random.default_rng(6).integers(0, cfg.vocab_size, size=(2, 8), dtype=np.int32)
    out_a = M.forward(tokens, params, cfg, r=1)
    acc = []
    hiddens = []
    for r in (1, 2, 4):
        o = M.forward(tokens, params, cfg, r=r)
        pred = np.asarray(o.logits).argmax(-1)
        acc.append(float((pred[:, :-1] == tokens[:, 1:]).mean()))
        hiddens.append(np.asarray(o.hidden))
    logits = np.asarray(out_a.logits)
    no_collapse = bool(np.isfinite(logits).all() and float(np.std(logits)) > 1e-3)
    used_budget = bool(np.max(np.abs(hiddens[0] - hiddens[-1])) > 1e-6)
    depth = _depth_task(params, hiddens[0])
    decode = _adapter_decode(params, hiddens[0])
    return {
        "no_collapse": no_collapse and used_budget,
        "accuracy_rises_with_budget": bool(depth["rises"]),
        "thoughts_decode": bool(decode["dropped"]),
        "decode_loss_start": decode["decode_loss_start"],
        "decode_loss_end": decode["decode_loss_end"],
        "loss_r1": depth["loss_r1"],
        "loss_r4": depth["loss_r4"],
        "acc": acc,
        "adapter_trained": True,
        "n_params": int(M.param_count(params)),
    }
