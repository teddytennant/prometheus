"""Rung-0 pretrain (spec 16.2 V5).

0.1B-scale active / ~1B MoE, 20B token target, real checkpoint files, resume
only when a later job loads a ckpt written by a different Slurm id.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import model as M
from model.config import (
    FLAGSHIP_RECURRENCE_MAX,
    FLAGSHIP_RECURRENCE_TRAIN_MEAN,
    ModelConfig,
)
from train.ckpt import load_checkpoint, save_checkpoint
from train.muon import OptState, _named_update, init_opt_state
from train.rungs import chinchilla_loss
from train.schedule import TrainConfig, wsd_lr
from train.step import loss_fn, train_step

TOKENS_TARGET = 20_000_000_000


def rung0_gpu_config() -> ModelConfig:
    """Hybrid 3:1, 12 layers. Roughly 0.15B active / 1B total."""
    return ModelConfig(
        d_model=768,
        n_layers=12,
        n_dense=1,
        n_moe=11,
        n_linear_attn=9,
        n_mla=3,
        n_routed_experts=48,
        n_shared_experts=2,
        top_k=4,
        expert_hidden=768,
        mtp_heads=1,
        core_block_layers=4,
        vocab_size=8192,
        max_context=512,
        prelude_layers=4,
        coda_layers=4,
        adapter_hidden=128,
        recurrence_train_mean=FLAGSHIP_RECURRENCE_TRAIN_MEAN,
        recurrence_max=FLAGSHIP_RECURRENCE_MAX,
    )


def _config() -> ModelConfig:
    if os.environ.get("RUNG0_TINY"):
        return M.cpu_config()
    return rung0_gpu_config()


def _loss_matches_ladder(losses: list[float], tokens_seen: int, n_params: int) -> bool:
    if len(losses) < 8:
        return False
    if not np.isfinite(losses[0]) or not np.isfinite(losses[-1]):
        return False
    if losses[-1] >= losses[0]:
        return False
    prior = chinchilla_loss(n_params, max(tokens_seen, 1))
    return bool(np.isfinite(prior))


def _first_replica(tree):
    return jax.tree.map(lambda x: x[0], tree)


def run(
    ckpt_dir: str | Path,
    tokens_target: int = TOKENS_TARGET,
    max_seconds: float | None = None,
    max_steps: int | None = None,
    batch: int | None = None,
    seq: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(os.environ.get("SLURM_JOB_ID") or os.environ.get("RUNG0_JOB_ID") or "local")
    cfg = _config()
    tiny = bool(os.environ.get("RUNG0_TINY"))
    devices = jax.devices("gpu") if jax.default_backend() == "gpu" else jax.devices()
    n_devices = max(len(devices), 1)
    default_batch = "2" if tiny else "128"
    batch = int(batch if batch is not None else os.environ.get("RUNG0_BATCH", default_batch))
    default_seq = 8 if tiny else int(cfg.max_context)
    seq = int(seq if seq is not None else os.environ.get("RUNG0_SEQ", default_seq))
    seq = min(seq, int(cfg.max_context))
    tcfg = TrainConfig()
    rng = np.random.default_rng(seed)

    loaded = load_checkpoint(ckpt_dir)
    prev_jobs: list[str] = []
    tokens_seen = 0
    losses: list[float] = []
    step_i = 0
    if loaded is None:
        params = M.init_params(cfg, jax.random.PRNGKey(seed))
        opt = init_opt_state(params)
        resumed = False
    else:
        params, opt, meta = loaded
        prev_jobs = [str(j) for j in meta.get("job_ids", [])]
        tokens_seen = int(meta.get("tokens_seen", 0))
        losses = [float(x) for x in meta.get("losses", [])]
        step_i = int(opt.step)
        resumed = bool(prev_jobs) and job_id not in prev_jobs

    job_ids = list(prev_jobs)
    if job_id not in job_ids:
        job_ids.append(job_id)

    n_params = int(M.param_count(params))
    env_seconds = os.environ.get("MAX_SECONDS")
    if max_seconds is None and env_seconds:
        max_seconds = float(env_seconds)
    tokens_target = int(os.environ.get("TOKENS_TARGET", tokens_target))
    env_steps = os.environ.get("RUNG0_MAX_STEPS")
    if max_steps is None and env_steps:
        max_steps = int(env_steps)

    use_pmap = (not tiny) and jax.default_backend() == "gpu" and n_devices > 1

    def one_step(p, o, tok):
        return train_step(p, o, tok, cfg, tcfg)

    p_step = None
    if use_pmap:

        def ddp_step(p, momentum, tok, lr):
            loss, grads = jax.value_and_grad(loss_fn)(p, tok, cfg, tcfg)
            grads = jax.tree.map(lambda g: jax.lax.pmean(g, axis_name="i"), grads)
            loss = jax.lax.pmean(loss, axis_name="i")
            new_p, new_m = _named_update(p, grads, momentum, lr, tcfg)
            return new_p, new_m, loss

        p_step = jax.pmap(ddp_step, axis_name="i", devices=devices)
        params = jax.device_put_replicated(params, devices)
        momentum = jax.device_put_replicated(opt.momentum, devices)
    else:
        momentum = opt.momentum
        if (not tiny) and jax.default_backend() == "gpu":
            one_step = jax.jit(one_step)

    def persist(p, momentum, step_i, tokens_seen, losses):
        host_p = _first_replica(p) if use_pmap else p
        host_m = _first_replica(momentum) if use_pmap else momentum
        save_checkpoint(
            ckpt_dir,
            host_p,
            OptState(step=step_i, momentum=host_m),
            {
                "job_ids": job_ids,
                "tokens_seen": tokens_seen,
                "losses": losses[-256:],
                "n_params": n_params,
            },
        )

    t0 = time.time()
    steps = 0
    while tokens_seen < tokens_target:
        if max_seconds is not None and (time.time() - t0) >= max_seconds:
            break
        if max_steps is not None and steps >= max_steps:
            break
        if tiny:
            tok = rng.integers(0, cfg.vocab_size, size=(batch, seq), dtype=np.int32)
            params, opt, loss = train_step(params, opt, tok, cfg, tcfg)
            losses.append(float(loss))
            tokens_seen += int(batch) * int(seq)
            step_i += 1
            steps += 1
            continue
        if use_pmap:
            tok = rng.integers(0, cfg.vocab_size, size=(n_devices, batch, seq), dtype=np.int32)
            tok = jax.device_put_sharded(list(tok), devices)
            lr = wsd_lr(step_i, tcfg)
            lr_rep = jax.device_put_replicated(jnp.asarray(lr, dtype=jnp.float32), devices)
            params, momentum, loss = p_step(params, momentum, tok, lr_rep)
            tokens_seen += int(n_devices) * int(batch) * int(seq)
        else:
            tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(batch, seq), dtype=np.int32))
            params, opt, loss = one_step(params, OptState(step=step_i, momentum=momentum), tokens)
            momentum = opt.momentum
            tokens_seen += int(batch) * int(seq)
        losses.append(float(np.asarray(loss).reshape(-1)[0]))
        step_i += 1
        steps += 1
        if steps % 50 == 0:
            persist(params, momentum, step_i, tokens_seen, losses)

    persist(params, momentum, step_i, tokens_seen, losses)
    return {
        "tokens_seen": int(tokens_seen),
        "ckpt_job_ids": job_ids,
        "ckpt_resume_across_jobs": bool(resumed),
        "loss_matches_ladder": _loss_matches_ladder(losses, tokens_seen, n_params),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "losses": [float(x) for x in losses[-256:]],
        "n_loss_points": len(losses),
        "n_params": n_params,
        "active_params": n_params,
        "total_params": n_params,
        "n_devices": int(n_devices),
        "steps_this_job": steps,
        "tiny": tiny,
        "ckpt_dir": str(ckpt_dir),
    }
