"""Loss and one train step (spec 5)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, value_and_grad

from model import forward as model_forward
from train.muon import OptState, _named_update
from train.schedule import TrainConfig, wsd_lr


def cross_entropy(logits: Array, targets: Array) -> Array:
    log_z = jax.nn.logsumexp(logits, axis=-1)
    gather = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return (log_z - gather).mean()


def z_loss(logits: Array) -> Array:
    log_z = jax.nn.logsumexp(logits, axis=-1)
    return jnp.square(log_z).mean()


def loss_fn(params, tokens, config, train_cfg: TrainConfig | None = None):
    if train_cfg is None:
        train_cfg = TrainConfig()
    out = model_forward(tokens[:, :-1], params, config, r=1)
    ce = cross_entropy(out.logits, tokens[:, 1:])
    extra = 0.0
    if out.mtp_logits:
        for i, mtp in enumerate(out.mtp_logits, start=2):
            if tokens.shape[1] > i:
                extra = extra + cross_entropy(mtp[:, : tokens.shape[1] - i], tokens[:, i:])
        extra = extra / len(out.mtp_logits)
    zl = z_loss(out.logits)
    return ce + train_cfg.mtp_weight * extra + train_cfg.z_loss_weight * zl


def train_step(params, opt: OptState, tokens, model_cfg, train_cfg: TrainConfig):
    tokens = jnp.asarray(tokens)
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, model_cfg, train_cfg)
    lr = wsd_lr(opt.step, train_cfg)
    new_p, new_m = _named_update(params, grads, opt.momentum, lr, train_cfg)
    return new_p, OptState(step=opt.step + 1, momentum=new_m), loss


