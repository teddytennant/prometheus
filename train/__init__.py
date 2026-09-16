"""Train package (spec 5): MuonClip, WSD, step, rung-0 loop."""

from __future__ import annotations

from train.loop import overfit_one_batch, v1_parity
from train.muon import OptState, init_opt_state, newton_schulz, qk_clip
from train.schedule import TrainConfig, wsd_lr
from train.step import cross_entropy, loss_fn, train_step, z_loss

__all__ = [
    "TrainConfig",
    "OptState",
    "newton_schulz",
    "qk_clip",
    "wsd_lr",
    "cross_entropy",
    "z_loss",
    "init_opt_state",
    "loss_fn",
    "train_step",
    "overfit_one_batch",
    "v1_parity",
]
