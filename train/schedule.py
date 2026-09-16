"""WSD schedule and train hyperparams (spec 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    warmup: int = 10
    stable: int = 40
    decay: int = 20
    peak_lr: float = 3e-3
    weight_decay: float = 0.01
    qk_clip: float = 100.0
    ns_steps: int = 5
    z_loss_weight: float = 1e-4
    mtp_weight: float = 0.1
    muon_lr: float = 0.02




def wsd_lr(step: int, cfg: TrainConfig) -> float:
    """Warmup-stable-decay schedule. Stable-phase ckpts can decay later."""
    s = max(int(step), 0)
    if s < cfg.warmup:
        return cfg.peak_lr * (s / max(cfg.warmup, 1))
    if s < cfg.warmup + cfg.stable:
        return cfg.peak_lr
    t = s - cfg.warmup - cfg.stable
    return cfg.peak_lr * max(0.0, 1.0 - t / max(cfg.decay, 1))


