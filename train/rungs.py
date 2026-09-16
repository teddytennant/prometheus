"""Rung 0 and 1 configs plus a tiny scaling-law fit (spec 6, A7, I1)."""

from __future__ import annotations

from dataclasses import dataclass

import model as M


@dataclass(frozen=True)
class Rung:
    index: int
    active_params: float
    total_params: float
    tokens: float


RUNG0 = Rung(0, 1.5e8, 1.5e9, 2e10)
RUNG1 = Rung(1, 1e9, 1.5e10, 2e11)


def rung_config(rung: Rung) -> M.ModelConfig:
    """CPU-testable configs. GPU shapes live in train.rung0 / train.rung1."""
    if rung.index == 0:
        return M.tiny_config()
    if rung.index == 1:
        return M.tiny_config()
    raise ValueError(rung.index)


def gpu_config(rung: int) -> M.ModelConfig:
    if rung == 0:
        from train.rung0 import rung0_gpu_config

        return rung0_gpu_config()
    if rung == 1:
        from train.rung1 import rung1_gpu_config

        return rung1_gpu_config()
    raise ValueError(rung)


def table() -> list[Rung]:
    return [RUNG0, RUNG1, Rung(2, 8e9, 1.2e11, 1.5e12), Rung(3, 4e10, 7e11, 6e12)]


def chinchilla_loss(n_params: float, n_tokens: float, a=6.49, b=7.2, e=0.3) -> float:
    """Kaplan/Hoffmann-style fit used as the ladder prior."""
    return float(e + (n_params / 1e9) ** (-a / 10) + (n_tokens / 1e9) ** (-b / 10))


def rung0_tiny_fit() -> dict:
    """CPU overfit slope vs the ladder prior. Not a V5 result."""
    from train import overfit_one_batch

    fit = overfit_one_batch(steps=20)
    prior = chinchilla_loss(RUNG0.active_params, 20 * 4 * 8)
    return {
        "loss_matches_ladder": bool(fit["ok"]),
        "ckpt_resume_across_jobs": False,
        "tokens_seen": 20 * 4 * 8,
        "tiny": True,
        "losses": [fit["loss_start"], fit["loss_end"]],
        "prior": prior,
        "rung0_config": rung_config(RUNG0).n_layers,
    }
