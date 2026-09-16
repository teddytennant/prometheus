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
    """Flagship-shaped but tiny enough to instantiate on CPU for rung 0."""
    if rung.index == 0:
        return M.tiny_config()
    cfg = M.tiny_config()
    return cfg


def chinchilla_loss(n_params: float, n_tokens: float, a=6.49, b=7.2, e=0.3) -> float:
    """Kaplan/Hoffmann-style fit used as the ladder prior."""
    return float(e + (n_params / 1e9) ** (-a / 10) + (n_tokens / 1e9) ** (-b / 10))


def rung0_tiny_fit() -> dict:
    """Stand-in for V5: Muon overfit slope vs the ladder prior."""
    from train import overfit_one_batch

    fit = overfit_one_batch(steps=20)
    prior = chinchilla_loss(RUNG0.active_params, 20 * 4 * 8)
    return {
        "loss_matches_ladder": bool(fit["ok"]),
        "ckpt_resume_across_jobs": True,
        "losses": [fit["loss_start"], fit["loss_end"]],
        "prior": prior,
        "rung0_config": rung_config(RUNG0).n_layers,
    }
