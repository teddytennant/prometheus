"""S1 rung-1, lab v0 cycle, Stage A/B losses (spec 4, 6, 14.6)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from model.config import validate_config
from train.lab import LabCycle, planted_cycle
from train.latent import (
    budget_accuracy,
    halt_kl_geometric,
    hidden_align_mse,
    jacobi_chunk,
    noisy_latent_sample,
    stage_b_total_loss,
    thought_decode_ce,
)
from train.rung1 import mup_lr, rung1_gpu_config
from train.rungs import RUNG1, gpu_config, table


def test_rung1_gpu_config_hybrid():
    cfg = rung1_gpu_config()
    validate_config(cfg)
    assert cfg.n_linear_attn == 3 * cfg.n_mla
    assert cfg.n_dense + cfg.n_moe == cfg.n_layers
    assert cfg.prelude_layers + cfg.core_block_layers + cfg.coda_layers == cfg.n_layers
    assert gpu_config(1).d_model == cfg.d_model
    assert table()[1].tokens == RUNG1.tokens


def test_mup_lr_scales_inverse_width():
    assert abs(mup_lr(1e-3, 1536, 768) - 5e-4) < 1e-12
    assert mup_lr(1e-3, 768, 768) == 1e-3


def test_rung1_tiny_loop(tmp_path: Path):
    os.environ["RUNG1_TINY"] = "1"
    os.environ["RUNG1_MAX_STEPS"] = "2"
    from train.rung1 import run

    out = run(tmp_path / "ckpt", tokens_target=32, max_steps=2, batch=2, seq=8, seed=0)
    assert out["rung"] == 1
    assert out["tiny"] is True
    assert out["n_params"] > 0
    assert (tmp_path / "ckpt").exists()
    os.environ.pop("RUNG1_TINY", None)
    os.environ.pop("RUNG1_MAX_STEPS", None)


def test_lab_cycle_fail_closed_without_replicate(tmp_path: Path):
    lab = LabCycle(tmp_path / "ledger.sqlite")
    lab.pre_register("x", "idea", "method", "win", "cpu")
    lab.run_experiment("x", lambda: {"score": 1.0, "baseline": 0.0, "held_out": 1.0})
    row = lab.record("x")
    assert row["recorded"] == "negative"
    assert row["replicated"] is False


def test_planted_cycle(tmp_path: Path):
    out = planted_cycle(tmp_path / "ledger.sqlite")
    assert out["planted_positive_replicated"] is True
    assert out["planted_negative_recorded"] is True
    assert out["n_experiments"] == 2


def test_stage_b_losses():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((2, 4, 8))
    ids = rng.integers(0, 8, size=(2, 4))
    ce = thought_decode_ce(logits, ids)
    assert ce > 0
    halt = rng.standard_normal((3, 8))
    kl = halt_kl_geometric(halt, p=0.3)
    assert kl >= 0
    align = hidden_align_mse(np.ones((2, 4)), np.zeros((2, 4)))
    assert abs(align - 1.0) < 1e-9
    total = stage_b_total_loss(1.0, ce, kl, align)
    assert total > 1.0


def test_jacobi_and_noisy_latent():
    def step(x):
        return 0.5 * x

    out = jacobi_chunk(np.ones(4), step, iters=20, eps=1e-8)
    assert float(np.max(np.abs(out))) < 1e-6
    rng = np.random.default_rng(1)
    z, logp = noisy_latent_sample(np.zeros(3), np.zeros(3), rng)
    assert z.shape == (3,)
    assert np.isfinite(logp)
    assert budget_accuracy([0.1, 0.2, 0.4]) is True
    assert budget_accuracy([0.4, 0.3]) is False
