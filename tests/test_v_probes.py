"""CPU tests that V-stage probes are real package code, not toys."""

from pathlib import Path

import numpy as np

from kernels.ep import ep_moe_match
from kernels.quant import two_precision_train
from rl.loss import rl_end_to_end_probe
from train.latent import stage_ab_probe
from train.loop import _finite_diff_grad_check, v1_parity
from verify.ncshare.probes import fault_probe, lab_dry_run


def test_v1_parity_flagship_and_real_grad():
    p = v1_parity()
    assert p["flagship_shape"]
    assert p["n_params"] > 1e6
    assert p["grad_check"]
    assert _finite_diff_grad_check()


def test_ep_moe_not_identity():
    r = ep_moe_match(n_ep=2)
    assert r["n_ep"] == 2
    assert not r["identity_mesh"]
    assert r["relative_err"] < 1e-4


def test_two_precision_train_has_steps():
    r = two_precision_train(steps=8)
    assert r["n_steps"] == 8
    assert np.isfinite(r["fp8_vs_bf16_rel"])
    assert r["nvfp4_numerics_ok"]


def test_fault_probe_writes_ckpt(tmp_path):
    r = fault_probe()
    assert r["resume_bitwise_equal"]
    assert r["sdc_caught_flip"]
    assert r["n_param_leaves"] >= 2
    assert r["ckpt_bytes"] > 64
    assert Path(r["ckpt_path"]).exists()


def test_latent_decode_trains():
    r = stage_ab_probe()
    assert r["adapter_trained"]
    assert r["decode_loss_end"] < r["decode_loss_start"]
    assert r["thoughts_decode"]
    assert r["accuracy_rises_with_budget"]
    assert r["no_collapse"]


def test_rl_uses_gspo():
    r = rl_end_to_end_probe()
    assert r["used_gspo"]
    assert r["planted_write_flagged"]
    assert r["drift_halted"]


def test_lab_writes_sqlite():
    r = lab_dry_run()
    assert r["ledger_rows"] >= 2
    assert Path(r["ledger_path"]).exists()
    assert r["planted_positive_replicated"]
    assert r["planted_negative_recorded"]


def test_task_queue_idempotent(tmp_path):
    from harness.chaos import TaskQueue

    q = TaskQueue(tmp_path)
    a, st = q.run("t1", lambda: {"v": 1})
    b, st2 = q.run("t1", lambda: {"v": 2})
    assert st == "ok"
    assert st2 == "dup_prevented"
    assert b["v"] == 1
    (q.out / "x.tmp").write_text("{")
    assert q.recover_incomplete() == 1
