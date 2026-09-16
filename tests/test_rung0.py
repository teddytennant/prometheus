"""Rung-0 ckpt/resume across two job ids. Tiny config, CPU only."""

from __future__ import annotations

import os

from train.rung0 import run


def test_rung0_resume_two_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNG0_TINY", "1")
    monkeypatch.setenv("RUNG0_JOB_ID", "111")
    r1 = run(tmp_path, tokens_target=32, max_steps=2, batch=2, seq=8)
    assert r1["tokens_seen"] > 0
    assert (tmp_path / "params.npz").is_file()
    assert r1["ckpt_resume_across_jobs"] is False
    monkeypatch.setenv("RUNG0_JOB_ID", "222")
    r2 = run(tmp_path, tokens_target=64, max_steps=2, batch=2, seq=8)
    assert r2["ckpt_resume_across_jobs"] is True
    assert r2["ckpt_job_ids"] == ["111", "222"]
    assert r2["tokens_seen"] > r1["tokens_seen"]
    assert r2["tiny"] is True
