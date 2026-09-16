"""PP FLOP balance, CP ring, sharding (spec 5.2)."""

from __future__ import annotations

import numpy as np
import pytest

from parallel import (
    Mesh,
    context_parallel_ring,
    pipeline_stages,
    shard_params,
    unshard_params,
    validate_mesh,
)


def test_pipeline_core_gets_extra_stage_capacity():
    stages = pipeline_stages(12, pp=4, core_start=4, core_len=4, r=3.0)
    assert sum(len(s) for s in stages) == 12
    assert [i for s in stages for i in s] == list(range(12))
    core_lens = [len(s) for s in stages if any(4 <= i < 8 for i in s)]
    edge_lens = [len(s) for s in stages if all(i < 4 or i >= 8 for i in s)]
    assert core_lens and edge_lens
    assert max(core_lens) < max(edge_lens)


def test_context_parallel_ring_reconstructs():
    x = np.arange(16, dtype=np.float32).reshape(1, 16, 1)
    out = context_parallel_ring(x, cp=4)
    np.testing.assert_allclose(out["full"], x)
    assert out["ranks_agree"]
    assert out["n_ring_steps"] == 3


def test_s3_ep_rejected_on_construct():
    with pytest.raises(ValueError, match="S3"):
        Mesh(ep=72)


def test_shard_unshard_roundtrip():
    mesh = Mesh(fsdp=4)
    params = {"w": np.arange(16, dtype=np.float32)}
    sh = shard_params(params, mesh)
    assert sh["w"].shape[0] == 4
    back = unshard_params(sh, mesh, orig_len=16)
    np.testing.assert_array_equal(back["w"], params["w"])
    validate_mesh(Mesh(ep=8, fsdp=2, pp=2, dp=2, cp=2))
