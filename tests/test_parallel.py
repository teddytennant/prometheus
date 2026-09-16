"""Mesh rules (A4)."""

from __future__ import annotations

import pytest

from parallel import Mesh, parallel_equivalence, validate_mesh


def test_mesh_product():
    m = Mesh(ep=2, fsdp=2, pp=1, dp=1, cp=1)
    assert m.size() == 4
    validate_mesh(m)


def test_s3_ep_rejected():
    with pytest.raises(ValueError, match="S3"):
        validate_mesh(Mesh(ep=72))


def test_equivalence_identity_mesh():
    r = parallel_equivalence()
    assert r["routing_identical"]
    assert r["relative_loss_err"] < 1e-6
