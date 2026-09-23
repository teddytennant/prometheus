"""Oracle tests for SPMD circular pipeline placement (spec 5.1, A4-pp-place).

Production ``parallel.spmd_circular_pipeline`` is a stub
(``NotImplementedError``). Every test calls it and must fail on the stub.
Do not skip, xfail, or mark ``gpu``.

The function is the Praxis placement: ``jax.lax.scan`` of the circular
schedule inside ``jax.shard_map``, activation moved with ``jax.lax.ppermute``
on mesh axis ``"pp"``. A host loop that calls ``circular_pipeline`` is not an
implementation. These tests check the numerical contract and the faults the
docstring names. They do not import JAX, so they stay valid in a one-device
process. Device count and the jaxpr ``ppermute`` are the implementer's job
under ``XLA_FLAGS=--xla_force_host_platform_device_count=N``.

``stage_fn(x, stage)`` keeps the activation shape. Stage ids in the reference
are Python ints. Floating compare is ``rtol=1e-5``, ``atol=1e-5``. Integer
compare is bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest

import parallel
from tests.reference.placement import sequential_compose


def _run(microbatches, n_stages, stage_fn):
    return parallel.spmd_circular_pipeline(
        microbatches, n_stages=n_stages, stage_fn=stage_fn
    )


def _close(got, expected) -> None:
    g = np.asarray(got)
    e = np.asarray(expected)
    assert g.shape == e.shape
    assert g.dtype == e.dtype
    if np.issubdtype(e.dtype, np.floating):
        assert np.allclose(g, e, rtol=1e-5, atol=1e-5)
    else:
        assert g.tobytes() == e.tobytes()


def test_integer_two_stages_matches_sequential() -> None:
    mb = np.arange(12, dtype=np.int32).reshape(3, 4)

    def stage_fn(x, stage):
        return x + np.int32(stage + 1)

    got = _run(mb, 2, stage_fn)
    _close(got, sequential_compose(mb, 2, stage_fn))


def test_literal_two_stage_golden() -> None:
    mb = np.array([[1, 2], [3, 4]], dtype=np.int32)

    def stage_fn(x, stage):
        return x * np.int32(10) + np.int32(stage)

    got = _run(mb, 2, stage_fn)
    # stage 0: x*10+0, stage 1: that*10+1
    expected = np.array([[101, 201], [301, 401]], dtype=np.int32)
    _close(got, expected)


def test_float32_three_stages() -> None:
    mb = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(2, 4)

    def stage_fn(x, stage):
        return x * np.float32(0.5) + np.float32(stage)

    got = _run(mb, 3, stage_fn)
    _close(got, sequential_compose(mb, 3, stage_fn))


def test_single_stage_keeps_shape() -> None:
    mb = np.arange(6, dtype=np.int64).reshape(2, 3)

    def stage_fn(x, stage):
        return x + np.int64(7)

    got = _run(mb, 1, stage_fn)
    expected = mb + np.int64(7)
    _close(got, expected)
    assert np.asarray(got).shape == (2, 3)


def test_does_not_mutate_microbatches() -> None:
    mb = np.array([[1, 2], [3, 4]], dtype=np.int32)
    original = mb.copy()

    def stage_fn(x, stage):
        return np.asarray(x) + np.int32(1)

    _run(mb, 2, stage_fn)
    assert mb.tobytes() == original.tobytes()


def test_n_stages_not_int_is_mesh_error() -> None:
    mb = np.ones((1, 2), dtype=np.float32)
    with pytest.raises(parallel.MeshError):
        _run(mb, 1.5, lambda x, s: x)


def test_bool_n_stages_is_mesh_error() -> None:
    mb = np.ones((1, 2), dtype=np.float32)
    with pytest.raises(parallel.MeshError):
        _run(mb, True, lambda x, s: x)


def test_n_stages_below_one_is_mesh_error() -> None:
    mb = np.ones((1, 2), dtype=np.float32)
    with pytest.raises(parallel.MeshError):
        _run(mb, 0, lambda x, s: x)


def test_empty_leading_axis_is_mesh_error() -> None:
    mb = np.zeros((0, 2), dtype=np.float32)
    with pytest.raises(parallel.MeshError):
        _run(mb, 1, lambda x, s: x)


def test_non_array_is_mesh_error() -> None:
    with pytest.raises(parallel.MeshError):
        _run([[1, 2]], 1, lambda x, s: x)


def test_stage_fn_not_callable_is_mesh_error() -> None:
    mb = np.ones((1, 2), dtype=np.float32)
    with pytest.raises(parallel.MeshError):
        _run(mb, 1, "not-a-function")


def test_shape_change_is_mesh_error() -> None:
    mb = np.ones((1, 4), dtype=np.float32)

    def stage_fn(x, stage):
        return np.asarray(x).reshape(2, 2)

    with pytest.raises(parallel.MeshError):
        _run(mb, 1, stage_fn)
