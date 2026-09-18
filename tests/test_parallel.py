"""Oracle tests for A4 ``parallel/`` (spec 5.1-5.2, 15.5 A4).

Constants, enums, dataclasses, ``flagship_mesh``, and ``tiny_mesh`` may pass
against the committed stubs. Every test that calls ``validate_mesh``,
``device_ids``, ``coord_of``, ``fsdp_partition``, ``pipeline_stage_layers``,
``context_parallel_on``, ``ep_dispatch``, or ``ep_combine`` must FAIL on the
stub with ``NotImplementedError``. GPU checks are marked ``gpu`` (A4 V2);
CPU runs skip them.

Coverage
--------
flagship_mesh: EP=72, FSDP=72, PP=12, DP=115, CP=1, n_devices=864*115.
tiny_mesh: 8 devices, ep=fsdp=2, pp=2, dp=2, cp=1.
validate_mesh: accepts tiny and flagship; rejects non-positive counts;
    rejects fsdp != ep.
device_ids / coord_of: every device unique; coord round-trip; FSDP index
    equals EP. C-order layout (dp, pp, ep, cp) with CP fastest.
fsdp_partition: ATTENTION/DENSE/SHARED_EXPERT shard on fsdp; ROUTED_EXPERT
    replicate (None,).
pipeline_stage_layers: covers [0, n_layers) without overlap; FLOP-balanced
    using recurrence on the last core_block_layers unique layers (later
    stages get fewer unique layers because they run r times); n_pp=12 with
    n_layers=60, core_block_layers=8, recurrence=4 (flagship-ish); tiny
    n_pp=2.
context_parallel_on: False when seq_len <= max_context_pretrain, True when
    greater.
ep_dispatch / ep_combine: wrap kernels.ep_dispatch / kernels.ep_combine;
    vs reference on a tiny DispatchMeta (A3 contract).
MeshError on bad layouts.

Frozen goldens live in this file and were computed from
``tests.reference.parallel``, not from production.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import kernels
import model
import parallel
from tests.reference import kernels as ref_kernels
from tests.reference import parallel as ref

TOL = dict(rtol=1e-5, atol=1e-5)

# Locked from tests.reference.parallel.coord_of on tiny_mesh.
GOLDEN_TINY_COORDS = (
    (0, 0, 0, 0),
    (0, 0, 1, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (1, 0, 0, 0),
    (1, 0, 1, 0),
    (1, 1, 0, 0),
    (1, 1, 1, 0),
)

# Locked from tests.reference.parallel.coord_of on flagship_mesh.
GOLDEN_FLAGSHIP_COORDS = {
    0: (0, 0, 0, 0),
    71: (0, 0, 71, 0),
    72: (0, 1, 0, 0),
    863: (0, 11, 71, 0),
    864: (1, 0, 0, 0),
    99359: (114, 11, 71, 0),
}

# Locked from tests.reference.parallel.pipeline_stage_layers.
GOLDEN_PP_TINY = ((0, 6), (6, 8))
GOLDEN_PP_FLAGSHIP_ISH = (
    (0, 7),
    (7, 14),
    (14, 21),
    (21, 28),
    (28, 35),
    (35, 42),
    (42, 49),
    (49, 53),
    (53, 55),
    (55, 56),
    (56, 58),
    (58, 60),
)
GOLDEN_PP_FLAGSHIP = (
    (0, 9),
    (9, 19),
    (19, 28),
    (28, 37),
    (37, 47),
    (47, 56),
    (56, 65),
    (65, 75),
    (75, 84),
    (84, 90),
    (90, 93),
    (93, 96),
)

# Locked from tests.reference.parallel.ep_dispatch (A3 tiny layout).
GOLDEN_EP_DISPATCHED = np.array(
    [[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [0.0, 0.0]]],
    dtype=np.float32,
)


def _np(x):
    return np.asarray(x)


def _require_gpu():
    jax = pytest.importorskip("jax")
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "tpu")]
    if not gpus:
        pytest.skip("V2 GPU test requires a GPU device")
    return jax


def _pairs(ranges) -> tuple[tuple[int, int], ...]:
    return tuple((int(r.start), int(r.end)) for r in ranges)


def _assert_cover(ranges, n_layers: int, n_pp: int) -> None:
    assert len(ranges) == n_pp
    assert ranges[0].start == 0
    assert ranges[-1].end == n_layers
    for i, r in enumerate(ranges):
        assert isinstance(r, parallel.StageRange)
        assert r.start < r.end
        if i + 1 < len(ranges):
            assert r.end == ranges[i + 1].start


def _assert_core_gets_fewer_unique(
    ranges, n_layers: int, core_block_layers: int
) -> None:
    if core_block_layers <= 0:
        return
    core_start = n_layers - core_block_layers
    non_core = []
    core = []
    for r in ranges:
        unique = r.end - r.start
        if r.end > core_start and r.start < n_layers:
            core.append(unique)
        else:
            non_core.append(unique)
    if non_core and core:
        assert max(core) <= min(non_core)


def _dispatch_meta(expert_ids, probs, racks, n_experts, max_racks=None):
    if max_racks is None:
        max_racks = kernels.MAX_RACKS
    return kernels.DispatchMeta(
        expert_ids=np.asarray(expert_ids, dtype=np.int32),
        probs=np.asarray(probs, dtype=np.float32),
        racks=np.asarray(racks, dtype=np.int32),
        n_experts=int(n_experts),
        max_racks=int(max_racks),
    )


def _tiny_ep_meta():
    tokens = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1], [0]],
        probs=[[1.0], [1.0], [1.0]],
        racks=[0, 1],
        n_experts=2,
    )
    return tokens, meta


# ---------------------------------------------------------------------------
# constants / dataclasses / mesh constructors (may pass against the stub)
# ---------------------------------------------------------------------------


def test_flagship_axis_constants():
    assert parallel.FLAGSHIP_EP == 72
    assert parallel.FLAGSHIP_FSDP == 72
    assert parallel.FLAGSHIP_PP == 12
    assert parallel.FLAGSHIP_DP == 115
    assert parallel.FLAGSHIP_CP_PRETRAIN == 1
    assert parallel.FLAGSHIP_GPUS_PER_REPLICA == 72 * 12
    assert parallel.FLAGSHIP_GPUS_PER_REPLICA == 864


def test_axis_enum_values():
    assert parallel.Axis.DP == "dp"
    assert parallel.Axis.FSDP == "fsdp"
    assert parallel.Axis.EP == "ep"
    assert parallel.Axis.PP == "pp"
    assert parallel.Axis.CP == "cp"
    assert set(parallel.Axis) == {
        parallel.Axis.DP,
        parallel.Axis.FSDP,
        parallel.Axis.EP,
        parallel.Axis.PP,
        parallel.Axis.CP,
    }


def test_param_kind_enum_values():
    assert parallel.ParamKind.ATTENTION == "attention"
    assert parallel.ParamKind.DENSE == "dense"
    assert parallel.ParamKind.SHARED_EXPERT == "shared_expert"
    assert parallel.ParamKind.ROUTED_EXPERT == "routed_expert"
    assert parallel.ParamKind.OTHER == "other"


def test_mesh_error_is_value_error():
    assert issubclass(parallel.MeshError, ValueError)


def test_mesh_spec_n_devices_and_frozen():
    spec = parallel.MeshSpec(dp=2, fsdp=2, ep=2, pp=2, cp=1)
    assert spec.n_devices == 8
    with pytest.raises(FrozenInstanceError):
        spec.dp = 3  # type: ignore[misc]


def test_stage_range_and_partition_spec_frozen():
    sr = parallel.StageRange(start=0, end=4)
    assert sr.start == 0 and sr.end == 4
    with pytest.raises(FrozenInstanceError):
        sr.start = 1  # type: ignore[misc]
    ps = parallel.PartitionSpec(axes=("fsdp",))
    assert ps.axes == ("fsdp",)
    with pytest.raises(FrozenInstanceError):
        ps.axes = (None,)  # type: ignore[misc]


def test_flagship_mesh_counts():
    mesh = parallel.flagship_mesh()
    assert mesh.ep == 72
    assert mesh.fsdp == 72
    assert mesh.pp == 12
    assert mesh.dp == 115
    assert mesh.cp == 1
    assert mesh.n_devices == 864 * 115
    assert mesh.n_devices == (
        parallel.FLAGSHIP_DP
        * parallel.FLAGSHIP_PP
        * parallel.FLAGSHIP_EP
        * parallel.FLAGSHIP_CP_PRETRAIN
    )
    assert mesh.fsdp == mesh.ep


def test_tiny_mesh_counts():
    mesh = parallel.tiny_mesh()
    assert mesh.n_devices == 8
    assert mesh.ep == 2
    assert mesh.fsdp == 2
    assert mesh.pp == 2
    assert mesh.dp == 2
    assert mesh.cp == 1
    assert mesh.n_devices == mesh.dp * mesh.pp * mesh.ep * mesh.cp
    assert mesh.fsdp == mesh.ep


# ---------------------------------------------------------------------------
# validate_mesh
# ---------------------------------------------------------------------------


def test_validate_mesh_accepts_tiny_and_flagship():
    assert parallel.validate_mesh(parallel.tiny_mesh()) is None
    assert parallel.validate_mesh(parallel.flagship_mesh()) is None
    assert ref.validate_mesh(parallel.tiny_mesh()) is None
    assert ref.validate_mesh(parallel.flagship_mesh()) is None


@pytest.mark.parametrize("field", ["dp", "fsdp", "ep", "pp", "cp"])
@pytest.mark.parametrize("bad", [0, -1])
def test_validate_mesh_rejects_non_positive_counts(field, bad):
    kwargs = {"dp": 2, "fsdp": 2, "ep": 2, "pp": 2, "cp": 1, field: bad}
    # Keep fsdp == ep so the only fault is a non-positive count.
    if field == "ep":
        kwargs["fsdp"] = bad
    if field == "fsdp":
        kwargs["ep"] = bad
    spec = parallel.MeshSpec(**kwargs)
    with pytest.raises(parallel.MeshError):
        parallel.validate_mesh(spec)


def test_validate_mesh_rejects_fsdp_ne_ep():
    spec = replace(parallel.tiny_mesh(), fsdp=1)
    assert spec.fsdp != spec.ep
    with pytest.raises(parallel.MeshError):
        parallel.validate_mesh(spec)
    spec = replace(parallel.flagship_mesh(), fsdp=36)
    with pytest.raises(parallel.MeshError):
        parallel.validate_mesh(spec)


def test_validate_mesh_accepts_custom_ep_equals_fsdp():
    spec = parallel.MeshSpec(dp=1, fsdp=4, ep=4, pp=1, cp=2)
    assert spec.n_devices == 8
    assert parallel.validate_mesh(spec) is None


# ---------------------------------------------------------------------------
# device_ids / coord_of
# ---------------------------------------------------------------------------


def test_tiny_device_ids_unique_and_coord_round_trip():
    spec = parallel.tiny_mesh()
    ids = parallel.device_ids(spec)
    assert ids == tuple(range(8))
    assert ids == ref.device_ids(spec)
    assert len(set(ids)) == spec.n_devices
    coords = []
    for d in ids:
        got = parallel.coord_of(d, spec)
        exp = ref.coord_of(d, spec)
        assert got == exp
        assert got == GOLDEN_TINY_COORDS[d]
        dp, pp, ep, cp = got
        assert 0 <= dp < spec.dp
        assert 0 <= pp < spec.pp
        assert 0 <= ep < spec.ep
        assert 0 <= cp < spec.cp
        assert ep < spec.fsdp
        assert ref.device_from_coord(dp, pp, ep, cp, spec) == d
        coords.append(got)
    assert len(set(coords)) == spec.n_devices


def test_flagship_device_ids_unique_and_coord_round_trip():
    spec = parallel.flagship_mesh()
    ids = parallel.device_ids(spec)
    assert ids == tuple(range(864 * 115))
    assert len(ids) == spec.n_devices
    assert len(set(ids)) == spec.n_devices
    seen = set()
    for d in ids:
        dp, pp, ep, cp = parallel.coord_of(d, spec)
        assert (dp, pp, ep, cp) == ref.coord_of(d, spec)
        assert 0 <= dp < spec.dp
        assert 0 <= pp < spec.pp
        assert 0 <= ep < spec.ep == spec.fsdp
        assert cp == 0
        assert ref.device_from_coord(dp, pp, ep, cp, spec) == d
        seen.add((dp, pp, ep, cp))
    assert len(seen) == spec.n_devices
    for device, exp in GOLDEN_FLAGSHIP_COORDS.items():
        assert parallel.coord_of(device, spec) == exp


def test_coord_of_fsdp_index_equals_ep():
    for spec in (
        parallel.tiny_mesh(),
        parallel.flagship_mesh(),
        parallel.MeshSpec(dp=2, fsdp=2, ep=2, pp=2, cp=2),
    ):
        for d in parallel.device_ids(spec):
            dp, pp, ep, cp = parallel.coord_of(d, spec)
            # Four-tuple has no independent FSDP axis; EP is the FSDP index.
            assert ep == parallel.coord_of(d, spec)[2]
            assert spec.fsdp == spec.ep
            assert 0 <= ep < spec.fsdp


def test_coord_of_cp_changes_fastest():
    spec = parallel.MeshSpec(dp=2, fsdp=2, ep=2, pp=2, cp=2)
    assert parallel.validate_mesh(spec) is None
    assert parallel.coord_of(0, spec) == (0, 0, 0, 0)
    assert parallel.coord_of(1, spec) == (0, 0, 0, 1)
    assert parallel.coord_of(2, spec) == (0, 0, 1, 0)
    assert parallel.coord_of(3, spec) == (0, 0, 1, 1)
    assert parallel.coord_of(4, spec) == (0, 1, 0, 0)
    for d in parallel.device_ids(spec):
        dp, pp, ep, cp = parallel.coord_of(d, spec)
        assert ref.device_from_coord(dp, pp, ep, cp, spec) == d
        assert (dp, pp, ep, cp) == ref.coord_of(d, spec)


def test_coord_of_and_device_ids_mesh_error_on_bad_layout():
    bad = replace(parallel.tiny_mesh(), fsdp=1)
    with pytest.raises(parallel.MeshError):
        parallel.device_ids(bad)
    with pytest.raises(parallel.MeshError):
        parallel.coord_of(0, bad)
    spec = parallel.tiny_mesh()
    with pytest.raises(parallel.MeshError):
        parallel.coord_of(-1, spec)
    with pytest.raises(parallel.MeshError):
        parallel.coord_of(spec.n_devices, spec)


# ---------------------------------------------------------------------------
# fsdp_partition
# ---------------------------------------------------------------------------


def test_fsdp_partition_shards_attention_dense_shared():
    for kind in (
        parallel.ParamKind.ATTENTION,
        parallel.ParamKind.DENSE,
        parallel.ParamKind.SHARED_EXPERT,
        parallel.ParamKind.OTHER,
    ):
        got = parallel.fsdp_partition(kind)
        assert isinstance(got, parallel.PartitionSpec)
        assert got.axes == (parallel.Axis.FSDP,)
        assert got.axes == ref.fsdp_partition(kind)


def test_fsdp_partition_replicates_routed_experts():
    got = parallel.fsdp_partition(parallel.ParamKind.ROUTED_EXPERT)
    assert isinstance(got, parallel.PartitionSpec)
    assert got.axes == (None,)
    assert got.axes == ref.fsdp_partition(parallel.ParamKind.ROUTED_EXPERT)


def test_fsdp_partition_every_param_kind():
    kinds = set(parallel.ParamKind)
    assert parallel.ParamKind.ROUTED_EXPERT in kinds
    for kind in kinds:
        got = parallel.fsdp_partition(kind)
        assert isinstance(got, parallel.PartitionSpec)
        assert got.axes == ref.fsdp_partition(kind)
        if kind is parallel.ParamKind.ROUTED_EXPERT:
            assert got.axes == (None,)
        else:
            assert got.axes == ("fsdp",)


# ---------------------------------------------------------------------------
# pipeline_stage_layers
# ---------------------------------------------------------------------------


def test_pipeline_stage_layers_tiny_n_pp_2():
    cfg = model.tiny_config()
    n_pp = parallel.tiny_mesh().pp
    got = parallel.pipeline_stage_layers(
        n_layers=cfg.n_layers,
        n_pp=n_pp,
        core_block_layers=cfg.core_block_layers,
        recurrence=3,
    )
    exp = ref.pipeline_stage_layers(
        cfg.n_layers, n_pp, cfg.core_block_layers, 3
    )
    _assert_cover(got, cfg.n_layers, n_pp)
    assert _pairs(got) == exp
    assert _pairs(got) == GOLDEN_PP_TINY
    _assert_core_gets_fewer_unique(got, cfg.n_layers, cfg.core_block_layers)
    assert (got[0].end - got[0].start) > (got[1].end - got[1].start)


def test_pipeline_stage_layers_flagship_ish_n_pp_12():
    n_layers, n_pp, core, r = 60, 12, 8, 4
    got = parallel.pipeline_stage_layers(n_layers, n_pp, core, r)
    exp = ref.pipeline_stage_layers(n_layers, n_pp, core, r)
    _assert_cover(got, n_layers, n_pp)
    assert _pairs(got) == exp
    assert _pairs(got) == GOLDEN_PP_FLAGSHIP_ISH
    _assert_core_gets_fewer_unique(got, n_layers, core)
    unique = [e - s for s, e in GOLDEN_PP_FLAGSHIP_ISH]
    assert unique[:7] == [7] * 7
    assert unique[-1] < unique[0]


def test_pipeline_stage_layers_flagship_96_vs_reference():
    n_layers = model.FLAGSHIP_N_LAYERS
    n_pp = parallel.FLAGSHIP_PP
    core = model.FLAGSHIP_CORE_BLOCK_LAYERS
    r = 3
    got = parallel.pipeline_stage_layers(n_layers, n_pp, core, r)
    exp = ref.pipeline_stage_layers(n_layers, n_pp, core, r)
    _assert_cover(got, n_layers, n_pp)
    assert _pairs(got) == exp
    assert _pairs(got) == GOLDEN_PP_FLAGSHIP
    _assert_core_gets_fewer_unique(got, n_layers, core)


def test_pipeline_stage_layers_n_pp_1_and_one_layer_each():
    got = parallel.pipeline_stage_layers(8, 1, 2, 3)
    _assert_cover(got, 8, 1)
    assert _pairs(got) == ref.pipeline_stage_layers(8, 1, 2, 3)
    assert _pairs(got) == ((0, 8),)
    got = parallel.pipeline_stage_layers(8, 8, 2, 3)
    _assert_cover(got, 8, 8)
    assert _pairs(got) == tuple((i, i + 1) for i in range(8))


def test_pipeline_stage_layers_recurrence_1_splits_counts():
    got = parallel.pipeline_stage_layers(10, 3, 2, 1)
    _assert_cover(got, 10, 3)
    assert _pairs(got) == ref.pipeline_stage_layers(10, 3, 2, 1)
    assert _pairs(got) == ((0, 3), (3, 7), (7, 10))


def test_pipeline_stage_layers_mesh_error_on_bad_layouts():
    with pytest.raises(parallel.MeshError):
        parallel.pipeline_stage_layers(8, 9, 2, 3)
    with pytest.raises(parallel.MeshError):
        parallel.pipeline_stage_layers(0, 1, 0, 1)
    with pytest.raises(parallel.MeshError):
        parallel.pipeline_stage_layers(8, 0, 2, 3)
    with pytest.raises(parallel.MeshError):
        parallel.pipeline_stage_layers(8, 2, 9, 3)
    with pytest.raises(parallel.MeshError):
        parallel.pipeline_stage_layers(8, 2, -1, 3)
    with pytest.raises(parallel.MeshError):
        parallel.pipeline_stage_layers(8, 2, 2, 0)


# ---------------------------------------------------------------------------
# context_parallel_on
# ---------------------------------------------------------------------------


def test_context_parallel_on_false_when_seq_at_or_below_pretrain():
    max_ctx = model.FLAGSHIP_MAX_CONTEXT
    assert max_ctx == 16_384
    assert parallel.context_parallel_on(max_ctx, max_ctx) is False
    assert parallel.context_parallel_on(max_ctx - 1, max_ctx) is False
    assert parallel.context_parallel_on(1, max_ctx) is False
    assert parallel.context_parallel_on(0, max_ctx) is False
    assert ref.context_parallel_on(max_ctx, max_ctx) is False


def test_context_parallel_on_true_when_seq_exceeds_pretrain():
    max_ctx = model.FLAGSHIP_MAX_CONTEXT
    assert parallel.context_parallel_on(max_ctx + 1, max_ctx) is True
    assert parallel.context_parallel_on(256_000, max_ctx) is True
    assert parallel.context_parallel_on(1_000_000, max_ctx) is True
    assert ref.context_parallel_on(256_000, max_ctx) is True


# ---------------------------------------------------------------------------
# ep_dispatch / ep_combine (wrap kernels; vs A3 + A4 references)
# ---------------------------------------------------------------------------


def test_ep_dispatch_tiny_meta_vs_reference_and_kernels():
    tokens, meta = _tiny_ep_meta()
    got_d, got_r = parallel.ep_dispatch(tokens, meta)
    ref_d, ref_r = ref.ep_dispatch(tokens, meta)
    kref_d, kref_r = ref_kernels.ep_dispatch(tokens, meta)
    kern_d, kern_r = kernels.ep_dispatch(tokens, meta)
    np.testing.assert_allclose(_np(got_d), ref_d, **TOL)
    np.testing.assert_allclose(_np(got_d), kref_d, **TOL)
    np.testing.assert_allclose(_np(got_d), _np(kern_d), **TOL)
    np.testing.assert_allclose(_np(got_d), GOLDEN_EP_DISPATCHED, **TOL)
    assert _np(got_d).shape == (2, 2, 2)
    assert _np(got_d).dtype == np.float32

    got_c = parallel.ep_combine(got_d, meta, got_r)
    ref_c = ref.ep_combine(ref_d, meta, ref_r)
    kref_c = ref_kernels.ep_combine(kref_d, meta, kref_r)
    kern_c = kernels.ep_combine(kern_d, meta, kern_r)
    np.testing.assert_allclose(_np(got_c), ref_c, **TOL)
    np.testing.assert_allclose(_np(got_c), kref_c, **TOL)
    np.testing.assert_allclose(_np(got_c), _np(kern_c), **TOL)
    np.testing.assert_allclose(_np(got_c), tokens, **TOL)
    wrap_c = kernels.ep_combine(got_d, meta, got_r)
    np.testing.assert_allclose(_np(wrap_c), tokens, **TOL)


def test_ep_dispatch_topk2_vs_reference():
    tokens = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
    )
    meta = _dispatch_meta(
        expert_ids=[[0, 1], [1, 0], [0, 0]],
        probs=[[0.7, 0.3], [0.4, 0.6], [1.0, 0.0]],
        racks=[0, 0],
        n_experts=2,
    )
    got_d, got_r = parallel.ep_dispatch(tokens, meta)
    ref_d, ref_r = ref.ep_dispatch(tokens, meta)
    kern_d, kern_r = kernels.ep_dispatch(tokens, meta)
    np.testing.assert_allclose(_np(got_d), ref_d, **TOL)
    np.testing.assert_allclose(_np(got_d), _np(kern_d), **TOL)
    got_c = parallel.ep_combine(got_d, meta, got_r)
    ref_c = ref.ep_combine(ref_d, meta, ref_r)
    np.testing.assert_allclose(_np(got_c), ref_c, **TOL)
    kern_c = kernels.ep_combine(kern_d, meta, kern_r)
    np.testing.assert_allclose(_np(got_c), _np(kern_c), **TOL)
    wrap_from_kern = parallel.ep_combine(kern_d, meta, kern_r)
    np.testing.assert_allclose(_np(wrap_from_kern), ref_c, **TOL)


def test_ep_combine_is_linear_in_expert_out():
    tokens, meta = _tiny_ep_meta()
    dispatched, residual = parallel.ep_dispatch(tokens, meta)
    y = _np(dispatched)
    c1 = _np(parallel.ep_combine(y, meta, residual))
    c2 = _np(parallel.ep_combine(2.0 * y, meta, residual))
    np.testing.assert_allclose(c2, 2.0 * c1, **TOL)
    # Finite-difference d(combine)/d(scale) at scale=1 equals combine(y).
    eps = 1e-3
    c_plus = _np(parallel.ep_combine((1.0 + eps) * y, meta, residual))
    fd = (c_plus - c1) / eps
    np.testing.assert_allclose(fd, c1, rtol=1e-3, atol=1e-3)


def test_ep_dispatch_kernel_error_on_rack_span():
    tokens = np.ones((1, 2), dtype=np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0, 1]],
        probs=[[0.5, 0.5]],
        racks=[0, 1],
        n_experts=2,
        max_racks=1,
    )
    with pytest.raises(kernels.KernelError):
        parallel.ep_dispatch(tokens, meta)


def test_ep_dispatch_shape_dtype():
    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((5, 4)).astype(np.float32)
    meta = _dispatch_meta(
        expert_ids=[[0], [1], [2], [0], [1]],
        probs=[[1.0], [1.0], [1.0], [1.0], [1.0]],
        racks=[0, 0, 1],
        n_experts=3,
    )
    dispatched, residual = parallel.ep_dispatch(tokens, meta)
    arr = _np(dispatched)
    assert arr.ndim == 3
    assert arr.shape[0] == 3
    assert arr.shape[2] == 4
    assert arr.dtype == np.float32
    combined = _np(parallel.ep_combine(dispatched, meta, residual))
    assert combined.shape == tokens.shape
    np.testing.assert_allclose(combined, tokens, **TOL)


# ---------------------------------------------------------------------------
# GPU (V2). Mesh math stays on CPU; this is the EP wrap on a device array.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v2_gpu_ep_dispatch_matches_reference():
    """V2: GPU ep_dispatch/ep_combine wrap vs numpy reference, 1e-5."""
    jax = _require_gpu()
    tokens, meta = _tiny_ep_meta()
    jtokens = jax.device_put(tokens)
    got_d, got_r = parallel.ep_dispatch(jtokens, meta)
    ref_d, ref_r = ref.ep_dispatch(tokens, meta)
    np.testing.assert_allclose(_np(got_d), ref_d, **TOL)
    got_c = parallel.ep_combine(got_d, meta, got_r)
    ref_c = ref.ep_combine(ref_d, meta, ref_r)
    np.testing.assert_allclose(_np(got_c), ref_c, **TOL)
