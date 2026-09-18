"""Implementer-facing tests for C2 SGLang model definition (spec 3.1, 13.1, 15.5).

These import production ``sglang_fork.model_def``. Constants and frozen
dataclass construction may pass against the stub. Every test that calls
``attention_kind``, ``ffn_kind``, ``in_core_block``, ``from_counts``,
``mtp_head_count``, or ``layer_table`` MUST FAIL on the stub with
``NotImplementedError``.

No GPU marker. Discrete architecture only (no tensors, no gradients).
C2 does not import production ``model``; tiny-scale integers are copied
from A1 ``tiny_config()``.

Groups:
- constants / interface: LINEAR_TO_MLA == 3, enum values linear/mla and
  dense/moe, ModelDefError is ValueError, frozen dataclasses.
- attention_kind: index 0 linear; 3:1 repeating; out of range raises
  ModelDefError.
- ffn_kind: first n_dense dense, rest moe; out of range raises.
- in_core_block: last core_block_layers unique layers True; OOB raises.
- from_counts: fills layers; n_dense+n_moe == n_layers;
  n_linear_attn+n_mla == n_layers; 3:1 via LINEAR_TO_MLA; mtp_heads must
  be 2; mismatch raises ModelDefError.
- layer_table: length == n_layers; matches attention_kind / ffn_kind /
  in_core_block; length mismatch raises.
- mtp_head_count == 2 on a valid definition.
- tiny-scale counts matching A1 tiny_config integers.
- flagship-ish: n_layers=60, n_dense=4, n_moe=56, 3:1 attention,
  core_block_layers=8, mtp_heads=2.
- properties: period-4 identity, count tallies, layer.index == i.
- edges / faults: all-dense, all-moe, core = 1 or n_layers, negative
  n_dense with compensating n_moe, core_block_layers out of range.
"""

from __future__ import annotations

import dataclasses

import pytest
from sglang_fork import model_def

from tests.reference import sglang_model_def as ref

# Copied from A1 model.tiny_config() integers. Do not import production model.
TINY = {
    "d_model": 128,
    "n_layers": 8,
    "n_dense": 1,
    "n_moe": 7,
    "n_linear_attn": 6,
    "n_mla": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 2,
    "top_k": 2,
    "expert_hidden": 256,
    "mtp_heads": 2,
    "core_block_layers": 2,
    "recurrence_train_mean": 3.0,
    "recurrence_max": 16,
    "vocab_size": 256,
    "max_context": 128,
    "max_racks": 4,
}

# Distinct stored fields so a swap would fail; 3:1 on 60 layers is 45:15.
FLAGSHIP_ISH = {
    "d_model": 1024,
    "n_layers": 60,
    "n_dense": 4,
    "n_moe": 56,
    "n_linear_attn": 45,
    "n_mla": 15,
    "n_routed_experts": 32,
    "n_shared_experts": 2,
    "top_k": 4,
    "expert_hidden": 512,
    "mtp_heads": 2,
    "core_block_layers": 8,
    "recurrence_train_mean": 3.0,
    "recurrence_max": 16,
    "vocab_size": 1024,
    "max_context": 2048,
    "max_racks": 4,
}

_SCALAR_FIELDS = (
    "d_model",
    "n_layers",
    "n_dense",
    "n_moe",
    "n_linear_attn",
    "n_mla",
    "n_routed_experts",
    "n_shared_experts",
    "top_k",
    "expert_hidden",
    "mtp_heads",
    "core_block_layers",
    "recurrence_train_mean",
    "recurrence_max",
    "vocab_size",
    "max_context",
    "max_racks",
)


def _kw(**overrides):
    fields = dict(TINY)
    fields.update(overrides)
    return fields


def _tiny_layers() -> tuple[model_def.SglangLayer, ...]:
    # 8 unique layers, 3:1, first dense, last 2 in the core block.
    spec = (
        (0, model_def.AttentionKind.LINEAR, model_def.FfnKind.DENSE, False),
        (1, model_def.AttentionKind.LINEAR, model_def.FfnKind.MOE, False),
        (2, model_def.AttentionKind.LINEAR, model_def.FfnKind.MOE, False),
        (3, model_def.AttentionKind.MLA, model_def.FfnKind.MOE, False),
        (4, model_def.AttentionKind.LINEAR, model_def.FfnKind.MOE, False),
        (5, model_def.AttentionKind.LINEAR, model_def.FfnKind.MOE, False),
        (6, model_def.AttentionKind.LINEAR, model_def.FfnKind.MOE, True),
        (7, model_def.AttentionKind.MLA, model_def.FfnKind.MOE, True),
    )
    return tuple(
        model_def.SglangLayer(index=i, attention=attn, ffn=ffn, in_core_block=core)
        for i, attn, ffn, core in spec
    )


def _tiny_defn(**overrides) -> model_def.SglangModelDef:
    fields = _kw(**overrides)
    return model_def.SglangModelDef(**fields)


def _assert_layer_vs_ref(got: object, exp: object) -> None:
    assert got.index == exp.index
    assert got.attention == exp.attention
    assert got.ffn == exp.ffn
    assert got.in_core_block == exp.in_core_block


def _assert_defn_vs_ref(got: object, exp: object) -> None:
    for name in _SCALAR_FIELDS:
        assert getattr(got, name) == getattr(exp, name), name
    assert len(got.layers) == len(exp.layers)
    for g, e in zip(got.layers, exp.layers, strict=True):
        _assert_layer_vs_ref(g, e)


def _assert_table_matches_kinds(
    table: tuple[model_def.SglangLayer, ...],
    n_layers: int,
    n_dense: int,
    core_block_layers: int,
) -> None:
    assert isinstance(table, tuple)
    assert len(table) == n_layers
    for i, layer in enumerate(table):
        assert isinstance(layer, model_def.SglangLayer)
        assert layer.index == i
        assert layer.attention is model_def.attention_kind(i, n_layers)
        assert layer.ffn is model_def.ffn_kind(i, n_dense, n_layers)
        assert layer.in_core_block == model_def.in_core_block(i, n_layers, core_block_layers)
        assert isinstance(layer.in_core_block, bool)


# ---------------------------------------------------------------------------
# constants / interface (may pass against the stub)
# ---------------------------------------------------------------------------


def test_linear_to_mla_is_three():
    assert model_def.LINEAR_TO_MLA == 3
    assert model_def.LINEAR_TO_MLA == ref.LINEAR_TO_MLA


def test_attention_kind_enum_values():
    assert model_def.AttentionKind.LINEAR == "linear"
    assert model_def.AttentionKind.MLA == "mla"
    assert set(model_def.AttentionKind) == {
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.MLA,
    }


def test_ffn_kind_enum_values():
    assert model_def.FfnKind.DENSE == "dense"
    assert model_def.FfnKind.MOE == "moe"
    assert set(model_def.FfnKind) == {
        model_def.FfnKind.DENSE,
        model_def.FfnKind.MOE,
    }


def test_model_def_error_is_value_error():
    assert issubclass(model_def.ModelDefError, ValueError)


def test_sglang_layer_is_frozen():
    layer = model_def.SglangLayer(
        index=0,
        attention=model_def.AttentionKind.LINEAR,
        ffn=model_def.FfnKind.DENSE,
        in_core_block=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        layer.index = 1  # type: ignore[misc]


def test_sglang_model_def_default_layers_empty():
    defn = _tiny_defn()
    assert defn.layers == ()
    assert defn.max_racks == 4


# ---------------------------------------------------------------------------
# attention_kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [4, 8, 60])
def test_attention_kind_index_zero_is_linear(n_layers: int):
    got = model_def.attention_kind(0, n_layers)
    assert got is model_def.AttentionKind.LINEAR
    assert got == ref.attention_kind(0, n_layers)


@pytest.mark.parametrize("n_layers", [4, 8, 60])
def test_attention_kind_three_to_one_repeating(n_layers: int):
    period = model_def.LINEAR_TO_MLA + 1
    assert period == 4
    for i in range(n_layers):
        got = model_def.attention_kind(i, n_layers)
        exp = ref.attention_kind(i, n_layers)
        assert got == exp
        if i % period == model_def.LINEAR_TO_MLA:
            assert got is model_def.AttentionKind.MLA
        else:
            assert got is model_def.AttentionKind.LINEAR


def test_attention_kind_tiny_explicit_sequence():
    n = TINY["n_layers"]
    want = [
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.MLA,
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.MLA,
    ]
    got = [model_def.attention_kind(i, n) for i in range(n)]
    assert got == want
    assert [ref.attention_kind(i, n) for i in range(n)] == want


@pytest.mark.parametrize("idx", [-1, 8, 9])
def test_attention_kind_out_of_range_raises(idx: int):
    with pytest.raises(model_def.ModelDefError):
        model_def.attention_kind(idx, 8)


def test_attention_kind_n_layers_zero_raises():
    with pytest.raises(model_def.ModelDefError):
        model_def.attention_kind(0, 0)


# ---------------------------------------------------------------------------
# ffn_kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_dense,n_layers", [(1, 8), (4, 60), (0, 8), (8, 8)])
def test_ffn_kind_first_n_dense_then_moe(n_dense: int, n_layers: int):
    for i in range(n_layers):
        got = model_def.ffn_kind(i, n_dense, n_layers)
        exp = ref.ffn_kind(i, n_dense, n_layers)
        assert got == exp
        if i < n_dense:
            assert got is model_def.FfnKind.DENSE
        else:
            assert got is model_def.FfnKind.MOE


def test_ffn_kind_tiny_explicit_sequence():
    n = TINY["n_layers"]
    n_dense = TINY["n_dense"]
    got = [model_def.ffn_kind(i, n_dense, n) for i in range(n)]
    assert got[0] is model_def.FfnKind.DENSE
    assert all(k is model_def.FfnKind.MOE for k in got[1:])
    assert got == [ref.ffn_kind(i, n_dense, n) for i in range(n)]


@pytest.mark.parametrize("idx", [-1, 8, 9])
def test_ffn_kind_out_of_range_raises(idx: int):
    with pytest.raises(model_def.ModelDefError):
        model_def.ffn_kind(idx, TINY["n_dense"], 8)


# ---------------------------------------------------------------------------
# in_core_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_layers,core",
    [(8, 2), (60, 8), (4, 1), (8, 8), (8, 1)],
)
def test_in_core_block_last_core_layers(n_layers: int, core: int):
    start = n_layers - core
    for i in range(n_layers):
        got = model_def.in_core_block(i, n_layers, core)
        exp = ref.in_core_block(i, n_layers, core)
        assert got == exp
        assert got is (i >= start)
        assert isinstance(got, bool)


def test_in_core_block_tiny_last_two():
    n = TINY["n_layers"]
    core = TINY["core_block_layers"]
    flags = [model_def.in_core_block(i, n, core) for i in range(n)]
    assert flags == [False, False, False, False, False, False, True, True]


@pytest.mark.parametrize("idx", [-1, 8, 9])
def test_in_core_block_out_of_range_raises(idx: int):
    with pytest.raises(model_def.ModelDefError):
        model_def.in_core_block(idx, 8, 2)


# ---------------------------------------------------------------------------
# from_counts
# ---------------------------------------------------------------------------


def test_from_counts_fills_layers_tiny():
    got = model_def.from_counts(**TINY)
    exp = ref.from_counts(**TINY)
    assert isinstance(got, model_def.SglangModelDef)
    _assert_defn_vs_ref(got, exp)
    assert len(got.layers) == TINY["n_layers"]
    _assert_table_matches_kinds(
        got.layers,
        TINY["n_layers"],
        TINY["n_dense"],
        TINY["core_block_layers"],
    )


def test_from_counts_stores_every_scalar():
    got = model_def.from_counts(**TINY)
    for name, value in TINY.items():
        assert getattr(got, name) == value, name


def test_from_counts_default_max_racks():
    fields = dict(TINY)
    fields.pop("max_racks")
    got = model_def.from_counts(**fields)
    assert got.max_racks == 4


def test_from_counts_n_dense_plus_n_moe_must_equal_n_layers():
    with pytest.raises(model_def.ModelDefError):
        model_def.from_counts(**_kw(n_dense=2, n_moe=7))


def test_from_counts_n_linear_plus_n_mla_must_equal_n_layers():
    with pytest.raises(model_def.ModelDefError):
        model_def.from_counts(**_kw(n_linear_attn=5, n_mla=2))


def test_from_counts_three_to_one_mismatch_raises():
    # Sum matches n_layers=8 but 5:3 is not 3:1.
    with pytest.raises(model_def.ModelDefError):
        model_def.from_counts(**_kw(n_linear_attn=5, n_mla=3))


@pytest.mark.parametrize("mtp_heads", [0, 1, 3, 4])
def test_from_counts_mtp_heads_must_be_two(mtp_heads: int):
    with pytest.raises(model_def.ModelDefError):
        model_def.from_counts(**_kw(mtp_heads=mtp_heads))


def test_from_counts_negative_n_dense_raises_even_if_sum_matches():
    with pytest.raises(model_def.ModelDefError):
        model_def.from_counts(**_kw(n_dense=-1, n_moe=9))


@pytest.mark.parametrize("core", [0, 9])
def test_from_counts_core_block_layers_out_of_range_raises(core: int):
    with pytest.raises(model_def.ModelDefError):
        model_def.from_counts(**_kw(core_block_layers=core))


# ---------------------------------------------------------------------------
# layer_table
# ---------------------------------------------------------------------------


def test_layer_table_length_and_kinds_tiny():
    defn = model_def.from_counts(**TINY)
    table = model_def.layer_table(defn)
    assert len(table) == TINY["n_layers"]
    assert table == defn.layers
    _assert_table_matches_kinds(
        table,
        TINY["n_layers"],
        TINY["n_dense"],
        TINY["core_block_layers"],
    )
    exp = ref.layer_table(ref.from_counts(**TINY))
    assert len(table) == len(exp)
    for g, e in zip(table, exp, strict=True):
        _assert_layer_vs_ref(g, e)


def test_layer_table_on_hand_built_tiny_layers():
    defn = _tiny_defn(layers=_tiny_layers())
    table = model_def.layer_table(defn)
    assert len(table) == 8
    _assert_table_matches_kinds(table, 8, TINY["n_dense"], TINY["core_block_layers"])


def test_layer_table_length_mismatch_raises():
    defn = _tiny_defn(layers=())
    with pytest.raises(model_def.ModelDefError):
        model_def.layer_table(defn)


def test_layer_table_wrong_attention_raises():
    layers = list(_tiny_layers())
    layers[0] = model_def.SglangLayer(
        index=0,
        attention=model_def.AttentionKind.MLA,
        ffn=model_def.FfnKind.DENSE,
        in_core_block=False,
    )
    defn = _tiny_defn(layers=tuple(layers))
    with pytest.raises(model_def.ModelDefError):
        model_def.layer_table(defn)


# ---------------------------------------------------------------------------
# mtp_head_count
# ---------------------------------------------------------------------------


def test_mtp_head_count_is_two_on_valid_defn():
    defn = _tiny_defn(layers=_tiny_layers())
    got = model_def.mtp_head_count(defn)
    assert got == 2
    assert got == ref.mtp_head_count(ref.from_counts(**TINY))
    assert isinstance(got, int)


def test_mtp_head_count_on_from_counts():
    defn = model_def.from_counts(**TINY)
    assert model_def.mtp_head_count(defn) == 2


# ---------------------------------------------------------------------------
# tiny-scale (A1 tiny_config integers)
# ---------------------------------------------------------------------------


def test_tiny_scale_counts():
    defn = model_def.from_counts(**TINY)
    assert defn.n_layers == 8
    assert defn.n_dense == 1
    assert defn.n_moe == 7
    assert defn.n_dense + defn.n_moe == defn.n_layers
    assert defn.n_linear_attn == 6
    assert defn.n_mla == 2
    assert defn.n_linear_attn + defn.n_mla == defn.n_layers
    assert defn.n_linear_attn == model_def.LINEAR_TO_MLA * defn.n_mla
    assert defn.mtp_heads == 2
    assert defn.core_block_layers == 2
    assert defn.d_model == 128
    assert defn.n_routed_experts == 8
    assert defn.n_shared_experts == 2
    assert defn.top_k == 2
    assert defn.expert_hidden == 256
    assert defn.vocab_size == 256
    assert defn.max_context == 128
    assert defn.recurrence_train_mean == 3.0
    assert defn.recurrence_max == 16
    assert defn.max_racks == 4
    lin = sum(1 for L in defn.layers if L.attention is model_def.AttentionKind.LINEAR)
    mla = sum(1 for L in defn.layers if L.attention is model_def.AttentionKind.MLA)
    dense = sum(1 for L in defn.layers if L.ffn is model_def.FfnKind.DENSE)
    moe = sum(1 for L in defn.layers if L.ffn is model_def.FfnKind.MOE)
    core = sum(1 for L in defn.layers if L.in_core_block)
    assert lin == 6
    assert mla == 2
    assert dense == 1
    assert moe == 7
    assert core == 2


# ---------------------------------------------------------------------------
# flagship-ish
# ---------------------------------------------------------------------------


def test_flagship_ish_counts_and_kinds():
    got = model_def.from_counts(**FLAGSHIP_ISH)
    exp = ref.from_counts(**FLAGSHIP_ISH)
    _assert_defn_vs_ref(got, exp)
    assert got.n_layers == 60
    assert got.n_dense == 4
    assert got.n_moe == 56
    assert got.n_dense + got.n_moe == 60
    assert got.n_linear_attn == 45
    assert got.n_mla == 15
    assert got.n_linear_attn == model_def.LINEAR_TO_MLA * got.n_mla
    assert got.core_block_layers == 8
    assert got.mtp_heads == 2
    assert model_def.mtp_head_count(got) == 2
    table = model_def.layer_table(got)
    assert len(table) == 60
    _assert_table_matches_kinds(table, 60, 4, 8)
    assert table[0].attention is model_def.AttentionKind.LINEAR
    assert table[3].attention is model_def.AttentionKind.MLA
    assert table[0].ffn is model_def.FfnKind.DENSE
    assert table[3].ffn is model_def.FfnKind.DENSE
    assert table[4].ffn is model_def.FfnKind.MOE
    assert table[51].in_core_block is False
    assert table[52].in_core_block is True
    assert table[59].in_core_block is True
    lin = sum(1 for L in table if L.attention is model_def.AttentionKind.LINEAR)
    mla = sum(1 for L in table if L.attention is model_def.AttentionKind.MLA)
    dense = sum(1 for L in table if L.ffn is model_def.FfnKind.DENSE)
    moe = sum(1 for L in table if L.ffn is model_def.FfnKind.MOE)
    core = sum(1 for L in table if L.in_core_block)
    assert (lin, mla, dense, moe, core) == (45, 15, 4, 56, 8)


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [TINY, FLAGSHIP_ISH])
def test_property_layer_index_matches_position(kwargs: dict):
    defn = model_def.from_counts(**kwargs)
    for i, layer in enumerate(model_def.layer_table(defn)):
        assert layer.index == i


@pytest.mark.parametrize("kwargs", [TINY, FLAGSHIP_ISH])
def test_property_kind_counts_match_declared(kwargs: dict):
    defn = model_def.from_counts(**kwargs)
    table = model_def.layer_table(defn)
    assert sum(L.attention is model_def.AttentionKind.LINEAR for L in table) == defn.n_linear_attn
    assert sum(L.attention is model_def.AttentionKind.MLA for L in table) == defn.n_mla
    assert sum(L.ffn is model_def.FfnKind.DENSE for L in table) == defn.n_dense
    assert sum(L.ffn is model_def.FfnKind.MOE for L in table) == defn.n_moe
    assert sum(L.in_core_block for L in table) == defn.core_block_layers


@pytest.mark.parametrize("n_layers", [4, 8, 60])
def test_property_attention_period_repeats(n_layers: int):
    period = model_def.LINEAR_TO_MLA + 1
    base = [model_def.attention_kind(i, n_layers) for i in range(period)]
    assert base.count(model_def.AttentionKind.LINEAR) == model_def.LINEAR_TO_MLA
    assert base.count(model_def.AttentionKind.MLA) == 1
    for i in range(n_layers):
        assert model_def.attention_kind(i, n_layers) is base[i % period]


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def test_from_counts_all_dense():
    kwargs = _kw(n_dense=8, n_moe=0)
    defn = model_def.from_counts(**kwargs)
    assert all(L.ffn is model_def.FfnKind.DENSE for L in defn.layers)
    _assert_defn_vs_ref(defn, ref.from_counts(**kwargs))


def test_from_counts_all_moe():
    kwargs = _kw(n_dense=0, n_moe=8)
    defn = model_def.from_counts(**kwargs)
    assert all(L.ffn is model_def.FfnKind.MOE for L in defn.layers)
    _assert_defn_vs_ref(defn, ref.from_counts(**kwargs))


def test_from_counts_core_is_entire_stack():
    kwargs = _kw(core_block_layers=8)
    defn = model_def.from_counts(**kwargs)
    assert all(L.in_core_block for L in defn.layers)


def test_from_counts_core_is_last_layer_only():
    kwargs = _kw(core_block_layers=1)
    defn = model_def.from_counts(**kwargs)
    assert [L.in_core_block for L in defn.layers] == [False] * 7 + [True]


def test_from_counts_minimum_period_four_layers():
    kwargs = _kw(
        n_layers=4,
        n_dense=1,
        n_moe=3,
        n_linear_attn=3,
        n_mla=1,
        core_block_layers=2,
    )
    defn = model_def.from_counts(**kwargs)
    assert [L.attention for L in defn.layers] == [
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.LINEAR,
        model_def.AttentionKind.MLA,
    ]
    _assert_defn_vs_ref(defn, ref.from_counts(**kwargs))
