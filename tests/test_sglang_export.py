"""Implementer-facing tests for C1 JAX→SGLang weight export (spec 13.1, 15.5).

These import production ``sglang_fork``. Constants may pass against the stub.
Every test that calls ``flatten_params``, ``quantize``, ``dequantize``,
``export_weights``, ``parity_max_abs_err``, or ``to_contract`` MUST FAIL on
the stub with ``NotImplementedError``.

CPU tensor parity only — no serving, no GPU marker, no A1 flagship init.

Groups:
- Constants: SCHEMA_ID, SCHEMA_VERSION, TARGET, DTYPE_*, DTYPES.
- flatten_params: dotted names, list/tuple indices, sorted keys, jax leaves,
  non-array ExportError.
- quantize / dequantize: exact match vs NumPy reference, dtypes/shapes,
  bf16 identity, fp8 scale = amax/448, nvfp4 block-16 pad crop, zeros.
- hashes: SHA-256 of little-endian raw bytes; scale_hash is None iff scale is.
- export_weights: WeightExport fields, shard_count, files, invalid dtype/empty.
- parity_max_abs_err: bf16 is 0, fp8/nvfp4 match reference, pad does not leak.
- to_contract: F1 prometheus.weight_export via contracts.validate[_named].
- properties / finite-diff: dequantize linear in fp8 scale.
- edges / faults: NaN/Inf, bad dtype, name mismatch, invalid contract.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sglang_fork

import contracts
from tests.reference import sglang_export as ref

TOL = {"rtol": 1e-5, "atol": 1e-5}
APPROX = {"rel": 1e-5, "abs": 1e-5}

# Frozen goldens from tests/reference/sglang_export.py on ones((4, 8)).
_FP8_ONES_CONTENT = "af9613760f72635fbdb44a5a0a63c39f12af30f950a6ee5c971be188e89c4051"
_FP8_ONES_SCALE = "4abeb7e04407af6d4256f29b91c71479bd701e60ee7468c778f55b1df0247dc7"
_BF16_ONES_CONTENT = "b638277a8690e175a9137feff1e43c067f9faf4e2f600caf468fb05b0403b717"
_CREATED = "2026-09-16T12:00:00Z"
_SHA = "2c9a93d84ad4df28371eab9ca3e915b257ae530b5d77e66a548628892464c228"


def _tiny() -> dict[str, Any]:
    return {
        "w": np.ones((4, 8), dtype=np.float32),
        "block": [{"ln": np.ones((8,), dtype=np.float32)}],
    }


def _random_tree(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "w": rng.standard_normal((4, 8)).astype(np.float32),
        "block": [{"ln": rng.standard_normal((8,)).astype(np.float32)}],
    }


def _np(x: Any) -> np.ndarray:
    return np.asarray(x)


def _sha256_le(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    endian = a.dtype.byteorder
    if endian == ">" or (endian == "=" and sys.byteorder != "little"):
        a = a.byteswap().view(a.dtype.newbyteorder("<"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _crop(dequantized: Any, source: np.ndarray) -> np.ndarray:
    dq = _np(dequantized).astype(np.float32, copy=False)
    if dq.shape == source.shape:
        return dq
    return dq[..., : source.shape[-1]]


def _shard_file(out_dir: str, shard: sglang_fork.Shard) -> Path:
    assert shard.path, "Shard.path must be filled"
    path = Path(shard.path)
    return path if path.is_absolute() else Path(out_dir) / path


def _validate_payload(payload: dict[str, Any]) -> None:
    named = getattr(contracts, "validate_named", None)
    if callable(named):
        named("prometheus.weight_export", payload)
        return
    contracts.validate(payload)


def _assert_export_meta(got: sglang_fork.WeightExport, exp: sglang_fork.WeightExport) -> None:
    assert got.export_id == exp.export_id
    assert got.source_checkpoint_id == exp.source_checkpoint_id
    assert got.tokenizer_id == exp.tokenizer_id
    assert got.target == exp.target == sglang_fork.TARGET
    assert got.dtype == exp.dtype
    assert got.created_at == exp.created_at
    assert got.schema_id == exp.schema_id == sglang_fork.SCHEMA_ID
    assert got.schema_version == exp.schema_version == sglang_fork.SCHEMA_VERSION
    assert got.routing_table_hash == exp.routing_table_hash
    assert got.shard_count == exp.shard_count == len(got.shards) == len(exp.shards)
    assert isinstance(got.shards, tuple)
    for g_shard, e_shard in zip(got.shards, exp.shards, strict=True):
        assert g_shard.name == e_shard.name
        assert g_shard.dtype == e_shard.dtype
        assert g_shard.content_hash == e_shard.content_hash
        assert g_shard.scale_hash == e_shard.scale_hash


def _export_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "export_id": "e1",
        "source_checkpoint_id": "ckpt1",
        "tokenizer_id": "tok1",
        "dtype": sglang_fork.DTYPE_FP8,
        "created_at": _CREATED,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Constants (may pass on the stub)
# ---------------------------------------------------------------------------


def test_schema_id_and_version():
    assert sglang_fork.SCHEMA_ID == "prometheus.weight_export"
    assert sglang_fork.SCHEMA_VERSION == 1


def test_target_is_sglang():
    assert sglang_fork.TARGET == "sglang"


def test_dtypes_locked():
    assert sglang_fork.DTYPE_BF16 == "bf16"
    assert sglang_fork.DTYPE_FP8 == "fp8"
    assert sglang_fork.DTYPE_NVFP4 == "nvfp4"
    assert sglang_fork.DTYPES == ("bf16", "fp8", "nvfp4")


# ---------------------------------------------------------------------------
# flatten_params
# ---------------------------------------------------------------------------


def test_flatten_params_dotted_and_index_keys_sorted():
    flat = sglang_fork.flatten_params(_tiny())
    assert list(flat) == ["block.0.ln", "w"]
    exp = ref.flatten_params(_tiny())
    assert list(flat) == list(exp)
    np.testing.assert_allclose(_np(flat["w"]), exp["w"], **TOL)
    np.testing.assert_allclose(_np(flat["block.0.ln"]), exp["block.0.ln"], **TOL)
    assert _np(flat["w"]).shape == (4, 8)
    assert _np(flat["block.0.ln"]).shape == (8,)
    assert _np(flat["w"]).dtype == np.float32


def test_flatten_params_tuple_matches_list():
    listed = {"blocks": [{"w": np.ones((2, 2), dtype=np.float32)}]}
    tupled = {"blocks": ({"w": np.ones((2, 2), dtype=np.float32)},)}
    got_l = sglang_fork.flatten_params(listed)
    got_t = sglang_fork.flatten_params(tupled)
    assert list(got_l) == list(got_t) == ["blocks.0.w"]
    np.testing.assert_allclose(_np(got_l["blocks.0.w"]), _np(got_t["blocks.0.w"]), **TOL)


def test_flatten_params_three_level_nest_and_sort():
    params = {
        "z": np.ones((2,), dtype=np.float32),
        "a": {"b": {"c": np.arange(4, dtype=np.float32).reshape(2, 2)}},
    }
    flat = sglang_fork.flatten_params(params)
    assert list(flat) == ["a.b.c", "z"]
    np.testing.assert_array_equal(_np(flat["a.b.c"]), np.arange(4, dtype=np.float32).reshape(2, 2))


def test_flatten_params_empty_mapping():
    assert sglang_fork.flatten_params({}) == {}


def test_flatten_params_rejects_non_array_leaf():
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.flatten_params({"w": "not-an-array"})
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.flatten_params({"w": 3})
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.flatten_params({"w": None})


def test_flatten_params_accepts_jax_arrays():
    jnp = pytest.importorskip("jax.numpy")
    params = {"w": jnp.ones((4, 8), dtype=jnp.float32)}
    flat = sglang_fork.flatten_params(params)
    np.testing.assert_allclose(_np(flat["w"]), np.ones((4, 8), dtype=np.float32), **TOL)
    assert list(flat) == ["w"]


# ---------------------------------------------------------------------------
# quantize / dequantize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "nvfp4"])
def test_quantize_matches_reference_arrays(dtype: str):
    x = _random_tree(1)["w"]
    q_got, s_got = sglang_fork.quantize(x, dtype)
    q_exp, s_exp = ref.quantize(x, dtype)
    np.testing.assert_array_equal(_np(q_got), q_exp)
    if s_exp is None:
        assert s_got is None
    else:
        np.testing.assert_allclose(_np(s_got), s_exp, **TOL)
        assert _np(s_got).dtype == np.float32


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "nvfp4"])
def test_dequantize_matches_reference(dtype: str):
    x = _random_tree(2)["w"]
    q, s = ref.quantize(x, dtype)
    got = _np(sglang_fork.dequantize(q, s, dtype))
    exp = ref.dequantize(q, s, dtype)
    np.testing.assert_allclose(got, exp, **TOL)
    assert got.dtype == np.float32


def test_bf16_is_float32_identity():
    x = _random_tree(3)["w"]
    q, s = sglang_fork.quantize(x, "bf16")
    assert s is None
    assert _np(q).dtype == np.float32
    assert _np(q).shape == x.shape
    np.testing.assert_allclose(_np(q), x, **TOL)
    dq = _np(sglang_fork.dequantize(q, None, "bf16"))
    np.testing.assert_allclose(dq, x, **TOL)


def test_fp8_scale_is_amax_over_448_and_uint8_codes():
    x = _random_tree(4)["w"]
    q, s = sglang_fork.quantize(x, "fp8")
    assert _np(q).dtype == np.uint8
    assert _np(q).shape == x.shape
    scale = _np(s)
    assert scale.shape == ()
    assert scale.dtype == np.float32
    amax = float(np.max(np.abs(x)))
    assert float(scale) == pytest.approx(amax / 448.0, **APPROX)


def test_nvfp4_packed_uint8_and_block_scale():
    x = _random_tree(5)["w"]
    q, s = sglang_fork.quantize(x, "nvfp4")
    q_exp, s_exp = ref.quantize(x, "nvfp4")
    assert _np(q).dtype == np.uint8
    assert _np(q).shape == q_exp.shape == (4, 8)
    np.testing.assert_allclose(_np(s), s_exp, **TOL)
    assert _np(s).shape == (4, 1)
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    np.testing.assert_allclose(_np(s), amax / 6.0, **TOL)


def test_nvfp4_dequantize_pads_last_axis_and_crop_matches_source_len():
    x = _random_tree(6)["w"]
    q, s = sglang_fork.quantize(x, "nvfp4")
    dq = _np(sglang_fork.dequantize(q, s, "nvfp4"))
    assert dq.shape == (4, 16)
    cropped = _crop(dq, x)
    assert cropped.shape == (4, 8)
    np.testing.assert_allclose(cropped, _crop(ref.dequantize(q, s, "nvfp4"), x), **TOL)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "nvfp4"])
def test_quantize_zeros_roundtrip(dtype: str):
    x = np.zeros((4, 8), dtype=np.float32)
    q, s = sglang_fork.quantize(x, dtype)
    dq = _crop(sglang_fork.dequantize(q, s, dtype), x)
    np.testing.assert_allclose(dq, x, **TOL)


def test_fp8_ones_roundtrip_and_golden_hashes():
    x = np.ones((4, 8), dtype=np.float32)
    q, s = sglang_fork.quantize(x, "fp8")
    np.testing.assert_allclose(_crop(sglang_fork.dequantize(q, s, "fp8"), x), x, **TOL)
    assert _sha256_le(_np(q)) == _FP8_ONES_CONTENT
    assert _sha256_le(_np(s)) == _FP8_ONES_SCALE


def test_bf16_ones_golden_content_hash():
    q, s = sglang_fork.quantize(np.ones((4, 8), dtype=np.float32), "bf16")
    assert s is None
    assert _sha256_le(_np(q)) == _BF16_ONES_CONTENT


# ---------------------------------------------------------------------------
# hashes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "nvfp4"])
def test_export_hashes_are_sha256_of_le_bytes(dtype: str, tmp_path: Path):
    params = _tiny()
    got = sglang_fork.export_weights(
        params, out_dir=str(tmp_path / "got"), **_export_kwargs(dtype=dtype)
    )
    flat = ref.flatten_params(params)
    for shard in got.shards:
        q, s = ref.quantize(flat[shard.name], dtype)
        assert shard.content_hash == _sha256_le(np.asarray(q))
        if s is None:
            assert shard.scale_hash is None
        else:
            assert shard.scale_hash == _sha256_le(np.asarray(s))


def test_scale_hash_none_iff_bf16(tmp_path: Path):
    params = _tiny()
    bf = sglang_fork.export_weights(
        params, out_dir=str(tmp_path / "bf"), **_export_kwargs(dtype="bf16")
    )
    fp = sglang_fork.export_weights(
        params, out_dir=str(tmp_path / "fp"), **_export_kwargs(dtype="fp8")
    )
    assert all(s.scale_hash is None for s in bf.shards)
    assert all(s.scale_hash is not None and len(s.scale_hash) == 64 for s in fp.shards)


# ---------------------------------------------------------------------------
# export_weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "nvfp4"])
def test_export_weights_matches_reference(dtype: str, tmp_path: Path):
    params = _tiny()
    kwargs = _export_kwargs(dtype=dtype)
    got = sglang_fork.export_weights(params, out_dir=str(tmp_path / "got"), **kwargs)
    exp = ref.export_weights(params, out_dir=str(tmp_path / "exp"), **kwargs)
    _assert_export_meta(got, exp)
    assert got.shard_count == 2
    for shard in got.shards:
        assert _shard_file(str(tmp_path / "got"), shard).is_file()


def test_export_weights_target_always_sglang(tmp_path: Path):
    got = sglang_fork.export_weights(_tiny(), out_dir=str(tmp_path), **_export_kwargs())
    assert got.target == "sglang"


def test_export_weights_routing_table_hash_roundtrip(tmp_path: Path):
    got = sglang_fork.export_weights(
        _tiny(), out_dir=str(tmp_path), **_export_kwargs(routing_table_hash=_SHA)
    )
    assert got.routing_table_hash == _SHA


def test_export_weights_rejects_bad_dtype(tmp_path: Path):
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.export_weights(
            _tiny(), out_dir=str(tmp_path), **_export_kwargs(dtype="fp32")
        )


def test_export_weights_rejects_empty_params(tmp_path: Path):
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.export_weights({}, out_dir=str(tmp_path), **_export_kwargs())


def test_export_weights_rejects_bad_routing_hash(tmp_path: Path):
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.export_weights(
            _tiny(),
            out_dir=str(tmp_path),
            **_export_kwargs(routing_table_hash="not-a-hash"),
        )


# ---------------------------------------------------------------------------
# parity_max_abs_err
# ---------------------------------------------------------------------------


def test_parity_reads_reference_export_layout(tmp_path: Path):
    """Call production parity on files written by the NumPy reference."""
    params = _random_tree(11)
    out = str(tmp_path)
    exported = ref.export_weights(params, out_dir=out, **_export_kwargs(dtype="fp8"))
    got = sglang_fork.parity_max_abs_err(params, exported, out)
    exp = ref.parity_max_abs_err(params, exported, out)
    assert got == pytest.approx(exp, **APPROX)


def test_parity_bf16_is_zero(tmp_path: Path):
    params = _random_tree(7)
    out = str(tmp_path)
    exported = sglang_fork.export_weights(params, out_dir=out, **_export_kwargs(dtype="bf16"))
    err = sglang_fork.parity_max_abs_err(params, exported, out)
    assert err == pytest.approx(0.0, **APPROX)


@pytest.mark.parametrize("dtype", ["bf16", "fp8", "nvfp4"])
def test_parity_matches_reference(dtype: str, tmp_path: Path):
    params = _random_tree(8)
    kwargs = _export_kwargs(dtype=dtype)
    got_dir = str(tmp_path / "got")
    exp_dir = str(tmp_path / "exp")
    got = sglang_fork.export_weights(params, out_dir=got_dir, **kwargs)
    exp = ref.export_weights(params, out_dir=exp_dir, **kwargs)
    got_err = sglang_fork.parity_max_abs_err(params, got, got_dir)
    exp_err = ref.parity_max_abs_err(params, exp, exp_dir)
    assert got_err == pytest.approx(exp_err, **APPROX)
    assert got_err >= 0.0


def test_parity_nvfp4_uses_original_shape_not_pad(tmp_path: Path):
    params = _tiny()
    out = str(tmp_path)
    exported = sglang_fork.export_weights(params, out_dir=out, **_export_kwargs(dtype="nvfp4"))
    err = sglang_fork.parity_max_abs_err(params, exported, out)
    # ones((4,8)) is exact in E2M1 after scale; pad zeros must not inflate error.
    assert err == pytest.approx(0.0, **APPROX)


def test_parity_rejects_name_mismatch(tmp_path: Path):
    params = _tiny()
    out = str(tmp_path)
    exported = sglang_fork.export_weights(params, out_dir=out, **_export_kwargs())
    other = {"only": np.ones((4, 8), dtype=np.float32)}
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.parity_max_abs_err(other, exported, out)


# ---------------------------------------------------------------------------
# to_contract
# ---------------------------------------------------------------------------


def test_to_contract_validates_f1_and_matches_reference(tmp_path: Path):
    exported = sglang_fork.export_weights(_tiny(), out_dir=str(tmp_path), **_export_kwargs())
    payload = sglang_fork.to_contract(exported)
    _validate_payload(payload)
    expected = ref.to_contract(exported)
    assert payload == expected
    assert payload["schema_id"] == "prometheus.weight_export"
    assert payload["schema_version"] == 1
    assert payload["target"] == "sglang"
    assert payload["dtype"] == "fp8"
    assert "shard_count" not in payload
    assert payload["model_config_hash"] == ref.model_config_hash(
        exported.tokenizer_id, [s.name for s in exported.shards]
    )


def test_to_contract_omits_null_optional_fields():
    digest = "a" * 64
    export = sglang_fork.WeightExport(
        export_id="e1",
        source_checkpoint_id="ckpt1",
        tokenizer_id="tok1",
        target="sglang",
        dtype="bf16",
        shard_count=1,
        shards=(
            sglang_fork.Shard(name="w", dtype="bf16", content_hash=digest, path="w.npy"),
        ),
        routing_table_hash=None,
        created_at=_CREATED,
    )
    payload = sglang_fork.to_contract(export)
    _validate_payload(payload)
    assert "routing_table_hash" not in payload
    assert "scale_hash" not in payload["shards"][0]
    assert payload["shards"][0]["path"] == "w.npy"


def test_to_contract_includes_routing_table_hash():
    digest = "a" * 64
    export = sglang_fork.WeightExport(
        export_id="e1",
        source_checkpoint_id="ckpt1",
        tokenizer_id="tok1",
        target="sglang",
        dtype="bf16",
        shard_count=1,
        shards=(
            sglang_fork.Shard(name="w", dtype="bf16", content_hash=digest, path="w.npy"),
        ),
        routing_table_hash=_SHA,
        created_at=_CREATED,
    )
    payload = sglang_fork.to_contract(export)
    _validate_payload(payload)
    assert payload["routing_table_hash"] == _SHA


def test_to_contract_rejects_empty_shards():
    export = sglang_fork.WeightExport(
        export_id="e1",
        source_checkpoint_id="ckpt1",
        tokenizer_id="tok1",
        target="sglang",
        dtype="bf16",
        shard_count=0,
        shards=(),
        routing_table_hash=None,
        created_at=_CREATED,
    )
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.to_contract(export)


# ---------------------------------------------------------------------------
# properties / finite differences
# ---------------------------------------------------------------------------


def test_fp8_dequantize_linear_in_scale_finite_diff():
    x = _random_tree(9)["w"]
    q, s = sglang_fork.quantize(x, "fp8")
    s0 = float(np.asarray(s))
    eps = 1e-3
    y1 = _np(sglang_fork.dequantize(q, np.asarray(np.float32(s0 + eps)), "fp8"))
    y0 = _np(sglang_fork.dequantize(q, np.asarray(np.float32(s0 - eps)), "fp8"))
    numeric = (y1 - y0) / (2.0 * eps)
    analytic = _np(sglang_fork.dequantize(q, np.asarray(np.float32(1.0)), "fp8"))
    np.testing.assert_allclose(numeric, analytic, rtol=1e-4, atol=1e-4)


def test_quantize_rejects_unknown_dtype():
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.quantize(np.ones((4, 8), dtype=np.float32), "fp32")
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.dequantize(np.ones((4, 8), dtype=np.float32), None, "fp16")


# ---------------------------------------------------------------------------
# edges / fault injection
# ---------------------------------------------------------------------------


def test_quantize_rejects_nan_and_inf():
    nan = np.array([1.0, np.nan], dtype=np.float32)
    inf = np.array([1.0, np.inf], dtype=np.float32)
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.quantize(nan, "fp8")
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.quantize(inf, "nvfp4")


def test_dequantize_bf16_rejects_scale():
    q, _ = ref.quantize(np.ones((4, 8), dtype=np.float32), "bf16")
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.dequantize(q, np.asarray(1.0, dtype=np.float32), "bf16")


def test_dequantize_fp8_requires_scale():
    q, _ = ref.quantize(np.ones((4, 8), dtype=np.float32), "fp8")
    with pytest.raises(sglang_fork.ExportError):
        sglang_fork.dequantize(q, None, "fp8")


def test_negative_and_mixed_values_match_reference():
    x = np.array([[-6.0, -1.0, 0.0, 1.0, 3.0, 6.0, 0.5, -0.5]], dtype=np.float32)
    for dtype in sglang_fork.DTYPES:
        q_g, s_g = sglang_fork.quantize(x, dtype)
        q_e, s_e = ref.quantize(x, dtype)
        np.testing.assert_array_equal(_np(q_g), q_e)
        if s_e is None:
            assert s_g is None
        else:
            np.testing.assert_allclose(_np(s_g), s_e, **TOL)
        np.testing.assert_allclose(
            _np(sglang_fork.dequantize(q_g, s_g, dtype)),
            ref.dequantize(q_e, s_e, dtype),
            **TOL,
        )
