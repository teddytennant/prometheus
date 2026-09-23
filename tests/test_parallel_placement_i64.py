"""Defect lock for ``parallel.spmd_circular_pipeline`` int64 (spec 5.1).

The pytest process has one CPU device, and device count is fixed at the first
JAX import. Cases that need devices run in a fresh interpreter with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` and ``JAX_ENABLE_X64``
set before that child imports JAX or ``parallel``. This module does not import
JAX at collection.

Two children, no skip / xfail / gpu marker:

* ``JAX_ENABLE_X64=0``: eager int64 stays int64 (the body enables 64-bit mode)
  and matches ``sequential_compose`` bitwise, including ``n_stages == 1`` and
  values ``>= 2**31``. float32, ``n_stages`` 1 and 2, jit matches eager within
  rtol=1e-5 atol=1e-5. This child does not jit int64. With 64-bit mode off,
  JAX converts a jit input to int32 before the body runs; requiring the
  original bits back, or locking the truncated int32 result, would test that
  cast rather than this function.
* ``JAX_ENABLE_X64=1``: ``jax.jit`` with ``n_stages`` and ``stage_fn`` static
  matches the eager int64 result bitwise, and the jit jaxpr contains a
  ``ppermute`` primitive.

``stage_fn(x, stage) = x + (stage + 1)``. The parent compares production
outputs to ``tests.reference.placement.sequential_compose``.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_THIS = Path(__file__).resolve()
_N_DEVICES = 4
_XLA_FLAGS = f"--xla_force_host_platform_device_count={_N_DEVICES}"
_STATIC = ("n_stages", "stage_fn")
_RTOL = 1e-5
_ATOL = 1e-5

# (n_stages, shape, base). Every base is >= 2**31 so a truncation to int32
# cannot match sequential_compose bitwise.
_INT64_CASES = (
    (1, (2, 2), 1 << 41),
    (2, (2, 2), 1 << 41),
    (1, (2, 1), 1 << 31),
)
_FLOAT32_CASES = (
    (1, (3, 4)),
    (2, (4, 3)),
)


def _stage_fn(x, stage):
    return x + (stage + 1)


def _int64_microbatches(shape: tuple[int, ...], base: int) -> np.ndarray:
    count = int(np.prod(shape))
    return (np.int64(base) + np.arange(count, dtype=np.int64)).reshape(shape)


def _float32_microbatches(shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    return np.linspace(-1.5, 2.5, count, dtype=np.float32).reshape(shape)


def _host(value) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value))


def _primitive_names(obj, acc: list[str], seen: set[int]) -> None:
    """Collect JAX primitive names, descending into nested jaxprs."""
    marker = id(obj)
    if marker in seen:
        return
    seen.add(marker)
    eqns = getattr(obj, "eqns", None)
    if eqns is not None:
        for eqn in eqns:
            acc.append(str(eqn.primitive))
            params = getattr(eqn, "params", None) or {}
            for value in params.values():
                _primitive_names(value, acc, seen)
        return
    inner = getattr(obj, "jaxpr", None)
    if inner is not None and inner is not obj:
        _primitive_names(inner, acc, seen)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _primitive_names(value, acc, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _primitive_names(value, acc, seen)


def _assert_env(mode: str) -> str:
    """Fail before importing JAX if the child was not configured at spawn."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    x64_env = os.environ.get("JAX_ENABLE_X64")
    expected = "1" if mode == "x64_on" else "0"
    if xla_flags != _XLA_FLAGS:
        raise RuntimeError(f"XLA_FLAGS must be set before JAX import, got {xla_flags!r}")
    if x64_env != expected:
        raise RuntimeError(
            f"JAX_ENABLE_X64 must be {expected!r} before JAX import, got {x64_env!r}"
        )
    return x64_env


def _worker(mode: str) -> dict:
    x64_env = _assert_env(mode)
    root = str(_ROOT)
    sys.path.insert(0, root)

    # 64-bit mode is whatever the environment set. Do not call
    # jax.config.update("jax_enable_x64", ...) after this import.
    import jax

    import parallel
    import parallel.placement as placement_mod

    placement_file = str(Path(placement_mod.__file__).resolve())
    if not placement_file.startswith(root + os.sep):
        raise RuntimeError(f"imported {placement_file}, expected under {root}")
    x64_on = mode == "x64_on"
    if bool(jax.config.jax_enable_x64) != x64_on:
        raise RuntimeError(
            f"jax_enable_x64={jax.config.jax_enable_x64!r} after import; "
            "refusing to flip it after JAX has been imported"
        )
    if jax.local_device_count() != _N_DEVICES or jax.device_count() != _N_DEVICES:
        raise RuntimeError(
            f"expected {_N_DEVICES} host devices, "
            f"local={jax.local_device_count()} global={jax.device_count()}"
        )

    fn = parallel.spmd_circular_pipeline
    if fn is not placement_mod.spmd_circular_pipeline:
        raise RuntimeError("parallel.spmd_circular_pipeline is not the placement function")
    jitted = jax.jit(fn, static_argnames=_STATIC)
    cases: list[dict] = []

    for n_stages, shape, base in _INT64_CASES:
        microbatches = _int64_microbatches(shape, base)
        eager = fn(microbatches, n_stages=n_stages, stage_fn=_stage_fn)
        case = {
            "kind": "int64",
            "n_stages": n_stages,
            "shape": shape,
            "base": base,
            "microbatches": _host(microbatches),
            "eager": _host(eager),
            "eager_is_jax": isinstance(eager, jax.Array),
        }
        if x64_on:
            jit_out = jitted(microbatches, n_stages=n_stages, stage_fn=_stage_fn)
            traced = jitted.trace(microbatches, n_stages=n_stages, stage_fn=_stage_fn)
            names: list[str] = []
            _primitive_names(traced.jaxpr, names, set())
            case["jit"] = _host(jit_out)
            case["jit_is_jax"] = isinstance(jit_out, jax.Array)
            case["primitives"] = names
        cases.append(case)

    if not x64_on:
        # float32 jit stays in the x64-off child so an int64-only dtype change
        # cannot drop this path. Do not jit the int64 cases above.
        for n_stages, shape in _FLOAT32_CASES:
            microbatches = _float32_microbatches(shape)
            eager = fn(microbatches, n_stages=n_stages, stage_fn=_stage_fn)
            jit_out = jitted(microbatches, n_stages=n_stages, stage_fn=_stage_fn)
            cases.append(
                {
                    "kind": "float32",
                    "n_stages": n_stages,
                    "shape": shape,
                    "microbatches": _host(microbatches),
                    "eager": _host(eager),
                    "jit": _host(jit_out),
                    "eager_is_jax": isinstance(eager, jax.Array),
                    "jit_is_jax": isinstance(jit_out, jax.Array),
                }
            )

    return {
        "mode": mode,
        "jax_enable_x64_env": x64_env,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "n_devices": int(jax.local_device_count()),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "static_argnames": list(_STATIC),
        "cases": cases,
    }


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _spawn(mode: str) -> dict:
    fd, name = tempfile.mkstemp(prefix=f"pp_i64_{mode}_", suffix=".pkl")
    os.close(fd)
    out = Path(name)
    env = os.environ.copy()
    env["XLA_FLAGS"] = _XLA_FLAGS
    env["JAX_ENABLE_X64"] = "1" if mode == "x64_on" else "0"
    env["PYTHONPATH"] = str(_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [sys.executable, str(_THIS), mode, str(out)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            raise AssertionError(
                f"worker {mode} failed rc={proc.returncode}\n"
                f"stdout:\n{_tail(proc.stdout)}\nstderr:\n{_tail(proc.stderr)}"
            )
        with out.open("rb") as fh:
            payload = pickle.load(fh)
    finally:
        out.unlink(missing_ok=True)
    return payload


@pytest.fixture(scope="module")
def x64_off() -> dict:
    return _spawn("x64_off")


@pytest.fixture(scope="module")
def x64_on() -> dict:
    return _spawn("x64_on")


def _require_harness(payload: dict, *, x64: bool) -> None:
    expected_env = "1" if x64 else "0"
    assert payload["mode"] == ("x64_on" if x64 else "x64_off")
    assert payload["jax_enable_x64_env"] == expected_env
    assert payload["jax_enable_x64"] is x64
    assert payload["n_devices"] == _N_DEVICES
    assert payload["xla_flags"] == _XLA_FLAGS
    assert payload["static_argnames"] == list(_STATIC)


def _find(payload: dict, kind: str, n_stages: int, shape: tuple[int, ...]) -> dict:
    matches = [
        case
        for case in payload["cases"]
        if case["kind"] == kind and case["n_stages"] == n_stages and case["shape"] == shape
    ]
    assert len(matches) == 1, (kind, n_stages, shape, len(matches))
    return matches[0]


def _sequential(microbatches: np.ndarray, n_stages: int) -> np.ndarray:
    from tests.reference.placement import sequential_compose

    return np.ascontiguousarray(sequential_compose(microbatches, n_stages, _stage_fn))


def _assert_int64_bitwise(got: np.ndarray, expected: np.ndarray, what: str) -> None:
    g = np.ascontiguousarray(np.asarray(got))
    e = np.ascontiguousarray(np.asarray(expected))
    assert g.dtype == np.dtype(np.int64), f"{what} dtype {g.dtype}, expected int64"
    assert e.dtype == np.dtype(np.int64), f"{what} reference dtype {e.dtype}"
    assert g.shape == e.shape, f"{what} shape {g.shape} != {e.shape}"
    assert g.tobytes() == e.tobytes(), f"{what} bits differ\ngot {g}\nexpected {e}"


def _assert_float32_close(got: np.ndarray, expected: np.ndarray, what: str) -> None:
    g = np.asarray(got)
    e = np.asarray(expected)
    assert g.dtype == np.dtype(np.float32), f"{what} dtype {g.dtype}, expected float32"
    assert e.dtype == np.dtype(np.float32), f"{what} reference dtype {e.dtype}"
    assert g.shape == e.shape, f"{what} shape {g.shape} != {e.shape}"
    np.testing.assert_allclose(g, e, rtol=_RTOL, atol=_ATOL, err_msg=what)


@pytest.mark.parametrize(
    ("n_stages", "shape", "base"),
    _INT64_CASES,
    ids=["n_stages1-ge-2**41", "n_stages2-ge-2**41", "n_stages1-ge-2**31"],
)
def test_eager_int64_x64_off_matches_sequential(x64_off: dict, n_stages, shape, base) -> None:
    """Eager int64, JAX_ENABLE_X64=0, bitwise vs sequential_compose.

    Covers ``n_stages == 1`` and values ``>= 2**31``. The child does not jit
    these inputs.
    """
    _require_harness(x64_off, x64=False)
    assert base >= 2**31
    expected_in = _int64_microbatches(shape, base)
    assert int(expected_in.min()) >= 2**31
    case = _find(x64_off, "int64", n_stages, shape)
    assert case["base"] == base
    assert case["eager_is_jax"] is True
    _assert_int64_bitwise(case["microbatches"], expected_in, "x64-off input")
    assert "jit" not in case
    reference = _sequential(expected_in, n_stages)
    _assert_int64_bitwise(case["eager"], reference, f"x64-off eager n_stages={n_stages}")


@pytest.mark.parametrize(
    ("n_stages", "shape", "base"),
    _INT64_CASES,
    ids=["n_stages1-ge-2**41", "n_stages2-ge-2**41", "n_stages1-ge-2**31"],
)
def test_jit_int64_x64_on_matches_eager_bitwise(x64_on: dict, n_stages, shape, base) -> None:
    """jit int64 matches eager bitwise when 64-bit mode is on before the jit.

    ``n_stages`` and ``stage_fn`` are static. Parent also compares both to
    ``sequential_compose``.
    """
    _require_harness(x64_on, x64=True)
    assert base >= 2**31
    expected_in = _int64_microbatches(shape, base)
    assert int(expected_in.min()) >= 2**31
    case = _find(x64_on, "int64", n_stages, shape)
    assert case["base"] == base
    assert case["eager_is_jax"] is True
    assert case["jit_is_jax"] is True
    _assert_int64_bitwise(case["microbatches"], expected_in, "x64-on input")
    reference = _sequential(expected_in, n_stages)
    _assert_int64_bitwise(case["eager"], reference, f"x64-on eager n_stages={n_stages}")
    _assert_int64_bitwise(case["jit"], case["eager"], f"x64-on jit vs eager n_stages={n_stages}")
    _assert_int64_bitwise(case["jit"], reference, f"x64-on jit vs sequential n_stages={n_stages}")


@pytest.mark.parametrize(
    ("n_stages", "shape", "base"),
    _INT64_CASES,
    ids=["n_stages1-ge-2**41", "n_stages2-ge-2**41", "n_stages1-ge-2**31"],
)
def test_jit_jaxpr_contains_ppermute(x64_on: dict, n_stages, shape, base) -> None:
    """The x64-on jit jaxpr contains a ppermute primitive."""
    _require_harness(x64_on, x64=True)
    case = _find(x64_on, "int64", n_stages, shape)
    assert case["base"] == base
    primitives = case["primitives"]
    assert isinstance(primitives, list)
    assert primitives, "jit jaxpr had no primitives"
    assert "ppermute" in primitives, primitives


@pytest.mark.parametrize(
    ("n_stages", "shape"),
    _FLOAT32_CASES,
    ids=["n_stages1", "n_stages2"],
)
def test_float32_jit_matches_eager_x64_off(x64_off: dict, n_stages, shape) -> None:
    """float32 jit matches eager within rtol=1e-5 atol=1e-5, x64 off.

    Runs in the x64-off child so a dtype change cannot drop the float path.
    """
    _require_harness(x64_off, x64=False)
    expected_in = _float32_microbatches(shape)
    case = _find(x64_off, "float32", n_stages, shape)
    assert case["eager_is_jax"] is True
    assert case["jit_is_jax"] is True
    got_in = np.asarray(case["microbatches"])
    assert got_in.dtype == np.dtype(np.float32)
    assert got_in.shape == expected_in.shape
    np.testing.assert_array_equal(got_in, expected_in)
    reference = _sequential(expected_in, n_stages)
    _assert_float32_close(case["eager"], reference, f"float32 eager n_stages={n_stages}")
    _assert_float32_close(
        case["jit"], case["eager"], f"float32 jit vs eager n_stages={n_stages}"
    )
    _assert_float32_close(case["jit"], reference, f"float32 jit vs sequential n_stages={n_stages}")


def _main(mode: str, out_path: str) -> None:
    if mode not in {"x64_off", "x64_on"}:
        raise SystemExit(f"unknown worker mode {mode!r}")
    payload = _worker(mode)
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {_THIS.name} <x64_off|x64_on> <out.pkl>")
    _main(sys.argv[1], sys.argv[2])
