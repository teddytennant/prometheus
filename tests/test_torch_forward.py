"""Oracle tests for ``prometheus.verify._torch_forward`` (spec 16.2 V1).

Public iface is ``prometheus.verify._torch_forward.forward``. The independent
slow math lives in ``tests.reference.torch_forward``. Production must never
import ``tests/``.

Every test in this file is written to fail against the NotImplementedError
stub (or the pre-implementation wiring: numpy V1, no default torch dep,
gpu marker still saying numpy).
"""

from __future__ import annotations

import ast
import inspect
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import model
from prometheus.verify import _torch_forward as tf
from prometheus.verify import v1_parity as v1

LOGITS_MAX_ABS = 1e-5
WRONG_FORWARD_SHIFT = 1e-3
FD_EPS = 1e-3
GRAD_RTOL = 5e-2
GRAD_ATOL = 5e-3
UNEMBED_COORDS = ((0, 0), (0, 1), (1, 0), (2, 3), (3, 5), (5, 7))

_ROOT = Path(__file__).resolve().parents[1]
_PROMETHEUS = _ROOT / "prometheus"


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        try:
            return np.asarray(x.numpy(), dtype=np.float32)
        except Exception:
            pass
    return np.asarray(x, dtype=np.float32)


def _logits_max_abs_diff(a: Any, b: Any) -> float:
    x = _to_numpy(a).astype(np.float64, copy=False)
    y = _to_numpy(b).astype(np.float64, copy=False)
    if x.shape != y.shape:
        raise ValueError(f"logit shape mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x - y)))


def _numpy_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _numpy_tree(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_numpy_tree(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_numpy_tree(v) for v in tree)
    if hasattr(tree, "detach"):
        tree = tree.detach()
    if hasattr(tree, "cpu"):
        tree = tree.cpu()
    # JAX DeviceArray views from np.asarray are read-only; FD perturbs unembed in place.
    return np.array(tree, dtype=np.float32, copy=True)


def _tiny_inputs() -> tuple[np.ndarray, dict[str, Any], model.ModelConfig]:
    cfg = model.tiny_config()
    params = model.init_params(cfg, rng=0)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, 4), dtype=np.int32)
    return tokens, params, cfg


def _prod_forward(tokens: Any, params: Any, cfg: Any, *, r: int | None = 1) -> Any:
    if r is None:
        return tf.forward(tokens, params, cfg)
    return tf.forward(tokens, params, cfg, r=r)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            mods.add(node.module)
            for alias in node.names:
                mods.add(f"{node.module}.{alias.name}")
    return mods


def _numpy_math_attrs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    numpy_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy" or alias.name.startswith("numpy."):
                    numpy_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            root = node.module.split(".")[0]
            if root == "numpy":
                for alias in node.names:
                    numpy_names.add(alias.asname or alias.name)
    forbidden = {
        "matmul",
        "dot",
        "einsum",
        "tensordot",
        "inner",
        "outer",
        "vdot",
        "softmax",
        "exp",
        "log_softmax",
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in numpy_names and node.attr in forbidden:
                found.add(node.attr)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "numpy" or node.module.startswith("numpy."):
                for alias in node.names:
                    if alias.name in forbidden:
                        found.add(alias.name)
    return found


def _require_gpu():
    jax = pytest.importorskip("jax")
    gpus = [d for d in jax.devices() if getattr(d, "platform", None) == "gpu"]
    if not gpus:
        pytest.skip("no GPU")
    return jax


# ---------------------------------------------------------------------------
# Wiring / source: must fail on the current stub (numpy V1, no torch dep)
# ---------------------------------------------------------------------------


def test_pyproject_default_dependencies_include_cpu_torch() -> None:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = list(data["project"]["dependencies"])
    joined = "\n".join(deps)
    names = []
    for dep in deps:
        name = dep.split(";")[0].strip()
        for sep in (">=", "==", "~=", ">", "<", "["):
            name = name.split(sep, 1)[0].strip()
        names.append(name)
    assert "torch" in names, "default project.dependencies must include torch"
    assert "torch[cuda]" not in joined
    assert not any("torch[cuda]" in d for d in deps)


def test_gpu_marker_says_pytorch_reference_not_numpy() -> None:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = list(data["tool"]["pytest"]["ini_options"]["markers"])
    gpu = [m for m in markers if m.startswith("gpu")]
    assert gpu, "pytest marker gpu must exist"
    text = gpu[0]
    assert "pytorch" in text.lower(), "gpu marker must say PyTorch reference"
    assert "numpy reference" not in text.lower()


def test_v1_parity_binds_independent_forward_to_torch_forward() -> None:
    v1_path = Path(v1.__file__).resolve()
    mods = _imported_modules(v1_path)
    assert not any("_numpy_forward" in m for m in mods), (
        "prometheus.verify.v1_parity must not import _numpy_forward"
    )
    assert any(m == "prometheus.verify._torch_forward" or m.startswith(
        "prometheus.verify._torch_forward."
    ) for m in mods), "importing v1_parity must bind the independent forward to _torch_forward"
    bound = False
    for obj in vars(v1).values():
        if obj is tf.forward:
            bound = True
            break
        if inspect.ismodule(obj) and getattr(obj, "forward", None) is tf.forward:
            bound = True
            break
    src = inspect.getsource(v1)
    if "prometheus.verify._torch_forward" in src and "forward" in src:
        bound = True
    assert bound, (
        "importing prometheus.verify.v1_parity must bind the independent "
        "forward to prometheus.verify._torch_forward.forward"
    )


def test_torch_forward_module_imports_torch_not_numpy_math() -> None:
    path = Path(tf.__file__).resolve()
    mods = _imported_modules(path)
    assert any(m == "torch" or m.startswith("torch.") for m in mods), (
        "_torch_forward.py must import torch"
    )
    assert not any(m == "tests" or m.startswith("tests.") for m in mods)
    assert not any(m == "jax" or m.startswith("jax.") for m in mods)
    assert not any("_numpy_forward" in m for m in mods)
    assert not _numpy_math_attrs(path), (
        "do not implement flagship math in NumPy then wrap with torch.tensor; "
        f"numpy math attrs: {_numpy_math_attrs(path)}"
    )


# ---------------------------------------------------------------------------
# Public iface: signature, shape, dtype, JAX 1e-5, reference, properties
# ---------------------------------------------------------------------------


def test_forward_signature_shape_dtype_float32_logits() -> None:
    sig = inspect.signature(tf.forward)
    names = list(sig.parameters)
    assert names == ["tokens", "params", "config", "r"]
    assert sig.parameters["r"].kind is inspect.Parameter.KEYWORD_ONLY
    tokens, params, cfg = _tiny_inputs()
    logits = _prod_forward(tokens, params, cfg, r=1)
    arr = _to_numpy(logits)
    assert arr.dtype == np.float32
    assert arr.shape == (tokens.shape[0], tokens.shape[1], cfg.vocab_size)
    assert np.isfinite(arr).all()


def test_v1_jax_vs_torch_forward_logits_match_1e_5() -> None:
    """V1 gate (CPU analog allowed): JAX model.forward vs _torch_forward.forward."""
    tokens, params, cfg = _tiny_inputs()
    jax_out = model.forward(tokens, params, cfg, r=1)
    torch_out = _prod_forward(tokens, params, cfg, r=1)
    diff = _logits_max_abs_diff(jax_out.logits, torch_out)
    assert np.isfinite(diff)
    assert diff <= LOGITS_MAX_ABS, f"V1 FP32 logit max-abs {diff} > {LOGITS_MAX_ABS}"


def test_production_matches_independent_torch_reference() -> None:
    tokens, params, cfg = _tiny_inputs()
    prod = _prod_forward(tokens, params, cfg, r=1)
    from tests.reference.torch_forward import forward as ref_forward

    ref = ref_forward(tokens, params, cfg, r=1)
    diff = _logits_max_abs_diff(prod, ref)
    assert np.isfinite(diff)
    assert diff <= LOGITS_MAX_ABS, f"prod vs oracle torch ref {diff} > {LOGITS_MAX_ABS}"


def test_omitted_r_returns_float32_logits_shape() -> None:
    tokens, params, cfg = _tiny_inputs()
    logits = tf.forward(tokens, params, cfg)
    arr = _to_numpy(logits)
    assert arr.dtype == np.float32
    assert arr.shape == (tokens.shape[0], tokens.shape[1], cfg.vocab_size)
    assert np.isfinite(arr).all()


def test_r_two_differs_from_r_one() -> None:
    tokens, params, cfg = _tiny_inputs()
    a = _prod_forward(tokens, params, cfg, r=1)
    b = _prod_forward(tokens, params, cfg, r=2)
    diff = _logits_max_abs_diff(a, b)
    assert np.isfinite(diff)
    assert diff > LOGITS_MAX_ABS


def test_batch1_seq1_shape() -> None:
    cfg = model.tiny_config()
    params = model.init_params(cfg, rng=1)
    tokens = np.array([[3]], dtype=np.int32)
    logits = _to_numpy(_prod_forward(tokens, params, cfg, r=1))
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()


def test_fault_injection_shifted_logits_exceed_1e_5() -> None:
    tokens, params, cfg = _tiny_inputs()
    jax_logits = _to_numpy(model.forward(tokens, params, cfg, r=1).logits)
    prod = _to_numpy(_prod_forward(tokens, params, cfg, r=1))
    match = _logits_max_abs_diff(jax_logits, prod)
    assert np.isfinite(match) and match <= LOGITS_MAX_ABS
    shifted = prod + np.float32(WRONG_FORWARD_SHIFT)
    bad = _logits_max_abs_diff(jax_logits, shifted)
    assert np.isfinite(bad)
    assert bad > LOGITS_MAX_ABS


def test_finite_difference_unembed_matches_jax_grad() -> None:
    tokens, params, cfg = _tiny_inputs()
    # Hit the iface first so the stub fails this test.
    _ = _prod_forward(tokens, params, cfg, r=1)

    import jax
    import jax.numpy as jnp

    params_np = _numpy_tree(params)
    tokens_np = np.asarray(tokens, dtype=np.int32)

    def _mean_prod(tree: dict[str, Any]) -> float:
        logits = _to_numpy(_prod_forward(tokens_np, tree, cfg, r=1))
        return float(np.mean(logits, dtype=np.float64))

    def _jax_mean(unembed: Any) -> Any:
        p = {**params, "unembed": unembed}
        return jnp.mean(model.forward(tokens_np, p, cfg, r=1).logits)

    jax_grad = np.asarray(jax.grad(_jax_mean)(params["unembed"]), dtype=np.float64)
    base = _mean_prod(params_np)
    assert np.isfinite(base)
    for i, j in UNEMBED_COORDS:
        plus = _numpy_tree(params_np)
        minus = _numpy_tree(params_np)
        plus["unembed"][i, j] = np.float32(plus["unembed"][i, j] + FD_EPS)
        minus["unembed"][i, j] = np.float32(minus["unembed"][i, j] - FD_EPS)
        fd = (_mean_prod(plus) - _mean_prod(minus)) / (2.0 * FD_EPS)
        analytic = float(jax_grad[i, j])
        assert np.isfinite(fd) and np.isfinite(analytic)
        assert abs(fd - analytic) <= GRAD_ATOL + GRAD_RTOL * abs(analytic)


def test_production_prometheus_does_not_import_tests() -> None:
    tokens, params, cfg = _tiny_inputs()
    _ = _prod_forward(tokens, params, cfg, r=1)
    offenders: list[str] = []
    for path in _PROMETHEUS.rglob("*.py"):
        try:
            mods = _imported_modules(path)
        except SyntaxError:
            continue
        if any(m == "tests" or m.startswith("tests.") for m in mods):
            offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, f"production imported tests/: {offenders}"


# ---------------------------------------------------------------------------
# GPU-only V1 (skipped cleanly on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_v1_gpu_jax_vs_pytorch_reference_logits_1e_5() -> None:
    """V1: 1 GPU, tiny_config, JAX vs PyTorch reference, FP32 1e-5."""
    _require_gpu()
    tokens, params, cfg = _tiny_inputs()
    jax_out = model.forward(tokens, params, cfg, r=1)
    torch_out = _prod_forward(tokens, params, cfg, r=1)
    diff = _logits_max_abs_diff(jax_out.logits, torch_out)
    assert np.isfinite(diff)
    assert diff <= LOGITS_MAX_ABS, f"V1 GPU logit max-abs {diff} > {LOGITS_MAX_ABS}"
