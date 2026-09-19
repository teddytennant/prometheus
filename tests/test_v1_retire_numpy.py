"""Oracle tests that V1 independent logits are torch-only (F4-v1-retire-numpy).

Spec 16.2 V1: JAX vs PyTorch reference logits, 1e-5 FP32.
``prometheus.verify._numpy_forward`` is retired: the module must not exist and
must not be importable. ``verify/ncshare/templates/v1.sh`` must fail-closed on
``import torch`` after the existing JAX GPU assert (same style as the
``jax.devices`` / GPU platform check). ``V1_INDEPENDENT_FORWARD`` stays
``\"torch\"``.

Every test in this file is written to fail on the pre-retirement tree
(``_numpy_forward.py`` still present, ``v1.sh`` has no ``import torch``).
Do not add a check that already holds on that tree (constant name alone,
``v1_parity`` not importing numpy, torch already a default CPU dep).
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_NUMPY_FORWARD_PY = _ROOT / "prometheus" / "verify" / "_numpy_forward.py"
_NUMPY_FORWARD_PKG = _ROOT / "prometheus" / "verify" / "_numpy_forward"
_V1_SH = _ROOT / "verify" / "ncshare" / "templates" / "v1.sh"
_LIB_RS = _ROOT / "verify" / "ncshare" / "src" / "lib.rs"

_V1_INDEPENDENT_FORWARD_TORCH = 'pub const V1_INDEPENDENT_FORWARD: &str = "torch";'


def _strip_trailing_hash_comment(line: str) -> str:
    in_s: str | None = None
    i = 0
    while i < len(line):
        c = line[i]
        if in_s is not None:
            if c == "\\" and in_s in "\"'":
                i += 2
                continue
            if c == in_s:
                in_s = None
            i += 1
            continue
        if c in "\"'":
            in_s = c
            i += 1
            continue
        if c == "#":
            return line[:i].rstrip()
        i += 1
    return line.rstrip()


def _code_lines(script: str) -> list[str]:
    out: list[str] = []
    for raw in script.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        out.append(_strip_trailing_hash_comment(raw))
    return out


def _is_jax_gpu_fail_closed_blob(blob: str) -> bool:
    low = blob.lower()
    if "jax" not in low:
        return False
    if not any(t in low for t in ("devices", "default_backend", "get_backend", "platform")):
        return False
    if not any(t in low for t in ("gpu", "cuda")):
        return False
    return any(t in low for t in ("assert", "raise", "sys.exit", "systemexit"))


def _is_import_torch_stmt(stmt: str) -> bool:
    t = stmt.strip()
    if t.startswith("from torch ") or t.startswith("from torch."):
        return True
    if not t.startswith("import "):
        return False
    rest = t[len("import ") :]
    for part in rest.split(","):
        tok = part.strip()
        if not tok:
            continue
        name = tok.split()[0]
        if name == "torch" or name.startswith("torch."):
            return True
    return False


_PYTHON_C = re.compile(r"""python3?\s+-c\s+(['"])(.*?)\1""", re.IGNORECASE)


def _line_has_top_level_import_torch(line: str) -> bool:
    if line.startswith((" ", "\t")):
        return False
    payloads = [line]
    payloads.extend(m.group(2) for m in _PYTHON_C.finditer(line))
    for payload in payloads:
        for stmt in payload.split(";"):
            if _is_import_torch_stmt(stmt):
                return True
    return False


def fail_closed_on_import_torch_after_jax_gpu(script: str) -> bool:
    """True iff a top-level ``import torch`` runs after the JAX GPU fail-closed check.

    Comments (shell or Python ``#``) do not count. An ``import torch`` that appears
    only *before* the JAX GPU assert does not count. Indented ``try/except``
    imports do not count (not the same style as the top-level JAX GPU assert).
    """
    lines = _code_lines(script)
    acc: list[str] = []
    jax_end: int | None = None
    for i, line in enumerate(lines):
        acc.append(line)
        if _is_jax_gpu_fail_closed_blob("\n".join(acc)):
            jax_end = i
            break
    if jax_end is None:
        return False
    same = lines[jax_end]
    if ";" in same:
        after = ";".join(same.split(";")[1:])
        if not same.startswith((" ", "\t")) and any(
            _is_import_torch_stmt(s) for s in after.split(";")
        ):
            return True
    return any(_line_has_top_level_import_torch(line) for line in lines[jax_end + 1 :])


def _assert_v1_independent_forward_stays_torch() -> None:
    lib = _LIB_RS.read_text(encoding="utf-8")
    assert _V1_INDEPENDENT_FORWARD_TORCH in lib, (
        "V1_INDEPENDENT_FORWARD must stay \"torch\" in verify/ncshare/src/lib.rs"
    )


# ---------------------------------------------------------------------------
# Group: prometheus.verify._numpy_forward is retired (not importable, no file)
# ---------------------------------------------------------------------------


def test_numpy_forward_source_must_not_exist() -> None:
    _assert_v1_independent_forward_stays_torch()
    assert not _NUMPY_FORWARD_PY.exists(), (
        "prometheus.verify._numpy_forward is retired: "
        f"{_NUMPY_FORWARD_PY} must not exist"
    )
    assert not _NUMPY_FORWARD_PKG.exists(), (
        "prometheus.verify._numpy_forward is retired: "
        f"{_NUMPY_FORWARD_PKG} must not exist"
    )


def test_numpy_forward_module_is_not_importable() -> None:
    _assert_v1_independent_forward_stays_torch()
    spec = importlib.util.find_spec("prometheus.verify._numpy_forward")
    assert spec is None, (
        "prometheus.verify._numpy_forward must not be importable (retired); "
        f"find_spec returned {spec!r}"
    )
    try:
        importlib.import_module("prometheus.verify._numpy_forward")
    except ModuleNotFoundError:
        return
    raise AssertionError(
        "prometheus.verify._numpy_forward must raise ModuleNotFoundError "
        "(retired); import succeeded"
    )


def test_numpy_forward_not_listed_as_prometheus_verify_submodule() -> None:
    _assert_v1_independent_forward_stays_torch()
    import prometheus.verify as pv

    names = {m.name for m in pkgutil.iter_modules(pv.__path__)}
    assert "_numpy_forward" not in names, (
        "prometheus.verify must not expose _numpy_forward as a submodule; "
        f"found {sorted(names)}"
    )


# ---------------------------------------------------------------------------
# Group: templates/v1.sh fail-closed on import torch after JAX GPU assert
# ---------------------------------------------------------------------------


def test_v1_sh_fail_closed_on_import_torch_after_jax_gpu_assert() -> None:
    _assert_v1_independent_forward_stays_torch()
    script = _V1_SH.read_text(encoding="utf-8")
    assert _is_jax_gpu_fail_closed_blob("\n".join(_code_lines(script))), (
        "templates/v1.sh must keep a JAX GPU fail-closed check "
        "(assert/raise on jax GPU devices/platform)"
    )
    assert fail_closed_on_import_torch_after_jax_gpu(script), (
        "templates/v1.sh must fail-closed on `import torch` after the JAX GPU "
        "assert (same style as the jax GPU check: top-level, not a comment, "
        "not only before the JAX assert). Torch is already a default CPU dep; "
        "this is not a new optional extra.\n--- v1.sh ---\n"
        f"{script}"
    )


def test_v1_sh_import_torch_is_not_only_a_comment() -> None:
    _assert_v1_independent_forward_stays_torch()
    script = _V1_SH.read_text(encoding="utf-8")
    raw_has_import = "import torch" in script or "from torch" in script
    live = fail_closed_on_import_torch_after_jax_gpu(script)
    assert live, (
        "templates/v1.sh must contain a live top-level `import torch` after the "
        "JAX GPU assert; a comment does not fail-close the job"
        + ("" if raw_has_import else " (no import torch text found at all)")
        + f"\n--- v1.sh ---\n{script}"
    )
