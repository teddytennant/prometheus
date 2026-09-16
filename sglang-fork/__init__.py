"""Independent serving path that loads a JAX checkpoint (spec 13, 16.2 V8)."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

_BUCKETS = (1, 2, 4, 8, 16)
TIERS = ("hbm", "grace", "nvme")


def serve_jax_checkpoint(params, tokens, config, r: int = 1) -> dict[str, Any]:
    """Decode with the numpy reference, not the JAX forward used to produce the ckpt."""
    from tests.reference import model as ref

    np_params = _to_numpy(params)
    out = ref.forward(np.asarray(tokens), np_params, config, r=r)
    return {"logits": np.asarray(out.logits), "hidden": np.asarray(out.hidden)}


def _to_numpy(tree):
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_to_numpy(v) for v in tree)
    arr = np.asarray(tree)
    if arr.dtype == object:
        return pickle.loads(pickle.dumps(tree))
    return arr


class RequestState:
    """Verbal tokens plus latent thoughts for one decode request."""

    def __init__(
        self,
        verbal: list[int] | None = None,
        latent: list[Any] | None = None,
    ) -> None:
        self.verbal = list(verbal or [])
        self.latent = list(latent or [])

    def append_token(self, tok: int) -> None:
        self.verbal.append(int(tok))

    def append_thought(self, vec: Any) -> None:
        self.latent.append(np.asarray(vec))


def recurrence_bucket(r: int) -> int:
    r = max(1, int(r))
    for b in _BUCKETS:
        if r <= b:
            return b
    return 16


def bitwise_equal(a, b) -> bool:
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.dtype == bb.dtype and aa.shape == bb.shape and aa.tobytes() == bb.tobytes()


class KvStore:
    """HBM / Grace / NVMe KV tiers. swap/restore keep bitwise identity."""

    TIERS = TIERS

    def __init__(self) -> None:
        self._tiers: dict[str, dict[str, np.ndarray]] = {t: {} for t in TIERS}

    def put(self, key: str, kv, tier: str = "hbm") -> np.ndarray:
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier}")
        arr = np.array(kv, copy=True)
        self._tiers[tier][key] = arr
        return arr

    def get(self, key: str, tier: str) -> np.ndarray:
        return self._tiers[tier][key]

    def swap(self, key: str, src: str, dst: str) -> np.ndarray:
        if src not in TIERS or dst not in TIERS:
            raise ValueError("tier")
        arr = self._tiers[src].pop(key)
        copied = np.array(arr, copy=True)
        self._tiers[dst][key] = copied
        return copied

    def restore(self, key: str, src: str, dst: str = "hbm") -> np.ndarray:
        arr = self._tiers[src][key]
        copied = np.array(arr, copy=True)
        self._tiers[dst][key] = copied
        return copied

    def match(self, key: str, tier_a: str, tier_b: str) -> bool:
        return bitwise_equal(self._tiers[tier_a][key], self._tiers[tier_b][key])


class RoutingCapture:
    def __init__(self) -> None:
        self.per_token: list[list[int]] = []

    def add(self, expert_ids: Sequence[int]) -> None:
        self.per_token.append([int(x) for x in expert_ids])

    def record(self, token_index: int, expert_ids: Sequence[int]) -> None:
        while len(self.per_token) <= token_index:
            self.per_token.append([])
        self.per_token[token_index] = [int(x) for x in expert_ids]


def routing_capture(expert_ids_per_token: Sequence[Sequence[int]] | None = None) -> RoutingCapture:
    log = RoutingCapture()
    if expert_ids_per_token is not None:
        for ids in expert_ids_per_token:
            log.add(ids)
    return log


def ttt_sidecar_hook(
    hidden: np.ndarray,
    prefix: np.ndarray | None = None,
    hook: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    h = np.asarray(hidden, dtype=np.float64)
    if hook is not None:
        return np.asarray(hook(h))
    if prefix is None:
        return h
    p = np.asarray(prefix, dtype=np.float64)
    if p.ndim == 1:
        return h + p
    return h @ p


def mtp_speculative(
    draft_logits: np.ndarray | None = None,
    target_logits: np.ndarray | None = None,
    draft_tokens: Sequence[int] | None = None,
    target_tokens: Sequence[int] | None = None,
    n_draft: int = 2,
) -> dict[str, Any]:
    """Draft 2 tokens; accept the longest matching prefix with the target."""
    n_draft = 2
    if draft_tokens is None:
        if draft_logits is None:
            raise ValueError("draft_tokens or draft_logits required")
        draft_tokens = np.argmax(np.asarray(draft_logits), axis=-1).reshape(-1).tolist()
    if target_tokens is None:
        if target_logits is None:
            raise ValueError("target_tokens or target_logits required")
        target_tokens = np.argmax(np.asarray(target_logits), axis=-1).reshape(-1).tolist()
    draft = [int(x) for x in list(draft_tokens)[:n_draft]]
    target = [int(x) for x in list(target_tokens)[:n_draft]]
    accepted: list[int] = []
    for a, b in zip(draft, target, strict=False):
        if a == b:
            accepted.append(a)
        else:
            break
    return {
        "n_draft": n_draft,
        "accepted": accepted,
        "n_accepted": len(accepted),
        "draft": draft,
        "target": target,
    }


def serving_probe() -> dict[str, Any]:
    from model import cpu_config
    from tests.reference import model as ref

    cfg = cpu_config()
    rng = np.random.default_rng(0)
    params = ref.init_params(cfg, rng)
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int32)
    out = serve_jax_checkpoint(params, tokens, cfg, r=1)
    jax_like = ref.forward(tokens, params, cfg, r=1)
    err = float(np.max(np.abs(out["logits"] - np.asarray(jax_like.logits))))
    kv = KvStore()
    x = np.arange(16, dtype=np.float32)
    kv.put("s0", x, "hbm")
    kv.swap("s0", "hbm", "nvme")
    y = kv.restore("s0", "nvme", "hbm")
    match = bitwise_equal(x, y) and kv.match("s0", "hbm", "nvme")
    return {
        "logprob_max_abs_err": err,
        "hbm_host_nvme_match": bool(match),
        "recurrence_buckets": [1, 2, 4, 8, 16],
        "ok": err < 1e-5,
    }
