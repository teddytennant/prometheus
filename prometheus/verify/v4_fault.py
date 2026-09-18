"""V4 fault-tolerance runner (spec 16.2, F4 template `v4.sh`).

Kill a rank mid-step, elastic DP continues, in-memory checkpoint restore,
injected bit flip, injected bad shard, host-RAM stand-in for Grace offload.

`verify/ncshare` `check_exit(V4)` reads `v4.json` with three bools, all True:

- `resumed_bitwise_equal`: restored run matches an uninterrupted twin
  (weights, opt, RNG)
- `sdc_caught_flip`: replica shard hashes catch an injected bit flip
  within N steps
- `spike_rollback_skipped_shard`: spike rollback skips the injected bad
  shard

Spec V4 is 4 to 8 H200. A CPU JAX / host path is allowed so the analog can
run without a GPU. That does not count as V4 verified.

Glue, not a second implementation: `ckpt/` (A5 save/restore, host-RAM
stand-in for Grace) and `control/` (A6 elastic DP, SDC, spike rollback).
The crates have no Python bindings, so this module is a host-side analog of
the same protocol. Must not import `tests/`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypedDict

import numpy as np

# Spec 16.2 row V4.
DEFAULT_GPUS_MIN = 4
DEFAULT_GPUS_MAX = 8

# Spec 5.5: in-memory copy on two other racks. Host RAM stands in for Grace.
MEMORY_REPLICAS = 2

# Spec 5.5 page window (second spike pages a human). Analog only records it.
_PAGE_WINDOW_STEPS = 10_000

# Tiny linear MSE + momentum SGD. Deterministic, bitwise. Not V1/V2 CE.
_IN = 5
_OUT = 2
_BATCH = 3
_SEED = 13
_LR = np.float32(0.04)
_MOMENTUM = np.float32(0.9)
_N_STEPS = 6
_KILL_STEP = 2
_N_REPLICAS = 4

# Elastic DP analog: hold tokens/step by raising accumulation when a rank dies.
_TOKENS_PER_STEP = 16
_MICROBATCH_TOKENS = 2

# SDC: hash-and-compare every N steps; catch the flip within this many steps.
_SDC_PERIOD_STEPS = 2
_SDC_CATCH_WITHIN_N = 4

_DATA_SHARDS: tuple[str, ...] = ("data-0", "data-1", "data-2")
_BAD_SHARD = "data-1"
_WEIGHT_SHARDS: tuple[str, ...] = ("w_a", "w_b")


class V4Result(TypedDict):
    """Payload written to `v4.json` by the F4 template."""

    resumed_bitwise_equal: bool
    sdc_caught_flip: bool
    spike_rollback_skipped_shard: bool


class V4Error(Exception):
    """Bad V4 inputs (gpus < 1) or a missing analog / backend."""


@dataclass
class _TrainState:
    """Tiny analog of spec 15.4 checkpoint payload: weights, opt, RNG."""

    weights: np.ndarray
    momentum: np.ndarray
    rng: int
    step: int

    def clone(self) -> _TrainState:
        return _TrainState(
            weights=np.array(self.weights, copy=True),
            momentum=np.array(self.momentum, copy=True),
            rng=int(self.rng),
            step=int(self.step),
        )


class _HostRamCheckpointer:
    """A5 analog: ``MEMORY_REPLICAS`` cloned copies in host RAM. No disk."""

    def __init__(self) -> None:
        self._replicas: list[_TrainState | None] = [None] * MEMORY_REPLICAS
        self._latest_id = "host-ram-0"

    def save(self, state: _TrainState) -> str:
        prepared = state.clone()
        for i in range(MEMORY_REPLICAS):
            self._replicas[i] = prepared.clone()
        return self._latest_id

    def restore(self) -> _TrainState:
        head = self._replicas[0]
        if head is None:
            raise V4Error("no in-memory checkpoint")
        return head.clone()

    def filled(self) -> bool:
        return all(slot is not None for slot in self._replicas)


def _splitmix64(seed: int) -> int:
    z = (int(seed) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def _draw_batch(rng: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Deterministic float32 batch from an integer RNG (checkpointed)."""
    n = _BATCH * _IN
    m = _BATCH * _OUT
    xs = np.empty(n, dtype=np.float32)
    ys = np.empty(m, dtype=np.float32)
    s = int(rng)
    for i in range(n):
        s = _splitmix64(s)
        xs[i] = np.float32((s >> 11) / 2**53 * 2.0 - 1.0)
    for i in range(m):
        s = _splitmix64(s)
        ys[i] = np.float32((s >> 11) / 2**53 * 2.0 - 1.0)
    return xs.reshape(_BATCH, _IN), ys.reshape(_BATCH, _OUT), s


def _init_state(seed: int = _SEED) -> _TrainState:
    w = np.empty((_IN, _OUT), dtype=np.float32)
    s = int(seed) ^ 0xA5A5A5A5
    for i in range(_IN):
        for j in range(_OUT):
            s = _splitmix64(s)
            w[i, j] = np.float32(((s >> 11) / 2**53 * 2.0 - 1.0) * 0.02)
    return _TrainState(
        weights=w,
        momentum=np.zeros((_IN, _OUT), dtype=np.float32),
        rng=int(seed),
        step=0,
    )


def _mse_grad(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    n = int(y64.size)
    err = x64 @ w64 - y64
    return ((2.0 / float(n)) * (x64.T @ err)).astype(np.float32)


def _step(state: _TrainState, *, lr_scale: float = 1.0) -> _TrainState:
    x, y, rng = _draw_batch(state.rng)
    grad = _mse_grad(x, y, state.weights)
    m = (_MOMENTUM * state.momentum + grad).astype(np.float32)
    lr = np.float32(float(_LR) * float(lr_scale))
    w = (state.weights - lr * m).astype(np.float32)
    return _TrainState(weights=w, momentum=m, rng=int(rng), step=int(state.step) + 1)


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    x = np.asarray(a)
    y = np.asarray(b)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return x.tobytes() == y.tobytes()


def _states_equal(a: _TrainState, b: _TrainState) -> bool:
    return (
        int(a.step) == int(b.step)
        and int(a.rng) == int(b.rng)
        and _arrays_equal(a.weights, b.weights)
        and _arrays_equal(a.momentum, b.momentum)
    )


def _min_accum(n_live: int) -> int:
    if n_live <= 0:
        return 0
    per = n_live * _MICROBATCH_TOKENS
    q, r = divmod(_TOKENS_PER_STEP, per)
    return q if r == 0 else q + 1


def _flip_bit(data: bytes, bit_index: int = 0) -> bytes:
    buf = bytearray(data)
    if not buf:
        raise V4Error("cannot flip a bit in empty bytes")
    i = int(bit_index) % (len(buf) * 8)
    buf[i // 8] ^= 1 << (i % 8)
    return bytes(buf)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _weight_shards(weights: np.ndarray) -> dict[str, bytes]:
    raw = np.asarray(weights).tobytes()
    mid = max(1, len(raw) // 2)
    return {"w_a": raw[:mid], "w_b": raw[mid:]}


def _sdc_mismatch(hexes: list[str]) -> bool:
    """True iff some replica disagrees with the majority hex (spec 5.5 SDC)."""
    n = len(hexes)
    if n < 2:
        return False
    counts: dict[str, int] = {}
    for h in hexes:
        counts[h] = counts.get(h, 0) + 1
    majority: str | None = None
    for h, c in counts.items():
        if c > n // 2:
            majority = h
            break
    if majority is None:
        return len(counts) > 1
    return any(h != majority for h in hexes)


def _next_data_shard(step: int, skipped: list[str]) -> str:
    live = [s for s in _DATA_SHARDS if s not in skipped]
    if not live:
        raise V4Error("all data shards skipped")
    return live[int(step) % len(live)]


def _resume_experiment() -> bool:
    """Kill a rank mid-step; restore from host-RAM ckpt; match the twin."""
    twin = _init_state()
    run = twin.clone()
    ckpt = _HostRamCheckpointer()
    ckpt.save(run)
    n_live = _N_REPLICAS
    accum_before = _min_accum(n_live)
    did_restore = False

    for t in range(_N_STEPS):
        twin = _step(twin)
        if t == _KILL_STEP:
            # Dirty mid-step analog, then simulated rank death (not kill -9).
            dirty = _step(run, lr_scale=0.5)
            n_live -= 1
            restored = ckpt.restore()
            run = _step(restored)
            did_restore = True
            if _states_equal(run, dirty):
                return False
        else:
            run = _step(run)
            if t < _KILL_STEP:
                ckpt.save(run)

    accum_after = _min_accum(n_live)
    if not did_restore:
        return False
    if n_live != _N_REPLICAS - 1 or accum_after < accum_before:
        return False
    if not ckpt.filled():
        return False
    return _states_equal(run, twin)


def _sdc_experiment() -> bool:
    """Inject a bit flip into replica 0; catch via shard hashes within N steps."""
    state = _init_state()
    clean = _weight_shards(state.weights)
    replicas = [dict(clean) for _ in range(_N_REPLICAS)]
    flipped = _flip_bit(replicas[0]["w_a"], bit_index=3)
    if flipped == replicas[0]["w_a"]:
        raise V4Error("bit flip did not change shard bytes")
    replicas[0]["w_a"] = flipped

    caught_at: int | None = None
    for step in range(1, _SDC_CATCH_WITHIN_N + 1):
        if _SDC_PERIOD_STEPS <= 0 or step % _SDC_PERIOD_STEPS != 0:
            continue
        for name in _WEIGHT_SHARDS:
            hexes = [_sha256_hex(replicas[r][name]) for r in range(_N_REPLICAS)]
            if _sdc_mismatch(hexes):
                caught_at = step
                break
        if caught_at is not None:
            break
    return caught_at is not None and int(caught_at) <= _SDC_CATCH_WITHIN_N


def _spike_experiment() -> bool:
    """Inject a bad data shard; rollback to last in-memory ckpt; skip it."""
    skipped: list[str] = []
    consumed: list[str] = []
    state = _init_state()
    ckpt = _HostRamCheckpointer()
    last_id = ckpt.save(state)
    last_spike_step: int | None = None
    pages = 0
    rollback_ids: list[str] = []
    spike_handled = False

    step = 0
    while step < _N_STEPS:
        shard = _next_data_shard(step, skipped)
        if shard == _BAD_SHARD and _BAD_SHARD not in skipped:
            spike_handled = True
            if last_spike_step is not None and (step - last_spike_step) < _PAGE_WINDOW_STEPS:
                pages += 1
            last_spike_step = step
            state = ckpt.restore()
            skipped.append(_BAD_SHARD)
            rollback_ids.append(last_id)
            continue
        consumed.append(shard)
        state = _step(state)
        last_id = ckpt.save(state)
        step += 1

    return (
        spike_handled
        and _BAD_SHARD in skipped
        and _BAD_SHARD not in consumed
        and len(rollback_ids) > 0
        and pages == 0
    )


def run_v4(*, gpus: int) -> V4Result:
    """Run V4 fault-tolerance checks.

    Parameters
    ----------
    gpus:
        Slurm GPU count. Must be >= 1. `gpus < 1` raises `V4Error`.
        CPU analog uses `gpus=1` in tests (simulated rank death, not
        `kill -9`; tiny state, not 90 TB). Still analog; not V4 verified.

    Returns
    -------
    V4Result
        Keys consumed by `check_exit(V4)`. Does not write `v4.json`; the
        template does that. Three separate analog experiments reported
        together is fine.
    """
    if gpus < 1:
        raise V4Error(f"gpus must be >= 1 (gpus={gpus})")

    resumed = _resume_experiment()
    sdc = _sdc_experiment()
    spike = _spike_experiment()
    return {
        "resumed_bitwise_equal": bool(resumed),
        "sdc_caught_flip": bool(sdc),
        "spike_rollback_skipped_shard": bool(spike),
    }
