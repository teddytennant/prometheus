"""Independent V4 fault-tolerance protocol (spec 16.2 / 5.5).

Slow and obvious. Does **not** import JAX, torch, ``prometheus.verify.v4_fault``,
``model/``, ``train/``, ``kernels/``, or the Rust crates. Production ``run_v4``
must not import this module; tests import both.

V4 meaning (CPU analog; tiny state, not 90 TB; simulated rank death, not
``kill -9``). This analog is **not** V4 verified.

``resumed_bitwise_equal``
    Restore from an in-memory checkpoint (host RAM stand-in for Grace,
    ``MEMORY_REPLICAS = 2``) matches an uninterrupted twin on weights,
    optimizer, and RNG. Must not be True because restore was skipped.
``sdc_caught_flip``
    An injected bit flip is caught by replica shard hashes within N steps.
    Must not be True because no flip was injected.
``spike_rollback_skipped_shard``
    Rollback after an injected bad data shard skips that shard. Must not be
    True because spike handling was skipped.

The three gates are **separate** analog experiments reported together. Glue
in production is ``ckpt/`` (A5) + ``control/`` (A6); this file is an
independent Python stand-in of that protocol, not a second kernel.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray

# Spec 16.2 row V4.
DEFAULT_GPUS_MIN = 4
DEFAULT_GPUS_MAX = 8

# Spec 5.5: in-memory copy on two other racks. Host RAM stands in for Grace.
MEMORY_REPLICAS = 2

# Spec 5.5 page window (second spike pages a human). Analog only records it.
PAGE_WINDOW_STEPS = 10_000

# SDC: hash-and-compare every N steps; catch the flip within this many steps.
SDC_PERIOD_STEPS = 2
SDC_CATCH_WITHIN_N = 4

# Tiny linear MSE + momentum SGD (not V1/V2 CE). Deterministic, bitwise.
TOY_IN = 4
TOY_OUT = 3
TOY_BATCH = 2
TOY_SEED = 7
TOY_LR = np.float32(0.05)
TOY_MOMENTUM = np.float32(0.9)
TOY_N_STEPS = 8
TOY_KILL_STEP = 3
TOY_N_REPLICAS = 4
TOY_FD_EPS = 1e-3
TOY_GRAD_RTOL = 5e-2
TOY_GRAD_ATOL = 5e-3

# Elastic DP analog: hold tokens/step by raising accumulation when a rank dies.
TOKENS_PER_STEP = 16
MICROBATCH_TOKENS = 2

SHARD_IDS: tuple[str, ...] = ("shard-0", "shard-1", "shard-2")
BAD_SHARD = "shard-1"
WEIGHT_SHARD_NAMES: tuple[str, ...] = ("w0", "w1")

V4_RESULT_KEYS: frozenset[str] = frozenset(
    {
        "resumed_bitwise_equal",
        "sdc_caught_flip",
        "spike_rollback_skipped_shard",
    }
)


@dataclass
class TrainState:
    """Tiny analog of weights + optimizer + RNG (spec 15.4 checkpoint payload)."""

    weights: Array
    momentum: Array
    rng_state: dict[str, Any]
    step: int

    def clone(self) -> TrainState:
        return TrainState(
            weights=np.array(self.weights, copy=True),
            momentum=np.array(self.momentum, copy=True),
            rng_state=copy.deepcopy(self.rng_state),
            step=int(self.step),
        )


class HostRamCheckpointer:
    """In-memory checkpoints: ``MEMORY_REPLICAS`` cloned copies in host RAM.

    Stand-in for Grace offload (spec 5.5 / A5). Does not touch disk.
    """

    def __init__(self) -> None:
        self.replicas: list[TrainState | None] = [None] * MEMORY_REPLICAS

    def save_in_memory(self, state: TrainState) -> str:
        prepared = state.clone()
        for i in range(MEMORY_REPLICAS):
            self.replicas[i] = prepared.clone()
        return "host-ram-0"

    def restore_latest_memory(self) -> TrainState:
        head = self.replicas[0]
        if head is None:
            raise LookupError("no in-memory checkpoint")
        return head.clone()

    def replica_filled(self, index: int) -> bool:
        if index < 0 or index >= MEMORY_REPLICAS:
            raise IndexError(f"replica index {index} out of range")
        return self.replicas[index] is not None


def make_rng(seed: int = TOY_SEED) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(seed)))


def rng_from_state(state: Mapping[str, Any]) -> np.random.Generator:
    bg = np.random.PCG64()
    bg.state = copy.deepcopy(dict(state))
    return np.random.Generator(bg)


def toy_init(seed: int = TOY_SEED) -> TrainState:
    rng = make_rng(seed)
    scale = np.float32(0.02)
    weights = (rng.standard_normal((TOY_IN, TOY_OUT)).astype(np.float32) * scale).astype(np.float32)
    momentum = np.zeros((TOY_IN, TOY_OUT), dtype=np.float32)
    return TrainState(
        weights=weights,
        momentum=momentum,
        rng_state=copy.deepcopy(rng.bit_generator.state),
        step=0,
    )


def mse_mean(pred: Array, target: Array) -> float:
    """Mean squared error over every element. Empty -> 0.0."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"mse shape mismatch: {p.shape} vs {t.shape}")
    if p.size == 0:
        return 0.0
    err = p - t
    return float(np.mean(err * err))


def linear_mse_grad(x: Array, y: Array, w: Array) -> Array:
    """``d mean((x @ w - y)^2) / d w`` as float64, same shape as ``w``."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    if x64.ndim != 2 or y64.ndim != 2 or w64.ndim != 2:
        raise ValueError("x, y, w must be 2D")
    if x64.shape[0] != y64.shape[0] or x64.shape[1] != w64.shape[0] or y64.shape[1] != w64.shape[1]:
        raise ValueError("linear mse shape mismatch")
    n = int(y64.size)
    if n == 0:
        return np.zeros_like(w64)
    err = x64 @ w64 - y64
    return (2.0 / float(n)) * (x64.T @ err)


def grad_match_ok(
    analytic: Array,
    finite_diff: Array,
    *,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    """Elementwise reverse-mode vs central differences (relative + absolute)."""
    g = np.asarray(analytic, dtype=np.float64)
    h = np.asarray(finite_diff, dtype=np.float64)
    if g.shape != h.shape:
        raise ValueError(f"grad shape mismatch: {g.shape} vs {h.shape}")
    if g.size == 0:
        return True
    if not (np.isfinite(g).all() and np.isfinite(h).all()):
        return False
    scale = np.maximum(np.abs(g), np.abs(h))
    abs_err = np.abs(g - h)
    return bool((abs_err <= float(atol) + float(rtol) * scale).all())


def toy_finite_diff_w_row0(
    x: Array,
    y: Array,
    w: Array,
    *,
    eps: float = TOY_FD_EPS,
) -> tuple[Array, Array]:
    """Analytic vs central FD on row 0 of ``w``. Both returned as float32."""
    w64 = np.asarray(w, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    analytic = linear_mse_grad(x64, y64, w64)
    numeric = np.zeros_like(w64)
    e = float(eps)
    for j in range(int(w64.shape[1])):
        plus = w64.copy()
        minus = w64.copy()
        plus[0, j] += e
        minus[0, j] -= e
        lp = mse_mean(x64 @ plus, y64)
        lm = mse_mean(x64 @ minus, y64)
        numeric[0, j] = (lp - lm) / (2.0 * e)
    # Compare only the FD'd row; other analytic rows are ignored here.
    return analytic[0].astype(np.float32), numeric[0].astype(np.float32)


def toy_grad_ok(seed: int = TOY_SEED) -> bool:
    rng = make_rng(int(seed) + 11)
    x = rng.standard_normal((TOY_BATCH, TOY_IN)).astype(np.float32)
    y = rng.standard_normal((TOY_BATCH, TOY_OUT)).astype(np.float32)
    w = (rng.standard_normal((TOY_IN, TOY_OUT)).astype(np.float32) * np.float32(0.02)).astype(
        np.float32
    )
    analytic, numeric = toy_finite_diff_w_row0(x, y, w)
    return grad_match_ok(analytic, numeric)


def _sgd_update(
    weights: Array,
    momentum: Array,
    grad: Array,
    *,
    lr_scale: float,
) -> tuple[Array, Array]:
    g = np.asarray(grad, dtype=np.float32)
    m = (TOY_MOMENTUM * np.asarray(momentum, dtype=np.float32) + g).astype(np.float32)
    lr = np.float32(float(TOY_LR) * float(lr_scale))
    w = (np.asarray(weights, dtype=np.float32) - lr * m).astype(np.float32)
    return w, m


def toy_step(state: TrainState, *, lr_scale: float = 1.0) -> TrainState:
    """One SGD+momentum step. Consumes RNG for a ``(B, IN)`` / ``(B, OUT)`` batch."""
    rng = rng_from_state(state.rng_state)
    x = rng.standard_normal((TOY_BATCH, TOY_IN)).astype(np.float32)
    y = rng.standard_normal((TOY_BATCH, TOY_OUT)).astype(np.float32)
    grad = linear_mse_grad(x, y, state.weights).astype(np.float32)
    w, m = _sgd_update(state.weights, state.momentum, grad, lr_scale=lr_scale)
    return TrainState(
        weights=w,
        momentum=m,
        rng_state=copy.deepcopy(rng.bit_generator.state),
        step=int(state.step) + 1,
    )


def toy_half_step(state: TrainState) -> TrainState:
    """Mid-step analog: same sample, half learning rate (dirty, not committed)."""
    return toy_step(state, lr_scale=0.5)


def bitwise_equal_arrays(a: Array, b: Array) -> bool:
    x = np.asarray(a)
    y = np.asarray(b)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    return x.tobytes() == y.tobytes()


def states_bitwise_equal(a: TrainState, b: TrainState) -> bool:
    """True iff weights, optimizer, RNG, and step match bit-for-bit."""
    if int(a.step) != int(b.step):
        return False
    if not bitwise_equal_arrays(a.weights, b.weights):
        return False
    if not bitwise_equal_arrays(a.momentum, b.momentum):
        return False
    return a.rng_state == b.rng_state


def flip_bit(data: bytes, bit_index: int = 0) -> bytes:
    """XOR a single bit. Empty input raises."""
    buf = bytearray(data)
    if not buf:
        raise ValueError("cannot flip a bit in empty bytes")
    nbits = len(buf) * 8
    i = int(bit_index) % nbits
    buf[i // 8] ^= 1 << (i % 8)
    return bytes(buf)


def shard_hash_hex(data: bytes) -> str:
    """SHA-256 hex of raw shard bytes (stdlib; not the Rust ``sha2`` crate)."""
    return hashlib.sha256(data).hexdigest()


def weight_shards(weights: Array) -> dict[str, bytes]:
    """Split the weight payload into two named shards (byte mid-point)."""
    raw = np.asarray(weights).tobytes()
    mid = max(1, len(raw) // 2)
    return {"w0": raw[:mid], "w1": raw[mid:]}


def min_grad_accumulation(n_live: int, microbatch_tokens: int, tokens_per_step: int) -> int:
    """Smallest accum such that ``n_live * microbatch * accum >= tokens_per_step``."""
    if int(tokens_per_step) <= 0:
        return 0
    n = int(n_live)
    mb = int(microbatch_tokens)
    if n <= 0 or mb <= 0:
        return 0
    per = n * mb
    q, r = divmod(int(tokens_per_step), per)
    return q if r == 0 else q + 1


def check_sdc_hashes(reports: Sequence[tuple[str, str]]) -> bool:
    """True iff some replica disagrees with the majority hex (spec 5.5 SDC).

    ``reports`` is ``(replica_id, hex)`` for one shard. No majority and more
    than one distinct hex also counts as a catch. A single replica cannot catch.
    """
    n = len(reports)
    if n < 2:
        return False
    counts: dict[str, int] = {}
    for _, hex_ in reports:
        counts[hex_] = counts.get(hex_, 0) + 1
    majority: str | None = None
    for hex_, c in counts.items():
        if c > n // 2:
            majority = hex_
            break
    if majority is None:
        return len(counts) > 1
    return any(hex_ != majority for _, hex_ in reports)


def next_shard(step: int, skipped: Sequence[str]) -> str:
    live = [s for s in SHARD_IDS if s not in set(skipped)]
    if not live:
        raise RuntimeError("all data shards skipped")
    return live[int(step) % len(live)]


def meets_v4_gates(result: Mapping[str, Any]) -> bool:
    """True iff all three V4 bools are exactly True."""
    return (
        result.get("resumed_bitwise_equal") is True
        and result.get("sdc_caught_flip") is True
        and result.get("spike_rollback_skipped_shard") is True
    )


def run_resume_experiment(
    *,
    n_steps: int = TOY_N_STEPS,
    kill_step: int = TOY_KILL_STEP,
    kill_rank: int = 1,
    n_replicas: int = TOY_N_REPLICAS,
    do_restore: bool = True,
    restore_corrupt: bool = False,
) -> dict[str, Any]:
    """Kill a rank mid-step; elastic DP continues; optional in-memory restore.

    Uninterrupted twin never sees the kill. Restored run must match the twin
    on weights, opt, and RNG. Skipping restore forces ``equal`` False.
    """
    n_steps = int(n_steps)
    kill_step = int(kill_step)
    n_replicas = int(n_replicas)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if kill_step < 0 or kill_step >= n_steps:
        raise ValueError("kill_step out of range")
    if n_replicas < 2:
        raise ValueError("n_replicas must be >= 2")
    kill_rank = int(kill_rank) % n_replicas

    twin = toy_init()
    run = twin.clone()
    ckpt = HostRamCheckpointer()
    ckpt.save_in_memory(run)
    live = list(range(n_replicas))
    accum_before = min_grad_accumulation(len(live), MICROBATCH_TOKENS, TOKENS_PER_STEP)
    did_restore = False

    for step in range(n_steps):
        twin = toy_step(twin)
        if step == kill_step:
            live = [r for r in live if r != kill_rank]
            if do_restore:
                restored = ckpt.restore_latest_memory()
                if restore_corrupt:
                    raw = restored.weights.tobytes()
                    flipped = flip_bit(raw, bit_index=0)
                    restored.weights = (
                        np.frombuffer(flipped, dtype=restored.weights.dtype)
                        .reshape(restored.weights.shape)
                        .copy()
                    )
                run = toy_step(restored)
                did_restore = True
            else:
                run = toy_half_step(run)
        else:
            run = toy_step(run)
            if step < kill_step:
                ckpt.save_in_memory(run)

    raw_equal = states_bitwise_equal(run, twin)
    # Must not report True because restore was skipped.
    equal = bool(did_restore) and (not restore_corrupt) and raw_equal
    accum_after = min_grad_accumulation(len(live), MICROBATCH_TOKENS, TOKENS_PER_STEP)
    replicas_filled = all(ckpt.replica_filled(i) for i in range(MEMORY_REPLICAS))
    return {
        "equal": equal,
        "raw_equal": raw_equal,
        "did_restore": did_restore,
        "restore_corrupt": bool(restore_corrupt),
        "n_live_after_kill": len(live),
        "killed_rank": kill_rank,
        "accum_before": accum_before,
        "accum_after": accum_after,
        "memory_replicas": MEMORY_REPLICAS,
        "replicas_filled": replicas_filled,
        "weights_equal": bitwise_equal_arrays(run.weights, twin.weights),
        "opt_equal": bitwise_equal_arrays(run.momentum, twin.momentum),
        "rng_equal": run.rng_state == twin.rng_state,
        "run_step": int(run.step),
        "twin_step": int(twin.step),
        "weights_dtype": str(run.weights.dtype),
        "weights_shape": tuple(int(x) for x in run.weights.shape),
        "momentum_dtype": str(run.momentum.dtype),
        "momentum_shape": tuple(int(x) for x in run.momentum.shape),
    }


def run_sdc_experiment(
    *,
    n_replicas: int = TOY_N_REPLICAS,
    n_steps: int = SDC_CATCH_WITHIN_N,
    sdc_period_steps: int = SDC_PERIOD_STEPS,
    inject_flip: bool = True,
    report_flipped_hash: bool = True,
    flip_bit_index: int = 0,
) -> dict[str, Any]:
    """Inject a bit flip into replica 0's ``w0`` shard; catch via hashes.

    If the flipped replica reports the *pre-flip* hash, the mismatch is hidden
    and ``caught`` is False. No injection => ``caught`` is False.
    """
    n_replicas = int(n_replicas)
    n_steps = int(n_steps)
    period = int(sdc_period_steps)
    if n_replicas < 2:
        raise ValueError("n_replicas must be >= 2")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    state = toy_init()
    clean = weight_shards(state.weights)
    replica_bytes: list[dict[str, bytes]] = [dict(clean) for _ in range(n_replicas)]
    flip_injected = bool(inject_flip)
    if flip_injected:
        replica_bytes[0]["w0"] = flip_bit(replica_bytes[0]["w0"], bit_index=int(flip_bit_index))

    caught_at: int | None = None
    mismatch_found = False
    for step in range(1, n_steps + 1):
        if period <= 0 or step % period != 0:
            continue
        for shard in WEIGHT_SHARD_NAMES:
            reports: list[tuple[str, str]] = []
            for r in range(n_replicas):
                payload = replica_bytes[r][shard]
                if r == 0 and flip_injected and shard == "w0" and not report_flipped_hash:
                    payload = clean[shard]
                reports.append((f"r{r}", shard_hash_hex(payload)))
            if check_sdc_hashes(reports):
                mismatch_found = True
                caught_at = step
                break
        if mismatch_found:
            break

    within_n = caught_at is not None and int(caught_at) <= int(SDC_CATCH_WITHIN_N)
    # Must not report True because no flip was injected.
    caught = bool(flip_injected) and mismatch_found and within_n
    return {
        "caught": caught,
        "flip_injected": flip_injected,
        "report_flipped_hash": bool(report_flipped_hash),
        "mismatch_found": mismatch_found,
        "caught_at_step": caught_at,
        "within_n": within_n,
        "sdc_period_steps": period,
        "sdc_catch_within_n": int(SDC_CATCH_WITHIN_N),
        "clean_w0_hash": shard_hash_hex(clean["w0"]),
        "flipped_w0_hash": shard_hash_hex(replica_bytes[0]["w0"]),
    }


def run_spike_experiment(
    *,
    n_steps: int = TOY_N_STEPS,
    inject_bad_shard: bool = True,
    skip_bad_shard: bool = True,
    bad_shard: str = BAD_SHARD,
) -> dict[str, Any]:
    """Inject a bad data shard; rollback to last in-memory ckpt; skip it.

    If spike handling is skipped (no injection) or rollback does not skip the
    shard, ``skipped_ok`` is False.
    """
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    skipped: list[str] = []
    consumed: list[str] = []
    state = toy_init()
    ckpt = HostRamCheckpointer()
    last_id = ckpt.save_in_memory(state)
    last_spike_step: int | None = None
    pages = 0
    rollback_ids: list[str] = []
    spike_handled = False

    step = 0
    while step < n_steps:
        shard = next_shard(step, skipped)
        if inject_bad_shard and shard == bad_shard and bad_shard not in skipped:
            spike_handled = True
            page = last_spike_step is not None and (step - last_spike_step) < PAGE_WINDOW_STEPS
            if page:
                pages += 1
            last_spike_step = step
            if skip_bad_shard:
                state = ckpt.restore_latest_memory()
                skipped.append(bad_shard)
                rollback_ids.append(last_id)
                # Redo this step index with the shard now skipped.
                continue
            consumed.append(shard)
            state = toy_step(state)
            step += 1
            continue
        consumed.append(shard)
        state = toy_step(state)
        last_id = ckpt.save_in_memory(state)
        step += 1

    skipped_ok = bool(spike_handled) and (bad_shard in skipped)
    return {
        "skipped_ok": skipped_ok,
        "spike_handled": spike_handled,
        "inject_bad_shard": bool(inject_bad_shard),
        "skip_bad_shard": bool(skip_bad_shard),
        "skipped_shards": list(skipped),
        "consumed_shards": list(consumed),
        "bad_shard": bad_shard,
        "pages": pages,
        "rollback_checkpoint_ids": list(rollback_ids),
        "bad_shard_consumed": bad_shard in consumed,
    }


def evaluate_v4_protocol(
    *,
    n_steps: int = TOY_N_STEPS,
    n_replicas: int = TOY_N_REPLICAS,
    kill_step: int = TOY_KILL_STEP,
    kill_rank: int = 1,
    do_restore: bool = True,
    restore_corrupt: bool = False,
    inject_flip: bool = True,
    report_flipped_hash: bool = True,
    inject_bad_shard: bool = True,
    skip_bad_shard: bool = True,
) -> dict[str, Any]:
    """Run the three independent V4 analog experiments and report the gates."""
    resume = run_resume_experiment(
        n_steps=n_steps,
        kill_step=kill_step,
        kill_rank=kill_rank,
        n_replicas=n_replicas,
        do_restore=do_restore,
        restore_corrupt=restore_corrupt,
    )
    sdc = run_sdc_experiment(
        n_replicas=n_replicas,
        inject_flip=inject_flip,
        report_flipped_hash=report_flipped_hash,
    )
    spike = run_spike_experiment(
        n_steps=n_steps,
        inject_bad_shard=inject_bad_shard,
        skip_bad_shard=skip_bad_shard,
    )
    result: dict[str, Any] = {
        "resumed_bitwise_equal": bool(resume["equal"]),
        "sdc_caught_flip": bool(sdc["caught"]),
        "spike_rollback_skipped_shard": bool(spike["skipped_ok"]),
        "resume": resume,
        "sdc": sdc,
        "spike": spike,
    }
    result["meets_gates"] = meets_v4_gates(result)
    result["grad_ok"] = toy_grad_ok()
    return result
