"""Independent V10 soak protocol (spec 16.2 / 15.2 H11).

Slow and obvious host Python + NumPy. Does **not** import JAX, torch, the V10
soak runner, ``model/``, ``train/``, ``kernels/``, or the Rust crates.
Production ``run_v10`` must not import this module; tests import both.

CPU analog: simulated-time event-log stand-in of the harness chaos suite
(process-kill / partition / replica, plus broker death and token expiry).
Nothing reads the wall clock and nothing sleeps 72 hours. Passing these tests
is **not** V10 verified. Spec V10 is a shared 72h soak while V5 to V9 run.

Exit JSON (F4 ``check_exit(V10)``):
    ``hours`` >= 72, ``lost_tasks`` == 0, ``duplicated_outputs`` == 0,
    ``dead_tokens`` == 0.

CPU analog may accept ``hours >= 1`` and still return ``hours`` as the
requested length.

Reviewer throwaways drop a task, duplicate an output, or mint a dead token
and expect non-zero counts. Those knobs live on ``SoakFaults`` /
``evaluate_v10_protocol``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Analog constants the production CPU stand-in must match (spec 16.2 V10).
DEFAULT_HOURS = 72.0
SOAK_HOURS_MIN = 72.0
CPU_ANALOG_HOURS_MIN = 1.0
MS_PER_HOUR = 3_600_000

MIN_REPLICAS = 2
DEFAULT_PROCESSES = 4
DEFAULT_TASKS = 4
DEFAULT_SEED = 10

LEASE_TTL_MS = 10_000.0
TOKEN_TTL_MS = 5_000.0
CLOCK_SKEW_MS = 30_000
FAULT_TICK_MS = 1_000

GENESIS_HASH = "0" * 64

EVENT_ENQUEUED = "chaos.enqueued"
EVENT_COMPLETED = "chaos.completed"
EVENT_FAULT = "chaos.fault"
EVENT_RECOVER = "chaos.recover"
EVENT_TOKEN = "chaos.token"

V10_RESULT_KEYS: tuple[str, ...] = (
    "hours",
    "lost_tasks",
    "duplicated_outputs",
    "dead_tokens",
)

TOY_FD_EPS = 1e-3
TOY_GRAD_RTOL = 5e-2
TOY_GRAD_ATOL = 5e-3


def _digest(prev_hash: str, body: Mapping[str, Any]) -> str:
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256()
    h.update(prev_hash.encode("ascii"))
    h.update(blob)
    return h.hexdigest()


@dataclass
class StoredEvent:
    event_type: str
    task_id: str | None
    attempt: int | None
    prev_hash: str
    digest: str
    at_ms: int
    body: dict[str, Any]


@dataclass
class Replica:
    index: int
    events: list[StoredEvent] = field(default_factory=list)
    lost: bool = False
    disk_full: bool = False
    corrupt: bool = False

    @property
    def writeable(self) -> bool:
        return not self.lost and not self.disk_full and not self.corrupt

    @property
    def tip_hash(self) -> str:
        if not self.events:
            return GENESIS_HASH
        return self.events[-1].digest


@dataclass
class Process:
    name: str
    alive: bool = True


@dataclass
class SoakFaults:
    """Injectable analog faults. Production happy-path must leave all counts 0."""

    drop_task: bool = False
    duplicate_output: bool = False
    mint_dead_token: bool = False


class AnalogError(ValueError):
    """Raised by the reference analog for illegal hours / clock skew."""


class World:
    """In-memory harness world. ``now_ms`` is injected; never wall-clock."""

    def __init__(
        self,
        *,
        n_replicas: int = MIN_REPLICAS,
        n_processes: int = DEFAULT_PROCESSES,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if n_replicas < MIN_REPLICAS:
            raise AnalogError(f"need at least {MIN_REPLICAS} replicas, have {n_replicas}")
        self.now_ms = 0
        self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(int(seed)))
        self.replicas = [Replica(index=i) for i in range(int(n_replicas))]
        names = ["coordinator"] + [f"worker-{i}" for i in range(max(0, int(n_processes) - 1))]
        self.processes = [Process(name=n) for n in names]
        # Directed partition edges drop replication from→to.
        self.partitions: set[tuple[int, int]] = set()
        self.enqueued: list[str] = []
        self.complete_counts: dict[tuple[str, int], int] = {}
        self.faults_applied: list[str] = []
        self.dropped_task_id: str | None = None
        # Token broker (D4 analog).
        self.broker_alive = True
        self.refresh_token = "refresh-0"
        self.committed_refresh = "refresh-0"
        self.access_token = "access-0"
        self.access_expires_at = TOKEN_TTL_MS
        self.dead_tokens: set[str] = set()
        self.dead_token_uses = 0
        self._token_seq = 0

    def advance(self, ms: int | float) -> None:
        self.now_ms += int(ms)

    def advance_to_hours(self, hours: float) -> None:
        target = int(float(hours) * MS_PER_HOUR)
        if target > self.now_ms:
            self.now_ms = target

    def apply_clock_skew(self, delta_ms: int) -> None:
        if abs(int(delta_ms)) > CLOCK_SKEW_MS:
            raise AnalogError(f"clock skew {delta_ms} ms exceeds ±{CLOCK_SKEW_MS} ms")
        self.now_ms += int(delta_ms)

    def _can_replicate(self, src: int, dst: int) -> bool:
        if src == dst:
            return True
        if (src, dst) in self.partitions:
            return False
        return self.replicas[dst].writeable

    def _origin(self, prefer: int) -> int | None:
        n = len(self.replicas)
        order = [prefer % n] + [i for i in range(n) if i != prefer % n]
        for i in order:
            if self.replicas[i].writeable:
                return i
        return None

    def _append(
        self,
        origin: int,
        event_type: str,
        *,
        task_id: str | None = None,
        attempt: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        body: dict[str, Any] = {
            "event_type": event_type,
            "task_id": task_id,
            "attempt": attempt,
            "at_ms": self.now_ms,
        }
        if extra:
            body.update(dict(extra))
        for i, replica in enumerate(self.replicas):
            if not replica.writeable:
                continue
            if not self._can_replicate(origin, i):
                continue
            prev = replica.tip_hash
            digest = _digest(prev, body)
            replica.events.append(
                StoredEvent(
                    event_type=event_type,
                    task_id=task_id,
                    attempt=attempt,
                    prev_hash=prev,
                    digest=digest,
                    at_ms=self.now_ms,
                    body=dict(body),
                )
            )

    def enqueue_task(self, task_id: str) -> None:
        origin = self._origin(0)
        if origin is None:
            raise AnalogError("no writeable replica for enqueue")
        self.enqueued.append(task_id)
        self._append(origin, EVENT_ENQUEUED, task_id=task_id)
        self.advance(FAULT_TICK_MS)

    def complete_task(self, task_id: str, *, attempt: int = 1) -> None:
        origin = self._origin(1 if len(self.replicas) > 1 else 0)
        if origin is None:
            raise AnalogError("no writeable replica for complete")
        key = (task_id, int(attempt))
        self.complete_counts[key] = self.complete_counts.get(key, 0) + 1
        self._append(origin, EVENT_COMPLETED, task_id=task_id, attempt=int(attempt))
        self.advance(FAULT_TICK_MS)

    def kill_process(self, name: str) -> None:
        for p in self.processes:
            if p.name == name:
                p.alive = False
                self.faults_applied.append("kill")
                origin = self._origin(0)
                if origin is not None:
                    self._append(origin, EVENT_FAULT, extra={"kind": "kill", "process": name})
                self.advance(FAULT_TICK_MS)
                return
        raise AnalogError(f"unknown process: {name}")

    def partition(self, src: int, dst: int, *, asymmetric: bool = True) -> None:
        self.partitions.add((int(src), int(dst)))
        if not asymmetric:
            self.partitions.add((int(dst), int(src)))
        self.faults_applied.append("partition")
        origin = self._origin(0)
        if origin is not None:
            self._append(
                origin,
                EVENT_FAULT,
                extra={
                    "kind": "partition",
                    "src": int(src),
                    "dst": int(dst),
                    "asymmetric": asymmetric,
                },
            )
        self.advance(FAULT_TICK_MS)

    def heal(self) -> None:
        self.partitions.clear()
        self.faults_applied.append("heal")
        origin = self._origin(0)
        if origin is not None:
            self._append(origin, EVENT_RECOVER, extra={"kind": "heal"})
        self.advance(FAULT_TICK_MS)

    def node_loss(self, index: int) -> None:
        self.replicas[int(index)].lost = True
        self.faults_applied.append("replica")
        origin = self._origin(0)
        if origin is not None:
            self._append(origin, EVENT_FAULT, extra={"kind": "replica", "index": int(index)})
        self.advance(FAULT_TICK_MS)

    def recover(self) -> None:
        survivor = next((r for r in self.replicas if r.writeable), None)
        for r in self.replicas:
            r.lost = False
            if survivor is not None and not r.events:
                r.events = [
                    StoredEvent(
                        event_type=e.event_type,
                        task_id=e.task_id,
                        attempt=e.attempt,
                        prev_hash=e.prev_hash,
                        digest=e.digest,
                        at_ms=e.at_ms,
                        body=dict(e.body),
                    )
                    for e in survivor.events
                ]
        for p in self.processes:
            p.alive = True
        self.faults_applied.append("recover")
        origin = self._origin(0)
        if origin is not None:
            self._append(origin, EVENT_RECOVER, extra={"kind": "recover"})
        self.advance(FAULT_TICK_MS)

    def _rotate_locked(self) -> None:
        self._token_seq += 1
        old_refresh = self.refresh_token
        self.dead_tokens.add(old_refresh)
        self.dead_tokens.add(self.access_token)
        self.refresh_token = f"refresh-{self._token_seq}"
        self.committed_refresh = self.refresh_token
        self.access_token = f"access-{self._token_seq}"
        self.access_expires_at = self.now_ms + TOKEN_TTL_MS
        origin = self._origin(0)
        if origin is not None:
            self._append(
                origin,
                EVENT_TOKEN,
                extra={"kind": "rotate", "refresh": self.committed_refresh},
            )

    def broker_death_mid_refresh(self) -> None:
        # New refresh minted in memory but not committed. Successor must use the
        # last committed refresh, not the uncommitted one.
        self._token_seq += 1
        uncommitted = f"refresh-uncommitted-{self._token_seq}"
        self.dead_tokens.add(uncommitted)
        self.broker_alive = False
        self.faults_applied.append("broker_death")
        origin = self._origin(0)
        if origin is not None:
            self._append(origin, EVENT_FAULT, extra={"kind": "broker_death"})
        self.advance(FAULT_TICK_MS)

    def recover_broker(self) -> None:
        self.broker_alive = True
        self.refresh_token = self.committed_refresh
        self._rotate_locked()
        self.faults_applied.append("broker_recover")
        origin = self._origin(0)
        if origin is not None:
            self._append(origin, EVENT_RECOVER, extra={"kind": "broker"})
        self.advance(FAULT_TICK_MS)

    def token_expiry(self) -> None:
        self.dead_tokens.add(self.access_token)
        self.access_expires_at = self.now_ms
        self.faults_applied.append("token_expiry")
        origin = self._origin(0)
        if origin is not None:
            self._append(origin, EVENT_FAULT, extra={"kind": "token_expiry"})
        self.advance(FAULT_TICK_MS)

    def rotate_token(self) -> None:
        if not self.broker_alive:
            raise AnalogError("broker dead")
        self._rotate_locked()
        self.advance(FAULT_TICK_MS)

    def use_access_token(self) -> None:
        self.use_token(self.access_token)

    def use_token(self, token: str) -> None:
        expired = self.now_ms >= self.access_expires_at and token == self.access_token
        dead = token in self.dead_tokens or expired or (not self.broker_alive)
        live = token == self.access_token and not dead
        if not live:
            self.dead_token_uses += 1
        origin = self._origin(0)
        if origin is not None:
            self._append(origin, EVENT_TOKEN, extra={"kind": "use", "live": live})
        self.advance(FAULT_TICK_MS)

    def mint_dead_token(self) -> str:
        token = f"dead-token-{self._token_seq}"
        self.dead_tokens.add(token)
        self.use_token(token)
        return token

    def chain_ok(self, index: int = 0) -> bool:
        replica = self.replicas[int(index)]
        prev = GENESIS_HASH
        for event in replica.events:
            if event.prev_hash != prev:
                return False
            if event.digest != _digest(prev, event.body):
                return False
            prev = event.digest
        return True

    def liveness_vector(self) -> np.ndarray:
        return np.array([1.0 if r.writeable else 0.0 for r in self.replicas], dtype=np.float64)

    def invariants(self) -> tuple[float, float, float]:
        return count_invariants(self.enqueued, self.complete_counts, self.dead_token_uses)

    def v10_result(self, hours: float) -> dict[str, float]:
        lost, dup, dead = self.invariants()
        return {
            "hours": float(hours),
            "lost_tasks": float(lost),
            "duplicated_outputs": float(dup),
            "dead_tokens": float(dead),
        }

    def run_chaos_suite(self, faults: SoakFaults | None = None) -> None:
        faults = faults or SoakFaults()
        for i in range(DEFAULT_TASKS):
            self.enqueue_task(f"task-{i}")

        victim = self.processes[-1].name if self.processes else "coordinator"
        self.kill_process(victim)
        self.recover()

        if len(self.replicas) >= 2:
            self.partition(0, 1, asymmetric=True)
            self.advance(int(LEASE_TTL_MS) + 1)
            self.heal()
            self.node_loss(1)
            self.recover()

        self.broker_death_mid_refresh()
        self.recover_broker()
        self.token_expiry()
        self.rotate_token()
        self.use_access_token()

        remaining = list(dict.fromkeys(self.enqueued))
        if faults.drop_task and remaining:
            self.dropped_task_id = remaining.pop()
        for tid in remaining:
            self.complete_task(tid, attempt=1)
        if faults.duplicate_output and remaining:
            self.complete_task(remaining[0], attempt=1)
        if faults.mint_dead_token:
            self.mint_dead_token()


def count_invariants(
    enqueued: list[str],
    complete_counts: Mapping[tuple[str, int], int],
    dead_token_uses: int,
) -> tuple[float, float, float]:
    """lost_tasks / duplicated_outputs / dead_tokens as floats."""
    completed_ids = {tid for (tid, _attempt), n in complete_counts.items() if int(n) > 0}
    unique: list[str] = []
    seen: set[str] = set()
    for tid in enqueued:
        if tid not in seen:
            seen.add(tid)
            unique.append(tid)
    if unique:
        lost_flags = np.array([tid not in completed_ids for tid in unique], dtype=np.int64)
        lost = float(np.sum(lost_flags))
    else:
        lost = 0.0
    if complete_counts:
        counts = np.array([int(n) for n in complete_counts.values()], dtype=np.int64)
        dup = float(np.sum(np.maximum(counts - 1, 0)))
    else:
        dup = 0.0
    return lost, dup, float(dead_token_uses)


def require_hours(hours: float) -> float:
    try:
        value = float(hours)
    except (TypeError, ValueError) as exc:
        raise AnalogError("hours must be a finite number >= 1") from exc
    if not math.isfinite(value) or value < CPU_ANALOG_HOURS_MIN:
        raise AnalogError(f"hours {hours!r} is below the analog floor {CPU_ANALOG_HOURS_MIN}")
    return value


def meets_v10_gates(result: Mapping[str, Any]) -> bool:
    """F4 GPU gate: hours >= 72 and the three counts == 0."""
    try:
        hours = float(result["hours"])
        lost = float(result["lost_tasks"])
        dup = float(result["duplicated_outputs"])
        dead = float(result["dead_tokens"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(x) for x in (hours, lost, dup, dead)):
        return False
    return hours >= SOAK_HOURS_MIN and lost == 0.0 and dup == 0.0 and dead == 0.0


def analog_ok(result: Mapping[str, Any]) -> bool:
    """CPU analog success: hours >= 1 and the three counts == 0. Not V10 verified."""
    try:
        hours = float(result["hours"])
        lost = float(result["lost_tasks"])
        dup = float(result["duplicated_outputs"])
        dead = float(result["dead_tokens"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(x) for x in (hours, lost, dup, dead)):
        return False
    return hours >= CPU_ANALOG_HOURS_MIN and lost == 0.0 and dup == 0.0 and dead == 0.0


def lease_remaining_ms(elapsed_ms: float, ttl_ms: float = LEASE_TTL_MS) -> float:
    return float(np.float64(ttl_ms) - np.float64(elapsed_ms))


def lease_expired(elapsed_ms: float, ttl_ms: float = LEASE_TTL_MS) -> bool:
    return lease_remaining_ms(elapsed_ms, ttl_ms) <= 0.0


def lease_d_elapsed(ttl_ms: float = LEASE_TTL_MS) -> float:
    """Analytic d(lease_remaining)/d(elapsed) = -1, independent of ttl."""
    _ = ttl_ms
    return -1.0


def lease_finite_diff(
    elapsed_ms: float,
    *,
    ttl_ms: float = LEASE_TTL_MS,
    eps: float = TOY_FD_EPS,
) -> float:
    up = lease_remaining_ms(elapsed_ms + eps, ttl_ms)
    dn = lease_remaining_ms(elapsed_ms - eps, ttl_ms)
    return float((up - dn) / (2.0 * eps))


def lease_grad_ok(
    *,
    eps: float = TOY_FD_EPS,
    rtol: float = TOY_GRAD_RTOL,
    atol: float = TOY_GRAD_ATOL,
) -> bool:
    elapsed = np.float64(100.0)
    analytic = lease_d_elapsed()
    numeric = lease_finite_diff(float(elapsed), eps=eps)
    return bool(np.allclose(analytic, numeric, rtol=rtol, atol=atol))


def evaluate_v10_protocol(
    *,
    hours: float = CPU_ANALOG_HOURS_MIN,
    drop_task: bool = False,
    duplicate_output: bool = False,
    mint_dead_token: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run the host chaos analog and report F4 fields plus protocol traces."""
    hours_f = require_hours(hours)
    world = World(seed=seed)
    world.run_chaos_suite(
        SoakFaults(
            drop_task=drop_task,
            duplicate_output=duplicate_output,
            mint_dead_token=mint_dead_token,
        )
    )
    world.advance_to_hours(hours_f)
    result = world.v10_result(hours_f)
    return {
        "result": result,
        "meets_gates": meets_v10_gates(result),
        "analog_ok": analog_ok(result),
        "did_kill": "kill" in world.faults_applied,
        "did_partition": "partition" in world.faults_applied,
        "did_replica_fault": "replica" in world.faults_applied,
        "did_recover": "recover" in world.faults_applied,
        "did_broker_death": "broker_death" in world.faults_applied,
        "did_token_expiry": "token_expiry" in world.faults_applied,
        "sim_ms": int(world.now_ms),
        "n_enqueued": len(list(dict.fromkeys(world.enqueued))),
        "n_completed": int(sum(1 for n in world.complete_counts.values() if n > 0)),
        "n_replicas": len(world.replicas),
        "n_processes": len(world.processes),
        "faults_applied": list(world.faults_applied),
        "chain_ok": all(world.chain_ok(i) for i in range(len(world.replicas))),
        "hours_requested": hours_f,
        "dropped_task_id": world.dropped_task_id,
        "liveness": world.liveness_vector().tolist(),
        "grad_ok": lease_grad_ok(),
    }
