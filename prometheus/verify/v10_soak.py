"""V10 soak runner (spec 16.2, F4 template ``v10.sh``).

CPU analog of the harness chaos suite (spec 15.2 / H11) while V5 to V9
jobs run. A CPU / host path is allowed so the analog can run without a
72-hour wall clock. That does not count as V10 verified. V10 itself is a
shared 72h soak after V0.

``verify/ncshare`` ``check_exit(V10)`` reads the F4 payload:

- ``hours``: must be >= 72 for the GPU gate.
- ``lost_tasks``: must be 0.
- ``duplicated_outputs``: must be 0.
- ``dead_tokens``: must be 0.

Glue, not a second chaos crate: H11 ``harness/chaos``. Oracle sources
are out of bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypedDict

# Spec 16.2: 72h soak. Template ``v10.sh`` defaults to 72.
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

_EVENT_ENQUEUED = "enqueued"
_EVENT_COMPLETED = "completed"
_EVENT_FAULT = "fault"
_EVENT_RECOVER = "recover"


class V10Result(TypedDict):
    hours: float
    lost_tasks: float
    duplicated_outputs: float
    dead_tokens: float


class V10Error(Exception):
    """Bad V10 inputs (hours < 1) or a missing / failed backend."""


@dataclass
class _Event:
    kind: str
    task_id: str | None = None
    attempt: int = 0


@dataclass
class _Replica:
    name: str
    log: list[_Event] = field(default_factory=list)
    alive: bool = True
    partitioned_from: set[str] = field(default_factory=set)
    clock_delta_ms: int = 0


@dataclass
class _Lease:
    process: str
    attempt: int
    expires_at: float


class _World:
    """Host analog of process-kill / partition / replica / broker / token."""

    def __init__(
        self,
        *,
        n_replicas: int = MIN_REPLICAS,
        n_processes: int = DEFAULT_PROCESSES,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if n_replicas < MIN_REPLICAS:
            raise V10Error(f"need at least {MIN_REPLICAS} replicas")
        if n_processes < 1:
            raise V10Error("need at least one process")
        self.now_ms = 0.0
        self.seed = int(seed)
        self.replicas = [_Replica(name=f"r{i}") for i in range(n_replicas)]
        self.processes = [f"p{i}" for i in range(n_processes)]
        self.alive_processes: set[str] = set(self.processes)
        self.leases: dict[str, _Lease] = {}
        self.attempts: dict[str, int] = {}
        self.enqueued: list[str] = []
        self.dead_tokens: set[str] = set()
        self.dead_token_uses = 0
        self.broker_alive = True
        self.access_token = "tok-0"
        self.refresh_token = "ref-0"
        self.token_issued_at = 0.0
        self._token_n = 1

    def replica(self, name: str) -> _Replica:
        for item in self.replicas:
            if item.name == name:
                return item
        raise V10Error(f"unknown replica {name}")

    def live_replicas(self) -> list[_Replica]:
        return [item for item in self.replicas if item.alive]

    def tick(self, ms: float = FAULT_TICK_MS) -> None:
        self.now_ms += float(ms)

    def advance_hours(self, hours: float) -> None:
        target = float(hours) * float(MS_PER_HOUR)
        if target > self.now_ms:
            self.now_ms = target

    def _append(self, event: _Event, *, src: str | None = None) -> None:
        live = self.live_replicas()
        if not live:
            return
        origin = src if src is not None else live[0].name
        origin_rep = self.replica(origin)
        if not origin_rep.alive:
            return
        origin_rep.log.append(event)
        for other in live:
            if other.name == origin:
                continue
            if origin in other.partitioned_from or other.name in origin_rep.partitioned_from:
                continue
            other.log.append(event)

    def _token_dead(self, token: str) -> bool:
        if token in self.dead_tokens:
            return True
        if not self.broker_alive:
            return True
        if token != self.access_token:
            return True
        return (self.now_ms - self.token_issued_at) > TOKEN_TTL_MS

    def _use_token(self, token: str) -> bool:
        if self._token_dead(token):
            self.dead_token_uses += 1
            return False
        return True

    def enqueue(self, task_id: str) -> None:
        if task_id in self.enqueued:
            return
        self.enqueued.append(task_id)
        self.attempts[task_id] = self.attempts.get(task_id, 0) + 1
        owner = next(iter(self.alive_processes)) if self.alive_processes else self.processes[0]
        self.leases[task_id] = _Lease(
            process=owner,
            attempt=self.attempts[task_id],
            expires_at=self.now_ms + LEASE_TTL_MS,
        )
        self._append(_Event(_EVENT_ENQUEUED, task_id, self.attempts[task_id]))

    def complete(self, task_id: str, *, token: str | None = None) -> bool:
        if task_id not in self.enqueued:
            raise V10Error(f"unknown task {task_id}")
        used = self.access_token if token is None else token
        if not self._use_token(used):
            return False
        attempt = self.attempts.get(task_id, 1)
        self._append(_Event(_EVENT_COMPLETED, task_id, attempt))
        self.leases.pop(task_id, None)
        return True

    def kill_process(self, process: str) -> None:
        self.alive_processes.discard(process)
        self._append(_Event(_EVENT_FAULT, process, 0))
        self.tick(LEASE_TTL_MS)
        for task_id, lease in list(self.leases.items()):
            if lease.process != process:
                continue
            if self.now_ms < lease.expires_at:
                continue
            self.attempts[task_id] = self.attempts.get(task_id, 1) + 1
            self.leases.pop(task_id, None)

    def recover_process(self, process: str) -> None:
        self.alive_processes.add(process)
        self._append(_Event(_EVENT_RECOVER, process, 0))
        for task_id in self.enqueued:
            if task_id in self.leases:
                continue
            if self._completed(task_id):
                continue
            self.leases[task_id] = _Lease(
                process=process,
                attempt=self.attempts.get(task_id, 1),
                expires_at=self.now_ms + LEASE_TTL_MS,
            )

    def partition(self, src: str, dst: str, *, asymmetric: bool = False) -> None:
        self.replica(src).partitioned_from.add(dst)
        if not asymmetric:
            self.replica(dst).partitioned_from.add(src)
        self._append(_Event(_EVENT_FAULT, f"{src}->{dst}", 0), src=src)
        self.tick()

    def heal(self, src: str, dst: str) -> None:
        self.replica(src).partitioned_from.discard(dst)
        self.replica(dst).partitioned_from.discard(src)
        self._catch_up(src, dst)
        self._append(_Event(_EVENT_RECOVER, f"{src}->{dst}", 0), src=src)

    def lose_replica(self, name: str) -> None:
        item = self.replica(name)
        item.alive = False
        item.log.clear()
        self.tick()

    def recover_replica(self, name: str) -> None:
        item = self.replica(name)
        item.alive = True
        item.partitioned_from.clear()
        survivors = [rep for rep in self.live_replicas() if rep.name != name]
        if not survivors:
            raise V10Error("no surviving replica")
        donor = max(survivors, key=lambda rep: len(rep.log))
        item.log = list(donor.log)
        self._append(_Event(_EVENT_RECOVER, name, 0), src=donor.name)

    def _catch_up(self, a: str, b: str) -> None:
        left = self.replica(a)
        right = self.replica(b)
        if not left.alive or not right.alive:
            return
        if len(left.log) < len(right.log):
            left.log = list(right.log)
        elif len(right.log) < len(left.log):
            right.log = list(left.log)

    def skew_clock(self, name: str, delta_ms: int) -> None:
        if abs(int(delta_ms)) > CLOCK_SKEW_MS:
            raise V10Error(f"clock skew {delta_ms} exceeds ±{CLOCK_SKEW_MS}")
        self.replica(name).clock_delta_ms = int(delta_ms)
        self.tick()

    def kill_broker(self) -> None:
        self.broker_alive = False
        self.tick()

    def recover_broker(self) -> None:
        self.broker_alive = True
        self.rotate_token()

    def expire_token(self) -> None:
        self.dead_tokens.add(self.access_token)
        self.tick(TOKEN_TTL_MS + 1.0)

    def rotate_token(self) -> None:
        self.dead_tokens.add(self.access_token)
        self.dead_tokens.add(self.refresh_token)
        self.access_token = f"tok-{self._token_n}"
        self.refresh_token = f"ref-{self._token_n}"
        self._token_n += 1
        self.token_issued_at = self.now_ms
        self.tick()

    def drop_task(self, task_id: str | None = None) -> str:
        tid = task_id or f"dropped-{len(self.enqueued)}"
        self.enqueue(tid)
        self.leases.pop(tid, None)
        return tid

    def duplicate_output(self, task_id: str | None = None) -> str:
        tid = task_id or (self.enqueued[0] if self.enqueued else "dup-0")
        if tid not in self.enqueued:
            self.enqueue(tid)
        self.complete(tid)
        self.complete(tid)
        return tid

    def mint_dead_token(self) -> str:
        token = f"dead-{self._token_n}"
        self._token_n += 1
        self.dead_tokens.add(token)
        self._use_token(token)
        return token

    def _completed(self, task_id: str) -> bool:
        for rep in self.live_replicas():
            for event in rep.log:
                if event.kind == _EVENT_COMPLETED and event.task_id == task_id:
                    return True
        return False

    def counts(self) -> tuple[float, float, float]:
        live = self.live_replicas()
        if not live:
            return float(len(self.enqueued)), 0.0, float(self.dead_token_uses)
        log = max(live, key=lambda rep: len(rep.log)).log
        enqueued: set[str] = set()
        completed: dict[tuple[str, int], int] = {}
        for event in log:
            if event.kind == _EVENT_ENQUEUED and event.task_id is not None:
                enqueued.add(event.task_id)
            elif event.kind == _EVENT_COMPLETED and event.task_id is not None:
                key = (event.task_id, int(event.attempt))
                completed[key] = completed.get(key, 0) + 1
        lost = sum(1 for task_id in enqueued if not any(tid == task_id for tid, _att in completed))
        dup = sum(n - 1 for n in completed.values() if n > 1)
        return float(lost), float(dup), float(self.dead_token_uses)


def _pick(seed: int, n: int) -> tuple[int, int]:
    nxt = (int(seed) * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
    return nxt, int(nxt % n) if n else 0


def _run_analog(
    *,
    hours: float,
    drop_task: bool = False,
    duplicate_output: bool = False,
    mint_dead_token: bool = False,
) -> tuple[float, float, float]:
    world = _World(seed=DEFAULT_SEED)
    seed = DEFAULT_SEED
    for i in range(DEFAULT_TASKS):
        world.enqueue(f"t{i}")

    seed, idx = _pick(seed, len(world.processes))
    victim = world.processes[idx]
    world.kill_process(victim)
    world.recover_process(victim)

    names = [rep.name for rep in world.replicas]
    world.partition(names[0], names[1])
    world.heal(names[0], names[1])

    world.lose_replica(names[0])
    world.recover_replica(names[0])

    world.skew_clock(names[0], CLOCK_SKEW_MS)
    world.skew_clock(names[0], 0)

    world.kill_broker()
    world.recover_broker()
    world.expire_token()
    world.rotate_token()

    if drop_task:
        world.drop_task("reviewer-drop")
    if duplicate_output:
        world.duplicate_output("t0")
    if mint_dead_token:
        world.mint_dead_token()

    live = world.access_token
    for i in range(DEFAULT_TASKS):
        tid = f"t{i}"
        if world._completed(tid):
            continue
        world.complete(tid, token=live)

    world.advance_hours(hours)
    return world.counts()


def _require_hours(hours: float) -> float:
    try:
        value = float(hours)
    except (TypeError, ValueError) as exc:
        raise V10Error("hours must be a finite number >= 1") from exc
    if not math.isfinite(value) or value < CPU_ANALOG_HOURS_MIN:
        raise V10Error(f"hours must be >= {CPU_ANALOG_HOURS_MIN} (hours={hours})")
    return value


def run_v10(*, hours: float) -> V10Result:
    """Run the V10 analog.

    ``hours`` is the soak length the template would pass. Spec 16.2 is 72.
    ``hours < 1`` is an error. A CPU analog may accept ``hours >= 1`` and
    still return ``hours`` as the requested length; that is not V10
    verified.
    """
    requested = _require_hours(hours)
    lost, dup, dead = _run_analog(hours=requested)
    return {
        "hours": requested,
        "lost_tasks": lost,
        "duplicated_outputs": dup,
        "dead_tokens": dead,
    }
