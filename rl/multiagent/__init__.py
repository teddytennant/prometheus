"""Multi-agent roles, topologies, bus, credit, distill-back (spec 9, D7, I7)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

ALLOWED_HANDOFFS = {
    ("researcher", "implementer"),
    ("implementer", "reviewer"),
    ("reviewer", "implementer"),
}

TOPOLOGIES = ("star", "ring", "hierarchy")


@dataclass
class Agent:
    role: str
    budget_tokens: int


def default_crew() -> list[Agent]:
    return [
        Agent("researcher", 8_000),
        Agent("implementer", 16_000),
        Agent("reviewer", 8_000),
    ]


def handoff(from_role: str, to_role: str, artifact: str) -> dict:
    if (from_role, to_role) not in ALLOWED_HANDOFFS:
        raise ValueError(f"bad handoff {from_role}->{to_role}")
    return {"from": from_role, "to": to_role, "artifact": artifact}


def topology(kind: str, agents: Sequence[Agent] | None = None) -> dict[str, Any]:
    if kind not in TOPOLOGIES:
        raise ValueError(f"unknown topology {kind}")
    crew = list(agents) if agents is not None else default_crew()
    roles = [a.role for a in crew]
    if len(roles) < 2:
        raise ValueError("need at least two agents")
    if kind == "star":
        hub = roles[0]
        edges = [(hub, r) for r in roles[1:]] + [(r, hub) for r in roles[1:]]
    elif kind == "ring":
        n = len(roles)
        edges = [(roles[i], roles[(i + 1) % n]) for i in range(n)]
    else:
        edges = [(roles[i], roles[i + 1]) for i in range(len(roles) - 1)]
        if "reviewer" in roles and "implementer" in roles:
            edges.append(("reviewer", "implementer"))
    return {"kind": kind, "nodes": roles, "edges": edges}


@dataclass
class MessageBus:
    agents: list[Agent] = field(default_factory=default_crew)
    topology_kind: str = "hierarchy"
    used: dict[str, int] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.budget = {a.role: int(a.budget_tokens) for a in self.agents}
        if not self.used:
            self.used = {a.role: 0 for a in self.agents}
        self.topo = topology(self.topology_kind, self.agents)

    def remaining(self, role: str) -> int:
        return self.budget.get(role, 0) - self.used.get(role, 0)

    def send(self, from_role: str, to_role: str, artifact: str, tokens: int = 1) -> dict:
        rec = handoff(from_role, to_role, artifact)
        tokens = int(tokens)
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        if from_role not in self.budget:
            raise ValueError(f"unknown role {from_role}")
        if self.used[from_role] + tokens > self.budget[from_role]:
            raise ValueError(f"budget exceeded for {from_role}")
        self.used[from_role] += tokens
        rec = {**rec, "tokens": tokens}
        self.messages.append(rec)
        return rec

    def total_used(self) -> int:
        return int(sum(self.used.values()))


def credit_assignment(
    outcome: float,
    roles: Sequence[str] | None = None,
    traces: Iterable[dict] | None = None,
    weights: Sequence[float] | None = None,
    sub_rewards: dict[str, float] | None = None,
) -> dict[str, float]:
    """Share `outcome` across roles. Returned values always sum to outcome."""
    role_list: list[str]
    if roles is not None:
        role_list = list(roles)
    elif traces is not None:
        role_list = []
        for t in traces:
            r = t.get("from") or t.get("role")
            if r and r not in role_list:
                role_list.append(str(r))
    else:
        role_list = [a.role for a in default_crew()]
    n = len(role_list)
    if n == 0:
        return {}
    if weights is None:
        w = [1.0] * n
        if traces is not None:
            tok = {r: 0.0 for r in role_list}
            for t in traces:
                r = t.get("from") or t.get("role")
                if r in tok:
                    tok[r] += float(t.get("tokens", 1))
            w = [tok[r] for r in role_list]
            if sum(w) == 0:
                w = [1.0] * n
    else:
        w = [float(x) for x in weights]
        if len(w) != n:
            raise ValueError("weights length")
    s = sum(w)
    out = float(outcome)
    if s == 0:
        credits = {r: 0.0 for r in role_list}
    else:
        credits = {r: out * (wi / s) for r, wi in zip(role_list, w, strict=True)}
    if sub_rewards:
        for r, v in sub_rewards.items():
            credits[r] = credits.get(r, 0.0) + float(v)
        tot = sum(credits.values())
        if tot == 0:
            credits = {r: out / len(credits) for r in credits} if out else dict(credits)
        else:
            credits = {r: out * (c / tot) for r, c in credits.items()}
    return credits


def distill_back(
    traces: Sequence[dict],
    outcome: float,
    *,
    single_outcome: float | None = None,
) -> list[dict]:
    """Compress crew traces into single-agent training items (spec 9.5)."""
    if single_outcome is not None and float(outcome) <= float(single_outcome):
        return []
    items = []
    for t in traces:
        items.append(
            {
                "prompt": str(t.get("artifact", t.get("prompt", ""))),
                "role": t.get("from", t.get("role", "agent")),
                "response": str(t.get("response", t.get("to", ""))),
                "reward": float(outcome),
                "off_policy": True,
            }
        )
    return items
