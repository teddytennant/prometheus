"""Multi-agent roles: researcher / implementer / reviewer (spec 9, D7)."""

from __future__ import annotations

from dataclasses import dataclass


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
    allowed = {
        ("researcher", "implementer"),
        ("implementer", "reviewer"),
        ("reviewer", "implementer"),
    }
    if (from_role, to_role) not in allowed:
        raise ValueError(f"bad handoff {from_role}->{to_role}")
    return {"from": from_role, "to": to_role, "artifact": artifact}
