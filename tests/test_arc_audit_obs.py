"""ARC, audit hash chain, obs tripwires."""

from arc import pass_at_k, rotate90
from audit import AuditLog
from obs import Registry, tripwire


def test_arc_pass_at_2():
    g = [[1, 0], [0, 1]]
    assert pass_at_k([rotate90([[0, 1], [1, 0]]), [[0]]], g, k=2)


def test_audit_chain():
    log = AuditLog()
    log.append({"kind": "train", "step": 1})
    log.append({"kind": "eval", "step": 1})
    assert log.verify()
    log.entries[0]["event"]["step"] = 99
    assert not log.verify()


def test_obs_tripwire():
    r = Registry()
    r.counter("tokens").inc(10)
    assert tripwire("tokens", r.counter("tokens").value, 5) is not None
    assert tripwire("tokens", 1, 5) is None
