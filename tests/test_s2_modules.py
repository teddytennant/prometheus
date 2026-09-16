"""New S2 APIs (spec 9-13, 15.5). These fail on the old 30-line stubs."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from arc import (
    ArcEnv,
    ArcTask,
    color_permutation,
    consistent_program,
    dihedral_aug,
    execute,
    generate_task,
    held_out_task,
    inverse_aug,
    pass_at_k,
    rotate90,
    ttt_prefix,
)
from audit import AuditLog
from obs import (
    Histogram,
    Registry,
    RunRegistry,
    Tracer,
    check_tripwires,
    dashboard_payload,
    tripwire,
)
from rl.multiagent import (
    MessageBus,
    credit_assignment,
    default_crew,
    distill_back,
    handoff,
    topology,
)


def _sglang():
    path = Path(__file__).resolve().parents[1] / "sglang-fork" / "__init__.py"
    spec = importlib.util.spec_from_file_location("sglang_fork", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_illegal_handoff() -> None:
    with pytest.raises(ValueError, match="bad handoff"):
        handoff("researcher", "reviewer", "x")
    with pytest.raises(ValueError, match="bad handoff"):
        handoff("implementer", "researcher", "x")
    assert handoff("reviewer", "implementer", "nits")["to"] == "implementer"


def test_budget_enforcement() -> None:
    bus = MessageBus(default_crew())
    bus.send("researcher", "implementer", "spec", tokens=8000)
    with pytest.raises(ValueError, match="budget"):
        bus.send("researcher", "implementer", "more", tokens=1)
    bus.send("implementer", "reviewer", "diff", tokens=16_000)
    with pytest.raises(ValueError, match="budget"):
        bus.send("implementer", "reviewer", "again", tokens=1)


def test_credit_sums_to_outcome() -> None:
    traces = [
        {"from": "researcher", "tokens": 10},
        {"from": "implementer", "tokens": 30},
        {"from": "reviewer", "tokens": 10},
    ]
    credits = credit_assignment(1.5, traces=traces)
    assert credits.keys() == {"researcher", "implementer", "reviewer"}
    assert abs(sum(credits.values()) - 1.5) < 1e-12
    assert credits["implementer"] > credits["researcher"]
    equal = credit_assignment(3.0, roles=["a", "b", "c"])
    assert equal == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_topologies_and_distill() -> None:
    for kind in ("star", "ring", "hierarchy"):
        topo = topology(kind)
        assert topo["kind"] == kind
        assert len(topo["nodes"]) == 3
        assert topo["edges"]
    traces = [handoff("researcher", "implementer", "plan")]
    items = distill_back(traces, outcome=1.0, single_outcome=0.2)
    assert items and items[0]["off_policy"] is True
    assert distill_back(traces, outcome=0.1, single_outcome=0.5) == []


def test_dihedral_and_inverse() -> None:
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    augs = dihedral_aug(g)
    assert len(augs) == 8
    named = dihedral_aug(g, named=True)
    for name, tg in named:
        assert inverse_aug(tg, name) == g
    perm = {1: 9, 9: 1}
    colored = color_permutation(g, perm)
    assert colored[0][0] == 9
    assert color_permutation(colored, {9: 1, 1: 9}) == g


def test_dsl_consistent_program() -> None:
    task = generate_task(0, "rotate")
    prog = consistent_program(task)
    assert prog is not None
    assert all(execute(inp, prog) == out for inp, out in task.train)
    assert execute(task.test_in, prog) == task.test_out
    assert pass_at_k([task.test_out, [[0]]], task.test_out, k=2) == 1.0


def test_ttt_prefix_identity() -> None:
    task = generate_task(1, "identity")
    pred = ttt_prefix(task)
    assert pred == task.test_out


def test_arc_env_sparse_reward_and_bonus() -> None:
    task = generate_task(2, "rotate")
    env = ArcEnv(task)
    frame, reward, info = env.step({"op": "rotate", "k": 1})
    assert frame == task.test_out
    assert reward == 1.0
    assert info["bonus"] > 0
    _, reward2, info2 = env.step({"op": "submit"})
    assert reward2 == 1.0
    assert info2["bonus"] == 0.0


def test_held_out_never_mixed_into_generators() -> None:
    g = generate_task(3, "flip")
    h = held_out_task(3, "flip")
    assert g.held_out is False
    assert h.held_out is True
    assert g.test_in != h.test_in
    for s in range(8):
        assert generate_task(s).held_out is False


def test_audit_worm_export_probes_query() -> None:
    log = AuditLog()
    log.append({"kind": "train", "step": 1})
    t0 = log.entries[0]["event"]["time"]
    log.discrete_vs_latent(problem_id="p1", discrete="2+2=4", latent="4")
    log.thought_decode(thought_id="t1", decoded="add", gold="add")
    assert log.verify()
    exported = log.export()
    assert isinstance(exported, list) and exported[0]["event"]["kind"] == "train"
    by_kind = log.query(kind="thought_decode")
    assert len(by_kind) == 1
    by_time = log.query(since=t0)
    assert len(by_time) == 3
    log.entries[0]["event"]["step"] = 99
    assert log.verify() is False


def test_audit_worm_persist(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog()
    log.append({"kind": "a", "step": 0})
    log.persist(path)
    log.append({"kind": "b", "step": 1})
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    json.loads(lines[0])
    log.entries[0]["event"]["kind"] = "tamper"
    assert log.verify() is False
    disk = [json.loads(x) for x in path.read_text().splitlines()]
    assert disk[0]["event"]["kind"] == "a"


def test_obs_histogram_tracer_run_dashboard() -> None:
    r = Registry()
    r.counter("tok").inc(3)
    hist = r.histogram("lat")
    hist.observe(0.5)
    assert isinstance(hist, Histogram)
    assert sum(hist.counts) == 1
    tr = Tracer()
    sp = tr.start("fwd")
    tr.end(sp)
    assert sp.duration is not None and sp.duration >= 0
    runs = RunRegistry()
    run = runs.register("r0", {"lr": 1e-3}, git_sha="abc")
    assert len(run.config_hash) == 64
    runs.snapshot("r0", {"loss": 1.2})
    payload = dashboard_payload(r, tr, runs)
    json.dumps(payload)
    assert payload["counters"]["tok"] == 3
    assert payload["runs"][0]["id"] == "r0"
    assert tripwire("tokens", 10, 20) is None
    assert tripwire("tokens", 21, 20)["name"] == "tokens"
    hits = check_tripwires({"tokens": 2e6, "loss_spike": 0.1, "kl_drift": 0.5})
    names = {h["name"] for h in hits}
    assert names == {"tokens", "kl_drift"}


def test_sglang_request_kv_routing_mtp() -> None:
    s = _sglang()
    st = s.RequestState()
    st.append_token(7)
    st.append_thought([0.1, 0.2])
    assert st.verbal == [7]
    assert len(st.latent) == 1
    assert s.recurrence_bucket(1) == 1
    assert s.recurrence_bucket(3) == 4
    assert s.recurrence_bucket(16) == 16
    assert s.recurrence_bucket(99) == 16
    kv = s.KvStore()
    x = np.arange(8, dtype=np.float32)
    kv.put("k", x, "hbm")
    kv.swap("k", "hbm", "nvme")
    y = kv.restore("k", "nvme", "hbm")
    assert s.bitwise_equal(x, y)
    assert kv.match("k", "hbm", "nvme")
    cap = s.routing_capture([[0, 3], [1, 2]])
    assert cap.per_token == [[0, 3], [1, 2]]
    hooked = s.ttt_sidecar_hook(np.ones(3), prefix=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(hooked, [2.0, 3.0, 4.0])
    mtp = s.mtp_speculative(draft_tokens=[5, 6], target_tokens=[5, 9])
    assert mtp["n_draft"] == 2
    assert mtp["accepted"] == [5]
    probe = s.serving_probe()
    assert probe["hbm_host_nvme_match"] is True
    assert set(probe) >= {"logprob_max_abs_err", "hbm_host_nvme_match", "recurrence_buckets", "ok"}


def test_arc_task_color_counts_rotate90_kept() -> None:
    task = ArcTask(train=[([[1]], [[1]])], test_in=[[1, 2], [3, 4]], test_out=[[3, 1], [4, 2]])
    assert rotate90(task.test_in) == task.test_out
    from arc import color_counts

    assert color_counts([[1, 1], [2, 1]]) == {1: 3, 2: 1}


def test_forecasting_and_latent_scaling_via_evals() -> None:
    import evals

    assert evals.brier(1.0, 1.0) == 0.0
    assert math.isclose(evals.log_score(0.5, 1.0), math.log(0.5))
    curve = evals.latent_scaling(
        {1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4, 16: 0.5},
        0.35,
        {1: 10, 2: 20, 4: 40, 8: 80, 16: 160},
        80,
    )
    assert curve["matched_budget"] == 8
    assert curve["beats_discrete_at_matched_flop"] is True
