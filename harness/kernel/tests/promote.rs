//! Group: propose_patch / patch / promote_genome / promote_weights.

mod common;
mod reference;

use common::{
    assert_no_quorum, assert_patch_not_found, fresh_world, open_kernel, quota, root_spawn,
    unwrap_err, NOW0, WEIGHT_SIGN_OFF,
};
use prometheus_kernel::{
    AgentId, HumanId, PatchId, PatchState, PatchTarget, Role, Signature, WeightPromoteState,
    CANARY_MS,
};
use reference::RefKernel;

fn researcher(k: &mut prometheus_kernel::Kernel) -> AgentId {
    k.spawn(root_spawn(Role::Researcher, quota(10, 10, 10), "r"), NOW0)
        .expect("researcher")
        .id
}

fn sig(human: &str, bytes: &[u8]) -> Signature {
    Signature {
        human: HumanId(human.into()),
        bytes: bytes.to_vec(),
    }
}

fn drive_to_canary(k: &mut prometheus_kernel::Kernel, id: &PatchId, now: u64) {
    assert_eq!(k.promote_genome(id, now).unwrap(), PatchState::SmokePassed);
    assert_eq!(k.promote_genome(id, now).unwrap(), PatchState::GatePassed);
    assert_eq!(k.promote_genome(id, now).unwrap(), PatchState::Canary);
}

#[test]
fn propose_then_patch_roundtrip() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let a = researcher(&mut k);
    let ar = r
        .spawn(root_spawn(Role::Researcher, quota(10, 10, 10), "r"), NOW0)
        .unwrap()
        .id;
    let id = k
        .propose_patch(&a, "diff-1", "l1-test", PatchTarget::Genome, NOW0)
        .expect("propose");
    let ir = r
        .propose_patch(&ar, "diff-1", "l1-test", PatchTarget::Genome, NOW0)
        .expect("r propose");
    let p = k.patch(&id).expect("patch");
    let pr = r.patch(&ir).expect("r patch");
    assert_eq!(p.target, PatchTarget::Genome);
    assert_eq!(p.diff, "diff-1");
    assert_eq!(p.rationale, "l1-test");
    assert_eq!(p.state, PatchState::Proposed);
    assert_eq!(p.author, a);
    assert_eq!(pr.state, PatchState::Proposed);
    assert_eq!(pr.diff, p.diff);
    assert_eq!(pr.rationale, p.rationale);
}

#[test]
fn propose_unknown_author_is_not_found() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let ghost = AgentId("ghost".into());
    common::assert_not_found_agent(
        &unwrap_err(
            k.propose_patch(&ghost, "d", "l1-test", PatchTarget::Synth, NOW0),
            "ghost",
        ),
        &ghost,
    );
}

#[test]
fn unknown_patch_is_not_found() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let id = PatchId("p-missing".into());
    assert_patch_not_found(&unwrap_err(k.patch(&id), "missing"), "p-missing");
}

#[test]
fn promote_unknown_patch_is_not_found() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let id = PatchId("p-missing".into());
    assert_patch_not_found(
        &unwrap_err(k.promote_genome(&id, NOW0), "promote missing"),
        "p-missing",
    );
}

#[test]
fn genome_steps_smoke_gate_canary_rollout() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let a = researcher(&mut k);
    let ar = r
        .spawn(root_spawn(Role::Researcher, quota(10, 10, 10), "r"), NOW0)
        .unwrap()
        .id;
    let id = k
        .propose_patch(&a, "ok-diff", "l1-test", PatchTarget::RlTasks, NOW0)
        .unwrap();
    let ir = r
        .propose_patch(&ar, "ok-diff", "l1-test", PatchTarget::RlTasks, NOW0)
        .unwrap();
    assert_eq!(
        k.promote_genome(&id, NOW0).unwrap(),
        PatchState::SmokePassed
    );
    assert_eq!(k.patch(&id).unwrap().state, PatchState::SmokePassed);
    assert_eq!(
        r.promote_genome(&ir, NOW0).unwrap(),
        PatchState::SmokePassed
    );
    assert_eq!(k.promote_genome(&id, NOW0).unwrap(), PatchState::GatePassed);
    assert_eq!(k.promote_genome(&id, NOW0).unwrap(), PatchState::Canary);
    assert_eq!(r.promote_genome(&ir, NOW0).unwrap(), PatchState::GatePassed);
    assert_eq!(r.promote_genome(&ir, NOW0).unwrap(), PatchState::Canary);
    // Still canary before CANARY_MS of injected time.
    assert_eq!(
        k.promote_genome(&id, NOW0 + CANARY_MS - 1).unwrap(),
        PatchState::Canary
    );
    assert_eq!(
        k.promote_genome(&id, NOW0 + CANARY_MS).unwrap(),
        PatchState::RolledOut
    );
    assert_eq!(k.patch(&id).unwrap().state, PatchState::RolledOut);
    assert_eq!(
        k.promote_genome(&id, NOW0 + CANARY_MS + 10).unwrap(),
        PatchState::RolledOut
    );
    assert_eq!(
        r.promote_genome(&ir, NOW0 + CANARY_MS).unwrap(),
        PatchState::RolledOut
    );
}

#[test]
fn fail_smoke_refuses() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = researcher(&mut k);
    let id = k
        .propose_patch(
            &a,
            "introduces FAIL_SMOKE",
            "l1-test",
            PatchTarget::Genome,
            NOW0,
        )
        .unwrap();
    assert_eq!(k.promote_genome(&id, NOW0).unwrap(), PatchState::Refused);
    assert_eq!(k.patch(&id).unwrap().state, PatchState::Refused);
    match k.promote_genome(&id, NOW0) {
        Err(prometheus_kernel::Error::PromoteRefused(_)) => {}
        other => panic!("expected PromoteRefused, got {other:?}"),
    }
}

#[test]
fn fail_gate_refuses_after_smoke() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = researcher(&mut k);
    let id = k
        .propose_patch(&a, "FAIL_GATE inside", "l1-test", PatchTarget::Recipe, NOW0)
        .unwrap();
    assert_eq!(
        k.promote_genome(&id, NOW0).unwrap(),
        PatchState::SmokePassed
    );
    assert_eq!(k.promote_genome(&id, NOW0).unwrap(), PatchState::Refused);
}

#[test]
fn fail_canary_reverts_after_injected_window() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = researcher(&mut k);
    let id = k
        .propose_patch(
            &a,
            "canary FAIL_CANARY",
            "l1-test",
            PatchTarget::Synth,
            NOW0,
        )
        .unwrap();
    drive_to_canary(&mut k, &id, NOW0);
    assert_eq!(
        k.promote_genome(&id, NOW0 + CANARY_MS - 1).unwrap(),
        PatchState::Canary
    );
    assert_eq!(
        k.promote_genome(&id, NOW0 + CANARY_MS).unwrap(),
        PatchState::Reverted
    );
}

#[test]
fn canary_ignores_wall_clock() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = researcher(&mut k);
    let id = k
        .propose_patch(&a, "ok", "l1-test", PatchTarget::Genome, NOW0)
        .unwrap();
    drive_to_canary(&mut k, &id, 0);
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert_eq!(k.promote_genome(&id, 1).unwrap(), PatchState::Canary);
    assert_eq!(
        k.promote_genome(&id, CANARY_MS).unwrap(),
        PatchState::RolledOut
    );
}

#[test]
fn weights_need_two_distinct_nonempty_humans() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let _ = researcher(&mut k);
    let err = unwrap_err(
        k.promote_weights("ckpt-a", &[sig("alice", b"sig")], NOW0),
        "one human",
    );
    assert_no_quorum(&err, 1, WEIGHT_SIGN_OFF);
    let err = unwrap_err(
        k.promote_weights(
            "ckpt-a",
            &[sig("alice", b"x"), sig("alice", b"y"), sig("bob", b"")],
            NOW0,
        ),
        "dup + empty",
    );
    assert_no_quorum(&err, 1, WEIGHT_SIGN_OFF);
    assert_eq!(
        k.promote_weights("ckpt-a", &[sig("alice", b"x"), sig("bob", b"y")], NOW0)
            .unwrap(),
        WeightPromoteState::Serving
    );
    assert_eq!(
        k.promote_weights("ckpt-a", &[], NOW0).unwrap(),
        WeightPromoteState::Serving
    );
}

#[test]
fn weights_fail_evals_or_bench_refuse() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let sigs = [sig("alice", b"x"), sig("bob", b"y")];
    assert_eq!(
        k.promote_weights("ckpt-FAIL_EVALS", &sigs, NOW0).unwrap(),
        WeightPromoteState::Refused
    );
    assert_eq!(
        k.promote_weights("ckpt-FAIL_BENCH", &sigs, NOW0).unwrap(),
        WeightPromoteState::Refused
    );
}

#[test]
fn empty_checkpoint_is_promote_refused() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    match k.promote_weights("", &[sig("a", b"1"), sig("b", b"2")], NOW0) {
        Err(prometheus_kernel::Error::PromoteRefused(s)) => assert!(!s.is_empty()),
        other => panic!("expected PromoteRefused, got {other:?}"),
    }
}

#[test]
fn all_patch_targets_are_promotable() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = researcher(&mut k);
    for target in [
        PatchTarget::Genome,
        PatchTarget::RlTasks,
        PatchTarget::Synth,
        PatchTarget::Recipe,
    ] {
        let id = k.propose_patch(&a, "ok", "l1-test", target, NOW0).unwrap();
        drive_to_canary(&mut k, &id, NOW0);
        assert_eq!(
            k.promote_genome(&id, NOW0 + CANARY_MS).unwrap(),
            PatchState::RolledOut
        );
    }
}
