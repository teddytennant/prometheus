//! Group: per-agent remaining quota, spawn/submit_job charges, hard stop.

mod common;
mod reference;

use common::{
    assert_budget_exhausted, assert_quota_refused, child_spawn, fresh_world, job, open_kernel,
    quota, root_spawn, unwrap_err, NOW0,
};
use prometheus_kernel::{Role, Rung};
use reference::RefKernel;

#[test]
fn remaining_unknown_agent_is_not_found() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let id = prometheus_kernel::AgentId("ghost".into());
    let err = unwrap_err(k.remaining(&id), "ghost");
    common::assert_not_found_agent(&err, &id);
}

#[test]
fn root_remaining_equals_requested_budget() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let req = root_spawn(Role::Researcher, quota(11, 22, 33), "b");
    let h = k.spawn(req.clone(), NOW0).expect("spawn");
    let hr = r.spawn(req, NOW0).expect("ref");
    assert_eq!(k.remaining(&h.id).unwrap(), quota(11, 22, 33));
    assert_eq!(r.remaining(&hr.id).unwrap(), quota(11, 22, 33));
}

#[test]
fn child_spawn_charges_parent_componentwise() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let parent_req = root_spawn(Role::Researcher, quota(100, 80, 60), "p");
    let p = k.spawn(parent_req.clone(), NOW0).expect("p");
    let pr = r.spawn(parent_req, NOW0).expect("pr");
    let child_req = child_spawn(p.id.clone(), Role::Researcher, quota(40, 10, 20), "c");
    // Reference uses its own parent id.
    let child_req_r = child_spawn(pr.id.clone(), Role::Researcher, quota(40, 10, 20), "c");
    let c = k.spawn(child_req, NOW0).expect("c");
    let cr = r.spawn(child_req_r, NOW0).expect("cr");
    assert_eq!(k.remaining(&c.id).unwrap(), quota(40, 10, 20));
    assert_eq!(k.remaining(&p.id).unwrap(), quota(60, 70, 40));
    assert_eq!(r.remaining(&cr.id).unwrap(), quota(40, 10, 20));
    assert_eq!(r.remaining(&pr.id).unwrap(), quota(60, 70, 40));
}

#[test]
fn spawn_refuses_when_tokens_cannot_cover() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let p = k
        .spawn(root_spawn(Role::Researcher, quota(5, 100, 100), "p"), NOW0)
        .expect("p");
    let err = unwrap_err(
        k.spawn(
            child_spawn(p.id.clone(), Role::Researcher, quota(6, 1, 1), "c"),
            NOW0,
        ),
        "tokens",
    );
    assert_quota_refused(&err, "tokens");
    assert_eq!(k.remaining(&p.id).unwrap(), quota(5, 100, 100));
}

#[test]
fn spawn_refuses_when_gpu_ms_cannot_cover() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let p = k
        .spawn(root_spawn(Role::Researcher, quota(100, 3, 100), "p"), NOW0)
        .expect("p");
    let err = unwrap_err(
        k.spawn(
            child_spawn(p.id.clone(), Role::Researcher, quota(1, 4, 1), "c"),
            NOW0,
        ),
        "gpu_ms",
    );
    assert_quota_refused(&err, "gpu_ms");
    assert_eq!(k.remaining(&p.id).unwrap(), quota(100, 3, 100));
}

#[test]
fn spawn_refuses_when_wall_ms_cannot_cover() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let p = k
        .spawn(root_spawn(Role::Researcher, quota(100, 100, 2), "p"), NOW0)
        .expect("p");
    let err = unwrap_err(
        k.spawn(
            child_spawn(p.id.clone(), Role::Researcher, quota(1, 1, 3), "c"),
            NOW0,
        ),
        "wall_ms",
    );
    assert_quota_refused(&err, "wall_ms");
}

#[test]
fn exhausted_parent_is_budget_exhausted() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let p = k
        .spawn(root_spawn(Role::Researcher, quota(0, 0, 0), "p"), NOW0)
        .expect("p");
    assert!(k.remaining(&p.id).unwrap().exhausted());
    let err = unwrap_err(
        k.spawn(
            child_spawn(p.id.clone(), Role::Researcher, quota(1, 0, 0), "c"),
            NOW0,
        ),
        "exhausted",
    );
    assert_budget_exhausted(&err);
}

#[test]
fn zero_charge_against_exhausted_parent_is_ok() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let p = k
        .spawn(root_spawn(Role::Researcher, quota(0, 0, 0), "p"), NOW0)
        .expect("p");
    let c = k
        .spawn(
            child_spawn(p.id.clone(), Role::Researcher, quota(0, 0, 0), "c"),
            NOW0,
        )
        .expect("zero child");
    assert_eq!(k.remaining(&c.id).unwrap(), quota(0, 0, 0));
}

#[test]
fn failed_depth_does_not_charge() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut parent = k
        .spawn(
            root_spawn(Role::Researcher, quota(1000, 1000, 1000), "p"),
            NOW0,
        )
        .expect("root");
    for _ in 0..3 {
        parent = k
            .spawn(
                child_spawn(parent.id.clone(), Role::Researcher, quota(1, 1, 1), "c"),
                NOW0,
            )
            .expect("chain");
    }
    assert_eq!(parent.depth, 3);
    let before = k.remaining(&parent.id).expect("leaf remaining");
    let err = unwrap_err(
        k.spawn(
            child_spawn(parent.id.clone(), Role::Researcher, quota(1, 1, 1), "nope"),
            NOW0,
        ),
        "depth",
    );
    common::assert_spawn_depth(&err, 4);
    assert_eq!(k.remaining(&parent.id).unwrap(), before);
}

#[test]
fn submit_job_charges_agent_budget() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let req = root_spawn(Role::Researcher, quota(100, 100, 100), "p");
    let h = k.spawn(req.clone(), NOW0).expect("k");
    let hr = r.spawn(req, NOW0).expect("r");
    let job_k = job(h.id.clone(), quota(10, 20, 30), 1, 30, Rung::R0, &[]);
    let job_r = job(hr.id.clone(), quota(10, 20, 30), 1, 30, Rung::R0, &[]);
    let jk = k.submit_job(job_k, NOW0).expect("job");
    let jr = r.submit_job(job_r, NOW0).expect("ref job");
    assert_eq!(jk.gpus, 1);
    assert_eq!(jk.wall_ms, 30);
    assert_eq!(jk.rung, Rung::R0);
    assert_eq!(jr.rung, Rung::R0);
    assert_eq!(k.remaining(&h.id).unwrap(), quota(90, 80, 70));
    assert_eq!(r.remaining(&hr.id).unwrap(), quota(90, 80, 70));
}

#[test]
fn submit_job_refuses_when_over_budget_and_does_not_charge() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(root_spawn(Role::Researcher, quota(5, 5, 5), "p"), NOW0)
        .expect("p");
    let err = unwrap_err(
        k.submit_job(job(h.id.clone(), quota(6, 1, 1), 1, 1, Rung::R0, &[]), NOW0),
        "over",
    );
    assert_quota_refused(&err, "tokens");
    assert_eq!(k.remaining(&h.id).unwrap(), quota(5, 5, 5));
}

#[test]
fn exact_cover_drains_remaining() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(root_spawn(Role::Researcher, quota(7, 8, 9), "p"), NOW0)
        .expect("p");
    k.submit_job(job(h.id.clone(), quota(7, 8, 9), 0, 9, Rung::R0, &[]), NOW0)
        .expect("exact");
    assert!(k.remaining(&h.id).unwrap().exhausted());
}
