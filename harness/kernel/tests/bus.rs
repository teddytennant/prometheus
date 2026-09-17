//! Group: send / recv message bus.

mod common;
mod reference;

use common::{fresh_world, open_kernel, quota, root_spawn, unwrap_err, NOW0};
use prometheus_kernel::{AgentId, Role};
use reference::RefKernel;

#[test]
fn send_recv_fifo_and_empty_is_none() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let a = k
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "a"), NOW0)
        .expect("a");
    let b = k
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "b"), NOW0)
        .expect("b");
    let ar = r
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "a"), NOW0)
        .expect("ar");
    let br = r
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "b"), NOW0)
        .expect("br");

    k.send(&a.id, &b.id, "one", NOW0).expect("s1");
    k.send(&a.id, &b.id, "two", NOW0 + 1).expect("s2");
    r.send(&ar.id, &br.id, "one", NOW0).expect("rs1");
    r.send(&ar.id, &br.id, "two", NOW0 + 1).expect("rs2");

    let m1 = k.recv(&b.id, 0, NOW0).expect("r1").expect("m1");
    let m2 = k.recv(&b.id, 50, NOW0 + 10).expect("r2").expect("m2");
    assert_eq!(m1.body, "one");
    assert_eq!(m1.from, a.id);
    assert_eq!(m1.to, b.id);
    assert_eq!(m1.created_ms, NOW0);
    assert_eq!(m2.body, "two");
    assert_eq!(m2.created_ms, NOW0 + 1);
    assert!(k.recv(&b.id, 100, NOW0).expect("empty").is_none());

    let rm1 = r.recv(&br.id, 0, NOW0).expect("rr1").expect("rm1");
    let rm2 = r.recv(&br.id, 50, NOW0).expect("rr2").expect("rm2");
    assert_eq!(rm1.body, m1.body);
    assert_eq!(rm2.body, m2.body);
    assert!(r.recv(&br.id, 0, NOW0).expect("rempty").is_none());
}

#[test]
fn recv_does_not_block_when_queue_empty() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = k
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "a"), NOW0)
        .expect("a");
    let start = std::time::Instant::now();
    let got = k.recv(&a.id, 24 * 3600 * 1000, NOW0).expect("recv");
    assert!(got.is_none());
    assert!(
        start.elapsed().as_secs() < 5,
        "recv must not sleep for timeout_ms of injected time"
    );
}

#[test]
fn send_unknown_from_or_to_is_not_found() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = k
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "a"), NOW0)
        .expect("a");
    let ghost = AgentId("ghost".into());
    common::assert_not_found_agent(
        &unwrap_err(k.send(&ghost, &a.id, "x", NOW0), "from"),
        &ghost,
    );
    common::assert_not_found_agent(&unwrap_err(k.send(&a.id, &ghost, "x", NOW0), "to"), &ghost);
    common::assert_not_found_agent(&unwrap_err(k.recv(&ghost, 0, NOW0), "recv"), &ghost);
}

#[test]
fn send_to_self_is_allowed() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = k
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "a"), NOW0)
        .expect("a");
    k.send(&a.id, &a.id, "loop", NOW0).expect("self");
    let m = k.recv(&a.id, 0, NOW0).expect("recv").expect("msg");
    assert_eq!(m.body, "loop");
    assert_eq!(m.from, a.id);
    assert_eq!(m.to, a.id);
}
