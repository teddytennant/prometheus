//! Group: escalate / ticket.

mod common;
mod reference;

use common::{
    assert_ticket_not_found, fresh_world, open_kernel, quota, root_spawn, unwrap_err, NOW0,
};
use prometheus_kernel::{AgentId, Role, TicketId};
use reference::RefKernel;

#[test]
fn escalate_opens_a_ticket() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let a = k
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "a"), NOW0)
        .unwrap();
    let ar = r
        .spawn(root_spawn(Role::Researcher, quota(1, 1, 1), "a"), NOW0)
        .unwrap();
    let id = k
        .escalate(
            &a.id,
            "grader disagrees with spec 14.3",
            "log line 12",
            NOW0,
        )
        .expect("escalate");
    let ir = r
        .escalate(
            &ar.id,
            "grader disagrees with spec 14.3",
            "log line 12",
            NOW0,
        )
        .expect("r escalate");
    let t = k.ticket(&id).expect("ticket");
    let tr = r.ticket(&ir).expect("r ticket");
    assert_eq!(t.author, a.id);
    assert_eq!(t.summary, "grader disagrees with spec 14.3");
    assert_eq!(t.evidence, "log line 12");
    assert_eq!(tr.summary, t.summary);
    assert_eq!(tr.evidence, t.evidence);
}

#[test]
fn escalate_unknown_author_is_not_found() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let ghost = AgentId("ghost".into());
    common::assert_not_found_agent(
        &unwrap_err(k.escalate(&ghost, "x", "evidence", NOW0), "ghost"),
        &ghost,
    );
}

#[test]
fn unknown_ticket_is_not_found() {
    let world = fresh_world();
    let k = open_kernel(&world);
    assert_ticket_not_found(
        &unwrap_err(k.ticket(&TicketId("t-missing".into())), "missing"),
        "t-missing",
    );
}

#[test]
fn tickets_are_unique_and_independent() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let a = k
        .spawn(root_spawn(Role::Engineer, quota(1, 1, 1), "a"), NOW0)
        .unwrap();
    let t1 = k.escalate(&a.id, "one", "e1", NOW0).unwrap();
    let t2 = k.escalate(&a.id, "two", "e2", NOW0).unwrap();
    assert_ne!(t1, t2);
    assert_eq!(k.ticket(&t1).unwrap().summary, "one");
    assert_eq!(k.ticket(&t2).unwrap().summary, "two");
    assert_eq!(k.ticket(&t1).unwrap().evidence, "e1");
    assert_eq!(k.ticket(&t2).unwrap().evidence, "e2");
}
