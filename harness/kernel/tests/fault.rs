//! Group: fault injection (refuse closed, no partial charge, no escape).

mod common;
mod reference;

use common::{
    child_spawn, fresh_world, job, open_kernel, quota, rec, root_spawn, unwrap_err, NOW0,
};
use prometheus_kernel::{AgentId, PatchId, Role, Rung, TicketId};

#[test]
fn remaining_after_failed_spawn_is_unchanged() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let p = k
        .spawn(root_spawn(Role::Researcher, quota(9, 8, 7), "p"), NOW0)
        .unwrap();
    let before = k.remaining(&p.id).unwrap();
    let _ = unwrap_err(
        k.spawn(
            child_spawn(p.id.clone(), Role::Researcher, quota(10, 1, 1), "no"),
            NOW0,
        ),
        "over",
    );
    assert_eq!(k.remaining(&p.id).unwrap(), before);
}

#[test]
fn remaining_after_failed_job_is_unchanged() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let p = k
        .spawn(root_spawn(Role::Researcher, quota(9, 8, 7), "p"), NOW0)
        .unwrap();
    let before = k.remaining(&p.id).unwrap();
    let _ = unwrap_err(
        k.submit_job(
            job(p.id.clone(), quota(1, 1, 1), 1, 1, Rung::R1, &["no-rows"]),
            NOW0,
        ),
        "rung",
    );
    assert_eq!(k.remaining(&p.id).unwrap(), before);
}

#[test]
fn ledger_duplicate_does_not_drop_the_first_row() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    k.ledger_append(rec("dup", "first")).unwrap();
    let _ = unwrap_err(k.ledger_append(rec("dup", "second")), "dup");
    let got = k.ledger_query("experiment_id=dup").unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].hypothesis, "first");
}

#[test]
fn lookups_on_garbage_ids_are_not_found() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let ghost = AgentId("".into());
    // Empty id is still unknown (open does not spawn anyone).
    match k.remaining(&ghost) {
        Err(prometheus_kernel::Error::AgentNotFound(_)) => {}
        other => panic!("expected AgentNotFound, got {other:?}"),
    }
    match k.patch(&PatchId("".into())) {
        Err(prometheus_kernel::Error::PatchNotFound(_)) => {}
        other => panic!("expected PatchNotFound, got {other:?}"),
    }
    match k.ticket(&TicketId("".into())) {
        Err(prometheus_kernel::Error::TicketNotFound(_)) => {}
        other => panic!("expected TicketNotFound, got {other:?}"),
    }
}

#[test]
fn fetch_nul_and_empty_are_not_mirrored() {
    let world = fresh_world();
    let k = open_kernel(&world);
    for url in ["", "https://", "https://arxiv.org/\0x"] {
        match k.fetch(url) {
            Err(prometheus_kernel::Error::NotMirrored(_)) => {}
            other => panic!("expected NotMirrored for {url:?}, got {other:?}"),
        }
    }
}
