//! Group: submit_job, rung-0 vs higher (spec 14.6).

mod common;
mod reference;

use common::{
    assert_rung_not_cleared, fresh_world, job, open_kernel, quota, rec, root_spawn, unwrap_err,
    NOW0,
};
use prometheus_kernel::{Role, Rung};
use reference::RefKernel;

fn seed_rung_rows(k: &mut prometheus_kernel::Kernel, key: &str) {
    k.ledger_append(rec(&format!("{key}-check"), &format!("CHECK:{key}")))
        .expect("check row");
    k.ledger_append(rec(
        &format!("{key}-prereg"),
        &format!("PREREG:{key} hypothesis"),
    ))
    .expect("prereg row");
}

#[test]
fn r0_does_not_need_ledger_rows() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(
            root_spawn(Role::Researcher, quota(100, 100, 100), "p"),
            NOW0,
        )
        .expect("p");
    let jh = k
        .submit_job(job(h.id.clone(), quota(1, 1, 1), 1, 1, Rung::R0, &[]), NOW0)
        .expect("r0");
    assert_eq!(jh.agent, h.id);
    assert_eq!(jh.rung, Rung::R0);
    assert_eq!(jh.gpus, 1);
}

#[test]
fn r1_without_tags_is_not_cleared() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(
            root_spawn(Role::Researcher, quota(100, 100, 100), "p"),
            NOW0,
        )
        .expect("p");
    let err = unwrap_err(
        k.submit_job(job(h.id.clone(), quota(1, 1, 1), 1, 1, Rung::R1, &[]), NOW0),
        "no tags",
    );
    assert_rung_not_cleared(&err, Rung::R1);
    assert_eq!(k.remaining(&h.id).unwrap(), quota(100, 100, 100));
}

#[test]
fn r1_without_check_row_is_not_cleared() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(
            root_spawn(Role::Researcher, quota(100, 100, 100), "p"),
            NOW0,
        )
        .expect("p");
    k.ledger_append(rec("run-a-prereg", "PREREG:run-a"))
        .expect("prereg only");
    let err = unwrap_err(
        k.submit_job(
            job(h.id.clone(), quota(1, 1, 1), 1, 1, Rung::R1, &["run-a"]),
            NOW0,
        ),
        "no check",
    );
    assert_rung_not_cleared(&err, Rung::R1);
}

#[test]
fn r1_without_prereg_row_is_not_cleared() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(
            root_spawn(Role::Researcher, quota(100, 100, 100), "p"),
            NOW0,
        )
        .expect("p");
    k.ledger_append(rec("run-a-check", "CHECK:run-a"))
        .expect("check only");
    let err = unwrap_err(
        k.submit_job(
            job(h.id.clone(), quota(1, 1, 1), 1, 1, Rung::R1, &["run-a"]),
            NOW0,
        ),
        "no prereg",
    );
    assert_rung_not_cleared(&err, Rung::R1);
}

#[test]
fn r1_with_both_rows_is_accepted() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let req = root_spawn(Role::Researcher, quota(100, 100, 100), "p");
    let h = k.spawn(req.clone(), NOW0).expect("p");
    let hr = r.spawn(req, NOW0).expect("pr");
    seed_rung_rows(&mut k, "run-a");
    r.ledger_append(rec("run-a-check", "CHECK:run-a"))
        .expect("r check");
    r.ledger_append(rec("run-a-prereg", "PREREG:run-a hypothesis"))
        .expect("r prereg");
    let jk = k
        .submit_job(
            job(h.id.clone(), quota(2, 3, 4), 2, 4, Rung::R1, &["run-a"]),
            NOW0,
        )
        .expect("k job");
    let jr = r
        .submit_job(
            job(hr.id.clone(), quota(2, 3, 4), 2, 4, Rung::R1, &["run-a"]),
            NOW0,
        )
        .expect("r job");
    assert_eq!(jk.rung, Rung::R1);
    assert_eq!(jr.rung, Rung::R1);
    assert_eq!(k.remaining(&h.id).unwrap(), quota(98, 97, 96));
    assert_eq!(r.remaining(&hr.id).unwrap(), quota(98, 97, 96));
}

#[test]
fn r2_and_r3_use_the_same_clearance() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(
            root_spawn(Role::Researcher, quota(100, 100, 100), "p"),
            NOW0,
        )
        .expect("p");
    seed_rung_rows(&mut k, "run-b");
    k.submit_job(
        job(h.id.clone(), quota(1, 1, 1), 1, 1, Rung::R2, &["run-b"]),
        NOW0,
    )
    .expect("r2");
    k.submit_job(
        job(h.id.clone(), quota(1, 1, 1), 1, 1, Rung::R3, &["run-b"]),
        NOW0,
    )
    .expect("r3");
}

#[test]
fn wrong_key_does_not_clear() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(
            root_spawn(Role::Researcher, quota(100, 100, 100), "p"),
            NOW0,
        )
        .expect("p");
    seed_rung_rows(&mut k, "run-a");
    let err = unwrap_err(
        k.submit_job(
            job(h.id.clone(), quota(1, 1, 1), 1, 1, Rung::R1, &["run-b"]),
            NOW0,
        ),
        "wrong key",
    );
    assert_rung_not_cleared(&err, Rung::R1);
}

#[test]
fn failed_rung_does_not_charge() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let h = k
        .spawn(root_spawn(Role::Researcher, quota(50, 50, 50), "p"), NOW0)
        .expect("p");
    let _ = unwrap_err(
        k.submit_job(
            job(h.id.clone(), quota(10, 10, 10), 1, 1, Rung::R1, &["nope"]),
            NOW0,
        ),
        "rung",
    );
    assert_eq!(k.remaining(&h.id).unwrap(), quota(50, 50, 50));
}

#[test]
fn unknown_agent_job_is_not_found() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let ghost = prometheus_kernel::AgentId("ghost".into());
    let err = unwrap_err(
        k.submit_job(
            job(ghost.clone(), quota(0, 0, 0), 0, 0, Rung::R0, &[]),
            NOW0,
        ),
        "ghost",
    );
    common::assert_not_found_agent(&err, &ghost);
}
