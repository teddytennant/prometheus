//! Group: Kernel vs independent `RefKernel` (not `prometheus_kernel::Kernel`).

mod common;
mod reference;

use common::{
    child_spawn, fresh_world, job, open_kernel, place_mirror, quota, rec, root_spawn, NOW0,
};
use prometheus_kernel::{PatchState, PatchTarget, Role, Rung, CANARY_MS};
use reference::RefKernel;

#[test]
fn spawn_send_job_ledger_fetch_promote_match_reference() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);

    let root_req = root_spawn(Role::Researcher, quota(500, 500, 500), "root");
    let hk = k.spawn(root_req.clone(), NOW0).unwrap();
    let hr = r.spawn(root_req, NOW0).unwrap();
    assert_eq!(hk.depth, hr.depth);
    assert_eq!(k.remaining(&hk.id).unwrap(), r.remaining(&hr.id).unwrap());

    let child_k = child_spawn(hk.id.clone(), Role::Engineer, quota(40, 30, 20), "child");
    let child_r = child_spawn(hr.id.clone(), Role::Engineer, quota(40, 30, 20), "child");
    let ck = k.spawn(child_k, NOW0).unwrap();
    let cr = r.spawn(child_r, NOW0).unwrap();
    assert_eq!(ck.depth, 1);
    assert_eq!(cr.depth, 1);
    assert_eq!(k.remaining(&hk.id).unwrap(), quota(460, 470, 480));
    assert_eq!(r.remaining(&hr.id).unwrap(), quota(460, 470, 480));

    k.send(&hk.id, &ck.id, "hello", NOW0).unwrap();
    r.send(&hr.id, &cr.id, "hello", NOW0).unwrap();
    assert_eq!(
        k.recv(&ck.id, 0, NOW0).unwrap().unwrap().body,
        r.recv(&cr.id, 0, NOW0).unwrap().unwrap().body
    );

    k.ledger_append(rec("p-check", "CHECK:prop")).unwrap();
    k.ledger_append(rec("p-prereg", "PREREG:prop")).unwrap();
    r.ledger_append(rec("p-check", "CHECK:prop")).unwrap();
    r.ledger_append(rec("p-prereg", "PREREG:prop")).unwrap();
    let qk = k.ledger_query("kind=check AND kind=prereg").unwrap();
    let qr = r.ledger_query("kind=check AND kind=prereg").unwrap();
    assert_eq!(qk.len(), qr.len());
    assert_eq!(qk.len(), 0, "AND of both kinds matches neither row");
    assert_eq!(k.ledger_query("kind=check").unwrap().len(), 1);
    assert_eq!(r.ledger_query("kind=check").unwrap().len(), 1);

    let jk = k
        .submit_job(
            job(ck.id.clone(), quota(1, 1, 1), 1, 1, Rung::R1, &["prop"]),
            NOW0,
        )
        .unwrap();
    let jr = r
        .submit_job(
            job(cr.id.clone(), quota(1, 1, 1), 1, 1, Rung::R1, &["prop"]),
            NOW0,
        )
        .unwrap();
    assert_eq!(jk.rung, jr.rung);
    assert_eq!(k.remaining(&ck.id).unwrap(), r.remaining(&cr.id).unwrap());

    let url = "https://github.com/prometheus/l1";
    place_mirror(&world, url, b"src");
    assert_eq!(k.fetch(url).unwrap(), r.fetch(url).unwrap());

    let pk = k
        .propose_patch(&hk.id, "ok", "l1-test", PatchTarget::Genome, NOW0)
        .unwrap();
    let pr = r
        .propose_patch(&hr.id, "ok", "l1-test", PatchTarget::Genome, NOW0)
        .unwrap();
    for _ in 0..3 {
        let sk = k.promote_genome(&pk, NOW0).unwrap();
        let sr = r.promote_genome(&pr, NOW0).unwrap();
        assert_eq!(sk, sr);
    }
    assert_eq!(k.patch(&pk).unwrap().state, PatchState::Canary);
    assert_eq!(
        k.promote_genome(&pk, NOW0 + CANARY_MS).unwrap(),
        r.promote_genome(&pr, NOW0 + CANARY_MS).unwrap()
    );
    assert_eq!(k.patch(&pk).unwrap().state, PatchState::RolledOut);
}

#[test]
fn covering_math_matches_reference_across_dimensions() {
    let world = fresh_world();
    let mut k = open_kernel(&world);
    let mut r = RefKernel::open(&world.cfg);
    let cases = [
        quota(10, 10, 10),
        quota(0, 5, 5),
        quota(5, 0, 5),
        quota(5, 5, 0),
        quota(1, 2, 3),
    ];
    for (i, parent_q) in cases.into_iter().enumerate() {
        let req = root_spawn(Role::Researcher, parent_q, "p");
        let hk = k.spawn(req.clone(), NOW0).unwrap();
        let hr = r.spawn(req, NOW0).unwrap();
        let need = quota(2, 2, 2);
        let ck = k.spawn(
            child_spawn(hk.id.clone(), Role::Researcher, need, "c"),
            NOW0,
        );
        let cr = r.spawn(
            child_spawn(hr.id.clone(), Role::Researcher, need, "c"),
            NOW0,
        );
        match (ck, cr) {
            (Ok(_), Ok(_)) => {
                assert_eq!(
                    k.remaining(&hk.id).unwrap(),
                    r.remaining(&hr.id).unwrap(),
                    "case {i} remaining"
                );
            }
            (Err(e1), Err(e2)) => match (e1, e2) {
                (
                    prometheus_kernel::Error::QuotaRefused(a),
                    prometheus_kernel::Error::QuotaRefused(b),
                ) => assert_eq!(a, b, "case {i}"),
                (
                    prometheus_kernel::Error::BudgetExhausted(a),
                    prometheus_kernel::Error::BudgetExhausted(b),
                ) => assert_eq!(a, b, "case {i}"),
                (a, b) => panic!("case {i} error mismatch {a:?} vs {b:?}"),
            },
            (a, b) => panic!("case {i} Ok/Err mismatch {a:?} vs {b:?}"),
        }
    }
}
