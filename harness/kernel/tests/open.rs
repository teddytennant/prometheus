//! Group: `Kernel::open`, reopen persistence.

mod common;
mod reference;

use common::{fresh_world, open_kernel, quota, rec, root_spawn, NOW0};
use prometheus_kernel::{Kernel, Role};
use reference::RefKernel;

#[test]
fn open_returns_a_kernel() {
    let world = fresh_world();
    let _k = open_kernel(&world);
    let _r = RefKernel::open(&world.cfg);
}

#[test]
fn open_is_idempotent_on_empty_paths() {
    let world = fresh_world();
    drop(open_kernel(&world));
    let _k = Kernel::open(world.cfg.clone()).expect("reopen empty");
}

#[test]
fn reopen_restores_remaining_quota() {
    let world = fresh_world();
    let id;
    let left;
    {
        let mut k = open_kernel(&world);
        let h = k
            .spawn(
                root_spawn(Role::Researcher, quota(100, 50, 25), "root"),
                NOW0,
            )
            .expect("spawn");
        id = h.id;
        left = k.remaining(&id).expect("remaining");
        assert_eq!(left, quota(100, 50, 25));
    }
    let k = Kernel::open(world.cfg.clone()).expect("reopen");
    assert_eq!(k.remaining(&id).expect("restored remaining"), left);
}

#[test]
fn reopen_restores_ledger_rows() {
    let world = fresh_world();
    let row = rec("exp-reopen", "CHECK:reopen");
    {
        let mut k = open_kernel(&world);
        k.ledger_append(row.clone()).expect("append");
    }
    let k = Kernel::open(world.cfg.clone()).expect("reopen");
    let got = k.ledger_query("experiment_id=exp-reopen").expect("query");
    assert_eq!(got.len(), 1);
    common::assert_records_eq(&got[0], &row);
}
