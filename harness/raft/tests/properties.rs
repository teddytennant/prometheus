//! Group: property tests. Random leader mutations on a 3-node group match the
//! in-memory reference (membership set, roles, token, kernel).

mod common;
mod reference;

use common::{
    assert_state_eq, ephemeral, err_kind, fresh_base, replicate, short_config, start3, this_id_of,
    tick_until_leader, voter, FAKE_TOKEN, FAKE_TOKEN_2,
};
use prometheus_raft::{Error, NodeId, Role};
use reference::RefGroup;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn pick(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn random_leader_ops_match_reference() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let mut now = 100_000u64;
    let mut rng = Lcg(0xC0FFEE);
    let mut workers: Vec<NodeId> = Vec::new();
    let mut wseq = 0u32;

    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);
    assert_state_eq(
        group.get(leader).expect("l").state(),
        &refer.committed_state(),
        "initial",
    );
    let _ = rleader;

    for step in 0..40 {
        now += 25;
        let leader = tick_until_leader(&mut group, now);
        refer.tick(now).expect("ref tick");
        let rleader = refer.leader_index().expect("ref leader");
        let ids = [
            this_id_of(&mut group, 0),
            this_id_of(&mut group, 1),
            this_id_of(&mut group, 2),
        ];
        let holder = &ids[rng.pick(3) as usize];
        let role = if rng.pick(2) == 0 {
            Role::Coordinator
        } else {
            Role::TokenBroker
        };

        match rng.pick(7) {
            0 => {
                let blob = if rng.pick(2) == 0 {
                    FAKE_TOKEN.to_vec()
                } else {
                    FAKE_TOKEN_2.to_vec()
                };
                let p = group.get(leader).expect("l").commit_token(blob.clone());
                let r = refer.commit_token(rleader, blob);
                assert_same_result(&p, &r, step, "commit_token");
            }
            1 => {
                let ver = format!("k{}", rng.pick(8));
                let p = group
                    .get(leader)
                    .expect("l")
                    .set_kernel_version(ver.clone());
                let r = refer.set_kernel_version(rleader, ver);
                assert_same_result(&p, &r, step, "set_kernel_version");
            }
            2 => {
                let p = group.get(leader).expect("l").claim_role(role, holder, now);
                let r = refer.claim_role(rleader, role, holder, now);
                assert_same_result(&p, &r, step, "claim_role");
            }
            3 => {
                let p = group
                    .get(leader)
                    .expect("l")
                    .heartbeat_role(role, holder, now);
                let r = refer.heartbeat_role(rleader, role, holder, now);
                assert_same_result(&p, &r, step, "heartbeat_role");
            }
            4 => {
                let p = group.get(leader).expect("l").expire_roles(now);
                let r = refer.expire_roles(rleader, now);
                assert_same_result(&p, &r, step, "expire_roles");
            }
            5 => {
                wseq += 1;
                let w = ephemeral(&format!("w{wseq}"), &format!("local:w{wseq}"), true);
                let p = group.get(leader).expect("l").add_worker(w.clone());
                let r = refer.add_worker(rleader, w.clone());
                assert_same_result(&p, &r, step, "add_worker");
                if p.is_ok() {
                    workers.push(w.id);
                }
            }
            _ => {
                if let Some(id) = workers.last().cloned() {
                    let p = group.get(leader).expect("l").remove_node(&id);
                    let r = refer.remove_node(rleader, &id);
                    assert_same_result(&p, &r, step, "remove_worker");
                    if p.is_ok() {
                        workers.pop();
                    }
                } else {
                    let extra = voter("ghost-4", "local:ghost-4");
                    let p = group.get(leader).expect("l").add_voter(extra.clone());
                    let r = refer.add_voter(rleader, extra);
                    assert_same_result(&p, &r, step, "add_voter extra");
                    if p.is_ok() {
                        let id = common::node_id("ghost-4");
                        let p2 = group.get(leader).expect("l").remove_node(&id);
                        let r2 = refer.remove_node(rleader, &id);
                        assert_same_result(&p2, &r2, step, "remove extra voter");
                    }
                }
            }
        }

        replicate(&mut group, now);
        refer.tick(now).expect("ref replicate");
        let leader = tick_until_leader(&mut group, now);
        assert_state_eq(
            group.get(leader).expect("l").state(),
            &refer.committed_state(),
            &format!("step {step}"),
        );
    }
}

fn assert_same_result<T: std::fmt::Debug + PartialEq>(
    prod: &Result<T, Error>,
    refer: &Result<T, Error>,
    step: i32,
    what: &str,
) {
    match (prod, refer) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "step {step} {what} value"),
        (Err(a), Err(b)) => assert_eq!(err_kind(a), err_kind(b), "step {step} {what} err"),
        (Ok(a), Err(b)) => panic!("step {step} {what}: prod Ok({a:?}) ref Err({b:?})"),
        (Err(a), Ok(b)) => panic!("step {step} {what}: prod Err({a:?}) ref Ok({b:?})"),
    }
}

#[test]
fn at_most_one_leader_after_stable_election() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 1_000;
    let _ = tick_until_leader(&mut group, now);
    let leaders = common::leader_indices(&mut group);
    assert_eq!(leaders.len(), 1, "stable 3-node election: {leaders:?}");
}
