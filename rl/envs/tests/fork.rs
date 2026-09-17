//! Group: snapshot, `fork_group(n=16)`, `fork_pair`, kill, live_count, capacity.

mod common;
mod reference;

use common::{sample_image, src_does_not_import_tests, WORKSPACE_README};
use prometheus_envs::{Error, Pool, GRPO_GROUP};
use reference::RefPool;

fn boot(
    capacity: usize,
) -> (
    Pool<prometheus_envs::InProcess>,
    RefPool,
    prometheus_envs::SnapshotId,
    prometheus_envs::SnapshotId,
) {
    src_does_not_import_tests();
    let cfg = common::cfg_cap(capacity);
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-fork");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    assert_eq!(pid, rid);
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    (prod, refer, ps, rs)
}

#[test]
fn snapshot_from_image_then_fork_group_16_identical_agent_views() {
    let (mut prod, mut refer, ps, rs) = boot(GRPO_GROUP);
    assert_eq!(prod.live_count(), 0);
    assert_eq!(refer.live_count(), 0);
    let pg = prod
        .fork_group(&ps, GRPO_GROUP, 0)
        .expect("prod fork_group");
    let rg = refer
        .fork_group(&rs, GRPO_GROUP, 0)
        .expect("ref fork_group");
    assert_eq!(pg.sandboxes.len(), GRPO_GROUP);
    assert_eq!(rg.sandboxes.len(), GRPO_GROUP);
    assert_eq!(pg.durations_ms.len(), GRPO_GROUP);
    assert!(pg.all_under_budget());
    assert!(rg.all_under_budget());
    assert_eq!(prod.live_count(), GRPO_GROUP);
    assert_eq!(refer.live_count(), GRPO_GROUP);
    let first = prod.agent_view(&pg.sandboxes[0]).unwrap();
    for sb in &pg.sandboxes {
        assert_eq!(prod.agent_view(sb).unwrap(), first);
    }
    let rfirst = refer.agent_view(&rg.sandboxes[0]).unwrap();
    assert_eq!(first, rfirst);
    assert_eq!(
        first.get(WORKSPACE_README).map(Vec::as_slice),
        Some(common::README_BODY)
    );
}

#[test]
fn live_plus_n_gt_capacity_is_pool_exhausted() {
    let (mut prod, mut refer, ps, rs) = boot(GRPO_GROUP);
    let _ = prod.fork_group(&ps, GRPO_GROUP, 0).unwrap();
    let _ = refer.fork_group(&rs, GRPO_GROUP, 0).unwrap();
    assert_eq!(
        prod.fork_group(&ps, 1, 1),
        Err(Error::PoolExhausted {
            capacity: GRPO_GROUP
        })
    );
    assert_eq!(
        refer.fork_group(&rs, 1, 1),
        Err(Error::PoolExhausted {
            capacity: GRPO_GROUP
        })
    );
}

#[test]
fn fork_group_n_gt_capacity_from_empty_is_exhausted() {
    let (mut prod, mut refer, ps, rs) = boot(4);
    assert_eq!(
        prod.fork_group(&ps, 5, 0),
        Err(Error::PoolExhausted { capacity: 4 })
    );
    assert_eq!(
        refer.fork_group(&rs, 5, 0),
        Err(Error::PoolExhausted { capacity: 4 })
    );
    assert_eq!(prod.live_count(), 0);
    assert_eq!(refer.live_count(), 0);
}

#[test]
fn fork_pair_two_distinct_sandboxes_from_one_snapshot() {
    let (mut prod, mut refer, ps, rs) = boot(8);
    let (a, b) = prod.fork_pair(&ps, 0).expect("prod pair");
    let (ra, rb) = refer.fork_pair(&rs, 0).expect("ref pair");
    assert_ne!(a, b);
    assert_ne!(ra, rb);
    assert_eq!(prod.live_count(), 2);
    assert_eq!(refer.live_count(), 2);
    assert_eq!(prod.agent_view(&a).unwrap(), prod.agent_view(&b).unwrap());
    assert_eq!(
        refer.agent_view(&ra).unwrap(),
        refer.agent_view(&rb).unwrap()
    );
    assert_eq!(prod.agent_view(&a).unwrap(), refer.agent_view(&ra).unwrap());
}

#[test]
fn mutating_one_sandbox_does_not_mutate_the_other() {
    let (mut prod, mut refer, ps, rs) = boot(8);
    let (a, b) = prod.fork_pair(&ps, 0).unwrap();
    let (ra, rb) = refer.fork_pair(&rs, 0).unwrap();
    let write = common::editor_write("c1", &ps.0, "/workspace/readme.txt", "mutated");
    prod.call(&a, write.clone(), 10).expect("prod write");
    refer.call(&ra, write, 10).expect("ref write");
    let pa = prod.agent_view(&a).unwrap();
    let pb = prod.agent_view(&b).unwrap();
    assert_eq!(
        pa.get(WORKSPACE_README).map(Vec::as_slice),
        Some(&b"mutated"[..])
    );
    assert_eq!(
        pb.get(WORKSPACE_README).map(Vec::as_slice),
        Some(common::README_BODY)
    );
    let ra_v = refer.agent_view(&ra).unwrap();
    let rb_v = refer.agent_view(&rb).unwrap();
    assert_eq!(pa, ra_v);
    assert_eq!(pb, rb_v);
}

#[test]
fn kill_drops_live_count_and_frees_capacity() {
    let (mut prod, mut refer, ps, rs) = boot(2);
    let (a, b) = prod.fork_pair(&ps, 0).unwrap();
    let (ra, rb) = refer.fork_pair(&rs, 0).unwrap();
    prod.kill(&a, 3).unwrap();
    refer.kill(&ra, 3).unwrap();
    assert_eq!(prod.live_count(), 1);
    assert_eq!(refer.live_count(), 1);
    prod.kill(&b, 4).unwrap();
    refer.kill(&rb, 4).unwrap();
    assert_eq!(prod.live_count(), 0);
    assert_eq!(refer.live_count(), 0);
    let g = prod.fork_group(&ps, 2, 5).expect("reuse capacity");
    let _ = refer.fork_group(&rs, 2, 5).unwrap();
    assert_eq!(g.sandboxes.len(), 2);
    assert_eq!(prod.live_count(), 2);
}

#[test]
fn pool_config_roundtrip_via_in_process() {
    let cfg = common::cpu_cfg();
    let pool = Pool::in_process(cfg.clone());
    assert_eq!(pool.config().capacity, cfg.capacity);
    assert_eq!(pool.config().default_timeout_s, cfg.default_timeout_s);
    assert!(pool.config().hold());
}
