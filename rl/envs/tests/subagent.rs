//! Group: subagent tool forks a child sandbox; `parent_call_id` links.

mod common;
mod reference;

use common::{editor_write, sample_image, src_does_not_import_tests, subagent, WORKSPACE_README};
use prometheus_envs::{Error, Pool, SandboxId};
use reference::RefPool;

fn pair(
    cap: usize,
) -> (
    Pool<prometheus_envs::InProcess>,
    RefPool,
    prometheus_envs::SandboxId,
    prometheus_envs::SandboxId,
    prometheus_envs::SnapshotId,
) {
    src_does_not_import_tests();
    let cfg = common::cfg_cap(cap);
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-sub");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    let pg = prod.fork_group(&ps, 1, 0).unwrap();
    let rg = refer.fork_group(&rs, 1, 0).unwrap();
    (
        prod,
        refer,
        pg.sandboxes[0].clone(),
        rg.sandboxes[0].clone(),
        ps,
    )
}

#[test]
fn subagent_forks_child_and_parent_call_id_links() {
    let (mut prod, mut refer, parent, rparent, ps) = pair(8);
    let req = subagent("call-sub", &ps.0, "solve piece", Some("call-parent"));
    assert_eq!(req.parent_call_id.as_deref(), Some("call-parent"));
    let pr = prod.call(&parent, req.clone(), 7).unwrap();
    let rr = refer.call(&rparent, req, 7).unwrap();
    assert!(pr.ok && rr.ok);
    let child = SandboxId(
        pr.stdout_artifact
            .path
            .clone()
            .expect("child sandbox id in stdout_artifact.path"),
    );
    let rchild = SandboxId(rr.stdout_artifact.path.clone().expect("ref child id"));
    assert_ne!(child, parent);
    assert_eq!(prod.live_count(), 2);
    assert_eq!(refer.live_count(), 2);
    let w = editor_write("mut-child", &ps.0, WORKSPACE_README, "from-child");
    prod.call(&child, w.clone(), 8).unwrap();
    refer.call(&rchild, w, 8).unwrap();
    assert_eq!(
        prod.agent_view(&child)
            .unwrap()
            .get(WORKSPACE_README)
            .map(Vec::as_slice),
        Some(&b"from-child"[..])
    );
    assert_eq!(
        prod.agent_view(&parent)
            .unwrap()
            .get(WORKSPACE_README)
            .map(Vec::as_slice),
        Some(common::README_BODY)
    );
    assert_eq!(
        prod.agent_view(&parent).unwrap(),
        refer.agent_view(&rparent).unwrap()
    );
}

#[test]
fn subagent_at_capacity_is_pool_exhausted() {
    let (mut prod, mut refer, parent, rparent, ps) = pair(1);
    let req = subagent("call-sub", &ps.0, "no room", Some("call-parent"));
    assert_eq!(
        prod.call(&parent, req.clone(), 0),
        Err(Error::PoolExhausted { capacity: 1 })
    );
    assert_eq!(
        refer.call(&rparent, req, 0),
        Err(Error::PoolExhausted { capacity: 1 })
    );
    assert_eq!(prod.live_count(), 1);
}
