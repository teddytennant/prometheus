//! Group: fault injection — not-found, timeout, bad tool request schema.

mod common;
mod reference;

use common::{python, sample_image, src_does_not_import_tests};
use prometheus_envs::{
    Backend, Error, ImageId, Pool, SandboxId, SnapshotId, Tool, ToolRequest, SCHEMA_TOOL_REQUEST,
    SCHEMA_VERSION,
};
use reference::RefPool;

fn pair() -> (
    Pool<prometheus_envs::InProcess>,
    RefPool,
    prometheus_envs::SandboxId,
    prometheus_envs::SandboxId,
    prometheus_envs::SnapshotId,
) {
    src_does_not_import_tests();
    let cfg = common::cpu_cfg();
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-fault");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    let a = prod.fork_group(&ps, 1, 0).unwrap().sandboxes[0].clone();
    let ra = refer.fork_group(&rs, 1, 0).unwrap().sandboxes[0].clone();
    (prod, refer, a, ra, ps)
}

#[test]
fn snapshot_not_found() {
    let (mut prod, mut refer, _, _, _) = pair();
    let missing = SnapshotId("no-such-snap".into());
    assert_eq!(
        prod.fork_group(&missing, 1, 0),
        Err(Error::SnapshotNotFound)
    );
    assert_eq!(
        refer.fork_group(&missing, 1, 0),
        Err(Error::SnapshotNotFound)
    );
}

#[test]
fn sandbox_not_found() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let missing = SandboxId("no-such-sb".into());
    assert_eq!(prod.kill(&missing, 0), Err(Error::SandboxNotFound));
    assert_eq!(refer.kill(&missing, 0), Err(Error::SandboxNotFound));
    prod.kill(&a, 1).unwrap();
    refer.kill(&ra, 1).unwrap();
    assert_eq!(
        prod.call(&a, python("x", &ps.0, "print(1)"), 2),
        Err(Error::SandboxNotFound)
    );
    assert_eq!(
        refer.call(&ra, python("x", &ps.0, "print(1)"), 2),
        Err(Error::SandboxNotFound)
    );
}

#[test]
fn image_not_found() {
    let (mut prod, mut refer, _, _, _) = pair();
    let missing = ImageId("no-such-img".into());
    assert_eq!(
        prod.snapshot_from_image(&missing, 0),
        Err(Error::ImageNotFound)
    );
    assert_eq!(
        refer.snapshot_from_image(&missing, 0),
        Err(Error::ImageNotFound)
    );
}

#[test]
fn bad_tool_request_wrong_schema_id() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let mut req = ToolRequest::new("bad-id", ps.0.clone(), Tool::Python);
    req.payload = serde_json::json!({"code": "print(1)"});
    req.schema_id = "prometheus.not_a_tool".into();
    match prod.call(&a, req.clone(), 0) {
        Err(Error::BadToolRequest(msg)) => assert!(!msg.is_empty()),
        other => panic!("prod {other:?}"),
    }
    match refer.call(&ra, req, 0) {
        Err(Error::BadToolRequest(_)) => {}
        other => panic!("ref {other:?}"),
    }
}

#[test]
fn bad_tool_request_wrong_schema_version() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let mut req = ToolRequest::new("bad-ver", ps.0.clone(), Tool::Python);
    req.payload = serde_json::json!({"code": "print(1)"});
    req.schema_version = SCHEMA_VERSION + 1;
    assert_ne!(req.schema_id, "");
    assert_eq!(req.schema_id, SCHEMA_TOOL_REQUEST);
    match prod.call(&a, req.clone(), 0) {
        Err(Error::BadToolRequest(_)) => {}
        other => panic!("prod {other:?}"),
    }
    match refer.call(&ra, req, 0) {
        Err(Error::BadToolRequest(_)) => {}
        other => panic!("ref {other:?}"),
    }
}

#[test]
fn in_process_backend_methods_exist() {
    let mut backend = prometheus_envs::InProcess::new(common::cpu_cfg());
    let img = sample_image("img-be");
    let sb = backend.boot(&img, 0).expect("boot");
    let snap = backend.snapshot(&sb, 1).expect("snapshot");
    let (forked, ms) = backend.fork(&snap, 2).expect("fork");
    assert!(ms <= prometheus_envs::FORK_BUDGET_MS);
    let _ = backend.list(&forked, prometheus_envs::View::Agent).unwrap();
    backend.kill(&forked, 3).unwrap();
}
