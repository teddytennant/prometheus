//! Group: untrusted Mesh::create + pull is Untrusted and does not claim.

mod common;
mod reference;

use common::{
    assert_queued, assert_untrusted, caps_cpu, fresh_node_dir, item_plain, open_queue,
    untrusted_config, T0,
};
use prometheus_leases::TaskState;
use prometheus_mesh::Mesh;
use reference::RefMesh;

#[test]
fn untrusted_create_then_pull_is_untrusted() {
    let (_parent, dir) = fresh_node_dir();
    let cfg = untrusted_config("evil", "key-evil");
    let mut mesh = Mesh::create(&dir, cfg.clone()).expect("create untrusted");
    assert!(!mesh.config().trusted);
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("t1"), T0).expect("enqueue");
    let err = mesh.pull(&mut queue, T0).expect_err("untrusted pull");
    assert_untrusted(&err, "pull");
    let task = queue
        .get(&prometheus_leases::TaskId("t1".into()))
        .expect("t1");
    assert_queued(task, "t1");
    assert_eq!(task.state, TaskState::Queued);

    let mut refer = RefMesh::new(cfg);
    let err = refer.pull(&mut queue, T0).expect_err("ref untrusted pull");
    assert_untrusted(&err, "pull");
}

#[test]
fn untrusted_pull_without_work_is_still_untrusted() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, untrusted_config("u", "key-u")).expect("create");
    let (_qparent, mut queue) = open_queue();
    let err = mesh.pull(&mut queue, T0).expect_err("empty untrusted pull");
    assert_untrusted(&err, "pull");
}

#[test]
fn untrusted_may_advertise_but_cannot_pull() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, untrusted_config("u", "key-u")).expect("create");
    // Advertise is not locked as Untrusted; pull is.
    let _ = mesh.advertise(caps_cpu(), T0);
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("t1"), T0).expect("enqueue");
    let err = mesh.pull(&mut queue, T0).expect_err("pull");
    assert_untrusted(&err, "pull");
    assert_queued(
        queue
            .get(&prometheus_leases::TaskId("t1".into()))
            .as_ref()
            .unwrap(),
        "t1",
    );
}
