//! Group: freeze then pull/advertise Frozen; BadSignature rejected;
//! good freeze durable across open.

mod common;
mod reference;

use common::{
    assert_bad_signature, assert_frozen, assert_queued, caps_cpu, fresh_node_dir, good_freeze,
    bad_freeze, item_plain, open_queue, slot_config, T0,
};
use prometheus_mesh::Mesh;
use reference::{signature_ok, RefMesh};

#[test]
fn freeze_then_pull_is_frozen() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    mesh.freeze(good_freeze("halt", "key-0")).expect("freeze");
    assert!(mesh.is_frozen());
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("t1"), T0).expect("enq");
    let err = mesh.pull(&mut queue, T0).expect_err("frozen pull");
    assert_frozen(&err, "pull");
    assert_queued(
        queue
            .get(&prometheus_leases::TaskId("t1".into()))
            .as_ref()
            .unwrap(),
        "t1",
    );
}

#[test]
fn freeze_then_advertise_is_frozen() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv before freeze");
    mesh.freeze(good_freeze("halt", "key-0")).expect("freeze");
    let err = mesh
        .advertise(caps_cpu(), T0 + 1)
        .expect_err("frozen advertise");
    assert_frozen(&err, "advertise");
}

#[test]
fn bad_signature_rejected_not_frozen() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    let cmd = bad_freeze("halt", "key-0");
    assert!(!signature_ok(&cmd));
    let err = mesh.freeze(cmd).expect_err("bad sig");
    assert_bad_signature(&err);
    assert!(!mesh.is_frozen());
    mesh.advertise(caps_cpu(), T0).expect("still can advertise");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("t1"), T0).expect("enq");
    let lease = mesh.pull(&mut queue, T0).expect("pull").expect("claimed");
    assert_eq!(lease.task_id.0, "t1");
}

#[test]
fn good_freeze_durable_across_open() {
    let (_parent, dir) = fresh_node_dir();
    let cfg = slot_config(0);
    {
        let mut mesh = Mesh::create(&dir, cfg.clone()).expect("create");
        assert!(!mesh.is_frozen());
        mesh.freeze(good_freeze("stop-the-line", "key-0"))
            .expect("freeze");
        assert!(mesh.is_frozen());
    }
    let mut mesh = Mesh::open(&dir, cfg).expect("open");
    assert!(mesh.is_frozen(), "freeze must survive open");
    let err = mesh
        .advertise(caps_cpu(), T0)
        .expect_err("still frozen after open");
    assert_frozen(&err, "advertise after open");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("t1"), T0).expect("enq");
    let err = mesh.pull(&mut queue, T0).expect_err("pull after open");
    assert_frozen(&err, "pull after open");
}

#[test]
fn empty_signature_is_bad_unless_empty_payload() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    let err = mesh
        .freeze(prometheus_mesh::Freeze {
            payload: "x".into(),
            signature: vec![],
            signer: prometheus_mesh::PublicKey("k".into()),
        })
        .expect_err("empty sig");
    assert_bad_signature(&err);
    mesh.freeze(good_freeze("", "k")).expect("empty payload + empty sig");
    assert!(mesh.is_frozen());
}

#[test]
fn freeze_is_idempotent_with_good_signature() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.freeze(good_freeze("a", "key-0")).expect("first");
    mesh.freeze(good_freeze("b", "key-0")).expect("second");
    assert!(mesh.is_frozen());
}

#[test]
fn reference_freeze_matches_production_errors() {
    let mut refer = RefMesh::new(slot_config(0));
    refer
        .freeze(bad_freeze("halt", "key-0"))
        .expect_err("ref bad");
    assert!(!refer.is_frozen());
    refer.freeze(good_freeze("halt", "key-0")).expect("ref good");
    assert!(refer.is_frozen());
    let err = refer.advertise(caps_cpu(), T0).expect_err("ref frozen adv");
    assert_frozen(&err, "ref advertise");

    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.freeze(good_freeze("halt", "key-0")).expect("prod");
    assert_eq!(mesh.is_frozen(), refer.is_frozen());
}
