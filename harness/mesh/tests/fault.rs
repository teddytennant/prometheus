//! Group: fault injection — partition freeze/adverts, tick without heal,
//! isolated advertise does not leak, freeze-while-partitioned on the isolated node.

mod common;
mod reference;

use common::{
    advert_set, assert_frozen, caps_cpu, caps_gpu, fresh_base, good_freeze, item_plain, open_queue,
    start3, T0,
};
use reference::RefLocalMesh;

#[test]
fn tick_without_heal_does_not_freeze_isolated() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    mesh.partition(&[2]).expect("part");
    mesh.get(0)
        .expect("0")
        .freeze(good_freeze("halt", "key-0"))
        .expect("freeze");
    for _ in 0..4 {
        mesh.tick(T0).expect("tick");
        assert!(
            !mesh.get(2).expect("2").is_frozen(),
            "repeated tick without heal must not freeze isolated"
        );
    }
    assert!(mesh.get(0).expect("0").is_frozen());
    assert!(mesh.get(1).expect("1").is_frozen());
}

#[test]
fn isolated_advertise_does_not_reach_majority_until_heal_tick() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    let mut refer = RefLocalMesh::start(3).expect("ref");
    mesh.partition(&[2]).expect("part");
    refer.partition(&[2]).expect("ref part");
    mesh.get(2)
        .expect("2")
        .advertise(caps_gpu(9), T0)
        .expect("isolated adv");
    refer
        .get(2)
        .expect("r2")
        .advertise(caps_gpu(9), T0)
        .expect("ref isolated adv");
    mesh.tick(T0).expect("tick");
    refer.tick(T0).expect("ref tick");
    for i in 0..2 {
        let ads = mesh.get(i).expect("get").adverts();
        assert!(
            ads.iter().all(|a| a.node.0 != "2"),
            "majority node {i} must not see isolated advert"
        );
        assert_eq!(
            advert_set(ads),
            advert_set(refer.get(i).expect("ref").adverts())
        );
    }
    mesh.heal().expect("heal");
    refer.heal().expect("ref heal");
    mesh.tick(T0 + 1).expect("tick");
    refer.tick(T0 + 1).expect("ref tick");
    for i in 0..2 {
        let ads = mesh.get(i).expect("get").adverts();
        assert!(
            ads.iter().any(|a| a.node.0 == "2" && a.caps.gpus == 9),
            "after heal+tick node {i} sees isolated advert"
        );
    }
}

#[test]
fn freeze_on_isolated_node_spreads_after_heal_tick() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    mesh.partition(&[2]).expect("part");
    mesh.get(2)
        .expect("2")
        .freeze(good_freeze("halt", "key-2"))
        .expect("freeze isolated");
    assert!(mesh.get(2).expect("2").is_frozen());
    mesh.tick(T0).expect("tick");
    assert!(
        !mesh.get(0).expect("0").is_frozen(),
        "majority must not freeze from isolated until heal"
    );
    assert!(!mesh.get(1).expect("1").is_frozen());
    mesh.heal().expect("heal");
    mesh.tick(T0 + 1).expect("tick");
    assert!(mesh.get(0).expect("0").is_frozen());
    assert!(mesh.get(1).expect("1").is_frozen());
}

#[test]
fn frozen_connected_node_cannot_pull() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    mesh.get(0)
        .expect("0")
        .advertise(caps_cpu(), T0)
        .expect("adv");
    mesh.get(0)
        .expect("0")
        .freeze(good_freeze("halt", "key-0"))
        .expect("freeze");
    mesh.tick(T0).expect("tick");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("t1"), T0).expect("enq");
    let err = mesh
        .get(1)
        .expect("1")
        .pull(&mut queue, T0)
        .expect_err("peer frozen via gossip");
    assert_frozen(&err, "gossiped freeze pull");
}

#[test]
fn partition_replaces_isolated_set() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    mesh.partition(&[2]).expect("first");
    mesh.get(0)
        .expect("0")
        .advertise(caps_gpu(1), T0)
        .expect("adv");
    mesh.tick(T0).expect("tick");
    mesh.partition(&[1]).expect("replace");
    mesh.get(0)
        .expect("0")
        .advertise(caps_gpu(2), T0 + 1)
        .expect("adv2");
    mesh.tick(T0 + 1).expect("tick");
    // node 2 is no longer isolated; node 1 is.
    let ads2 = mesh.get(2).expect("2").adverts();
    assert!(
        ads2.iter().any(|a| a.node.0 == "0" && a.caps.gpus == 2),
        "node 2 (healed by replace) sees latest advert"
    );
    let ads1 = mesh.get(1).expect("1").adverts();
    assert!(
        ads1.iter().all(|a| a.caps.gpus != 2 || a.node.0 != "0"),
        "node 1 newly isolated must not see gpus=2 advert"
    );
}
