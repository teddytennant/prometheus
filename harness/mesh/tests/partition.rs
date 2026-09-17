//! Group: partition isolates gossip; heal then tick applies freeze;
//! isolated nodes stay in the watcher ring.

mod common;
mod reference;

use common::{
    advert_set, caps_gpu, fresh_base, good_freeze, start3, watch_pairs, T0,
};
use prometheus_mesh::MIN_WATCHERS;
use reference::RefLocalMesh;

#[test]
fn partitioned_node_applies_freeze_on_tick_after_heal() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    let mut refer = RefLocalMesh::start(3).expect("ref");

    mesh.partition(&[2]).expect("partition");
    refer.partition(&[2]).expect("ref partition");

    mesh.get(0)
        .expect("0")
        .freeze(good_freeze("halt", "key-0"))
        .expect("freeze");
    refer
        .get(0)
        .expect("r0")
        .freeze(good_freeze("halt", "key-0"))
        .expect("ref freeze");
    assert!(mesh.get(0).expect("0").is_frozen());
    assert!(refer.get(0).expect("r0").is_frozen());

    mesh.tick(T0).expect("tick while partitioned");
    refer.tick(T0).expect("ref tick");
    assert!(mesh.get(1).expect("1").is_frozen(), "connected peer freezes");
    assert!(
        !mesh.get(2).expect("2").is_frozen(),
        "partitioned node must not freeze before heal"
    );
    assert!(!refer.get(2).expect("r2").is_frozen());

    mesh.heal().expect("heal");
    refer.heal().expect("ref heal");
    assert!(
        !mesh.get(2).expect("2").is_frozen(),
        "heal itself does not apply freeze"
    );

    mesh.tick(T0 + 1).expect("tick after heal");
    refer.tick(T0 + 1).expect("ref tick after heal");
    assert!(
        mesh.get(2).expect("2").is_frozen(),
        "partitioned node applies freeze on tick after heal"
    );
    assert!(refer.get(2).expect("r2").is_frozen());
}

#[test]
fn partitioned_node_does_not_see_adverts_until_heal_tick() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    let mut refer = RefLocalMesh::start(3).expect("ref");
    mesh.partition(&[2]).expect("part");
    refer.partition(&[2]).expect("ref part");
    mesh.get(0)
        .expect("0")
        .advertise(caps_gpu(3), T0)
        .expect("adv");
    refer
        .get(0)
        .expect("r0")
        .advertise(caps_gpu(3), T0)
        .expect("ref adv");
    mesh.tick(T0).expect("tick");
    refer.tick(T0).expect("ref tick");

    let ads1 = mesh.get(1).expect("1").adverts();
    assert!(
        ads1.iter().any(|a| a.node.0 == "0" && a.caps.gpus == 3),
        "connected node 1 sees advert"
    );
    let ads2 = mesh.get(2).expect("2").adverts();
    assert!(
        ads2.iter().all(|a| a.node.0 != "0"),
        "isolated node 2 must not see node 0 advert yet, got {ads2:?}"
    );
    assert_eq!(
        advert_set(ads2),
        advert_set(refer.get(2).expect("r2").adverts())
    );

    mesh.heal().expect("heal");
    refer.heal().expect("ref heal");
    mesh.tick(T0 + 1).expect("tick");
    refer.tick(T0 + 1).expect("ref tick");
    let ads2 = mesh.get(2).expect("2").adverts();
    assert!(
        ads2.iter().any(|a| a.node.0 == "0"),
        "after heal+tick isolated node sees advert"
    );
    assert_eq!(
        advert_set(ads2),
        advert_set(refer.get(2).expect("r2").adverts())
    );
}

#[test]
fn isolated_nodes_remain_in_watcher_ring() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    let before: Vec<(String, String)> = watch_pairs(mesh.get(0).expect("0").watches());
    assert_eq!(before.len(), 6);
    mesh.partition(&[2]).expect("part");
    let after = watch_pairs(mesh.get(0).expect("0").watches());
    assert_eq!(after, before, "partition must not drop ring entries");
    assert!(
        after.iter().any(|(_, watchee)| watchee == "2"),
        "isolated watchee still in ring"
    );
    assert_eq!(MIN_WATCHERS, 2);
    let count2 = after.iter().filter(|(_, w)| w == "2").count();
    assert_eq!(count2, MIN_WATCHERS);
}
