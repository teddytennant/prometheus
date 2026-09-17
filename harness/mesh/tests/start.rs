//! Group: LocalMesh::start(3) slot layout + advert gossip after tick;
//! start(n<3) is TooFewNodes.

mod common;
mod reference;

use common::{
    advert_set, assert_too_few_nodes, caps_gpu, fresh_base, slot_config, start3, start_n,
    unwrap_err, T0,
};
use prometheus_mesh::{LocalMesh, MIN_RING};
use reference::RefLocalMesh;

#[test]
fn start_two_is_too_few_nodes() {
    let (_parent, dir) = fresh_base();
    assert_eq!(MIN_RING, 3);
    let err = unwrap_err(LocalMesh::start(2, &dir), "start(2)");
    assert_too_few_nodes(&err, 2);
    let rerr = unwrap_err(RefLocalMesh::start(2), "ref start(2)");
    assert_too_few_nodes(&rerr, 2);
}

#[test]
fn start_zero_and_one_are_too_few_nodes() {
    let (_parent, dir) = fresh_base();
    for n in [0usize, 1] {
        let err = unwrap_err(LocalMesh::start(n, &dir), "start n<3");
        assert_too_few_nodes(&err, n);
        let rerr = unwrap_err(RefLocalMesh::start(n), "ref start n<3");
        assert_too_few_nodes(&rerr, n);
    }
}

#[test]
fn start_three_slots_trusted_key_dir_and_len() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    let mut refer = RefLocalMesh::start(3).expect("ref start");
    assert_eq!(mesh.len(), 3);
    assert!(!mesh.is_empty());
    assert_eq!(refer.len(), 3);
    for i in 0..3 {
        let m = mesh.get(i).unwrap_or_else(|e| panic!("get({i}): {e}"));
        let cfg = slot_config(i);
        assert_eq!(m.dir(), dir.join(i.to_string()).as_path(), "slot {i} dir");
        assert_eq!(m.config().this_id, cfg.this_id, "slot {i} id");
        assert_eq!(m.config().this_key, cfg.this_key, "slot {i} key");
        assert!(m.config().trusted, "slot {i} trusted");
        assert_eq!(refer.get(i).expect("ref").config().this_id, cfg.this_id);
    }
}

#[test]
fn start_three_adverts_gossip_after_tick() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    let mut refer = RefLocalMesh::start(3).expect("ref start");
    for i in 0..3 {
        let caps = caps_gpu(i as u32);
        mesh.get(i)
            .expect("get")
            .advertise(caps.clone(), T0)
            .expect("advertise");
        refer
            .get(i)
            .expect("ref get")
            .advertise(caps, T0)
            .expect("ref advertise");
    }
    mesh.tick(T0).expect("tick");
    refer.tick(T0).expect("ref tick");
    for i in 0..3 {
        let ads = mesh.get(i).expect("get").adverts();
        assert_eq!(ads.len(), 3, "node {i} should see 3 adverts after tick");
        let r_ads = refer.get(i).expect("ref").adverts();
        assert_eq!(
            advert_set(ads),
            advert_set(r_ads),
            "node {i} adverts vs reference"
        );
        for j in 0..3 {
            let found = ads.iter().find(|a| a.node.0 == j.to_string());
            let a = found.unwrap_or_else(|| panic!("node {i} missing advert {j}"));
            assert_eq!(a.key.0, format!("key-{j}"));
            assert_eq!(a.caps.gpus, j as u32);
            assert_eq!(a.at, T0);
        }
    }
}

#[test]
fn advertise_on_one_node_reaches_others_only_after_tick() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start_n(3, &dir);
    let mut refer = RefLocalMesh::start(3).expect("ref");
    mesh.get(0)
        .expect("0")
        .advertise(caps_gpu(2), T0)
        .expect("adv");
    refer
        .get(0)
        .expect("r0")
        .advertise(caps_gpu(2), T0)
        .expect("ref adv");
    mesh.tick(T0).expect("tick");
    refer.tick(T0).expect("ref tick");
    for i in 1..3 {
        let ads = mesh.get(i).expect("get").adverts();
        assert!(
            ads.iter().any(|a| a.node.0 == "0" && a.caps.gpus == 2),
            "node {i} should see node 0 advert after tick"
        );
        assert_eq!(
            advert_set(ads),
            advert_set(refer.get(i).expect("ref").adverts())
        );
    }
}

#[test]
fn start_four_succeeds_and_gossips() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start_n(4, &dir);
    assert_eq!(mesh.len(), 4);
    for i in 0..4 {
        mesh.get(i)
            .expect("get")
            .advertise(caps_gpu(i as u32), T0)
            .expect("adv");
    }
    mesh.tick(T0).expect("tick");
    for i in 0..4 {
        assert_eq!(mesh.get(i).expect("get").adverts().len(), 4);
    }
}
