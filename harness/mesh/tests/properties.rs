//! Group: property checks of production vs the in-memory reference.

mod common;
mod reference;

use common::{
    advert_set, caps_cpu, caps_gpu, fresh_base, fresh_node_dir, item, open_queue, slot_config,
    start_n, watch_pairs, T0,
};
use prometheus_mesh::{Capabilities, LocalMesh, MIN_WATCHERS};
use reference::{caps_satisfy, expected_watches, first_fitting, RefLocalMesh};
use serde_json::json;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
}

#[test]
fn watches_len_is_n_times_min_watchers_for_n_in_3_to_6() {
    for n in 3usize..=6 {
        let (_parent, dir) = fresh_base();
        let mut mesh = start_n(n, &dir);
        let mut refer = RefLocalMesh::start(n).expect("ref");
        let expect = expected_watches(n);
        assert_eq!(expect.len(), n * MIN_WATCHERS);
        for i in 0..n {
            let got = mesh.get(i).expect("get").watches();
            assert_eq!(got.len(), n * MIN_WATCHERS, "n={n} node {i}");
            assert_eq!(watch_pairs(got), watch_pairs(&expect));
            assert_eq!(
                watch_pairs(got),
                watch_pairs(refer.get(i).expect("ref").watches())
            );
        }
    }
}

#[test]
fn each_node_watches_exactly_two_and_is_watched_exactly_twice() {
    for n in 3usize..=5 {
        let (_parent, dir) = fresh_base();
        let mut mesh = LocalMesh::start(n, &dir).expect("start");
        let got = mesh.get(0).expect("0").watches();
        for i in 0..n {
            let id = i.to_string();
            let as_watcher = got.iter().filter(|w| w.watcher.0 == id).count();
            let as_watchee = got.iter().filter(|w| w.watchee.0 == id).count();
            assert_eq!(as_watcher, MIN_WATCHERS, "n={n} watcher {i}");
            assert_eq!(as_watchee, MIN_WATCHERS, "n={n} watchee {i}");
        }
    }
}

#[test]
fn random_advert_gossip_matches_reference() {
    let (_parent, dir) = fresh_base();
    let n = 4usize;
    let mut mesh = start_n(n, &dir);
    let mut refer = RefLocalMesh::start(n).expect("ref");
    let mut rng = Lcg(0xC0FFEE);
    let mut now = T0;
    for _ in 0..24 {
        let i = rng.pick(n);
        let caps = Capabilities {
            gpus: rng.pick(4) as u32,
            cpus: (rng.pick(8) as u32) + 1,
            kvm: rng.pick(2) == 1,
            providers: if rng.pick(2) == 1 {
                vec!["cuda".into()]
            } else {
                vec![]
            },
        };
        mesh.get(i)
            .expect("get")
            .advertise(caps.clone(), now)
            .expect("adv");
        refer
            .get(i)
            .expect("ref")
            .advertise(caps, now)
            .expect("ref adv");
        now += 1;
        mesh.tick(now).expect("tick");
        refer.tick(now).expect("ref tick");
        for j in 0..n {
            assert_eq!(
                advert_set(mesh.get(j).expect("j").adverts()),
                advert_set(refer.get(j).expect("rj").adverts()),
                "node {j} after gossip"
            );
        }
    }
}

#[test]
fn random_requires_vs_caps_satisfy_matches_pull() {
    let mut rng = Lcg(0xBEEF);
    for trial in 0..32 {
        let caps = Capabilities {
            gpus: rng.pick(3) as u32,
            cpus: (rng.pick(4) as u32) + 1,
            kvm: rng.pick(2) == 1,
            providers: match rng.pick(3) {
                0 => vec![],
                1 => vec!["cuda".into()],
                _ => vec!["cuda".into(), "rocm".into()],
            },
        };
        let req_gpus = rng.pick(4) as u32;
        let payload = json!({"requires": {"gpus": req_gpus}});
        let fits = caps_satisfy(&caps, &payload);

        let (_parent, dir) = fresh_node_dir();
        let mut mesh = prometheus_mesh::Mesh::create(&dir, slot_config(0)).expect("create");
        mesh.advertise(caps.clone(), T0).expect("adv");
        let (_qparent, mut queue) = open_queue();
        queue.enqueue(item("t", payload), T0).expect("enq");
        let oracle = first_fitting(&queue, &caps);
        let got = mesh.pull(&mut queue, T0).expect("pull");
        if fits {
            assert!(oracle.is_some(), "trial {trial} oracle");
            let lease = got.unwrap_or_else(|| panic!("trial {trial}: expected claim"));
            assert_eq!(lease.task_id.0, "t");
            assert_eq!(lease.attempt, 1);
        } else {
            assert!(oracle.is_none(), "trial {trial} oracle none");
            assert!(
                got.is_none(),
                "trial {trial}: expected Ok(None), got {got:?}"
            );
        }
    }
}

#[test]
fn second_advertise_replaces_caps_then_gossips() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start_n(3, &dir);
    mesh.get(0)
        .expect("0")
        .advertise(caps_cpu(), T0)
        .expect("first");
    mesh.get(0)
        .expect("0")
        .advertise(caps_gpu(7), T0 + 5)
        .expect("second");
    mesh.tick(T0 + 5).expect("tick");
    for i in 0..3 {
        let ads = mesh.get(i).expect("get").adverts();
        let a = ads.iter().find(|a| a.node.0 == "0").expect("have node 0");
        assert_eq!(a.caps.gpus, 7);
        assert_eq!(a.at, T0 + 5);
        assert_eq!(
            ads.iter().filter(|a| a.node.0 == "0").count(),
            1,
            "one advert per node"
        );
    }
}
