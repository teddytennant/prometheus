//! Group: watcher ring for 3 nodes, 6 Watch entries, each watchee twice;
//! deterministic (i+1)%n and (i+2)%n order.

mod common;
mod reference;

use common::{fresh_base, start3, start_n, watch_pairs};
use prometheus_mesh::{Watch, MIN_RING, MIN_WATCHERS};
use reference::{expected_watches, RefLocalMesh};

fn count_watchee(watches: &[Watch], id: &str) -> usize {
    watches.iter().filter(|w| w.watchee.0 == id).count()
}

#[test]
fn watches_ring_three_nodes_six_entries_each_watchee_twice() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start3(&dir);
    let mut refer = RefLocalMesh::start(3).expect("ref");
    assert_eq!(MIN_RING, 3);
    assert_eq!(MIN_WATCHERS, 2);

    let expect = expected_watches(3);
    assert_eq!(expect.len(), 6);
    let pairs = watch_pairs(&expect);
    assert_eq!(
        pairs,
        vec![
            ("1".into(), "0".into()),
            ("2".into(), "0".into()),
            ("2".into(), "1".into()),
            ("0".into(), "1".into()),
            ("0".into(), "2".into()),
            ("1".into(), "2".into()),
        ]
    );

    for i in 0..3 {
        let got = mesh.get(i).expect("get").watches();
        assert_eq!(got.len(), 6, "node {i} watches len");
        assert_eq!(watch_pairs(got), pairs, "node {i} ring order");
        for w in 0..3 {
            assert_eq!(
                count_watchee(got, &w.to_string()),
                MIN_WATCHERS,
                "watchee {w} count on node {i}"
            );
        }
        let rgot = refer.get(i).expect("ref").watches();
        assert_eq!(watch_pairs(got), watch_pairs(rgot));
    }
}

#[test]
fn watches_ring_four_nodes() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start_n(4, &dir);
    let expect = expected_watches(4);
    assert_eq!(expect.len(), 8);
    let got = mesh.get(0).expect("0").watches();
    assert_eq!(watch_pairs(got), watch_pairs(&expect));
    for w in 0..4 {
        assert_eq!(count_watchee(got, &w.to_string()), MIN_WATCHERS);
    }
}

#[test]
fn watcher_is_never_self() {
    let (_parent, dir) = fresh_base();
    let mut mesh = start_n(5, &dir);
    let got = mesh.get(0).expect("0").watches();
    assert_eq!(got.len(), 10);
    for w in got {
        assert_ne!(w.watcher, w.watchee, "self-watch");
    }
}
