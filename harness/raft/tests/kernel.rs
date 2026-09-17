//! Group: set_kernel_version replicates; restart sees it.

mod common;
mod reference;

use common::{
    assert_state_eq, follower_index, fresh_base, replicate, short_config, start3, tick_until_leader,
};
use reference::RefGroup;

#[test]
fn set_kernel_version_replicates_and_restart_sees_it() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let mut refer = RefGroup::start(3, short_config()).expect("ref");
    let now = 3_000;
    let leader = tick_until_leader(&mut group, now);
    let rleader = refer.tick_until_leader(now);

    assert_eq!(group.get(leader).expect("l").state().kernel_version, "0");
    group
        .get(leader)
        .expect("l")
        .set_kernel_version("k-7".into())
        .expect("set");
    refer
        .set_kernel_version(rleader, "k-7".into())
        .expect("ref set");
    assert_eq!(group.get(leader).expect("l").state().kernel_version, "k-7");
    replicate(&mut group, now);
    refer.tick(now).expect("ref tick");

    for i in 0..3 {
        assert_eq!(
            group.get(i).expect("n").state().kernel_version,
            "k-7",
            "node {i}"
        );
    }
    assert_state_eq(
        group.get(leader).expect("l").state(),
        &refer.committed_state(),
        "kernel vs ref",
    );

    let f = follower_index(&mut group);
    group.crash(f).expect("crash");
    refer.crash(f).expect("ref crash");
    group.restart(f).expect("restart");
    refer.restart(f).expect("ref restart");
    assert_eq!(
        group.get(f).expect("restarted").state().kernel_version,
        "k-7"
    );
    assert_eq!(refer.get(f).expect("ref f").state.kernel_version, "k-7");
}

#[test]
fn set_kernel_version_can_overwrite() {
    let (_parent, dir) = fresh_base();
    let mut group = start3(&dir);
    let now = 4_000;
    let leader = tick_until_leader(&mut group, now);
    group
        .get(leader)
        .expect("l")
        .set_kernel_version("a".into())
        .expect("a");
    group
        .get(leader)
        .expect("l")
        .set_kernel_version("b".into())
        .expect("b");
    assert_eq!(group.get(leader).expect("l").state().kernel_version, "b");
}
