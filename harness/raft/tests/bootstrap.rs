//! Group: single-node Cluster::bootstrap / open smoke, grow via add_voter,
//! refuse to empty the cluster.

mod common;
mod reference;

use common::{
    assert_ephemeral_voter, assert_err, assert_initial_control_state, assert_membership_eq,
    assert_state_eq, assert_too_few_voters, assert_untrusted, bootstrap_solo, default_config,
    ephemeral, fresh_base, node_id, short_config, untrusted_voter, voter, FAKE_TOKEN,
};
use prometheus_raft::{Cluster, NodeId};
use reference::RefCluster;
use std::fs;

#[test]
fn bootstrap_trusted_always_on_is_leader_with_initial_state() {
    let (_parent, dir) = fresh_base();
    let cfg = short_config();
    let this = voter("solo", "local:solo");
    let refer = RefCluster::bootstrap(this.clone(), cfg.clone()).expect("ref bootstrap");
    let c = Cluster::bootstrap(&dir, this.clone(), cfg).expect("bootstrap");

    assert_eq!(c.this_id(), &NodeId("solo".into()));
    assert_eq!(c.dir(), dir.as_path());
    assert_eq!(c.config().heartbeat_period_ms, 1_000);
    assert_eq!(c.config().missed_heartbeats, 2);
    assert_eq!(c.config().lease_ttl_ms(), 2_000);
    assert!(c.is_leader(), "single-node bootstrap is leader");
    assert_initial_control_state(c.state(), "bootstrap");
    assert_eq!(c.state().membership.len(), 1);
    assert_eq!(c.state().membership[0].id, this.id);
    assert_eq!(c.state().membership[0].kind, this.kind);
    assert_eq!(c.state().membership[0].trusted, this.trusted);
    assert!(dir.is_dir(), "bootstrap creates dir");
    assert_state_eq(c.state(), &refer.state(), "bootstrap vs ref");
    let _ = refer;
}

#[test]
fn bootstrap_default_config_ttl_is_30000_and_kernel_is_zero() {
    let (_parent, dir) = fresh_base();
    let c = bootstrap_solo(&dir, default_config());
    assert_eq!(c.config().lease_ttl_ms(), 15_000 * 2);
    assert_eq!(c.state().kernel_version, "0");
    assert!(c.is_leader());
}

#[test]
fn bootstrap_fails_if_dir_exists() {
    let (_parent, dir) = fresh_base();
    fs::create_dir_all(&dir).expect("mkdir");
    let this = voter("solo", "local:solo");
    assert_err(
        Cluster::bootstrap(&dir, this, short_config()),
        "bootstrap over existing dir",
    );
}

#[test]
fn bootstrap_ephemeral_is_ephemeral_voter() {
    let (_parent, dir) = fresh_base();
    let this = ephemeral("e", "local:e", true);
    let err = common::unwrap_err(Cluster::bootstrap(&dir, this, short_config()), "ephemeral");
    assert_ephemeral_voter(&err, "bootstrap ephemeral");
}

#[test]
fn bootstrap_untrusted_is_untrusted() {
    let (_parent, dir) = fresh_base();
    let this = untrusted_voter("u", "local:u");
    let err = common::unwrap_err(Cluster::bootstrap(&dir, this, short_config()), "untrusted");
    assert_untrusted(&err, "bootstrap untrusted");
}

#[test]
fn bootstrap_add_voter_grows_membership() {
    let (_parent, dir) = fresh_base();
    let cfg = short_config();
    let this = voter("solo", "local:solo");
    let mut refer = RefCluster::bootstrap(this.clone(), cfg.clone()).expect("ref");
    let mut c = Cluster::bootstrap(&dir, this, cfg).expect("bootstrap");
    assert!(c.is_leader());

    let peer = voter("peer", "local:peer");
    c.add_voter(peer.clone()).expect("add_voter");
    refer.add_voter(peer.clone()).expect("ref add_voter");
    assert_eq!(c.state().membership.len(), 2);
    let got = c
        .state()
        .membership
        .iter()
        .find(|n| n.id == peer.id)
        .expect("peer in membership");
    assert_eq!(got, &peer);
    assert_membership_eq(
        &c.state().membership,
        &refer.state().membership,
        "grown membership",
    );
}

#[test]
fn remove_last_voter_on_one_node_group_is_too_few_voters_zero() {
    let (_parent, dir) = fresh_base();
    let mut c = bootstrap_solo(&dir, short_config());
    let mut refer =
        RefCluster::bootstrap(voter("solo", "local:solo"), short_config()).expect("ref");
    let id = c.this_id().clone();
    let err = common::unwrap_err(c.remove_node(&id), "last voter");
    assert_too_few_voters(&err, 0, "remove last voter");
    assert_eq!(c.state().membership.len(), 1, "cluster must not go empty");
    let ref_err = refer.remove_node(&id).expect_err("ref last voter");
    assert_eq!(common::err_kind(&err), common::err_kind(&ref_err));
}

#[test]
fn open_after_bootstrap_restores_kernel_and_token() {
    let (_parent, dir) = fresh_base();
    let cfg = short_config();
    let this = voter("solo", "local:solo");
    {
        let mut c = Cluster::bootstrap(&dir, this.clone(), cfg.clone()).expect("bootstrap");
        assert!(c.is_leader());
        c.set_kernel_version("k1".into()).expect("kernel");
        let rec = c.commit_token(FAKE_TOKEN.to_vec()).expect("token");
        assert_eq!(rec.generation, 1);
        assert_eq!(rec.blob, FAKE_TOKEN);
        assert_eq!(c.state().kernel_version, "k1");
    }
    let c = Cluster::open(&dir, node_id("solo"), cfg).expect("open");
    assert_eq!(c.this_id(), &NodeId("solo".into()));
    assert_eq!(c.state().kernel_version, "k1");
    let tok = c.state().token.as_ref().expect("token restored");
    assert_eq!(tok.generation, 1);
    assert_eq!(tok.blob, FAKE_TOKEN);
}

#[test]
fn open_missing_dir_fails() {
    let (_parent, dir) = fresh_base();
    assert_err(
        Cluster::open(&dir, node_id("solo"), short_config()),
        "open missing",
    );
}
