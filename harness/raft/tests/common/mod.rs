//! Shared builders and assertions for H4 `prometheus-raft` oracle tests.
//!
//! Documented rules the implementer must match:
//! - `LocalGroup::start(n)` errors if `n < MIN_VOTERS` (3). Node `i` has id `"{i}"`
//!   (decimal, no leading zeros), dir `base_dir/<i>/`, kind AlwaysOn, trusted.
//!   Addr is implementation-defined and unique per node.
//! - Kernel version starts as `"0"`. Token is none at bootstrap. First
//!   `commit_token` is generation 1; each later commit is current+1. Tests use
//!   fake `b"fake-token"` blobs, never real secrets.
//! - First role claim is attempt 1. `expire_roles` does not bump; the next
//!   `claim_role` does. Due means `expires_at <= now`. TTL =
//!   `heartbeat_period_ms * missed_heartbeats.max(1)`.
//! - Mutations that change replicated state are leader only (`NotLeader` on
//!   followers). Tests `tick` until some node is leader.
//! - AlwaysOn + trusted = voter (`add_voter`). Ephemeral never a voter
//!   (`EphemeralVoter`). Untrusted cannot claim roles (`Untrusted`).
//! - `remove_node` refuses to drop below 3 voters once the group has reached 3.
//!   Removing the last voter of a 1-node bootstrap group is `TooFewVoters(0)`.
//! - `NowMs` is injected. No wall clock. Drive time with `tick(now)` and the
//!   `now` arguments on role APIs; do not sleep.
//! - `crash(i)` drops the in-memory node (`get(i)` is `NotFound`); disk stays.
//!   Indices stay stable. `len()` remains the started `n`. `restart(i)` reopens
//!   from disk, not leader until `tick`.
//! - `partition` replaces the isolated set. Isolated nodes get no Raft traffic.
//!   `heal` restores traffic. After `tick`, a majority of 2 (of 3) still commits.
//!
//! No GPU coverage in H4; these tests are CPU-only (no `gpu` marker).

#![allow(dead_code)]

use prometheus_raft::{
    Addr, Cluster, ClusterConfig, ControlState, Error, LocalGroup, NodeId, NodeInfo, NodeKind,
    NowMs, Result, Role, RoleLease, MIN_VOTERS,
};
use std::path::{Path, PathBuf};

pub const FAKE_TOKEN: &[u8] = b"fake-token";
pub const FAKE_TOKEN_2: &[u8] = b"fake-token-2";
pub const FAKE_TOKEN_3: &[u8] = b"fake-token-3";
pub const TICK_BUDGET: u32 = 128;
pub const SHORT_PERIOD_MS: u64 = 1_000;
pub const SHORT_MISSED: u32 = 2;

/// Parent temp dir plus a not-yet-created `raft/` child for bootstrap / start.
pub fn fresh_base() -> (tempfile::TempDir, PathBuf) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("raft");
    (parent, dir)
}

pub fn short_config() -> ClusterConfig {
    ClusterConfig {
        heartbeat_period_ms: SHORT_PERIOD_MS,
        missed_heartbeats: SHORT_MISSED,
    }
}

pub fn short_ttl() -> u64 {
    short_config().lease_ttl_ms()
}

pub fn default_config() -> ClusterConfig {
    ClusterConfig::default()
}

pub fn node_id(id: &str) -> NodeId {
    NodeId(id.to_string())
}

pub fn slot_id(i: usize) -> NodeId {
    NodeId(i.to_string())
}

pub fn voter(id: &str, addr: &str) -> NodeInfo {
    NodeInfo {
        id: node_id(id),
        addr: Addr(addr.to_string()),
        kind: NodeKind::AlwaysOn,
        trusted: true,
    }
}

pub fn untrusted_voter(id: &str, addr: &str) -> NodeInfo {
    NodeInfo {
        id: node_id(id),
        addr: Addr(addr.to_string()),
        kind: NodeKind::AlwaysOn,
        trusted: false,
    }
}

pub fn ephemeral(id: &str, addr: &str, trusted: bool) -> NodeInfo {
    NodeInfo {
        id: node_id(id),
        addr: Addr(addr.to_string()),
        kind: NodeKind::Ephemeral,
        trusted,
    }
}

pub fn start_n(n: usize, base: &Path, cfg: ClusterConfig) -> LocalGroup {
    LocalGroup::start(n, base, cfg).unwrap_or_else(|e| panic!("LocalGroup::start({n}): {e}"))
}

pub fn start3(base: &Path) -> LocalGroup {
    start_n(3, base, short_config())
}

/// Indices of nodes that currently report `is_leader`. Raft may briefly show
/// more than one across a partition; callers that need a unique leader after a
/// stable election should use [`tick_until_leader`].
pub fn leader_indices(group: &mut LocalGroup) -> Vec<usize> {
    let n = group.len();
    let mut out = Vec::new();
    for i in 0..n {
        if let Ok(c) = group.get(i) {
            if c.is_leader() {
                out.push(i);
            }
        }
    }
    out
}

pub fn leader_index(group: &mut LocalGroup) -> Option<usize> {
    leader_indices(group).into_iter().next()
}

pub fn tick_until_leader(group: &mut LocalGroup, now: NowMs) -> usize {
    for _ in 0..TICK_BUDGET {
        group.tick(now).expect("tick");
        if let Some(i) = leader_index(group) {
            return i;
        }
    }
    panic!("no leader after {TICK_BUDGET} ticks at now={now}");
}

pub fn tick_until_leader_among(group: &mut LocalGroup, now: NowMs, allowed: &[usize]) -> usize {
    for _ in 0..TICK_BUDGET {
        group.tick(now).expect("tick");
        for &i in allowed {
            if let Ok(c) = group.get(i) {
                if c.is_leader() {
                    return i;
                }
            }
        }
    }
    panic!("no leader among {allowed:?} after {TICK_BUDGET} ticks");
}

pub fn follower_index(group: &mut LocalGroup) -> usize {
    let n = group.len();
    for i in 0..n {
        match group.get(i) {
            Ok(c) if !c.is_leader() => return i,
            Ok(_) => {}
            Err(_) => {}
        }
    }
    panic!("no follower (every live node reports is_leader)");
}

pub fn this_id_of(group: &mut LocalGroup, i: usize) -> NodeId {
    group
        .get(i)
        .unwrap_or_else(|e| panic!("get({i}): {e}"))
        .this_id()
        .clone()
}

pub fn state_of(group: &mut LocalGroup, i: usize) -> ControlState {
    group
        .get(i)
        .unwrap_or_else(|e| panic!("get({i}): {e}"))
        .state()
        .clone()
}

pub fn live_indices(group: &mut LocalGroup) -> Vec<usize> {
    let n = group.len();
    let mut out = Vec::new();
    for i in 0..n {
        if group.get(i).is_ok() {
            out.push(i);
        }
    }
    out
}

pub fn unwrap_err<T>(r: Result<T>, what: &str) -> Error {
    match r {
        Ok(_) => panic!("expected error ({what})"),
        Err(e) => e,
    }
}

pub fn assert_err<T>(r: Result<T>, what: &str) {
    let _ = unwrap_err(r, what);
}

pub fn assert_not_leader<T>(r: Result<T>, what: &str) {
    match r {
        Err(Error::NotLeader) => {}
        Ok(_) => panic!("{what}: expected NotLeader, got Ok"),
        Err(other) => panic!("{what}: expected NotLeader, got {other:?}"),
    }
}

pub fn assert_not_found<T>(r: Result<T>, what: &str) {
    match r {
        Err(Error::NotFound(_)) => {}
        Ok(_) => panic!("{what}: expected NotFound, got Ok"),
        Err(other) => panic!("{what}: expected NotFound, got {other:?}"),
    }
}

pub fn assert_not_holder(err: &Error, role: Role, what: &str) {
    match err {
        Error::NotHolder(r, _) if *r == role => {}
        other => panic!("{what}: expected NotHolder({role:?}, _), got {other:?}"),
    }
}

pub fn assert_not_claimable(err: &Error, role: Role, what: &str) {
    match err {
        Error::NotClaimable(r) if *r == role => {}
        other => panic!("{what}: expected NotClaimable({role:?}), got {other:?}"),
    }
}

pub fn assert_too_few_voters(err: &Error, have: usize, what: &str) {
    match err {
        Error::TooFewVoters(n) if *n == have => {}
        other => panic!("{what}: expected TooFewVoters({have}), got {other:?}"),
    }
}

pub fn assert_duplicate(err: &Error, what: &str) {
    match err {
        Error::Duplicate(_) => {}
        other => panic!("{what}: expected Duplicate, got {other:?}"),
    }
}

pub fn assert_ephemeral_voter(err: &Error, what: &str) {
    match err {
        Error::EphemeralVoter(_) => {}
        other => panic!("{what}: expected EphemeralVoter, got {other:?}"),
    }
}

pub fn assert_untrusted(err: &Error, what: &str) {
    match err {
        Error::Untrusted(_) => {}
        other => panic!("{what}: expected Untrusted, got {other:?}"),
    }
}

pub fn err_kind(err: &Error) -> &'static str {
    match err {
        Error::NotLeader => "NotLeader",
        Error::NotFound(_) => "NotFound",
        Error::NotHolder(_, _) => "NotHolder",
        Error::NotClaimable(_) => "NotClaimable",
        Error::Untrusted(_) => "Untrusted",
        Error::EphemeralVoter(_) => "EphemeralVoter",
        Error::TooFewVoters(_) => "TooFewVoters",
        Error::Duplicate(_) => "Duplicate",
        Error::TokenGeneration { .. } => "TokenGeneration",
        Error::Other(_) => "Other",
    }
}

/// Membership compared as a set of (id, kind, trusted). Addr is not part of
/// start()-node identity; tests that pass a `NodeInfo` into `add_voter` /
/// `add_worker` also check addr on that record.
pub fn assert_membership_eq(got: &[NodeInfo], exp: &[NodeInfo], what: &str) {
    assert_eq!(got.len(), exp.len(), "{what}: membership len");
    for e in exp {
        let g = got
            .iter()
            .find(|n| n.id == e.id)
            .unwrap_or_else(|| panic!("{what}: missing member {}", e.id.0));
        assert_eq!(g.kind, e.kind, "{what}: kind {}", e.id.0);
        assert_eq!(g.trusted, e.trusted, "{what}: trusted {}", e.id.0);
    }
    for g in got {
        assert!(
            exp.iter().any(|e| e.id == g.id),
            "{what}: unexpected member {}",
            g.id.0
        );
    }
}

pub fn assert_state_eq(got: &ControlState, exp: &ControlState, what: &str) {
    assert_eq!(got.kernel_version, exp.kernel_version, "{what}: kernel");
    assert_eq!(got.token, exp.token, "{what}: token");
    assert_eq!(got.coordinator, exp.coordinator, "{what}: coordinator");
    assert_eq!(got.token_broker, exp.token_broker, "{what}: token_broker");
    assert_membership_eq(&got.membership, &exp.membership, what);
}

pub fn assert_initial_control_state(state: &ControlState, what: &str) {
    assert_eq!(state.kernel_version, "0", "{what}: kernel starts as \"0\"");
    assert!(state.token.is_none(), "{what}: no token at bootstrap");
    assert!(state.coordinator.is_none(), "{what}: no coordinator");
    assert!(state.token_broker.is_none(), "{what}: no token_broker");
}

pub fn assert_lease(
    lease: &RoleLease,
    role: Role,
    holder: &NodeId,
    attempt: u64,
    expires_at: NowMs,
) {
    assert_eq!(lease.role, role);
    assert_eq!(&lease.holder, holder);
    assert_eq!(lease.attempt, attempt);
    assert_eq!(lease.expires_at, expires_at);
}

pub fn member<'a>(state: &'a ControlState, id: &NodeId) -> &'a NodeInfo {
    state
        .membership
        .iter()
        .find(|n| n.id == *id)
        .unwrap_or_else(|| panic!("missing member {}", id.0))
}

pub fn voter_count(state: &ControlState) -> usize {
    state
        .membership
        .iter()
        .filter(|n| n.kind == NodeKind::AlwaysOn)
        .count()
}

pub fn assert_min_voters_lock() {
    assert_eq!(MIN_VOTERS, 3, "locked MIN_VOTERS");
}

/// After a leader mutation, drive apply so connected followers see committed state.
pub fn replicate(group: &mut LocalGroup, now: NowMs) {
    group.tick(now).expect("replicate tick");
}

pub fn bootstrap_solo(dir: &Path, cfg: ClusterConfig) -> Cluster {
    let this = voter("solo", "local:solo");
    Cluster::bootstrap(dir, this, cfg).expect("bootstrap")
}
