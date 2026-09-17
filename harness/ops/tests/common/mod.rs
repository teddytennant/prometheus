//! Shared builders and assertions for H12 `prometheus-ops` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//! - `Ops::create(dir, config)` fails if `dir` exists. Creates an H2 EventLog at
//!   `dir/log/`. No events on create. `open` replays the log; after drop+open the
//!   release state, node applied/healthy/role, and event types match.
//! - `has_quorum(release, quorum)` is true iff at least `quorum` *distinct*
//!   `HumanId`s have a signature with non-empty bytes. Duplicate humans count
//!   once. Empty signature bytes do not count. `DEFAULT_QUORUM` is 2.
//!   `OpsConfig::default().quorum == 2`.
//! - `propose` hashes `artifact` (SHA-256 lowercase hex, same as H8). If
//!   `release.digest` does not match: `Error::DigestMismatch { expected:
//!   release.digest, got: computed }` and no put, no pin, state stays Idle, no
//!   kernel change. Digest is checked before quorum.
//! - If `!has_quorum(release, config.quorum)`: `Error::NoQuorum { have, need }`,
//!   emit `ops.refuse`, state `Refused { reason }` (reason non-empty), do not
//!   put, do not pin, never `set_kernel_version`.
//! - If quorum: put artifact, pin `PIN_SHADOW` (`"kernel.shadow"`), state `Proposed`,
//!   emit `ops.propose`. Live `Cluster::kernel_version` is unchanged.
//! - `start_shadow(&mut store, now)`: `WrongState` unless `Proposed`. Pins
//!   shadow (must succeed even if `propose` already pinned `PIN_SHADOW`),
//!   state `Shadowing`, emit `ops.shadow`. Does not change live kernel_version.
//!   Membership nodes stay
//!   `NodeRole::Live` with `applied` still the live kernel. At least one
//!   `NodeStatus` has `role == Shadow`.
//! - `shadow_pass`: `WrongState` unless `Shadowing`. Emit `ops.shadow_result`.
//!   State stays `Shadowing` (no extra variant). `apply_one` is now allowed.
//! - `shadow_fail(reason, store, now)`: `WrongState` unless `Shadowing`. Unpin `PIN_SHADOW`,
//!   state `RolledBack { reason }`, emit `ops.shadow_result`. Live kernel
//!   unchanged.
//! - `apply_one`: `WrongState` until a successful `shadow_pass`. Unknown node
//!   `NotFound`. The shadow node is not a live node (`WrongState`). Rolls out
//!   one membership node: that node's `applied` becomes the proposed version,
//!   `healthy == true`, `role == Live`. Emit `ops.apply`. State `Rolling { node }`.
//!   Does not call `set_kernel_version`.
//! - `promote`: all *live* nodes must have `applied == proposed.version` and
//!   `healthy`. Else `WrongState`. Pin previous=current, current=new, unpin
//!   shadow. `Cluster::set_kernel_version` to the new version (caller passes
//!   the leader). State `Current`. Emit `ops.promote`. Shadow role is gone.
//! - `rollback(reason)`: `WrongState` from Idle/Refused/RolledBack. From
//!   `Current`, restore the previous pin and previous kernel_version. From an
//!   in-progress release (Proposed/Shadowing/Rolling), unpin shadow and leave
//!   the live kernel as it is. State `RolledBack { reason }`. Emit `ops.rollback`.
//! - `mark_unhealthy`: unknown node `NotFound`. Else emit `ops.unhealthy` and
//!   trigger `rollback` (same pin/kernel rules). `WrongState` from
//!   Idle/Refused/RolledBack.
//! - `propose` is allowed from Idle, Refused, RolledBack, Current. `WrongState`
//!   from Proposed, Shadowing, Rolling.
//! - `status.kernel_version` is `cluster.state().kernel_version`. Initial
//!   membership nodes are Live, `applied` equal to that kernel (`"0"` at
//!   bootstrap), `healthy true`.
//! - `status_json` is serde JSON of `Status` (snake_case enums). `status_text`
//!   is non-empty and contains the kernel version string.
//! - `cli(&["status"], ...)` returns `status_text`.
//!   `cli(&["--json", "status"], ...)` and `cli(&["status", "--json"], ...)`
//!   return `status_json`.
//! - Production `src/` must never import `tests/`.
//!
//! CPU-only. No `gpu` marker. Raft via `LocalGroup` (3 in-process nodes). CAS
//! via `Store::create` with two local backends.
//!
//! Production must never import this module.

#![allow(dead_code)]

use prometheus_cas::{Digest, Store, StoreConfig, EVENT_PIN, EVENT_UNPIN};
use prometheus_ops::{
    Error, HumanId, NodeRole, Ops, OpsConfig, Release, ReleaseState, Result, Signature, Status,
    DEFAULT_QUORUM,
};
use prometheus_raft::{Cluster, ClusterConfig, LocalGroup, NodeId, NowMs, MIN_VOTERS};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const TICK_BUDGET: u32 = 128;
pub const NOW0: NowMs = 1_000_000;

/// SHA-256("kernel-v1") lowercase hex.
pub const V1_BYTES: &[u8] = b"kernel-v1";
pub const V1_HEX: &str = "e535284b6f32cd691e98d2491929fa8280e183d7540f0983feacaec8ce6da61f";
/// SHA-256("kernel-v2") lowercase hex.
pub const V2_BYTES: &[u8] = b"kernel-v2";
pub const V2_HEX: &str = "3efa593dbcabdefb1b46b7e698bcdd398e0cf2a31e8c7bc0ddee10ebddc54475";
/// SHA-256("abc") lowercase hex.
pub const ABC_BYTES: &[u8] = b"abc";
pub const ABC_HEX: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

pub struct World {
    pub parent: tempfile::TempDir,
    pub ops_dir: PathBuf,
    pub store: Store,
    pub group: LocalGroup,
    pub now: NowMs,
}

pub fn short_raft_config() -> ClusterConfig {
    ClusterConfig {
        heartbeat_period_ms: 1_000,
        missed_heartbeats: 2,
    }
}

pub fn ops_cfg() -> OpsConfig {
    OpsConfig {
        quorum: DEFAULT_QUORUM,
    }
}

pub fn ops_cfg_quorum(quorum: usize) -> OpsConfig {
    OpsConfig { quorum }
}

pub fn nid(id: &str) -> NodeId {
    NodeId(id.to_string())
}

pub fn digest_hex_const(hex: &str) -> Digest {
    Digest(hex.to_string())
}

pub fn v1_digest() -> Digest {
    digest_hex_const(V1_HEX)
}

pub fn v2_digest() -> Digest {
    digest_hex_const(V2_HEX)
}

pub fn sig(human: &str, bytes: &[u8]) -> Signature {
    Signature {
        human: HumanId(human.to_string()),
        bytes: bytes.to_vec(),
    }
}

pub fn release(version: &str, digest: Digest, signatures: Vec<Signature>) -> Release {
    Release {
        version: version.to_string(),
        digest,
        commit: format!("commit-{version}"),
        signatures,
    }
}

pub fn two_sig_release(version: &str, digest: Digest) -> Release {
    release(
        version,
        digest,
        vec![sig("alice", b"sig-alice"), sig("bob", b"sig-bob")],
    )
}

pub fn one_sig_release(version: &str, digest: Digest) -> Release {
    release(version, digest, vec![sig("alice", b"sig-alice")])
}

pub fn v1_signed() -> Release {
    two_sig_release("v1", v1_digest())
}

pub fn v2_signed() -> Release {
    two_sig_release("v2", v2_digest())
}

/// Parent tempdir, CAS store, 3-node Raft group with a leader, ops dir not created.
pub fn fresh_world() -> World {
    let parent = tempfile::tempdir().expect("tempdir");
    let ops_dir = parent.path().join("ops");
    let store_dir = parent.path().join("store");
    let raft_dir = parent.path().join("raft");
    let b0 = parent.path().join("b0");
    let b1 = parent.path().join("b1");
    fs::create_dir(&b0).expect("b0");
    fs::create_dir(&b1).expect("b1");
    let store = Store::create(store_dir, vec![b0, b1], StoreConfig::default()).expect("store");
    let mut group = LocalGroup::start(MIN_VOTERS, raft_dir, short_raft_config())
        .unwrap_or_else(|e| panic!("LocalGroup::start: {e}"));
    let now = NOW0;
    tick_until_leader(&mut group, now);
    World {
        parent,
        ops_dir,
        store,
        group,
        now,
    }
}

pub fn create_ops(world: &World) -> Ops {
    Ops::create(&world.ops_dir, ops_cfg()).expect("Ops::create")
}

pub fn create_ops_cfg(world: &World, cfg: OpsConfig) -> Ops {
    Ops::create(&world.ops_dir, cfg).expect("Ops::create")
}

pub fn tick_until_leader(group: &mut LocalGroup, now: NowMs) -> usize {
    for _ in 0..TICK_BUDGET {
        group.tick(now).expect("tick");
        for i in 0..group.len() {
            if let Ok(c) = group.get(i) {
                if c.is_leader() {
                    return i;
                }
            }
        }
    }
    panic!("no leader after {TICK_BUDGET} ticks");
}

pub fn leader_idx(world: &mut World) -> usize {
    tick_until_leader(&mut world.group, world.now)
}

pub fn membership(world: &mut World) -> Vec<NodeId> {
    world
        .group
        .get(0)
        .expect("node 0")
        .state()
        .membership
        .iter()
        .map(|n| n.id.clone())
        .collect()
}

pub fn kernel_of(cluster: &Cluster) -> String {
    cluster.state().kernel_version.clone()
}

pub fn kernel_leader(world: &mut World) -> String {
    let i = leader_idx(world);
    kernel_of(world.group.get(i).expect("leader"))
}

pub fn event_types(ops: &Ops) -> Vec<String> {
    ops.log().iter().map(|e| e.event_type.clone()).collect()
}

pub fn assert_has_event(ops: &Ops, ty: &str) {
    let types = event_types(ops);
    assert!(
        types.iter().any(|t| t == ty),
        "missing event {ty}, had {types:?}"
    );
}

pub fn assert_no_event(ops: &Ops, ty: &str) {
    let types = event_types(ops);
    assert!(
        types.iter().all(|t| t != ty),
        "unexpected event {ty} in {types:?}"
    );
}

/// Replay CAS pin/unpin events into name → digest hex.
pub fn cas_pins(store: &Store) -> HashMap<String, String> {
    let mut pins = HashMap::new();
    for ev in store.log().iter() {
        match ev.event_type.as_str() {
            EVENT_PIN => {
                let name = ev
                    .payload
                    .get("name")
                    .and_then(|v| v.as_str())
                    .expect("cas.pin name")
                    .to_string();
                let digest = ev
                    .payload
                    .get("digest")
                    .and_then(|v| v.as_str())
                    .expect("cas.pin digest")
                    .to_string();
                pins.insert(name, digest);
            }
            EVENT_UNPIN => {
                if let Some(name) = ev.payload.get("name").and_then(|v| v.as_str()) {
                    pins.remove(name);
                }
            }
            _ => {}
        }
    }
    pins
}

pub fn assert_pin(store: &Store, name: &str, digest_hex: &str) {
    let pins = cas_pins(store);
    assert_eq!(
        pins.get(name).map(String::as_str),
        Some(digest_hex),
        "pin {name}, pins={pins:?}"
    );
}

pub fn assert_no_pin(store: &Store, name: &str) {
    let pins = cas_pins(store);
    assert!(
        !pins.contains_key(name),
        "pin {name} should be absent, pins={pins:?}"
    );
}

pub fn assert_wrong_state(err: &Error) {
    match err {
        Error::WrongState(_) => {}
        other => panic!("expected WrongState, got {other}"),
    }
}

pub fn assert_no_quorum(err: &Error, have: usize, need: usize) {
    match err {
        Error::NoQuorum { have: h, need: n } => {
            assert_eq!(*h, have, "have");
            assert_eq!(*n, need, "need");
        }
        other => panic!("expected NoQuorum {{ have: {have}, need: {need} }}, got {other}"),
    }
}

pub fn assert_digest_mismatch(err: &Error, expected: &str, got: &str) {
    match err {
        Error::DigestMismatch {
            expected: e,
            got: g,
        } => {
            assert_eq!(e, expected, "expected digest");
            assert_eq!(g, got, "got digest");
        }
        other => panic!("expected DigestMismatch, got {other}"),
    }
}

pub fn assert_not_found(err: &Error) {
    match err {
        Error::NotFound(_) => {}
        other => panic!("expected NotFound, got {other}"),
    }
}

pub fn err_tag(err: &Error) -> &'static str {
    match err {
        Error::NoQuorum { .. } => "no_quorum",
        Error::WrongState(_) => "wrong_state",
        Error::DigestMismatch { .. } => "digest_mismatch",
        Error::NotFound(_) => "not_found",
        Error::Refused(_) => "refused",
        Error::BadRelease(_) => "bad_release",
        Error::Cas(_) => "cas",
        Error::Raft(_) => "raft",
        Error::Log(_) => "log",
        Error::Other(_) => "other",
    }
}

pub fn assert_result_tag(got: &Result<()>, exp: &Result<()>, what: &str) {
    match (got, exp) {
        (Ok(()), Ok(())) => {}
        (Err(a), Err(b)) => {
            assert_eq!(err_tag(a), err_tag(b), "{what}: {a} vs {b}");
            if let (
                Error::NoQuorum { have: ha, need: na },
                Error::NoQuorum { have: hb, need: nb },
            ) = (a, b)
            {
                assert_eq!(ha, hb, "{what} have");
                assert_eq!(na, nb, "{what} need");
            }
            if let (
                Error::DigestMismatch {
                    expected: ea,
                    got: ga,
                },
                Error::DigestMismatch {
                    expected: eb,
                    got: gb,
                },
            ) = (a, b)
            {
                assert_eq!(ea, eb, "{what} expected");
                assert_eq!(ga, gb, "{what} got");
            }
        }
        (Ok(()), Err(b)) => panic!("{what}: ops Ok, reference Err({b})"),
        (Err(a), Ok(())) => panic!("{what}: ops Err({a}), reference Ok"),
    }
}

pub fn live_nodes(status: &Status) -> Vec<&prometheus_ops::NodeStatus> {
    status
        .nodes
        .iter()
        .filter(|n| n.role == NodeRole::Live)
        .collect()
}

pub fn has_shadow_role(status: &Status) -> bool {
    status.nodes.iter().any(|n| n.role == NodeRole::Shadow)
}

pub fn assert_live_applied(status: &Status, version: &str) {
    let live = live_nodes(status);
    assert!(!live.is_empty(), "no live nodes in {status:?}");
    for n in live {
        assert_eq!(n.applied, version, "live node {} applied", n.id.0);
        assert!(n.healthy, "live node {} healthy", n.id.0);
        assert_eq!(n.role, NodeRole::Live);
    }
}

pub fn assert_membership_live(status: &Status, members: &[NodeId]) {
    let mut got: Vec<String> = live_nodes(status)
        .into_iter()
        .map(|n| n.id.0.clone())
        .collect();
    let mut exp: Vec<String> = members.iter().map(|n| n.0.clone()).collect();
    got.sort();
    exp.sort();
    assert_eq!(got, exp, "live membership");
}

pub fn assert_state_refused(status: &Status) {
    match &status.state {
        ReleaseState::Refused { reason } => {
            assert!(!reason.is_empty(), "Refused reason must be non-empty");
        }
        other => panic!("expected Refused, got {other:?}"),
    }
}

pub fn assert_state_rolled_back(status: &Status, reason: &str) {
    match &status.state {
        ReleaseState::RolledBack { reason: r } => {
            assert_eq!(r, reason, "rollback reason");
        }
        other => panic!("expected RolledBack {{ reason: {reason:?} }}, got {other:?}"),
    }
}

pub fn status_of(ops: &Ops, world: &mut World) -> Status {
    let i = leader_idx(world);
    ops.status(world.group.get(i).expect("leader"))
}

pub fn arg(s: &str) -> String {
    s.to_string()
}

pub fn log_dir(ops_dir: &Path) -> PathBuf {
    ops_dir.join("log")
}

/// Walk `harness/ops/src` and fail if production imports the tests tree.
pub fn assert_src_does_not_import_tests() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for ent in fs::read_dir(dir).expect("read src") {
            let ent = ent.expect("entry");
            let p = ent.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    walk(&src, &mut files);
    for f in files {
        let text = fs::read_to_string(&f).unwrap_or_else(|e| panic!("read {f:?}: {e}"));
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with("//") {
                continue;
            }
            assert!(
                !t.contains("include!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/tests")
                    && !t.contains("path = \"../tests")
                    && !t.contains("prometheus_ops::tests")
                    && !t.contains("tests/reference")
                    && !t.contains("mod tests"),
                "{} must not import tests/: {line}",
                f.display()
            );
        }
    }
}
