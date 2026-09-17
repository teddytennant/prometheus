//! Shared builders and assertions for H6 `prometheus-providers` oracle tests.
//!
//! Documented rules the implementer must match (also see `tests/reference`):
//!
//! **`validate_response`** (check order, first match wins):
//! 1. Body is the exact ASCII bytes `rate_limit` in any case → `RateLimitText`.
//! 2. Body is not UTF-8 JSON, or the JSON value is not an object → `MalformedJson`.
//! 3. Object has string field `error` equal to `rate_limit` (ASCII case-insensitive)
//!    → `RateLimitText`. Extra fields do not matter.
//! 4. `choices` missing, null, non-array, or empty array → `EmptyCompletion`.
//! 5. Any choice with `finish_reason == "length"` and `tool_calls` present
//!    (on the choice or under `message`) that is truncated → `TruncatedToolCall`.
//!    Truncated means: `tool_calls` is a non-array, or a non-empty array whose
//!    item is missing `function.arguments` (no `function` object, no `arguments`
//!    key, or `arguments: null`), or `arguments` is a string that is not valid
//!    JSON (unclosed). Parsed object/array `arguments` are not truncated.
//!    `finish_reason` other than `"length"` is never `TruncatedToolCall`.
//! 6. Else return the parsed `serde_json::Value` unchanged.
//!
//! **AIMD** (integer window):
//! - `on_success`: if `latency_ms > latency_limit_ms` treat as congestion (same
//!   shrink as `on_429`); else `window = min(max_window, saturating_add(increase))`.
//!   Latency equal to the limit is additive success, not congestion.
//! - `on_429` / `on_outage` / congestion: `floor(window as f64 * decrease)` as
//!   `u32`. If that value is `>= window` (including `decrease >= 1`), subtract
//!   1 before clamp so a 429 always shrinks. Then `window = max(min_window, …)`.
//!
//! **Failover**:
//! - `current` is the first `providers` entry whose id is not in `down`, else
//!   `NoProvider`. Original list order.
//! - `mark_down` / `mark_up` of an id not in `providers` is `NotFound`.
//! - `mark_down` of already-down and `mark_up` of already-up are `Ok` (idempotent).
//!   `down` stores unique ids so `wait_backoff` uses the unique count.
//! - `wait_backoff` is `Duration::from_millis(100 * 2^min(down.len(), 8))`.
//!   Zero down is 100 ms. It returns `Duration`, never an error, including when
//!   every provider is down (`current` is `NoProvider` — wait, do not exit).
//!
//! **Broker::refresh**:
//! - If `!cluster.is_leader()` return `NotBroker`. CPU tests do not require a
//!   TokenBroker role claim; single-node `Cluster::bootstrap` is leader.
//! - Else `commit_token(refresh_blob)` first. Access token is lowercase hex
//!   SHA-256 of `refresh_blob` (64 chars `0-9a-f`). `expires_at = now.saturating_add(access_ttl_ms)`.
//! - Agents must not be able to read the refresh blob from `AccessToken`.
//! - Tests use fake blobs `b"fake-token"`. Cluster dirs come from tempfile.
//! - `NowMs` is injected; no wall clock.
//!
//! CPU only. No `gpu` marker. Production must never import this module.
//!
//! Do not write tests whose *only* assertions are the already-implemented
//! `Aimd::{new,config,window}`, `Failover::{new,providers,down}`,
//! `Broker::{new,provider}`, or `From<prometheus_raft::Error>`.

#![allow(dead_code)]

use prometheus_providers::{
    AccessToken, AimdConfig, Broker, Error, Failover, ProviderConfig, ProviderId, TransportFault,
};
use prometheus_raft::{
    Addr, Cluster, ClusterConfig, LocalGroup, NodeId, NodeInfo, NodeKind, NowMs, TokenRecord,
};
use std::path::PathBuf;
use std::time::Duration;

pub const FAKE_BLOB: &[u8] = b"fake-token";

/// SHA-256("fake-token") lowercase hex, independently computed.
pub const FAKE_ACCESS_HEX: &str =
    "e1466187c844c921b622aff2197444cfdc2c87489f7a6e71cef47b31a1602ced";

pub fn pid(s: &str) -> ProviderId {
    ProviderId(s.to_string())
}

pub fn provider(id: &str, access_ttl_ms: u64) -> ProviderConfig {
    ProviderConfig {
        id: pid(id),
        access_ttl_ms,
    }
}

pub fn aimd_cfg(
    min_window: u32,
    max_window: u32,
    increase: u32,
    decrease: f64,
    latency_limit_ms: u64,
) -> AimdConfig {
    AimdConfig {
        min_window,
        max_window,
        increase,
        decrease,
        latency_limit_ms,
    }
}

pub fn ids(names: &[&str]) -> Vec<ProviderId> {
    names.iter().map(|s| pid(s)).collect()
}

pub fn failover(names: &[&str]) -> Failover {
    Failover::new(ids(names))
}

pub fn broker(id: &str, ttl: u64) -> Broker {
    Broker::new(provider(id, ttl))
}

pub fn assert_invalid(err: &Error, kind: TransportFault) {
    match err {
        Error::InvalidResponse(got) => {
            assert_eq!(*got, kind, "InvalidResponse {got:?} != {kind:?}")
        }
        other => panic!("expected InvalidResponse({kind:?}), got {other:?}"),
    }
}

pub fn assert_not_found(err: &Error, id: &ProviderId) {
    match err {
        Error::NotFound(got) => assert_eq!(got, &id.0, "NotFound id"),
        other => panic!("expected NotFound({}), got {other:?}", id.0),
    }
}

pub fn assert_no_provider(err: &Error) {
    match err {
        Error::NoProvider => {}
        other => panic!("expected NoProvider, got {other:?}"),
    }
}

pub fn assert_not_broker(err: &Error) {
    match err {
        Error::NotBroker => {}
        other => panic!("expected NotBroker, got {other:?}"),
    }
}

pub fn assert_backoff(d: Duration, down_count: usize) {
    let exp = down_count.min(8) as u32;
    let want = Duration::from_millis(100u64.saturating_mul(1u64 << exp));
    assert_eq!(d, want, "wait_backoff for {down_count} down");
}

pub fn assert_access_token(tok: &AccessToken, provider: &ProviderId, blob: &[u8], expires_at: u64) {
    assert_eq!(&tok.provider, provider);
    assert_eq!(tok.expires_at, expires_at);
    assert_eq!(tok.token.len(), 64, "access token is 64 hex chars");
    assert!(
        tok.token
            .chars()
            .all(|c| matches!(c, '0'..='9' | 'a'..='f')),
        "access token must be lowercase hex, got {:?}",
        tok.token
    );
    assert_ne!(
        tok.token.as_bytes(),
        blob,
        "agents must not see the refresh blob"
    );
    if !blob.is_empty() {
        assert!(
            !tok.token.as_bytes().windows(blob.len()).any(|w| w == blob),
            "refresh blob leaked inside access token"
        );
    }
}

pub fn assert_record(rec: &TokenRecord, generation: u64, blob: &[u8]) {
    assert_eq!(rec.generation, generation);
    assert_eq!(rec.blob.as_slice(), blob);
}

pub fn solo_node(id: &str) -> NodeInfo {
    NodeInfo {
        id: NodeId(id.to_string()),
        addr: Addr(format!("local://{id}")),
        kind: NodeKind::AlwaysOn,
        trusted: true,
    }
}

/// Parent tempdir kept alive plus a bootstrap single-node leader cluster.
pub fn solo_cluster() -> (tempfile::TempDir, Cluster) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("raft");
    let cluster = Cluster::bootstrap(dir, solo_node("solo"), ClusterConfig::default())
        .expect("bootstrap leader");
    assert!(
        cluster.is_leader(),
        "Cluster::bootstrap is leader for CPU broker tests"
    );
    (parent, cluster)
}

pub fn group3() -> (tempfile::TempDir, LocalGroup) {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir: PathBuf = parent.path().join("raft");
    let group = LocalGroup::start(3, dir, ClusterConfig::default()).expect("LocalGroup 3");
    (parent, group)
}

pub fn tick_until_leader(group: &mut LocalGroup, now: NowMs) -> usize {
    for _ in 0..256 {
        group.tick(now).expect("tick");
        for i in 0..group.len() {
            if group.get(i).expect("node").is_leader() {
                return i;
            }
        }
    }
    panic!("no leader elected in 256 ticks");
}

pub fn follower_index(group: &mut LocalGroup, leader: usize) -> usize {
    for i in 0..group.len() {
        if i != leader && !group.get(i).expect("node").is_leader() {
            return i;
        }
    }
    panic!("no follower in 3-node group");
}
