//! Providers: token broker, transport validation, AIMD, failover
//! (spec 15.2, 15.5 H6, gate D4).
//!
//! Exactly one token broker refreshes. Each rotated refresh token is committed
//! to H4 Raft *before* it is used. Agents only ever see short-lived access
//! tokens. Responses are validated in the transport before any cache or log.
//! Swarm size follows AIMD. On a sustained outage the harness fails over; if
//! every provider is down it waits with backoff and never exits.
//!
//! CPU tests use fake tokens and in-process [`Cluster`]. `NowMs` is injected.

use prometheus_raft::{Cluster, TokenRecord};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

pub type NowMs = u64;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderId(pub String);

/// Short-lived access token. Never a refresh token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessToken {
    pub provider: ProviderId,
    pub token: String,
    pub expires_at: NowMs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportFault {
    RateLimitText,
    EmptyCompletion,
    TruncatedToolCall,
    MalformedJson,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: ProviderId,
    /// Access-token lifetime in ms. Refresh stays in Raft, not here.
    pub access_ttl_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AimdConfig {
    pub min_window: u32,
    pub max_window: u32,
    pub increase: u32,
    pub decrease: f64,
    pub latency_limit_ms: u64,
}

impl Default for AimdConfig {
    fn default() -> Self {
        Self {
            min_window: 1,
            max_window: 32,
            increase: 1,
            decrease: 0.5,
            latency_limit_ms: 30_000,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("not the token broker")]
    NotBroker,
    #[error("no provider available")]
    NoProvider,
    #[error("invalid response: {0:?}")]
    InvalidResponse(TransportFault),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("duplicate: {0}")]
    Duplicate(String),
    #[error("{0}")]
    Raft(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<prometheus_raft::Error> for Error {
    fn from(err: prometheus_raft::Error) -> Self {
        Error::Raft(err.to_string())
    }
}

/// Validate a provider body before cache or log. No silent default.
pub fn validate_response(_body: &[u8]) -> Result<Value> {
    unimplemented!("H6: validate_response")
}

/// Additive-increase / multiplicative-decrease window.
pub struct Aimd {
    config: AimdConfig,
    window: u32,
}

impl Aimd {
    pub fn new(config: AimdConfig) -> Self {
        let window = config.min_window.max(1);
        Self { config, window }
    }

    pub fn config(&self) -> &AimdConfig {
        &self.config
    }

    pub fn window(&self) -> u32 {
        self.window
    }

    pub fn on_success(&mut self, _latency_ms: u64) {
        unimplemented!("H6: Aimd::on_success")
    }

    pub fn on_429(&mut self) {
        unimplemented!("H6: Aimd::on_429")
    }

    pub fn on_outage(&mut self) {
        unimplemented!("H6: Aimd::on_outage")
    }
}

/// Ordered failover. Empty list is a wait, not an exit.
pub struct Failover {
    providers: Vec<ProviderId>,
    down: Vec<ProviderId>,
}

impl Failover {
    pub fn new(providers: Vec<ProviderId>) -> Self {
        Self {
            providers,
            down: Vec::new(),
        }
    }

    pub fn providers(&self) -> &[ProviderId] {
        &self.providers
    }

    pub fn down(&self) -> &[ProviderId] {
        &self.down
    }

    pub fn current(&self) -> Result<&ProviderId> {
        unimplemented!("H6: Failover::current")
    }

    pub fn mark_down(&mut self, _id: &ProviderId) -> Result<()> {
        unimplemented!("H6: Failover::mark_down")
    }

    pub fn mark_up(&mut self, _id: &ProviderId) -> Result<()> {
        unimplemented!("H6: Failover::mark_up")
    }

    /// Backoff when every provider is down. Never returns a "give up" error.
    pub fn wait_backoff(&self) -> Duration {
        unimplemented!("H6: Failover::wait_backoff")
    }
}

/// Token broker. Refresh commits to `cluster` before the access token is issued.
pub struct Broker {
    provider: ProviderConfig,
}

impl Broker {
    pub fn new(provider: ProviderConfig) -> Self {
        Self { provider }
    }

    pub fn provider(&self) -> &ProviderConfig {
        &self.provider
    }

    /// Caller must hold the H4 `TokenBroker` role. Commits a new refresh
    /// blob (tests: fake bytes) via [`Cluster::commit_token`], then returns
    /// a short-lived access token. Agents never see the refresh blob.
    pub fn refresh(
        &mut self,
        _cluster: &mut Cluster,
        _refresh_blob: &[u8],
        _now: NowMs,
    ) -> Result<(TokenRecord, AccessToken)> {
        unimplemented!("H6: Broker::refresh")
    }
}
