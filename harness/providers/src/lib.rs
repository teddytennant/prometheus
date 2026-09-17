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
use sha2::{Digest, Sha256};
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

fn invalid(kind: TransportFault) -> Error {
    Error::InvalidResponse(kind)
}

fn is_exact_rate_limit_text(body: &[u8]) -> bool {
    const NEEDLE: &[u8] = b"rate_limit";
    if body.len() != NEEDLE.len() {
        return false;
    }
    body.iter()
        .zip(NEEDLE.iter())
        .all(|(a, b)| a.to_ascii_lowercase() == *b)
}

fn json_error_is_rate_limit(obj: &serde_json::Map<String, Value>) -> bool {
    match obj.get("error") {
        Some(Value::String(s)) => s.eq_ignore_ascii_case("rate_limit"),
        _ => false,
    }
}

fn choices_usable(obj: &serde_json::Map<String, Value>) -> Option<&Vec<Value>> {
    match obj.get("choices") {
        Some(Value::Array(items)) if !items.is_empty() => Some(items),
        _ => None,
    }
}

fn tool_calls_of(choice: &Value) -> Option<&Value> {
    if let Some(tc) = choice.get("tool_calls") {
        return Some(tc);
    }
    choice.get("message").and_then(|m| m.get("tool_calls"))
}

fn arguments_truncated(call: &Value) -> bool {
    let Some(function) = call.get("function") else {
        return true;
    };
    if !function.is_object() {
        return true;
    }
    match function.get("arguments") {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => serde_json::from_str::<Value>(s).is_err(),
        Some(Value::Object(_)) | Some(Value::Array(_)) => false,
        Some(_) => true,
    }
}

fn choice_is_truncated_tool_call(choice: &Value) -> bool {
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    if finish != Some("length") {
        return false;
    }
    let Some(tool_calls) = tool_calls_of(choice) else {
        return false;
    };
    match tool_calls {
        Value::Array(items) => {
            if items.is_empty() {
                return false;
            }
            items.iter().any(arguments_truncated)
        }
        _ => true,
    }
}

/// Validate a provider body before cache or log. No silent default.
pub fn validate_response(body: &[u8]) -> Result<Value> {
    if is_exact_rate_limit_text(body) {
        return Err(invalid(TransportFault::RateLimitText));
    }
    let text = std::str::from_utf8(body).map_err(|_| invalid(TransportFault::MalformedJson))?;
    let value: Value =
        serde_json::from_str(text).map_err(|_| invalid(TransportFault::MalformedJson))?;
    let obj = match value.as_object() {
        Some(o) => o,
        None => return Err(invalid(TransportFault::MalformedJson)),
    };
    if json_error_is_rate_limit(obj) {
        return Err(invalid(TransportFault::RateLimitText));
    }
    let Some(choices) = choices_usable(obj) else {
        return Err(invalid(TransportFault::EmptyCompletion));
    };
    if choices.iter().any(choice_is_truncated_tool_call) {
        return Err(invalid(TransportFault::TruncatedToolCall));
    }
    Ok(value)
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

    pub fn on_success(&mut self, latency_ms: u64) {
        if latency_ms > self.config.latency_limit_ms {
            self.on_429();
            return;
        }
        self.window = self
            .window
            .saturating_add(self.config.increase)
            .min(self.config.max_window);
    }

    pub fn on_429(&mut self) {
        self.window = shrink_window(self.window, self.config.min_window, self.config.decrease);
    }

    pub fn on_outage(&mut self) {
        self.on_429();
    }
}

fn shrink_window(window: u32, min_window: u32, decrease: f64) -> u32 {
    let scaled = (window as f64) * decrease;
    let floored = if scaled.is_finite() {
        scaled.floor().max(0.0).min(u32::MAX as f64) as u32
    } else {
        0
    };
    // decrease >= 1 (or any multiplicative step that would not shrink): still
    // subtract 1 before the min_window clamp so a 429/outage shrinks.
    let shrunk = if floored >= window {
        window.saturating_sub(1)
    } else {
        floored
    };
    shrunk.max(min_window)
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
        self.providers
            .iter()
            .find(|p| !self.down.iter().any(|d| d == *p))
            .ok_or(Error::NoProvider)
    }

    pub fn mark_down(&mut self, id: &ProviderId) -> Result<()> {
        if !self.providers.iter().any(|p| p == id) {
            return Err(Error::NotFound(id.0.clone()));
        }
        if !self.down.iter().any(|d| d == id) {
            self.down.push(id.clone());
        }
        Ok(())
    }

    pub fn mark_up(&mut self, id: &ProviderId) -> Result<()> {
        if !self.providers.iter().any(|p| p == id) {
            return Err(Error::NotFound(id.0.clone()));
        }
        self.down.retain(|d| d != id);
        Ok(())
    }

    /// Backoff when every provider is down. Never returns a "give up" error.
    pub fn wait_backoff(&self) -> Duration {
        let exp = self.down.len().min(8) as u32;
        Duration::from_millis(100u64.saturating_mul(1u64 << exp))
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
        cluster: &mut Cluster,
        refresh_blob: &[u8],
        now: NowMs,
    ) -> Result<(TokenRecord, AccessToken)> {
        if !cluster.is_leader() {
            return Err(Error::NotBroker);
        }
        let rec = cluster.commit_token(refresh_blob.to_vec())?;
        let token = AccessToken {
            provider: self.provider.id.clone(),
            token: sha256_hex(refresh_blob),
            expires_at: now.saturating_add(self.provider.access_ttl_ms),
        };
        Ok((rec, token))
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}
