//! Independent H6 reference: transport validation, AIMD, failover, token broker.
//!
//! Slow and obvious. Compiled only as a submodule of the integration tests.
//! Production `src/lib.rs` must never import this module.

#![allow(dead_code)]

use prometheus_providers::{
    AccessToken, AimdConfig, Error, ProviderConfig, ProviderId, Result, TransportFault,
};
use prometheus_raft::{Cluster, TokenRecord};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::time::Duration;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Lowercase hex SHA-256. Independent of production (production has no sha2 dep).
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn invalid(kind: TransportFault, _detail: &str) -> Error {
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

/// Independent `validate_response`. See `tests/common` for the locked check order.
pub fn validate_response(body: &[u8]) -> Result<Value> {
    if is_exact_rate_limit_text(body) {
        return Err(invalid(
            TransportFault::RateLimitText,
            "exact ASCII rate_limit",
        ));
    }
    let text = std::str::from_utf8(body)
        .map_err(|_| invalid(TransportFault::MalformedJson, "body is not UTF-8"))?;
    let value: Value = serde_json::from_str(text)
        .map_err(|_| invalid(TransportFault::MalformedJson, "body is not JSON"))?;
    let obj = match value.as_object() {
        Some(o) => o,
        None => {
            return Err(invalid(
                TransportFault::MalformedJson,
                "JSON is not an object",
            ))
        }
    };
    if json_error_is_rate_limit(obj) {
        return Err(invalid(
            TransportFault::RateLimitText,
            "JSON error rate_limit",
        ));
    }
    let Some(choices) = choices_usable(obj) else {
        return Err(invalid(
            TransportFault::EmptyCompletion,
            "missing or empty choices",
        ));
    };
    if choices.iter().any(choice_is_truncated_tool_call) {
        return Err(invalid(
            TransportFault::TruncatedToolCall,
            "finish_reason length with truncated tool_calls",
        ));
    }
    Ok(value)
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

/// Independent AIMD controller. Starts at `min_window.max(1)` like `Aimd::new`.
#[derive(Debug, Clone)]
pub struct RefAimd {
    config: AimdConfig,
    window: u32,
}

impl RefAimd {
    pub fn new(config: AimdConfig) -> Self {
        let window = config.min_window.max(1);
        Self { config, window }
    }

    pub fn window(&self) -> u32 {
        self.window
    }

    pub fn config(&self) -> &AimdConfig {
        &self.config
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

/// Independent failover: original order, unique down set, exponential backoff.
#[derive(Debug, Clone)]
pub struct RefFailover {
    providers: Vec<ProviderId>,
    down: Vec<ProviderId>,
}

impl RefFailover {
    pub fn new(providers: Vec<ProviderId>) -> Self {
        Self {
            providers,
            down: Vec::new(),
        }
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

    pub fn wait_backoff(&self) -> Duration {
        let exp = self.down.len().min(8) as u32;
        Duration::from_millis(100u64.saturating_mul(1u64 << exp))
    }

    pub fn down(&self) -> &[ProviderId] {
        &self.down
    }
}

pub fn expected_access_token(
    provider: &ProviderId,
    blob: &[u8],
    now: u64,
    access_ttl_ms: u64,
) -> AccessToken {
    AccessToken {
        provider: provider.clone(),
        token: sha256_hex(blob),
        expires_at: now.saturating_add(access_ttl_ms),
    }
}

pub fn expected_record(generation: u64, blob: &[u8]) -> TokenRecord {
    TokenRecord {
        generation,
        blob: blob.to_vec(),
    }
}

/// Independent broker refresh. Does **not** call Raft; generation is an argument
/// so tests can compare production (which commits first) without double-commit.
pub fn expected_refresh(
    cfg: &ProviderConfig,
    blob: &[u8],
    now: u64,
    generation: u64,
) -> (TokenRecord, AccessToken) {
    (
        expected_record(generation, blob),
        expected_access_token(&cfg.id, blob, now, cfg.access_ttl_ms),
    )
}

/// Leader check used by production `Broker::refresh`. CPU tests skip role claim.
pub fn refresh_requires_leader(cluster: &Cluster) -> Result<()> {
    if cluster.is_leader() {
        Ok(())
    } else {
        Err(Error::NotBroker)
    }
}
