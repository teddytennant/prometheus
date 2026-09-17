//! Observability: metrics, tracing, run registry, dashboards, spike diagnosis
//! (spec 15.5 F2 + I10).
//!
//! F1 contracts this crate speaks:
//! - run records are identified by a `run_id` string
//! - the event log (when a run emits one) matches `prometheus.event_log`
//!   (`contracts/schemas/v1/event_log.schema.json`): hash-chained, seq,
//!   prev_hash, hash, timestamp, event_type, payload, payload_hash
//!
//! F2 is the in-process side (metrics, tracer, registry). I10 adds dashboards
//! (JSON snapshots a TS frontend can render) and the spec 5.5 loss-spike
//! policy: rollback, skip the shard, log it, page on the second spike within
//! 10k steps. Batch reconstruction for a spiked step is
//! [`prometheus_loader::reconstruct`] (spec 7.2).

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Batch reconstruction for spike diagnosis (spec 7.2).
pub use prometheus_loader::{
    reconstruct as reconstruct_batch, Batch, LoaderState, PackedSequence, Packing,
};

/// Failures from the registry or a metric name that is already a different kind.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unknown run {0}")]
    UnknownRun(String),
    #[error("run {0} already registered")]
    DuplicateRun(String),
    #[error("metric {0} already registered as a different kind")]
    KindMismatch(String),
    #[error("spike log is empty")]
    EmptyHistory,
    #[error("unknown metric {0}")]
    UnknownMetric(String),
    #[error("spike config: {0}")]
    BadSpikeConfig(&'static str),
    #[error("obs: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Sorted label map. Empty map is the unlabeled series.
pub type Labels = BTreeMap<String, String>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    Counter,
    Gauge,
    Histogram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// One registered training / eval / verify run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub kind: String,
    pub status: RunStatus,
    pub config_hash: String,
    pub created_at: String,
    pub parent_run_id: Option<String>,
}

/// Hash-chained event. Field names match the F1 `event_log` schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub schema_id: String,
    pub schema_version: u32,
    pub seq: u64,
    pub prev_hash: String,
    pub hash: String,
    pub timestamp: String,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub payload_hash: String,
    pub task_id: Option<String>,
    pub attempt: Option<u64>,
    pub node_id: Option<String>,
}

pub const EVENT_SCHEMA_ID: &str = "prometheus.event_log";
pub const EVENT_SCHEMA_VERSION: u32 = 1;
/// Genesis prev_hash: 64 zero hex chars.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// In-process metric registry. Names are dotted (`train.loss.ce`).
#[derive(Debug, Default)]
pub struct Metrics {
    kinds: BTreeMap<String, MetricKind>,
    series: BTreeMap<String, BTreeMap<Labels, f64>>,
}

impl Metrics {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_kind(&mut self, name: &str, kind: MetricKind) -> Result<()> {
        match self.kinds.get(name) {
            None => {
                self.kinds.insert(name.to_string(), kind);
                Ok(())
            }
            Some(existing) if *existing == kind => Ok(()),
            Some(_) => Err(Error::KindMismatch(name.to_string())),
        }
    }

    fn add(&mut self, name: &str, labels: &Labels, amount: f64) {
        let series = self.series.entry(name.to_string()).or_default();
        *series.entry(labels.clone()).or_insert(0.0) += amount;
    }

    pub fn inc(&mut self, name: &str, labels: &Labels, amount: f64) -> Result<()> {
        self.ensure_kind(name, MetricKind::Counter)?;
        self.add(name, labels, amount);
        Ok(())
    }

    pub fn set(&mut self, name: &str, labels: &Labels, value: f64) -> Result<()> {
        self.ensure_kind(name, MetricKind::Gauge)?;
        self.series
            .entry(name.to_string())
            .or_default()
            .insert(labels.clone(), value);
        Ok(())
    }

    pub fn observe(&mut self, name: &str, labels: &Labels, value: f64) -> Result<()> {
        self.ensure_kind(name, MetricKind::Histogram)?;
        self.add(name, labels, value);
        Ok(())
    }

    /// Snapshot of one series. Missing name is `Ok(None)`.
    pub fn get(&self, name: &str, labels: &Labels) -> Result<Option<f64>> {
        Ok(self
            .series
            .get(name)
            .and_then(|series| series.get(labels).copied()))
    }
}

/// One open span. Dropping it without `end` is a leak the tests catch.
#[derive(Debug)]
pub struct Span {
    pub name: String,
    pub start: SystemTime,
}

#[derive(Debug, Clone)]
struct OpenSpan {
    name: String,
    start: SystemTime,
}

/// In-process tracer. Spans nest; `current` is the innermost open span.
#[derive(Debug, Default)]
pub struct Tracer {
    stack: Vec<OpenSpan>,
}

impl Tracer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&mut self, name: &str) -> Result<Span> {
        let span = Span {
            name: name.to_string(),
            start: SystemTime::now(),
        };
        self.stack.push(OpenSpan {
            name: span.name.clone(),
            start: span.start,
        });
        Ok(span)
    }

    pub fn end(&mut self, span: Span) -> Result<u128> {
        let matches_innermost = self
            .stack
            .last()
            .is_some_and(|top| top.name == span.name && top.start == span.start);
        if !matches_innermost {
            return Err(Error::Other(format!(
                "span '{}' is not the innermost open span on this tracer",
                span.name
            )));
        }
        self.stack.pop();
        let ms = span.start.elapsed().map(|d| d.as_millis()).unwrap_or(0);
        Ok(ms)
    }

    pub fn current(&self) -> Option<&str> {
        self.stack.last().map(|s| s.name.as_str())
    }
}

/// In-process run registry. `run_id` is unique.
#[derive(Debug, Default)]
pub struct RunRegistry {
    runs: BTreeMap<String, RunRecord>,
    events: BTreeMap<String, Vec<Event>>,
}

impl RunRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, record: RunRecord) -> Result<()> {
        if self.runs.contains_key(&record.run_id) {
            return Err(Error::DuplicateRun(record.run_id));
        }
        self.runs.insert(record.run_id.clone(), record);
        Ok(())
    }

    pub fn get(&self, run_id: &str) -> Result<RunRecord> {
        self.runs
            .get(run_id)
            .cloned()
            .ok_or_else(|| Error::UnknownRun(run_id.to_string()))
    }

    pub fn set_status(&mut self, run_id: &str, status: RunStatus) -> Result<()> {
        match self.runs.get_mut(run_id) {
            Some(record) => {
                record.status = status;
                Ok(())
            }
            None => Err(Error::UnknownRun(run_id.to_string())),
        }
    }

    pub fn list(&self) -> Result<Vec<RunRecord>> {
        Ok(self.runs.values().cloned().collect())
    }

    /// Append a hash-chained event for `run_id`. First event uses `GENESIS_HASH`.
    pub fn append_event(
        &mut self,
        run_id: &str,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<Event> {
        if !self.runs.contains_key(run_id) {
            return Err(Error::UnknownRun(run_id.to_string()));
        }
        let chain = self.events.entry(run_id.to_string()).or_default();
        let (seq, prev_hash) = match chain.last() {
            None => (1_u64, GENESIS_HASH.to_string()),
            Some(prev) => (prev.seq + 1, prev.hash.clone()),
        };
        let timestamp = rfc3339_now();
        let payload_hash = payload_hash(&payload);
        let hash = event_hash(seq, &prev_hash, &payload_hash, &timestamp, event_type);
        let event = Event {
            schema_id: EVENT_SCHEMA_ID.to_string(),
            schema_version: EVENT_SCHEMA_VERSION,
            seq,
            prev_hash,
            hash,
            timestamp,
            event_type: event_type.to_string(),
            payload,
            payload_hash,
            task_id: None,
            attempt: None,
            node_id: None,
        };
        chain.push(event.clone());
        Ok(event)
    }

    pub fn events(&self, run_id: &str) -> Result<Vec<Event>> {
        if !self.runs.contains_key(run_id) {
            return Err(Error::UnknownRun(run_id.to_string()));
        }
        Ok(self.events.get(run_id).cloned().unwrap_or_default())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for &b in digest.as_slice() {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn canonicalize(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            let mut out = serde_json::Map::new();
            for k in keys {
                out.insert(k.clone(), canonicalize(&map[k]));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize).collect())
        }
        other => other.clone(),
    }
}

fn canonical_json_bytes(payload: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(&canonicalize(payload)).expect("canonical JSON is serializable")
}

/// SHA-256 of canonical JSON payload bytes, lowercase hex.
pub fn payload_hash(payload: &serde_json::Value) -> String {
    sha256_hex(&canonical_json_bytes(payload))
}

/// SHA-256 of `seq | prev_hash | payload_hash | timestamp | event_type`.
pub fn event_hash(
    seq: u64,
    prev_hash: &str,
    payload_hash: &str,
    timestamp: &str,
    event_type: &str,
) -> String {
    let concat = format!("{seq}|{prev_hash}|{payload_hash}|{timestamp}|{event_type}");
    sha256_hex(concat.as_bytes())
}

fn rfc3339_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    unix_secs_to_rfc3339_utc(secs)
}

fn unix_secs_to_rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3_600;
    let min = (rem % 3_600) / 60;
    let sec = rem % 60;
    let (year, month, day) = civil_from_unix_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Civil date from days since Unix epoch (Howard Hinnant `civil_from_days`).
fn civil_from_unix_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = i64::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    if month <= 2 {
        year += 1;
    }
    (year, month, day)
}

/// Spec 5.5: page a human on the second spike within this many steps.
pub const PAGE_WINDOW_STEPS: u64 = 10_000;

pub const EVENT_LOSS_SPIKE: &str = "loss_spike";
pub const EVENT_SKIP_SHARD: &str = "skip_shard";
pub const EVENT_PAGE: &str = "page";

/// One training-step loss observation, tagged with the data shard that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LossSample {
    pub step: u64,
    pub loss: f64,
    pub shard_id: String,
    pub run_id: String,
}

/// A detected loss spike (spec 5.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub step: u64,
    pub loss: f64,
    pub baseline: f64,
    pub shard_id: String,
    pub run_id: String,
}

/// Record that a shard was skipped after a spike.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkipShardRecord {
    pub shard_id: String,
    pub step: u64,
    pub run_id: String,
    pub reason: String,
}

/// Actions the spike policy emits for one observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum SpikeAction {
    /// Roll back to the last in-memory checkpoint (spec 5.5).
    Rollback { checkpoint_step: u64 },
    /// Skip the offending data shard and keep going.
    SkipShard { shard_id: String },
    /// Append a `loss_spike` event to the run log.
    Log { event: SpikeEvent },
    /// Page a human. Spec 5.5: second spike inside [`PAGE_WINDOW_STEPS`].
    Page { message: String },
}

/// Thresholds for spike detection. The 10k-step page window is spec-locked;
/// relative threshold and baseline window are oracle-locked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpikeConfig {
    pub page_window_steps: u64,
    pub relative_threshold: f64,
    pub baseline_window: usize,
}

impl SpikeConfig {
    pub fn new(page_window_steps: u64, relative_threshold: f64, baseline_window: usize) -> Self {
        Self {
            page_window_steps,
            relative_threshold,
            baseline_window,
        }
    }
}

/// Stateful loss-spike policy (spec 5.5).
///
/// On a spike: rollback to the last in-memory checkpoint, skip the shard, log
/// it. A second spike within `page_window_steps` also pages a human.
#[derive(Debug, Clone)]
pub struct SpikeLog {
    pub config: SpikeConfig,
    pub last_checkpoint_step: u64,
    skipped: Vec<SkipShardRecord>,
    spikes: Vec<SpikeEvent>,
}

impl SpikeLog {
    pub fn new(config: SpikeConfig, last_checkpoint_step: u64) -> Self {
        Self {
            config,
            last_checkpoint_step,
            skipped: Vec::new(),
            spikes: Vec::new(),
        }
    }

    /// Record one loss sample. Returns the actions to take (possibly empty).
    pub fn observe(&mut self, sample: &LossSample) -> Result<Vec<SpikeAction>> {
        let _ = sample;
        unimplemented!("I10 SpikeLog::observe")
    }

    pub fn skipped_shards(&self) -> &[SkipShardRecord] {
        &self.skipped
    }

    pub fn spikes(&self) -> &[SpikeEvent] {
        &self.spikes
    }
}

/// One (step, value) point on a dashboard series.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeriesPoint {
    pub step: u64,
    pub value: f64,
}

/// Step-stamped metric history. Distinct from [`Metrics`], which stores only
/// the current value of each series.
#[derive(Debug, Default, Clone)]
pub struct MetricLog {
    _private: (),
}

impl MetricLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, name: &str, labels: &Labels, step: u64, value: f64) -> Result<()> {
        let _ = (name, labels, step, value);
        unimplemented!("I10 MetricLog::record")
    }

    pub fn series(&self, name: &str, labels: &Labels) -> Result<Vec<SeriesPoint>> {
        let _ = (name, labels);
        unimplemented!("I10 MetricLog::series")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelKind {
    Timeseries,
    Gauge,
    Table,
    Log,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Panel {
    pub id: String,
    pub title: String,
    pub metric_name: String,
    pub kind: PanelKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dashboard {
    pub id: String,
    pub title: String,
    pub panels: Vec<Panel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelSnapshot {
    pub panel_id: String,
    pub points: Vec<SeriesPoint>,
}

/// JSON-serializable dashboard view for a TS frontend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub dashboard_id: String,
    pub generated_at: String,
    pub panels: Vec<PanelSnapshot>,
    pub recent_spikes: Vec<SpikeEvent>,
}

/// Render a dashboard from a metric log and recent spikes.
pub fn snapshot(
    dashboard: &Dashboard,
    log: &MetricLog,
    spikes: &[SpikeEvent],
    generated_at: &str,
) -> Result<DashboardSnapshot> {
    let _ = (dashboard, log, spikes, generated_at);
    unimplemented!("I10 snapshot")
}
