//! Independent I10 reference: spike detection, metric time series, dashboards.
//!
//! Slow and obvious. Production `prometheus_obs` must match this module exactly
//! (actions, error variants, messages, series order, snapshot shape). No GPU
//! surface; I10 is CPU-only (no `gpu` marker / feature).
//!
//! # Spike detection (locked; spec 5.5 does not give a formula)
//!
//! `SpikeConfig::new` / `SpikeLog::new` do **not** validate. Validation runs on
//! every `observe` call, reading the live `config` and `last_checkpoint_step`
//! fields (callers may mutate them between observes).
//!
//! 1. If `relative_threshold <= 0.0` → `Error::BadSpikeConfig("relative_threshold must be > 0")`.
//! 2. Else if `baseline_window == 0` → `Error::BadSpikeConfig("baseline_window must be > 0")`.
//! 3. Else if `sample.loss` is not finite (NaN / ±inf) →
//!    `Error::BadSpikeConfig("loss must be finite")`.
//!    Error paths leave history / `spikes` / `skipped_shards` unchanged.
//! 4. History is every **successfully** observed finite loss, in `observe` call
//!    order (warmup, non-spikes, and spikes all append).
//! 5. If history currently has fewer than `baseline_window` samples, append the
//!    loss and return `Ok([])` (warmup; never a spike).
//! 6. Else the baseline at this sample is the arithmetic mean of the **previous**
//!    `baseline_window` history values (the last `baseline_window` strictly
//!    earlier samples; sliding window, not an expanding mean and not a frozen
//!    first window). Then append the current loss.
//! 7. A spike iff `baseline > 0.0` AND
//!    `loss > baseline * (1.0 + relative_threshold)` (strict `>`).
//!    `baseline == 0` or negative ⇒ not a spike even for huge loss.
//!    Equality with the threshold is not a spike.
//! 8. On a spike, actions in this order:
//!    `Log { event }`, `SkipShard { shard_id }`,
//!    `Rollback { checkpoint_step: last_checkpoint_step }`.
//!    If **any already-recorded** spike has
//!    `|this.step - prev.step| <= page_window_steps` (inclusive), also append
//!    `Page { message }` where
//!    `message == format!("{EVENT_PAGE} run_id={} step={}", run_id, step)`
//!    (so it contains `EVENT_PAGE`, the run_id, and the decimal step).
//!    The current spike is not considered "already-recorded".
//! 9. After a spike, `spikes()` includes the `SpikeEvent` (baseline = the mean
//!    used) and `skipped_shards()` includes
//!    `SkipShardRecord { shard_id, step, run_id, reason: EVENT_LOSS_SPIKE }`.
//!
//! # MetricLog (locked)
//!
//! - `record` always `Ok(())` and stores `(name, labels) → SeriesPoint`.
//! - Same `step` overwrites that step's value; different steps keep all points.
//! - `series` returns points sorted by `step` ascending.
//! - Labels distinguish series. Empty labels are a distinct series.
//! - Completely empty log (no successful `record`) → `series` is
//!   `Error::EmptyHistory`.
//! - Otherwise unknown `(name, labels)` → `Error::UnknownMetric(name)`.
//!
//! # snapshot (locked)
//!
//! - `dashboard_id` / `generated_at` copied from the dashboard / argument.
//! - `recent_spikes` cloned in the given order.
//! - Panels emitted in `dashboard.panels` order as `PanelSnapshot { panel_id, points }`.
//! - `Timeseries` and `Gauge`: points = `log.series(panel.metric_name, empty labels)`.
//!   Missing metric / empty log errors propagate (`UnknownMetric` / `EmptyHistory`).
//!   If that call returned an empty vec, `Error::EmptyHistory`.
//! - `Table` and `Log`: `points = []`, do not query the log, never error for a
//!   missing metric.
//!
//! Snapshot JSON round-trips via `serde_json` (finite values).

#![allow(dead_code)]

use std::collections::BTreeMap;

use prometheus_obs::{
    Dashboard, DashboardSnapshot, Error, Labels, LossSample, MetricLog, PanelKind, PanelSnapshot,
    Result, SeriesPoint, SkipShardRecord, SpikeAction, SpikeConfig, SpikeEvent, SpikeLog,
    EVENT_LOSS_SPIKE, EVENT_PAGE,
};

/// Absolute tolerance for I10 f64 comparisons (spec-style FP32 parity).
pub const TOL: f64 = 1e-5;

pub fn empty_labels() -> Labels {
    Labels::new()
}

pub fn labels(pairs: &[(&str, &str)]) -> Labels {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect()
}

pub fn sample(step: u64, loss: f64, shard: &str, run: &str) -> LossSample {
    LossSample {
        step,
        loss,
        shard_id: shard.to_string(),
        run_id: run.to_string(),
    }
}

pub fn page_message(run_id: &str, step: u64) -> String {
    format!("{EVENT_PAGE} run_id={run_id} step={step}")
}

pub fn f64_close(got: f64, expected: f64) -> bool {
    if got.is_nan() && expected.is_nan() {
        return true;
    }
    if got.is_infinite() && expected.is_infinite() {
        return got.is_sign_positive() == expected.is_sign_positive();
    }
    (got - expected).abs() <= TOL
}

pub fn assert_f64_close(got: f64, expected: f64) {
    assert!(
        f64_close(got, expected),
        "f64 {got} not within {TOL} of {expected} (diff {})",
        (got - expected).abs()
    );
}

pub fn assert_spike_event_close(got: &SpikeEvent, want: &SpikeEvent) {
    assert_eq!(got.step, want.step, "SpikeEvent.step");
    assert_f64_close(got.loss, want.loss);
    assert_f64_close(got.baseline, want.baseline);
    assert_eq!(got.shard_id, want.shard_id, "SpikeEvent.shard_id");
    assert_eq!(got.run_id, want.run_id, "SpikeEvent.run_id");
}

pub fn assert_skip_eq(got: &SkipShardRecord, want: &SkipShardRecord) {
    assert_eq!(got.shard_id, want.shard_id);
    assert_eq!(got.step, want.step);
    assert_eq!(got.run_id, want.run_id);
    assert_eq!(got.reason, want.reason);
}

pub fn assert_actions_close(got: &[SpikeAction], want: &[SpikeAction]) {
    assert_eq!(
        got.len(),
        want.len(),
        "action count mismatch\n got: {got:?}\nwant: {want:?}"
    );
    for (g, w) in got.iter().zip(want) {
        match (g, w) {
            (SpikeAction::Log { event: ge }, SpikeAction::Log { event: we }) => {
                assert_spike_event_close(ge, we);
            }
            (SpikeAction::SkipShard { shard_id: a }, SpikeAction::SkipShard { shard_id: b }) => {
                assert_eq!(a, b, "SkipShard shard_id");
            }
            (
                SpikeAction::Rollback { checkpoint_step: a },
                SpikeAction::Rollback { checkpoint_step: b },
            ) => {
                assert_eq!(a, b, "Rollback checkpoint_step");
            }
            (SpikeAction::Page { message: a }, SpikeAction::Page { message: b }) => {
                assert_eq!(a, b, "Page message");
            }
            _ => panic!("action kind mismatch\n got: {g:?}\nwant: {w:?}"),
        }
    }
}

pub fn assert_error_eq(got: &Error, want: &Error) {
    match (got, want) {
        (Error::EmptyHistory, Error::EmptyHistory) => {}
        (Error::UnknownMetric(a), Error::UnknownMetric(b)) => {
            assert_eq!(a, b, "UnknownMetric name");
        }
        (Error::BadSpikeConfig(a), Error::BadSpikeConfig(b)) => {
            assert_eq!(a, b, "BadSpikeConfig message");
        }
        (Error::UnknownRun(a), Error::UnknownRun(b)) => assert_eq!(a, b),
        (Error::DuplicateRun(a), Error::DuplicateRun(b)) => assert_eq!(a, b),
        (Error::KindMismatch(a), Error::KindMismatch(b)) => assert_eq!(a, b),
        (Error::Other(a), Error::Other(b)) => assert_eq!(a, b),
        _ => panic!("error mismatch\n got: {got:?}\nwant: {want:?}"),
    }
}

pub fn assert_observe_result(got: Result<Vec<SpikeAction>>, want: Result<Vec<SpikeAction>>) {
    match (got, want) {
        (Ok(g), Ok(w)) => assert_actions_close(&g, &w),
        (Err(g), Err(w)) => assert_error_eq(&g, &w),
        (Ok(g), Err(w)) => panic!("prod Ok({g:?}), reference Err({w:?})"),
        (Err(g), Ok(w)) => panic!("prod Err({g:?}), reference Ok({w:?})"),
    }
}

pub fn assert_points_close(got: &[SeriesPoint], want: &[SeriesPoint]) {
    assert_eq!(
        got.len(),
        want.len(),
        "series length mismatch\n got: {got:?}\nwant: {want:?}"
    );
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.step, w.step, "SeriesPoint.step");
        assert_f64_close(g.value, w.value);
    }
}

pub fn assert_series_result(got: Result<Vec<SeriesPoint>>, want: Result<Vec<SeriesPoint>>) {
    match (got, want) {
        (Ok(g), Ok(w)) => assert_points_close(&g, &w),
        (Err(g), Err(w)) => assert_error_eq(&g, &w),
        (Ok(g), Err(w)) => panic!("prod Ok({g:?}), reference Err({w:?})"),
        (Err(g), Ok(w)) => panic!("prod Err({g:?}), reference Ok({w:?})"),
    }
}

/// Independent spike detector. Production `SpikeLog` must match this.
pub struct RefSpikeLog {
    pub config: SpikeConfig,
    pub last_checkpoint_step: u64,
    history: Vec<f64>,
    skipped: Vec<SkipShardRecord>,
    spikes: Vec<SpikeEvent>,
}

impl RefSpikeLog {
    pub fn new(config: SpikeConfig, last_checkpoint_step: u64) -> Self {
        Self {
            config,
            last_checkpoint_step,
            history: Vec::new(),
            skipped: Vec::new(),
            spikes: Vec::new(),
        }
    }

    pub fn skipped_shards(&self) -> &[SkipShardRecord] {
        &self.skipped
    }

    pub fn spikes(&self) -> &[SpikeEvent] {
        &self.spikes
    }

    pub fn observe(&mut self, sample: &LossSample) -> Result<Vec<SpikeAction>> {
        if self.config.relative_threshold <= 0.0 {
            return Err(Error::BadSpikeConfig("relative_threshold must be > 0"));
        }
        if self.config.baseline_window == 0 {
            return Err(Error::BadSpikeConfig("baseline_window must be > 0"));
        }
        if !sample.loss.is_finite() {
            return Err(Error::BadSpikeConfig("loss must be finite"));
        }

        let window = self.config.baseline_window;
        if self.history.len() < window {
            self.history.push(sample.loss);
            return Ok(Vec::new());
        }

        let start = self.history.len() - window;
        let mut sum = 0.0_f64;
        for &v in &self.history[start..] {
            sum += v;
        }
        let baseline = sum / (window as f64);

        let is_spike =
            baseline > 0.0 && sample.loss > baseline * (1.0 + self.config.relative_threshold);

        self.history.push(sample.loss);

        if !is_spike {
            return Ok(Vec::new());
        }

        let event = SpikeEvent {
            step: sample.step,
            loss: sample.loss,
            baseline,
            shard_id: sample.shard_id.clone(),
            run_id: sample.run_id.clone(),
        };

        let page = self.spikes.iter().any(|prev| {
            let dist = if sample.step >= prev.step {
                sample.step - prev.step
            } else {
                prev.step - sample.step
            };
            dist <= self.config.page_window_steps
        });

        self.spikes.push(event.clone());
        self.skipped.push(SkipShardRecord {
            shard_id: sample.shard_id.clone(),
            step: sample.step,
            run_id: sample.run_id.clone(),
            reason: EVENT_LOSS_SPIKE.to_string(),
        });

        let mut actions = vec![
            SpikeAction::Log {
                event: event.clone(),
            },
            SpikeAction::SkipShard {
                shard_id: sample.shard_id.clone(),
            },
            SpikeAction::Rollback {
                checkpoint_step: self.last_checkpoint_step,
            },
        ];
        if page {
            actions.push(SpikeAction::Page {
                message: page_message(&sample.run_id, sample.step),
            });
        }
        Ok(actions)
    }
}

/// Paired observe: production must match the reference (actions + log state).
pub fn observe_pair(
    prod: &mut SpikeLog,
    refer: &mut RefSpikeLog,
    sample: &LossSample,
) -> Result<Vec<SpikeAction>> {
    let want = refer.observe(sample);
    let got = prod.observe(sample);
    match (&got, &want) {
        (Ok(g), Ok(w)) => assert_actions_close(g, w),
        (Err(g), Err(w)) => assert_error_eq(g, w),
        (Ok(g), Err(w)) => panic!("prod Ok({g:?}), reference Err({w:?})"),
        (Err(g), Ok(w)) => panic!("prod Err({g:?}), reference Ok({w:?})"),
    }
    assert_eq!(prod.skipped_shards().len(), refer.skipped_shards().len());
    for (g, w) in prod.skipped_shards().iter().zip(refer.skipped_shards()) {
        assert_skip_eq(g, w);
    }
    assert_eq!(prod.spikes().len(), refer.spikes().len());
    for (g, w) in prod.spikes().iter().zip(refer.spikes()) {
        assert_spike_event_close(g, w);
    }
    got
}

/// Independent metric time-series store. Production `MetricLog` must match this.
#[derive(Clone, Default)]
pub struct RefMetricLog {
    /// name → labels → step → value. BTreeMap keeps steps sorted.
    inner: BTreeMap<String, BTreeMap<Labels, BTreeMap<u64, f64>>>,
}

impl RefMetricLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, name: &str, labels: &Labels, step: u64, value: f64) -> Result<()> {
        self.inner
            .entry(name.to_string())
            .or_default()
            .entry(labels.clone())
            .or_default()
            .insert(step, value);
        Ok(())
    }

    pub fn series(&self, name: &str, labels: &Labels) -> Result<Vec<SeriesPoint>> {
        if self.inner.is_empty() {
            return Err(Error::EmptyHistory);
        }
        match self
            .inner
            .get(name)
            .and_then(|by_labels| by_labels.get(labels))
        {
            None => Err(Error::UnknownMetric(name.to_string())),
            Some(by_step) => Ok(by_step
                .iter()
                .map(|(&step, &value)| SeriesPoint { step, value })
                .collect()),
        }
    }
}

pub fn record_pair(
    prod: &mut MetricLog,
    refer: &mut RefMetricLog,
    name: &str,
    labels: &Labels,
    step: u64,
    value: f64,
) {
    let want = refer.record(name, labels, step, value);
    let got = prod.record(name, labels, step, value);
    match (got, want) {
        (Ok(()), Ok(())) => {}
        (Err(g), Err(w)) => assert_error_eq(&g, &w),
        (Ok(()), Err(w)) => panic!("prod Ok, reference Err({w:?})"),
        (Err(g), Ok(())) => panic!("prod Err({g:?}), reference Ok"),
    }
}

pub fn series_pair(
    prod: &MetricLog,
    refer: &RefMetricLog,
    name: &str,
    labels: &Labels,
) -> Result<Vec<SeriesPoint>> {
    let want = refer.series(name, labels);
    let got = prod.series(name, labels);
    assert_series_result(got, want);
    refer.series(name, labels)
}

/// Independent dashboard snapshot. Production `snapshot` must match this.
pub fn snapshot(
    dashboard: &Dashboard,
    log: &RefMetricLog,
    spikes: &[SpikeEvent],
    generated_at: &str,
) -> Result<DashboardSnapshot> {
    let unlabeled = empty_labels();
    let mut panels = Vec::with_capacity(dashboard.panels.len());
    for panel in &dashboard.panels {
        let points = match panel.kind {
            PanelKind::Timeseries | PanelKind::Gauge => {
                let pts = log.series(&panel.metric_name, &unlabeled)?;
                if pts.is_empty() {
                    return Err(Error::EmptyHistory);
                }
                pts
            }
            PanelKind::Table | PanelKind::Log => Vec::new(),
        };
        panels.push(PanelSnapshot {
            panel_id: panel.id.clone(),
            points,
        });
    }
    Ok(DashboardSnapshot {
        dashboard_id: dashboard.id.clone(),
        generated_at: generated_at.to_string(),
        panels,
        recent_spikes: spikes.to_vec(),
    })
}

pub fn assert_snapshot_close(got: &DashboardSnapshot, want: &DashboardSnapshot) {
    assert_eq!(got.dashboard_id, want.dashboard_id, "dashboard_id");
    assert_eq!(got.generated_at, want.generated_at, "generated_at");
    assert_eq!(got.panels.len(), want.panels.len(), "panels length");
    for (g, w) in got.panels.iter().zip(&want.panels) {
        assert_eq!(g.panel_id, w.panel_id, "panel_id");
        assert_points_close(&g.points, &w.points);
    }
    assert_eq!(
        got.recent_spikes.len(),
        want.recent_spikes.len(),
        "recent_spikes length"
    );
    for (g, w) in got.recent_spikes.iter().zip(&want.recent_spikes) {
        assert_spike_event_close(g, w);
    }
}

pub fn snapshot_pair(
    dashboard: &Dashboard,
    prod_log: &MetricLog,
    ref_log: &RefMetricLog,
    spikes: &[SpikeEvent],
    generated_at: &str,
) -> Result<DashboardSnapshot> {
    let want = snapshot(dashboard, ref_log, spikes, generated_at);
    let got = prometheus_obs::snapshot(dashboard, prod_log, spikes, generated_at);
    match (&got, &want) {
        (Ok(g), Ok(w)) => assert_snapshot_close(g, w),
        (Err(g), Err(w)) => assert_error_eq(g, w),
        (Ok(g), Err(w)) => panic!("prod Ok({g:?}), reference Err({w:?})"),
        (Err(g), Ok(w)) => panic!("prod Err({g:?}), reference Ok({w:?})"),
    }
    got
}
