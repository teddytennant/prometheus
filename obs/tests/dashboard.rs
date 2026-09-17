//! Group: dashboard snapshot vs the independent reference.
//!
//! Checks: Timeseries/Gauge points from unlabeled series, Table/Log empty
//! points, missing metric, empty history, recent_spikes order, JSON round-trip.
//! Every test calls `snapshot` (and usually `record`) and must fail on the stub.

mod reference;

use prometheus_obs::{Dashboard, Error, MetricLog, Panel, PanelKind, SpikeEvent};

use reference::i10::{
    assert_snapshot_close, empty_labels, labels, record_pair, snapshot_pair, RefMetricLog,
};

fn panel(id: &str, title: &str, metric: &str, kind: PanelKind) -> Panel {
    Panel {
        id: id.to_string(),
        title: title.to_string(),
        metric_name: metric.to_string(),
        kind,
    }
}

fn dash(id: &str, title: &str, panels: Vec<Panel>) -> Dashboard {
    Dashboard {
        id: id.to_string(),
        title: title.to_string(),
        panels,
    }
}

#[test]
fn timeseries_points_come_from_unlabeled_series() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(&mut prod, &mut refer, "train.loss", &empty_labels(), 1, 1.0);
    record_pair(&mut prod, &mut refer, "train.loss", &empty_labels(), 2, 0.5);
    let d = dash(
        "run",
        "Training",
        vec![panel("p1", "Loss", "train.loss", PanelKind::Timeseries)],
    );
    let snap = snapshot_pair(&d, &prod, &refer, &[], "2026-09-16T12:00:00Z").expect("snapshot");
    assert_eq!(snap.dashboard_id, "run");
    assert_eq!(snap.generated_at, "2026-09-16T12:00:00Z");
    assert_eq!(snap.panels.len(), 1);
    assert_eq!(snap.panels[0].panel_id, "p1");
    assert_eq!(snap.panels[0].points.len(), 2);
    assert_eq!(snap.panels[0].points[0].step, 1);
    assert_eq!(snap.panels[0].points[1].step, 2);
    assert!(snap.recent_spikes.is_empty());
}

#[test]
fn gauge_also_loads_the_full_unlabeled_series() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(&mut prod, &mut refer, "lr", &empty_labels(), 0, 3e-4);
    record_pair(&mut prod, &mut refer, "lr", &empty_labels(), 10, 1e-4);
    let d = dash(
        "g",
        "Gauges",
        vec![panel("lr", "Learning rate", "lr", PanelKind::Gauge)],
    );
    let snap = snapshot_pair(&d, &prod, &refer, &[], "t0").expect("snapshot");
    assert_eq!(snap.panels[0].points.len(), 2);
}

#[test]
fn table_and_log_panels_have_empty_points_even_if_metric_missing() {
    let prod = MetricLog::new();
    let refer = RefMetricLog::new();
    let d = dash(
        "ops",
        "Ops",
        vec![
            panel("t", "Shards", "missing.table", PanelKind::Table),
            panel("l", "Events", "missing.log", PanelKind::Log),
        ],
    );
    let snap = snapshot_pair(&d, &prod, &refer, &[], "t0").expect("table/log on empty log");
    assert_eq!(snap.panels.len(), 2);
    assert!(snap.panels[0].points.is_empty());
    assert!(snap.panels[1].points.is_empty());
    assert_eq!(snap.panels[0].panel_id, "t");
    assert_eq!(snap.panels[1].panel_id, "l");
}

#[test]
fn timeseries_on_empty_log_is_empty_history() {
    let prod = MetricLog::new();
    let refer = RefMetricLog::new();
    let d = dash(
        "run",
        "Training",
        vec![panel("p1", "Loss", "train.loss", PanelKind::Timeseries)],
    );
    match snapshot_pair(&d, &prod, &refer, &[], "t0") {
        Err(Error::EmptyHistory) => {}
        other => panic!("expected EmptyHistory, got {other:?}"),
    }
}

#[test]
fn gauge_on_empty_log_is_empty_history() {
    let prod = MetricLog::new();
    let refer = RefMetricLog::new();
    let d = dash(
        "run",
        "Training",
        vec![panel("g", "LR", "lr", PanelKind::Gauge)],
    );
    match snapshot_pair(&d, &prod, &refer, &[], "t0") {
        Err(Error::EmptyHistory) => {}
        other => panic!("expected EmptyHistory, got {other:?}"),
    }
}

#[test]
fn missing_metric_on_non_empty_log_is_unknown_metric() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(&mut prod, &mut refer, "other", &empty_labels(), 0, 1.0);
    let d = dash(
        "run",
        "Training",
        vec![panel("p1", "Loss", "train.loss", PanelKind::Timeseries)],
    );
    match snapshot_pair(&d, &prod, &refer, &[], "t0") {
        Err(Error::UnknownMetric(n)) => assert_eq!(n, "train.loss"),
        other => panic!("expected UnknownMetric(train.loss), got {other:?}"),
    }
}

#[test]
fn labeled_series_is_not_used_by_snapshot() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(
        &mut prod,
        &mut refer,
        "train.loss",
        &labels(&[("gpu", "0")]),
        0,
        1.0,
    );
    let d = dash(
        "run",
        "Training",
        vec![panel("p1", "Loss", "train.loss", PanelKind::Timeseries)],
    );
    match snapshot_pair(&d, &prod, &refer, &[], "t0") {
        Err(Error::UnknownMetric(n)) => assert_eq!(n, "train.loss"),
        other => panic!("snapshot looks up empty labels, got {other:?}"),
    }
}

#[test]
fn mixed_panels_preserve_order_and_skip_log_lookup_for_table() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(&mut prod, &mut refer, "loss", &empty_labels(), 1, 2.0);
    let d = dash(
        "mix",
        "Mix",
        vec![
            panel("a", "Loss", "loss", PanelKind::Timeseries),
            panel("b", "Table", "nope", PanelKind::Table),
            panel("c", "Loss gauge", "loss", PanelKind::Gauge),
            panel("d", "Log", "nope", PanelKind::Log),
        ],
    );
    let snap = snapshot_pair(&d, &prod, &refer, &[], "t0").expect("mixed");
    assert_eq!(
        snap.panels
            .iter()
            .map(|p| p.panel_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "c", "d"]
    );
    assert_eq!(snap.panels[0].points.len(), 1);
    assert!(snap.panels[1].points.is_empty());
    assert_eq!(snap.panels[2].points.len(), 1);
    assert!(snap.panels[3].points.is_empty());
}

#[test]
fn recent_spikes_copied_in_given_order() {
    let prod = MetricLog::new();
    let refer = RefMetricLog::new();
    let d = dash(
        "run",
        "Training",
        vec![panel("l", "Events", "x", PanelKind::Log)],
    );
    let spikes = vec![
        SpikeEvent {
            step: 10,
            loss: 9.0,
            baseline: 1.0,
            shard_id: "s0".into(),
            run_id: "r".into(),
        },
        SpikeEvent {
            step: 20,
            loss: 8.0,
            baseline: 1.1,
            shard_id: "s1".into(),
            run_id: "r".into(),
        },
    ];
    let snap = snapshot_pair(&d, &prod, &refer, &spikes, "t0").expect("spikes");
    assert_eq!(snap.recent_spikes.len(), 2);
    assert_eq!(snap.recent_spikes[0].step, 10);
    assert_eq!(snap.recent_spikes[1].shard_id, "s1");
}

#[test]
fn snapshot_json_round_trips() {
    let mut prod = MetricLog::new();
    let mut refer = RefMetricLog::new();
    record_pair(&mut prod, &mut refer, "loss", &empty_labels(), 4, 0.25);
    let d = dash(
        "run-json",
        "JSON",
        vec![
            panel("ts", "Loss", "loss", PanelKind::Timeseries),
            panel("tb", "Table", "x", PanelKind::Table),
        ],
    );
    let spikes = vec![SpikeEvent {
        step: 4,
        loss: 0.25,
        baseline: 0.1,
        shard_id: "sh".into(),
        run_id: "run-json".into(),
    }];
    let snap = snapshot_pair(&d, &prod, &refer, &spikes, "2026-09-16T12:00:00Z").expect("snap");
    let value = serde_json::to_value(&snap).expect("to_value");
    let back: prometheus_obs::DashboardSnapshot =
        serde_json::from_value(value).expect("from_value");
    assert_snapshot_close(&back, &snap);
}

#[test]
fn empty_dashboard_snapshots_even_with_empty_log() {
    let prod = MetricLog::new();
    let refer = RefMetricLog::new();
    let d = dash("empty", "Empty", vec![]);
    let snap = snapshot_pair(&d, &prod, &refer, &[], "t0").expect("empty dash");
    assert!(snap.panels.is_empty());
    assert_eq!(snap.dashboard_id, "empty");
}
