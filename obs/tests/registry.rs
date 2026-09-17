//! Group: RunRegistry register / get / list / set_status and fault cases.

mod common;

use prometheus_obs::{RunRegistry, RunStatus};

use common::{assert_duplicate_run, assert_unknown_run, sample_run};

#[test]
fn register_then_get_returns_the_record() {
    let mut reg = RunRegistry::new();
    let rec = sample_run("run-a");
    reg.register(rec.clone()).expect("register");
    let got = reg.get("run-a").expect("get");
    assert_eq!(got, rec);
}

#[test]
fn get_unknown_run_errors() {
    let reg = RunRegistry::new();
    assert_unknown_run(reg.get("missing"), "missing");
}

#[test]
fn duplicate_register_errors_and_does_not_replace() {
    let mut reg = RunRegistry::new();
    let mut first = sample_run("run-dup");
    first.status = RunStatus::Pending;
    reg.register(first.clone()).expect("first");

    let mut second = sample_run("run-dup");
    second.status = RunStatus::Running;
    second.kind = "eval".to_string();
    assert_duplicate_run(reg.register(second), "run-dup");

    let got = reg.get("run-dup").expect("get after duplicate");
    assert_eq!(got.status, RunStatus::Pending);
    assert_eq!(got.kind, "train");
}

#[test]
fn list_empty_then_all_registered_ids() {
    let mut reg = RunRegistry::new();
    assert!(
        reg.list().expect("list empty").is_empty(),
        "fresh registry lists no runs"
    );
    reg.register(sample_run("r1")).expect("r1");
    reg.register(sample_run("r2")).expect("r2");
    let mut ids: Vec<String> = reg
        .list()
        .expect("list")
        .into_iter()
        .map(|r| r.run_id)
        .collect();
    ids.sort();
    assert_eq!(ids, vec!["r1".to_string(), "r2".to_string()]);
}

#[test]
fn set_status_updates_only_status() {
    let mut reg = RunRegistry::new();
    let rec = sample_run("run-st");
    reg.register(rec.clone()).expect("register");
    reg.set_status("run-st", RunStatus::Running)
        .expect("running");
    let got = reg.get("run-st").expect("get");
    assert_eq!(got.status, RunStatus::Running);
    assert_eq!(got.run_id, rec.run_id);
    assert_eq!(got.kind, rec.kind);
    assert_eq!(got.config_hash, rec.config_hash);
    assert_eq!(got.created_at, rec.created_at);
    assert_eq!(got.parent_run_id, rec.parent_run_id);

    reg.set_status("run-st", RunStatus::Completed)
        .expect("completed");
    assert_eq!(reg.get("run-st").expect("get").status, RunStatus::Completed);
}

#[test]
fn set_status_unknown_run_errors() {
    let mut reg = RunRegistry::new();
    assert_unknown_run(reg.set_status("nope", RunStatus::Failed), "nope");
}

#[test]
fn list_reflects_status_changes() {
    let mut reg = RunRegistry::new();
    reg.register(sample_run("only")).expect("register");
    reg.set_status("only", RunStatus::Cancelled)
        .expect("cancel");
    let list = reg.list().expect("list");
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].run_id, "only");
    assert_eq!(list[0].status, RunStatus::Cancelled);
}

#[test]
fn parent_run_id_is_stored() {
    let mut reg = RunRegistry::new();
    let mut rec = sample_run("child");
    rec.parent_run_id = Some("parent".to_string());
    rec.kind = "eval".to_string();
    reg.register(rec.clone()).expect("register");
    assert_eq!(reg.get("child").expect("get"), rec);
}
