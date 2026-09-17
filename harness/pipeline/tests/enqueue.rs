//! Group: work_payload schema and tasks enqueued by start_*.

mod common;
mod reference;

use common::{
    default_pipeline_config, default_queue_config, drive_to_implement, drive_to_stub_must_fail,
    fresh_dir, implementer_extra, module, payloads_with_role, round_extra, NOW,
};
use prometheus_pipeline::{work_payload, Pipeline, ROLE_IMPLEMENTER, ROLE_ORACLE, ROLE_REVIEWER};
use reference::ref_work_payload;
use serde_json::{json, Value};

fn assert_payload_eq(prod: prometheus_leases::WorkItem, refer: prometheus_leases::WorkItem) {
    assert_eq!(prod.task_id.0, refer.task_id.0, "task_id");
    assert_eq!(prod.payload, refer.payload, "payload");
}

#[test]
fn work_payload_sets_module_and_role() {
    let m = module();
    let extra = json!({ "round": 0u64 });
    let prod = work_payload(&m, ROLE_ORACLE, extra.clone());
    let refer = ref_work_payload(&m, ROLE_ORACLE, extra);
    assert_payload_eq(prod.clone(), refer);
    assert_eq!(prod.payload["module"], m.0);
    assert_eq!(prod.payload["role"], ROLE_ORACLE);
    assert_eq!(prod.payload["round"], 0);
    assert_eq!(prod.task_id.0, format!("oracle/{}/0", m.0));
}

#[test]
fn work_payload_merges_extra_object_and_args_win() {
    let m = module();
    let extra = json!({
        "round": 2,
        "candidate": "7",
        "module": "spoof",
        "role": "spoof",
        "angle": "x",
    });
    let prod = work_payload(&m, ROLE_IMPLEMENTER, extra.clone());
    let refer = ref_work_payload(&m, ROLE_IMPLEMENTER, extra);
    assert_payload_eq(prod.clone(), refer);
    assert_eq!(prod.payload["module"], m.0, "args win over extra.module");
    assert_eq!(prod.payload["role"], ROLE_IMPLEMENTER);
    assert_eq!(prod.payload["candidate"], "7");
    assert_eq!(prod.payload["angle"], "x");
    assert_eq!(prod.task_id.0, format!("implementer/{}/2/7", m.0));
}

#[test]
fn work_payload_non_object_extra_nested() {
    let m = module();
    let extra = json!("not-an-object");
    let prod = work_payload(&m, ROLE_REVIEWER, extra.clone());
    let refer = ref_work_payload(&m, ROLE_REVIEWER, extra);
    assert_payload_eq(prod.clone(), refer);
    assert_eq!(prod.payload["extra"], "not-an-object");
    assert_eq!(prod.task_id.0, format!("reviewer/{}/0", m.0));
}

#[test]
fn work_payload_op_and_candidate_in_task_id() {
    let m = module();
    let extra = json!({ "round": 1, "op": "submit_candidate", "candidate": "0" });
    let prod = work_payload(&m, "pipeline", extra.clone());
    let refer = ref_work_payload(&m, "pipeline", extra);
    assert_payload_eq(prod.clone(), refer);
    assert_eq!(
        prod.task_id.0,
        format!("pipeline/{}/1/submit_candidate/0", m.0)
    );
}

#[test]
fn work_payload_is_deterministic() {
    let m = module();
    let extra = round_extra(0);
    let a = work_payload(&m, ROLE_ORACLE, extra.clone());
    let b = work_payload(&m, ROLE_ORACLE, extra);
    assert_eq!(a.task_id.0, b.task_id.0);
    assert_eq!(a.payload, b.payload);
}

#[test]
fn start_oracle_enqueues_one_oracle_task() {
    let (_parent, dir) = fresh_dir();
    let m = module();
    let mut p = Pipeline::create(
        &dir,
        m.clone(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("create");
    let tid = p.start_oracle(NOW).expect("start_oracle");
    let expected = work_payload(&m, ROLE_ORACLE, round_extra(0));
    assert_eq!(tid.0, expected.task_id.0);
    let oracles = payloads_with_role(p.queue(), ROLE_ORACLE);
    assert_eq!(oracles.len(), 1);
    assert_eq!(oracles[0]["role"], ROLE_ORACLE);
    assert_eq!(oracles[0]["module"], m.0);
    assert_eq!(oracles[0]["round"], 0);
    assert!(payloads_with_role(p.queue(), ROLE_IMPLEMENTER).is_empty());
    assert!(payloads_with_role(p.queue(), ROLE_REVIEWER).is_empty());
}

#[test]
fn start_implementers_enqueues_n_with_candidate_ids() {
    let (_parent, dir) = fresh_dir();
    let m = module();
    let mut p = Pipeline::create(
        &dir,
        m.clone(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("create");
    drive_to_stub_must_fail(&mut p, NOW);
    let ids = p.start_implementers(NOW).expect("start_implementers");
    assert_eq!(ids.len(), 3);
    let items = payloads_with_role(p.queue(), ROLE_IMPLEMENTER);
    assert_eq!(items.len(), 3);
    for (i, payload) in items.iter().enumerate() {
        assert_eq!(payload["role"], ROLE_IMPLEMENTER);
        assert_eq!(payload["module"], m.0);
        assert_eq!(payload["round"], 0);
        assert_eq!(payload["candidate"], i.to_string());
        let expected = work_payload(&m, ROLE_IMPLEMENTER, implementer_extra(0, &i.to_string()));
        assert_eq!(ids[i].0, expected.task_id.0);
    }
}

#[test]
fn start_implementers_respects_n_implementers() {
    let (_parent, dir) = fresh_dir();
    let cfg = prometheus_pipeline::PipelineConfig {
        n_implementers: 1,
        max_rounds: 3,
    };
    let mut p = Pipeline::create(&dir, module(), cfg, default_queue_config()).expect("create");
    drive_to_stub_must_fail(&mut p, NOW);
    let ids = p.start_implementers(NOW).expect("start_implementers");
    assert_eq!(ids.len(), 1);
    assert_eq!(payloads_with_role(p.queue(), ROLE_IMPLEMENTER).len(), 1);
}

#[test]
fn start_review_enqueues_one_reviewer_task() {
    let (_parent, dir) = fresh_dir();
    let m = module();
    let mut p = Pipeline::create(
        &dir,
        m.clone(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("create");
    drive_to_implement(&mut p, NOW);
    let tid = p.start_review(NOW).expect("start_review");
    let expected = work_payload(&m, ROLE_REVIEWER, round_extra(0));
    assert_eq!(tid.0, expected.task_id.0);
    let revs = payloads_with_role(p.queue(), ROLE_REVIEWER);
    assert_eq!(revs.len(), 1);
    assert_eq!(revs[0]["role"], ROLE_REVIEWER);
    assert_eq!(revs[0]["module"], m.0);
    assert_eq!(revs[0]["round"], 0);
}

#[test]
fn work_payload_null_and_array_extra() {
    let m = module();
    for extra in [Value::Null, json!([1, 2, 3]), json!(42)] {
        let prod = work_payload(&m, ROLE_ORACLE, extra.clone());
        let refer = ref_work_payload(&m, ROLE_ORACLE, extra.clone());
        assert_payload_eq(prod, refer);
    }
}
