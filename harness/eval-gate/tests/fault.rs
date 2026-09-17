//! Group: fault-injection. Torn / duplicate / corrupt suite files fail closed.
//! Error strings must still not contain planted task text.

mod common;
mod reference;

use prometheus_eval_gate::{Error, EvalGate, Subject};

use common::{
    assert_no_leak, canary_items, cfg, fresh_root, seed_re_bench, seed_standard, CANARY_BODY, HV,
};

fn scan(err: &Error, where_: &str) {
    assert_no_leak(&format!("{err}"), where_);
    assert_no_leak(&format!("{err:?}"), where_);
}

#[test]
fn torn_current_json_fails_closed() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    std::fs::write(root.join("re_bench").join("current.json"), b"{").unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.suite_hash("re_bench") {
        Err(e) => {
            scan(&e, "torn current suite_hash");
            match e {
                Error::Message(_) => {}
                other => panic!("expected Message, got {other:?}"),
            }
        }
        Ok(h) => panic!("torn current.json must not succeed, got {h}"),
    }
}

#[test]
fn torn_items_jsonl_fails_closed_without_leaking_canary() {
    let (_tmp, root) = fresh_root();
    let hash = seed_re_bench(&root);
    common::seed_perfect_genome(&root);
    let path = common::generation_items_path(&root, "re_bench", &hash);
    let mut body = serde_json::to_string(&canary_items()[0]).unwrap();
    body.push('\n');
    body.push_str("{\"id\":\"i2\",\"prompt\":\"");
    body.push_str(CANARY_BODY);
    // torn, no closing quote
    std::fs::write(&path, body).unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.request(Subject::GenomeRev("perfect".into()), "re_bench", 0) {
        Err(e) => scan(&e, "torn items"),
        Ok(s) => panic!("torn items must fail, got {s:?}"),
    }
}

#[test]
fn duplicate_item_ids_fail_closed() {
    let (_tmp, root) = fresh_root();
    let hash = seed_re_bench(&root);
    common::seed_perfect_genome(&root);
    let path = common::generation_items_path(&root, "re_bench", &hash);
    let line = serde_json::to_string(&canary_items()[0]).unwrap();
    std::fs::write(&path, format!("{line}\n{line}\n")).unwrap();
    // current.json still has the old hash; either hash mismatch or duplicate should fail.
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.request(Subject::GenomeRev("perfect".into()), "re_bench", 0) {
        Err(e) => scan(&e, "duplicate ids"),
        Ok(s) => panic!("duplicate ids must fail, got {s:?}"),
    }
}

#[test]
fn subject_json_array_fails_closed() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let path = root.join("subjects").join("genome").join("perfect.json");
    std::fs::write(&path, b"[1,2,3]").unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.request(Subject::GenomeRev("perfect".into()), "re_bench", 0) {
        Err(e) => {
            scan(&e, "subject array");
            match e {
                Error::Message(_) | Error::SubjectNotFound(_) => {}
                other => panic!("expected Message or SubjectNotFound, got {other:?}"),
            }
        }
        Ok(s) => panic!("array subject must fail, got {s:?}"),
    }
}

#[test]
fn extra_item_keys_fail_closed() {
    let (_tmp, root) = fresh_root();
    let hash = seed_re_bench(&root);
    common::seed_perfect_genome(&root);
    let path = common::generation_items_path(&root, "re_bench", &hash);
    std::fs::write(
        &path,
        format!(
            "{{\"id\":\"i1\",\"prompt\":\"{CANARY_BODY}\",\"answer\":\"alpha\",\"weight_milli\":400,\"secret\":\"leak-me\"}}\n"
        ),
    )
    .unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.request(Subject::GenomeRev("perfect".into()), "re_bench", 0) {
        Err(e) => {
            scan(&e, "extra keys");
            assert!(
                !format!("{e}").contains("leak-me"),
                "extra item field leaked in error: {e}"
            );
        }
        Ok(_) => panic!("unknown item fields must fail closed"),
    }
}

#[test]
fn missing_items_file_fails_closed() {
    let (_tmp, root) = fresh_root();
    let hash = seed_re_bench(&root);
    common::seed_perfect_genome(&root);
    std::fs::remove_file(common::generation_items_path(&root, "re_bench", &hash)).unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.request(Subject::GenomeRev("perfect".into()), "re_bench", 0) {
        Err(e) => scan(&e, "missing items"),
        Ok(s) => panic!("missing items must fail, got {s:?}"),
    }
}

#[test]
fn hash_mismatch_between_current_and_items_fails_closed() {
    let (_tmp, root) = fresh_root();
    seed_re_bench(&root);
    common::seed_perfect_genome(&root);
    let cur_path = root.join("re_bench").join("current.json");
    let mut cur: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&cur_path).unwrap()).unwrap();
    cur["suite_hash"] =
        serde_json::json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    std::fs::write(&cur_path, serde_json::to_vec(&cur).unwrap()).unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.suite_hash("re_bench") {
        Err(e) => scan(&e, "hash mismatch"),
        Ok(h) => panic!("mismatched current.json must fail, got {h}"),
    }
}

#[test]
fn pending_with_duplicate_ids_refuses_rotate() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let dir = root.join("re_bench").join("pending");
    std::fs::create_dir_all(&dir).unwrap();
    let line = r#"{"id":"dup","prompt":"CANARY_PROMPT_NEVER_LEAVE_GATE_7f3a9c2e","answer":"a","weight_milli":1}"#;
    std::fs::write(dir.join("items.jsonl"), format!("{line}\n{line}\n")).unwrap();
    std::fs::write(dir.join("meta.json"), r#"{"written_ms":50,"compute_ms":1}"#).unwrap();
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.rotate("re_bench", 1, prometheus_eval_gate::ROTATION_MS) {
        Err(e) => scan(&e, "pending dups"),
        Ok(h) => panic!("duplicate pending ids must not rotate, got {h}"),
    }
}

#[test]
fn corrupt_suite_does_not_break_sibling_suite() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    std::fs::write(root.join("re_bench").join("current.json"), b"not-json").unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.suite_hash("re_bench") {
        Err(e) => scan(&e, "corrupt re_bench"),
        Ok(_) => panic!("corrupt re_bench must fail"),
    }
    let h = gate.suite_hash("mle_bench").expect("sibling suite");
    common::assert_hash_shape(&h);
}
