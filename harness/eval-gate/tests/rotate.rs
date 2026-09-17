//! Group: `rotate` installs a suite written after `cutoff_ms`; old suite_hash
//! stays on disk; pending is consumed; quarterly interval is enforced.

mod common;
mod reference;

use prometheus_eval_gate::{Error, EvalGate, Subject, ROTATION_MS};

use common::{
    assert_hash_shape, cfg, fresh_root, generation_items_path, leak_scan_scores, pending_items,
    seed_standard, HV,
};

fn setup_pending(root: &std::path::Path, suite: &str, n: usize, written_ms: u64, compute_ms: u64) {
    reference::install_pending(root, suite, pending_items(suite, n), written_ms, compute_ms)
        .expect("pending");
}

#[test]
fn rotate_installs_new_hash_and_request_uses_it() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let old = {
        let gate = EvalGate::open(cfg(&root, HV)).unwrap();
        gate.suite_hash("re_bench").unwrap()
    };
    setup_pending(&root, "re_bench", 3, 10, 9_000);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let new_hash = gate.rotate("re_bench", 1, ROTATION_MS).expect("rotate");
    assert_hash_shape(&new_hash);
    assert_ne!(new_hash, old);
    assert_eq!(
        new_hash,
        reference::hash_items(&pending_items("re_bench", 3))
    );
    assert_eq!(gate.suite_hash("re_bench").unwrap(), new_hash);

    let mut answers = std::collections::BTreeMap::new();
    for it in pending_items("re_bench", 3) {
        answers.insert(it.id, it.answer);
    }
    reference::install_subject(&root, &Subject::GenomeRev("after-rot".into()), &answers).unwrap();
    let scores = gate
        .request(
            Subject::GenomeRev("after-rot".into()),
            "re_bench",
            ROTATION_MS,
        )
        .unwrap();
    assert_eq!(scores.suite_hash, new_hash);
    assert_eq!(scores.n_items, 3);
    assert_eq!(scores.rci_milli, 750);
    assert_eq!(scores.compute_ms, 9_000);
    leak_scan_scores(&scores, "after rotate");
}

#[test]
fn rotate_leaves_old_generation_on_disk() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let gate0 = EvalGate::open(cfg(&root, HV)).unwrap();
    let old = gate0.suite_hash("re_bench").unwrap();
    drop(gate0);
    assert!(generation_items_path(&root, "re_bench", &old).is_file());
    setup_pending(&root, "re_bench", 2, 50, 8_000);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let new_hash = gate.rotate("re_bench", 1, ROTATION_MS).unwrap();
    assert!(
        generation_items_path(&root, "re_bench", &old).is_file(),
        "old suite_hash must stay queryable on disk"
    );
    assert_ne!(new_hash, old);
}

#[test]
fn rotate_refused_if_interval_not_elapsed() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    setup_pending(&root, "re_bench", 2, 50, 8_000);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.rotate("re_bench", 1, ROTATION_MS - 1) {
        Err(Error::RotationRefused(msg)) => {
            common::assert_no_leak(&msg, "early rotate");
            common::assert_no_leak(
                &format!("{}", Error::RotationRefused(msg.clone())),
                "display",
            );
        }
        other => panic!("expected RotationRefused, got {other:?}"),
    }
}

#[test]
fn rotate_refused_if_items_not_written_after_cutoff() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    setup_pending(&root, "re_bench", 2, 10, 8_000);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.rotate("re_bench", 10, ROTATION_MS) {
        Err(Error::RotationRefused(msg)) => common::assert_no_leak(&msg, "cutoff equal"),
        other => panic!("expected RotationRefused, got {other:?}"),
    }
    match gate.rotate("re_bench", 11, ROTATION_MS) {
        Err(Error::RotationRefused(msg)) => common::assert_no_leak(&msg, "cutoff after written"),
        other => panic!("expected RotationRefused, got {other:?}"),
    }
}

#[test]
fn rotate_refused_without_pending() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.rotate("paperbench", 0, ROTATION_MS) {
        Err(Error::RotationRefused(msg)) => common::assert_no_leak(&msg, "no pending"),
        other => panic!("expected RotationRefused, got {other:?}"),
    }
}

#[test]
fn rotate_refused_empty_pending() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    reference::install_pending(&root, "speedrun", vec![], 50, 1).unwrap();
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.rotate("speedrun", 1, ROTATION_MS) {
        Err(Error::RotationRefused(msg)) => common::assert_no_leak(&msg, "empty pending"),
        other => panic!("expected RotationRefused, got {other:?}"),
    }
}

#[test]
fn rotate_consumes_pending_so_second_needs_new_items() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    setup_pending(&root, "mle_bench", 2, 40, 7_000);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let _ = gate
        .rotate("mle_bench", 1, ROTATION_MS)
        .expect("first rotate");
    match gate.rotate("mle_bench", 1, ROTATION_MS.saturating_mul(2)) {
        Err(Error::RotationRefused(_)) => {}
        other => panic!("second rotate without pending: {other:?}"),
    }
}

#[test]
fn rotate_then_reopen_sees_new_hash() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    setup_pending(&root, "paperbench", 4, 80, 11_000);
    let new_hash = {
        let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
        gate.rotate("paperbench", 1, ROTATION_MS).unwrap()
    };
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    assert_eq!(gate.suite_hash("paperbench").unwrap(), new_hash);
}

#[test]
fn rotate_return_value_is_not_task_text() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    setup_pending(&root, "re_bench", 2, 20, 1);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let h = gate.rotate("re_bench", 1, ROTATION_MS).unwrap();
    common::assert_no_leak(&h, "rotate return");
    assert_hash_shape(&h);
}

#[test]
fn first_install_via_rotate_allowed_at_now_zero() {
    let (_tmp, root) = fresh_root();
    setup_pending(&root, "re_bench", 1, 5, 100);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let h = gate.rotate("re_bench", 0, 0).expect("first rotate");
    assert_eq!(h, reference::hash_items(&pending_items("re_bench", 1)));
}

#[test]
fn rotate_refused_if_cutoff_after_now() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    setup_pending(&root, "re_bench", 2, 9_999_999, 1);
    let mut gate = EvalGate::open(cfg(&root, HV)).unwrap();
    match gate.rotate("re_bench", ROTATION_MS + 1, ROTATION_MS) {
        Err(Error::RotationRefused(msg)) => common::assert_no_leak(&msg, "future cutoff"),
        other => panic!("expected RotationRefused, got {other:?}"),
    }
}

#[test]
fn rotate_matches_reference_on_copied_tree() {
    let (_tmp, root) = fresh_root();
    seed_standard(&root);
    setup_pending(&root, "speedrun", 5, 77, 12_000);
    let prod_root = root.join("prod");
    let ref_root = root.join("refer");
    copy_tree(&root, &prod_root);
    copy_tree(&root, &ref_root);
    let mut prod = EvalGate::open(cfg(&prod_root, HV)).unwrap();
    let mut refer = reference::RefGate::open(cfg(&ref_root, HV)).unwrap();
    let ph = prod.rotate("speedrun", 2, ROTATION_MS).unwrap();
    let rh = refer.rotate("speedrun", 2, ROTATION_MS).unwrap();
    assert_eq!(ph, rh);
}

fn copy_tree(src: &std::path::Path, dst: &std::path::Path) {
    for suite in prometheus_eval_gate::HELD_OUT_SUITES {
        let from = src.join(suite);
        if from.exists() {
            copy_dir(&from, &dst.join(suite));
        }
    }
    let subj = src.join("subjects");
    if subj.exists() {
        copy_dir(&subj, &dst.join("subjects"));
    }
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}
