//! Group: attempt-keyed output bytes, disk layout, hash vs reference, branches.

mod common;
mod reference;

use common::{
    assert_completed, fresh_queue_dir, item_n, output_path, short_config, worker,
};
use prometheus_leases::{Queue, TaskId};
use reference::{assert_self_consistent_goldens, output_hash, GOLDEN_EMPTY_OUTPUT_HASH, RefQueue};

#[test]
fn empty_output_hash_matches_reference_golden() {
    assert_self_consistent_goldens();
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    q.complete(&TaskId("t1".into()), &worker("w"), 1, b"", 1)
        .expect("complete");
    let got = q.get(&TaskId("t1".into())).expect("get");
    assert_eq!(got.output_hash.as_deref(), Some(GOLDEN_EMPTY_OUTPUT_HASH));
    assert_eq!(output_hash(b""), GOLDEN_EMPTY_OUTPUT_HASH);
    assert_eq!(
        q.get_output(&TaskId("t1".into()), 1)
            .expect("bytes")
            .as_deref(),
        Some(&b""[..])
    );
}

#[test]
fn get_output_roundtrip_and_disk_path() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    let bytes: Vec<u8> = (0..255).collect();
    q.complete(&TaskId("t1".into()), &worker("w"), 1, &bytes, 1)
        .expect("complete");
    let hash = output_hash(&bytes);
    assert_completed(q.get(&TaskId("t1".into())).expect("get"), "t1", 1, &hash);
    assert_eq!(
        q.get_output(&TaskId("t1".into()), 1)
            .expect("get_output")
            .as_deref(),
        Some(bytes.as_slice())
    );
    let path = output_path(&dir, &TaskId("t1".into()), 1);
    assert!(path.exists(), "expected {}", path.display());
    assert_eq!(std::fs::read(&path).expect("read disk"), bytes);
}

#[test]
fn get_output_missing_attempt_is_ok_none() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    assert!(
        q.get_output(&TaskId("t1".into()), 1)
            .expect("get_output")
            .is_none()
    );
}

#[test]
fn get_output_unknown_task_is_not_found() {
    let (_parent, dir) = fresh_queue_dir();
    let q = Queue::create(&dir, short_config()).expect("create");
    let err = q
        .get_output(&TaskId("nope".into()), 1)
        .expect_err("unknown");
    common::assert_not_found(&err, "nope");
}

#[test]
fn checkpoint_branch_matches_claimed_attempt() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    let lease = q.claim(&worker("w"), 0).expect("claim").expect("lease");
    assert_eq!(
        Queue::checkpoint_branch(&lease.task_id, lease.attempt),
        "task/t1/attempt/1"
    );
    assert_eq!(
        Queue::checkpoint_branch(&TaskId("abc".into()), 12),
        "task/abc/attempt/12"
    );
}

#[test]
fn output_hash_matches_in_memory_reference() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    let mut q = Queue::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefQueue::new(cfg);
    q.enqueue(item_n(1), 0).expect("enqueue");
    refer.enqueue(item_n(1), 0).expect("ref enqueue");
    q.claim(&worker("w"), 0).expect("claim");
    refer.claim(&worker("w"), 0).expect("ref claim");
    let out = b"xyz\x00\xff";
    q.complete(&TaskId("t1".into()), &worker("w"), 1, out, 1)
        .expect("complete");
    refer
        .complete(&TaskId("t1".into()), &worker("w"), 1, out, 1)
        .expect("ref complete");
    assert_eq!(
        q.get(&TaskId("t1".into())).expect("get").output_hash,
        refer.get(&TaskId("t1".into())).expect("ref").output_hash
    );
    assert_eq!(
        q.get_output(&TaskId("t1".into()), 1).expect("out"),
        refer.get_output(&TaskId("t1".into()), 1).expect("ref out")
    );
}
