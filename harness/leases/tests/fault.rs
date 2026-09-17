//! Group: fail-closed open on a corrupt log; create collisions; refused writes.

mod common;
mod reference;

use common::{
    assert_err, events_jsonl, fresh_queue_dir, item_n, short_config, worker,
};
use prometheus_leases::{Queue, TaskId};

#[test]
fn open_fails_on_garbage_events_jsonl() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), 0).expect("enqueue");
    }
    let path = events_jsonl(&dir);
    assert!(path.exists(), "expected {}", path.display());
    std::fs::write(&path, b"{this is not jsonl\n").expect("corrupt");
    assert_err(Queue::open(&dir, cfg), "open corrupt log");
}

#[test]
fn open_fails_on_truncated_jsonl_line() {
    let (_parent, dir) = fresh_queue_dir();
    let cfg = short_config();
    {
        let mut q = Queue::create(&dir, cfg.clone()).expect("create");
        q.enqueue(item_n(1), 0).expect("enqueue");
        q.claim(&worker("w"), 0).expect("claim");
    }
    let path = events_jsonl(&dir);
    let mut data = std::fs::read(&path).expect("read jsonl");
    assert!(!data.is_empty(), "log should have bytes after enqueue");
    // H2 recovers a torn last line (no newline). A mutated complete line
    // must fail-closed. Flip a payload byte that is not a newline.
    let mut idx = data.len() / 2;
    if data[idx] == b'\n' {
        idx = idx.saturating_sub(1);
    }
    data[idx] ^= 0x7f;
    std::fs::write(&path, &data).expect("corrupt complete line");
    assert_err(Queue::open(&dir, cfg), "open truncated log");
}

#[test]
fn create_fails_if_path_is_a_file() {
    let parent = tempfile::tempdir().expect("tempdir");
    let dir = parent.path().join("queue");
    std::fs::write(&dir, b"not a dir").expect("file");
    assert_err(Queue::create(&dir, short_config()), "create on file");
}

#[test]
fn heartbeat_unknown_task_is_not_found() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let err = q
        .heartbeat(&TaskId("ghost".into()), &worker("w"), 1, 0)
        .expect_err("not found");
    common::assert_not_found(&err, "ghost");
}

#[test]
fn fail_unknown_task_is_not_found() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    let err = q
        .fail(&TaskId("ghost".into()), &worker("w"), 1, "x", 0)
        .expect_err("not found");
    common::assert_not_found(&err, "ghost");
}

#[test]
fn refused_complete_does_not_create_output_file() {
    let (_parent, dir) = fresh_queue_dir();
    let mut q = Queue::create(&dir, short_config()).expect("create");
    q.enqueue(item_n(1), 0).expect("enqueue");
    q.claim(&worker("owner"), 0).expect("claim");
    let _ = q
        .complete(&TaskId("t1".into()), &worker("thief"), 1, b"nope", 1)
        .expect_err("refused");
    let path = common::output_path(&dir, &TaskId("t1".into()), 1);
    assert!(
        !path.exists(),
        "refused complete must not create {}",
        path.display()
    );
}
