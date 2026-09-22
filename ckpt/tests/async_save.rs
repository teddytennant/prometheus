//! Async persistent save (spec 5.5): the train step must not stall on the write.
//!
//! CPU contract is the docstrings on `Checkpointer::save_persistent_async` and
//! `PendingSave`. A synchronous `put` must not be able to hang this file: every
//! wait is bounded by [`TIMEOUT`], and the gate is released on unwind.

mod common;
mod reference;

use common::{
    assert_checkpoint_roundtrip, assert_hash_mismatch, assert_missing_shard,
    checkpoint_from_shards, tiny_checkpoint, tiny_checkpoint_with_id, PanicStore,
};
use prometheus_ckpt::{
    Checkpoint, Checkpointer, CkptError, Dtype, MemoryStore, OptimizerKind, Store,
};
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, ScopedJoinHandle};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(2);

struct GateInner {
    /// `put` calls that have entered, incremented before the condvar wait.
    entered: usize,
    keys: Vec<String>,
    release: bool,
    map: HashMap<String, Vec<u8>>,
}

struct Gate {
    mu: Mutex<GateInner>,
    cv: Condvar,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            mu: Mutex::new(GateInner {
                entered: 0,
                keys: Vec::new(),
                release: false,
                map: HashMap::new(),
            }),
            cv: Condvar::new(),
        })
    }

    fn entered_count(&self) -> usize {
        self.mu.lock().expect("gate").entered
    }

    fn keys(&self) -> Vec<String> {
        self.mu.lock().expect("gate").keys.clone()
    }

    fn is_released(&self) -> bool {
        self.mu.lock().expect("gate").release
    }

    fn release(&self) {
        let mut guard = self.mu.lock().expect("gate");
        guard.release = true;
        self.cv.notify_all();
    }

    fn contains(&self, key: &str) -> bool {
        self.mu.lock().expect("gate").map.contains_key(key)
    }

    fn snapshot(&self) -> HashMap<String, Vec<u8>> {
        self.mu.lock().expect("gate").map.clone()
    }

    fn wait_for_enter(&self, timeout: Duration) {
        let guard = self.mu.lock().expect("gate");
        if guard.entered > 0 {
            return;
        }
        let (guard, _) = self.cv.wait_timeout(guard, timeout).expect("gate wait");
        drop(guard);
    }
}

/// `put` signals entry, then blocks until [`Gate::release`]. Bytes land in the
/// map only after the wait, so a blocked put is not yet durable.
struct GateStore {
    gate: Arc<Gate>,
}

impl GateStore {
    fn new(gate: Arc<Gate>) -> Self {
        Self { gate }
    }
}

impl Store for GateStore {
    fn put(&mut self, key: &str, bytes: &[u8]) -> prometheus_ckpt::Result<()> {
        let mut guard = self.gate.mu.lock().expect("gate");
        guard.entered += 1;
        guard.keys.push(key.to_string());
        self.gate.cv.notify_all();
        while !guard.release {
            guard = self.gate.cv.wait(guard).expect("gate wait");
        }
        guard.map.insert(key.to_string(), bytes.to_vec());
        self.gate.cv.notify_all();
        Ok(())
    }

    fn get(&self, key: &str) -> prometheus_ckpt::Result<Vec<u8>> {
        self.gate
            .mu
            .lock()
            .expect("gate")
            .map
            .get(key)
            .cloned()
            .ok_or_else(|| CkptError::NotFound(key.to_string()))
    }

    fn contains(&self, key: &str) -> prometheus_ckpt::Result<bool> {
        Ok(self.contains_key(key))
    }
}

impl GateStore {
    fn contains_key(&self, key: &str) -> bool {
        self.gate.contains(key)
    }
}

/// Releases the gate and sets `cancel` so a helper blocked in `put` or in a
/// park loop can exit when the test unwinds. Drop order inside `thread::scope`
/// runs this before the scope joins.
struct Unblock<'a> {
    gate: &'a Gate,
    cancel: &'a AtomicBool,
}

impl Drop for Unblock<'_> {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.gate.release();
    }
}

enum Wait {
    Ready,
    HelperDead,
    TimedOut,
}

fn three_flags(gate: &Gate, returned: &AtomicBool) -> (bool, bool, bool) {
    let entered = gate.entered_count() > 0;
    let call_returned = returned.load(Ordering::SeqCst);
    let release_still_false = !gate.is_released();
    (entered, call_returned, release_still_false)
}

fn assert_returned_while_put_blocked(gate: &Gate, returned: &AtomicBool) {
    let (entered, call_returned, release_still_false) = three_flags(gate, returned);
    assert!(
        entered && call_returned && release_still_false,
        "put entered={entered}, call returned={call_returned}, release still false={release_still_false}"
    );
}

fn wait_ready(gate: &Gate, returned: &AtomicBool, helper_finished: impl Fn() -> bool) -> Wait {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let (entered, call_returned, _) = three_flags(gate, returned);
        if entered && call_returned {
            return Wait::Ready;
        }
        if helper_finished() && !call_returned {
            return Wait::HelperDead;
        }
        if Instant::now() >= deadline {
            return Wait::TimedOut;
        }
        let slice = deadline.saturating_duration_since(Instant::now());
        gate.wait_for_enter(slice.min(Duration::from_millis(20)));
    }
}

fn fail_timeout(gate: &Gate, returned: &AtomicBool) -> ! {
    let (entered, call_returned, release_still_false) = three_flags(gate, returned);
    panic!(
        "timed out after 2s: put entered={entered}, call returned={call_returned}, release still false={release_still_false}"
    );
}

fn propagate_helper_panic<T>(handle: ScopedJoinHandle<'_, T>) -> T {
    match handle.join() {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn park_until(flag: &AtomicBool, cancel: &AtomicBool) -> bool {
    let deadline = Instant::now() + TIMEOUT;
    while !flag.load(Ordering::SeqCst) {
        if cancel.load(Ordering::SeqCst) || Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(5));
    }
    !cancel.load(Ordering::SeqCst)
}

fn assert_matches_sync_save(got: &Checkpoint, src: &Checkpoint) {
    let mut sync = Checkpointer::new(Box::new(MemoryStore::new()));
    let id = sync.save_persistent(src).expect("sync save_persistent");
    let expect = sync.restore(&id).expect("sync restore");
    assert_eq!(got.manifest.checkpoint_id, expect.manifest.checkpoint_id);
    assert_eq!(
        got.weights.len(),
        expect.weights.len(),
        "weight shard count"
    );
    assert_eq!(
        got.optimizer.len(),
        expect.optimizer.len(),
        "optimizer shard count"
    );
    for (got_blob, expect_blob) in got.weights.iter().zip(expect.weights.iter()) {
        assert_eq!(got_blob.name, expect_blob.name);
        assert_eq!(got_blob.shard_rank, expect_blob.shard_rank);
        assert_eq!(
            got_blob.bytes, expect_blob.bytes,
            "bitwise weight shard bytes for {}/{}",
            got_blob.name, got_blob.shard_rank
        );
    }
    for (got_blob, expect_blob) in got.optimizer.iter().zip(expect.optimizer.iter()) {
        assert_eq!(got_blob.name, expect_blob.name);
        assert_eq!(got_blob.shard_rank, expect_blob.shard_rank);
        assert_eq!(
            got_blob.bytes, expect_blob.bytes,
            "bitwise optimizer shard bytes for {}/{}",
            got_blob.name, got_blob.shard_rank
        );
    }
    for (got_meta, expect_meta) in got
        .manifest
        .weights
        .iter()
        .zip(expect.manifest.weights.iter())
    {
        assert_eq!(got_meta.content_hash, expect_meta.content_hash);
    }
    for (got_meta, expect_meta) in got
        .manifest
        .optimizer
        .iter()
        .zip(expect.manifest.optimizer.iter())
    {
        assert_eq!(got_meta.content_hash, expect_meta.content_hash);
    }
    assert_checkpoint_roundtrip(got, src);
    assert_checkpoint_roundtrip(&expect, src);
}

fn shard_bytes_landed(gate: &Gate, src: &Checkpoint) -> bool {
    if !gate.contains(&src.manifest.checkpoint_id) {
        return false;
    }
    let map = gate.snapshot();
    src.weights
        .iter()
        .chain(src.optimizer.iter())
        .all(|blob| map.values().any(|stored| stored == &blob.bytes))
}

fn poll_manifest_and_shards(gate: &Gate, src: &Checkpoint) -> bool {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        if shard_bytes_landed(gate, src) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Shared prefix: call `save_persistent_async` on a helper so a synchronous
/// stub cannot hang the suite, then require put-entered AND call-returned AND
/// release-still-false. `body` runs on the helper after the call returns and
/// must park on `cancel` (and any extra flag) so the scope can join.
fn assert_async_return_then<F>(src: Checkpoint, body: F)
where
    F: FnOnce(&mut Checkpointer, prometheus_ckpt::PendingSave, &AtomicBool, &Gate) + Send,
{
    let gate = Gate::new();
    let cancel = AtomicBool::new(false);
    let returned = AtomicBool::new(false);
    thread::scope(|scope| {
        let unblock = Unblock {
            gate: gate.as_ref(),
            cancel: &cancel,
        };
        let handle = scope.spawn(|| {
            let mut ckpt = Checkpointer::new(Box::new(GateStore::new(Arc::clone(&gate))));
            let pending = match ckpt.save_persistent_async(&src) {
                Ok(pending) => pending,
                Err(err) => panic!("save_persistent_async failed before a blocked put: {err}"),
            };
            returned.store(true, Ordering::SeqCst);
            body(&mut ckpt, pending, &cancel, gate.as_ref());
        });
        match wait_ready(&gate, &returned, || handle.is_finished()) {
            Wait::HelperDead => {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper finished without save_persistent_async returning");
            }
            Wait::TimedOut => fail_timeout(&gate, &returned),
            Wait::Ready => assert_returned_while_put_blocked(&gate, &returned),
        }
        drop(unblock);
        propagate_helper_panic(handle);
    });
}

#[test]
fn save_persistent_async_returns_while_first_put_blocked() {
    let src = tiny_checkpoint_with_id("async-returns-before-put");
    assert_async_return_then(src, |_ckpt, pending, cancel, _gate| {
        // Hold the handle so Drop cannot hide a synchronous return. The call
        // has already returned; release is still false until the test asserts.
        let _held = pending;
        let _ = park_until(&AtomicBool::new(false), cancel);
    });
}

fn multi_shard(id: &str) -> Checkpoint {
    let mut src = checkpoint_from_shards(
        id,
        vec![
            (
                "embed",
                0,
                vec![2],
                Dtype::Fp32,
                format!("w0-{id}").into_bytes(),
                vec!["fsdp".into()],
            ),
            (
                "lm_head",
                1,
                vec![4, 2],
                Dtype::Bf16,
                format!("w1-{id}").into_bytes(),
                vec!["tp".into()],
            ),
        ],
        vec![
            (
                "embed.m",
                0,
                OptimizerKind::Muon,
                format!("o0-{id}").into_bytes(),
            ),
            (
                "lm_head.m",
                1,
                OptimizerKind::Adamw,
                format!("o1-{id}").into_bytes(),
            ),
        ],
    );
    src.manifest.parent_checkpoint_id = Some("parent-async".into());
    src.manifest.step = 42;
    src
}

#[test]
fn wait_then_restore_matches_save_persistent_bitwise() {
    let src = multi_shard("async-restore-bitwise");
    let gate = Gate::new();
    let cancel = AtomicBool::new(false);
    let returned = AtomicBool::new(false);
    let proceed = AtomicBool::new(false);
    let compared = AtomicBool::new(false);

    thread::scope(|scope| {
        let unblock = Unblock {
            gate: gate.as_ref(),
            cancel: &cancel,
        };
        let handle = scope.spawn(|| {
            let mut ckpt = Checkpointer::new(Box::new(GateStore::new(Arc::clone(&gate))));
            let pending = match ckpt.save_persistent_async(&src) {
                Ok(pending) => pending,
                Err(err) => panic!("save_persistent_async failed: {err}"),
            };
            assert_eq!(pending.checkpoint_id(), src.manifest.checkpoint_id);
            returned.store(true, Ordering::SeqCst);
            if !park_until(&proceed, &cancel) {
                return;
            }
            pending.wait().expect("PendingSave::wait");
            let got = ckpt
                .restore(&src.manifest.checkpoint_id)
                .expect("restore after wait");
            assert_matches_sync_save(&got, &src);
            compared.store(true, Ordering::SeqCst);
        });

        match wait_ready(&gate, &returned, || handle.is_finished()) {
            Wait::HelperDead => {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper finished without save_persistent_async returning");
            }
            Wait::TimedOut => fail_timeout(&gate, &returned),
            Wait::Ready => assert_returned_while_put_blocked(&gate, &returned),
        }

        // Unblock the object-store write only after the call has returned.
        gate.release();
        proceed.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + TIMEOUT;
        while !compared.load(Ordering::SeqCst) {
            if handle.is_finished() {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper exited before restore comparison");
            }
            if Instant::now() >= deadline {
                panic!("wait/restore did not finish within 2s");
            }
            thread::sleep(Duration::from_millis(5));
        }
        drop(unblock);
        propagate_helper_panic(handle);
    });
}

#[test]
fn second_save_before_wait_is_save_in_flight_and_starts_no_put() {
    let first = tiny_checkpoint_with_id("async-inflight-first");
    let second = tiny_checkpoint_with_id("async-inflight-second");
    let third = tiny_checkpoint_with_id("async-inflight-third");
    let gate = Gate::new();
    let cancel = AtomicBool::new(false);
    let returned = AtomicBool::new(false);
    let try_second = AtomicBool::new(false);
    let do_finish = AtomicBool::new(false);
    let second_done = AtomicBool::new(false);
    let third_ok = AtomicBool::new(false);
    let second_was_ok = AtomicBool::new(false);
    let second_err: Mutex<Option<CkptError>> = Mutex::new(None);
    let entered_at_second: Mutex<Option<(usize, usize)>> = Mutex::new(None);

    thread::scope(|scope| {
        let unblock = Unblock {
            gate: gate.as_ref(),
            cancel: &cancel,
        };
        let handle = scope.spawn(|| {
            let mut ckpt = Checkpointer::new(Box::new(GateStore::new(Arc::clone(&gate))));
            let pending = match ckpt.save_persistent_async(&first) {
                Ok(pending) => pending,
                Err(err) => panic!("first save_persistent_async failed: {err}"),
            };
            returned.store(true, Ordering::SeqCst);
            if !park_until(&try_second, &cancel) {
                return;
            }
            let before = gate.entered_count();
            match ckpt.save_persistent_async(&second) {
                Ok(extra) => {
                    drop(extra);
                    second_was_ok.store(true, Ordering::SeqCst);
                }
                Err(err) => {
                    *second_err.lock().expect("err slot") = Some(err);
                }
            }
            let after = gate.entered_count();
            *entered_at_second.lock().expect("count slot") = Some((before, after));
            second_done.store(true, Ordering::SeqCst);
            if !park_until(&do_finish, &cancel) {
                drop(pending);
                return;
            }
            pending.wait().expect("wait after in-flight check");
            match ckpt.save_persistent_async(&third) {
                Ok(next) => {
                    next.wait().expect("third wait");
                    let got = ckpt
                        .restore(&third.manifest.checkpoint_id)
                        .expect("restore third");
                    assert_matches_sync_save(&got, &third);
                    third_ok.store(true, Ordering::SeqCst);
                }
                Err(err) => panic!("save after wait must be allowed, got {err}"),
            }
        });

        match wait_ready(&gate, &returned, || handle.is_finished()) {
            Wait::HelperDead => {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper finished without save_persistent_async returning");
            }
            Wait::TimedOut => fail_timeout(&gate, &returned),
            Wait::Ready => assert_returned_while_put_blocked(&gate, &returned),
        }

        let stable = wait_entered_stable(&gate, Duration::from_millis(100), TIMEOUT);
        let stable = stable.unwrap_or_else(|| {
            panic!(
                "in-flight puts did not settle within 2s (entered={})",
                gate.entered_count()
            )
        });
        let keys_before = gate.keys();
        try_second.store(true, Ordering::SeqCst);

        let deadline = Instant::now() + TIMEOUT;
        while !second_done.load(Ordering::SeqCst) {
            if handle.is_finished() {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper exited before the second save returned");
            }
            if Instant::now() >= deadline {
                panic!("second save_persistent_async did not return within 2s");
            }
            thread::sleep(Duration::from_millis(5));
        }

        assert!(
            !second_was_ok.load(Ordering::SeqCst),
            "second save_persistent_async returned Ok while the first save was in flight"
        );
        let err = second_err.lock().expect("err slot").take();
        match err {
            Some(CkptError::SaveInFlight) => {}
            Some(other) => panic!("expected CkptError::SaveInFlight, got {other}"),
            None => panic!("second save produced no error"),
        }
        let (before, after) = entered_at_second
            .lock()
            .expect("count slot")
            .take()
            .expect("entered counts");
        assert_eq!(before, stable, "entered count moved before the second save");
        assert_eq!(
            before, after,
            "second save_persistent_async started an extra put"
        );
        assert_eq!(
            gate.keys(),
            keys_before,
            "second save started a put under a new key"
        );
        let watch_until = Instant::now() + Duration::from_millis(150);
        while Instant::now() < watch_until {
            assert_eq!(
                gate.entered_count(),
                before,
                "extra put entered after the second save returned"
            );
            assert!(
                !gate.keys().iter().any(|key| key == "async-inflight-second"),
                "second checkpoint id was written while the first save was in flight"
            );
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !gate.is_released(),
            "release must still be false while checking SaveInFlight"
        );

        gate.release();
        do_finish.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + TIMEOUT;
        while !third_ok.load(Ordering::SeqCst) {
            if handle.is_finished() {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper exited before the post-wait save finished");
            }
            if Instant::now() >= deadline {
                panic!("save after wait did not finish within 2s");
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            gate.contains("async-inflight-first"),
            "first manifest missing after wait"
        );
        assert!(
            gate.contains("async-inflight-third"),
            "third manifest missing after the allowed save"
        );
        assert!(
            !gate.contains("async-inflight-second"),
            "SaveInFlight save must not write a manifest"
        );
        drop(unblock);
        propagate_helper_panic(handle);
    });
}

fn wait_entered_stable(gate: &Gate, stable_for: Duration, timeout: Duration) -> Option<usize> {
    let deadline = Instant::now() + timeout;
    let mut last = gate.entered_count();
    let mut since = Instant::now();
    while Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
        let count = gate.entered_count();
        if count != last {
            last = count;
            since = Instant::now();
            continue;
        }
        if last > 0 && since.elapsed() >= stable_for {
            return Some(last);
        }
    }
    None
}

#[test]
fn drop_pending_save_does_not_cancel_write() {
    let src = tiny_checkpoint_with_id("async-drop-no-cancel");
    let gate = Gate::new();
    let cancel = AtomicBool::new(false);
    let returned = AtomicBool::new(false);

    thread::scope(|scope| {
        let unblock = Unblock {
            gate: gate.as_ref(),
            cancel: &cancel,
        };
        let handle = scope.spawn(|| {
            let mut ckpt = Checkpointer::new(Box::new(GateStore::new(Arc::clone(&gate))));
            let pending = match ckpt.save_persistent_async(&src) {
                Ok(pending) => pending,
                Err(err) => panic!("save_persistent_async failed: {err}"),
            };
            // Do not call wait. Drop must not cancel, and must not block on put.
            drop(pending);
            returned.store(true, Ordering::SeqCst);
            // Keep the checkpointer alive until the test has polled the store.
            let _ = park_until(&AtomicBool::new(false), &cancel);
            drop(ckpt);
        });

        match wait_ready(&gate, &returned, || handle.is_finished()) {
            Wait::HelperDead => {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper finished without dropping PendingSave");
            }
            Wait::TimedOut => fail_timeout(&gate, &returned),
            Wait::Ready => assert_returned_while_put_blocked(&gate, &returned),
        }

        // Release only after the three-part assertion. Poll the store; do not
        // call PendingSave::wait.
        gate.release();
        assert!(
            poll_manifest_and_shards(&gate, &src),
            "dropping PendingSave cancelled the write: manifest id {} not present within 2s",
            src.manifest.checkpoint_id
        );
        drop(unblock);
        propagate_helper_panic(handle);
    });
}

#[test]
fn preparation_failure_returns_same_error_and_starts_no_write() {
    let mut missing = tiny_checkpoint();
    let missing_name = missing.manifest.weights[0].name.clone();
    let missing_rank = missing.manifest.weights[0].shard_rank;
    missing.weights.clear();

    let mut hashed = tiny_checkpoint_with_id("async-bad-hash");
    let wrong_hash = "0".repeat(64);
    hashed.manifest.weights[0].content_hash = wrong_hash.clone();

    assert_prep_rejected(&missing, |err| {
        assert_missing_shard(err, &missing_name, missing_rank)
    });
    assert_prep_rejected(&hashed, |err| assert_hash_mismatch(err, Some(&wrong_hash)));
}

fn assert_prep_rejected(bad: &Checkpoint, check_sync: impl Fn(&CkptError)) {
    let sync_err = {
        let mut sync = Checkpointer::new(Box::new(MemoryStore::new()));
        sync.save_persistent(bad)
            .expect_err("save_persistent must already reject this checkpoint")
    };
    check_sync(&sync_err);

    let bad = bad.clone();
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let caught = catch_unwind(AssertUnwindSafe(|| {
            let mut ckpt = Checkpointer::new(Box::new(PanicStore));
            match ckpt.save_persistent_async(&bad) {
                Ok(pending) => Err(format!(
                    "expected preparation error, got Ok({})",
                    pending.checkpoint_id()
                )),
                Err(err) => Ok(err),
            }
        }));
        let _ = tx.send(caught);
    });
    let caught = match rx.recv_timeout(TIMEOUT) {
        Ok(caught) => caught,
        Err(_) => {
            panic!("save_persistent_async did not return within 2s on a rejected checkpoint")
        }
    };
    match caught {
        Ok(Ok(err)) => assert_eq!(err, sync_err),
        Ok(Err(msg)) => panic!("{msg}"),
        Err(payload) => std::panic::resume_unwind(payload),
    }
    let _ = handle.join();
}

#[test]
fn is_finished_false_while_blocked_true_after_wait() {
    let src = tiny_checkpoint_with_id("async-is-finished");
    let gate = Gate::new();
    let cancel = AtomicBool::new(false);
    let returned = AtomicBool::new(false);
    let proceed = AtomicBool::new(false);
    let saw_false = AtomicBool::new(false);
    let saw_true = AtomicBool::new(false);
    let wait_ok = AtomicBool::new(false);

    thread::scope(|scope| {
        let unblock = Unblock {
            gate: gate.as_ref(),
            cancel: &cancel,
        };
        let handle = scope.spawn(|| {
            let mut ckpt = Checkpointer::new(Box::new(GateStore::new(Arc::clone(&gate))));
            let pending = match ckpt.save_persistent_async(&src) {
                Ok(pending) => pending,
                Err(err) => panic!("save_persistent_async failed: {err}"),
            };
            // Observable while the first put is still blocked.
            if pending.is_finished() {
                panic!("is_finished was true before puts completed");
            }
            saw_false.store(true, Ordering::SeqCst);
            returned.store(true, Ordering::SeqCst);
            if !park_until(&proceed, &cancel) {
                return;
            }
            // `wait` consumes `self`, so durability is observed before the
            // consuming call. The docstring says `is_finished` is true once
            // every shard and the manifest are in the store, without waiting
            // for `wait` itself. `wait` must then return Ok.
            let deadline = Instant::now() + TIMEOUT;
            while !pending.is_finished() {
                if cancel.load(Ordering::SeqCst) || Instant::now() >= deadline {
                    panic!("is_finished stayed false for 2s after the blocked put was released");
                }
                thread::sleep(Duration::from_millis(5));
            }
            assert!(
                pending.is_finished(),
                "is_finished must be true once puts have completed"
            );
            saw_true.store(true, Ordering::SeqCst);
            pending.wait().expect("wait");
            wait_ok.store(true, Ordering::SeqCst);
            let got = ckpt
                .restore(&src.manifest.checkpoint_id)
                .expect("restore after is_finished");
            assert_matches_sync_save(&got, &src);
            drop(ckpt);
        });

        match wait_ready(&gate, &returned, || handle.is_finished()) {
            Wait::HelperDead => {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper finished without save_persistent_async returning");
            }
            Wait::TimedOut => fail_timeout(&gate, &returned),
            Wait::Ready => assert_returned_while_put_blocked(&gate, &returned),
        }
        assert!(
            saw_false.load(Ordering::SeqCst),
            "is_finished was not observed false while put was blocked"
        );
        assert!(!gate.is_released(), "release must still be false");

        gate.release();
        proceed.store(true, Ordering::SeqCst);
        let deadline = Instant::now() + TIMEOUT;
        while !wait_ok.load(Ordering::SeqCst) {
            if handle.is_finished() {
                drop(unblock);
                propagate_helper_panic(handle);
                panic!("helper exited before wait returned");
            }
            if Instant::now() >= deadline {
                panic!(
                    "is_finished/wait did not finish within 2s (saw_true={}, wait_ok={})",
                    saw_true.load(Ordering::SeqCst),
                    wait_ok.load(Ordering::SeqCst)
                );
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(saw_true.load(Ordering::SeqCst));
        assert!(wait_ok.load(Ordering::SeqCst));
        drop(unblock);
        propagate_helper_panic(handle);
    });
}
