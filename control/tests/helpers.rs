//! Group: helpers that do not call `Controller` (may pass against the stub).
//!
//! Constants, Display, type equality, and the independent accum formula.

mod reference;

use prometheus_control::{ControlError, ReplicaId, ReplicaState, PAGE_WINDOW_STEPS};
use reference::min_grad_accumulation;

#[test]
fn page_window_steps_is_10_000() {
    assert_eq!(PAGE_WINDOW_STEPS, 10_000);
}

#[test]
fn min_grad_accumulation_smallest_u64_holding_tokens() {
    // 4 live * 128 micro * 2 = 1024.
    assert_eq!(min_grad_accumulation(4, 128, 1024), 2);
    // 3 live * 128 * 3 = 1152 >= 1024.
    assert_eq!(min_grad_accumulation(3, 128, 1024), 3);
    // 2 live * 128 * 4 = 1024.
    assert_eq!(min_grad_accumulation(2, 128, 1024), 4);
    // 1 live * 128 * 8 = 1024.
    assert_eq!(min_grad_accumulation(1, 128, 1024), 8);
    // Uneven: 3 * 128 = 384, ceil(1000/384) = 3, 1152 >= 1000.
    assert_eq!(min_grad_accumulation(3, 128, 1000), 3);
    // microbatch already covers the target: accum = 1.
    assert_eq!(min_grad_accumulation(1, 1000, 100), 1);
    // tokens_per_step 0 -> accum 0.
    assert_eq!(min_grad_accumulation(4, 128, 0), 0);
}

#[test]
fn no_live_replicas_display() {
    let err = ControlError::NoLiveReplicas;
    assert_eq!(err.to_string(), "no live replicas remain");
}

#[test]
fn replica_id_eq_and_replica_state_copy() {
    assert_eq!(ReplicaId("a".into()), ReplicaId("a".into()));
    assert_ne!(ReplicaId("a".into()), ReplicaId("b".into()));
    let s = ReplicaState::Spare;
    let c = s;
    assert_eq!(s, c);
    assert_eq!(s, ReplicaState::Spare);
    assert_ne!(s, ReplicaState::Live);
}
