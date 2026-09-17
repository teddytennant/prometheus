//! Group: fault injection — malformed transport, truncated tools, all-down
//! never-exit, follower NotBroker.

mod common;
mod reference;

use common::{
    assert_invalid, assert_no_provider, assert_not_broker, broker, failover, follower_index,
    group3, pid, tick_until_leader, FAKE_BLOB,
};
use prometheus_providers::{validate_response, TransportFault};
use reference::validate_response as ref_validate;
use serde_json::json;
use std::time::Duration;

#[test]
fn rate_limit_text_is_not_malformed_json() {
    let err = validate_response(b"rate_limit").unwrap_err();
    assert_invalid(&err, TransportFault::RateLimitText);
    match ref_validate(b"rate_limit") {
        Err(prometheus_providers::Error::InvalidResponse(kind)) => {
            assert_eq!(kind, TransportFault::RateLimitText)
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn truncated_arguments_empty_string() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "tool_calls": [{
                    "function": { "name": "f", "arguments": "" }
                }]
            }
        }]
    }))
    .unwrap();
    let err = validate_response(&body).unwrap_err();
    assert_invalid(&err, TransportFault::TruncatedToolCall);
    match (validate_response(&body), ref_validate(&body)) {
        (
            Err(prometheus_providers::Error::InvalidResponse(a)),
            Err(prometheus_providers::Error::InvalidResponse(b)),
        ) => assert_eq!(a, b),
        other => panic!("{other:?}"),
    }
}

#[test]
fn truncated_null_arguments() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "tool_calls": [{
                "function": { "name": "f", "arguments": null }
            }]
        }]
    }))
    .unwrap();
    let err = validate_response(&body).unwrap_err();
    assert_invalid(&err, TransportFault::TruncatedToolCall);
}

#[test]
fn truncated_non_array_tool_calls() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "message": { "tool_calls": { "function": { "name": "f" } } }
        }]
    }))
    .unwrap();
    let err = validate_response(&body).unwrap_err();
    assert_invalid(&err, TransportFault::TruncatedToolCall);
}

#[test]
fn trailing_junk_json_is_malformed() {
    let err = validate_response(br#"{"choices":[{"finish_reason":"stop"}]} extra"#).unwrap_err();
    assert_invalid(&err, TransportFault::MalformedJson);
}

#[test]
fn all_down_current_fault_then_backoff_loop_never_exits() {
    let mut fo = failover(&["primary", "secondary"]);
    fo.mark_down(&pid("primary")).unwrap();
    fo.mark_down(&pid("secondary")).unwrap();
    for _ in 0..3 {
        assert_no_provider(&fo.current().unwrap_err());
        let d = fo.wait_backoff();
        assert_eq!(d, Duration::from_millis(400));
        assert!(d > Duration::from_millis(0));
    }
}

#[test]
fn follower_refresh_is_not_broker_even_after_leader_exists() {
    let (_parent, mut group) = group3();
    let now = 50_000u64;
    let leader = tick_until_leader(&mut group, now);
    let follower = follower_index(&mut group, leader);
    let mut br = broker("xai", 100);
    let err = br
        .refresh(group.get(follower).unwrap(), FAKE_BLOB, now)
        .unwrap_err();
    assert_not_broker(&err);
}

#[test]
fn arguments_object_not_string_is_not_truncated() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "tool_calls": [{
                    "function": { "name": "f", "arguments": { "q": 1 } }
                }]
            }
        }]
    }))
    .unwrap();
    let got = validate_response(&body).expect("object arguments are closed");
    let refer = ref_validate(&body).expect("ref");
    assert_eq!(got, refer);
}
