//! Group: `validate_response` faults and success vs the independent reference.

mod common;
mod reference;

use common::assert_invalid;
use prometheus_providers::{validate_response, Error, TransportFault};
use reference::validate_response as ref_validate;
use serde_json::{json, Value};

fn assert_same(body: &[u8]) {
    let prod = validate_response(body);
    let refer = ref_validate(body);
    match (prod, refer) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "parsed Value vs reference"),
        (Err(Error::InvalidResponse(a)), Err(Error::InvalidResponse(b))) => {
            assert_eq!(a, b, "fault kind vs reference for body {body:?}")
        }
        (Ok(v), Err(e)) => panic!("production Ok({v:?}) reference Err({e:?})"),
        (Err(e), Ok(v)) => panic!("production Err({e:?}) reference Ok({v:?})"),
        (Err(a), Err(b)) => panic!("error variant mismatch {a:?} vs {b:?}"),
    }
}

fn err_kind(body: &[u8], kind: TransportFault) {
    let err = validate_response(body).expect_err("expected fault");
    assert_invalid(&err, kind);
    assert_same(body);
}

fn ok_value(body: &[u8]) -> Value {
    let got = validate_response(body).expect("expected success");
    let refer = ref_validate(body).expect("reference success");
    assert_eq!(got, refer);
    got
}

#[test]
fn exact_ascii_rate_limit_is_rate_limit_text() {
    err_kind(b"rate_limit", TransportFault::RateLimitText);
}

#[test]
fn rate_limit_any_ascii_case() {
    for body in [b"RATE_LIMIT".as_slice(), b"Rate_Limit", b"rAtE_lImIt"] {
        err_kind(body, TransportFault::RateLimitText);
    }
}

#[test]
fn json_error_rate_limit_object() {
    err_kind(br#"{"error":"rate_limit"}"#, TransportFault::RateLimitText);
}

#[test]
fn json_error_rate_limit_case_insensitive_and_extra_fields() {
    err_kind(
        br#"{"error":"RATE_LIMIT","choices":[]}"#,
        TransportFault::RateLimitText,
    );
}

#[test]
fn json_rate_limit_beats_empty_choices() {
    err_kind(
        br#"{"error":"rate_limit","choices":[]}"#,
        TransportFault::RateLimitText,
    );
}

#[test]
fn missing_choices_is_empty_completion() {
    err_kind(br#"{"id":"x"}"#, TransportFault::EmptyCompletion);
}

#[test]
fn empty_choices_array_is_empty_completion() {
    err_kind(br#"{"choices":[]}"#, TransportFault::EmptyCompletion);
}

#[test]
fn null_or_non_array_choices_is_empty_completion() {
    err_kind(br#"{"choices":null}"#, TransportFault::EmptyCompletion);
    err_kind(br#"{"choices":{}}"#, TransportFault::EmptyCompletion);
}

#[test]
fn not_json_is_malformed() {
    err_kind(b"not-json", TransportFault::MalformedJson);
    err_kind(b"", TransportFault::MalformedJson);
    err_kind(b" rate_limit ", TransportFault::MalformedJson);
}

#[test]
fn json_non_object_is_malformed() {
    err_kind(b"[]", TransportFault::MalformedJson);
    err_kind(b"\"rate_limit\"", TransportFault::MalformedJson);
    err_kind(b"true", TransportFault::MalformedJson);
    err_kind(b"1", TransportFault::MalformedJson);
}

#[test]
fn invalid_utf8_is_malformed() {
    err_kind(&[0xff, 0xfe, 0xfd], TransportFault::MalformedJson);
}

#[test]
fn truncated_tool_call_missing_arguments() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": { "name": "search" }
                }]
            }
        }]
    }))
    .unwrap();
    err_kind(&body, TransportFault::TruncatedToolCall);
}

#[test]
fn truncated_tool_call_unclosed_arguments_string() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "search",
                        "arguments": "{\"q\": "
                    }
                }]
            }
        }]
    }))
    .unwrap();
    err_kind(&body, TransportFault::TruncatedToolCall);
}

#[test]
fn truncated_tool_call_on_choice_not_message() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "tool_calls": [{ "id": "c1" }]
        }]
    }))
    .unwrap();
    err_kind(&body, TransportFault::TruncatedToolCall);
}

#[test]
fn finish_reason_length_with_closed_arguments_is_success() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "search",
                        "arguments": "{\"q\": 1}"
                    }
                }]
            }
        }]
    }))
    .unwrap();
    let v = ok_value(&body);
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

#[test]
fn finish_reason_stop_with_unclosed_arguments_is_not_truncated() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "tool_calls": [{
                    "function": { "name": "search", "arguments": "{\"q\": " }
                }]
            }
        }]
    }))
    .unwrap();
    let _ = ok_value(&body);
}

#[test]
fn finish_reason_length_without_tool_calls_is_success() {
    let body = serde_json::to_vec(&json!({
        "choices": [{
            "finish_reason": "length",
            "message": { "role": "assistant", "content": "cut" }
        }]
    }))
    .unwrap();
    let _ = ok_value(&body);
}

#[test]
fn success_returns_parsed_object() {
    let value = json!({
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hi" },
            "finish_reason": "stop"
        }]
    });
    let body = serde_json::to_vec(&value).unwrap();
    let got = ok_value(&body);
    assert_eq!(got, value);
}

#[test]
fn one_truncated_choice_among_many_is_truncated() {
    let body = serde_json::to_vec(&json!({
        "choices": [
            {
                "finish_reason": "stop",
                "message": { "content": "ok" }
            },
            {
                "finish_reason": "length",
                "message": {
                    "tool_calls": [{
                        "function": { "name": "f", "arguments": "{" }
                    }]
                }
            }
        ]
    }))
    .unwrap();
    err_kind(&body, TransportFault::TruncatedToolCall);
}
