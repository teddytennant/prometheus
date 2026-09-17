//! Group: GenerateResponse::openai_body shape and H6 validate_response.

mod common;
mod reference;

use common::{assert_openai_shape, sample_response};
use prometheus_providers::{validate_response, TransportFault};
use prometheus_serve::{Completion, GenerateResponse};
use serde_json::Value;

#[test]
fn openai_body_validates_and_matches_locked_shape() {
    let resp = sample_response();
    let body = resp.openai_body();
    assert_openai_shape(&body, &resp);
}

#[test]
fn openai_body_is_pure_same_bytes_twice() {
    let resp = sample_response();
    let a = resp.openai_body();
    let b = resp.openai_body();
    assert_eq!(a, b, "openai_body must not read the wall clock or RNG");
}

#[test]
fn openai_body_one_choice_per_completion_in_order() {
    let resp = sample_response();
    let body = resp.openai_body();
    let v: Value = serde_json::from_slice(&body).unwrap();
    let choices = v["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["message"]["content"], "alpha");
    assert_eq!(choices[0]["finish_reason"], "stop");
    assert_eq!(choices[1]["message"]["content"], "beta");
    assert_eq!(choices[1]["finish_reason"], "length");
}

#[test]
fn openai_body_not_rate_limit_or_malformed() {
    let resp = sample_response();
    let body = resp.openai_body();
    match validate_response(&body) {
        Ok(_) => {}
        Err(prometheus_providers::Error::InvalidResponse(kind)) => {
            panic!(
                "unexpected transport fault {kind:?}; body={}",
                String::from_utf8_lossy(&body)
            );
        }
        Err(e) => panic!("{e}"),
    }
    assert_ne!(
        match validate_response(&body) {
            Err(prometheus_providers::Error::InvalidResponse(k)) => Some(k),
            _ => None,
        },
        Some(TransportFault::RateLimitText)
    );
}

#[test]
fn openai_body_unicode_content() {
    let resp = GenerateResponse {
        model: "open-weights".into(),
        completions: vec![Completion {
            prompt_index: 0,
            text: "你好 π".into(),
            finish_reason: "stop".into(),
        }],
    };
    let body = resp.openai_body();
    assert_openai_shape(&body, &resp);
}

#[test]
fn reference_openai_body_validates_locked_shape() {
    // Documents the minimal body. Production is not required to match bytes.
    let resp = sample_response();
    let refer = reference::openai_body(&resp);
    assert_openai_shape(&refer, &resp);
    let prod = resp.openai_body();
    assert_openai_shape(&prod, &resp);
}

#[test]
fn extra_fields_allowed_but_choices_required() {
    let resp = sample_response();
    let body = resp.openai_body();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v.get("choices").is_some(), "missing choices is a test fail");
    let choices = v["choices"].as_array().expect("choices array");
    assert!(!choices.is_empty(), "empty choices is a test fail");
}
