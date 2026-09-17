//! Group: property tests — production matches the independent reference.

mod common;
mod reference;

use common::{aimd_cfg, assert_invalid, ids, pid, provider, solo_cluster, FAKE_BLOB};
use prometheus_providers::{validate_response, Aimd, Error, Failover, TransportFault};
use reference::{
    expected_refresh, sha256_hex, validate_response as ref_validate, RefAimd, RefFailover,
};
use serde_json::json;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn u32n(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        (self.next() as u32) % n
    }
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.u32n(items.len() as u32) as usize]
    }
}

fn assert_validate_eq(body: &[u8]) {
    match (validate_response(body), ref_validate(body)) {
        (Ok(a), Ok(b)) => assert_eq!(a, b),
        (Err(Error::InvalidResponse(a)), Err(Error::InvalidResponse(b))) => assert_eq!(a, b),
        (p, r) => panic!("validate mismatch prod={p:?} ref={r:?} body={body:?}"),
    }
}

#[test]
fn validate_matches_reference_on_corpus_and_random() {
    let corpus: Vec<Vec<u8>> = vec![
        b"rate_limit".to_vec(),
        b"RATE_LIMIT".to_vec(),
        b"".to_vec(),
        b"not json".to_vec(),
        b"[]".to_vec(),
        br#"{"error":"rate_limit"}"#.to_vec(),
        br#"{"choices":[]}"#.to_vec(),
        br#"{"id":1}"#.to_vec(),
        serde_json::to_vec(&json!({"choices":[{"finish_reason":"stop"}]})).unwrap(),
        serde_json::to_vec(&json!({
            "choices": [{
                "finish_reason": "length",
                "message": { "tool_calls": [{ "function": { "name": "f" } }] }
            }]
        }))
        .unwrap(),
        serde_json::to_vec(&json!({
            "choices": [{
                "finish_reason": "length",
                "message": { "tool_calls": [{ "function": { "name": "f", "arguments": "{" } }] }
            }]
        }))
        .unwrap(),
        vec![0xff, 0x00],
    ];
    for body in &corpus {
        assert_validate_eq(body);
    }

    let mut rng = Lcg(0x4836);
    for i in 0..64 {
        let n = 1 + rng.u32n(24);
        let mut body = vec![0u8; n as usize];
        for b in &mut body {
            *b = rng.u32n(256) as u8;
        }
        if i % 7 == 0 {
            body = b"rate_limit".to_vec();
        }
        if i % 11 == 0 {
            body =
                serde_json::to_vec(&json!({"choices":[{"finish_reason":"stop","i":i}]})).unwrap();
        }
        assert_validate_eq(&body);
    }
}

#[test]
fn aimd_sequence_matches_reference() {
    let mut rng = Lcg(2026);
    let decreases = [0.25_f64, 0.5, 0.75, 1.0, 1.5];
    for _trial in 0..32 {
        let min_window = rng.u32n(5);
        let max_window = min_window + 1 + rng.u32n(16);
        let increase = 1 + rng.u32n(4);
        let decrease = *rng.pick(&decreases);
        let latency_limit = 50 + rng.next() % 500;
        let cfg = aimd_cfg(min_window, max_window, increase, decrease, latency_limit);
        let mut prod = Aimd::new(cfg.clone());
        let mut refer = RefAimd::new(cfg);
        for _op in 0..24 {
            match rng.u32n(4) {
                0 => {
                    let lat = rng.next() % (latency_limit.saturating_mul(2).saturating_add(1));
                    prod.on_success(lat);
                    refer.on_success(lat);
                }
                1 => {
                    prod.on_429();
                    refer.on_429();
                }
                _ => {
                    prod.on_outage();
                    refer.on_outage();
                }
            }
            assert_eq!(prod.window(), refer.window());
        }
    }
}

#[test]
fn failover_ops_match_reference() {
    let mut rng = Lcg(99);
    let names = ["a", "b", "c", "d", "e"];
    for n in 1..=5 {
        let slice = &names[..n];
        let mut prod = Failover::new(ids(slice));
        let mut refer = RefFailover::new(ids(slice));
        for _ in 0..40 {
            let id = pid(slice[rng.u32n(slice.len() as u32) as usize]);
            match rng.u32n(5) {
                0 => {
                    let p = prod.mark_down(&id);
                    let r = refer.mark_down(&id);
                    match (p, r) {
                        (Ok(()), Ok(())) => {}
                        (Err(Error::NotFound(a)), Err(Error::NotFound(b))) => assert_eq!(a, b),
                        other => panic!("mark_down {other:?}"),
                    }
                }
                1 => {
                    let p = prod.mark_up(&id);
                    let r = refer.mark_up(&id);
                    match (p, r) {
                        (Ok(()), Ok(())) => {}
                        (Err(Error::NotFound(a)), Err(Error::NotFound(b))) => assert_eq!(a, b),
                        other => panic!("mark_up {other:?}"),
                    }
                }
                2 => match (prod.current(), refer.current()) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b),
                    (Err(Error::NoProvider), Err(Error::NoProvider)) => {}
                    other => panic!("current {other:?}"),
                },
                _ => {
                    assert_eq!(prod.wait_backoff(), refer.wait_backoff());
                }
            }
        }
        // unknown id
        let ghost = pid("ghost");
        let p = prod.mark_down(&ghost).unwrap_err();
        let r = refer.mark_down(&ghost).unwrap_err();
        match (p, r) {
            (Error::NotFound(a), Error::NotFound(b)) => assert_eq!(a, b),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn broker_refresh_matches_reference_token_and_generation() {
    let (_parent, mut cluster) = solo_cluster();
    let cfg = provider("xai", 12_345);
    let mut br = prometheus_providers::Broker::new(cfg.clone());
    let blobs: [&[u8]; 3] = [FAKE_BLOB, FAKE_BLOB, b"fake-token"];
    for (i, blob) in blobs.iter().enumerate() {
        let now = 1_000 * (i as u64 + 1);
        let (rec, tok) = br.refresh(&mut cluster, blob, now).expect("refresh");
        let (wrec, wtok) = expected_refresh(&cfg, blob, now, (i as u64) + 1);
        assert_eq!(rec, wrec);
        assert_eq!(tok, wtok);
        assert_eq!(tok.token, sha256_hex(blob));
    }
}

#[test]
fn validate_fault_kinds_are_stable_against_reference() {
    let cases: &[(&[u8], TransportFault)] = &[
        (b"rate_limit", TransportFault::RateLimitText),
        (br#"{"error":"rate_limit"}"#, TransportFault::RateLimitText),
        (br#"{"choices":[]}"#, TransportFault::EmptyCompletion),
        (b"{", TransportFault::MalformedJson),
    ];
    for (body, kind) in cases {
        let err = validate_response(*body).unwrap_err();
        assert_invalid(&err, *kind);
        assert_validate_eq(*body);
    }
}
