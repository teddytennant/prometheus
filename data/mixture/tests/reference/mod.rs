//! Independent B6 reference: shells out to `mixture.py`.
//!
//! Compiled only as a submodule of the integration tests. Production
//! `src/lib.rs` must stay `unimplemented!` except `flagship_catalog` and
//! `Source::all`.
#![allow(dead_code)]

use prometheus_mixture::{Error, Mix, Phase, Result, Source};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reference")
        .join("mixture.py")
}

fn special_f64(x: f64) -> serde_json::Value {
    if x.is_nan() {
        json!("nan")
    } else if x == f64::INFINITY {
        json!("inf")
    } else if x == f64::NEG_INFINITY {
        json!("-inf")
    } else {
        json!(x)
    }
}

fn call(request: serde_json::Value) -> serde_json::Value {
    let script = script();
    let mut child = Command::new("python3")
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn python3 {script:?}: {e}"));
    {
        let mut stdin = child.stdin.take().expect("python stdin");
        serde_json::to_writer(&mut stdin, &request).expect("write mixture.py request");
        stdin.flush().ok();
    }
    let out = child.wait_with_output().expect("wait python3");
    if !out.status.success() {
        panic!(
            "mixture.py exited {}: stdout={} stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "mixture.py json {e}: stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

fn parse<T: serde::de::DeserializeOwned>(response: serde_json::Value) -> Result<T> {
    let ok = response
        .get("ok")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if ok {
        let value = serde_json::from_value(
            response
                .get("value")
                .cloned()
                .unwrap_or(serde_json::Value::Null),
        )
        .unwrap_or_else(|e| panic!("mixture.py value: {e} in {response}"));
        Ok(value)
    } else {
        let kind = response
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("config");
        let msg = response
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if kind == "unknown_source" {
            Err(Error::UnknownSource(msg))
        } else {
            Err(Error::Config(msg))
        }
    }
}

fn mix_json(mix: &Mix) -> serde_json::Value {
    // serde_json rejects NaN/Inf; encode those as strings the Python
    // reference already accepts via `_parse_special_float`.
    let mut weights = serde_json::Map::new();
    for (src, w) in &mix.weights {
        weights.insert(source_name(*src), special_f64(*w));
    }
    let phase = match mix.phase {
        Phase::Pretrain => "pretrain",
        Phase::Decay => "decay",
    };
    json!({
        "mix_id": mix.mix_id,
        "mix_bucket": mix.mix_bucket,
        "phase": phase,
        "weights": weights,
    })
}

fn source_name(source: Source) -> String {
    serde_json::to_value(source)
        .expect("serialize Source")
        .as_str()
        .expect("Source string")
        .to_string()
}

pub fn validate_mix(mix: &Mix) -> Result<()> {
    parse(call(json!({ "op": "validate_mix", "mix": mix_json(mix) })))
}

pub fn canonical_json(mix: &Mix) -> Result<Vec<u8>> {
    let s: String = parse(call(json!({ "op": "canonical_json", "mix": mix_json(mix) })))?;
    Ok(s.into_bytes())
}

pub fn mix_hash(mix: &Mix) -> Result<String> {
    parse(call(json!({ "op": "mix_hash", "mix": mix_json(mix) })))
}

pub fn decay_reweight(mix: &Mix, factor: f64) -> Result<Mix> {
    parse(call(json!({
        "op": "decay_reweight",
        "mix": mix_json(mix),
        "factor": special_f64(factor),
    })))
}

pub fn drop_source(mix: &Mix, source: Source) -> Result<Mix> {
    parse(call(json!({
        "op": "drop_source",
        "mix": mix_json(mix),
        "source": source_name(source),
    })))
}

pub fn rung2_ablations(base: &Mix) -> Result<Vec<Mix>> {
    parse(call(json!({ "op": "rung2_ablations", "mix": mix_json(base) })))
}

pub fn tokens_seen(unique_tokens: Option<u64>, epochs: u32) -> Option<u64> {
    parse::<Option<u64>>(call(json!({
        "op": "tokens_seen",
        "unique_tokens": unique_tokens,
        "epochs": epochs,
    })))
    .expect("tokens_seen is infallible")
}

pub fn sample_source(mix: &Mix, u: f64) -> Result<Source> {
    parse(call(json!({
        "op": "sample_source",
        "mix": mix_json(mix),
        "u": special_f64(u),
    })))
}
