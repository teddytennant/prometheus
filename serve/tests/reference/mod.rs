//! Independent slow reference for F5 LocalEngine completions and openai_body.
//!
//! Not a language model. Token text is a hash mix so the production engine
//! has an exact, clock-free target. Plain integer arithmetic (no NumPy --
//! this module has no tensor math).

#![allow(dead_code)]

use prometheus_serve::{
    Completion, EngineConfig, GenerateRequest, GenerateResponse, LAST_RESORT_ID,
};
use prometheus_slurm::ALLOWED_GPUS;
use serde_json::{json, Value};

use crate::common::FINISH_STOP;

/// FNV-1a 32-bit over UTF-8 bytes.
pub fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in bytes {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// 32-bit mix (Murmur-style finalizer). Slow and obvious.
pub fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Seed from `(prompt, max_tokens, temperature.to_bits())`.
pub fn prompt_seed(prompt: &str, max_tokens: u32, temperature: f32) -> u32 {
    let mut seed = fnv1a32(prompt.as_bytes());
    seed ^= max_tokens.wrapping_mul(0x9E37_79B9);
    seed ^= temperature.to_bits();
    seed
}

/// Locked completion text.
///
/// - `max_tokens == 0` -> empty string
/// - else `max_tokens` tokens, each 8 lowercase hex digits, joined by a
///   single ASCII space, no leading/trailing space
/// - token `j` (0-based) is `mix32(seed.wrapping_add((j+1).wrapping_mul(0x85EBCA6B)))`
///   formatted as `{:08x}`
///
/// Golden (also asserted against production in engine tests):
/// - `("hi", 1, 0.0)` -> `75163d3c`
/// - `("hi", 2, 0.0)` -> `b15dfef7 9fc454b8`
/// - `("hi", 2, 0.7)` -> `8bdb0443 3375f8ad`
pub fn completion_text(prompt: &str, max_tokens: u32, temperature: f32) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    let seed = prompt_seed(prompt, max_tokens, temperature);
    let mut parts = Vec::with_capacity(max_tokens as usize);
    for j in 0..max_tokens {
        let step = (j.wrapping_add(1)).wrapping_mul(0x85EB_CA6B);
        let token = mix32(seed.wrapping_add(step));
        parts.push(format!("{token:08x}"));
    }
    parts.join(" ")
}

/// Independent generate. Same errors as the locked LocalEngine check order,
/// without an engine instance (caller handles EngineDown / is_up).
pub fn generate(cfg: &EngineConfig, req: &GenerateRequest) -> GenerateResponse {
    GenerateResponse {
        model: cfg.model.clone(),
        completions: req
            .prompts
            .iter()
            .enumerate()
            .map(|(i, p)| Completion {
                prompt_index: i,
                text: completion_text(p, req.max_tokens, req.temperature),
                finish_reason: FINISH_STOP.to_string(),
            })
            .collect(),
    }
}

/// Minimal OpenAI-style body that satisfies the locked shape. Production may
/// add extra fields; tests do not require byte equality with this function.
pub fn openai_body(resp: &GenerateResponse) -> Vec<u8> {
    let choices: Vec<Value> = resp
        .completions
        .iter()
        .map(|c| {
            json!({
                "index": c.prompt_index,
                "message": { "role": "assistant", "content": c.text },
                "finish_reason": c.finish_reason,
            })
        })
        .collect();
    serde_json::to_vec(&json!({
        "model": resp.model,
        "choices": choices,
    }))
    .expect("serialize reference openai_body")
}

pub fn start_rejects_gpus(gpus: u32) -> bool {
    !ALLOWED_GPUS.contains(&gpus)
}

pub fn last_resort_id() -> &'static str {
    LAST_RESORT_ID
}
