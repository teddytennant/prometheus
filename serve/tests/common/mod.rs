//! Locked F5 oracle rules. Implementers must match these, not invent them.
//!
//! This crate is **stock** SGLang as the H6 last-resort provider (spec 15.2,
//! 15.5 F5, gate D5). The SGLang fork (latent decode, recurrence, MTP) is
//! C1/C2/I4 -- do not implement or test it here. CPU tests drive
//! [`prometheus_serve::LocalEngine`]. There are no GPU-only tests in this
//! crate (`gpu` feature / V-stage markers are unused).
//!
//! `NowMs` is an argument everywhere. The engine, router, and
//! `GenerateResponse::openai_body` must not read the wall clock.
//!
//! # LocalEngine completions
//!
//! Deterministic given `(prompts, max_tokens, temperature)` only. Independent
//! of `model`, `gpus`, `port`, `max_batch` (except the batch-size gate), `now`,
//! and the wall clock. See `tests/reference` for the exact token generator.
//!
//! - `LocalEngine::start` rejects `gpus` not in `prometheus_slurm::ALLOWED_GPUS`
//!   (`[1, 2, 4]`) with `Error::BadGpuCount(gpus)`. Accepts 1, 2, 4.
//! - After `start`: `is_up() == true`, `provider_id() == ProviderId("sglang")`
//!   (`LAST_RESORT_ID`), `config()` equals the input.
//! - `mark_down` then `is_up() == false`; `generate` returns `Error::EngineDown`.
//!   `mark_up` restores `is_up() == true` and generate works again.
//! - Check order in `generate`:
//!   1. `!is_up()` -> `Error::EngineDown` (even if prompts are empty / too big)
//!   2. `prompts.is_empty()` -> `Error::EmptyBatch`
//!   3. `prompts.len() > max_batch as usize` -> `Error::BatchTooLarge(max_batch)`
//!      (payload is `max_batch`, not the request length)
//!   4. else one `Completion` per prompt
//! - Happy path: `prompt_index` is `0..n` in request order, `model` equals
//!   `config.model`, `finish_reason` is exactly `"stop"`. `max_tokens == 0`
//!   yields empty `text` still with `"stop"`.
//! - Calling `generate` twice with the same request (any `now`) yields equal
//!   responses. Completions match `tests/reference::generate`.
//!
//! # GenerateResponse::openai_body
//!
//! Pure function of the `GenerateResponse` fields. Extra JSON fields are
//! allowed; bytes must be identical across calls (no clock, no RNG).
//!
//! Required shape (OpenAI chat-completion style):
//! - JSON object
//! - `choices` is an array with **one element per completion**, same order
//! - `choices[i].message.content` equals `completions[i].text`
//! - `choices[i].finish_reason` equals `completions[i].finish_reason`
//! - missing `choices` or empty `choices` is a test fail
//! - the bytes pass `prometheus_providers::validate_response` (not
//!   `RateLimitText` / `MalformedJson` / `EmptyCompletion`)
//!
//! Engine-produced responses always have `n >= 1` completions, so their
//! `openai_body` always validates. Direct construction of an empty
//! `completions` vec is out of scope.
//!
//! # Deployer
//!
//! `new` stores `client` and `script`; getters return them (already
//! implemented on the stub -- constructors/getters may be green).
//! `submit` with `gpus` not in `ALLOWED_GPUS` returns `Error::BadGpuCount`
//! **without** calling slurm (no PATH lookup, no SSH). Live `sbatch` /
//! teardown is not required of CPU tests.
//!
//! # Router (D5)
//!
//! `Router::new(hosted, engine, aimd)`:
//! - Failover provider list is `hosted` with `ProviderId("sglang")` appended
//!   **iff** it is not already the last entry. Duplicate last-resort in
//!   `hosted` is fine as long as the engine id is still last.
//! - Empty `hosted` -> providers `["sglang"]`.
//! - AIMD starts as `Aimd::new(aimd)` then is **raised to `max_window`** by
//!   repeated `on_success(0)` until `window() == max_window` or the window
//!   stops increasing (latency 0 is never congestion under H6). This is the
//!   starting swarm size. Tests use `min_window < max_window` and `increase >= 1`.
//!
//! `failover().current()` is the first hosted id while that id is up.
//!
//! `generate` check order:
//! 1. `failover.current()` is `Err(NoProvider)` -> `Error::EngineDown`
//!    (NoProvider is mapped; not a raw H6 error)
//! 2. current id is not `sglang` -> `Error::Hosted(id.0)` (even if the
//!    engine is down or the batch is empty)
//! 3. else `engine.generate(req, now)` (EngineDown / EmptyBatch /
//!    BatchTooLarge / completions)
//!
//! After `mark_down` of every **hosted** id (not last-resort), current is
//! `sglang` and `generate` uses `LocalEngine`. This is D5: every hosted
//! provider down, swarm keeps working.
//!
//! # AIMD shrink on D5
//!
//! Each `Router::mark_down` of a hosted id (`id.0 != "sglang"`) that
//! **transitions that id from up to down** calls `Aimd::on_outage` once
//! (same shrink H6 uses on outage / 429). Re-marking an already-down hosted
//! id does not shrink again. `mark_down` of last-resort / `sglang` does not
//! shrink. `mark_up` does not grow the window.
//!
//! After N distinct hosted up->down transitions, `aimd().window()` equals
//! raising to max then calling `on_outage` N times. When at least one hosted
//! provider was marked down and `min_window <` starting window (`max_window`
//! after raise), the window is **strictly smaller** than the start.
//!
//! # Router mark_down / mark_up
//!
//! Unknown id -> `Error::Provider` wrapping H6 `Error::NotFound`, i.e.
//! `Error::Provider(format!("not found: {id}"))` via `From`. Not
//! `Error::NotFound`.
//!
//! `mark_down("sglang")` is a failover mark (the id is in the list). It does
//! not by itself call `engine.mark_down`. Combined with all hosted down,
//! `current` is `NoProvider` and `generate` is `EngineDown` even if the
//! engine is still up. `engine.mark_down()` while current is `sglang` also
//! yields `EngineDown`.
//!
//! `failover().wait_backoff()` still returns a `Duration` after everything
//! is down -- never an error (H6 rule; the type is `Duration`).
//!
//! Router does not read the wall clock: last-resort generate is independent
//! of `now`.

#![allow(dead_code)]

use prometheus_providers::{Aimd, AimdConfig, ProviderId};
use prometheus_serve::{
    Completion, EngineConfig, GenerateRequest, GenerateResponse, LocalEngine, NowMs, LAST_RESORT_ID,
};
use prometheus_slurm::ALLOWED_GPUS;
use serde_json::Value;

pub const NOW: NowMs = 1_700_000_000_000;
pub const FINISH_STOP: &str = "stop";

pub fn pid(id: &str) -> ProviderId {
    ProviderId(id.to_string())
}

pub fn last_resort() -> ProviderId {
    pid(LAST_RESORT_ID)
}

pub fn engine_config(gpus: u32, max_batch: u32) -> EngineConfig {
    EngineConfig {
        model: prometheus_serve::DEFAULT_MODEL.to_string(),
        gpus,
        max_batch,
        port: 30000,
    }
}

pub fn start_engine(gpus: u32, max_batch: u32) -> LocalEngine {
    LocalEngine::start(engine_config(gpus, max_batch)).expect("LocalEngine::start")
}

pub fn gen_req(prompts: &[&str], max_tokens: u32, temperature: f32) -> GenerateRequest {
    GenerateRequest {
        prompts: prompts.iter().map(|s| (*s).to_string()).collect(),
        max_tokens,
        temperature,
    }
}

pub fn swarm_aimd() -> AimdConfig {
    AimdConfig {
        min_window: 1,
        max_window: 16,
        increase: 1,
        decrease: 0.5,
        latency_limit_ms: 30_000,
    }
}

/// Grow AIMD to `max_window` the way `Router::new` must.
pub fn raise_aimd_to_max(aimd: &mut Aimd) {
    let max = aimd.config().max_window;
    for _ in 0..65_536 {
        if aimd.window() >= max {
            return;
        }
        let before = aimd.window();
        aimd.on_success(0);
        if aimd.window() <= before {
            return;
        }
    }
}

pub fn expected_window_after_hosted_downs(cfg: &AimdConfig, n: usize) -> u32 {
    let mut a = Aimd::new(cfg.clone());
    raise_aimd_to_max(&mut a);
    for _ in 0..n {
        a.on_outage();
    }
    a.window()
}

pub fn expected_failover_ids(hosted: &[ProviderId]) -> Vec<ProviderId> {
    let mut v = hosted.to_vec();
    let already_last = v
        .last()
        .map(|p| p.0.as_str() == LAST_RESORT_ID)
        .unwrap_or(false);
    if !already_last {
        v.push(last_resort());
    }
    v
}

pub fn disallowed_gpus() -> [u32; 6] {
    [0, 3, 5, 7, 8, 32]
}

pub fn allowed_gpus() -> [u32; 3] {
    ALLOWED_GPUS
}

pub fn assert_bad_gpu(err: &prometheus_serve::Error, n: u32) {
    match err {
        prometheus_serve::Error::BadGpuCount(got) => {
            assert_eq!(*got, n, "BadGpuCount payload")
        }
        other => panic!("expected BadGpuCount({n}), got {other:?}"),
    }
}

pub fn assert_engine_down(err: &prometheus_serve::Error) {
    match err {
        prometheus_serve::Error::EngineDown => {}
        other => panic!("expected EngineDown, got {other:?}"),
    }
}

pub fn assert_empty_batch(err: &prometheus_serve::Error) {
    match err {
        prometheus_serve::Error::EmptyBatch => {}
        other => panic!("expected EmptyBatch, got {other:?}"),
    }
}

pub fn assert_batch_too_large(err: &prometheus_serve::Error, max_batch: u32) {
    match err {
        prometheus_serve::Error::BatchTooLarge(got) => {
            assert_eq!(*got, max_batch, "BatchTooLarge payload is max_batch")
        }
        other => panic!("expected BatchTooLarge({max_batch}), got {other:?}"),
    }
}

pub fn assert_hosted(err: &prometheus_serve::Error, id: &str) {
    match err {
        prometheus_serve::Error::Hosted(got) => assert_eq!(got, id, "Hosted payload"),
        other => panic!("expected Hosted({id}), got {other:?}"),
    }
}

pub fn assert_provider_not_found(err: &prometheus_serve::Error, id: &str) {
    match err {
        prometheus_serve::Error::Provider(s) => {
            assert_eq!(
                s,
                &format!("not found: {id}"),
                "locked: Error::Provider wrapping H6 NotFound Display"
            );
        }
        prometheus_serve::Error::NotFound(_) => {
            panic!("locked Error::Provider wrapping NotFound, not Error::NotFound ({id})")
        }
        other => panic!("expected Provider(not found: {id}), got {other:?}"),
    }
}

pub fn assert_happy_completions(resp: &GenerateResponse, req: &GenerateRequest, model: &str) {
    assert_eq!(resp.model, model);
    assert_eq!(resp.completions.len(), req.prompts.len());
    for (i, c) in resp.completions.iter().enumerate() {
        assert_eq!(c.prompt_index, i, "prompt_index in request order");
        assert_eq!(c.finish_reason, FINISH_STOP);
    }
}

pub fn assert_match_reference(cfg: &EngineConfig, req: &GenerateRequest, got: &GenerateResponse) {
    let refer = crate::reference::generate(cfg, req);
    assert_eq!(got, &refer, "production generate vs tests/reference");
}

/// Required openai_body shape + H6 transport validation.
pub fn assert_openai_shape(body: &[u8], resp: &GenerateResponse) {
    assert!(
        !resp.completions.is_empty(),
        "openai_body shape checks need n>=1 completions"
    );
    let v: Value = serde_json::from_slice(body).unwrap_or_else(|e| {
        panic!(
            "openai_body must be JSON: {e}; bytes={:?}",
            String::from_utf8_lossy(body)
        )
    });
    assert!(v.is_object(), "openai_body must be a JSON object");
    let choices = v
        .get("choices")
        .unwrap_or_else(|| panic!("openai_body missing choices; body={v}"))
        .as_array()
        .unwrap_or_else(|| panic!("choices must be an array; body={v}"));
    assert!(
        !choices.is_empty(),
        "empty choices fails validate_response; body={v}"
    );
    assert_eq!(
        choices.len(),
        resp.completions.len(),
        "one choice per completion"
    );
    for (choice, comp) in choices.iter().zip(resp.completions.iter()) {
        let content = choice
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("choice missing message.content; choice={choice}"));
        assert_eq!(content, comp.text, "message.content == Completion.text");
        let fr = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("choice missing finish_reason; choice={choice}"));
        assert_eq!(fr, comp.finish_reason, "finish_reason");
    }
    prometheus_providers::validate_response(body)
        .unwrap_or_else(|e| panic!("openai_body must pass validate_response: {e}; body={v}"));
}

pub fn sample_response() -> GenerateResponse {
    GenerateResponse {
        model: "open-weights".into(),
        completions: vec![
            Completion {
                prompt_index: 0,
                text: "alpha".into(),
                finish_reason: FINISH_STOP.into(),
            },
            Completion {
                prompt_index: 1,
                text: "beta".into(),
                finish_reason: "length".into(),
            },
        ],
    }
}
