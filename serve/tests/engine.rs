//! Group: LocalEngine start, up/down, batch gates, deterministic completions.

mod common;
mod reference;

use common::{
    allowed_gpus, assert_bad_gpu, assert_batch_too_large, assert_empty_batch, assert_engine_down,
    assert_happy_completions, assert_match_reference, disallowed_gpus, engine_config, gen_req,
    start_engine, FINISH_STOP, NOW,
};
use prometheus_serve::{EngineConfig, LocalEngine, LAST_RESORT_ID};
use prometheus_slurm::ALLOWED_GPUS;

#[test]
fn start_rejects_gpus_outside_allowed() {
    for gpus in disallowed_gpus() {
        let err = match LocalEngine::start(engine_config(gpus, 8)) {
            Ok(_) => panic!("start must reject gpus={gpus}"),
            Err(e) => e,
        };
        assert_bad_gpu(&err, gpus);
        assert!(
            reference::start_rejects_gpus(gpus),
            "reference agrees {gpus} is illegal"
        );
    }
}

#[test]
fn start_accepts_allowed_gpus() {
    for gpus in allowed_gpus() {
        assert!(ALLOWED_GPUS.contains(&gpus));
        let cfg = engine_config(gpus, 8);
        let eng = LocalEngine::start(cfg.clone()).expect("accept 1, 2, 4");
        assert!(eng.is_up(), "after start is_up == true");
        assert_eq!(eng.provider_id().0, LAST_RESORT_ID);
        assert_eq!(eng.provider_id().0, "sglang");
        assert_eq!(eng.config(), &cfg);
    }
}

#[test]
fn start_preserves_full_config() {
    let cfg = EngineConfig {
        model: "llama-oracle".into(),
        gpus: 2,
        max_batch: 17,
        port: 12345,
    };
    let eng = LocalEngine::start(cfg.clone()).unwrap();
    assert_eq!(eng.config(), &cfg);
    assert_eq!(eng.provider_id().0, "sglang");
}

#[test]
fn mark_down_then_up() {
    let mut eng = start_engine(1, 8);
    assert!(eng.is_up());
    eng.mark_down();
    assert!(!eng.is_up());
    let err = eng
        .generate(gen_req(&["hi"], 1, 0.0), NOW)
        .expect_err("down");
    assert_engine_down(&err);
    eng.mark_up();
    assert!(eng.is_up());
    let got = eng.generate(gen_req(&["hi"], 1, 0.0), NOW).unwrap();
    assert_eq!(got.completions.len(), 1);
}

#[test]
fn generate_while_down_beats_empty_and_too_large() {
    let mut eng = start_engine(1, 1);
    eng.mark_down();
    assert_engine_down(
        &eng.generate(gen_req(&[], 1, 0.0), NOW)
            .expect_err("empty while down"),
    );
    assert_engine_down(
        &eng.generate(gen_req(&["a", "b"], 1, 0.0), NOW)
            .expect_err("too large while down"),
    );
}

#[test]
fn empty_prompts_is_empty_batch() {
    let mut eng = start_engine(1, 8);
    let err = eng.generate(gen_req(&[], 4, 0.0), NOW).expect_err("empty");
    assert_empty_batch(&err);
}

#[test]
fn prompts_over_max_batch_is_batch_too_large() {
    let mut eng = start_engine(1, 2);
    let err = eng
        .generate(gen_req(&["a", "b", "c"], 1, 0.0), NOW)
        .expect_err("too large");
    assert_batch_too_large(&err, 2);
}

#[test]
fn batch_equal_to_max_batch_is_ok() {
    let mut eng = start_engine(1, 2);
    let req = gen_req(&["a", "b"], 1, 0.0);
    let got = eng.generate(req.clone(), NOW).unwrap();
    assert_happy_completions(&got, &req, eng.config().model.as_str());
    assert_match_reference(eng.config(), &req, &got);
}

#[test]
fn golden_hi_one_token_temp_zero() {
    let mut eng = start_engine(1, 8);
    let req = gen_req(&["hi"], 1, 0.0);
    let got = eng.generate(req.clone(), NOW).unwrap();
    assert_eq!(got.model, prometheus_serve::DEFAULT_MODEL);
    assert_eq!(got.completions.len(), 1);
    assert_eq!(got.completions[0].prompt_index, 0);
    assert_eq!(got.completions[0].finish_reason, FINISH_STOP);
    assert_eq!(got.completions[0].text, "75163d3c");
    assert_eq!(
        got.completions[0].text,
        reference::completion_text("hi", 1, 0.0)
    );
}

#[test]
fn golden_hi_two_tokens_temp_zero_and_point_seven() {
    let mut eng = start_engine(4, 8);
    let a = eng.generate(gen_req(&["hi"], 2, 0.0), NOW).unwrap();
    assert_eq!(a.completions[0].text, "b15dfef7 9fc454b8");
    let b = eng.generate(gen_req(&["hi"], 2, 0.7), NOW).unwrap();
    assert_eq!(b.completions[0].text, "8bdb0443 3375f8ad");
    assert_ne!(a.completions[0].text, b.completions[0].text);
}

#[test]
fn batch_prompt_index_in_order() {
    let mut eng = start_engine(1, 8);
    let req = gen_req(&["one", "two", "three"], 1, 0.0);
    let got = eng.generate(req.clone(), NOW).unwrap();
    assert_happy_completions(&got, &req, eng.config().model.as_str());
    assert_match_reference(eng.config(), &req, &got);
    let texts: Vec<&str> = got.completions.iter().map(|c| c.text.as_str()).collect();
    assert_eq!(
        texts,
        vec![
            reference::completion_text("one", 1, 0.0).as_str(),
            reference::completion_text("two", 1, 0.0).as_str(),
            reference::completion_text("three", 1, 0.0).as_str(),
        ]
    );
}

#[test]
fn max_tokens_zero_is_empty_text_still_stop() {
    let mut eng = start_engine(1, 8);
    let req = gen_req(&["hi", ""], 0, 0.0);
    let got = eng.generate(req.clone(), NOW).unwrap();
    assert_happy_completions(&got, &req, eng.config().model.as_str());
    assert_eq!(got.completions[0].text, "");
    assert_eq!(got.completions[1].text, "");
    assert_match_reference(eng.config(), &req, &got);
}

#[test]
fn completions_ignore_now_and_are_deterministic() {
    let mut eng = start_engine(1, 8);
    let req = gen_req(&["clock", "free"], 3, 0.5);
    let a = eng.generate(req.clone(), 0).unwrap();
    let b = eng.generate(req.clone(), 1).unwrap();
    let c = eng.generate(req.clone(), u64::MAX).unwrap();
    let d = eng.generate(req.clone(), NOW).unwrap();
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(c, d);
    assert_match_reference(eng.config(), &req, &a);
}

#[test]
fn completions_ignore_gpus_port_max_batch_model_does_not_change_text() {
    let req = gen_req(&["same"], 2, 0.0);
    let mut a = LocalEngine::start(EngineConfig {
        model: "m-a".into(),
        gpus: 1,
        max_batch: 8,
        port: 1,
    })
    .unwrap();
    let mut b = LocalEngine::start(EngineConfig {
        model: "m-b".into(),
        gpus: 4,
        max_batch: 64,
        port: 9,
    })
    .unwrap();
    let ra = a.generate(req.clone(), NOW).unwrap();
    let rb = b.generate(req.clone(), 99).unwrap();
    assert_eq!(ra.model, "m-a");
    assert_eq!(rb.model, "m-b");
    assert_eq!(ra.completions[0].text, rb.completions[0].text);
    assert_eq!(
        ra.completions[0].text,
        reference::completion_text("same", 2, 0.0)
    );
}

#[test]
fn unicode_and_empty_prompt_string_are_valid_batch_items() {
    let mut eng = start_engine(2, 8);
    let req = gen_req(&["", "π", "你好", "a\nb"], 1, 0.0);
    let got = eng.generate(req.clone(), NOW).unwrap();
    assert_happy_completions(&got, &req, eng.config().model.as_str());
    assert_match_reference(eng.config(), &req, &got);
    assert_eq!(got.completions[1].text, "9466fd01");
}

#[test]
fn minus_zero_temperature_differs_from_plus_zero_via_to_bits() {
    let mut eng = start_engine(1, 4);
    let plus = eng.generate(gen_req(&["t"], 1, 0.0), NOW).unwrap();
    let minus = eng.generate(gen_req(&["t"], 1, -0.0), NOW).unwrap();
    assert_ne!(
        plus.completions[0].text, minus.completions[0].text,
        "f32 to_bits distinguishes -0.0 from 0.0"
    );
    assert_match_reference(eng.config(), &gen_req(&["t"], 1, 0.0), &plus);
    assert_match_reference(eng.config(), &gen_req(&["t"], 1, -0.0), &minus);
}
