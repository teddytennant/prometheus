//! Group: Router last-resort failover (gate D5) and AIMD reduced swarm size.

mod common;
mod reference;

use common::{
    assert_engine_down, assert_happy_completions, assert_hosted, assert_match_reference,
    assert_openai_shape, assert_provider_not_found, engine_config, expected_failover_ids,
    expected_window_after_hosted_downs, gen_req, last_resort, pid, start_engine, swarm_aimd, NOW,
};
use prometheus_providers::{AimdConfig, ProviderId};
use prometheus_serve::{LocalEngine, Router, LAST_RESORT_ID};
use std::time::Duration;

fn started() -> LocalEngine {
    start_engine(1, 8)
}

fn router(hosted: &[&str]) -> Router {
    let ids: Vec<ProviderId> = hosted.iter().map(|s| pid(s)).collect();
    Router::new(ids, started(), swarm_aimd()).expect("Router::new")
}

#[test]
fn new_appends_sglang_as_last_if_missing() {
    let r = router(&["xai", "openai"]);
    let got = r.failover().providers().to_vec();
    assert_eq!(got, expected_failover_ids(&[pid("xai"), pid("openai")]));
    assert_eq!(got.last().unwrap().0, LAST_RESORT_ID);
    assert_eq!(r.last_resort().0, "sglang");
    assert!(r.engine().is_up());
}

#[test]
fn new_does_not_append_when_sglang_already_last() {
    let r = router(&["xai", "sglang"]);
    let got = r.failover().providers().to_vec();
    assert_eq!(got, vec![pid("xai"), last_resort()]);
    assert_eq!(
        got.iter().filter(|p| p.0 == LAST_RESORT_ID).count(),
        1,
        "do not duplicate when already last"
    );
}

#[test]
fn duplicate_last_resort_in_the_middle_still_appends() {
    let r = router(&["sglang", "xai"]);
    let got = r.failover().providers().to_vec();
    assert_eq!(got, vec![last_resort(), pid("xai"), last_resort()]);
    assert_eq!(got.last().unwrap().0, LAST_RESORT_ID);
}

#[test]
fn empty_hosted_is_only_sglang() {
    let r = router(&[]);
    assert_eq!(r.failover().providers(), &[last_resort()]);
    match r.failover().current() {
        Ok(p) => assert_eq!(p.0, "sglang"),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn current_is_first_hosted_while_hosted_are_up() {
    let r = router(&["xai", "openai"]);
    match r.failover().current() {
        Ok(p) => assert_eq!(p.0, "xai"),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn generate_while_current_is_hosted_returns_hosted() {
    let mut r = router(&["xai", "openai"]);
    let err = r
        .generate(gen_req(&["hi"], 1, 0.0), NOW)
        .expect_err("no local backend");
    assert_hosted(&err, "xai");
}

#[test]
fn hosted_beats_empty_batch() {
    let mut r = router(&["xai"]);
    assert_hosted(
        &r.generate(gen_req(&[], 1, 0.0), NOW)
            .expect_err("empty still Hosted"),
        "xai",
    );
}

#[test]
fn d5_after_all_hosted_down_generate_uses_engine() {
    let mut r = router(&["xai", "openai"]);
    r.mark_down(&pid("xai")).unwrap();
    r.mark_down(&pid("openai")).unwrap();
    match r.failover().current() {
        Ok(p) => assert_eq!(p.0, "sglang"),
        Err(e) => panic!("D5 current must be sglang, got {e}"),
    }
    let req = gen_req(&["hi"], 1, 0.0);
    let got = r
        .generate(req.clone(), NOW)
        .expect("D5 swarm keeps working");
    assert_happy_completions(&got, &req, r.engine().config().model.as_str());
    assert_match_reference(r.engine().config(), &req, &got);
    assert_eq!(got.completions[0].text, "75163d3c");
    assert_openai_shape(&got.openai_body(), &got);
}

#[test]
fn aimd_window_starts_at_max_and_shrinks_once_per_hosted_down() {
    let cfg = swarm_aimd();
    let start = expected_window_after_hosted_downs(&cfg, 0);
    assert_eq!(start, cfg.max_window);
    assert!(cfg.min_window < start);

    let mut r = router(&["xai", "openai", "anthropic"]);
    assert_eq!(r.aimd().window(), start, "Router::new raises AIMD to max");

    r.mark_down(&pid("xai")).unwrap();
    assert_eq!(
        r.aimd().window(),
        expected_window_after_hosted_downs(&cfg, 1)
    );
    assert!(
        r.aimd().window() < start,
        "strictly smaller after one hosted down"
    );

    r.mark_down(&pid("xai")).unwrap();
    assert_eq!(
        r.aimd().window(),
        expected_window_after_hosted_downs(&cfg, 1),
        "idempotent re-down does not shrink again"
    );

    r.mark_down(&pid("openai")).unwrap();
    assert_eq!(
        r.aimd().window(),
        expected_window_after_hosted_downs(&cfg, 2)
    );

    r.mark_down(&last_resort()).unwrap();
    assert_eq!(
        r.aimd().window(),
        expected_window_after_hosted_downs(&cfg, 2),
        "last-resort down does not shrink AIMD"
    );

    r.mark_down(&pid("anthropic")).unwrap();
    assert_eq!(
        r.aimd().window(),
        expected_window_after_hosted_downs(&cfg, 3)
    );
}

#[test]
fn mark_up_does_not_grow_aimd() {
    let mut r = router(&["xai"]);
    r.mark_down(&pid("xai")).unwrap();
    let shrunk = r.aimd().window();
    r.mark_up(&pid("xai")).unwrap();
    assert_eq!(r.aimd().window(), shrunk);
    match r.failover().current() {
        Ok(p) => assert_eq!(p.0, "xai"),
        Err(e) => panic!("{e}"),
    }
    assert_hosted(
        &r.generate(gen_req(&["hi"], 1, 0.0), NOW)
            .expect_err("hosted again"),
        "xai",
    );
}

#[test]
fn mark_down_unknown_id_is_provider_not_found() {
    let mut r = router(&["xai"]);
    let err = r.mark_down(&pid("nope")).expect_err("unknown");
    assert_provider_not_found(&err, "nope");
    let err = r.mark_up(&pid("ghost")).expect_err("unknown up");
    assert_provider_not_found(&err, "ghost");
}

#[test]
fn last_resort_and_engine_down_is_engine_down() {
    let mut r = router(&["xai"]);
    r.mark_down(&pid("xai")).unwrap();
    r.mark_down(&last_resort()).unwrap();
    match r.failover().current() {
        Err(prometheus_providers::Error::NoProvider) => {}
        Ok(p) => panic!("all down -> NoProvider, got {}", p.0),
        Err(e) => panic!("all down -> NoProvider, got {e}"),
    }
    assert_engine_down(
        &r.generate(gen_req(&["hi"], 1, 0.0), NOW)
            .expect_err("NoProvider mapped to EngineDown"),
    );
    let d: Duration = r.failover().wait_backoff();
    assert!(d > Duration::ZERO, "wait_backoff never errors");
}

#[test]
fn engine_mark_down_while_current_is_sglang_is_engine_down() {
    let mut eng = LocalEngine::start(engine_config(1, 8)).unwrap();
    eng.mark_down();
    let mut r = Router::new(vec![], eng, swarm_aimd()).unwrap();
    match r.failover().current() {
        Ok(p) => assert_eq!(p.0, "sglang"),
        Err(e) => panic!("{e}"),
    }
    assert!(!r.engine().is_up());
    assert_engine_down(
        &r.generate(gen_req(&["hi"], 1, 0.0), NOW)
            .expect_err("engine down"),
    );
}

#[test]
fn generate_independent_of_now_on_last_resort() {
    let mut r = router(&[]);
    let req = gen_req(&["hi"], 2, 0.0);
    let a = r.generate(req.clone(), 0).unwrap();
    let b = r.generate(req.clone(), u64::MAX).unwrap();
    assert_eq!(a, b);
    assert_match_reference(r.engine().config(), &req, &a);
}

#[test]
fn custom_aimd_config_is_stored() {
    let cfg = AimdConfig {
        min_window: 2,
        max_window: 8,
        increase: 2,
        decrease: 0.5,
        latency_limit_ms: 10,
    };
    let r = Router::new(vec![pid("xai")], started(), cfg.clone()).unwrap();
    assert_eq!(r.aimd().config().min_window, 2);
    assert_eq!(r.aimd().config().max_window, 8);
    assert_eq!(r.aimd().window(), 8);
}
