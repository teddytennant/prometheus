//! Group: fault injection -- batch gates, unknown provider, Hosted, EngineDown.

mod common;
mod reference;

use common::{
    assert_batch_too_large, assert_empty_batch, assert_engine_down, assert_hosted,
    assert_provider_not_found, engine_config, gen_req, last_resort, pid, start_engine, swarm_aimd,
    NOW,
};
use prometheus_serve::{LocalEngine, Router};

#[test]
fn engine_empty_batch_and_too_large_payloads() {
    let mut eng = start_engine(1, 3);
    assert_empty_batch(&eng.generate(gen_req(&[], 8, 1.0), NOW).unwrap_err());
    assert_batch_too_large(
        &eng.generate(gen_req(&["a", "b", "c", "d"], 1, 0.0), NOW)
            .unwrap_err(),
        3,
    );
}

#[test]
fn engine_max_batch_zero_rejects_any_prompt() {
    let mut eng = LocalEngine::start(engine_config(1, 0)).unwrap();
    assert_empty_batch(&eng.generate(gen_req(&[], 1, 0.0), NOW).unwrap_err());
    assert_batch_too_large(&eng.generate(gen_req(&["x"], 1, 0.0), NOW).unwrap_err(), 0);
}

#[test]
fn router_empty_batch_on_last_resort() {
    let mut r = Router::new(vec![], start_engine(1, 8), swarm_aimd()).unwrap();
    assert_empty_batch(&r.generate(gen_req(&[], 1, 0.0), NOW).unwrap_err());
}

#[test]
fn router_batch_too_large_on_last_resort() {
    let mut r = Router::new(vec![], start_engine(1, 1), swarm_aimd()).unwrap();
    assert_batch_too_large(
        &r.generate(gen_req(&["a", "b"], 1, 0.0), NOW).unwrap_err(),
        1,
    );
}

#[test]
fn router_hosted_does_not_inspect_batch() {
    let mut r = Router::new(vec![pid("xai")], start_engine(1, 1), swarm_aimd()).unwrap();
    assert_hosted(&r.generate(gen_req(&[], 1, 0.0), NOW).unwrap_err(), "xai");
    assert_hosted(
        &r.generate(gen_req(&["a", "b"], 1, 0.0), NOW).unwrap_err(),
        "xai",
    );
}

#[test]
fn router_unknown_provider_does_not_change_current() {
    let mut r = Router::new(vec![pid("xai")], start_engine(1, 8), swarm_aimd()).unwrap();
    assert_provider_not_found(&r.mark_down(&pid("missing")).unwrap_err(), "missing");
    assert_hosted(
        &r.generate(gen_req(&["hi"], 1, 0.0), NOW).unwrap_err(),
        "xai",
    );
}

#[test]
fn mark_down_sglang_then_all_hosted_is_engine_down_wait_backoff_ok() {
    let mut r = Router::new(vec![pid("xai")], start_engine(1, 8), swarm_aimd()).unwrap();
    r.mark_down(&last_resort()).unwrap();
    r.mark_down(&pid("xai")).unwrap();
    assert_engine_down(&r.generate(gen_req(&["hi"], 1, 0.0), NOW).unwrap_err());
    let _d = r.failover().wait_backoff();
}

#[test]
fn start_bad_gpu_is_not_other_or_slurm() {
    let err = LocalEngine::start(engine_config(3, 8))
        .err()
        .expect("must fail");
    match err {
        prometheus_serve::Error::BadGpuCount(3) => {}
        other => panic!("expected BadGpuCount(3), got {other:?}"),
    }
}
