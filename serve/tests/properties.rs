//! Group: properties -- 100 random batches vs reference, openai_body, D5.

mod common;
mod reference;

use common::{
    assert_happy_completions, assert_match_reference, assert_openai_shape, engine_config,
    expected_window_after_hosted_downs, last_resort, pid, swarm_aimd, NOW,
};
use prometheus_providers::ProviderId;
use prometheus_serve::{GenerateRequest, LocalEngine, Router};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    fn bounded(&mut self, n: u32) -> u32 {
        assert!(n > 0);
        self.u32() % n
    }
}

fn random_prompt(rng: &mut Lcg) -> String {
    const ALPHA: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789 _-";
    let kind = rng.bounded(6);
    match kind {
        0 => String::new(),
        1 => "π".repeat(1 + rng.bounded(3) as usize),
        2 => "你好".to_string(),
        _ => {
            let n = rng.bounded(25);
            let mut s = String::new();
            for _ in 0..n {
                s.push(ALPHA[rng.bounded(ALPHA.len() as u32) as usize] as char);
            }
            s
        }
    }
}

fn random_request(rng: &mut Lcg, max_batch: u32) -> GenerateRequest {
    let n = 1 + rng.bounded(max_batch);
    let mut prompts = Vec::with_capacity(n as usize);
    for _ in 0..n {
        prompts.push(random_prompt(rng));
    }
    GenerateRequest {
        prompts,
        max_tokens: rng.bounded(9),
        temperature: f32::from_bits(rng.u32()),
    }
}

#[test]
fn engine_generate_matches_reference_on_100_random_batches() {
    let mut rng = Lcg(0xF5F5_F5F5_0000_0001);
    let cfg = engine_config(1, 8);
    let mut eng = LocalEngine::start(cfg.clone()).unwrap();
    for i in 0..100 {
        let req = random_request(&mut rng, cfg.max_batch);
        let now = if i % 3 == 0 {
            0
        } else {
            NOW.wrapping_add(i as u64)
        };
        let got = eng
            .generate(req.clone(), now)
            .unwrap_or_else(|e| panic!("batch {i} generate: {e}"));
        assert_happy_completions(&got, &req, cfg.model.as_str());
        assert_match_reference(&cfg, &req, &got);
        let body = got.openai_body();
        assert_openai_shape(&body, &got);
        let body2 = got.openai_body();
        assert_eq!(body, body2, "openai_body pure on batch {i}");
    }
}

#[test]
fn openai_body_always_validates_for_engine_output() {
    let mut rng = Lcg(7);
    let mut eng = LocalEngine::start(engine_config(2, 4)).unwrap();
    for _ in 0..32 {
        let req = random_request(&mut rng, 4);
        let got = eng.generate(req, NOW).unwrap();
        assert_openai_shape(&got.openai_body(), &got);
        prometheus_providers::validate_response(&got.openai_body()).unwrap();
    }
}

#[test]
fn d5_random_hosted_lists_generate_succeeds_after_all_hosted_down() {
    let mut rng = Lcg(0xD5D5_0001);
    const NAMES: [&str; 4] = ["xai", "openai", "anthropic", "grok"];
    let cfg = swarm_aimd();
    for trial in 0..50 {
        let n = rng.bounded(5) as usize; // 0..=4
        let mut hosted: Vec<ProviderId> = Vec::new();
        let mut used = [false; 4];
        while hosted.len() < n {
            let i = rng.bounded(4) as usize;
            if !used[i] {
                used[i] = true;
                hosted.push(pid(NAMES[i]));
            }
        }
        let eng = LocalEngine::start(engine_config(1, 8)).unwrap();
        let mut r = Router::new(hosted.clone(), eng, cfg.clone()).unwrap();
        for h in &hosted {
            r.mark_down(h).unwrap();
        }
        match r.failover().current() {
            Ok(p) => assert_eq!(p.0, "sglang", "trial {trial}"),
            Err(e) => panic!("trial {trial} current {e}"),
        }
        let start_w = expected_window_after_hosted_downs(&cfg, 0);
        let got_w = r.aimd().window();
        assert_eq!(
            got_w,
            expected_window_after_hosted_downs(&cfg, hosted.len()),
            "trial {trial} AIMD shrink"
        );
        if !hosted.is_empty() && cfg.min_window < start_w {
            assert!(got_w < start_w, "trial {trial} strictly smaller");
        }
        let req = GenerateRequest {
            prompts: vec![format!("trial-{trial}")],
            max_tokens: 2,
            temperature: 0.0,
        };
        let got = r
            .generate(req.clone(), NOW.wrapping_add(trial as u64))
            .unwrap_or_else(|e| panic!("D5 trial {trial}: {e}"));
        assert_match_reference(r.engine().config(), &req, &got);
        assert_openai_shape(&got.openai_body(), &got);
        assert_eq!(r.last_resort(), last_resort());
    }
}
