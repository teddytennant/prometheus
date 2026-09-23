//! Group: production A7 surface equals the independent reference.

mod reference;

use prometheus_control::rung::{
    rung_0_spec, rung_spec, validate_rung_config, RungConfig, RungError, RungId, RungRun, RungSpec,
};
use reference::rung::{self as ref_rung, RefRungRun};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn bounded(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next() % n
    }
}

fn assert_runs_match(prod: &RungRun, refer: &RefRungRun) {
    assert_eq!(prod.done(), refer.done());
    assert_eq!(prod.step_index(), refer.step_index());
    assert_eq!(prod.tokens_seen(), refer.tokens_seen());
    assert_eq!(prod.remaining_tokens(), refer.remaining_tokens());
    assert_eq!(prod.loss_curve(), refer.loss_curve());
    assert_eq!(prod.config(), refer.config());
    assert_eq!(prod.checkpoint(), refer.checkpoint());
}

fn bits_to_loss(bits: u64) -> f64 {
    match bits % 7 {
        0 => 0.0,
        1 => 1.0,
        2 => -0.25,
        3 => 1e-9,
        4 => f64::NAN,
        5 => f64::INFINITY,
        _ => f64::NEG_INFINITY,
    }
}

#[test]
fn every_rung_spec_matches_reference() {
    for id in [RungId::Zero, RungId::One, RungId::Two, RungId::Three] {
        assert_eq!(rung_spec(id), ref_rung::rung_spec(id));
    }
    assert_eq!(rung_0_spec(), ref_rung::rung_0_spec());
    assert_eq!(rung_0_spec(), rung_spec(RungId::Zero));
}

#[test]
fn validate_matches_reference_on_random_configs() {
    let ids = [RungId::Zero, RungId::One, RungId::Two, RungId::Three];
    let hashes = ["", " ", "h", "sha256:abc"];
    for trial in 0..64u64 {
        let mut rng = Lcg(0xA700_0000 + trial * 19);
        let id = ids[rng.bounded(4) as usize];
        let mut spec = ref_rung::rung_spec(id);
        if rng.bounded(4) == 0 {
            spec.tokens = rng.bounded(32);
        }
        let cfg = RungConfig {
            spec,
            token_budget: rng.bounded(40),
            tokens_per_step: rng.bounded(8),
            tokenizer_hash: hashes[rng.bounded(4) as usize].to_string(),
            seed: rng.next(),
        };
        assert_eq!(
            validate_rung_config(&cfg),
            ref_rung::validate_rung_config(&cfg),
            "cfg={cfg:?}"
        );
        assert_eq!(
            RungRun::new(cfg.clone()).map(|_| ()),
            RefRungRun::new(cfg).map(|_| ()),
        );
    }
}

#[test]
fn step_checkpoint_resume_match_reference_random_walk() {
    for trial in 0..24u64 {
        let mut rng = Lcg(0xA7A7_0000 + trial * 17);
        let token_budget = 1 + rng.bounded(24);
        let tokens_per_step = 1 + rng.bounded(12);
        let cfg = RungConfig {
            spec: ref_rung::rung_0_spec(),
            token_budget,
            tokens_per_step,
            tokenizer_hash: format!("tok-{trial}"),
            seed: rng.next(),
        };
        let mut prod = RungRun::new(cfg.clone()).expect("prod new");
        let mut refer = RefRungRun::new(cfg.clone()).expect("ref new");
        assert_runs_match(&prod, &refer);

        let mut last_ckpt = prod.checkpoint();
        for _ in 0..20 {
            match rng.bounded(5) {
                0..=2 => {
                    let loss = bits_to_loss(rng.next());
                    match (prod.step(loss), refer.step(loss)) {
                        (Ok(a), Ok(b)) => {
                            assert_eq!(a, b);
                            let _: f64 = a.loss;
                            let _: u64 = a.tokens_seen;
                        }
                        (Err(a), Err(b)) => assert_eq!(a, b),
                        other => panic!("step mismatch {other:?} loss={loss}"),
                    }
                }
                3 => {
                    last_ckpt = prod.checkpoint();
                    assert_eq!(last_ckpt, refer.checkpoint());
                    assert_eq!(prod.tokens_seen(), refer.tokens_seen());
                }
                _ => match (
                    RungRun::resume(cfg.clone(), &last_ckpt),
                    RefRungRun::resume(cfg.clone(), &last_ckpt),
                ) {
                    (Ok(p), Ok(r)) => {
                        prod = p;
                        refer = r;
                    }
                    (Err(a), Err(b)) => assert_eq!(a, b),
                    other => panic!("resume mismatch {other:?}"),
                },
            }
            assert_runs_match(&prod, &refer);
            assert!(prod.tokens_seen() <= token_budget);
            assert_eq!(
                prod.remaining_tokens(),
                token_budget.saturating_sub(prod.tokens_seen())
            );
            if prod.done() {
                assert_eq!(prod.tokens_seen(), token_budget);
            }
        }
    }
}

#[test]
fn resume_faults_match_reference() {
    let cfg = ref_rung::tiny_rung0_config(10, 2);
    let base = prometheus_control::rung::RungCheckpoint {
        step: 2,
        tokens_seen: 4,
        loss_curve: vec![1.0, 0.5],
        tokenizer_hash: cfg.tokenizer_hash.clone(),
        seed: cfg.seed,
        token_budget: cfg.token_budget,
        tokens_per_step: cfg.tokens_per_step,
        rung: RungId::Zero,
    };

    let mut empty_hash = cfg.clone();
    empty_hash.tokenizer_hash.clear();
    assert_eq!(
        RungRun::resume(empty_hash.clone(), &base).map(|_| ()),
        RefRungRun::resume(empty_hash, &base).map(|_| ()),
    );

    let mut changed = base.clone();
    changed.tokenizer_hash = "nope".into();
    assert_eq!(
        RungRun::resume(cfg.clone(), &changed).unwrap_err(),
        RungError::TokenizerChanged
    );
    assert_eq!(
        RefRungRun::resume(cfg.clone(), &changed).unwrap_err(),
        RungError::TokenizerChanged
    );

    let mut past = base.clone();
    past.tokens_seen = 11;
    assert_eq!(
        RungRun::resume(cfg.clone(), &past).unwrap_err(),
        RefRungRun::resume(cfg.clone(), &past).unwrap_err()
    );

    let mut seed = base;
    seed.seed = 0;
    assert_eq!(
        RungRun::resume(cfg.clone(), &seed).unwrap_err(),
        RungError::CheckpointMismatch
    );
    assert_eq!(
        RefRungRun::resume(cfg, &seed).unwrap_err(),
        RungError::CheckpointMismatch
    );
}

#[test]
fn u64_counts_never_exceed_budget_on_a_valid_run() {
    let cfg = RungConfig {
        spec: RungSpec {
            id: RungId::Zero,
            active_params: 100_000_000,
            total_params: 1_000_000_000,
            tokens: 20_000_000_000,
            gpus: 64,
        },
        token_budget: 17,
        tokens_per_step: 5,
        tokenizer_hash: "h".into(),
        seed: 0,
    };
    let mut prod = RungRun::new(cfg.clone()).unwrap();
    let mut refer = RefRungRun::new(cfg).unwrap();
    loop {
        match (prod.step(0.1), refer.step(0.1)) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a, b);
                assert!(a.tokens_seen <= 17);
            }
            (Err(RungError::AlreadyDone), Err(RungError::AlreadyDone)) => break,
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(prod.tokens_seen(), 17);
    assert_eq!(prod.loss_curve().len() as u64, prod.step_index());
}
