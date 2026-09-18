//! Group: held-out recovers the generating law; OLS residual identities.
//!
//! Two-point recovery uses rel tol 1e-9. Tiny synthetic N, D, loss.

mod reference;

use prometheus_control::rung::RungId;
use prometheus_control::scale::{fit, predict_held_out, RungObservation};
use reference::scale as ref_scale;

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
    fn unit(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
}

fn obs(id: RungId, n: u64, d: u64, a: f64, alpha: f64) -> RungObservation {
    ref_scale::observation(id, n, d, ref_scale::law_loss(a, alpha, n, d))
}

#[test]
fn two_point_held_out_recovers_generating_law_to_1e9() {
    let a = 2.0;
    let alpha = 0.5;
    let train = [
        obs(RungId::One, 3, 5, a, alpha),
        obs(RungId::Two, 7, 5, a, alpha),
    ];
    let held = ref_scale::observation(RungId::Three, 11, 13, 1.0).spec;
    let got = predict_held_out(&train, &held).unwrap();
    let expect = ref_scale::law_loss(a, alpha, 11, 13);
    ref_scale::assert_rel_close(got, expect, 1e-9, "held-out law");
    ref_scale::assert_rel_close(
        got,
        ref_scale::predict_held_out(&train, &held).unwrap(),
        1e-12,
        "vs ref",
    );

    let fitted = fit(&train).unwrap();
    ref_scale::assert_rel_close(fitted.a(), a, 1e-9, "A");
    ref_scale::assert_rel_close(fitted.alpha(), alpha, 1e-9, "alpha");
}

#[test]
fn three_point_colinear_ols_residuals_are_numerically_zero() {
    let a = 3.0;
    let alpha = 1.0;
    let points = [
        obs(RungId::One, 1, 1, a, alpha),
        obs(RungId::Two, 2, 1, a, alpha),
        obs(RungId::Three, 4, 1, a, alpha),
    ];
    let fitted = fit(&points).unwrap();
    ref_scale::assert_rel_close(fitted.a(), a, 1e-9, "A");
    ref_scale::assert_rel_close(fitted.alpha(), alpha, 1e-9, "alpha");
    for p in &points {
        let r = ref_scale::log_residual(
            fitted.a(),
            fitted.alpha(),
            p.spec.active_params,
            p.spec.tokens,
            p.loss,
        );
        assert!(r.abs() < 1e-12, "residual {r} for {:?}", p.spec.id);
    }
}

#[test]
fn ols_matches_reference_and_beats_a_perturbed_alpha() {
    let train = [
        ref_scale::observation(RungId::One, 1, 1, 2.0),
        ref_scale::observation(RungId::Two, 2, 1, 1.0),
        ref_scale::observation(RungId::Three, 4, 1, 1.0),
    ];
    let prod = fit(&train).unwrap();
    let refer = ref_scale::fit(&train).unwrap();
    ref_scale::assert_rel_close(prod.a(), refer.a(), 1e-12, "A");
    ref_scale::assert_rel_close(prod.alpha(), refer.alpha(), 1e-12, "alpha");

    let sse = |a: f64, alpha: f64| -> f64 {
        train
            .iter()
            .map(|p| {
                let r =
                    ref_scale::log_residual(a, alpha, p.spec.active_params, p.spec.tokens, p.loss);
                r * r
            })
            .sum()
    };
    let best = sse(prod.a(), prod.alpha());
    let perturbed = sse(prod.a(), prod.alpha() + 0.05);
    assert!(
        best < perturbed,
        "OLS sse {best} should beat perturbed {perturbed}"
    );

    // Normal equation: residuals orthogonal to (ln C - mean ln C).
    let lcs: Vec<f64> = train
        .iter()
        .map(|p| ref_scale::compute_c(p.spec.active_params, p.spec.tokens).ln())
        .collect();
    let mean = lcs.iter().sum::<f64>() / (lcs.len() as f64);
    let mut dot = 0.0;
    for (p, lc) in train.iter().zip(lcs.iter()) {
        let r = ref_scale::log_residual(
            prod.a(),
            prod.alpha(),
            p.spec.active_params,
            p.spec.tokens,
            p.loss,
        );
        dot += r * (lc - mean);
    }
    assert!(dot.abs() < 1e-12, "orthogonality dot={dot}");
}

#[test]
fn random_two_point_laws_recover_within_1e9() {
    for trial in 0..32u64 {
        let mut rng = Lcg(0x1100_0000 + trial * 29);
        let a = 0.25 + 4.0 * rng.unit();
        let alpha = 0.05 + 1.5 * rng.unit();
        let n1 = 1 + rng.bounded(16);
        let d1 = 1 + rng.bounded(16);
        let mut n2 = 1 + rng.bounded(16);
        let mut d2 = 1 + rng.bounded(16);
        while ref_scale::compute_c(n1, d1) == ref_scale::compute_c(n2, d2) {
            n2 = 1 + rng.bounded(32);
            d2 = 1 + rng.bounded(32);
        }
        let n3 = 1 + rng.bounded(24);
        let d3 = 1 + rng.bounded(24);
        let train = [
            obs(RungId::One, n1, d1, a, alpha),
            obs(RungId::Two, n2, d2, a, alpha),
        ];
        let fitted = fit(&train).expect("fit");
        ref_scale::assert_rel_close(fitted.a(), a, 1e-9, &format!("trial {trial} A"));
        ref_scale::assert_rel_close(fitted.alpha(), alpha, 1e-9, &format!("trial {trial} alpha"));
        let held = ref_scale::observation(RungId::Three, n3, d3, 1.0).spec;
        if n3 == 0 || d3 == 0 {
            continue;
        }
        let got = predict_held_out(&train, &held).unwrap();
        let expect = ref_scale::law_loss(a, alpha, n3, d3);
        ref_scale::assert_rel_close(got, expect, 1e-9, &format!("trial {trial} held-out"));
        let refer = ref_scale::fit(&train).unwrap();
        ref_scale::assert_rel_close(fitted.a(), refer.a(), 1e-12, "A vs ref");
        ref_scale::assert_rel_close(fitted.alpha(), refer.alpha(), 1e-12, "alpha vs ref");
    }
}

#[test]
fn production_errors_match_reference_on_random_faults() {
    let ids = [RungId::Zero, RungId::One, RungId::Two, RungId::Three];
    for trial in 0..48u64 {
        let mut rng = Lcg(0x11E0_0000 + trial * 13);
        let n_obs = rng.bounded(4) as usize;
        let mut train = Vec::with_capacity(n_obs);
        for _ in 0..n_obs {
            let id = ids[rng.bounded(4) as usize];
            let active = rng.bounded(4);
            let tokens = rng.bounded(4);
            let loss = match rng.bounded(6) {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => 0.0,
                3 => -0.5,
                4 => 1e-3,
                _ => 1.0 + rng.unit(),
            };
            train.push(ref_scale::observation(id, active, tokens, loss));
        }
        assert_eq!(
            fit(&train).map(|_| ()),
            ref_scale::fit(&train).map(|_| ()),
            "trial {trial} train={train:?}"
        );
        let held_id = ids[rng.bounded(4) as usize];
        let held =
            ref_scale::observation(held_id, 1 + rng.bounded(8), 1 + rng.bounded(8), 1.0).spec;
        assert_eq!(
            predict_held_out(&train, &held).map(|_| ()),
            ref_scale::predict_held_out(&train, &held).map(|_| ()),
            "trial {trial} held={held:?}"
        );
    }
}

#[test]
fn predict_is_monotone_in_c_when_alpha_positive() {
    let train = [
        obs(RungId::One, 1, 1, 3.0, 1.0),
        obs(RungId::Two, 2, 1, 3.0, 1.0),
    ];
    let fitted = fit(&train).unwrap();
    let p_small = fitted.predict(1, 1).unwrap();
    let p_mid = fitted.predict(2, 1).unwrap();
    let p_big = fitted.predict(4, 1).unwrap();
    assert!(
        p_small > p_mid && p_mid > p_big,
        "{p_small} {p_mid} {p_big}"
    );
}
