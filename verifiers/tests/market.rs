//! Group: MarketResolution relative log score, clamp, planted extremes, gradients.

mod common;
mod reference;

use common::{assert_meta, reward_request, SCORED_AT};
use prometheus_verifiers::{Answer, Error, MarketResolution, MarketTask, PROB_EPS};

fn checker(market_p: f64, outcome: bool) -> MarketResolution {
    MarketResolution::new("market-log", MarketTask::new(market_p, outcome))
}

fn pin_score(model_p: f64, market_p: f64, outcome: bool) {
    let want = reference::relative_log_score(model_p, market_p, outcome);
    let got = MarketResolution::relative_log_score(model_p, market_p, outcome);
    match (want, got) {
        (Ok(a), Ok(b)) => assert!(
            (a - b).abs() < 1e-12,
            "score {model_p} {market_p} {outcome}: {a} vs {b}"
        ),
        (Err(Error::BadProbability(_)), Err(Error::BadProbability(_))) => {}
        (w, g) => panic!("ref {w:?} prod {g:?}"),
    }
}

#[test]
fn relative_log_score_matches_reference() {
    pin_score(0.9, 0.5, true);
    pin_score(0.1, 0.5, true);
    pin_score(0.9, 0.5, false);
    pin_score(0.0, 0.5, true);
    pin_score(1.0, 0.5, false);
    pin_score(0.5, 0.5, true);
    pin_score(0.2, 0.8, false);
}

#[test]
fn equal_to_market_is_zero_and_does_not_pass() {
    let s = MarketResolution::relative_log_score(0.4, 0.4, true).unwrap();
    assert!((s - 0.0).abs() < 1e-12);
    let v = checker(0.4, true);
    let req = reward_request("market-log");
    let got = v
        .verify(&req, &Answer::Forecast { p: 0.4 }, SCORED_AT, 0)
        .unwrap();
    assert!(!got.passed);
    assert!((got.score - 0.0).abs() < 1e-12);
}

#[test]
fn beats_market_passes_with_positive_score() {
    let v = checker(0.2, true);
    let req = reward_request("market-log");
    let ans = Answer::Forecast { p: 0.8 };
    let got = v.verify(&req, &ans, SCORED_AT, common::NOW_MS).unwrap();
    let exp = reference::verify_market(&v.task, &req, &ans, SCORED_AT).unwrap();
    assert_meta(&got, &req, SCORED_AT);
    assert!(got.passed);
    assert!(got.score > 0.0);
    assert!((got.score - exp.score).abs() < 1e-12);
}

#[test]
fn planted_p_zero_when_true_does_not_pass() {
    let v = checker(0.5, true);
    let req = reward_request("market-log");
    let got = v
        .verify(&req, &Answer::Forecast { p: 0.0 }, SCORED_AT, 0)
        .unwrap();
    assert!(!got.passed);
    assert!(got.score < 0.0);
    let s = MarketResolution::relative_log_score(0.0, 0.5, true).unwrap();
    assert!((s - (PROB_EPS.ln() - 0.5_f64.ln())).abs() < 1e-12);
}

#[test]
fn planted_p_one_when_false_does_not_pass() {
    let v = checker(0.5, false);
    let req = reward_request("market-log");
    let got = v
        .verify(&req, &Answer::Forecast { p: 1.0 }, SCORED_AT, 0)
        .unwrap();
    assert!(!got.passed);
    assert!(got.score < 0.0);
}

#[test]
fn p_outside_unit_interval_is_bad_probability() {
    let v = checker(0.5, true);
    let req = reward_request("market-log");
    match v
        .verify(&req, &Answer::Forecast { p: 1.5 }, SCORED_AT, 0)
        .unwrap_err()
    {
        Error::BadProbability(p) => assert_eq!(p, 1.5),
        other => panic!("{other:?}"),
    }
    match MarketResolution::relative_log_score(-0.01, 0.5, true) {
        Err(Error::BadProbability(p)) => assert_eq!(p, -0.01),
        other => panic!("{other:?}"),
    }
    match MarketResolution::relative_log_score(0.5, 2.0, true) {
        Err(Error::BadProbability(p)) => assert_eq!(p, 2.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn outcome_false_uses_one_minus_p() {
    let a = MarketResolution::relative_log_score(0.8, 0.3, true).unwrap();
    let b = MarketResolution::relative_log_score(0.2, 0.7, false).unwrap();
    assert!((a - b).abs() < 1e-12);
    assert_eq!(a, reference::relative_log_score(0.8, 0.3, true).unwrap());
}

#[test]
fn finite_difference_gradient_matches_one_over_p() {
    let p = 0.3_f64;
    let m = 0.5_f64;
    let h = 1e-8;
    let s1 = MarketResolution::relative_log_score(p + h, m, true).unwrap();
    let s0 = MarketResolution::relative_log_score(p - h, m, true).unwrap();
    let numeric = (s1 - s0) / (2.0 * h);
    let analytic = 1.0 / p;
    assert!(
        (numeric - analytic).abs() < 1e-5,
        "numeric {numeric} analytic {analytic}"
    );
    let s1f = MarketResolution::relative_log_score(p + h, m, false).unwrap();
    let s0f = MarketResolution::relative_log_score(p - h, m, false).unwrap();
    let numeric_f = (s1f - s0f) / (2.0 * h);
    let analytic_f = -1.0 / (1.0 - p);
    assert!((numeric_f - analytic_f).abs() < 1e-5);
}

#[test]
fn clamp_uses_prob_eps_at_extremes() {
    let s = MarketResolution::relative_log_score(0.0, 0.0, true).unwrap();
    assert!((s - 0.0).abs() < 1e-12);
    let s = MarketResolution::relative_log_score(1.0, 1.0, true).unwrap();
    assert!((s - 0.0).abs() < 1e-12);
}

#[test]
fn wrong_kind() {
    let v = checker(0.5, true);
    let req = reward_request("market-log");
    let err = v
        .verify(
            &req,
            &Answer::Math {
                latex: "0.5".into(),
            },
            SCORED_AT,
            0,
        )
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);
}

#[test]
fn scored_at_is_caller_string() {
    let v = checker(0.9, true);
    let req = reward_request("market-log");
    let got = v
        .verify(&req, &Answer::Forecast { p: 0.95 }, "caller-ts", 123)
        .unwrap();
    assert_eq!(got.scored_at, "caller-ts");
    assert_eq!(got.request_id, req.request_id);
}
