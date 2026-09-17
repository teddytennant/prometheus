//! Group: SymbolicMath canonicalize, equivalence, planted wrong answers.

mod common;
mod reference;

use common::{assert_meta, has_kind, reward_request, SCORED_AT};
use prometheus_verifiers::{Answer, Error, EvidenceKind, MathTask, SymbolicMath};

fn pin_canon(input: &str, want: &str) {
    assert_eq!(reference::canonicalize(input), want, "reference {input:?}");
    assert_eq!(
        SymbolicMath::canonicalize(input),
        want,
        "production {input:?}"
    );
}

fn pin_eq(left: &str, right: &str, want: bool) {
    assert_eq!(
        reference::equivalent(left, right),
        want,
        "reference {left:?} ~ {right:?}"
    );
    assert_eq!(
        SymbolicMath::equivalent(left, right),
        want,
        "production {left:?} ~ {right:?}"
    );
    assert_eq!(
        reference::equivalent(right, left),
        want,
        "reference symmetry"
    );
    assert_eq!(
        SymbolicMath::equivalent(right, left),
        want,
        "production symmetry"
    );
}

#[test]
fn canonicalize_strips_whitespace_dollars_left_right() {
    pin_canon("1+2", "1+2");
    pin_canon(" 1 + 2 ", "1+2");
    pin_canon("$1+2$", "1+2");
    pin_canon(" $ 1 + 2 $ ", "1+2");
    pin_canon("\\left(1+2\\right)", "(1+2)");
    pin_canon("\\left (1+2 \\right )", "(1+2)");
    pin_canon("$\n\\left(1+2\\right)\n$", "(1+2)");
    pin_canon("\\frac{1}{2}", "\\frac{1}{2}");
    pin_canon("$\\frac{1}{2}$", "\\frac{1}{2}");
}

#[test]
fn canonicalize_is_idempotent() {
    for s in ["$1 + 2$", "\\left(1\\right)", "  a\tb\n"] {
        let once = SymbolicMath::canonicalize(s);
        assert_eq!(once, SymbolicMath::canonicalize(&once));
        assert_eq!(once, reference::canonicalize(s));
    }
}

#[test]
fn equivalent_integers_rationals_sums() {
    pin_eq("3", "3", true);
    pin_eq("1+2", "3", true);
    pin_eq(" 1 + 2 ", "3", true);
    pin_eq("$1+2$", "3", true);
    pin_eq("\\left(1+2\\right)", "3", true);
    pin_eq("1/2", "0.5", true);
    pin_eq("0.5", "\\frac{1}{2}", true);
    pin_eq("1/2", "\\frac{1}{2}", true);
    pin_eq("$1/2$", "0.50", true);
    pin_eq("2/4", "1/2", true);
    pin_eq("1+2+3", "6", true);
    pin_eq("-1+2", "1", true);
}

#[test]
fn planted_wrong_expression_rejected() {
    pin_eq("1+2", "4", false);
    pin_eq("1/2", "1/3", false);
    pin_eq("0.5", "0.6", false);
    pin_eq("3", "2", false);
    pin_eq("abc", "abd", false);
}

#[test]
fn equivalent_reflexive_on_garbage() {
    pin_eq("not-math", "not-math", true);
    pin_eq("not-math", "also-not", false);
}

#[test]
fn symbolic_verify_correct_scores_one() {
    let v = SymbolicMath::new("sympy-exact", MathTask::new("3"));
    let req = reward_request("sympy-exact");
    let ans = Answer::Math {
        latex: "$1+2$".into(),
    };
    let got = v.verify(&req, &ans, SCORED_AT, common::NOW_MS).unwrap();
    let exp = reference::verify_symbolic(&v.task, &req, &ans, SCORED_AT).unwrap();
    assert_meta(&got, &req, SCORED_AT);
    assert!(got.passed);
    assert_eq!(got.score, 1.0);
    assert!(has_kind(&got.evidence, EvidenceKind::Symbolic));
    assert_eq!(got.passed, exp.passed);
    assert_eq!(got.score, exp.score);
}

#[test]
fn symbolic_verify_planted_wrong_scores_zero() {
    let v = SymbolicMath::new("sympy-exact", MathTask::new("3"));
    let req = reward_request("sympy-exact");
    let ans = Answer::Math {
        latex: "1+3".into(),
    };
    let got = v.verify(&req, &ans, SCORED_AT, common::NOW_MS).unwrap();
    assert_meta(&got, &req, SCORED_AT);
    assert!(!got.passed);
    assert_eq!(got.score, 0.0);
}

#[test]
fn symbolic_frac_forms_pass() {
    let v = SymbolicMath::new("sympy-exact", MathTask::new(r"\frac{1}{2}"));
    let req = reward_request("sympy-exact");
    for latex in ["1/2", "0.5", r"$\frac{1}{2}$"] {
        let ans = Answer::Math {
            latex: latex.into(),
        };
        let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
        assert!(got.passed, "{latex}");
        assert_eq!(got.score, 1.0);
    }
}

#[test]
fn production_canonicalize_matches_reference_table() {
    for s in [
        "",
        " ",
        "$",
        r"\left\left",
        r"\right.",
        "3.0",
        r"\frac{1+1}{2}",
    ] {
        assert_eq!(
            SymbolicMath::canonicalize(s),
            reference::canonicalize(s),
            "{s:?}"
        );
        assert_eq!(
            SymbolicMath::equivalent(s, s),
            reference::equivalent(s, s),
            "{s:?}"
        );
    }
}

#[test]
fn wrong_kind_code_is_rejected() {
    let v = SymbolicMath::new("sympy-exact", MathTask::new("3"));
    let req = reward_request("sympy-exact");
    let err = v
        .verify(&req, &common::answer_ok(), SCORED_AT, 0)
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);
}
