//! Group: LeanCheck + TinyKernel. Pinned TinyLean language (no lean binary).

mod common;
mod reference;

use common::{assert_meta, has_kind, reward_request, SCORED_AT};
use prometheus_verifiers::{Answer, Error, EvidenceKind, Kernel, LeanCheck, LeanTask, TinyKernel};

fn pin_check(theorem: &str, proof: &str) {
    let want = reference::tiny_kernel_check(theorem, proof);
    let got = TinyKernel::new().check(theorem, proof);
    match (want, got) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{theorem:?} / {proof:?}"),
        (Err(Error::Lean(a)), Err(Error::Lean(b))) => {
            assert_eq!(a, b, "{theorem:?} / {proof:?}")
        }
        (w, g) => panic!("{theorem:?} / {proof:?}: ref {w:?} prod {g:?}"),
    }
}

#[test]
fn tiny_kernel_accepts_pinned_proofs() {
    pin_check("true", "trivial");
    pin_check("id", "fun x => x");
    pin_check(" true ", "  trivial  ");
    pin_check("id", "fun  x  =>  x");
    assert!(TinyKernel::new().check("true", "trivial").unwrap());
    assert!(TinyKernel::new().check("id", "fun x => x").unwrap());
}

#[test]
fn tiny_kernel_rejects_invalid_and_planted_wrong() {
    pin_check("true", "fun x => x");
    pin_check("id", "trivial");
    pin_check("true", "sorry");
    pin_check("id", "fun x => y");
    assert!(!TinyKernel::new().check("true", "fun x => x").unwrap());
    assert!(!TinyKernel::new().check("id", "trivial").unwrap());
}

#[test]
fn tiny_kernel_errors_on_empty_or_unknown() {
    pin_check("", "trivial");
    pin_check("true", "");
    pin_check("true", "   ");
    pin_check("and_comm", "trivial");
    match TinyKernel::new().check("foo", "trivial") {
        Err(Error::Lean(s)) => assert!(s.contains("unknown theorem"), "{s}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn lean_verify_true_trivial_passes() {
    let v = LeanCheck::new("lean-tiny", LeanTask::new("true"), TinyKernel::new());
    let req = reward_request("lean-tiny");
    let ans = Answer::Lean {
        proof: "trivial".into(),
    };
    let got = v.verify(&req, &ans, SCORED_AT, common::NOW_MS).unwrap();
    let exp = reference::verify_lean(&v.task, &req, &ans, SCORED_AT).unwrap();
    assert_meta(&got, &req, SCORED_AT);
    assert!(got.passed);
    assert_eq!(got.score, 1.0);
    assert!(has_kind(&got.evidence, EvidenceKind::Lean));
    assert_eq!(got.passed, exp.passed);
    assert_eq!(got.score, exp.score);
}

#[test]
fn lean_verify_id_fun_passes() {
    let v = LeanCheck::new("lean-tiny", LeanTask::new("id"), TinyKernel::new());
    let req = reward_request("lean-tiny");
    let ans = Answer::Lean {
        proof: "fun x => x".into(),
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(got.passed);
    assert_eq!(got.score, 1.0);
}

#[test]
fn lean_verify_planted_wrong_proof_rejected() {
    let v = LeanCheck::new("lean-tiny", LeanTask::new("true"), TinyKernel::new());
    let req = reward_request("lean-tiny");
    let ans = Answer::Lean {
        proof: "fun x => x".into(),
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(!got.passed);
    assert_eq!(got.score, 0.0);
}

#[test]
fn lean_verify_invalid_proof_rejected() {
    let v = LeanCheck::new("lean-tiny", LeanTask::new("id"), TinyKernel::new());
    let req = reward_request("lean-tiny");
    let ans = Answer::Lean {
        proof: "sorry".into(),
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(!got.passed);
    assert_eq!(got.score, 0.0);
}

#[test]
fn lean_wrong_kind() {
    let v = LeanCheck::new("lean-tiny", LeanTask::new("true"), TinyKernel::new());
    let req = reward_request("lean-tiny");
    let err = v
        .verify(
            &req,
            &Answer::Math {
                latex: "true".into(),
            },
            SCORED_AT,
            0,
        )
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);
}

#[test]
fn lean_kernel_field_is_used() {
    let v = LeanCheck::new("lean-tiny", LeanTask::new("true"), TinyKernel::new());
    assert!(v.kernel.check("true", "trivial").unwrap());
}
