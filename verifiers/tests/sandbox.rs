//! Group: SandboxedTests + mutation against the D1 InProcess python subset.

mod common;
mod reference;

use std::collections::BTreeMap;

use common::{
    answer_ok, answer_wrong, assert_meta, code_task, cpu_pool, evidence_kinds, has_kind,
    reward_request, strong_mutant, weak_mutant, CpuRegistry, HIDDEN_TEST, NOW_MS, SCORED_AT,
};
use prometheus_verifiers::{Answer, AnyVerifier, Error, EvidenceKind, Mutant, SandboxedTests};

fn code_checker(
    image_id: &str,
    mutants: Vec<Mutant>,
) -> SandboxedTests<prometheus_envs::InProcess> {
    SandboxedTests::new("code-py", code_task(image_id, mutants), cpu_pool())
}

#[test]
fn correct_code_passes_with_hidden_and_mutation_evidence() {
    let mut v = code_checker("img-ok", vec![strong_mutant()]);
    let req = reward_request("code-py");
    let ans = answer_ok();
    let got = v.verify(&req, &ans, SCORED_AT, NOW_MS).unwrap();
    let mut pool = cpu_pool();
    let exp = reference::verify_code(&mut pool, &v.task, &req, &ans, SCORED_AT, NOW_MS).unwrap();
    assert_meta(&got, &req, SCORED_AT);
    assert!(got.passed);
    assert_eq!(got.score, 1.0);
    assert!(has_kind(&got.evidence, EvidenceKind::HiddenTests));
    assert!(has_kind(&got.evidence, EvidenceKind::Mutation));
    assert_eq!(got.passed, exp.passed);
    assert_eq!(got.score, exp.score);
    assert_eq!(
        evidence_kinds(&got.evidence)
            .into_iter()
            .filter(|k| *k == EvidenceKind::HiddenTests || *k == EvidenceKind::Mutation)
            .count(),
        2
    );
}

#[test]
fn planted_wrong_answer_rejected() {
    let mut v = code_checker("img-wrong", vec![strong_mutant()]);
    let req = reward_request("code-py");
    let ans = answer_wrong();
    let got = v.verify(&req, &ans, SCORED_AT, NOW_MS).unwrap();
    assert!(!got.passed);
    assert_eq!(got.score, 0.0);
    assert!(has_kind(&got.evidence, EvidenceKind::HiddenTests));
}

#[test]
fn weak_mutant_that_still_prints_ok_is_weak_tests() {
    let mut v = code_checker("img-weak", vec![weak_mutant()]);
    let req = reward_request("code-py");
    let err = v.verify(&req, &answer_ok(), SCORED_AT, NOW_MS).unwrap_err();
    assert_eq!(
        err,
        Error::WeakTests {
            id: "still-ok".into()
        }
    );
    let mut pool = cpu_pool();
    let exp = reference::verify_code(&mut pool, &v.task, &req, &answer_ok(), SCORED_AT, NOW_MS)
        .unwrap_err();
    assert_eq!(err, exp);
}

#[test]
fn no_mutants_cannot_score_one() {
    let mut v = code_checker("img-nomut", vec![]);
    match v
        .verify(&req_code(), &answer_ok(), SCORED_AT, NOW_MS)
        .unwrap_err()
    {
        Error::Unverifiable(s) => assert!(s.contains("mutant"), "{s}"),
        other => panic!("{other:?}"),
    }
}

fn req_code() -> prometheus_verifiers::RewardRequest {
    reward_request("code-py")
}

#[test]
fn test_file_write_to_grader_errors() {
    let mut v = code_checker("img-write", vec![strong_mutant()]);
    let req = reward_request("code-py");
    let ans = Answer::Code {
        files: BTreeMap::from([(HIDDEN_TEST.into(), b"hacked".to_vec())]),
    };
    match v.verify(&req, &ans, SCORED_AT, NOW_MS).unwrap_err() {
        Error::TestFileWrite { path } => assert_eq!(path, HIDDEN_TEST),
        other => panic!("{other:?}"),
    }
}

#[test]
fn hidden_tests_in_agent_view_error() {
    let mut v = code_checker("img-leak", vec![strong_mutant()]);
    v.task
        .image
        .agent_files
        .insert(HIDDEN_TEST.into(), b"leaked".to_vec());
    let req = reward_request("code-py");
    match v.verify(&req, &answer_ok(), SCORED_AT, NOW_MS).unwrap_err() {
        Error::HiddenTestsVisible => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn registry_unknown_id_does_not_run_code() {
    let mut reg = CpuRegistry::new();
    let req = reward_request("missing-code");
    let err = reg
        .verify(&req, &answer_ok(), SCORED_AT, NOW_MS)
        .unwrap_err();
    assert_eq!(err, Error::UnknownVerifier("missing-code".into()));
}

#[test]
fn registry_schema_mismatch_on_code_request() {
    let mut reg = CpuRegistry::new();
    reg.insert(AnyVerifier::Code(Box::new(code_checker(
        "img-reg",
        vec![strong_mutant()],
    ))))
    .unwrap();
    let mut req = reward_request("code-py");
    req.schema_id = "nope".into();
    match reg.verify(&req, &answer_ok(), SCORED_AT, NOW_MS) {
        Err(Error::Schema(_)) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn wrong_kind_on_sandboxed_tests() {
    let mut v = code_checker("img-kind", vec![strong_mutant()]);
    let req = reward_request("code-py");
    let err = v
        .verify(
            &req,
            &Answer::Math {
                latex: "print(1)".into(),
            },
            SCORED_AT,
            NOW_MS,
        )
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);
}

#[test]
fn production_matches_reference_on_ok_and_wrong() {
    let mut v = code_checker("img-ref", vec![strong_mutant()]);
    let req = reward_request("code-py");
    let mut pool = cpu_pool();
    let ok_ref =
        reference::verify_code(&mut pool, &v.task, &req, &answer_ok(), SCORED_AT, NOW_MS).unwrap();
    assert!(ok_ref.passed);
    assert_eq!(ok_ref.score, 1.0);
    let bad_ref =
        reference::verify_code(&mut pool, &v.task, &req, &answer_wrong(), SCORED_AT, NOW_MS)
            .unwrap();
    assert!(!bad_ref.passed);
    assert_eq!(bad_ref.score, 0.0);

    let ok = v.verify(&req, &answer_ok(), SCORED_AT, NOW_MS).unwrap();
    assert_eq!(ok.passed, ok_ref.passed);
    assert_eq!(ok.score, ok_ref.score);
    let bad = v.verify(&req, &answer_wrong(), SCORED_AT, NOW_MS).unwrap();
    assert_eq!(bad.passed, bad_ref.passed);
    assert_eq!(bad.score, bad_ref.score);
}
