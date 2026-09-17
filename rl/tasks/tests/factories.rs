//! Group: wave-1 factory mint vs the reference. Must fail on the stub.

mod common;
mod reference;

use common::{
    assert_hex, code_row, math_row, math_row_b, src, swe_row, CODE_ID, CREATED, MATH_ID, NOW,
    SWE_ID,
};
use prometheus_tasks::{
    CodeFactory, Error, Factory, MathFactory, MintedTask, Source, SweFactory, TaskDomain,
    SCHEMA_TASK_SPEC, SCHEMA_VERSION, START_TOOL_CALLS,
};
use prometheus_verifiers::VerifierKind;

fn pin_math(sources: Vec<Source>, now: u64, created: &str) {
    let mut prod = MathFactory::new(MATH_ID, sources.clone());
    let mut refer = reference::RefFactory::math(MATH_ID, sources);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

fn pin_code(sources: Vec<Source>, now: u64, created: &str) {
    let mut prod = CodeFactory::new(CODE_ID, sources.clone());
    let mut refer = reference::RefFactory::code(CODE_ID, sources);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

fn pin_swe(sources: Vec<Source>, now: u64, created: &str) {
    let mut prod = SweFactory::new(SWE_ID, sources.clone());
    let mut refer = reference::RefFactory::swe(SWE_ID, sources);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

#[test]
fn empty_catalog_is_empty_catalog() {
    pin_math(vec![], NOW, CREATED);
    pin_code(vec![], NOW, CREATED);
    pin_swe(vec![], NOW, CREATED);
    assert_eq!(
        MathFactory::new(MATH_ID, vec![]).mint(NOW, CREATED),
        Err(Error::EmptyCatalog)
    );
}

#[test]
fn empty_body_is_empty_statement() {
    let rows = vec![src("e", "")];
    pin_math(rows.clone(), NOW, CREATED);
    pin_code(rows.clone(), NOW, CREATED);
    pin_swe(rows, NOW, CREATED);
    assert_eq!(
        MathFactory::new(MATH_ID, vec![src("e", "")]).mint(NOW, CREATED),
        Err(Error::EmptyStatement)
    );
}

#[test]
fn created_at_is_caller_string_never_now() {
    let mut f = MathFactory::new(MATH_ID, vec![math_row()]);
    let t = f.mint(NOW, CREATED).expect("mint");
    assert_eq!(t.spec.created_at, CREATED);
    assert_ne!(t.spec.created_at, NOW.to_string());
    assert!(!t.spec.created_at.contains(&NOW.to_string()));
    let mut f = MathFactory::new(MATH_ID, vec![math_row()]);
    let t = f.mint(42, "2024-01-01T00:00:00Z").expect("mint");
    assert_eq!(t.spec.created_at, "2024-01-01T00:00:00Z");
}

#[test]
fn now_is_not_encoded_in_the_task() {
    let rows = vec![math_row()];
    let mut a = MathFactory::new(MATH_ID, rows.clone());
    let mut b = MathFactory::new(MATH_ID, rows);
    let t1 = a.mint(1, CREATED).unwrap();
    let t2 = b.mint(999_999, CREATED).unwrap();
    assert_eq!(t1, t2);
}

#[test]
fn schema_id_and_version() {
    let t = MathFactory::new(MATH_ID, vec![math_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(t.spec.schema_id, SCHEMA_TASK_SPEC);
    assert_eq!(t.spec.schema_id, "prometheus.task_spec");
    assert_eq!(t.spec.schema_version, SCHEMA_VERSION);
    assert_eq!(t.spec.schema_version, 1);
    t.spec.schema_ok().unwrap();
}

#[test]
fn hashes_are_lowercase_hex_sha256() {
    let t = MathFactory::new(MATH_ID, vec![math_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_hex("statement_hash", &t.spec.statement_hash);
    assert_hex("hidden_tests_hash", &t.spec.hidden_tests_hash);
    assert_eq!(
        t.spec.statement_hash,
        reference::statement_hash(&t.statement)
    );
    let expected = t.math.as_ref().unwrap().expected.as_bytes();
    assert_eq!(t.spec.hidden_tests_hash, reference::sha256_hex(expected));
}

#[test]
fn math_factory_payload_and_horizon() {
    pin_math(vec![math_row()], NOW, CREATED);
    let t = MathFactory::new(MATH_ID, vec![math_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(t.spec.domain, TaskDomain::Math);
    assert_eq!(t.verifier_kind, VerifierKind::SymbolicMath);
    assert_eq!(t.verifier_id.0, reference::MATH_VERIFIER_ID);
    assert_eq!(t.spec.verifier_id, reference::MATH_VERIFIER_ID);
    assert_eq!(t.statement, math_row().body);
    assert!(t.math.is_some());
    assert!(t.code.is_none());
    let expected = &t.math.as_ref().unwrap().expected;
    assert!(!expected.is_empty());
    assert_eq!(*expected, reference::math_expected(&math_row().body));
    assert_ne!(*expected, t.statement);
    assert_eq!(t.spec.max_tool_calls, START_TOOL_CALLS);
    assert_eq!(t.spec.max_tool_calls, reference::MATH_MAX_TOOL_CALLS);
    assert_eq!(t.spec.horizon_s, reference::MATH_HORIZON_S);
    assert_eq!(t.spec.env_image, None);
    assert_eq!(t.spec.task_id, reference::task_id(MATH_ID, &math_row().id));
    assert_eq!(t.spec.provenance.source, math_row().provenance.source);
}

#[test]
fn code_factory_payload_and_horizon() {
    pin_code(vec![code_row()], NOW, CREATED);
    let t = CodeFactory::new(CODE_ID, vec![code_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_code_shape(&t, false);
}

#[test]
fn swe_factory_payload_and_horizon() {
    pin_swe(vec![swe_row()], NOW, CREATED);
    let t = SweFactory::new(SWE_ID, vec![swe_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_code_shape(&t, true);
    assert_eq!(t.spec.domain, TaskDomain::Code);
    assert_ne!(t.spec.domain, TaskDomain::Agent);
}

fn assert_code_shape(t: &MintedTask, swe: bool) {
    assert_eq!(t.spec.domain, TaskDomain::Code);
    assert_eq!(t.verifier_kind, VerifierKind::SandboxedTests);
    assert_eq!(t.verifier_id.0, reference::CODE_VERIFIER_ID);
    assert!(t.math.is_none());
    let code = t.code.as_ref().expect("code payload");
    assert!(
        !code.image.hidden_tests.is_empty(),
        "at least one hidden test"
    );
    assert!(!code.mutants.is_empty(), "at least one mutant");
    let stdout = String::from_utf8_lossy(&code.run.expected_stdout);
    assert!(
        !t.statement.contains(stdout.as_ref()),
        "public statement must not contain hidden expected stdout"
    );
    assert!(t.spec.env_image.is_some(), "env_image set");
    assert_eq!(t.spec.env_image.as_ref(), Some(&code.image.id.0));
    assert_hex("image.hidden_tests_hash", &code.image.hidden_tests_hash);
    assert_eq!(t.spec.hidden_tests_hash, code.image.hidden_tests_hash);
    assert_eq!(
        code.image.hidden_tests_hash,
        reference::hidden_files_hash(&code.image.hidden_tests)
    );
    if swe {
        assert_eq!(t.spec.max_tool_calls, reference::SWE_MAX_TOOL_CALLS);
        assert_eq!(t.spec.horizon_s, reference::SWE_HORIZON_S);
        assert_eq!(t.spec.max_tool_calls, START_TOOL_CALLS * 2);
        assert_eq!(code.mutants[0].id, reference::SWE_MUTANT_ID);
    } else {
        assert_eq!(t.spec.max_tool_calls, reference::CODE_MAX_TOOL_CALLS);
        assert_eq!(t.spec.horizon_s, reference::CODE_HORIZON_S);
        assert_eq!(t.spec.max_tool_calls, START_TOOL_CALLS);
        assert_eq!(code.mutants[0].id, reference::CODE_MUTANT_ID);
    }
}

#[test]
fn swe_is_strictly_longer_than_code_on_the_same_row() {
    let row = src("same", "shared body");
    let code = CodeFactory::new(CODE_ID, vec![row.clone()])
        .mint(NOW, CREATED)
        .unwrap();
    let swe = SweFactory::new(SWE_ID, vec![row])
        .mint(NOW, CREATED)
        .unwrap();
    assert!(
        swe.spec.max_tool_calls >= 2 * code.spec.max_tool_calls,
        "SWE max_tool_calls {} vs code {}",
        swe.spec.max_tool_calls,
        code.spec.max_tool_calls
    );
    assert!(
        swe.spec.horizon_s >= 2 * code.spec.horizon_s,
        "SWE horizon_s {} vs code {}",
        swe.spec.horizon_s,
        code.spec.horizon_s
    );
    assert_eq!(code.spec.max_tool_calls, 10);
    assert_eq!(swe.spec.max_tool_calls, 20);
    assert_eq!(code.spec.horizon_s, 30);
    assert_eq!(swe.spec.horizon_s, 60);
}

#[test]
fn provenance_source_comes_from_the_catalog_row() {
    let row = Source::new("id1", "gsm8k", "body");
    let t = MathFactory::new(MATH_ID, vec![row.clone()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(t.spec.provenance, row.provenance);
    assert_eq!(t.spec.provenance.source, "gsm8k");
}

#[test]
fn task_id_stable_given_source_id() {
    let rows = vec![math_row()];
    let mut f = MathFactory::new(MATH_ID, rows);
    let a = f.mint(NOW, CREATED).unwrap();
    let b = f.mint(NOW, CREATED).unwrap();
    assert_eq!(a.spec.task_id, b.spec.task_id);
    assert_eq!(a.spec.task_id, format!("{MATH_ID}:{}", math_row().id));
}

#[test]
fn one_row_catalog_two_mints_same_task_id() {
    pin_math(vec![math_row()], NOW, CREATED);
    let mut f = MathFactory::new(MATH_ID, vec![math_row()]);
    let a = f.mint(1, "t0").unwrap();
    let b = f.mint(2, "t1").unwrap();
    assert_eq!(a.spec.task_id, b.spec.task_id);
    assert_eq!(a.spec.created_at, "t0");
    assert_eq!(b.spec.created_at, "t1");
}

#[test]
fn round_robin_wraps() {
    let rows = vec![math_row(), math_row_b()];
    let mut prod = MathFactory::new(MATH_ID, rows.clone());
    let mut refer = reference::RefFactory::math(MATH_ID, rows.clone());
    for k in 0..5 {
        let got = prod.mint(NOW, CREATED);
        let want = refer.mint(NOW, CREATED);
        assert_eq!(got, want, "mint {k}");
    }
    let mut f = MathFactory::new(MATH_ID, rows);
    let a = f.mint(NOW, CREATED).unwrap();
    let b = f.mint(NOW, CREATED).unwrap();
    let c = f.mint(NOW, CREATED).unwrap();
    assert_eq!(a.spec.task_id, format!("{MATH_ID}:{}", math_row().id));
    assert_eq!(b.spec.task_id, format!("{MATH_ID}:{}", math_row_b().id));
    assert_eq!(c.spec.task_id, a.spec.task_id);
    assert_ne!(a.spec.task_id, b.spec.task_id);
}

#[test]
fn empty_statement_does_not_advance_cursor() {
    let rows = vec![src("bad", ""), math_row()];
    let mut f = MathFactory::new(MATH_ID, rows.clone());
    assert_eq!(f.mint(NOW, CREATED), Err(Error::EmptyStatement));
    assert_eq!(f.mint(NOW, CREATED), Err(Error::EmptyStatement));
    let mut refer = reference::RefFactory::math(MATH_ID, rows);
    assert_eq!(refer.mint(NOW, CREATED), Err(Error::EmptyStatement));
}

#[test]
fn sources_order_stable_across_mint() {
    let rows = vec![math_row(), math_row_b()];
    let mut f = MathFactory::new(MATH_ID, rows.clone());
    let _ = f.mint(NOW, CREATED).unwrap();
    assert_eq!(f.sources(), rows.as_slice());
}

#[test]
fn factory_trait_mint_matches_inherent() {
    let mut inherent = MathFactory::new(MATH_ID, vec![math_row()]);
    let mut as_trait: Box<dyn Factory> = Box::new(MathFactory::new(MATH_ID, vec![math_row()]));
    assert_eq!(
        inherent.mint(NOW, CREATED).unwrap(),
        as_trait.mint(NOW, CREATED).unwrap()
    );
}

#[test]
fn code_and_swe_round_robin_matches_reference() {
    let rows = vec![code_row(), src("h2", "second prompt")];
    let mut prod = CodeFactory::new(CODE_ID, rows.clone());
    let mut refer = reference::RefFactory::code(CODE_ID, rows);
    for _ in 0..3 {
        assert_eq!(prod.mint(NOW, CREATED), refer.mint(NOW, CREATED));
    }
    let rows = vec![swe_row(), src("s2", "repo@def")];
    let mut prod = SweFactory::new(SWE_ID, rows.clone());
    let mut refer = reference::RefFactory::swe(SWE_ID, rows);
    for _ in 0..3 {
        assert_eq!(prod.mint(NOW, CREATED), refer.mint(NOW, CREATED));
    }
}

#[test]
fn math_expected_hidden_from_statement() {
    let t = MathFactory::new(MATH_ID, vec![math_row()])
        .mint(NOW, CREATED)
        .unwrap();
    let expected = &t.math.as_ref().unwrap().expected;
    assert!(!t.statement.contains(expected));
}
