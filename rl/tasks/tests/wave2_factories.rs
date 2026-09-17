//! Group: I6 wave-2 factory mint vs the independent reference.
//! Must fail on the stub (`unimplemented!("I6 ...::mint")`).

mod common;
mod reference;

use common::*;
use prometheus_tasks::{
    ArcFactory, Error, Factory, ForecastFactory, LongHorizonFactory, OpenEndedFactory,
    ResearchFactory, ResearchKind, START_TOOL_CALLS,
};
use prometheus_verifiers::VerifierKind;

fn pin_research(
    rows: Vec<prometheus_tasks::ResearchSource>,
    now: prometheus_envs::NowMs,
    created: &str,
) {
    let mut prod = ResearchFactory::new(RESEARCH_ID, rows.clone());
    let mut refer = reference::RefResearch::from_prod(&prod);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

fn pin_lh(
    rows: Vec<prometheus_tasks::LongHorizonSource>,
    now: prometheus_envs::NowMs,
    created: &str,
) {
    let mut prod = LongHorizonFactory::new(LONG_HORIZON_ID, rows.clone());
    let mut refer = reference::RefLongHorizon::from_prod(&prod);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

fn pin_arc(rows: Vec<prometheus_tasks::ArcSource>, now: prometheus_envs::NowMs, created: &str) {
    let mut prod = ArcFactory::new(ARC_ID, rows.clone());
    let mut refer = reference::RefArc::from_prod(&prod);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

fn pin_forecast(
    rows: Vec<prometheus_tasks::ForecastSource>,
    now: prometheus_envs::NowMs,
    created: &str,
) {
    let mut prod = ForecastFactory::new(FORECAST_ID, rows.clone());
    let mut refer = reference::RefForecast::from_prod(&prod);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

fn pin_oe(
    rows: Vec<prometheus_tasks::OpenEndedSource>,
    now: prometheus_envs::NowMs,
    created: &str,
) {
    let mut prod = OpenEndedFactory::new(OPEN_ENDED_ID, rows.clone());
    let mut refer = reference::RefOpenEnded::from_prod(&prod);
    assert_eq!(prod.mint(now, created), refer.mint(now, created));
}

#[test]
fn research_mint_matches_reference() {
    pin_research(vec![research_kaggle_row()], NOW, CREATED);
}

#[test]
fn long_horizon_mint_matches_reference() {
    pin_lh(vec![lh_row()], NOW, CREATED);
}

#[test]
fn arc_mint_matches_reference() {
    pin_arc(vec![arc_row()], NOW, CREATED);
}

#[test]
fn forecast_mint_matches_reference() {
    pin_forecast(vec![forecast_row()], NOW, CREATED);
}

#[test]
fn open_ended_mint_matches_reference() {
    pin_oe(vec![oe_row()], NOW, CREATED);
}

#[test]
fn task_id_is_factory_id_colon_source_id() {
    let t = ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(
        t.spec.task_id,
        format!("{RESEARCH_ID}:{}", research_kaggle_row().source.id)
    );
    let t = ArcFactory::new(ARC_ID, vec![arc_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(t.spec.task_id, format!("{ARC_ID}:{}", arc_row().source.id));
    let t = ForecastFactory::new(FORECAST_ID, vec![forecast_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(
        t.spec.task_id,
        format!("{FORECAST_ID}:{}", forecast_row().source.id)
    );
    let t = LongHorizonFactory::new(LONG_HORIZON_ID, vec![lh_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(
        t.spec.task_id,
        format!("{LONG_HORIZON_ID}:{}", lh_row().source.id)
    );
    let t = OpenEndedFactory::new(OPEN_ENDED_ID, vec![oe_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(
        t.spec.task_id,
        format!("{OPEN_ENDED_ID}:{}", oe_row().source.id)
    );
}

#[test]
fn created_at_is_the_caller_string_now_unused() {
    let rows = vec![arc_row()];
    let a = ArcFactory::new(ARC_ID, rows.clone())
        .mint(1, "caller-stamp")
        .unwrap();
    let b = ArcFactory::new(ARC_ID, rows)
        .mint(9_999_999_999, "caller-stamp")
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(a.spec.created_at, "caller-stamp");
    assert_ne!(a.spec.created_at, NOW.to_string());
    pin_arc(vec![arc_row()], 1, "caller-stamp");
    pin_research(vec![research_kaggle_row()], 2, "caller-stamp");
    pin_forecast(vec![forecast_row()], 3, "caller-stamp");
    pin_lh(vec![lh_row()], 4, "caller-stamp");
    pin_oe(vec![oe_row()], 5, "caller-stamp");
}

#[test]
fn domains_are_pinned() {
    let r = ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(r.spec.domain, prometheus_tasks::TaskDomain::Science);
    let h = LongHorizonFactory::new(LONG_HORIZON_ID, vec![lh_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(h.spec.domain, prometheus_tasks::TaskDomain::Agent);
    let a = ArcFactory::new(ARC_ID, vec![arc_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(a.spec.domain, prometheus_tasks::TaskDomain::Arc);
    let f = ForecastFactory::new(FORECAST_ID, vec![forecast_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(f.spec.domain, prometheus_tasks::TaskDomain::Other);
    let o = OpenEndedFactory::new(OPEN_ENDED_ID, vec![oe_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(o.spec.domain, prometheus_tasks::TaskDomain::Other);
}

#[test]
fn verifier_kinds_and_payloads() {
    let a = ArcFactory::new(ARC_ID, vec![arc_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(a.verifier_kind, VerifierKind::GridMatch);
    assert_eq!(a.verifier_id.0, reference::ARC_VERIFIER_ID);
    assert!(a.grid.is_some());
    assert!(a.math.is_none() && a.code.is_none() && a.market.is_none());
    assert_eq!(a.grid.as_ref().unwrap().expected, arc_row().expected);

    let f = ForecastFactory::new(FORECAST_ID, vec![forecast_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(f.verifier_kind, VerifierKind::MarketResolution);
    assert_eq!(f.verifier_id.0, reference::FORECAST_VERIFIER_ID);
    assert!(f.market.is_some());
    assert!(f.grid.is_none());
    assert_eq!(f.market.as_ref().unwrap().outcome, true);
    assert_eq!(f.market.as_ref().unwrap().market_p, 0.4);

    let r = ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(r.verifier_kind, VerifierKind::LeanKernel);
    assert_eq!(r.verifier_id.0, reference::RESEARCH_VERIFIER_ID);
    let rt = r.research.as_ref().expect("ResearchTask");
    assert_eq!(rt.kind, ResearchKind::Kaggle);
    assert_eq!(rt.target, 0.85);
    assert!(r.grid.is_none() && r.market.is_none() && r.math.is_none());

    let h = LongHorizonFactory::new(LONG_HORIZON_ID, vec![lh_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(h.verifier_kind, VerifierKind::LeanKernel);
    assert_eq!(h.verifier_id.0, reference::LONG_HORIZON_VERIFIER_ID);
    let lh = h.long_horizon.as_ref().expect("LongHorizonTask");
    assert_eq!(lh.checkpoints.len(), 2);
    assert!(lh.final_math.is_some());
    assert!(h.grid.is_none() && h.market.is_none());

    let o = OpenEndedFactory::new(OPEN_ENDED_ID, vec![oe_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(o.verifier_kind, VerifierKind::LeanKernel);
    assert_eq!(o.verifier_id.0, reference::OPEN_ENDED_VERIFIER_ID);
    assert!(o.open_ended.is_some());
    assert!(o.math.is_none() && o.code.is_none() && o.grid.is_none() && o.market.is_none());
    assert!(o.research.is_none() && o.long_horizon.is_none());
}

#[test]
fn research_kinds_speedrun_paper_kaggle() {
    let k = ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap();
    let kt = k.research.unwrap();
    assert_eq!(kt.kind, ResearchKind::Kaggle);
    assert_eq!(kt.budget_gpu_minutes, None);
    assert!(kt.rubric.is_none());

    let s = ResearchFactory::new(RESEARCH_ID, vec![research_speedrun_row()])
        .mint(NOW, CREATED)
        .unwrap();
    let st = s.research.unwrap();
    assert_eq!(st.kind, ResearchKind::Speedrun);
    assert_eq!(st.budget_gpu_minutes, Some(8));
    assert_eq!(st.target, 42.0);

    let p = ResearchFactory::new(RESEARCH_ID, vec![research_paper_row()])
        .mint(NOW, CREATED)
        .unwrap();
    let pt = p.research.unwrap();
    assert_eq!(pt.kind, ResearchKind::PaperRepro);
    assert!(pt.rubric.is_some());
    assert_eq!(pt.target, 3.14);
    pin_research(vec![research_speedrun_row()], NOW, CREATED);
    pin_research(vec![research_paper_row()], NOW, CREATED);
}

#[test]
fn hidden_tests_never_in_public_statement() {
    let a = ArcFactory::new(ARC_ID, vec![arc_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(a.statement, arc_row().source.body);
    let expected_json = serde_json::to_string(&a.grid.as_ref().unwrap().expected.cells).unwrap();
    assert!(
        !a.statement.contains(&expected_json),
        "ARC expected grid leaked into statement: {}",
        a.statement
    );

    let f = ForecastFactory::new(FORECAST_ID, vec![forecast_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(f.statement, forecast_row().source.body);
    assert!(
        !f.statement.contains("true") && !f.statement.contains("false"),
        "forecast outcome leaked: {}",
        f.statement
    );

    let r = ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(r.statement, research_kaggle_row().source.body);
    assert!(
        !r.statement.contains("0.85"),
        "research target leaked: {}",
        r.statement
    );

    let h = LongHorizonFactory::new(LONG_HORIZON_ID, vec![lh_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(h.statement, lh_row().source.body);
    assert!(!h.statement.contains("alpha"));
    assert!(!h.statement.contains("beta"));
    assert!(!h.statement.contains("omega"));

    let o = OpenEndedFactory::new(OPEN_ENDED_ID, vec![oe_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(o.statement, oe_row().source.body);
    assert!(
        !o.statement.contains("GRADER_ONLY"),
        "rubric grader prompt leaked into statement: {}",
        o.statement
    );

    let p = ResearchFactory::new(RESEARCH_ID, vec![research_paper_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert!(!p.statement.contains("GRADER_ONLY"));
    assert!(!p.statement.contains("3.14"));
}

#[test]
fn statement_hash_is_lowercase_hex_sha256_of_public_statement() {
    for t in [
        ArcFactory::new(ARC_ID, vec![arc_row()])
            .mint(NOW, CREATED)
            .unwrap(),
        ForecastFactory::new(FORECAST_ID, vec![forecast_row()])
            .mint(NOW, CREATED)
            .unwrap(),
        ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
            .mint(NOW, CREATED)
            .unwrap(),
        LongHorizonFactory::new(LONG_HORIZON_ID, vec![lh_row()])
            .mint(NOW, CREATED)
            .unwrap(),
        OpenEndedFactory::new(OPEN_ENDED_ID, vec![oe_row()])
            .mint(NOW, CREATED)
            .unwrap(),
    ] {
        assert_eq!(
            t.spec.statement_hash,
            reference::statement_hash(&t.statement)
        );
        assert_hex("statement_hash", &t.spec.statement_hash);
        assert_hex("hidden_tests_hash", &t.spec.hidden_tests_hash);
        assert_ne!(t.spec.statement_hash, t.spec.hidden_tests_hash);
        assert!(t.spec.env_image.is_none());
    }
}

#[test]
fn empty_catalog_is_empty_catalog() {
    assert_eq!(
        ResearchFactory::new(RESEARCH_ID, vec![]).mint(NOW, CREATED),
        Err(Error::EmptyCatalog)
    );
    assert_eq!(
        LongHorizonFactory::new(LONG_HORIZON_ID, vec![]).mint(NOW, CREATED),
        Err(Error::EmptyCatalog)
    );
    assert_eq!(
        ArcFactory::new(ARC_ID, vec![]).mint(NOW, CREATED),
        Err(Error::EmptyCatalog)
    );
    assert_eq!(
        ForecastFactory::new(FORECAST_ID, vec![]).mint(NOW, CREATED),
        Err(Error::EmptyCatalog)
    );
    assert_eq!(
        OpenEndedFactory::new(OPEN_ENDED_ID, vec![]).mint(NOW, CREATED),
        Err(Error::EmptyCatalog)
    );
    pin_research(vec![], NOW, CREATED);
    pin_lh(vec![], NOW, CREATED);
    pin_arc(vec![], NOW, CREATED);
    pin_forecast(vec![], NOW, CREATED);
    pin_oe(vec![], NOW, CREATED);
}

#[test]
fn empty_statement_is_empty_statement() {
    let row = arc_src("empty", "", vec![vec![1]]);
    assert_eq!(
        ArcFactory::new(ARC_ID, vec![row.clone()]).mint(NOW, CREATED),
        Err(Error::EmptyStatement)
    );
    pin_arc(vec![row], NOW, CREATED);
    pin_research(
        vec![research_src("e", "", ResearchKind::Kaggle, 1.0, None, None)],
        NOW,
        CREATED,
    );
    pin_forecast(vec![forecast_src("e", "", 0.4, true)], NOW, CREATED);
    pin_oe(vec![oe_src("e", "", sample_rubric())], NOW, CREATED);
    pin_lh(
        vec![lh_src(
            "e",
            "",
            vec![math_checkpoint("c", "s", "x")],
            Some(prometheus_verifiers::MathTask::new("y")),
            None,
        )],
        NOW,
        CREATED,
    );
}

#[test]
fn empty_statement_does_not_advance_cursor() {
    let rows = vec![
        arc_src("bad", "", vec![vec![1]]),
        arc_row(),
    ];
    let mut f = ArcFactory::new(ARC_ID, rows.clone());
    assert_eq!(f.mint(NOW, CREATED), Err(Error::EmptyStatement));
    assert_eq!(f.mint(NOW, CREATED), Err(Error::EmptyStatement));
    let mut refer = reference::RefArc::from_prod(&ArcFactory::new(ARC_ID, rows));
    assert_eq!(refer.mint(NOW, CREATED), Err(Error::EmptyStatement));
    assert_eq!(refer.mint(NOW, CREATED), Err(Error::EmptyStatement));
}

#[test]
fn round_robin_advances_only_after_success_and_wraps() {
    let rows = vec![
        arc_src("g1", "grid one", vec![vec![1, 2], vec![3, 4]]),
        arc_src("g2", "grid two", vec![vec![5, 6], vec![7, 8]]),
        arc_src("g3", "grid three", vec![vec![0, 1], vec![1, 0]]),
    ];
    let mut prod = ArcFactory::new(ARC_ID, rows.clone());
    let mut refer = reference::RefArc::from_prod(&prod);
    let mut ids = Vec::new();
    for k in 0..6 {
        let got = prod.mint(NOW, CREATED);
        let want = refer.mint(NOW, CREATED);
        assert_eq!(got, want, "mint {k}");
        ids.push(got.unwrap().spec.task_id);
    }
    assert_eq!(
        ids,
        vec![
            format!("{ARC_ID}:g1"),
            format!("{ARC_ID}:g2"),
            format!("{ARC_ID}:g3"),
            format!("{ARC_ID}:g1"),
            format!("{ARC_ID}:g2"),
            format!("{ARC_ID}:g3"),
        ]
    );
}

#[test]
fn research_round_robin_matches_reference() {
    let rows = vec![research_kaggle_row(), research_speedrun_row(), research_paper_row()];
    let mut prod = ResearchFactory::new(RESEARCH_ID, rows);
    let mut refer = reference::RefResearch::from_prod(&prod);
    for k in 0..5 {
        assert_eq!(prod.mint(NOW, CREATED), refer.mint(NOW, CREATED), "mint {k}");
    }
}

#[test]
fn wave2_horizons_are_pinned_on_minted_spec() {
    let a = ArcFactory::new(ARC_ID, vec![arc_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(a.spec.horizon_s, reference::ARC_HORIZON_S);
    assert_eq!(a.spec.max_tool_calls, reference::ARC_MAX_TOOL_CALLS);
    assert_eq!(a.spec.horizon_s, 30);
    assert_eq!(a.spec.max_tool_calls, START_TOOL_CALLS);
    assert_ne!(a.spec.horizon_s, 60, "SWE 2× is D3-only");

    let f = ForecastFactory::new(FORECAST_ID, vec![forecast_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(f.spec.horizon_s, 30);
    assert_eq!(f.spec.max_tool_calls, START_TOOL_CALLS);

    let o = OpenEndedFactory::new(OPEN_ENDED_ID, vec![oe_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(o.spec.horizon_s, 30);
    assert_eq!(o.spec.max_tool_calls, START_TOOL_CALLS);

    let r = ResearchFactory::new(RESEARCH_ID, vec![research_kaggle_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(r.spec.horizon_s, reference::RESEARCH_HORIZON_S);
    assert_eq!(r.spec.horizon_s, 600);
    assert!(r.spec.horizon_s > 30);
    assert_eq!(r.spec.max_tool_calls, START_TOOL_CALLS);

    let h = LongHorizonFactory::new(LONG_HORIZON_ID, vec![lh_row()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(h.spec.horizon_s, reference::LONG_HORIZON_S);
    assert_eq!(h.spec.horizon_s, 3600);
    assert!(h.spec.horizon_s >= 3600, "hours-scale");
    assert_eq!(h.spec.max_tool_calls, reference::LONG_HORIZON_MAX_TOOL_CALLS);
    assert_eq!(h.spec.max_tool_calls, 2000);
}

#[test]
fn factory_trait_mint_matches_inherent() {
    let mut inherent = ArcFactory::new(ARC_ID, vec![arc_row()]);
    let mut as_trait: Box<dyn Factory> = Box::new(ArcFactory::new(ARC_ID, vec![arc_row()]));
    assert_eq!(
        inherent.mint(NOW, CREATED).unwrap(),
        as_trait.mint(NOW, CREATED).unwrap()
    );
}

#[test]
fn sources_order_stable_across_mint() {
    let rows = vec![arc_row(), arc_src("g2", "second grid", vec![vec![0]])];
    let mut f = ArcFactory::new(ARC_ID, rows.clone());
    let _ = f.mint(NOW, CREATED).unwrap();
    assert_eq!(f.sources(), rows.as_slice());
}

#[test]
fn provenance_comes_from_the_catalog_row() {
    let row = forecast_row();
    let t = ForecastFactory::new(FORECAST_ID, vec![row.clone()])
        .mint(NOW, CREATED)
        .unwrap();
    assert_eq!(t.spec.provenance, row.source.provenance);
}

#[test]
fn raise_if_still_strict_gt_half_after_wave2_mint() {
    let _ = ArcFactory::new(ARC_ID, vec![arc_row()])
        .mint(NOW, CREATED)
        .unwrap();
    let mut h = prometheus_tasks::Horizon::new();
    h.raise_if(0.5);
    assert_eq!(h.tool_calls(), START_TOOL_CALLS);
    h.raise_if(0.5000000000000001);
    assert_eq!(h.tool_calls(), START_TOOL_CALLS * 2);
}
