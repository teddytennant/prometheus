//! Group 6: send_message / inbox: FIFO, dropout keep and drop, None memo
//! stays None, from==to, isolation (A cannot see B's mail), UnknownAgent order.

mod common;
mod reference;

use common::*;
use prometheus_multiagent::{MultiError, Topology};

#[test]
fn send_fifo_per_inbox() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 32),
    );
    for text in ["one", "two", "three"] {
        let m = msg("orchestrator", "sub-00", text, None);
        let p = prod.send_message(&eid("e"), m.clone(), 0).unwrap();
        let r = refer.send_message(&eid("e"), m.clone(), 0).unwrap();
        assert_eq!(p, r);
        assert_eq!(p.text, text);
        assert_eq!(p.latent_memo, None);
    }
    let pin = prod.inbox(&eid("e"), &aid("sub-00")).unwrap();
    let rin = refer.inbox(&eid("e"), &aid("sub-00")).unwrap();
    assert_eq!(pin, rin);
    assert_eq!(
        pin.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
        vec!["one", "two", "three"]
    );
}

#[test]
fn none_memo_stays_none_even_when_draw_would_keep() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let m = msg("solo", "solo", "hi", None);
    let p = prod.send_message(&eid("e"), m.clone(), 0).unwrap();
    let r = refer.send_message(&eid("e"), m, 0).unwrap();
    assert_eq!(p, r);
    assert_eq!(p.latent_memo, None);
}

#[test]
fn none_memo_stays_none_even_when_draw_would_drop() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let m = msg("solo", "solo", "hi", None);
    let p = prod.send_message(&eid("e"), m.clone(), 50).unwrap();
    let r = refer.send_message(&eid("e"), m, 50).unwrap();
    assert_eq!(p, r);
    assert_eq!(p.latent_memo, None);
}

#[test]
fn latent_keep_when_draw_mod_denom_lt_numer() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let memo = latent(vec![1.0, 2.0]);
    // default numer=50 denom=100. 0 % 100 = 0 < 50 → keep. 49 keep.
    for draw in [0u64, 49, 100, 149] {
        let m = msg("solo", "solo", "k", memo.clone());
        let p = prod.send_message(&eid("e"), m.clone(), draw).unwrap();
        let r = refer.send_message(&eid("e"), m, draw).unwrap();
        assert_eq!(p, r, "draw {draw}");
        assert_eq!(p.latent_memo, memo, "draw {draw} should keep");
        assert_eq!(p.text, "k");
    }
}

#[test]
fn latent_drop_when_draw_mod_denom_ge_numer() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let memo = latent(vec![3.0]);
    // 50 % 100 = 50 < 50 is false → drop. 99 drop. 150 → 50 drop.
    for draw in [50u64, 99, 150, 199] {
        let m = msg("solo", "solo", "d", memo.clone());
        let p = prod.send_message(&eid("e"), m.clone(), draw).unwrap();
        let r = refer.send_message(&eid("e"), m, draw).unwrap();
        assert_eq!(p, r, "draw {draw}");
        assert_eq!(p.latent_memo, None, "draw {draw} should drop");
        assert_eq!(p.text, "d");
        assert_eq!(p.from, aid("solo"));
        assert_eq!(p.to, aid("solo"));
    }
}

#[test]
fn latent_always_keep_when_numer_equals_denom() {
    let mut cfg = default_config();
    cfg.latent_keep_numer = 1;
    cfg.latent_keep_denom = 1;
    let (mut prod, mut refer) = pair(cfg);
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let memo = latent(vec![9.0]);
    for draw in [0u64, 1, 2, 99, u64::MAX] {
        let m = msg("solo", "solo", "x", memo.clone());
        let p = prod.send_message(&eid("e"), m.clone(), draw).unwrap();
        let r = refer.send_message(&eid("e"), m, draw).unwrap();
        assert_eq!(p.latent_memo, memo, "draw {draw}");
        assert_eq!(r.latent_memo, memo);
    }
}

#[test]
fn latent_always_drop_when_numer_zero() {
    let mut cfg = default_config();
    cfg.latent_keep_numer = 0;
    cfg.latent_keep_denom = 100;
    let (mut prod, mut refer) = pair(cfg);
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let memo = latent(vec![1.0]);
    for draw in [0u64, 1, 49, 50, 99] {
        let m = msg("solo", "solo", "x", memo.clone());
        let p = prod.send_message(&eid("e"), m.clone(), draw).unwrap();
        let r = refer.send_message(&eid("e"), m, draw).unwrap();
        assert_eq!(p.latent_memo, None, "draw {draw}");
        assert_eq!(r.latent_memo, None);
    }
}

#[test]
fn returned_message_matches_stored_inbox_tail() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::AuthorReviewer, 0, 8),
    );
    let m = msg("author", "reviewer", "please", latent(vec![0.5]));
    let p = prod.send_message(&eid("e"), m.clone(), 0).unwrap();
    let r = refer.send_message(&eid("e"), m, 0).unwrap();
    assert_eq!(p, r);
    assert_eq!(prod.inbox(&eid("e"), &aid("reviewer")).unwrap(), vec![p]);
    assert_eq!(refer.inbox(&eid("e"), &aid("reviewer")).unwrap(), vec![r]);
}

#[test]
fn from_equals_to_self_message_lands_in_that_inbox_only() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::ProposerSolver, 0, 8),
    );
    let m = msg("proposer", "proposer", "note", None);
    assert_both_ok(
        prod.send_message(&eid("e"), m.clone(), 0),
        refer.send_message(&eid("e"), m, 0),
    );
    assert_eq!(prod.inbox(&eid("e"), &aid("proposer")).unwrap().len(), 1);
    assert_eq!(prod.inbox(&eid("e"), &aid("solver")).unwrap().len(), 0);
    assert_eq!(refer.inbox(&eid("e"), &aid("solver")).unwrap().len(), 0);
    assert_inbox_isolation(&prod, &refer, &eid("e"));
}

#[test]
fn isolation_a_cannot_see_b_mail() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::OrchestratorSubs, 2, 8),
    );
    let m = msg("orchestrator", "sub-00", "private", None);
    assert_both_ok(
        prod.send_message(&eid("e"), m.clone(), 0),
        refer.send_message(&eid("e"), m, 0),
    );
    assert_eq!(prod.inbox(&eid("e"), &aid("sub-00")).unwrap().len(), 1);
    assert_eq!(prod.inbox(&eid("e"), &aid("sub-01")).unwrap().len(), 0);
    assert_eq!(
        prod.inbox(&eid("e"), &aid("orchestrator")).unwrap().len(),
        0
    );
    assert_eq!(refer.inbox(&eid("e"), &aid("sub-01")).unwrap().len(), 0);
    assert_inbox_isolation(&prod, &refer, &eid("e"));
}

#[test]
fn empty_text_allowed() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let m = msg("solo", "solo", "", None);
    let p = prod.send_message(&eid("e"), m.clone(), 0).unwrap();
    let r = refer.send_message(&eid("e"), m, 0).unwrap();
    assert_eq!(p.text, "");
    assert_eq!(r.text, "");
    assert_eq!(prod.inbox(&eid("e"), &aid("solo")).unwrap()[0].text, "");
}

#[test]
fn unknown_from_checked_before_unknown_to() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let m = msg("ghost-from", "ghost-to", "x", None);
    let e = assert_both_err(
        prod.send_message(&eid("e"), m.clone(), 0),
        refer.send_message(&eid("e"), m, 0),
    );
    assert_unknown_agent(&e, &aid("ghost-from"));
    assert_eq!(prod.inbox(&eid("e"), &aid("solo")).unwrap().len(), 0);
}

#[test]
fn unknown_to_when_from_known() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let m = msg("solo", "ghost-to", "x", None);
    let e = assert_both_err(
        prod.send_message(&eid("e"), m.clone(), 0),
        refer.send_message(&eid("e"), m, 0),
    );
    assert_unknown_agent(&e, &aid("ghost-to"));
    assert_eq!(prod.inbox(&eid("e"), &aid("solo")).unwrap().len(), 0);
}

#[test]
fn unknown_episode_before_unknown_from() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let m = msg("ghost", "ghost", "x", None);
    let e = assert_both_err(
        prod.send_message(&eid("nope"), m.clone(), 0),
        refer.send_message(&eid("nope"), m, 0),
    );
    assert_unknown_episode(&e, &eid("nope"));
}

#[test]
fn inbox_unknown_agent_does_not_invent_empty() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let e = assert_both_err(
        prod.inbox(&eid("e"), &aid("ghost")),
        refer.inbox(&eid("e"), &aid("ghost")),
    );
    assert_unknown_agent(&e, &aid("ghost"));
}

#[test]
fn send_does_not_consult_token_budget() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 1),
    );
    assert_both_ok(
        prod.charge_tokens(&eid("e"), &aid("solo"), 1),
        refer.charge_tokens(&eid("e"), &aid("solo"), 1),
    );
    let m = msg("solo", "solo", "still-ok", None);
    assert_both_ok(
        prod.send_message(&eid("e"), m.clone(), 0),
        refer.send_message(&eid("e"), m, 0),
    );
    assert_eq!(prod.inbox(&eid("e"), &aid("solo")).unwrap().len(), 1);
}

#[test]
fn messages_do_not_cross_episodes() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("a", "t", Topology::Single, 0, 8),
    );
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("b", "t", Topology::Single, 0, 8),
    );
    let m = msg("solo", "solo", "only-a", None);
    assert_both_ok(
        prod.send_message(&eid("a"), m.clone(), 0),
        refer.send_message(&eid("a"), m, 0),
    );
    assert_eq!(prod.inbox(&eid("b"), &aid("solo")).unwrap().len(), 0);
    assert_eq!(refer.inbox(&eid("b"), &aid("solo")).unwrap().len(), 0);
}

#[test]
fn instance_latent_keep_not_production_constants() {
    // numer=1 denom=2: keep when draw%2 < 1 i.e. even.
    let mut cfg = default_config();
    cfg.latent_keep_numer = 1;
    cfg.latent_keep_denom = 2;
    let (mut prod, mut refer) = pair(cfg);
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::Single, 0, 8),
    );
    let memo = latent(vec![4.0]);
    let keep = prod
        .send_message(&eid("e"), msg("solo", "solo", "k", memo.clone()), 2)
        .unwrap();
    let keep_r = refer
        .send_message(&eid("e"), msg("solo", "solo", "k", memo.clone()), 2)
        .unwrap();
    assert_eq!(keep.latent_memo, memo);
    assert_eq!(keep_r.latent_memo, memo);
    let drop = prod
        .send_message(&eid("e"), msg("solo", "solo", "d", memo.clone()), 3)
        .unwrap();
    let drop_r = refer
        .send_message(&eid("e"), msg("solo", "solo", "d", memo), 3)
        .unwrap();
    assert_eq!(drop.latent_memo, None);
    assert_eq!(drop_r.latent_memo, None);
}

#[test]
fn send_unknown_from_known_to_does_not_deliver() {
    let (mut prod, mut refer) = pair_default();
    spawn_ok(
        &mut prod,
        &mut refer,
        spec("e", "t", Topology::AuthorReviewer, 0, 8),
    );
    let m = msg("ghost", "reviewer", "x", None);
    match assert_both_err(
        prod.send_message(&eid("e"), m.clone(), 0),
        refer.send_message(&eid("e"), m, 0),
    ) {
        MultiError::UnknownAgent(id) => assert_eq!(id, aid("ghost")),
        other => panic!("{other:?}"),
    }
    assert_eq!(prod.inbox(&eid("e"), &aid("reviewer")).unwrap().len(), 0);
}
