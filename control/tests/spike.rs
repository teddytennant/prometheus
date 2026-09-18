//! Group: loss-spike execution. Does not call ckpt.

mod common;
mod reference;

use common::*;
use prometheus_control::{ShardId, PAGE_WINDOW_STEPS};

fn shard(s: &str) -> ShardId {
    ShardId(s.to_string())
}

#[test]
fn first_spike_page_false_echoes_checkpoint_id_and_shard() {
    let cfg = default_config();
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    let ckpt = "mem-ckpt-7".to_string();
    let sh = shard("data-3");
    let prod_r = prod
        .on_loss_spike(100, sh.clone(), ckpt.clone())
        .expect("prod spike");
    let ref_r = refer
        .on_loss_spike(100, sh.clone(), ckpt.clone())
        .expect("ref spike");
    assert_eq!(prod_r.rollback_checkpoint_id, ckpt);
    assert_eq!(prod_r.skipped_shard, sh);
    assert!(!prod_r.page);
    assert_eq!(prod_r, ref_r);
    assert_eq!(prod.pages().unwrap(), 0);
    assert_eq!(prod.skipped_shards().unwrap(), vec![shard("data-3")]);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn second_spike_inside_window_pages() {
    let mut cfg = default_config();
    cfg.page_window_steps = 10;
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    let _ = prod.on_loss_spike(100, shard("a"), "c1".into()).unwrap();
    let _ = refer.on_loss_spike(100, shard("a"), "c1".into()).unwrap();
    let r = prod.on_loss_spike(109, shard("b"), "c2".into()).unwrap();
    let rr = refer.on_loss_spike(109, shard("b"), "c2".into()).unwrap();
    assert!(r.page);
    assert_eq!(r.rollback_checkpoint_id, "c2");
    assert_eq!(r.skipped_shard, shard("b"));
    assert_eq!(r, rr);
    assert_eq!(prod.pages().unwrap(), 1);
    assert_eq!(prod.skipped_shards().unwrap(), vec![shard("a"), shard("b")]);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn second_spike_at_or_beyond_window_does_not_page() {
    let mut cfg = default_config();
    cfg.page_window_steps = 10;
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    let _ = prod.on_loss_spike(100, shard("a"), "c1".into()).unwrap();
    let _ = refer.on_loss_spike(100, shard("a"), "c1".into()).unwrap();
    let r = prod.on_loss_spike(110, shard("b"), "c2".into()).unwrap();
    let rr = refer.on_loss_spike(110, shard("b"), "c2".into()).unwrap();
    assert!(
        !r.page,
        "delta == page_window_steps is not inside the window"
    );
    assert_eq!(r, rr);
    assert_eq!(prod.pages().unwrap(), 0);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn skipped_shards_lists_every_id_in_call_order() {
    let mut cfg = default_config();
    cfg.page_window_steps = 100;
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    for (step, name) in [(1u64, "s0"), (2, "s1"), (3, "s0")] {
        let _ = prod
            .on_loss_spike(step, shard(name), format!("c{step}"))
            .unwrap();
        let _ = refer
            .on_loss_spike(step, shard(name), format!("c{step}"))
            .unwrap();
    }
    assert_eq!(
        prod.skipped_shards().unwrap(),
        vec![shard("s0"), shard("s1"), shard("s0")]
    );
    // first page=false; second delta 1 < 100 page; third delta 1 < 100 page
    assert_eq!(prod.pages().unwrap(), 2);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn uses_config_page_window_not_only_the_constant() {
    let mut cfg = default_config();
    cfg.page_window_steps = 3;
    let (mut prod, mut refer) = pair(cfg, n_live_m_spare(2, 0));
    let _ = prod.on_loss_spike(0, shard("a"), "c".into()).unwrap();
    let _ = refer.on_loss_spike(0, shard("a"), "c".into()).unwrap();
    // delta 2 < 3 => page. Would not page if someone used PAGE_WINDOW_STEPS=10000.
    let r = prod.on_loss_spike(2, shard("b"), "c".into()).unwrap();
    let rr = refer.on_loss_spike(2, shard("b"), "c".into()).unwrap();
    assert!(r.page);
    assert_eq!(r, rr);
    assert_ne!(PAGE_WINDOW_STEPS, 3);
    assert_eq!(prod.pages().unwrap(), 1);
    assert_prod_matches_ref(&prod, &refer);
}

#[test]
fn empty_checkpoint_id_is_echoed_not_invented() {
    let (mut prod, mut refer) = pair(default_config(), n_live_m_spare(2, 0));
    let r = prod.on_loss_spike(0, shard("z"), String::new()).unwrap();
    let rr = refer.on_loss_spike(0, shard("z"), String::new()).unwrap();
    assert_eq!(r.rollback_checkpoint_id, "");
    assert_eq!(r, rr);
}

#[test]
fn spike_does_not_change_membership() {
    let cfg = default_config();
    let specs = n_live_m_spare(2, 1);
    let ids = spec_ids(&specs);
    let (mut prod, mut refer) = pair(cfg, specs);
    let before_states: Vec<_> = ids
        .iter()
        .map(|id| prod.replica_state(id).unwrap())
        .collect();
    let _ = prod.on_loss_spike(5, shard("x"), "m".into()).unwrap();
    let _ = refer.on_loss_spike(5, shard("x"), "m".into()).unwrap();
    assert_eq!(prod.n_live().unwrap(), 2);
    let after_states: Vec<_> = ids
        .iter()
        .map(|id| prod.replica_state(id).unwrap())
        .collect();
    assert_eq!(before_states, after_states);
    assert_prod_matches_ref(&prod, &refer);
}
