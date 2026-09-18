//! Group: injected `NowMs`. Violations and `frozen_at` carry the argument.
//! Nothing reads the wall clock.

mod common;
mod reference;

use common::{
    assert_frozen_pair, assert_has_violation, assert_result_eq, fresh_world, open_pair,
    ALL_BOUNDARIES, GRADER_ID, NOW0, NOW1, NOW_MAX,
};
use prometheus_monitors::Boundary;

#[test]
fn injected_now_appears_on_kill_snapshot() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    assert_result_eq(prod.kill("t", NOW0), refer.kill("t", NOW0), "kill now");
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
    assert_eq!(refer.snapshot().frozen_at, Some(NOW0));
    assert_ne!(prod.snapshot().frozen_at, Some(0));
}

#[test]
fn injected_now_appears_on_every_planted_violation() {
    for boundary in ALL_BOUNDARIES {
        let world = fresh_world();
        let (mut prod, mut refer) = open_pair(&world);
        let detail = match boundary {
            Boundary::Graders => GRADER_ID,
            _ => "d",
        };
        assert_result_eq(
            prod.plant(boundary, detail, NOW0),
            refer.plant(boundary, detail, NOW0),
            "plant now",
        );
        assert_frozen_pair(&prod, &refer);
        assert_has_violation(&prod.snapshot(), boundary, NOW0, detail);
        assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
    }
}

#[test]
fn now_zero_is_a_legal_timestamp() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let _ = prod.note_held_out_access("z", 0);
    let _ = refer.note_held_out_access("z", 0);
    assert_eq!(prod.snapshot().frozen_at, Some(0));
    assert_has_violation(&prod.snapshot(), Boundary::HeldOut, 0, "z");
}

#[test]
fn now_max_is_a_legal_timestamp() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let _ = prod.note_self_check("m", NOW_MAX);
    let _ = refer.note_self_check("m", NOW_MAX);
    assert_eq!(prod.snapshot().frozen_at, Some(NOW_MAX));
    assert_has_violation(&prod.snapshot(), Boundary::Monitors, NOW_MAX, "m");
}

#[test]
fn first_now_wins_on_later_trips() {
    let world = fresh_world();
    let (mut prod, mut refer) = open_pair(&world);
    let _ = prod.note_kernel_escape("a", NOW0);
    let _ = refer.note_kernel_escape("a", NOW0);
    let _ = prod.note_held_out_access("b", NOW1);
    let _ = refer.note_held_out_access("b", NOW1);
    assert_eq!(prod.snapshot().frozen_at, Some(NOW0));
    assert_eq!(refer.snapshot().frozen_at, Some(NOW0));
    assert_has_violation(&prod.snapshot(), Boundary::HeldOut, NOW1, "b");
}

#[test]
fn operations_do_not_sleep() {
    let world = fresh_world();
    let (mut prod, _) = open_pair(&world);
    let start = std::time::Instant::now();
    let _ = prod.kill("fast", NOW0);
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 2,
        "kill slept or blocked on the wall clock: {elapsed:?}"
    );
}
