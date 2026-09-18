//! Group: `predict_held_out` — fit rungs 1+2 predict 3, and permutations.
//!
//! Synthetic losses from a known `(A, alpha)`. Tiny N, D. HeldOutInTrain
//! is decided on `spec.id` before `fit`.

mod reference;

use prometheus_control::rung::{RungId, RungSpec};
use prometheus_control::scale::{predict_held_out, ScaleError};
use reference::scale as ref_scale;

fn obs(
    id: RungId,
    active: u64,
    tokens: u64,
    loss: f64,
) -> prometheus_control::scale::RungObservation {
    ref_scale::observation(id, active, tokens, loss)
}

fn spec(id: RungId, active: u64, tokens: u64) -> RungSpec {
    obs(id, active, tokens, 1.0).spec
}

#[test]
fn fit_one_two_predicts_three_on_the_generating_law() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let held = spec(RungId::Three, 4, 1);
    let got = predict_held_out(&train, &held).expect("held-out");
    let refer = ref_scale::predict_held_out(&train, &held).expect("ref");
    ref_scale::assert_rel_close(got, refer, 1e-12, "vs ref");
    ref_scale::assert_rel_close(got, 0.125, 1e-9, "L=3/C at C=24");
}

#[test]
fn fit_one_three_predicts_two() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Three, 4, 1, 0.125)];
    let held = spec(RungId::Two, 2, 1);
    let got = predict_held_out(&train, &held).unwrap();
    ref_scale::assert_rel_close(got, 0.25, 1e-9, "L at C=12");
    ref_scale::assert_rel_close(
        got,
        ref_scale::predict_held_out(&train, &held).unwrap(),
        1e-12,
        "vs ref",
    );
}

#[test]
fn fit_two_three_predicts_one() {
    let train = [
        obs(RungId::Two, 2, 1, 0.25),
        obs(RungId::Three, 4, 1, 0.125),
    ];
    let held = spec(RungId::One, 1, 1);
    let got = predict_held_out(&train, &held).unwrap();
    ref_scale::assert_rel_close(got, 0.5, 1e-9, "L at C=6");
}

#[test]
fn permutation_of_train_order_does_not_change_held_out() {
    let a = obs(RungId::One, 1, 1, 0.5);
    let b = obs(RungId::Two, 2, 1, 0.25);
    let held = spec(RungId::Three, 4, 1);
    let p1 = predict_held_out(&[a.clone(), b.clone()], &held).unwrap();
    let p2 = predict_held_out(&[b, a], &held).unwrap();
    ref_scale::assert_rel_close(p1, p2, 1e-12, "order");
    ref_scale::assert_rel_close(p1, 0.125, 1e-9, "value");
}

#[test]
fn held_out_id_in_train_is_held_out_in_train() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let held_same_id_different_nd = spec(RungId::One, 99, 99);
    assert_eq!(
        predict_held_out(&train, &held_same_id_different_nd),
        Err(ScaleError::HeldOutInTrain)
    );
    assert_eq!(
        ref_scale::predict_held_out(&train, &held_same_id_different_nd),
        Err(ScaleError::HeldOutInTrain)
    );

    let held_two = spec(RungId::Two, 2, 1);
    assert_eq!(
        predict_held_out(&train, &held_two),
        Err(ScaleError::HeldOutInTrain)
    );
}

#[test]
fn held_out_in_train_wins_over_invalid_train() {
    let train = [
        obs(RungId::Zero, 0, 0, f64::NAN),
        obs(RungId::One, 1, 1, 0.5),
    ];
    let held = spec(RungId::One, 8, 8);
    assert_eq!(
        predict_held_out(&train, &held),
        Err(ScaleError::HeldOutInTrain)
    );
    assert_eq!(
        ref_scale::predict_held_out(&train, &held),
        Err(ScaleError::HeldOutInTrain)
    );
}

#[test]
fn held_out_not_in_train_falls_through_to_fit() {
    let train = [obs(RungId::Zero, 1, 1, 0.5), obs(RungId::One, 2, 1, 0.25)];
    let held = spec(RungId::Two, 4, 1);
    assert_eq!(
        predict_held_out(&train, &held),
        Err(ScaleError::Rung0NotUsed)
    );
}

#[test]
fn held_out_rung_zero_after_a_valid_fit_is_rung0_not_used() {
    let train = [obs(RungId::One, 1, 1, 0.5), obs(RungId::Two, 2, 1, 0.25)];
    let held = spec(RungId::Zero, 4, 1);
    assert_eq!(
        predict_held_out(&train, &held),
        Err(ScaleError::Rung0NotUsed)
    );
    assert_eq!(
        ref_scale::predict_held_out(&train, &held),
        Err(ScaleError::Rung0NotUsed)
    );
}

#[test]
fn held_out_equals_fit_then_predict_spec() {
    let train = [
        obs(RungId::One, 3, 5, ref_scale::law_loss(2.0, 0.5, 3, 5)),
        obs(RungId::Two, 7, 5, ref_scale::law_loss(2.0, 0.5, 7, 5)),
    ];
    let held = spec(RungId::Three, 11, 5);
    let via_held = predict_held_out(&train, &held).unwrap();
    let via_fit = prometheus_control::scale::fit(&train)
        .unwrap()
        .predict_spec(&held)
        .unwrap();
    ref_scale::assert_rel_close(via_held, via_fit, 1e-12, "held vs fit.predict_spec");
    ref_scale::assert_rel_close(
        via_held,
        ref_scale::law_loss(2.0, 0.5, 11, 5),
        1e-9,
        "generating law",
    );
}

#[test]
fn empty_train_is_need_two_rungs_not_held_out_in_train() {
    let held = spec(RungId::Three, 4, 1);
    assert_eq!(predict_held_out(&[], &held), Err(ScaleError::NeedTwoRungs));
    assert_eq!(
        ref_scale::predict_held_out(&[], &held),
        Err(ScaleError::NeedTwoRungs)
    );
}
