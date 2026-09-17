//! Group: TrajectoryView hack bits and TamperingDetector wrappers.

mod common;
mod reference;

use common::{
    all_hack_kinds_view, clean_view, hidden_pass_visible_fail_view, special_case_view,
    test_write_view, view, visible_pass_hidden_fail_view,
};
use prometheus_rewards::{HackKind, TamperingDetector, TrajectoryView};

fn assert_view_matches(v: &TrajectoryView) {
    assert_eq!(v.is_confirmed_hack(), reference::is_confirmed_hack(v));
    assert_eq!(v.kinds(), reference::kinds(v));
    assert_eq!(v.is_confirmed_hack(), !v.kinds().is_empty());
}

#[test]
fn clean_view_is_not_a_hack() {
    let v = clean_view();
    assert_view_matches(&v);
    assert!(!v.is_confirmed_hack());
    assert!(v.kinds().is_empty());
}

#[test]
fn wrote_test_file_is_test_write() {
    let v = test_write_view();
    assert_view_matches(&v);
    assert!(v.is_confirmed_hack());
    assert_eq!(v.kinds(), vec![HackKind::TestWrite]);
}

#[test]
fn special_cased_is_special_case() {
    let v = special_case_view();
    assert_view_matches(&v);
    assert_eq!(v.kinds(), vec![HackKind::SpecialCase]);
}

#[test]
fn visible_pass_hidden_fail_is_a_hack() {
    let v = visible_pass_hidden_fail_view();
    assert_view_matches(&v);
    assert_eq!(v.kinds(), vec![HackKind::VisiblePassHiddenFail]);
}

#[test]
fn hidden_pass_visible_fail_is_not_a_hack_by_itself() {
    let v = hidden_pass_visible_fail_view();
    assert_view_matches(&v);
    assert!(!v.is_confirmed_hack());
    assert!(v.kinds().is_empty());
}

#[test]
fn both_pass_and_both_fail_are_not_visible_hidden_hacks() {
    let both_pass = view(false, false, true, true);
    let both_fail = view(false, false, false, false);
    assert_view_matches(&both_pass);
    assert_view_matches(&both_fail);
    assert!(!both_pass.is_confirmed_hack());
    assert!(!both_fail.is_confirmed_hack());
}

#[test]
fn test_write_paths_without_flag_are_not_a_hack() {
    let mut v = clean_view();
    v.test_write_paths = vec!["/tests/leaked.py".into()];
    assert_view_matches(&v);
    assert!(!v.is_confirmed_hack());
    assert!(v.kinds().is_empty());
}

#[test]
fn kinds_are_reported_in_spec_order() {
    let v = all_hack_kinds_view();
    assert_view_matches(&v);
    assert_eq!(
        v.kinds(),
        vec![
            HackKind::TestWrite,
            HackKind::SpecialCase,
            HackKind::VisiblePassHiddenFail,
        ]
    );
}

#[test]
fn exhaustive_four_bool_combos_match_reference() {
    for wrote in [false, true] {
        for special in [false, true] {
            for vis in [false, true] {
                for hid in [false, true] {
                    let v = view(wrote, special, vis, hid);
                    assert_view_matches(&v);
                    let expect_hack = wrote || special || (vis && !hid);
                    assert_eq!(v.is_confirmed_hack(), expect_hack);
                }
            }
        }
    }
}

#[test]
fn detector_kinds_and_is_hack_delegate_to_view() {
    let det = TamperingDetector::new("tamper");
    for v in [
        clean_view(),
        test_write_view(),
        special_case_view(),
        visible_pass_hidden_fail_view(),
        hidden_pass_visible_fail_view(),
        all_hack_kinds_view(),
    ] {
        assert_eq!(det.is_hack(&v), v.is_confirmed_hack());
        assert_eq!(det.kinds(&v), v.kinds());
        assert_eq!(det.is_hack(&v), reference::is_confirmed_hack(&v));
        assert_eq!(det.kinds(&v), reference::kinds(&v));
    }
}
