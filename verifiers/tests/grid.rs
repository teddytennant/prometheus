//! Group: GridMatch exact cells, pass@2, BadGrid, TooManyGrids, planted wrong.

mod common;
mod reference;

use common::{assert_meta, grid, has_kind, reward_request, SCORED_AT};
use prometheus_verifiers::{Answer, Error, EvidenceKind, Grid, GridMatch, GridTask};

fn expected_2x2() -> Grid {
    grid(vec![vec![1, 2], vec![3, 4]])
}

fn checker(expected: Grid) -> GridMatch {
    GridMatch::new("grid-arc", GridTask::new(expected))
}

#[test]
fn exact_match_passes() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![expected_2x2()],
    };
    let got = v.verify(&req, &ans, SCORED_AT, common::NOW_MS).unwrap();
    let exp = reference::verify_grid(&v.task.expected, &req, &ans, SCORED_AT).unwrap();
    assert_meta(&got, &req, SCORED_AT);
    assert!(got.passed);
    assert_eq!(got.score, 1.0);
    assert!(has_kind(&got.evidence, EvidenceKind::Grid));
    assert_eq!(got.passed, exp.passed);
    assert_eq!(got.score, exp.score);
}

#[test]
fn planted_wrong_grid_rejected() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![vec![9, 9], vec![9, 9]])],
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(!got.passed);
    assert_eq!(got.score, 0.0);
}

#[test]
fn pass_at_2_second_slot_matches() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![vec![0, 0], vec![0, 0]]), expected_2x2()],
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(got.passed);
    assert_eq!(got.score, 1.0);
}

#[test]
fn pass_at_2_first_slot_matches() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![expected_2x2(), grid(vec![vec![0, 0], vec![0, 0]])],
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(got.passed);
}

#[test]
fn dihedral_is_not_accepted() {
    // 90-degree rotation of [[1,2],[3,4]] is [[3,1],[4,2]].
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![vec![3, 1], vec![4, 2]])],
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(!got.passed);
    assert_eq!(got.score, 0.0);
}

#[test]
fn color_perm_is_not_accepted() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![vec![2, 1], vec![4, 3]])],
    };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(!got.passed);
}

#[test]
fn too_many_grids_errors_even_if_third_matches() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![
            grid(vec![vec![0, 0], vec![0, 0]]),
            grid(vec![vec![1, 1], vec![1, 1]]),
            expected_2x2(),
        ],
    };
    let err = v.verify(&req, &ans, SCORED_AT, 0).unwrap_err();
    assert_eq!(err, Error::TooManyGrids);
    assert_eq!(
        reference::grid_match(
            &expected_2x2(),
            &[expected_2x2(), expected_2x2(), expected_2x2()]
        )
        .unwrap_err(),
        Error::TooManyGrids
    );
}

#[test]
fn empty_grid_is_bad_grid() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![])],
    };
    assert_eq!(
        v.verify(&req, &ans, SCORED_AT, 0).unwrap_err(),
        Error::BadGrid
    );
}

#[test]
fn zero_cols_is_bad_grid() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![vec![], vec![]])],
    };
    assert_eq!(
        v.verify(&req, &ans, SCORED_AT, 0).unwrap_err(),
        Error::BadGrid
    );
}

#[test]
fn ragged_rows_are_bad_grid() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![vec![1, 2], vec![3]])],
    };
    assert_eq!(
        v.verify(&req, &ans, SCORED_AT, 0).unwrap_err(),
        Error::BadGrid
    );
}

#[test]
fn empty_prediction_list_does_not_pass() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let ans = Answer::Grid { grids: vec![] };
    let got = v.verify(&req, &ans, SCORED_AT, 0).unwrap();
    assert!(!got.passed);
    assert_eq!(got.score, 0.0);
}

#[test]
fn expected_empty_is_bad_grid() {
    let v = checker(grid(vec![]));
    let req = reward_request("grid-arc");
    let ans = Answer::Grid {
        grids: vec![grid(vec![vec![0]])],
    };
    assert_eq!(
        v.verify(&req, &ans, SCORED_AT, 0).unwrap_err(),
        Error::BadGrid
    );
}

#[test]
fn reference_grid_match_pins_exactness() {
    let e = expected_2x2();
    assert!(reference::grid_match(&e, std::slice::from_ref(&e)).unwrap());
    assert!(!reference::grid_match(&e, &[grid(vec![vec![1, 2], vec![3, 5]])]).unwrap());
    let v = checker(e.clone());
    let req = reward_request("grid-arc");
    let got = v
        .verify(
            &req,
            &Answer::Grid {
                grids: vec![e.clone()],
            },
            SCORED_AT,
            0,
        )
        .unwrap();
    assert!(got.passed);
}

#[test]
fn wrong_kind() {
    let v = checker(expected_2x2());
    let req = reward_request("grid-arc");
    let err = v
        .verify(
            &req,
            &Answer::Math {
                latex: "[[1]]".into(),
            },
            SCORED_AT,
            0,
        )
        .unwrap_err();
    assert_eq!(err, Error::WrongKind);
}

#[test]
fn rows_cols_helpers() {
    let g = expected_2x2();
    assert_eq!(g.rows(), 2);
    assert_eq!(g.cols(), 2);
}
