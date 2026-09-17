//! Group: refuse_exact — learned graders never cover exact-verifier domains.

mod common;
mod reference;

use prometheus_rewards::{refuse_exact, Error};

#[test]
fn refuse_exact_matches_reference_for_every_id() {
    for id in ["sympy-exact", "lean-kernel", "grid-match", "x", " "] {
        let got = refuse_exact(id);
        let exp = reference::refuse_exact(id);
        match (got, exp) {
            (Err(Error::ExactVerifier(a)), Err(Error::ExactVerifier(b))) => {
                assert_eq!(a, b);
                assert_eq!(a, id);
            }
            (g, e) => panic!("refuse_exact({id:?}): got {g:?} expected {e:?}"),
        }
    }
}

#[test]
fn refuse_exact_empty_string_is_still_exact_verifier() {
    let err = refuse_exact("").unwrap_err();
    assert_eq!(err, Error::ExactVerifier(String::new()));
}

#[test]
fn refuse_exact_never_succeeds() {
    assert!(refuse_exact("any").is_err());
    assert!(reference::refuse_exact("any").is_err());
}
