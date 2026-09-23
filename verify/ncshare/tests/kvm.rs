//! Oracle tests for `kvm_available`.
//!
//! `ref_*` must pass on the stub. `prod_*` call
//! `prometheus_verify_ncshare::kvm_available` only and must panic
//! `unimplemented!` until F4 is implemented. Do not `#[should_panic]`.

mod reference;

use prometheus_verify_ncshare::kvm_available;
use reference::{FACTS_KVM_ABSENT, FACTS_KVM_CHAR, FACTS_KVM_NO, FACTS_KVM_YES, FACTS_NO_KVM};

// ---------------------------------------------------------------------------
// Reference
// ---------------------------------------------------------------------------

#[test]
fn ref_kvm_available_char_device_true() {
    assert!(reference::kvm_available(FACTS_KVM_CHAR).unwrap());
}

#[test]
fn ref_kvm_available_explicit_yes_true() {
    assert!(reference::kvm_available(FACTS_KVM_YES).unwrap());
}

#[test]
fn ref_kvm_available_explicit_no_false() {
    assert!(!reference::kvm_available(FACTS_KVM_NO).unwrap());
}

#[test]
fn ref_kvm_available_absent_false() {
    assert!(!reference::kvm_available(FACTS_KVM_ABSENT).unwrap());
}

#[test]
fn ref_kvm_available_empty_errors() {
    assert!(reference::kvm_available("").is_err());
    assert!(reference::kvm_available("  \n\t").is_err());
}

#[test]
fn ref_kvm_available_no_mention_errors() {
    assert!(reference::kvm_available(FACTS_NO_KVM).is_err());
}

#[test]
fn ref_kvm_available_case_insensitive_yes() {
    assert!(reference::kvm_available("KVM: YES\n").unwrap());
    assert!(reference::kvm_available("kvm: True\n").unwrap());
    assert!(reference::kvm_available("kvm: present\n").unwrap());
}

#[test]
fn ref_kvm_available_explicit_wins_over_absence_line() {
    let facts = "stat: cannot stat '/dev/kvm': No such file or directory\nkvm: yes\n";
    assert!(reference::kvm_available(facts).unwrap());
}

#[test]
fn ref_kvm_available_character_device_phrase() {
    let facts = "/dev/kvm: character device\n";
    assert!(reference::kvm_available(facts).unwrap());
}

#[test]
fn ref_kvm_available_unrecognized_value_errors() {
    assert!(reference::kvm_available("kvm: maybe\n").is_err());
}

// ---------------------------------------------------------------------------
// Production public API — must fail on unimplemented! stub
// ---------------------------------------------------------------------------

#[test]
fn prod_kvm_available_char_device_true() {
    assert!(kvm_available(FACTS_KVM_CHAR).unwrap());
}

#[test]
fn prod_kvm_available_explicit_yes_true() {
    assert!(kvm_available(FACTS_KVM_YES).unwrap());
}

#[test]
fn prod_kvm_available_explicit_no_false() {
    assert!(!kvm_available(FACTS_KVM_NO).unwrap());
}

#[test]
fn prod_kvm_available_absent_false() {
    assert!(!kvm_available(FACTS_KVM_ABSENT).unwrap());
}

#[test]
fn prod_kvm_available_empty_errors() {
    assert!(kvm_available("").is_err());
    assert!(kvm_available("  \n\t").is_err());
}

#[test]
fn prod_kvm_available_no_mention_errors() {
    assert!(kvm_available(FACTS_NO_KVM).is_err());
}

#[test]
fn prod_kvm_available_matches_reference() {
    for facts in [
        FACTS_KVM_CHAR,
        FACTS_KVM_YES,
        FACTS_KVM_NO,
        FACTS_KVM_ABSENT,
        FACTS_NO_KVM,
        "",
        "kvm: yes\n",
        "kvm: no\n",
    ] {
        match (kvm_available(facts), reference::kvm_available(facts)) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "facts={facts:?}"),
            (Err(_), Err(_)) => {}
            (p, r) => panic!("prod={p:?} ref={r:?} facts={facts:?}"),
        }
    }
}
