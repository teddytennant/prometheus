//! Group: public constants, `Boundary::{parse,as_str}`, serde field set,
//! Error Display, method inventory, and the no-wall-clock source scan.
//!
//! These tests do **not** call
//! `Monitors::{open,frozen,snapshot,note_grader_hash,note_held_out_access,note_kernel_escape,note_self_check,plant,kill}`
//! and are allowed to pass against the unimplemented stub.

use prometheus_monitors::{Boundary, Error, KillSnapshot, Violation, BOUNDARIES};

mod reference;

#[test]
fn boundaries_four_names_in_order() {
    assert_eq!(BOUNDARIES, &["kernel", "graders", "held_out", "monitors"]);
    assert_eq!(BOUNDARIES.len(), 4);
    assert_eq!(BOUNDARIES, reference::BOUNDARIES);
}

#[test]
fn boundary_as_str_matches_boundaries() {
    assert_eq!(Boundary::Kernel.as_str(), "kernel");
    assert_eq!(Boundary::Graders.as_str(), "graders");
    assert_eq!(Boundary::HeldOut.as_str(), "held_out");
    assert_eq!(Boundary::Monitors.as_str(), "monitors");
    assert_eq!(
        [
            Boundary::Kernel.as_str(),
            Boundary::Graders.as_str(),
            Boundary::HeldOut.as_str(),
            Boundary::Monitors.as_str(),
        ],
        *BOUNDARIES
    );
}

#[test]
fn boundary_parse_four_names() {
    assert_eq!(Boundary::parse("kernel"), Ok(Boundary::Kernel));
    assert_eq!(Boundary::parse("graders"), Ok(Boundary::Graders));
    assert_eq!(Boundary::parse("held_out"), Ok(Boundary::HeldOut));
    assert_eq!(Boundary::parse("monitors"), Ok(Boundary::Monitors));
    for name in BOUNDARIES {
        let b = Boundary::parse(name).expect(name);
        assert_eq!(b.as_str(), *name);
        assert_eq!(reference::parse_boundary(name), Ok(b));
    }
}

#[test]
fn boundary_parse_unknown_is_error() {
    for bad in [
        "",
        "Kernel",
        "KERNEL",
        "held-out",
        "heldout",
        "held_out ",
        " kernel",
        "eval-gate",
        "eval_gate",
        "grader",
        "monitor",
        "foo",
    ] {
        match Boundary::parse(bad) {
            Err(Error::UnknownBoundary(s)) => assert_eq!(s, bad, "{bad}"),
            other => panic!("expected UnknownBoundary({bad:?}), got {other:?}"),
        }
        match reference::parse_boundary(bad) {
            Err(Error::UnknownBoundary(s)) => assert_eq!(s, bad),
            other => panic!("reference parse {bad:?} -> {other:?}"),
        }
    }
}

#[test]
fn boundary_parse_does_not_accept_serde_variant_names() {
    for name in ["Kernel", "Graders", "HeldOut", "Monitors"] {
        match Boundary::parse(name) {
            Err(Error::UnknownBoundary(s)) => assert_eq!(s, name),
            other => panic!("{name}: {other:?}"),
        }
    }
}

#[test]
fn error_display_strings() {
    assert_eq!(
        Error::UnknownBoundary("x".into()).to_string(),
        "unknown boundary x"
    );
    assert_eq!(
        Error::AlreadyFrozen.to_string(),
        "kill switch already frozen"
    );
    assert_eq!(
        Error::GraderHashMismatch.to_string(),
        "grader hash mismatch"
    );
    assert_eq!(Error::HeldOutAccess.to_string(), "held-out access blocked");
    assert_eq!(
        Error::MonitorSelfCheck.to_string(),
        "monitor self-check failed"
    );
    assert_eq!(Error::Message("boom".into()).to_string(), "boom");
}

#[test]
fn freeze_reason_matches_error_display_where_they_overlap() {
    assert_eq!(
        reference::freeze_reason(Boundary::Graders),
        Error::GraderHashMismatch.to_string()
    );
    assert_eq!(
        reference::freeze_reason(Boundary::HeldOut),
        Error::HeldOutAccess.to_string()
    );
    assert_eq!(
        reference::freeze_reason(Boundary::Monitors),
        Error::MonitorSelfCheck.to_string()
    );
    assert_eq!(reference::freeze_reason(Boundary::Kernel), "kernel escape");
}

#[test]
fn boundary_serde_is_variant_name_string() {
    for (b, name) in [
        (Boundary::Kernel, "Kernel"),
        (Boundary::Graders, "Graders"),
        (Boundary::HeldOut, "HeldOut"),
        (Boundary::Monitors, "Monitors"),
    ] {
        let v = serde_json::to_value(b).expect("ser");
        assert_eq!(v, serde_json::Value::String(name.to_string()));
        let back: Boundary = serde_json::from_value(v).expect("de");
        assert_eq!(back, b);
    }
}

#[test]
fn kill_snapshot_serde_keys() {
    let snap = KillSnapshot {
        frozen: true,
        frozen_at: Some(1),
        reason: Some("r".into()),
        violations: vec![Violation {
            boundary: Boundary::Kernel,
            at_ms: 1,
            detail: "d".into(),
        }],
    };
    let v = serde_json::to_value(&snap).expect("serde");
    let obj = v.as_object().expect("object");
    let keys: std::collections::BTreeSet<&str> = obj.keys().map(|s| s.as_str()).collect();
    let expected: std::collections::BTreeSet<&str> =
        ["frozen", "frozen_at", "reason", "violations"]
            .into_iter()
            .collect();
    assert_eq!(keys, expected);
    let back: KillSnapshot = serde_json::from_value(v).expect("de");
    assert_eq!(back, snap);
}

#[test]
fn violation_serde_keys() {
    let vln = Violation {
        boundary: Boundary::HeldOut,
        at_ms: 9,
        detail: "who".into(),
    };
    let v = serde_json::to_value(&vln).expect("serde");
    let obj = v.as_object().expect("object");
    let keys: std::collections::BTreeSet<&str> = obj.keys().map(|s| s.as_str()).collect();
    let expected: std::collections::BTreeSet<&str> =
        ["boundary", "at_ms", "detail"].into_iter().collect();
    assert_eq!(keys, expected);
}

#[test]
fn monitors_public_methods_are_exactly_the_kill_switch_surface() {
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
    let impl_idx = src.find("impl Monitors").expect("impl Monitors block");
    let rest = &src[impl_idx..];
    let mut names = Vec::new();
    for line in rest.lines() {
        let t = line.trim();
        if t.starts_with("pub fn ") {
            let after = t.trim_start_matches("pub fn ");
            let name = after.split(|c: char| c == '<' || c == '(').next().unwrap();
            names.push(name.to_string());
        }
        if t.starts_with("impl ") && !t.starts_with("impl Monitors") {
            break;
        }
    }
    names.sort();
    assert_eq!(
        names,
        [
            "frozen",
            "kill",
            "note_grader_hash",
            "note_held_out_access",
            "note_kernel_escape",
            "note_self_check",
            "open",
            "plant",
            "snapshot",
        ],
        "Monitors public methods changed"
    );
}

#[test]
fn lib_src_has_no_unfreeze_or_disable_api() {
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
    for needle in [
        "pub fn unfreeze",
        "pub fn thaw",
        "pub fn disable",
        "pub fn disable_boundary",
        "pub fn skip_monitor",
        "pub fn bypass",
        "pub fn disarm",
        "pub fn clear_freeze",
        "pub fn reset",
    ] {
        assert!(
            !src.contains(needle),
            "genome must not be able to disable monitors ({needle})"
        );
    }
}

#[test]
fn lib_src_does_not_read_the_wall_clock() {
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
    for needle in [
        "SystemTime",
        "Instant",
        "Utc::now",
        "Local::now",
        "chrono::",
        "time::OffsetDateTime",
        "unix_epoch",
        "std::thread::sleep",
        "std::time::sleep",
    ] {
        assert!(
            !src.contains(needle),
            "NowMs is injected; production must not use {needle}"
        );
    }
}

#[test]
fn lib_src_does_not_import_kernel_graders_or_eval_gate() {
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
    for needle in [
        "prometheus_kernel",
        "prometheus_eval_gate",
        "harness/kernel",
        "harness/eval-gate",
    ] {
        assert!(
            !src.contains(needle),
            "monitors freeze via the freeze file, not by importing {needle}"
        );
    }
}
