//! Group: Tracer start/end, duration unit, nesting, leaks, foreign spans.
//!
//! Leak rule (locked): dropping a `Span` without `end` does not close it.
//! After the handle is gone, `current()` still returns that span's name when
//! it is the innermost open span.

use std::thread;
use std::time::{Duration, SystemTime};

use prometheus_obs::{Span, Tracer};

#[test]
fn current_is_none_on_a_fresh_tracer() {
    let t = Tracer::new();
    assert_eq!(t.current(), None);
}

#[test]
fn start_makes_current_that_name() {
    let mut t = Tracer::new();
    let span = t.start("train.step").expect("start");
    assert_eq!(span.name, "train.step");
    assert_eq!(t.current(), Some("train.step"));
    let _ms = t.end(span).expect("end");
    assert_eq!(t.current(), None);
}

#[test]
fn nested_current_is_innermost() {
    let mut t = Tracer::new();
    let outer = t.start("outer").expect("outer");
    assert_eq!(t.current(), Some("outer"));
    let inner = t.start("inner").expect("inner");
    assert_eq!(t.current(), Some("inner"));
    let inner_ms = t.end(inner).expect("end inner");
    let _ = inner_ms;
    assert_eq!(t.current(), Some("outer"));
    let _ = t.end(outer).expect("end outer");
    assert_eq!(t.current(), None);
}

#[test]
fn end_duration_is_milliseconds_as_u128() {
    // Unit: milliseconds. A ~40ms sleep must not look like seconds (0) or
    // microseconds (~40_000).
    let mut t = Tracer::new();
    let span = t.start("sleep").expect("start");
    thread::sleep(Duration::from_millis(40));
    let ms: u128 = t.end(span).expect("end");
    assert!(
        ms >= 20,
        "duration {ms} ms is too small for a 40ms sleep; unit must be milliseconds"
    );
    assert!(
        ms < 10_000,
        "duration {ms} is too large for a 40ms sleep; unit must be milliseconds (not µs/ns)"
    );
}

#[test]
fn end_of_unknown_constructed_span_errors() {
    let mut t = Tracer::new();
    let fake = Span {
        name: "ghost".to_string(),
        start: SystemTime::now(),
    };
    assert!(
        t.end(fake).is_err(),
        "ending a span that was never start()ed on this tracer must error"
    );
    assert_eq!(t.current(), None);
}

#[test]
fn end_of_foreign_span_from_another_tracer_errors() {
    let mut t1 = Tracer::new();
    let mut t2 = Tracer::new();
    let span = t1.start("owned-by-t1").expect("start t1");
    assert!(
        t2.end(span).is_err(),
        "ending a span on a different Tracer must error"
    );
    assert_eq!(
        t1.current(),
        Some("owned-by-t1"),
        "failed foreign end must not close the span on the owner"
    );
}

#[test]
fn end_of_non_innermost_span_errors_lifo() {
    let mut t = Tracer::new();
    let outer = t.start("outer").expect("outer");
    let inner = t.start("inner").expect("inner");
    assert!(
        t.end(outer).is_err(),
        "ending a non-innermost span must error (LIFO)"
    );
    assert_eq!(t.current(), Some("inner"));
    let _ = t.end(inner).expect("end inner after failed outer");
    assert_eq!(
        t.current(),
        Some("outer"),
        "failed end of outer must leave outer open"
    );
}

#[test]
fn drop_without_end_is_detectable_via_current() {
    let mut t = Tracer::new();
    {
        let leaked = t.start("leaked").expect("start");
        assert_eq!(t.current(), Some("leaked"));
        drop(leaked);
    }
    assert_eq!(
        t.current(),
        Some("leaked"),
        "drop without end must not auto-close; current() is the leak signal"
    );
}

#[test]
fn drop_without_end_stays_on_the_stack_under_a_later_span() {
    let mut t = Tracer::new();
    let leaked = t.start("leaked").expect("start leaked");
    drop(leaked);
    let inner = t.start("ok").expect("start ok");
    assert_eq!(t.current(), Some("ok"));
    let _ = t.end(inner).expect("end ok");
    assert_eq!(
        t.current(),
        Some("leaked"),
        "after ending the later span, the leaked outer must still be current"
    );
}

#[test]
fn two_tracers_have_independent_current() {
    let mut t1 = Tracer::new();
    let mut t2 = Tracer::new();
    let a = t1.start("a").expect("t1");
    assert_eq!(t2.current(), None);
    let b = t2.start("b").expect("t2");
    assert_eq!(t1.current(), Some("a"));
    assert_eq!(t2.current(), Some("b"));
    let _ = t1.end(a).expect("end t1");
    assert_eq!(t1.current(), None);
    assert_eq!(t2.current(), Some("b"));
    let _ = t2.end(b).expect("end t2");
}
