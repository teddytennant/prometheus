//! Group: normalize — lowercase, collapse whitespace, strip zero-width.

mod common;
mod reference;

use prometheus_dedup::normalize;

#[derive(serde::Deserialize)]
struct Case {
    #[serde(rename = "in")]
    input: String,
    out: String,
}

fn goldens() -> Vec<Case> {
    serde_json::from_str(include_str!("goldens/normalize.json")).expect("normalize goldens")
}

#[test]
fn golden_strings_match_public_and_reference() {
    for case in goldens() {
        let got = normalize(&case.input);
        assert_eq!(
            got, case.out,
            "public normalize({:?}) must equal golden",
            case.input
        );
        assert_eq!(
            reference::normalize(&case.input),
            case.out,
            "reference normalize({:?}) must equal golden",
            case.input
        );
    }
}

#[test]
fn lowercase_ascii() {
    assert_eq!(normalize("ABC xyz DEF"), "abc xyz def");
    assert_eq!(
        reference::normalize("ABC xyz DEF"),
        normalize("ABC xyz DEF")
    );
}

#[test]
fn collapse_runs_of_spaces_tabs_newlines() {
    assert_eq!(normalize("a \t  b\n\n c"), "a b c");
}

#[test]
fn trim_leading_and_trailing_whitespace() {
    assert_eq!(normalize("  padded  "), "padded");
}

#[test]
fn strip_zero_width_space() {
    assert_eq!(normalize("ab\u{200B}cd"), "abcd");
}

#[test]
fn strip_zwnj_zwj_word_joiner_bom_mvs() {
    assert_eq!(normalize("a\u{200C}b"), "ab");
    assert_eq!(normalize("a\u{200D}b"), "ab");
    assert_eq!(normalize("a\u{2060}b"), "ab");
    assert_eq!(normalize("a\u{FEFF}b"), "ab");
    assert_eq!(normalize("a\u{180E}b"), "ab");
}

#[test]
fn zero_width_does_not_leave_a_space() {
    assert_eq!(normalize("Hello\u{200B}World"), "helloworld");
}

#[test]
fn empty_and_whitespace_only_become_empty() {
    assert_eq!(normalize(""), "");
    assert_eq!(normalize(" \t \n "), "");
}

#[test]
fn idempotent() {
    let samples = [
        "Hello World",
        "  A\u{200B}B  ",
        "",
        "naïve CAFÉ",
        common::alphabet(),
    ];
    for s in samples {
        let once = normalize(s);
        let twice = normalize(&once);
        assert_eq!(once, twice, "normalize must be idempotent on {s:?}");
        assert_eq!(once, reference::normalize(s));
    }
}

#[test]
fn matches_reference_on_crlf_and_unicode() {
    for s in ["A\r\nB", "A\rB", "Straße", "İ", "café π"] {
        assert_eq!(normalize(s), reference::normalize(s));
    }
}
