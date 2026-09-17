//! Group: paragraphs — blank-line split after normalize, drop empties.

mod common;
mod reference;

use prometheus_dedup::paragraphs;

#[test]
fn blank_line_split_two_paragraphs() {
    let got = paragraphs("Para one.\n\nPara two.");
    assert_eq!(got, vec!["para one.", "para two."]);
    assert_eq!(got, reference::paragraphs("Para one.\n\nPara two."));
}

#[test]
fn extra_blank_lines_and_whitespace_only_lines_are_dropped() {
    let src = "Para one.\n\n\n  \nPara two.";
    assert_eq!(paragraphs(src), vec!["para one.", "para two."]);
}

#[test]
fn intra_paragraph_newlines_collapse_to_spaces() {
    let src = "Para one\ncontinues here.\n\nPara two.";
    assert_eq!(
        paragraphs(src),
        vec!["para one continues here.", "para two."]
    );
}

#[test]
fn leading_trailing_blank_lines_dropped() {
    assert_eq!(paragraphs("\n\nOnly\n\n"), vec!["only"]);
}

#[test]
fn empty_input_is_empty_vec() {
    assert_eq!(paragraphs(""), Vec::<String>::new());
    assert_eq!(paragraphs("  \n\n  "), Vec::<String>::new());
}

#[test]
fn crlf_blank_line() {
    assert_eq!(paragraphs("A\r\n\r\nB"), vec!["a", "b"]);
}

#[test]
fn lowercase_and_zero_width_applied_before_split() {
    let src = "Hello\u{200B}World\n\nNEXT";
    assert_eq!(paragraphs(src), vec!["helloworld", "next"]);
}

#[test]
fn drop_empties_after_collapse() {
    // A block of only zero-width / whitespace is not a paragraph.
    let src = "Keep\n\n\u{200B}\n\nAlso keep";
    assert_eq!(paragraphs(src), vec!["keep", "also keep"]);
}

#[test]
fn matches_reference_on_fixture_corpus() {
    for d in common::fixture_corpus() {
        assert_eq!(
            paragraphs(&d.text),
            reference::paragraphs(&d.text),
            "paragraphs mismatch on {}",
            d.id
        );
    }
}
