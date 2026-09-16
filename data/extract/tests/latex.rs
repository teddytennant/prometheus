//! Group: LaTeX math remains as source; preserves_latex_math is true.

mod common;

use prometheus_extract::{extract, preserves_latex_math, Format};

#[test]
fn latex_keeps_inline_display_and_environment_math_as_source() {
    let doc = extract(common::MATH_TEX, Format::Latex).expect("latex extract");
    let phrases = common::phrases_from_expected(common::MATH_EXPECTED);
    common::assert_phrases_in_order(&doc.text, &phrases);
    common::assert_meta(&doc, Format::Latex, None, common::MATH_TEX.len() as u64);
}

#[test]
fn latex_does_not_unicode_approximate_math() {
    let doc = extract(common::MATH_TEX, Format::Latex).expect("latex extract");
    // Source forms that must survive:
    assert!(doc.text.contains("$E=mc^2$"), "got {:?}", doc.text);
    assert!(doc.text.contains("\\int_0^1"), "got {:?}", doc.text);
    assert!(doc.text.contains("\\alpha"), "got {:?}", doc.text);
    assert!(doc.text.contains("\\beta"), "got {:?}", doc.text);
    assert!(doc.text.contains("\\begin{equation}"), "got {:?}", doc.text);
    assert!(doc.text.contains("$$"), "got {:?}", doc.text);
    // Unicode approximations that must not appear:
    common::assert_forbidden(&doc.text, &["α", "β", "∑", "∫", "²"]);
}

#[test]
fn latex_extracted_text_preserves_latex_math() {
    let doc = extract(common::MATH_TEX, Format::Latex).expect("latex extract");
    assert!(
        preserves_latex_math(&doc.text),
        "preserves_latex_math must be true on extracted LaTeX, text={:?}",
        doc.text
    );
}

#[test]
fn latex_inline_snippet_round_trips_source() {
    let src = b"Plain $x^2$ and $$y$$.";
    let doc = extract(src, Format::Latex).expect("latex extract");
    assert!(doc.text.contains("$x^2$"), "got {:?}", doc.text);
    assert!(
        doc.text.contains("$$y$$") || doc.text.contains("$$"),
        "got {:?}",
        doc.text
    );
    assert!(!doc.text.contains('²'), "got {:?}", doc.text);
    assert!(preserves_latex_math(&doc.text));
}
