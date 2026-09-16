//! Group: golden corpus (HTML page, tiny PDF, LaTeX with math).

mod common;

use prometheus_extract::{extract, extract_from, preserves_latex_math, Format};

#[test]
fn golden_html_page() {
    let doc = extract(common::PAGE_HTML, Format::Html).expect("html golden");
    common::assert_phrases_in_order(
        &doc.text,
        &common::phrases_from_expected(common::PAGE_EXPECTED),
    );
    common::assert_forbidden(
        &doc.text,
        &[
            "<h1>",
            "<p>",
            "alert",
            "nope",
            "color: red",
            "secret-comment-should-not-appear",
        ],
    );
    assert!(
        doc.text.contains("$x^2$"),
        "HTML golden must keep inline math source, got {:?}",
        doc.text
    );
    common::assert_meta(&doc, Format::Html, None, common::PAGE_HTML.len() as u64);
}

#[test]
fn golden_pdf() {
    let doc = extract(common::TINY_PDF, Format::Pdf).expect("pdf golden");
    common::assert_phrases_in_order(
        &doc.text,
        &common::phrases_from_expected(common::TINY_EXPECTED),
    );
    common::assert_meta(&doc, Format::Pdf, None, common::TINY_PDF.len() as u64);
}

#[test]
fn golden_latex_math() {
    let doc = extract(common::MATH_TEX, Format::Latex).expect("latex golden");
    common::assert_phrases_in_order(
        &doc.text,
        &common::phrases_from_expected(common::MATH_EXPECTED),
    );
    common::assert_forbidden(&doc.text, &["α", "β", "∑", "∫", "²"]);
    assert!(
        preserves_latex_math(&doc.text),
        "golden LaTeX must still look like LaTeX math, got {:?}",
        doc.text
    );
    common::assert_meta(&doc, Format::Latex, None, common::MATH_TEX.len() as u64);
}

#[test]
fn golden_corpus_via_extract_from_paths() {
    let html = extract_from(common::PAGE_HTML, Some("page.html"), None).expect("html");
    let pdf = extract_from(common::TINY_PDF, Some("tiny.pdf"), None).expect("pdf");
    let tex = extract_from(common::MATH_TEX, Some("math.tex"), None).expect("tex");
    common::assert_phrases_in_order(
        &html.text,
        &common::phrases_from_expected(common::PAGE_EXPECTED),
    );
    common::assert_phrases_in_order(
        &pdf.text,
        &common::phrases_from_expected(common::TINY_EXPECTED),
    );
    common::assert_phrases_in_order(
        &tex.text,
        &common::phrases_from_expected(common::MATH_EXPECTED),
    );
}
