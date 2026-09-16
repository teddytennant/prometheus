//! Group: extract_from uses path / media_type to pick Format.

mod common;

use prometheus_extract::{extract, extract_from, Format};

#[test]
fn html_from_path() {
    let doc = extract_from(common::PAGE_HTML, Some("goldens/page.html"), None)
        .expect("extract_from html path");
    common::assert_meta(
        &doc,
        Format::Html,
        Some("goldens/page.html"),
        common::PAGE_HTML.len() as u64,
    );
    common::assert_phrases_in_order(
        &doc.text,
        &common::phrases_from_expected(common::PAGE_EXPECTED),
    );
}

#[test]
fn html_from_htm_and_xhtml_paths() {
    let a = extract_from(common::PAGE_HTML, Some("page.htm"), None).expect("htm");
    let b = extract_from(common::PAGE_HTML, Some("page.xhtml"), None).expect("xhtml");
    assert_eq!(a.format, Format::Html);
    assert_eq!(b.format, Format::Html);
}

#[test]
fn html_from_media_type() {
    let doc = extract_from(common::PAGE_HTML, None, Some("text/html")).expect("media html");
    common::assert_meta(&doc, Format::Html, None, common::PAGE_HTML.len() as u64);
}

#[test]
fn html_from_xhtml_media_type() {
    let doc =
        extract_from(common::PAGE_HTML, None, Some("application/xhtml+xml")).expect("xhtml media");
    assert_eq!(doc.format, Format::Html);
}

#[test]
fn pdf_from_path_and_media_type() {
    let via_path = extract_from(common::TINY_PDF, Some("tiny.pdf"), None).expect("pdf path");
    let via_media =
        extract_from(common::TINY_PDF, None, Some("application/pdf")).expect("pdf media");
    common::assert_meta(
        &via_path,
        Format::Pdf,
        Some("tiny.pdf"),
        common::TINY_PDF.len() as u64,
    );
    common::assert_meta(&via_media, Format::Pdf, None, common::TINY_PDF.len() as u64);
    common::assert_phrases_in_order(&via_path.text, &["Hello PDF"]);
    common::assert_phrases_in_order(&via_media.text, &["Hello PDF"]);
}

#[test]
fn latex_from_path_and_media_types() {
    let via_path = extract_from(common::MATH_TEX, Some("math.tex"), None).expect("tex path");
    let via_latex_ext =
        extract_from(common::MATH_TEX, Some("math.latex"), None).expect("latex ext");
    let via_media =
        extract_from(common::MATH_TEX, None, Some("application/x-latex")).expect("latex media");
    assert_eq!(via_path.format, Format::Latex);
    assert_eq!(via_latex_ext.format, Format::Latex);
    assert_eq!(via_media.format, Format::Latex);
    assert_eq!(via_path.source.as_deref(), Some("math.tex"));
    assert!(
        via_path.text.contains("$E=mc^2$"),
        "got {:?}",
        via_path.text
    );
}

#[test]
fn path_and_media_type_agreeing_html() {
    let doc = extract_from(common::PAGE_HTML, Some("page.html"), Some("text/html"))
        .expect("agreeing hints");
    assert_eq!(doc.format, Format::Html);
    assert_eq!(doc.source.as_deref(), Some("page.html"));
}

#[test]
fn extract_from_matches_extract_for_html() {
    let via_extract = extract(common::PAGE_HTML, Format::Html).expect("extract");
    let via_from = extract_from(common::PAGE_HTML, Some("page.html"), None).expect("from");
    assert_eq!(via_extract.text, via_from.text);
    assert_eq!(via_extract.content_hash, via_from.content_hash);
    assert_eq!(via_extract.format, via_from.format);
}

#[test]
fn neither_path_nor_media_type_is_unsupported() {
    common::assert_err_unsupported(extract_from(b"<p>hi</p>", None, None));
}

#[test]
fn unknown_path_and_media_type_is_unsupported() {
    common::assert_err_unsupported(extract_from(b"hello", Some("notes.txt"), None));
    common::assert_err_unsupported(extract_from(b"hello", None, Some("text/plain")));
}
