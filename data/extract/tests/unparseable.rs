//! Group: corrupt / undecodable bytes -> Error::Unparseable { format }.

mod common;

use prometheus_extract::{extract, extract_from, Format};

#[test]
fn invalid_utf8_html_is_unparseable() {
    common::assert_err_unparseable(extract(b"<p>\xff\xfe</p>", Format::Html), Format::Html);
}

#[test]
fn invalid_utf8_latex_is_unparseable() {
    common::assert_err_unparseable(extract(b"hello \xff $x$", Format::Latex), Format::Latex);
}

#[test]
fn non_pdf_bytes_as_pdf_are_unparseable() {
    common::assert_err_unparseable(extract(b"this is not a pdf", Format::Pdf), Format::Pdf);
}

#[test]
fn truncated_pdf_is_unparseable() {
    common::assert_err_unparseable(extract(b"%PDF-1.4\n1 0 obj\n", Format::Pdf), Format::Pdf);
}

#[test]
fn extract_from_garbage_pdf_path_is_unparseable() {
    common::assert_err_unparseable(
        extract_from(b"not-a-pdf", Some("tiny.pdf"), None),
        Format::Pdf,
    );
}

#[test]
fn extract_from_invalid_utf8_html_media_type_is_unparseable() {
    common::assert_err_unparseable(
        extract_from(b"<html>\xff</html>", None, Some("text/html")),
        Format::Html,
    );
}
