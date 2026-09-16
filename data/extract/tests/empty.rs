//! Group: empty input -> Error::Empty.

mod common;

use prometheus_extract::{extract, extract_from, Format};

#[test]
fn extract_empty_html_is_empty() {
    common::assert_err_empty(extract(b"", Format::Html));
}

#[test]
fn extract_empty_pdf_is_empty() {
    common::assert_err_empty(extract(b"", Format::Pdf));
}

#[test]
fn extract_empty_latex_is_empty() {
    common::assert_err_empty(extract(b"", Format::Latex));
}

#[test]
fn extract_from_empty_bytes_with_html_path_is_empty() {
    common::assert_err_empty(extract_from(b"", Some("page.html"), None));
}

#[test]
fn extract_from_empty_bytes_with_pdf_media_type_is_empty() {
    common::assert_err_empty(extract_from(b"", None, Some("application/pdf")));
}

#[test]
fn extract_from_empty_bytes_with_tex_path_is_empty() {
    common::assert_err_empty(extract_from(b"", Some("math.tex"), None));
}
