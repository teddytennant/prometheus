//! Group: PDF extracts text (golden).

mod common;

use prometheus_extract::extract;
use prometheus_extract::Format;

#[test]
fn pdf_golden_extracts_hello_pdf() {
    let doc = extract(common::TINY_PDF, Format::Pdf).expect("pdf extract");
    let phrases = common::phrases_from_expected(common::TINY_EXPECTED);
    common::assert_phrases_in_order(&doc.text, &phrases);
    common::assert_meta(&doc, Format::Pdf, None, common::TINY_PDF.len() as u64);
}

#[test]
fn pdf_extracted_text_is_not_raw_header_only() {
    let doc = extract(common::TINY_PDF, Format::Pdf).expect("pdf extract");
    assert!(
        common::collapse_ws(&doc.text).contains("Hello PDF"),
        "expected visible PDF string 'Hello PDF', got {:?}",
        doc.text
    );
    assert!(
        !doc.text.trim().starts_with("%PDF"),
        "extracted text should not be the raw PDF header, got {:?}",
        doc.text
    );
}
