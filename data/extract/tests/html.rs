//! Group: HTML strips tags and keeps text.

mod common;

use prometheus_extract::extract;
use prometheus_extract::Format;

#[test]
fn html_strips_tags_keeps_text() {
    let bytes = b"<html><body><h1>Title One</h1><p>Hello world.</p></body></html>";
    let doc = extract(bytes, Format::Html).expect("html extract");
    common::assert_phrases_in_order(&doc.text, &["Title One", "Hello world."]);
    common::assert_forbidden(&doc.text, &["<h1>", "<p>", "</p>", "<html>", "<body>"]);
    common::assert_meta(&doc, Format::Html, None, bytes.len() as u64);
}

#[test]
fn html_decodes_entities() {
    let doc = extract(b"<p>A &amp; B &lt; C &gt; D</p>", Format::Html).expect("html extract");
    assert!(
        doc.text.contains("A & B"),
        "ampersand entity must decode, got {:?}",
        doc.text
    );
    assert!(
        doc.text.contains("B < C") || common::collapse_ws(&doc.text).contains("B < C"),
        "less-than entity must decode, got {:?}",
        doc.text
    );
}

#[test]
fn html_drops_script_style_and_comments() {
    let html = b"<html><head><style>body { color: red; }</style>\
<script>alert(\"nope\")</script></head>\
<body><!-- secret-comment-should-not-appear --><p>Keep me</p></body></html>";
    let doc = extract(html, Format::Html).expect("html extract");
    assert!(doc.text.contains("Keep me"), "got {:?}", doc.text);
    common::assert_forbidden(
        &doc.text,
        &[
            "alert",
            "nope",
            "color: red",
            "secret-comment-should-not-appear",
            "<script>",
            "<style>",
        ],
    );
}

#[test]
fn html_keeps_nested_tag_inner_text() {
    let doc = extract(b"<p>a<b>b</b>c</p>", Format::Html).expect("html extract");
    common::assert_phrases_in_order(&doc.text, &["a", "b", "c"]);
}

#[test]
fn html_keeps_utf8_text() {
    let bytes = "<p>café π</p>".as_bytes();
    let doc = extract(bytes, Format::Html).expect("html extract");
    assert!(doc.text.contains("café"), "got {:?}", doc.text);
    assert!(doc.text.contains('π'), "got {:?}", doc.text);
}

#[test]
fn html_preserves_inline_latex_math() {
    let doc = extract(b"<p>Energy $E=mc^2$ holds.</p>", Format::Html).expect("html extract");
    assert!(
        doc.text.contains("$E=mc^2$"),
        "inline LaTeX in HTML must stay as source, got {:?}",
        doc.text
    );
    assert!(
        !doc.text.contains('²'),
        "must not rewrite $E=mc^2$ to Unicode superscript, got {:?}",
        doc.text
    );
}

#[test]
fn html_non_empty_markup_is_not_error_empty() {
    let doc = extract(b"<html></html>", Format::Html).expect("empty-ish html is still input");
    common::assert_meta(&doc, Format::Html, None, b"<html></html>".len() as u64);
}
