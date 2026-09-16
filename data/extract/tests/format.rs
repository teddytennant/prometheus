//! Group: Format::from_path and Format::from_media_type (implemented; may pass).

use prometheus_extract::Format;

#[test]
fn from_path_html_extensions() {
    assert_eq!(Format::from_path("page.html"), Some(Format::Html));
    assert_eq!(Format::from_path("page.htm"), Some(Format::Html));
    assert_eq!(Format::from_path("page.xhtml"), Some(Format::Html));
    assert_eq!(
        Format::from_path("dir/nested/Page.HTML"),
        Some(Format::Html)
    );
}

#[test]
fn from_path_pdf_and_latex_extensions() {
    assert_eq!(Format::from_path("tiny.pdf"), Some(Format::Pdf));
    assert_eq!(Format::from_path("/tmp/Tiny.PDF"), Some(Format::Pdf));
    assert_eq!(Format::from_path("math.tex"), Some(Format::Latex));
    assert_eq!(Format::from_path("math.latex"), Some(Format::Latex));
    assert_eq!(Format::from_path("paper.TEX"), Some(Format::Latex));
}

#[test]
fn from_path_unknown_or_missing_extension_is_none() {
    assert_eq!(Format::from_path("notes.txt"), None);
    assert_eq!(Format::from_path("README"), None);
    assert_eq!(Format::from_path(""), None);
    assert_eq!(Format::from_path("archive.html.bak"), None);
}

#[test]
fn from_path_uses_final_extension() {
    assert_eq!(Format::from_path("my.file.tex"), Some(Format::Latex));
    assert_eq!(Format::from_path("a.b.c.pdf"), Some(Format::Pdf));
}

#[test]
fn from_media_type_known_types() {
    assert_eq!(Format::from_media_type("text/html"), Some(Format::Html));
    assert_eq!(
        Format::from_media_type("application/xhtml+xml"),
        Some(Format::Html)
    );
    assert_eq!(
        Format::from_media_type("application/pdf"),
        Some(Format::Pdf)
    );
    assert_eq!(
        Format::from_media_type("application/x-latex"),
        Some(Format::Latex)
    );
    assert_eq!(
        Format::from_media_type("application/x-tex"),
        Some(Format::Latex)
    );
    assert_eq!(Format::from_media_type("text/x-latex"), Some(Format::Latex));
    assert_eq!(Format::from_media_type("text/x-tex"), Some(Format::Latex));
}

#[test]
fn from_media_type_unknown_is_none() {
    assert_eq!(Format::from_media_type("text/plain"), None);
    assert_eq!(Format::from_media_type("application/json"), None);
    assert_eq!(Format::from_media_type(""), None);
}
