use scraper::{ElementRef, Html, Node};

use crate::{Error, Format};

const SKIP_TAGS: &[&str] = &["script", "style", "noscript", "template"];

const BLOCK_TAGS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
];

pub(crate) fn extract_html(bytes: &[u8]) -> Result<String, Error> {
    let src = std::str::from_utf8(bytes).map_err(|_| Error::Unparseable {
        format: Format::Html,
    })?;
    let document = Html::parse_document(src);
    let mut out = String::new();
    collect(document.root_element(), &mut out, false);
    Ok(out)
}

fn collect(el: ElementRef<'_>, out: &mut String, skip: bool) {
    let name = el.value().name();
    let skip_here = skip || SKIP_TAGS.contains(&name);
    if !skip_here && is_block(name) {
        push_break(out);
    }
    for child in el.children() {
        match child.value() {
            Node::Text(text) if !skip_here => out.push_str(text),
            Node::Element(_) => {
                if let Some(child_el) = ElementRef::wrap(child) {
                    collect(child_el, out, skip_here);
                }
            }
            _ => {}
        }
    }
    if !skip_here && is_block(name) {
        push_break(out);
    }
}

fn is_block(name: &str) -> bool {
    BLOCK_TAGS.contains(&name)
}

fn push_break(out: &mut String) {
    if out.is_empty() || out.ends_with('\n') {
        return;
    }
    out.push('\n');
}
