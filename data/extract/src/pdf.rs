use lopdf::content::Operation;
use lopdf::{Document, Object};

use crate::{Error, Format};

pub(crate) fn extract_pdf(bytes: &[u8]) -> Result<String, Error> {
    let document = Document::load_mem(bytes).map_err(|_| Error::Unparseable {
        format: Format::Pdf,
    })?;
    let pages = document.get_pages();
    if pages.is_empty() {
        return Err(Error::Unparseable {
            format: Format::Pdf,
        });
    }

    let mut from_ops = String::new();
    for page_id in pages.values().copied() {
        let content =
            document
                .get_and_decode_page_content(page_id)
                .map_err(|_| Error::Unparseable {
                    format: Format::Pdf,
                })?;
        collect_ops(&content.operations, &mut from_ops);
    }

    if !from_ops.trim().is_empty() {
        return Ok(from_ops);
    }

    let page_nums: Vec<u32> = pages.keys().copied().collect();
    match document.extract_text(&page_nums) {
        Ok(text) if !text.trim().is_empty() && !text.trim_start().starts_with("%PDF") => Ok(text),
        Ok(_) => Ok(from_ops),
        Err(_) => Err(Error::Unparseable {
            format: Format::Pdf,
        }),
    }
}

fn collect_ops(ops: &[Operation], out: &mut String) {
    for op in ops {
        match op.operator.as_str() {
            "Tj" | "'" => {
                if let Some(obj) = op.operands.first() {
                    push_pdf_string(obj, out);
                }
            }
            "\"" => {
                if let Some(obj) = op.operands.last() {
                    push_pdf_string(obj, out);
                }
            }
            "TJ" => {
                if let Some(Object::Array(items)) = op.operands.first() {
                    for item in items {
                        push_pdf_string(item, out);
                    }
                }
            }
            "T*" | "Td" | "TD" if !out.is_empty() && !out.ends_with('\n') => {
                out.push('\n');
            }
            _ => {}
        }
    }
}

fn push_pdf_string(obj: &Object, out: &mut String) {
    let Object::String(bytes, _) = obj else {
        return;
    };
    match std::str::from_utf8(bytes) {
        Ok(s) => out.push_str(s),
        Err(_) => out.extend(bytes.iter().copied().map(char::from)),
    }
}
