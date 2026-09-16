use crate::{Error, Format};

const MATH_ENV_BEGIN: &[&str] = &[
    "\\begin{align}",
    "\\begin{align*}",
    "\\begin{alignat}",
    "\\begin{alignat*}",
    "\\begin{displaymath}",
    "\\begin{eqnarray}",
    "\\begin{eqnarray*}",
    "\\begin{equation}",
    "\\begin{equation*}",
    "\\begin{flalign}",
    "\\begin{flalign*}",
    "\\begin{gather}",
    "\\begin{gather*}",
    "\\begin{math}",
    "\\begin{multline}",
    "\\begin{multline*}",
];

pub(crate) fn extract_latex(bytes: &[u8]) -> Result<String, Error> {
    let src = std::str::from_utf8(bytes).map_err(|_| Error::Unparseable {
        format: Format::Latex,
    })?;
    Ok(strip_comments(src))
}

/// Drop TeX comments (`%` to end of line) while keeping escaped `\%`.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && (i == 0 || bytes[i - 1] != b'\\') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| src.to_owned())
}

pub(crate) fn preserves_latex_math(text: &str) -> bool {
    if text.contains("$$") {
        return true;
    }
    if MATH_ENV_BEGIN.iter().any(|needle| text.contains(needle)) {
        return true;
    }
    has_inline_dollar_math(text)
}

fn has_inline_dollar_math(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && !is_escaped(bytes, i) {
            let mut j = i + 1;
            while j < bytes.len() {
                if bytes[j] == b'$' && !is_escaped(bytes, j) {
                    return true;
                }
                j += 1;
            }
            return false;
        }
        i += 1;
    }
    false
}

fn is_escaped(bytes: &[u8], i: usize) -> bool {
    i > 0 && bytes[i - 1] == b'\\'
}
