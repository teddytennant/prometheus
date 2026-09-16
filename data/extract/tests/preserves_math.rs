//! Group: preserves_latex_math on known strings (must fail on the stub).

use prometheus_extract::preserves_latex_math;

#[test]
fn dollar_span_is_math() {
    assert!(preserves_latex_math("the value $x$ is small"));
}

#[test]
fn double_dollar_span_is_math() {
    assert!(preserves_latex_math("display $$x^2$$ here"));
}

#[test]
fn equation_environment_is_math() {
    assert!(preserves_latex_math(
        "\\begin{equation}\\alpha\\end{equation}"
    ));
}

#[test]
fn align_environment_is_math() {
    assert!(preserves_latex_math("\\begin{align}a&=b\\end{align}"));
}

#[test]
fn plain_text_is_not_math() {
    assert!(!preserves_latex_math("no math here"));
}

#[test]
fn empty_string_is_not_math() {
    assert!(!preserves_latex_math(""));
}

#[test]
fn document_environment_is_not_math() {
    assert!(!preserves_latex_math(
        "\\begin{document}hello\\end{document}"
    ));
}
