//! Eval-set n-gram decontamination (spec 7, 15.5 B7).

use std::collections::HashSet;

pub fn grams(text: &str, n: usize) -> HashSet<String> {
    let c: Vec<char> = text.chars().collect();
    if c.len() < n {
        return HashSet::new();
    }
    (0..=c.len() - n).map(|i| c[i..i + n].iter().collect()).collect()
}

pub fn overlap_frac(doc: &str, eval: &str, n: usize) -> f64 {
    let e = grams(eval, n);
    if e.is_empty() {
        return 0.0;
    }
    let d = grams(doc, n);
    let hit = d.intersection(&e).count();
    hit as f64 / e.len() as f64
}

pub fn contaminated(doc: &str, evals: &[&str], n: usize, thresh: f64) -> bool {
    evals.iter().any(|e| overlap_frac(doc, e, n) >= thresh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_copied_eval() {
        let eval = "the capital of france is paris and it is well known";
        assert!(contaminated(eval, &[eval], 5, 0.8));
        assert!(!contaminated("unrelated document about cats", &[eval], 5, 0.5));
    }
}
