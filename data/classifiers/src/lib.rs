//! Quality / language / register heuristics (spec 7, 15.5 B3).

#[derive(Debug, Clone, PartialEq)]
pub struct Scores {
    pub quality: f64,
    pub english: f64,
    pub code_like: f64,
}

pub fn score(text: &str) -> Scores {
    let n = text.chars().count().max(1) as f64;
    let letters = text.chars().filter(|c| c.is_ascii_alphabetic()).count() as f64;
    let spaces = text.chars().filter(|c| c.is_whitespace()).count() as f64;
    let punct = text.chars().filter(|c| matches!(c, '{' | '}' | ';' | '(')).count() as f64;
    let quality = (letters / n) * (1.0 - (spaces / n - 0.15).abs().min(1.0));
    Scores {
        quality: quality.clamp(0.0, 1.0),
        english: letters / n,
        code_like: (punct / n * 8.0).clamp(0.0, 1.0),
    }
}

pub fn keep(text: &str, min_quality: f64) -> bool {
    score(text).quality >= min_quality && text.len() >= 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prose_beats_garbage() {
        let p = score("This is a reasonably long English sentence about science.");
        let g = score("@@@@####$$$$%%%%^^^^");
        assert!(p.quality > g.quality);
        assert!(keep("This is a reasonably long English sentence about science.", 0.2));
        assert!(!keep("@@", 0.2));
    }
}
