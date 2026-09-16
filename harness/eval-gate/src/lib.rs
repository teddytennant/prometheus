//! Eval gate: a rung cannot promote if a listed eval fails (spec 15.2 H11).

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub name: String,
    pub passed: bool,
}

pub fn promote(evals: &[EvalResult]) -> Result<(), String> {
    let failed: Vec<_> = evals.iter().filter(|e| !e.passed).map(|e| e.name.clone()).collect();
    if failed.is_empty() {
        Ok(())
    } else {
        Err(failed.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_fail() {
        let e = vec![
            EvalResult { name: "gsm8k".into(), passed: true },
            EvalResult { name: "arc".into(), passed: false },
        ];
        assert!(promote(&e).is_err());
        assert!(promote(&[EvalResult { name: "a".into(), passed: true }]).is_ok());
    }
}
