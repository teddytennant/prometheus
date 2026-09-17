//! Restricted ledger query language (not SQL, not embedding search).

use crate::{Error, Record, Result};

pub fn run(records: &[Record], query: &str) -> Result<Vec<Record>> {
    parse_and_filter(records, query).map_err(Error::Message)
}

enum Pred {
    ExperimentId(String),
    AuthorRole(String),
    Rung(Option<u32>),
    HypothesisContains(String),
    KindCheck,
    KindPrereg,
}

impl Pred {
    fn matches(&self, r: &Record) -> bool {
        match self {
            Pred::ExperimentId(v) => r.experiment_id == *v,
            Pred::AuthorRole(v) => r.author_role == *v,
            Pred::Rung(v) => r.rung == *v,
            Pred::HypothesisContains(v) => r.hypothesis.contains(v),
            Pred::KindCheck => r.hypothesis.starts_with("CHECK:"),
            Pred::KindPrereg => r.hypothesis.starts_with("PREREG:"),
        }
    }
}

fn parse_and_filter(records: &[Record], query: &str) -> std::result::Result<Vec<Record>, String> {
    let q = query.trim();
    if q.is_empty() || q == "*" || q == "all" {
        return Ok(records.to_vec());
    }
    let upper = q.to_ascii_uppercase();
    if upper.contains("SELECT ")
        || upper.contains("DROP ")
        || upper.contains("INSERT ")
        || upper.contains("DELETE ")
    {
        return Err("query language is not SQL".into());
    }
    let parts = split_and(q)?;
    let mut preds = Vec::new();
    for p in parts {
        preds.push(parse_pred(&p)?);
    }
    Ok(records
        .iter()
        .filter(|r| preds.iter().all(|p| p.matches(r)))
        .cloned()
        .collect())
}

fn split_and(q: &str) -> std::result::Result<Vec<String>, String> {
    let chars: Vec<char> = q.chars().collect();
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '"' {
            in_quotes = !in_quotes;
            cur.push('"');
            i += 1;
            continue;
        }
        if !in_quotes && chars[i] == ' ' && i + 5 <= chars.len() {
            let slice: String = chars[i..i + 5].iter().collect();
            if slice == " AND " {
                parts.push(cur.trim().to_string());
                cur.clear();
                i += 5;
                continue;
            }
        }
        cur.push(chars[i]);
        i += 1;
    }
    if in_quotes {
        return Err("unterminated quote".into());
    }
    parts.push(cur.trim().to_string());
    if parts.iter().any(|p| p.is_empty()) {
        return Err("empty predicate".into());
    }
    Ok(parts)
}

fn parse_pred(p: &str) -> std::result::Result<Pred, String> {
    let p = p.trim();
    if let Some(v) = p.strip_prefix("kind=") {
        return match unquote(v)?.as_str() {
            "check" => Ok(Pred::KindCheck),
            "prereg" => Ok(Pred::KindPrereg),
            other => Err(format!("unknown kind {other}")),
        };
    }
    if let Some(v) = p.strip_prefix("experiment_id=") {
        return Ok(Pred::ExperimentId(unquote(v)?));
    }
    if let Some(v) = p.strip_prefix("author_role=") {
        return Ok(Pred::AuthorRole(unquote(v)?));
    }
    if let Some(v) = p.strip_prefix("rung=") {
        let v = unquote(v)?;
        if v == "none" {
            return Ok(Pred::Rung(None));
        }
        let n: u32 = v.parse().map_err(|_| format!("bad rung {v}"))?;
        return Ok(Pred::Rung(Some(n)));
    }
    if let Some(v) = p.strip_prefix("hypothesis~") {
        return Ok(Pred::HypothesisContains(unquote(v)?));
    }
    Err(format!("unknown predicate {p}"))
}

fn unquote(s: &str) -> std::result::Result<String, String> {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix('"').and_then(|x| x.strip_suffix('"')) {
        return Ok(inner.to_string());
    }
    if s.split_whitespace().nth(1).is_some() {
        return Err("unquoted whitespace".into());
    }
    Ok(s.to_string())
}
