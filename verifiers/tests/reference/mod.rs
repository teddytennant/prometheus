//! Independent slow D2 reference. Production must never call this module.
//!
//! # Canonicalize (`SymbolicMath::canonicalize`)
//! 1. Delete every `$`.
//! 2. Delete every `\left` and `\right` substring (TeX delimiter commands).
//! 3. Delete every Unicode whitespace character (`char::is_whitespace`).
//! The result is not otherwise rewritten (`\frac` stays `\frac`).
//!
//! # Equivalence (`SymbolicMath::equivalent`)
//! After canonicalize, identical strings are equivalent. Otherwise both
//! sides parse as rationals over this grammar (no variables):
//!
//! ```text
//! expr     = term { ('+' | '-') term }
//! term     = unary { ('*' | '/') unary }
//! unary    = '-' unary | primary
//! primary  = number | frac | '(' expr ')'
//! frac     = '\frac' '{' expr '}' '{' expr '}'
//! number   = digits [ '.' digits ] | '.' digits
//! ```
//!
//! Values are exact `i128` rationals reduced by gcd. `1/2`, `0.5`, and
//! `\frac{1}{2}` are the same. `1+2` equals `3`. If either side fails to
//! parse fully, they are not equivalent.
//!
//! # TinyLean (`TinyKernel::check`, no `lean` binary)
//! Normalize theorem/proof by trimming and collapsing whitespace to single
//! spaces. Two theorems:
//! - `true` is proved by `trivial`
//! - `id` is proved by `fun x => x`
//! Empty theorem/proof or an unknown theorem is `Error::Lean`. A known
//! theorem with any other non-empty proof is `Ok(false)` (planted wrong).
//!
//! # Grid
//! Exact `cells` match. At most `GRID_PASS_K` predictions. Empty means no
//! rows or a row of length 0. Ragged means row lengths differ.
//!
//! # Market
//! `ln p(outcome) - ln p_market(outcome)` with `p` clamped into
//! `[PROB_EPS, 1-PROB_EPS]`. `p` is P(yes); if `outcome == false` use `1-p`.
//! `passed` iff the relative score is strictly positive.

#![allow(dead_code)]

use std::collections::BTreeMap;

use prometheus_envs::{Backend, Image, NowMs, Pool, ToolRequest, GRADER_ROOT};
use prometheus_verifiers::{
    Answer, CodeTask, Error, Evidence, EvidenceKind, Grid, LeanTask, MarketTask, MathTask, Mutant,
    Result, RewardRequest, RewardResponse, TestRun, GRID_PASS_K, PROB_EPS,
};
use sha2::{Digest, Sha256};

const HEX: &[u8] = b"0123456789abcdef";

pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(&Sha256::digest(bytes))
}

pub fn is_grader_path(path: &str) -> bool {
    path == GRADER_ROOT || path.starts_with("/grader/")
}

pub fn canonicalize(s: &str) -> String {
    let stripped = s.replace('$', "");
    let stripped = stripped.replace("\\left", "");
    let stripped = stripped.replace("\\right", "");
    stripped.chars().filter(|c| !c.is_whitespace()).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rat {
    n: i128,
    d: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    if a < 0 {
        a = -a;
    }
    if b < 0 {
        b = -b;
    }
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    if a == 0 {
        1
    } else {
        a
    }
}

fn rat(n: i128, d: i128) -> Option<Rat> {
    if d == 0 {
        return None;
    }
    let g = gcd(n, d);
    let mut n = n / g;
    let mut d = d / g;
    if d < 0 {
        n = -n;
        d = -d;
    }
    Some(Rat { n, d })
}

impl Rat {
    fn add(self, other: Rat) -> Option<Rat> {
        let n = self
            .n
            .checked_mul(other.d)?
            .checked_add(other.n.checked_mul(self.d)?)?;
        let d = self.d.checked_mul(other.d)?;
        rat(n, d)
    }
    fn sub(self, other: Rat) -> Option<Rat> {
        self.add(Rat {
            n: other.n.checked_neg()?,
            d: other.d,
        })
    }
    fn mul(self, other: Rat) -> Option<Rat> {
        rat(self.n.checked_mul(other.n)?, self.d.checked_mul(other.d)?)
    }
    fn div(self, other: Rat) -> Option<Rat> {
        if other.n == 0 {
            return None;
        }
        rat(self.n.checked_mul(other.d)?, self.d.checked_mul(other.n)?)
    }
    fn neg(self) -> Option<Rat> {
        Some(Rat {
            n: self.n.checked_neg()?,
            d: self.d,
        })
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn bump(&mut self) {
        self.i += 1;
    }
    fn eat(&mut self, c: u8) -> Option<()> {
        if self.peek() == Some(c) {
            self.bump();
            Some(())
        } else {
            None
        }
    }
    fn eat_slice(&mut self, p: &[u8]) -> bool {
        if self.s[self.i..].starts_with(p) {
            self.i += p.len();
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Option<Rat> {
        let mut acc = self.term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.bump();
                    acc = acc.add(self.term()?)?;
                }
                Some(b'-') => {
                    self.bump();
                    acc = acc.sub(self.term()?)?;
                }
                _ => break,
            }
        }
        Some(acc)
    }
    fn term(&mut self) -> Option<Rat> {
        let mut acc = self.unary()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.bump();
                    acc = acc.mul(self.unary()?)?;
                }
                Some(b'/') => {
                    self.bump();
                    acc = acc.div(self.unary()?)?;
                }
                _ => break,
            }
        }
        Some(acc)
    }
    fn unary(&mut self) -> Option<Rat> {
        if self.eat(b'-').is_some() {
            return self.unary()?.neg();
        }
        self.primary()
    }
    fn primary(&mut self) -> Option<Rat> {
        if self.eat_slice(br"\frac") {
            self.eat(b'{')?;
            let a = self.expr()?;
            self.eat(b'}')?;
            self.eat(b'{')?;
            let b = self.expr()?;
            self.eat(b'}')?;
            return a.div(b);
        }
        if self.eat(b'(').is_some() {
            let e = self.expr()?;
            self.eat(b')')?;
            return Some(e);
        }
        self.number()
    }
    fn number(&mut self) -> Option<Rat> {
        let mut saw_int = false;
        let mut int_part: i128 = 0;
        while let Some(d @ b'0'..=b'9') = self.peek() {
            saw_int = true;
            self.bump();
            int_part = int_part
                .checked_mul(10)?
                .checked_add(i128::from(d - b'0'))?;
        }
        if self.peek() == Some(b'.') {
            self.bump();
            let mut frac: i128 = 0;
            let mut den: i128 = 1;
            let mut saw_frac = false;
            while let Some(d @ b'0'..=b'9') = self.peek() {
                saw_frac = true;
                self.bump();
                frac = frac.checked_mul(10)?.checked_add(i128::from(d - b'0'))?;
                den = den.checked_mul(10)?;
            }
            if !saw_int && !saw_frac {
                return None;
            }
            return rat(int_part, 1)?.add(rat(frac, den)?);
        }
        if !saw_int {
            return None;
        }
        rat(int_part, 1)
    }
}

fn eval_canon(s: &str) -> Option<Rat> {
    let mut p = Parser {
        s: s.as_bytes(),
        i: 0,
    };
    let v = p.expr()?;
    if p.i != p.s.len() {
        return None;
    }
    Some(v)
}

pub fn equivalent(left: &str, right: &str) -> bool {
    let l = canonicalize(left);
    let r = canonicalize(right);
    if l == r {
        return true;
    }
    match (eval_canon(&l), eval_canon(&r)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

pub fn tiny_normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn tiny_kernel_check(theorem: &str, proof: &str) -> Result<bool> {
    let th = tiny_normalize(theorem);
    let pr = tiny_normalize(proof);
    if th.is_empty() {
        return Err(Error::Lean("empty theorem".into()));
    }
    if pr.is_empty() {
        return Err(Error::Lean("empty proof".into()));
    }
    let expected = match th.as_str() {
        "true" => "trivial",
        "id" => "fun x => x",
        _ => return Err(Error::Lean(format!("unknown theorem: {th}"))),
    };
    Ok(pr == expected)
}

pub fn grid_valid(g: &Grid) -> Result<()> {
    if g.cells.is_empty() || g.cells.iter().any(|r| r.is_empty()) {
        return Err(Error::BadGrid);
    }
    let w = g.cols();
    if g.cells.iter().any(|r| r.len() != w) {
        return Err(Error::BadGrid);
    }
    Ok(())
}

pub fn grid_match(expected: &Grid, grids: &[Grid]) -> Result<bool> {
    if grids.len() > GRID_PASS_K {
        return Err(Error::TooManyGrids);
    }
    grid_valid(expected)?;
    for g in grids {
        grid_valid(g)?;
    }
    Ok(grids.iter().any(|g| g == expected))
}

fn in_unit_interval(p: f64) -> bool {
    (0.0..=1.0).contains(&p)
}

fn clamp_prob(p: f64) -> f64 {
    p.clamp(PROB_EPS, 1.0 - PROB_EPS)
}

fn p_outcome(p_yes: f64, outcome: bool) -> f64 {
    if outcome {
        p_yes
    } else {
        1.0 - p_yes
    }
}

pub fn relative_log_score(model_p: f64, market_p: f64, outcome: bool) -> Result<f64> {
    if !in_unit_interval(model_p) {
        return Err(Error::BadProbability(model_p));
    }
    if !in_unit_interval(market_p) {
        return Err(Error::BadProbability(market_p));
    }
    let pm = clamp_prob(p_outcome(model_p, outcome));
    let pk = clamp_prob(p_outcome(market_p, outcome));
    Ok(pm.ln() - pk.ln())
}

fn base_response(req: &RewardRequest, scored_at: &str, score: f64, passed: bool) -> RewardResponse {
    let mut resp = RewardResponse::new(req.request_id.clone(), scored_at);
    resp.score = score;
    resp.passed = passed;
    resp
}

fn ev(kind: EvidenceKind, payload: &str) -> Evidence {
    Evidence {
        kind,
        hash: sha256_hex(payload.as_bytes()),
        summary: None,
    }
}

pub fn verify_symbolic(
    task: &MathTask,
    req: &RewardRequest,
    answer: &Answer,
    scored_at: &str,
) -> Result<RewardResponse> {
    let Answer::Math { latex } = answer else {
        return Err(Error::WrongKind);
    };
    let ok = equivalent(latex, &task.expected);
    let mut resp = base_response(req, scored_at, if ok { 1.0 } else { 0.0 }, ok);
    resp.evidence.push(ev(EvidenceKind::Symbolic, latex));
    Ok(resp)
}

pub fn verify_lean(
    task: &LeanTask,
    req: &RewardRequest,
    answer: &Answer,
    scored_at: &str,
) -> Result<RewardResponse> {
    let Answer::Lean { proof } = answer else {
        return Err(Error::WrongKind);
    };
    let ok = tiny_kernel_check(&task.theorem, proof)?;
    let mut resp = base_response(req, scored_at, if ok { 1.0 } else { 0.0 }, ok);
    resp.evidence.push(ev(EvidenceKind::Lean, proof));
    Ok(resp)
}

pub fn verify_grid(
    expected: &Grid,
    req: &RewardRequest,
    answer: &Answer,
    scored_at: &str,
) -> Result<RewardResponse> {
    let Answer::Grid { grids } = answer else {
        return Err(Error::WrongKind);
    };
    let ok = grid_match(expected, grids)?;
    let mut resp = base_response(req, scored_at, if ok { 1.0 } else { 0.0 }, ok);
    resp.evidence
        .push(ev(EvidenceKind::Grid, &format!("{expected:?}")));
    Ok(resp)
}

pub fn verify_market(
    task: &MarketTask,
    req: &RewardRequest,
    answer: &Answer,
    scored_at: &str,
) -> Result<RewardResponse> {
    let Answer::Forecast { p } = answer else {
        return Err(Error::WrongKind);
    };
    let score = relative_log_score(*p, task.market_p, task.outcome)?;
    Ok(base_response(req, scored_at, score, score > 0.0))
}

fn check_image_isolation(image: &Image) -> Result<()> {
    if image.agent_files.keys().any(|p| is_grader_path(p)) {
        return Err(Error::HiddenTestsVisible);
    }
    if image.hidden_tests.keys().any(|p| !is_grader_path(p)) {
        return Err(Error::HiddenTestsVisible);
    }
    Ok(())
}

fn reject_grader_writes(files: &BTreeMap<String, Vec<u8>>) -> Result<()> {
    if let Some(path) = files.keys().find(|p| is_grader_path(p)) {
        return Err(Error::TestFileWrite {
            path: (*path).clone(),
        });
    }
    Ok(())
}

fn overlay(image: &Image, files: &BTreeMap<String, Vec<u8>>, id: &str) -> Result<Image> {
    reject_grader_writes(files)?;
    let mut out = image.clone();
    out.id = prometheus_envs::ImageId(id.to_string());
    for (p, b) in files {
        out.agent_files.insert(p.clone(), b.clone());
    }
    check_image_isolation(&out)?;
    Ok(out)
}

fn run_hidden<B: Backend>(
    pool: &mut Pool<B>,
    image: Image,
    run: &TestRun,
    now: NowMs,
) -> Result<bool> {
    check_image_isolation(&image)?;
    let image_id = pool.register_image(image)?;
    let snap = pool.snapshot_from_image(&image_id, now)?;
    let fork = pool.fork_group(&snap, 1, now)?;
    let sb = fork
        .sandboxes
        .first()
        .ok_or_else(|| Error::Sandbox("empty fork".into()))?
        .clone();
    let mut treq = ToolRequest::new("grade", snap.0.clone(), run.tool);
    treq.payload = run.payload.clone();
    treq.timeout_s = Some(run.timeout_s);
    let call = pool.call(&sb, treq, now);
    let _ = pool.kill(&sb, now);
    let resp = call?;
    let want = sha256_hex(&run.expected_stdout);
    Ok(resp.ok && resp.stdout_artifact.content_hash == want)
}

fn mutant_passes<B: Backend>(
    pool: &mut Pool<B>,
    task: &CodeTask,
    mutant: &Mutant,
    now: NowMs,
) -> Result<bool> {
    let image = overlay(&task.image, &mutant.files, &format!("mut-{}", mutant.id))?;
    run_hidden(pool, image, &task.run, now)
}

pub fn verify_code<B: Backend>(
    pool: &mut Pool<B>,
    task: &CodeTask,
    req: &RewardRequest,
    answer: &Answer,
    scored_at: &str,
    now: NowMs,
) -> Result<RewardResponse> {
    let Answer::Code { files } = answer else {
        return Err(Error::WrongKind);
    };
    check_image_isolation(&task.image)?;
    reject_grader_writes(files)?;

    let mut any_mutant_failed = false;
    for m in &task.mutants {
        if mutant_passes(pool, task, m, now)? {
            return Err(Error::WeakTests { id: m.id.clone() });
        }
        any_mutant_failed = true;
    }

    let image = overlay(&task.image, files, "ans")?;
    let passed_tests = run_hidden(pool, image, &task.run, now)?;

    if passed_tests && !any_mutant_failed {
        return Err(Error::Unverifiable("no mutants".into()));
    }

    let score = if passed_tests { 1.0 } else { 0.0 };
    let mut resp = base_response(req, scored_at, score, passed_tests);
    resp.evidence
        .push(ev(EvidenceKind::HiddenTests, "hidden_tests"));
    resp.evidence.push(ev(EvidenceKind::Mutation, "mutation"));
    Ok(resp)
}
