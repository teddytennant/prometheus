//! Canonicalization, exact rational parse, and TinyLean whitespace.

use crate::{Error, Result};

pub(crate) fn canonicalize(s: &str) -> String {
    let stripped = s.replace('$', "");
    let stripped = stripped.replace("\\left", "");
    let stripped = stripped.replace("\\right", "");
    stripped.chars().filter(|c| !c.is_whitespace()).collect()
}

pub(crate) fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn equivalent(left: &str, right: &str) -> bool {
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

pub(crate) fn tiny_kernel_check(theorem: &str, proof: &str) -> Result<bool> {
    let th = collapse_ws(theorem);
    let pr = collapse_ws(proof);
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
