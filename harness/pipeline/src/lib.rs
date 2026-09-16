//! Build pipeline records (spec 15.3, 15.5 H5).

mod stages;

pub use stages::{Case, RunError, RunResult, Stage, Suite};

#[derive(Debug, Clone)]
pub struct Record {
    pub role: String,
    pub ok: bool,
}

#[derive(Debug, Default)]
pub struct Pipeline {
    pub records: Vec<Record>,
}

impl Pipeline {
    pub fn push(&mut self, r: Record) {
        self.records.push(r);
    }

    pub fn last_ok(&self) -> bool {
        self.records.last().map(|r| r.ok).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_roles() {
        let mut p = Pipeline::default();
        p.push(Record {
            role: "oracle".into(),
            ok: true,
        });
        p.push(Record {
            role: "impl".into(),
            ok: true,
        });
        assert!(p.last_ok());
        assert_eq!(p.records.len(), 2);
    }
}
