//! nccl-tests all_reduce busbw parser (out-of-place column of the last data row).

use crate::{Error, Result};

/// Out-of-place busbw (GB/s) of the last finite data row.
pub(crate) fn parse_busbw_gbps(nccl_stdout: &str) -> Result<f64> {
    if nccl_stdout.trim().is_empty() {
        return Err(Error::Other("empty nccl stdout".into()));
    }
    let mut last: Option<f64> = None;
    for line in nccl_stdout.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        // size count type redop root time algbw busbw #wrong  [in-place...]
        if cols.len() < 9 {
            continue;
        }
        if cols[0].parse::<u64>().is_err() {
            continue;
        }
        if cols[1].parse::<u64>().is_err() {
            continue;
        }
        match cols[7].parse::<f64>() {
            Ok(v) if v.is_finite() => last = Some(v),
            _ => continue,
        }
    }
    last.ok_or_else(|| Error::Other("no busbw data row".into()))
}
