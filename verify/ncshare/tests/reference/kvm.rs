//! Slow node-facts kvm parser.

use prometheus_verify_ncshare::{Error, Result};

/// True when facts show `/dev/kvm` as a character device or `kvm: yes`.
pub fn kvm_available(node_facts: &str) -> Result<bool> {
    if node_facts.trim().is_empty() {
        return Err(Error::Other("empty node facts".into()));
    }

    let mut explicit: Option<bool> = None;
    let mut char_device = false;
    let mut absent = false;
    let mut mentioned = false;

    for line in node_facts.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let lower = line.to_ascii_lowercase();

        if let Some(rest) = strip_kvm_key(&lower) {
            mentioned = true;
            let value = rest.trim();
            explicit = Some(match value {
                "yes" | "true" | "1" | "present" => true,
                "no" | "false" | "0" | "absent" => false,
                other => {
                    return Err(Error::Other(format!("unrecognized kvm value: {other}")));
                }
            });
            continue;
        }

        if lower.contains("/dev/kvm") {
            mentioned = true;
            let first = lower.split_whitespace().next().unwrap_or("");
            if first.starts_with('c') && !first.starts_with("cannot") {
                char_device = true;
            }
            if lower.contains("character device") || lower.contains("character special") {
                char_device = true;
            }
            if lower.contains("no such file")
                || lower.contains("not found")
                || lower.contains("cannot stat")
                || lower.contains("missing")
                || lower.contains("absent")
            {
                absent = true;
            }
        }
    }

    if let Some(v) = explicit {
        return Ok(v);
    }
    if char_device {
        return Ok(true);
    }
    if absent {
        return Ok(false);
    }
    if mentioned {
        return Err(Error::Other(
            "kvm mentioned but not a character device and not an absence line".into(),
        ));
    }
    Err(Error::Other("node facts do not mention kvm".into()))
}

fn strip_kvm_key(lower_line: &str) -> Option<&str> {
    let trimmed = lower_line.trim();
    if let Some(rest) = trimmed.strip_prefix("kvm:") {
        return Some(rest);
    }
    None
}
