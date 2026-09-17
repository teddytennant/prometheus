//! `harness` CLI (spec 15.2, 15.5 H12).
//!
//! Grammar matches [`prometheus_ops::cli`]: `harness [--json] status`.
//! In-process callers pass open [`prometheus_ops::Ops`] and
//! [`prometheus_raft::Cluster`] handles into `cli`. This binary is the same
//! dispatcher for operators; it needs those handles from the library API.

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = argv.iter().map(String::as_str).collect();
    match dispatch(&args) {
        Ok(out) => print!("{out}"),
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

fn dispatch(args: &[&str]) -> Result<String, String> {
    let json = args.contains(&"--json");
    let rest: Vec<&str> = args.iter().copied().filter(|a| *a != "--json").collect();
    match rest.as_slice() {
        ["status"] => {
            let _ = json;
            Err(
                "harness status requires open Ops and Cluster handles; use prometheus_ops::cli"
                    .into(),
            )
        }
        [] | ["-h"] | ["--help"] => Ok("usage: harness [--json] status\n".into()),
        other => Err(format!("unknown command: {}", other.join(" "))),
    }
}
