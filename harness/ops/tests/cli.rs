//! Group: `cli` — `harness [--json] status`.

mod common;
mod reference;

use common::{arg, create_ops, fresh_world, leader_idx, membership, ops_cfg, v1_signed, V1_BYTES};
use prometheus_ops::{cli, Status};
use reference::{ref_cli, RefOps};

#[test]
fn cli_status_returns_status_text() {
    let mut world = fresh_world();
    let ops = create_ops(&world);
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();
    let text = ops.status_text(cluster);
    let out = cli(&[arg("status")], &ops, cluster).expect("cli status");
    assert_eq!(out, text);
}

#[test]
fn cli_json_flag_before_status() {
    let mut world = fresh_world();
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();
    let json = ops.status_json(cluster).expect("json");
    let out = cli(&[arg("--json"), arg("status")], &ops, cluster).expect("cli --json status");
    assert_eq!(out, json);
    let parsed: Status = serde_json::from_str(&out).expect("json of Status");
    assert_eq!(parsed, ops.status(cluster));
}

#[test]
fn cli_json_flag_after_status() {
    let mut world = fresh_world();
    let ops = create_ops(&world);
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();
    let json = ops.status_json(cluster).expect("json");
    let out = cli(&[arg("status"), arg("--json")], &ops, cluster).expect("cli status --json");
    assert_eq!(out, json);
}

#[test]
fn cli_unknown_command_errors() {
    let mut world = fresh_world();
    let ops = create_ops(&world);
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();
    cli(&[arg("not-a-command")], &ops, cluster).expect_err("unknown");
    cli(&[], &ops, cluster).expect_err("empty");
}

#[test]
fn cli_matches_reference_dispatch() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = common::kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members, kernel);
    let mut ops = create_ops(&world);
    ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now)
        .unwrap();
    refer.propose(v1_signed(), V1_BYTES).unwrap();
    let i = leader_idx(&mut world);
    let cluster = world.group.get(i).unwrap();

    let got = cli(&[arg("status")], &ops, cluster);
    let exp = ref_cli(&[arg("status")], &refer);
    assert!(got.is_ok() && exp.is_ok(), "status text both ok");
    assert_eq!(got.unwrap(), ops.status_text(cluster));

    let got = cli(&[arg("--json"), arg("status")], &ops, cluster).unwrap();
    let exp = ref_cli(&[arg("--json"), arg("status")], &refer).unwrap();
    let got_st: Status = serde_json::from_str(&got).unwrap();
    let exp_st: Status = serde_json::from_str(&exp).unwrap();
    assert_eq!(got_st.state, exp_st.state);
    assert_eq!(got_st.release, exp_st.release);
    assert_eq!(got_st.kernel_version, exp_st.kernel_version);
}
