//! Group: property tests vs the independent reference; src must not import tests/.

mod common;
mod reference;

use common::{
    assert_result_tag, assert_src_does_not_import_tests, cas_pins, create_ops, fresh_world,
    has_shadow_role, kernel_leader, leader_idx, live_nodes, membership, ops_cfg, release, sig,
    status_of, v1_digest, v1_signed, v2_signed, V1_BYTES, V2_BYTES,
};
use prometheus_ops::{has_quorum, NodeRole, Ops, Release, ReleaseState, DEFAULT_QUORUM};
use reference::{ref_has_quorum, RefOps};

#[test]
fn src_does_not_import_tests_and_has_quorum_agrees() {
    assert_src_does_not_import_tests();
    let rel = v1_signed();
    assert_eq!(
        has_quorum(&rel, DEFAULT_QUORUM),
        ref_has_quorum(&rel, DEFAULT_QUORUM)
    );
    assert!(has_quorum(&rel, 2));
}

#[test]
fn has_quorum_matches_reference_on_random_signature_lists() {
    let mut seed: u64 = 0xC0FFEE;
    let next = |s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *s
    };
    for trial in 0..40 {
        let n = (next(&mut seed) % 6) as usize;
        let mut signatures = Vec::new();
        for i in 0..n {
            let human = format!("h{}", next(&mut seed) % 4);
            let empty = next(&mut seed) % 5 == 0;
            let bytes = if empty {
                vec![]
            } else {
                vec![(next(&mut seed) % 256) as u8, i as u8]
            };
            signatures.push(sig(&human, &bytes));
        }
        let rel = release("v", v1_digest(), signatures);
        let q = (next(&mut seed) % 4) as usize;
        assert_eq!(
            has_quorum(&rel, q),
            ref_has_quorum(&rel, q),
            "trial {trial} q={q} rel={rel:?}"
        );
    }
}

#[test]
fn random_ops_match_reference_errors_state_pins_kernel() {
    let mut seed: u64 = 0xA11CE;
    let next = |s: &mut u64| {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *s
    };
    for trial in 0..8 {
        let mut world = fresh_world();
        let members = membership(&mut world);
        let kernel = kernel_leader(&mut world);
        let mut refer = RefOps::new(ops_cfg(), members.clone(), kernel);
        let mut ops = create_ops(&world);
        for step in 0..16 {
            let op = next(&mut seed) % 8;
            match op {
                0 => {
                    let use_v2 = next(&mut seed) % 2 == 0;
                    let (rel, bytes): (Release, &[u8]) = if use_v2 {
                        (v2_signed(), V2_BYTES)
                    } else {
                        (v1_signed(), V1_BYTES)
                    };
                    let got = ops.propose(rel.clone(), bytes, &mut world.store, world.now);
                    let exp = refer.propose(rel, bytes);
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} propose"));
                }
                1 => {
                    let got = ops.start_shadow(&mut world.store, world.now);
                    let exp = refer.start_shadow();
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} start_shadow"));
                }
                2 => {
                    let got = ops.shadow_pass(world.now);
                    let exp = refer.shadow_pass();
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} pass"));
                }
                3 => {
                    let got = ops.shadow_fail("prop-fail", &mut world.store, world.now);
                    let exp = refer.shadow_fail("prop-fail");
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} fail"));
                }
                4 => {
                    let idx = (next(&mut seed) as usize) % members.len();
                    let got = ops.apply_one(&members[idx], world.now);
                    let exp = refer.apply_one(&members[idx]);
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} apply"));
                }
                5 => {
                    let i = leader_idx(&mut world);
                    let got = ops.promote(world.group.get(i).unwrap(), &mut world.store, world.now);
                    let exp = refer.promote();
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} promote"));
                }
                6 => {
                    let i = leader_idx(&mut world);
                    let got = ops.rollback(
                        "prop-rb",
                        world.group.get(i).unwrap(),
                        &mut world.store,
                        world.now,
                    );
                    let exp = refer.rollback("prop-rb");
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} rollback"));
                }
                _ => {
                    let idx = (next(&mut seed) as usize) % members.len();
                    let i = leader_idx(&mut world);
                    let got = ops.mark_unhealthy(
                        &members[idx],
                        "prop-unh",
                        world.group.get(i).unwrap(),
                        &mut world.store,
                        world.now,
                    );
                    let exp = refer.mark_unhealthy(&members[idx], "prop-unh");
                    assert_result_tag(&got, &exp, &format!("t{trial}s{step} unhealthy"));
                }
            }
            let st = status_of(&ops, &mut world);
            assert_eq!(st.state, refer.state, "t{trial}s{step} state");
            assert_eq!(st.release, refer.release, "t{trial}s{step} release");
            assert_eq!(
                kernel_leader(&mut world),
                refer.kernel_version,
                "t{trial}s{step} kernel"
            );
            assert_eq!(
                st.kernel_version, refer.kernel_version,
                "t{trial}s{step} status kernel"
            );
            assert_eq!(
                has_shadow_role(&st),
                refer.nodes.iter().any(|n| n.role == NodeRole::Shadow),
                "t{trial}s{step} shadow"
            );
            let mut got_live: Vec<(String, String, bool)> = live_nodes(&st)
                .into_iter()
                .map(|n| (n.id.0.clone(), n.applied.clone(), n.healthy))
                .collect();
            let mut exp_live: Vec<(String, String, bool)> = refer
                .nodes
                .iter()
                .filter(|n| n.role == NodeRole::Live)
                .map(|n| (n.id.0.clone(), n.applied.clone(), n.healthy))
                .collect();
            got_live.sort();
            exp_live.sort();
            assert_eq!(got_live, exp_live, "t{trial}s{step} live nodes");
            assert_eq!(cas_pins(&world.store), refer.pins, "t{trial}s{step} pins");
        }
        drop(ops);
        let ops2 = Ops::open(&world.ops_dir, ops_cfg()).expect("replay");
        let st = status_of(&ops2, &mut world);
        assert_eq!(st.state, refer.state, "t{trial} replay state");
        assert_eq!(st.release, refer.release, "t{trial} replay release");
    }
}

#[test]
fn refuse_then_valid_propose_then_shadow_never_touches_kernel_until_promote() {
    let mut world = fresh_world();
    let members = membership(&mut world);
    let kernel = kernel_leader(&mut world);
    let mut refer = RefOps::new(ops_cfg(), members.clone(), kernel.clone());
    let mut ops = create_ops(&world);
    let bad = release("v1", v1_digest(), vec![sig("only", b"x")]);
    assert_result_tag(
        &ops.propose(bad.clone(), V1_BYTES, &mut world.store, world.now),
        &refer.propose(bad, V1_BYTES),
        "refuse",
    );
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_result_tag(
        &ops.propose(v1_signed(), V1_BYTES, &mut world.store, world.now),
        &refer.propose(v1_signed(), V1_BYTES),
        "ok",
    );
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_result_tag(
        &ops.start_shadow(&mut world.store, world.now),
        &refer.start_shadow(),
        "sh",
    );
    assert_eq!(kernel_leader(&mut world), kernel);
    assert_result_tag(&ops.shadow_pass(world.now), &refer.shadow_pass(), "pass");
    for m in &members {
        ops.apply_one(m, world.now).unwrap();
        refer.apply_one(m).unwrap();
        assert_eq!(kernel_leader(&mut world), kernel);
    }
    let i = leader_idx(&mut world);
    ops.promote(world.group.get(i).unwrap(), &mut world.store, world.now)
        .unwrap();
    refer.promote().unwrap();
    assert_eq!(kernel_leader(&mut world), "v1");
    assert_eq!(status_of(&ops, &mut world).state, ReleaseState::Current);
}
