//! Group: pull claims attempt 1 via H3; requires.gpus skipped by 0-gpu and
//! claimed by 1-gpu; scan past mismatch; cpus/kvm/providers; empty = None.

mod common;
mod reference;

use common::{
    assert_lease, assert_leased, assert_queued, caps_cpu, caps_gpu, caps_kvm, caps_providers,
    fresh_node_dir, item, item_plain, item_requires_cpus, item_requires_gpus, item_requires_kvm,
    item_requires_providers, open_queue, short_ttl, slot_config, start3, task_id, trusted_config,
    T0,
};
use prometheus_leases::TaskState;
use prometheus_mesh::Mesh;
use reference::{caps_satisfy, first_fitting, RefMesh};
use serde_json::json;

#[test]
fn pull_empty_queue_is_none() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    let got = mesh.pull(&mut queue, T0).expect("pull");
    assert!(got.is_none(), "empty queue → Ok(None)");
}

#[test]
fn pull_claims_attempt_one_worker_is_node_id() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, trusted_config("n7", "key-n7")).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("job-a"), T0).expect("enqueue");
    let lease = mesh.pull(&mut queue, T0).expect("pull").expect("claimed");
    assert_lease(&lease, "job-a", "n7", 1, T0 + short_ttl());
    let task = queue.get(&task_id("job-a")).expect("get");
    assert_leased(task, "job-a", "n7", 1);

    let mut refer = RefMesh::new(trusted_config("n7", "key-n7"));
    refer.advertise(caps_cpu(), T0).expect("ref adv");
    let (_q2, mut q2) = open_queue();
    q2.enqueue(item_plain("job-a"), T0).expect("enq");
    let rlease = refer
        .pull(&mut q2, T0)
        .expect("ref pull")
        .expect("ref claimed");
    assert_eq!(lease, rlease);
}

#[test]
fn requires_gpus_one_skipped_by_zero_gpu_claimed_by_one_gpu() {
    let (_parent, dir0) = fresh_node_dir();
    let mut cpu = Mesh::create(&dir0, slot_config(0)).expect("cpu node");
    cpu.advertise(caps_cpu(), T0).expect("cpu adv");

    let (_p1, dir1) = fresh_node_dir();
    let mut gpu = Mesh::create(&dir1, slot_config(1)).expect("gpu node");
    gpu.advertise(caps_gpu(1), T0).expect("gpu adv");

    let (_qparent, mut queue) = open_queue();
    queue
        .enqueue(item_requires_gpus("need-gpu", 1), T0)
        .expect("enqueue");

    let skip = cpu.pull(&mut queue, T0).expect("cpu pull");
    assert!(
        skip.is_none(),
        "0-gpu node must skip requires.gpus=1 with Ok(None), got {skip:?}"
    );
    let still = queue.get(&task_id("need-gpu")).expect("still there");
    assert_queued(still, "need-gpu");

    let lease = gpu
        .pull(&mut queue, T0)
        .expect("gpu pull")
        .expect("gpu claimed");
    assert_lease(&lease, "need-gpu", "1", 1, T0 + short_ttl());
    let taken = queue.get(&task_id("need-gpu")).expect("leased");
    assert_leased(taken, "need-gpu", "1", 1);
    assert_eq!(taken.state, TaskState::Leased);
}

#[test]
fn pull_scans_past_mismatch_and_claims_later_fit() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue
        .enqueue(item_requires_gpus("gpu-job", 1), T0)
        .expect("enq gpu");
    queue.enqueue(item_plain("cpu-job"), T0).expect("enq cpu");

    let expect = first_fitting(&queue, &caps_cpu()).expect("oracle fit");
    assert_eq!(expect.0, "cpu-job");

    let lease = mesh
        .pull(&mut queue, T0)
        .expect("pull")
        .expect("should claim later fitting task");
    assert_eq!(lease.task_id.0, "cpu-job");
    assert_eq!(lease.worker_id.0, "0");
    assert_eq!(lease.attempt, 1);
    assert_queued(queue.get(&task_id("gpu-job")).as_ref().unwrap(), "gpu-job");
    assert_leased(
        queue.get(&task_id("cpu-job")).as_ref().unwrap(),
        "cpu-job",
        "0",
        1,
    );
}

#[test]
fn pull_scans_two_mismatches_then_fit() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_requires_gpus("g1", 1), T0).expect("g1");
    queue.enqueue(item_requires_gpus("g2", 2), T0).expect("g2");
    queue.enqueue(item_plain("ok"), T0).expect("ok");
    let lease = mesh.pull(&mut queue, T0).expect("pull").expect("claimed");
    assert_eq!(lease.task_id.0, "ok");
    assert_queued(queue.get(&task_id("g1")).as_ref().unwrap(), "g1");
    assert_queued(queue.get(&task_id("g2")).as_ref().unwrap(), "g2");
}

#[test]
fn all_mismatched_queued_tasks_yield_none_none_claimed() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_requires_gpus("g1", 1), T0).expect("g1");
    queue.enqueue(item_requires_gpus("g2", 1), T0).expect("g2");
    let got = mesh.pull(&mut queue, T0).expect("pull");
    assert!(got.is_none());
    assert_queued(queue.get(&task_id("g1")).as_ref().unwrap(), "g1");
    assert_queued(queue.get(&task_id("g2")).as_ref().unwrap(), "g2");
}

#[test]
fn both_fit_claims_fifo_head() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("first"), T0).expect("first");
    queue.enqueue(item_plain("second"), T0).expect("second");
    let lease = mesh.pull(&mut queue, T0).expect("pull").expect("claimed");
    assert_eq!(lease.task_id.0, "first");
    assert_queued(queue.get(&task_id("second")).as_ref().unwrap(), "second");
}

#[test]
fn missing_requires_claimed_by_zero_gpu_node() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_plain("anyone"), T0).expect("enq");
    let lease = mesh.pull(&mut queue, T0).expect("pull").expect("claimed");
    assert_eq!(lease.task_id.0, "anyone");
}

#[test]
fn no_advert_default_caps_skip_gpu_require() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_requires_gpus("g", 1), T0).expect("enq");
    let got = mesh.pull(&mut queue, T0).expect("pull");
    assert!(
        got.is_none(),
        "no advert / default caps cannot satisfy gpus=1"
    );
    assert_queued(queue.get(&task_id("g")).as_ref().unwrap(), "g");
}

#[test]
fn requires_cpus_skipped_when_short() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv"); // cpus: 4
    let (_qparent, mut queue) = open_queue();
    queue
        .enqueue(item_requires_cpus("need-8", 8), T0)
        .expect("enq");
    assert!(mesh.pull(&mut queue, T0).expect("pull").is_none());
    assert_queued(queue.get(&task_id("need-8")).as_ref().unwrap(), "need-8");
}

#[test]
fn requires_kvm_true_skipped_without_kvm() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_cpu(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue
        .enqueue(item_requires_kvm("need-kvm", true), T0)
        .expect("enq");
    assert!(mesh.pull(&mut queue, T0).expect("pull").is_none());
}

#[test]
fn requires_kvm_true_claimed_by_kvm_node() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_kvm(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue
        .enqueue(item_requires_kvm("need-kvm", true), T0)
        .expect("enq");
    let lease = mesh.pull(&mut queue, T0).expect("pull").expect("claimed");
    assert_eq!(lease.task_id.0, "need-kvm");
}

#[test]
fn requires_providers_must_include_all() {
    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps_providers(&["cuda"]), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue
        .enqueue(item_requires_providers("need-both", &["cuda", "rocm"]), T0)
        .expect("enq");
    assert!(mesh.pull(&mut queue, T0).expect("pull").is_none());

    let (_p2, dir2) = fresh_node_dir();
    let mut rich = Mesh::create(&dir2, slot_config(1)).expect("create");
    rich.advertise(caps_providers(&["cuda", "rocm", "xai"]), T0)
        .expect("adv");
    let lease = rich.pull(&mut queue, T0).expect("pull").expect("claimed");
    assert_eq!(lease.task_id.0, "need-both");
    assert_eq!(lease.worker_id.0, "1");
}

#[test]
fn local_mesh_node_pull_uses_slot_id_as_worker() {
    let (_parent, dir) = common::fresh_base();
    let mut mesh = start3(&dir);
    mesh.get(1)
        .expect("1")
        .advertise(caps_gpu(1), T0)
        .expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue.enqueue(item_requires_gpus("g", 1), T0).expect("enq");
    let lease = mesh
        .get(1)
        .expect("1")
        .pull(&mut queue, T0)
        .expect("pull")
        .expect("claimed");
    assert_eq!(lease.worker_id.0, "1");
    assert_eq!(lease.attempt, 1);
}

#[test]
fn caps_satisfy_oracle_matches_pull_skip_and_claim() {
    let caps = caps_gpu(1);
    assert!(caps_satisfy(&caps, &json!({"job": 1})));
    assert!(caps_satisfy(&caps, &json!({"requires": {"gpus": 1}})));
    assert!(!caps_satisfy(&caps, &json!({"requires": {"gpus": 2}})));
    assert!(!caps_satisfy(
        &caps_cpu(),
        &json!({"requires": {"gpus": 1}})
    ));
    assert!(caps_satisfy(
        &caps,
        &json!({"requires": {"providers": ["cuda"]}})
    ));
    assert!(!caps_satisfy(
        &caps,
        &json!({"requires": {"providers": ["rocm"]}})
    ));

    let (_parent, dir) = fresh_node_dir();
    let mut mesh = Mesh::create(&dir, slot_config(0)).expect("create");
    mesh.advertise(caps.clone(), T0).expect("adv");
    let (_qparent, mut queue) = open_queue();
    queue
        .enqueue(item("too-big", json!({"requires": {"gpus": 2}})), T0)
        .expect("enq");
    assert!(first_fitting(&queue, &caps).is_none());
    assert!(mesh.pull(&mut queue, T0).expect("pull").is_none());
}
