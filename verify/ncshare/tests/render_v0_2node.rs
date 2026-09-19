//! Oracle tests for `render_v0_2node` (spec 16.2 second allocation).
//!
//! Tests always go through `render_v0_2node` / `render_job` (and the
//! `tests/reference/` renderer), never inspect template files alone.
//! `ref_*` pass today: the reference substitutes placeholders in an independent
//! 2-node stub. Do not `#[should_panic]`.
//!
//! NCShare sbatch rejects combining typed `--gres=gpu:h200:N` with
//! `--gpus-per-task` ("Invalid GRES specification (with and without type
//! identification)"). The rendered 2-node script must not emit
//! `#SBATCH --gpus-per-task` or `srun --gpus-per-task`. One rank per GPU is
//! `--ntasks=8` and `--ntasks-per-node=4` under `--gres=gpu:h200:4`.
//!
//! `render_job(Stage::V0, ...)` is the 1-GPU first allocation and must stay
//! 1-node. That regression is allowed to pass on the current tree.

mod reference;

use prometheus_verify_ncshare::{
    render_job, render_v0_2node, Error, Stage, JOB_PREFIX, V0_2NODE_GPUS_PER_NODE, V0_2NODE_NODES,
};

fn expected_job_name_line(run_id: &str) -> String {
    format!("#SBATCH -J {JOB_PREFIX}-{run_id}-{}", Stage::V0.as_str())
}

fn ranks() -> u32 {
    V0_2NODE_NODES * V0_2NODE_GPUS_PER_NODE
}

fn is_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('#') && !t.starts_with("#SBATCH")
}

fn sbatch_rest(line: &str) -> Option<&str> {
    line.trim().strip_prefix("#SBATCH").map(str::trim)
}

fn sbatch_has_kv(script: &str, key: &str, value: u32) -> bool {
    let value = value.to_string();
    let eq = format!("{key}={value}");
    let glued = format!("{key}{value}");
    script.lines().any(|line| {
        let Some(rest) = sbatch_rest(line) else {
            return false;
        };
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.iter().any(|t| *t == eq || *t == glued) {
            return true;
        }
        tokens.windows(2).any(|w| w[0] == key && w[1] == value)
    })
}

fn sbatch_requests_nodes(script: &str, n: u32) -> bool {
    sbatch_has_kv(script, "--nodes", n) || sbatch_has_kv(script, "-N", n)
}

fn sbatch_has_h200_count(script: &str, n: u32) -> bool {
    let needle = format!("gpu:h200:{n}");
    script.lines().any(|line| {
        if sbatch_rest(line).is_none() {
            return false;
        }
        let Some(idx) = line.find(&needle) else {
            return false;
        };
        let after = &line[idx + needle.len()..];
        after.is_empty() || !after.starts_with(|c: char| c.is_ascii_digit())
    })
}

fn non_mpi_all_reduce_token(line: &str) -> bool {
    let mut rest = line;
    while let Some(idx) = rest.find("all_reduce_perf") {
        let after = &rest[idx + "all_reduce_perf".len()..];
        if !after.starts_with("_mpi") {
            return true;
        }
        rest = after;
    }
    false
}

fn line_has_two_nodes(line: &str) -> bool {
    let n = V0_2NODE_NODES;
    line.contains(&format!("-N {n}"))
        || line.contains(&format!("-N{n}"))
        || line.contains(&format!("-N={n}"))
        || line.contains(&format!("--nodes={n}"))
        || line.contains(&format!("--nodes {n}"))
}

fn launches_non_mpi_all_reduce_across_nodes(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        line.contains("srun") && line_has_two_nodes(line) && non_mpi_all_reduce_token(line)
    })
}

fn has_mpi_pmix(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        line.contains("--mpi=pmix") || line.contains("--mpi pmix")
    })
}

fn has_all_reduce_perf_mpi(script: &str) -> bool {
    script
        .lines()
        .any(|line| !is_comment(line) && line.contains("all_reduce_perf_mpi"))
}

fn has_srun(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        line.split_whitespace().any(|tok| tok == "srun")
    })
}

fn non_comment_contains(script: &str, needle: &str) -> bool {
    script
        .lines()
        .any(|line| !is_comment(line) && line.contains(needle))
}

fn has_venv_inside_job(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        let t = line.trim();
        t.contains("python3 -m venv") || t.contains("python -m venv")
    })
}

fn token_is_gpus_per_task(tok: &str) -> bool {
    tok == "--gpus-per-task" || tok.starts_with("--gpus-per-task=")
}

fn line_has_gpus_per_task(line: &str) -> bool {
    line.split_whitespace().any(token_is_gpus_per_task)
}

/// `#SBATCH --gpus-per-task` (any value / spacing). `#SBATCH` is not a comment.
fn has_sbatch_gpus_per_task(script: &str) -> bool {
    script
        .lines()
        .any(|line| sbatch_rest(line).is_some() && line_has_gpus_per_task(line))
}

/// `srun --gpus-per-task` on a non-comment, non-#SBATCH line (including
/// continuation lines that only carry the flag).
fn has_srun_gpus_per_task(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) || sbatch_rest(line).is_some() {
            return false;
        }
        line_has_gpus_per_task(line)
    })
}

fn assert_forbids_gpus_per_task(script: &str) {
    let sbatch = has_sbatch_gpus_per_task(script);
    let srun = has_srun_gpus_per_task(script);
    assert!(
        !sbatch && !srun,
        "NCShare rejects combining typed --gres=gpu:h200:N with --gpus-per-task \
         (Invalid GRES specification (with and without type identification)). \
         Do not emit #SBATCH --gpus-per-task or srun --gpus-per-task. \
         sbatch_gpus_per_task={sbatch} srun_gpus_per_task={srun}: {script}"
    );
}

fn assert_rejects_empty(result: prometheus_verify_ncshare::Result<String>) {
    match result {
        Err(Error::Other(_)) => {}
        other => panic!("expected Error::Other, got {other:?}"),
    }
}

fn assert_v0_2node_script(out: &str, run_id: &str, walltime: &str) {
    assert!(
        !out.contains("{{RUN_ID}}"),
        "unsubstituted {{RUN_ID}}: {out}"
    );
    assert!(
        !out.contains("{{WALLTIME}}"),
        "unsubstituted {{WALLTIME}}: {out}"
    );
    assert!(
        out.contains(run_id),
        "rendered script missing run_id {run_id:?}: {out}"
    );
    assert!(
        out.contains(walltime),
        "rendered script missing walltime {walltime:?}: {out}"
    );

    let job = expected_job_name_line(run_id);
    assert!(
        out.contains(&job),
        "job name must be {job} (JOB_PREFIX + stage v0): {out}"
    );

    assert!(
        sbatch_requests_nodes(out, V0_2NODE_NODES),
        "must request {} nodes (#SBATCH --nodes={} or equivalent): {out}",
        V0_2NODE_NODES,
        V0_2NODE_NODES
    );
    assert!(
        sbatch_has_h200_count(out, V0_2NODE_GPUS_PER_NODE),
        "must request gpu:h200:{} (V0_2NODE_GPUS_PER_NODE): {out}",
        V0_2NODE_GPUS_PER_NODE
    );

    let ntasks = ranks();
    assert!(
        sbatch_has_kv(out, "--ntasks", ntasks),
        "must launch {ntasks} ranks (#SBATCH --ntasks={ntasks} = {V0_2NODE_NODES}×{V0_2NODE_GPUS_PER_NODE}): {out}"
    );
    assert!(
        sbatch_has_kv(out, "--ntasks-per-node", V0_2NODE_GPUS_PER_NODE),
        "must set #SBATCH --ntasks-per-node={} (one rank per GPU): {out}",
        V0_2NODE_GPUS_PER_NODE
    );

    assert!(
        has_srun(out),
        "cross-node all_reduce must be launched with srun: {out}"
    );
    assert!(
        has_mpi_pmix(out),
        "multi-rank nccl-tests on this cluster only works as srun --mpi=pmix: {out}"
    );
    assert!(
        has_all_reduce_perf_mpi(out),
        "cross-node all_reduce must use the MPI binary all_reduce_perf_mpi: {out}"
    );
    assert!(
        !launches_non_mpi_all_reduce_across_nodes(out),
        "must not srun -N {V0_2NODE_NODES} the non-MPI all_reduce_perf: {out}"
    );

    assert!(
        non_comment_contains(out, "busbw_gbps"),
        "must write busbw_gbps (check_exit(V0) reads it) from the 2-node all_reduce: {out}"
    );
    assert!(
        non_comment_contains(out, "node_facts.txt"),
        "must write node_facts.txt (check_exit(V0) reads it): {out}"
    );
    let lower = out.to_ascii_lowercase();
    assert!(
        lower.contains("kvm") && non_comment_contains(out, "/dev/kvm"),
        "node_facts must record /dev/kvm: {out}"
    );
    assert!(lower.contains("nvme"), "node_facts must record NVMe: {out}");

    assert!(
        has_venv_inside_job(out),
        "venv must be built inside the job (python3 -m venv), not on the login node: {out}"
    );
}

// ---------------------------------------------------------------------------
// ref_* — independent 2-node stub; these pass today
// ---------------------------------------------------------------------------

#[test]
fn ref_render_v0_2node_rejects_empty_run_id_or_walltime() {
    assert_rejects_empty(reference::render_v0_2node("", "1:00:00"));
    assert_rejects_empty(reference::render_v0_2node("run1", ""));
    assert_rejects_empty(reference::render_v0_2node("", ""));
}

#[test]
fn ref_render_v0_2node_substitutes_run_id_walltime_and_job_name() {
    let out = reference::render_v0_2node("run42", "04:00:00").unwrap();
    assert_v0_2node_script(&out, "run42", "04:00:00");
}

#[test]
fn ref_render_v0_2node_requests_2_nodes_and_4_h200s_per_node() {
    let out = reference::render_v0_2node("alloc", "02:00:00").unwrap();
    assert!(sbatch_requests_nodes(&out, V0_2NODE_NODES));
    assert!(sbatch_has_h200_count(&out, V0_2NODE_GPUS_PER_NODE));
}

#[test]
fn ref_render_v0_2node_launches_8_ranks_one_per_gpu() {
    let out = reference::render_v0_2node("ranks", "01:00:00").unwrap();
    assert!(sbatch_has_kv(&out, "--ntasks", ranks()));
    assert!(sbatch_has_kv(
        &out,
        "--ntasks-per-node",
        V0_2NODE_GPUS_PER_NODE
    ));
}

#[test]
fn ref_render_v0_2node_forbids_gpus_per_task() {
    let out = reference::render_v0_2node("gres", "01:00:00").unwrap();
    assert_forbids_gpus_per_task(&out);
}

#[test]
fn ref_render_v0_2node_uses_mpi_all_reduce_perf_mpi() {
    let out = reference::render_v0_2node("nccl", "01:00:00").unwrap();
    assert!(has_srun(&out));
    assert!(has_mpi_pmix(&out));
    assert!(has_all_reduce_perf_mpi(&out));
    assert!(!launches_non_mpi_all_reduce_across_nodes(&out));
}

#[test]
fn ref_render_v0_2node_writes_busbw_and_node_facts() {
    let out = reference::render_v0_2node("facts", "01:00:00").unwrap();
    assert!(non_comment_contains(&out, "busbw_gbps"));
    assert!(non_comment_contains(&out, "node_facts.txt"));
    assert!(non_comment_contains(&out, "/dev/kvm"));
    assert!(out.to_ascii_lowercase().contains("nvme"));
}

#[test]
fn ref_render_v0_2node_builds_venv_inside_job() {
    let out = reference::render_v0_2node("venv", "01:00:00").unwrap();
    assert!(has_venv_inside_job(&out));
}

// ---------------------------------------------------------------------------
// prod_* — call render_v0_2node. The GRES forbid test must fail until
// production drops #SBATCH --gpus-per-task / srun --gpus-per-task.
// ---------------------------------------------------------------------------

#[test]
fn prod_render_v0_2node_rejects_empty_run_id_or_walltime() {
    assert_rejects_empty(render_v0_2node("", "1:00:00"));
    assert_rejects_empty(render_v0_2node("run1", ""));
    assert_rejects_empty(render_v0_2node("", ""));
}

#[test]
fn prod_render_v0_2node_substitutes_run_id_walltime_and_job_name() {
    let out = render_v0_2node("run42", "04:00:00").unwrap();
    assert_v0_2node_script(&out, "run42", "04:00:00");
}

#[test]
fn prod_render_v0_2node_requests_2_nodes_and_4_h200s_per_node() {
    let out = render_v0_2node("alloc", "02:00:00").unwrap();
    assert!(
        sbatch_requests_nodes(&out, V0_2NODE_NODES),
        "must request {} nodes: {out}",
        V0_2NODE_NODES
    );
    assert!(
        sbatch_has_h200_count(&out, V0_2NODE_GPUS_PER_NODE),
        "must request gpu:h200:{}: {out}",
        V0_2NODE_GPUS_PER_NODE
    );
}

#[test]
fn prod_render_v0_2node_launches_8_ranks_one_per_gpu() {
    let out = render_v0_2node("ranks", "01:00:00").unwrap();
    assert!(
        sbatch_has_kv(&out, "--ntasks", ranks()),
        "must launch {} ranks: {out}",
        ranks()
    );
    assert!(
        sbatch_has_kv(&out, "--ntasks-per-node", V0_2NODE_GPUS_PER_NODE),
        "must set --ntasks-per-node={}: {out}",
        V0_2NODE_GPUS_PER_NODE
    );
}

#[test]
fn prod_render_v0_2node_forbids_gpus_per_task() {
    let out = render_v0_2node("gres", "01:00:00").unwrap();
    assert_forbids_gpus_per_task(&out);
}

#[test]
fn prod_render_v0_2node_uses_mpi_all_reduce_perf_mpi() {
    let out = render_v0_2node("nccl", "01:00:00").unwrap();
    assert!(has_srun(&out), "must srun the MPI all_reduce: {out}");
    assert!(
        has_mpi_pmix(&out),
        "must use srun --mpi=pmix, not srun -N {} of non-MPI all_reduce_perf: {out}",
        V0_2NODE_NODES
    );
    assert!(
        has_all_reduce_perf_mpi(&out),
        "must use all_reduce_perf_mpi: {out}"
    );
    assert!(
        !launches_non_mpi_all_reduce_across_nodes(&out),
        "must not srun -N {} the non-MPI all_reduce_perf: {out}",
        V0_2NODE_NODES
    );
}

#[test]
fn prod_render_v0_2node_writes_busbw_and_node_facts() {
    let out = render_v0_2node("facts", "01:00:00").unwrap();
    assert!(
        non_comment_contains(&out, "busbw_gbps"),
        "must write busbw_gbps from the 2-node all_reduce: {out}"
    );
    assert!(
        non_comment_contains(&out, "node_facts.txt"),
        "must write node_facts.txt: {out}"
    );
    assert!(
        non_comment_contains(&out, "/dev/kvm"),
        "node_facts must record /dev/kvm: {out}"
    );
    assert!(
        out.to_ascii_lowercase().contains("nvme"),
        "node_facts must record NVMe: {out}"
    );
}

#[test]
fn prod_render_v0_2node_builds_venv_inside_job() {
    let out = render_v0_2node("venv", "01:00:00").unwrap();
    assert!(
        has_venv_inside_job(&out),
        "venv must be built inside the job, not on the login node: {out}"
    );
}

// ---------------------------------------------------------------------------
// Regression: 1-GPU first allocation must stay 1-node (passes today)
// ---------------------------------------------------------------------------

#[test]
fn prod_render_job_v0_still_1node() {
    let out = render_job(Stage::V0, "run42", "04:00:00", 1).unwrap();
    assert!(
        sbatch_requests_nodes(&out, 1),
        "render_job(Stage::V0) is the 1-GPU first allocation and must request 1 node: {out}"
    );
    assert!(
        !sbatch_requests_nodes(&out, V0_2NODE_NODES),
        "render_job(Stage::V0) must not become the 2-node allocation: {out}"
    );
    assert!(
        !sbatch_has_h200_count(&out, V0_2NODE_GPUS_PER_NODE),
        "1-GPU V0 must not request gpu:h200:{} per node: {out}",
        V0_2NODE_GPUS_PER_NODE
    );
    assert!(
        out.contains(&expected_job_name_line("run42")),
        "1-GPU V0 job name must stay fv-{{run_id}}-v0: {out}"
    );
}
