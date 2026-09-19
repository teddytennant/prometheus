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
//! Cross-node all_reduce on this cluster is `srun --mpi=pmix` of the
//! `all_reduce_perf_mpi` ELF (not an `mpirun` wrapper). The job must also
//! emit the PMIx env that actually works on NCShare (734353 / 734382):
//! non-comment `PMIX_MCA_gds=hash`, `unset OMPI_MCA_mca_base_component_path`,
//! and `LD_LIBRARY_PATH` including a system OpenMPI lib dir (`openmpi/lib`).
//! Binary override is `NCCL_TESTS_ALL_REDUCE_MPI` — do not require a
//! hardcoded site path as the only way to find the ELF. Job 734144
//! (`fv-mpifix`) SIGSEGV'd in `PMIx_Init` after copying a tools-prefix MCA
//! path and launching via `mpirun`.
//!
//! Spec 16.2 also requires intra-node all_reduce (4 GPUs on one node,
//! non-MPI `all_reduce_perf` with `-g 4` or `-g=4`, override
//! `NCCL_TESTS_ALL_REDUCE`, stdout not `busbw_gbps` /
//! `nccl_allreduce_2node.txt`) and `/dev/kvm` plus NVMe recorded on every
//! allocated compute node via a non-comment `srun` with `-N 2` or
//! `--nodes=2` and `--ntasks-per-node=1` (not only local hostname on the
//! batch host). Do not `srun -N 2` the non-MPI `all_reduce_perf`.
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

fn strip_shell_quotes(s: &str) -> String {
    s.replace(['"', '\''], "")
}

fn assignment_tokens(line: &str) -> Vec<String> {
    strip_shell_quotes(line)
        .split_whitespace()
        .map(|tok| tok.trim_matches(|c| c == ';' || c == '&').to_string())
        .collect()
}

/// Non-comment `export PMIX_MCA_gds=hash` or `PMIX_MCA_gds=hash` (quotes ok).
fn has_pmix_mca_gds_hash(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        assignment_tokens(line)
            .iter()
            .any(|tok| tok == "PMIX_MCA_gds=hash")
    })
}

/// Non-comment `unset OMPI_MCA_mca_base_component_path` (`unset -v` ok).
fn has_unset_ompi_mca_component_path(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        let tokens = assignment_tokens(line);
        tokens.iter().any(|t| t == "unset")
            && tokens
                .iter()
                .any(|t| t == "OMPI_MCA_mca_base_component_path")
    })
}

/// Non-comment `LD_LIBRARY_PATH` assignment/export whose value mentions
/// a system OpenMPI lib dir. `openmpi/lib` is enough; do not require a
/// hardcoded `/work/ttennant1` (or any other) site prefix.
fn has_ld_library_path_openmpi_lib(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        line.contains("LD_LIBRARY_PATH") && line.contains("openmpi/lib")
    })
}

fn token_is_mpirun(tok: &str) -> bool {
    let tok = tok.trim_matches(|c: char| c == '"' || c == '\'' || c == ';' || c == '&');
    tok == "mpirun" || tok.ends_with("/mpirun")
}

/// Non-comment launch via `mpirun` (the 734144 wrapper). Comments that
/// mention mpirun do not count.
fn launches_via_mpirun(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        line.split_whitespace().any(token_is_mpirun)
    })
}

/// Non-comment `NCCL_TESTS_ALL_REDUCE_MPI` override. Do not require a
/// hardcoded site path as the only way to find `all_reduce_perf_mpi`.
fn has_nccl_tests_all_reduce_mpi_override(script: &str) -> bool {
    non_comment_contains(script, "NCCL_TESTS_ALL_REDUCE_MPI")
}

fn assert_pmix_launch_env(script: &str) {
    let mut missing = Vec::new();
    if !has_pmix_mca_gds_hash(script) {
        missing.push("non-comment PMIX_MCA_gds=hash (export or assignment)");
    }
    if !has_unset_ompi_mca_component_path(script) {
        missing.push("non-comment unset OMPI_MCA_mca_base_component_path");
    }
    if !has_ld_library_path_openmpi_lib(script) {
        missing.push("non-comment LD_LIBRARY_PATH including system OpenMPI lib dir (openmpi/lib)");
    }
    if launches_via_mpirun(script) {
        missing.push("must not launch via mpirun (job 734144 wrapper)");
    }
    if !has_nccl_tests_all_reduce_mpi_override(script) {
        missing.push("NCCL_TESTS_ALL_REDUCE_MPI binary override");
    }
    assert!(
        missing.is_empty(),
        "pmix launch env missing {missing:?}. \
         Job 734144 SIGSEGV'd in PMIx_Init (gds_shmem) after copying a \
         tools-prefix MCA path and launching via mpirun. Jobs 734353/734382 \
         passed with this env. srun --mpi=pmix of all_reduce_perf_mpi stays \
         required; do not require a hardcoded /work/ttennant1 path. \
         Comments do not count: {script}"
    );
    assert!(
        has_srun(script),
        "must still invoke srun (not mpirun): {script}"
    );
    assert!(
        has_mpi_pmix(script),
        "must still pass --mpi=pmix to srun: {script}"
    );
    assert!(
        has_all_reduce_perf_mpi(script),
        "must still launch all_reduce_perf_mpi: {script}"
    );
}

/// `NCCL_TESTS_ALL_REDUCE` as its own override, not only
/// `NCCL_TESTS_ALL_REDUCE_MPI` (which contains that prefix).
fn has_nccl_tests_all_reduce_override(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) {
            return false;
        }
        let mut rest = line;
        while let Some(idx) = rest.find("NCCL_TESTS_ALL_REDUCE") {
            let after = &rest[idx + "NCCL_TESTS_ALL_REDUCE".len()..];
            if !after.starts_with("_MPI") {
                return true;
            }
            rest = after;
        }
        false
    })
}

fn has_non_mpi_all_reduce_perf(script: &str) -> bool {
    script
        .lines()
        .any(|line| !is_comment(line) && non_mpi_all_reduce_token(line))
}

/// Digit-bounded flag match so `-g 4` does not hit `-g 40`.
fn contains_flag_count(line: &str, needles: &[String]) -> bool {
    needles.iter().any(|needle| {
        let mut rest = line;
        while let Some(idx) = rest.find(needle.as_str()) {
            let after = &rest[idx + needle.len()..];
            if after
                .chars()
                .next()
                .map(|c| !c.is_ascii_digit())
                .unwrap_or(true)
            {
                return true;
            }
            rest = &rest[idx + 1..];
        }
        false
    })
}

fn line_has_g_count(line: &str, n: u32) -> bool {
    let nstr = n.to_string();
    contains_flag_count(
        line,
        &[
            format!("-g {nstr}"),
            format!("-g{nstr}"),
            format!("-g={nstr}"),
        ],
    )
}

fn line_has_mpi_pmix(line: &str) -> bool {
    line.split_whitespace().any(|tok| {
        let tok = tok.trim_matches(|c| c == '"' || c == '\'');
        tok == "--mpi=pmix"
    })
}

/// Non-MPI `-g 4` / `-g=4` / `-g4` on one node (not `srun -N 2`, not pmix).
fn has_intra_node_all_reduce_g4(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) || sbatch_rest(line).is_some() {
            return false;
        }
        line_has_g_count(line, 4) && !line_has_mpi_pmix(line) && !line_has_two_nodes(line)
    })
}

fn has_intra_node_log(script: &str) -> bool {
    non_comment_contains(script, "nccl_allreduce_intra.txt")
        || non_comment_contains(script, "nccl_allreduce.txt")
}

fn intra_g4_line_writes_forbidden_log(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) || sbatch_rest(line).is_some() {
            return false;
        }
        line_has_g_count(line, 4)
            && (line.contains("busbw_gbps") || line.contains("nccl_allreduce_2node.txt"))
    })
}

fn has_g_count(script: &str, n: u32) -> bool {
    script
        .lines()
        .any(|line| !is_comment(line) && sbatch_rest(line).is_none() && line_has_g_count(line, n))
}

fn assert_intra_node_all_reduce(script: &str) {
    let mut missing = Vec::new();
    if !has_nccl_tests_all_reduce_override(script) {
        missing.push("NCCL_TESTS_ALL_REDUCE binary override (not only NCCL_TESTS_ALL_REDUCE_MPI)");
    }
    if !has_non_mpi_all_reduce_perf(script) {
        missing.push("non-MPI all_reduce_perf (not all_reduce_perf_mpi)");
    }
    if !has_intra_node_all_reduce_g4(script) {
        missing.push("intra-node all_reduce_perf with -g 4 or -g=4 on one node");
    }
    if !has_intra_node_log(script) {
        missing.push(
            "intra-node stdout file (nccl_allreduce.txt or nccl_allreduce_intra.txt, \
             not busbw_gbps / nccl_allreduce_2node.txt)",
        );
    }
    if intra_g4_line_writes_forbidden_log(script) {
        missing.push("intra-node -g 4 stdout must not be busbw_gbps or nccl_allreduce_2node.txt");
    }
    if launches_non_mpi_all_reduce_across_nodes(script) {
        missing.push("must not srun -N 2 the non-MPI all_reduce_perf");
    }
    assert!(
        missing.is_empty(),
        "intra-node all_reduce missing {missing:?}. \
         Spec 16.2: nccl-tests all_reduce intra-node (4 GPUs on one node, \
         non-MPI all_reduce_perf -g 4) and across 2 nodes. Override via \
         NCCL_TESTS_ALL_REDUCE. Intra stdout is a separate file. \
         Comments do not count: {script}"
    );
    assert!(
        has_all_reduce_perf_mpi(script),
        "must still launch all_reduce_perf_mpi: {script}"
    );
    assert!(
        has_srun(script),
        "must still invoke srun for the 2-node MPI all_reduce: {script}"
    );
    assert!(
        has_mpi_pmix(script),
        "must still pass --mpi=pmix to srun: {script}"
    );
    assert!(
        non_comment_contains(script, "busbw_gbps"),
        "busbw_gbps must still come from the 2-node log: {script}"
    );
    assert!(
        has_g_count(script, 1),
        "2-node MPI all_reduce must still use -g 1: {script}"
    );
}

fn line_has_ntasks_per_node(line: &str, n: u32) -> bool {
    let nstr = n.to_string();
    contains_flag_count(
        line,
        &[
            format!("--ntasks-per-node={nstr}"),
            format!("--ntasks-per-node {nstr}"),
        ],
    )
}

fn line_has_srun(line: &str) -> bool {
    line.split_whitespace().any(|tok| {
        let tok = tok.trim_matches(|c: char| c == '"' || c == '\'' || c == ';' || c == '&');
        tok == "srun" || tok.ends_with("/srun")
    })
}

/// Non-comment `srun` with `-N 2`/`--nodes=2` and `--ntasks-per-node=1`,
/// not the all_reduce launch.
fn records_facts_via_srun_on_every_node(script: &str) -> bool {
    script.lines().any(|line| {
        if is_comment(line) || sbatch_rest(line).is_some() {
            return false;
        }
        line_has_srun(line)
            && line_has_two_nodes(line)
            && line_has_ntasks_per_node(line, 1)
            && !non_mpi_all_reduce_token(line)
            && !line.contains("all_reduce_perf_mpi")
    })
}

fn assert_node_facts_on_every_node(script: &str) {
    assert!(
        records_facts_via_srun_on_every_node(script),
        "node facts must be recorded on every allocated compute node via a \
         non-comment srun with -N 2 or --nodes=2 and --ntasks-per-node=1 \
         (not only local hostname on the batch host). Spec 16.2: check \
         /dev/kvm and NVMe on compute nodes. Comments do not count: {script}"
    );
    assert!(
        non_comment_contains(script, "/dev/kvm"),
        "missing /dev/kvm in node facts: {script}"
    );
    assert!(
        non_comment_contains(script, "nvme"),
        "missing nvme in node facts: {script}"
    );
    assert!(
        non_comment_contains(script, "node_facts.txt"),
        "missing node_facts.txt: {script}"
    );
    assert!(
        !launches_non_mpi_all_reduce_across_nodes(script),
        "srun -N 2 of non-MPI all_reduce_perf is still forbidden: {script}"
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
fn ref_render_v0_2node_pmix_launch_env() {
    let out = reference::render_v0_2node("pmix", "01:00:00").unwrap();
    assert_pmix_launch_env(&out);
}

#[test]
fn ref_render_v0_2node_intra_node_all_reduce() {
    let out = reference::render_v0_2node("intra", "01:00:00").unwrap();
    assert_intra_node_all_reduce(&out);
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
fn ref_render_v0_2node_node_facts_on_every_node() {
    let out = reference::render_v0_2node("facts-all", "01:00:00").unwrap();
    assert_node_facts_on_every_node(&out);
}

#[test]
fn ref_render_v0_2node_builds_venv_inside_job() {
    let out = reference::render_v0_2node("venv", "01:00:00").unwrap();
    assert!(has_venv_inside_job(&out));
}

// ---------------------------------------------------------------------------
// prod_* — call render_v0_2node. Intra-node all_reduce and per-node facts
// tests must fail until production emits non-MPI all_reduce_perf -g 4
// (override NCCL_TESTS_ALL_REDUCE; stdout not busbw_gbps /
// nccl_allreduce_2node.txt) and records /dev/kvm + NVMe on every node via
// srun -N 2 --ntasks-per-node=1. Do not srun -N 2 the non-MPI binary.
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
fn prod_render_v0_2node_pmix_launch_env() {
    let out = render_v0_2node("pmix", "01:00:00").unwrap();
    assert_pmix_launch_env(&out);
}

#[test]
fn prod_render_v0_2node_intra_node_all_reduce() {
    let out = render_v0_2node("intra", "01:00:00").unwrap();
    assert_intra_node_all_reduce(&out);
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
fn prod_render_v0_2node_node_facts_on_every_node() {
    let out = render_v0_2node("facts-all", "01:00:00").unwrap();
    assert_node_facts_on_every_node(&out);
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
