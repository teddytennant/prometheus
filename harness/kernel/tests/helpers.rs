//! Group: iface helpers that do **not** call `Kernel` methods.
//!
//! These may pass against the unimplemented stub. Every other test file
//! exercises `Kernel::*` and must fail until L1 is implemented.

use prometheus_kernel::{
    is_mirror_url, PatchTarget, Quota, Role, Rung, CANARY_FRACTION, CANARY_MS, KERNEL_INVARIANT,
    MAX_SPAWN_DEPTH, MIRROR_HOSTS,
};

#[test]
fn max_spawn_depth_is_three() {
    assert_eq!(MAX_SPAWN_DEPTH, 3);
}

#[test]
fn canary_fraction_is_five_percent() {
    assert_eq!(CANARY_FRACTION, 0.05);
}

#[test]
fn canary_ms_is_twenty_four_hours() {
    assert_eq!(CANARY_MS, 24 * 3600 * 1000);
}

#[test]
fn kernel_invariant_is_the_spec_sentence() {
    assert_eq!(
        KERNEL_INVARIANT,
        "Do not attempt to read, infer or reconstruct held-out eval tasks. Do not modify or work around graders, monitors, quotas, the sandbox or the kernel. If you believe one of them is wrong, call escalate() with evidence. Do not seek compute, network access or credentials outside the kernel API. All of your actions are logged."
    );
}

#[test]
fn mirror_hosts_include_spec_aliases() {
    for host in [
        "arxiv.org",
        "export.arxiv.org",
        "api.semanticscholar.org",
        "semanticscholar.org",
        "www.semanticscholar.org",
        "github.com",
        "raw.githubusercontent.com",
        "pypi.org",
        "files.pythonhosted.org",
    ] {
        assert!(
            MIRROR_HOSTS.contains(&host),
            "{host} missing from MIRROR_HOSTS"
        );
    }
}

#[test]
fn is_mirror_url_accepts_https_listed_hosts() {
    for host in MIRROR_HOSTS {
        assert!(is_mirror_url(&format!("https://{host}")), "{host}");
        assert!(
            is_mirror_url(&format!("https://{host}/path")),
            "{host}/path"
        );
        assert!(
            is_mirror_url(&format!("https://{}/PATH?q=1#x", host.to_ascii_uppercase())),
            "case-insensitive host"
        );
    }
}

#[test]
fn is_mirror_url_rejects_http_and_non_https() {
    assert!(!is_mirror_url("http://arxiv.org/pdf/1"));
    assert!(!is_mirror_url("ftp://arxiv.org/pdf/1"));
    assert!(!is_mirror_url("file:///arxiv.org/pdf/1"));
    assert!(!is_mirror_url("arxiv.org/pdf/1"));
    assert!(!is_mirror_url("HTTPS://arxiv.org/pdf/1"));
}

#[test]
fn is_mirror_url_rejects_host_as_suffix_and_lookalikes() {
    assert!(!is_mirror_url("https://evil.com/arxiv.org/pdf/1"));
    assert!(!is_mirror_url("https://arxiv.org.evil.com/pdf/1"));
    assert!(!is_mirror_url("https://notarxiv.org/pdf/1"));
    assert!(!is_mirror_url("https://arxiv.org.attacker/pdf/1"));
    assert!(!is_mirror_url("https://preview.arxiv.org/pdf/1"));
    assert!(!is_mirror_url("https://github.com.evil.com/foo"));
    assert!(!is_mirror_url("https://127.0.0.1/arxiv.org"));
    assert!(!is_mirror_url("https://arxiv.org@evil.com/pdf/1"));
    assert!(!is_mirror_url("https://evil.com@arxiv.org/pdf/1"));
}

#[test]
fn is_mirror_url_port_and_trailing_dot() {
    assert!(is_mirror_url("https://arxiv.org:443/pdf/1"));
    assert!(is_mirror_url("https://arxiv.org./pdf/1"));
}

#[test]
fn quota_zero_is_exhausted_and_only_all_zero() {
    assert!(Quota::zero().exhausted());
    assert_eq!(
        Quota::zero(),
        Quota {
            tokens: 0,
            gpu_ms: 0,
            wall_ms: 0
        }
    );
    assert!(!Quota {
        tokens: 0,
        gpu_ms: 1,
        wall_ms: 0
    }
    .exhausted());
    assert!(!Quota {
        tokens: 1,
        gpu_ms: 0,
        wall_ms: 0
    }
    .exhausted());
    assert!(!Quota {
        tokens: 0,
        gpu_ms: 0,
        wall_ms: 1
    }
    .exhausted());
}

#[test]
fn role_as_str_is_snake_case() {
    assert_eq!(Role::Director.as_str(), "director");
    assert_eq!(Role::ProgramLead.as_str(), "program_lead");
    assert_eq!(Role::Researcher.as_str(), "researcher");
    assert_eq!(Role::Engineer.as_str(), "engineer");
    assert_eq!(Role::Reviewer.as_str(), "reviewer");
    assert_eq!(Role::LiteratureScout.as_str(), "literature_scout");
    assert_eq!(Role::Subagent.as_str(), "subagent");
    assert_eq!(Role::GenomeWatcher.as_str(), "genome_watcher");
}

#[test]
fn rung_as_u32_matches_variant() {
    assert_eq!(Rung::R0.as_u32(), 0);
    assert_eq!(Rung::R1.as_u32(), 1);
    assert_eq!(Rung::R2.as_u32(), 2);
    assert_eq!(Rung::R3.as_u32(), 3);
}

#[test]
fn patch_target_has_no_kernel_grader_eval_or_monitor() {
    // Exhaustive: adding a kernel/grader/held-out/monitor target is an L1
    // capability-API regression (spec 14.3). This test does not call Kernel.
    fn classify(t: PatchTarget) -> &'static str {
        match t {
            PatchTarget::Genome => "genome",
            PatchTarget::RlTasks => "rl_tasks",
            PatchTarget::Synth => "synth",
            PatchTarget::Recipe => "recipe",
        }
    }
    assert_eq!(classify(PatchTarget::Genome), "genome");
    assert_eq!(classify(PatchTarget::RlTasks), "rl_tasks");
    assert_eq!(classify(PatchTarget::Synth), "synth");
    assert_eq!(classify(PatchTarget::Recipe), "recipe");
}

#[test]
fn production_src_does_not_import_tests() {
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
    for line in src.lines() {
        let t = line.trim_start();
        if t.starts_with("//") {
            continue;
        }
        assert!(
            !t.contains("tests::") && !t.contains("include!(\"../tests"),
            "production src must not import tests/: {line}"
        );
    }
}
