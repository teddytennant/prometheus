//! Oracle tests for `render_job`.
//!
//! Tests always go through `render_job` (never inspect stub template files
//! alone, which would pass today). `ref_*` pass on the current stub templates
//! because the reference only substitutes placeholders. `prod_*` panic
//! `unimplemented!` until F4 implements `render_job`.
//!
//! `gpus == 0` is `Error::Other` for every stage, including V10: pass a dummy
//! `gpus >= 1`. V10 does not interpolate `{{GPUS}}`.
//!
//! A few `prod_*` tests additionally require filled template bodies (no
//! "interface stub" marker). Those stay red until the implementer writes the
//! real job scripts; they fail today via `unimplemented!`, not by reading
//! files.

mod reference;

use prometheus_verify_ncshare::{render_job, Error, Stage, JOB_PREFIX};

fn expected_job_name(run_id: &str, stage: Stage) -> String {
    format!("#SBATCH -J {JOB_PREFIX}-{run_id}-{}", stage.as_str())
}

// ---------------------------------------------------------------------------
// Reference
// ---------------------------------------------------------------------------

#[test]
fn ref_render_job_substitutes_placeholders() {
    let out = reference::render_job(Stage::V0, "run42", "04:00:00", 8).unwrap();
    assert!(!out.contains("{{RUN_ID}}"), "{out}");
    assert!(!out.contains("{{WALLTIME}}"), "{out}");
    assert!(!out.contains("{{GPUS}}"), "{out}");
    assert!(out.contains("run42"), "{out}");
    assert!(out.contains("04:00:00"), "{out}");
    assert!(out.contains("8"), "{out}");
    assert!(
        out.contains(&expected_job_name("run42", Stage::V0)),
        "{out}"
    );
}

#[test]
fn ref_render_job_all_stages_job_name() {
    for stage in Stage::ALL {
        let out = reference::render_job(stage, "abc", "1:00:00", 4).unwrap();
        assert!(
            out.contains(&expected_job_name("abc", stage)),
            "missing job name for {}: {out}",
            stage.as_str()
        );
        assert!(!out.contains("{{RUN_ID}}"));
        assert!(!out.contains("{{WALLTIME}}"));
        assert!(!out.contains("{{GPUS}}"));
    }
}

#[test]
fn ref_render_job_rejects_gpus_zero() {
    let err = reference::render_job(Stage::V0, "r", "1:00:00", 0).unwrap_err();
    assert!(
        matches!(err, Error::Other(_)),
        "gpus=0 must be Error::Other, got {err:?}"
    );
    let err = reference::render_job(Stage::V10, "r", "1:00:00", 0).unwrap_err();
    assert!(matches!(err, Error::Other(_)), "{err:?}");
}

#[test]
fn ref_render_job_rejects_empty_run_id_or_walltime() {
    assert!(reference::render_job(Stage::V1, "", "1:00:00", 1).is_err());
    assert!(reference::render_job(Stage::V1, "r", "", 1).is_err());
}

#[test]
fn ref_render_job_gpus_decimal() {
    let out = reference::render_job(Stage::V2, "r", "2:00:00", 16).unwrap();
    assert!(out.contains("16"), "{out}");
    assert!(!out.contains("{{GPUS}}"));
}

// ---------------------------------------------------------------------------
// Production public API — must fail on unimplemented! stub
// ---------------------------------------------------------------------------

#[test]
fn prod_render_job_substitutes_placeholders() {
    let out = render_job(Stage::V0, "run42", "04:00:00", 8).unwrap();
    assert!(!out.contains("{{RUN_ID}}"), "{out}");
    assert!(!out.contains("{{WALLTIME}}"), "{out}");
    assert!(!out.contains("{{GPUS}}"), "{out}");
    assert!(out.contains("run42"), "{out}");
    assert!(out.contains("04:00:00"), "{out}");
    assert!(
        out.contains(&expected_job_name("run42", Stage::V0)),
        "{out}"
    );
}

#[test]
fn prod_render_job_all_stages_job_name() {
    for stage in Stage::ALL {
        let out = render_job(stage, "abc", "1:00:00", 4).unwrap();
        assert!(
            out.contains(&expected_job_name("abc", stage)),
            "missing job name for {}: {out}",
            stage.as_str()
        );
        assert!(!out.contains("{{RUN_ID}}"));
        assert!(!out.contains("{{WALLTIME}}"));
        assert!(!out.contains("{{GPUS}}"));
    }
}

#[test]
fn prod_render_job_rejects_gpus_zero() {
    let err = render_job(Stage::V0, "r", "1:00:00", 0).unwrap_err();
    assert!(
        matches!(err, Error::Other(_)),
        "gpus=0 must be Error::Other, got {err:?}"
    );
}

#[test]
fn prod_render_job_rejects_empty_run_id_or_walltime() {
    assert!(render_job(Stage::V1, "", "1:00:00", 1).is_err());
    assert!(render_job(Stage::V1, "r", "", 1).is_err());
}

#[test]
fn prod_render_job_matches_reference_substitution() {
    let prod = render_job(Stage::V3, "id9", "08:00:00", 8).unwrap();
    let refer = reference::render_job(Stage::V3, "id9", "08:00:00", 8).unwrap();
    assert_eq!(prod, refer);
}

/// Passes only once V0's template body is filled (not the interface stub).
/// Fails today because `render_job` is `unimplemented!`.
#[test]
fn prod_render_job_v0_not_interface_stub() {
    let out = render_job(Stage::V0, "run42", "04:00:00", 8).unwrap();
    let lower = out.to_ascii_lowercase();
    assert!(
        !lower.contains("interface stub"),
        "V0 template is still the interface stub"
    );
}

/// Passes once the rendered V0 script mentions nccl (filled template or, after
/// substitution-only, the stock comment). Fails today via `unimplemented!`.
#[test]
fn prod_render_job_v0_mentions_nccl() {
    let out = render_job(Stage::V0, "run42", "04:00:00", 8).unwrap();
    let lower = out.to_ascii_lowercase();
    assert!(
        lower.contains("nccl"),
        "rendered V0 must mention nccl: {out}"
    );
}

#[test]
fn prod_render_job_v0_mentions_kvm() {
    let out = render_job(Stage::V0, "run42", "04:00:00", 8).unwrap();
    let lower = out.to_ascii_lowercase();
    assert!(lower.contains("kvm"), "rendered V0 must mention kvm: {out}");
}
