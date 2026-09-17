//! Group: GPU-ish calls fail closed without `GpuConfig`; walltime → `GpuTimeLimit`.
//! Real-device coverage is gated on cargo feature `gpu` (V-stage D1, spec 9.4).

mod common;
mod reference;

use common::{python, sample_image, src_does_not_import_tests};
use prometheus_envs::{Error, GpuConfig, Pool, PoolConfig};
use reference::RefPool;

fn boot(
    gpu: Option<GpuConfig>,
) -> (
    Pool<prometheus_envs::InProcess>,
    RefPool,
    prometheus_envs::SandboxId,
    prometheus_envs::SandboxId,
    prometheus_envs::SnapshotId,
) {
    src_does_not_import_tests();
    let cfg = PoolConfig {
        gpu,
        ..common::cpu_cfg()
    };
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-gpu");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    let a = prod.fork_group(&ps, 1, 0).unwrap().sandboxes[0].clone();
    let ra = refer.fork_group(&rs, 1, 0).unwrap().sandboxes[0].clone();
    (prod, refer, a, ra, ps)
}

fn gpu_python(snap: &str, work_s: u32) -> prometheus_envs::ToolRequest {
    // Tiny interpreter still accepts `print`; GPU-ish-ness is the payload flag.
    let mut r = python("gpu-py", snap, "print(1)");
    r.payload
        .as_object_mut()
        .unwrap()
        .insert("work_s".into(), serde_json::json!(work_s));
    r.payload
        .as_object_mut()
        .unwrap()
        .insert("gpu".into(), serde_json::json!(true));
    r
}

#[test]
fn gpu_config_none_fails_closed() {
    let (mut prod, mut refer, a, ra, ps) = boot(None);
    let req = gpu_python(&ps.0, 1);
    let pe = prod.call(&a, req.clone(), 0);
    let re = refer.call(&ra, req, 0);
    match pe {
        Err(Error::Backend(_)) => {}
        Ok(r) => assert!(!r.ok, "gpu-less call must fail closed"),
        other => panic!("expected fail closed, got {other:?}"),
    }
    match re {
        Err(Error::Backend(_)) | Err(Error::GpuTimeLimit) => {}
        Ok(r) if !r.ok => {}
        other => panic!("ref {other:?}"),
    }
}

#[test]
fn gpu_walltime_exceeded_is_gpu_time_limit() {
    let gpu = GpuConfig {
        gpus: 1,
        walltime_s: 3,
        mig: false,
    };
    let (mut prod, mut refer, a, ra, ps) = boot(Some(gpu));
    let req = gpu_python(&ps.0, 10);
    assert_eq!(prod.call(&a, req.clone(), 0), Err(Error::GpuTimeLimit));
    assert_eq!(refer.call(&ra, req, 0), Err(Error::GpuTimeLimit));
}

#[test]
fn gpu_walltime_accumulates_across_calls() {
    let gpu = GpuConfig {
        gpus: 1,
        walltime_s: 5,
        mig: false,
    };
    let (mut prod, mut refer, a, ra, ps) = boot(Some(gpu));
    let first = gpu_python(&ps.0, 3);
    assert!(prod.call(&a, first.clone(), 0).unwrap().ok);
    assert!(refer.call(&ra, first, 0).unwrap().ok);
    let second = gpu_python(&ps.0, 3);
    assert_eq!(prod.call(&a, second.clone(), 1), Err(Error::GpuTimeLimit));
    assert_eq!(refer.call(&ra, second, 1), Err(Error::GpuTimeLimit));
}

/// V-stage D1 (spec 9.4): real GPU device path, skipped on CPU `cargo test`.
#[cfg(feature = "gpu")]
#[test]
fn gpu_feature_walltime_on_configured_device() {
    let gpu = GpuConfig {
        gpus: 1,
        walltime_s: 1,
        mig: false,
    };
    let (mut prod, mut refer, a, ra, ps) = boot(Some(gpu));
    let req = gpu_python(&ps.0, 30);
    assert_eq!(prod.call(&a, req.clone(), 0), Err(Error::GpuTimeLimit));
    assert_eq!(refer.call(&ra, req, 0), Err(Error::GpuTimeLimit));
}
