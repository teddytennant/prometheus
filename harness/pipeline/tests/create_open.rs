//! Group: create / open, dir layout, create-fails-if-exists, empty replay.

mod common;
mod reference;

use common::{
    assert_fresh_interface, assert_state, default_pipeline_config, default_queue_config, fresh_dir,
    module, queue_dir, unwrap_err,
};
use prometheus_pipeline::{
    Pipeline, PipelineConfig, DEFAULT_MAX_ROUNDS, DEFAULT_N_IMPLEMENTERS,
};
use reference::RefPipeline;

#[test]
fn create_starts_at_interface_round_zero() {
    let (_parent, dir) = fresh_dir();
    let cfg = default_pipeline_config();
    let qcfg = default_queue_config();
    let m = module();
    let p = Pipeline::create(&dir, m.clone(), cfg.clone(), qcfg.clone()).expect("create");
    let r = RefPipeline::create(
        _parent.path().join("refer"),
        m.clone(),
        cfg.clone(),
        qcfg,
    )
    .expect("ref create");
    assert_fresh_interface(&p, &dir, &m, &cfg);
    assert_fresh_interface(&r, &_parent.path().join("refer"), &m, &cfg);
    assert_state(&p, &r, "create");
    assert_eq!(p.config().n_implementers, DEFAULT_N_IMPLEMENTERS);
    assert_eq!(p.config().max_rounds, DEFAULT_MAX_ROUNDS);
    assert_eq!(p.queue().log().len(), 0, "create writes no queue events");
}

#[test]
fn create_owns_queue_at_dir_queue() {
    let (_parent, dir) = fresh_dir();
    let p = Pipeline::create(
        &dir,
        module(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("create");
    assert_eq!(p.dir(), dir.as_path());
    assert_eq!(p.queue().dir(), queue_dir(&dir));
    assert!(queue_dir(&dir).is_dir(), "queue subdir must exist");
}

#[test]
fn create_fails_if_dir_exists() {
    let (_parent, dir) = fresh_dir();
    std::fs::create_dir_all(&dir).expect("mkdir");
    let err = unwrap_err(
        Pipeline::create(
            &dir,
            module(),
            default_pipeline_config(),
            default_queue_config(),
        ),
        "create existing",
    );
    let _ = err;
    let rerr = RefPipeline::create(
        &dir,
        module(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect_err("ref create existing");
    let _ = rerr;
}

#[test]
fn create_second_time_fails() {
    let (_parent, dir) = fresh_dir();
    let _p = Pipeline::create(
        &dir,
        module(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("first create");
    unwrap_err(
        Pipeline::create(
            &dir,
            module(),
            default_pipeline_config(),
            default_queue_config(),
        ),
        "second create",
    );
}

#[test]
fn open_missing_dir_fails() {
    let (_parent, dir) = fresh_dir();
    unwrap_err(
        Pipeline::open(
            &dir,
            module(),
            default_pipeline_config(),
            default_queue_config(),
        ),
        "open missing",
    );
}

#[test]
fn open_after_create_restores_interface() {
    let (_parent, dir) = fresh_dir();
    let cfg = default_pipeline_config();
    let qcfg = default_queue_config();
    let m = module();
    {
        let p = Pipeline::create(&dir, m.clone(), cfg.clone(), qcfg.clone()).expect("create");
        assert_eq!(p.stage(), prometheus_pipeline::Stage::Interface);
        assert_eq!(p.round(), 0);
        assert!(!p.stub_failed());
    }
    let p = Pipeline::open(&dir, m.clone(), cfg.clone(), qcfg).expect("open");
    assert_fresh_interface(&p, &dir, &m, &cfg);
}

#[test]
fn default_config_is_three_and_three() {
    let (_parent, dir) = fresh_dir();
    let cfg = PipelineConfig::default();
    let p = Pipeline::create(&dir, module(), cfg, default_queue_config()).expect("create");
    assert_eq!(p.config().n_implementers, 3);
    assert_eq!(p.config().max_rounds, 3);
    assert_eq!(DEFAULT_N_IMPLEMENTERS, 3);
    assert_eq!(DEFAULT_MAX_ROUNDS, 3);
}

#[test]
fn custom_config_is_stored() {
    let (_parent, dir) = fresh_dir();
    let cfg = PipelineConfig {
        n_implementers: 5,
        max_rounds: 1,
    };
    let p = Pipeline::create(&dir, module(), cfg.clone(), default_queue_config()).expect("create");
    assert_eq!(p.config().n_implementers, 5);
    assert_eq!(p.config().max_rounds, 1);
    assert_eq!(p.module().0, "mod-h10");
}

#[test]
fn open_uses_caller_module_and_config() {
    let (_parent, dir) = fresh_dir();
    let created = PipelineConfig {
        n_implementers: 3,
        max_rounds: 3,
    };
    Pipeline::create(&dir, module(), created, default_queue_config()).expect("create");
    let opened_mod = prometheus_pipeline::ModuleId("other".into());
    let opened_cfg = PipelineConfig {
        n_implementers: 1,
        max_rounds: 9,
    };
    let p = Pipeline::open(
        &dir,
        opened_mod.clone(),
        opened_cfg.clone(),
        default_queue_config(),
    )
    .expect("open");
    assert_eq!(p.module(), &opened_mod);
    assert_eq!(p.config().n_implementers, 1);
    assert_eq!(p.config().max_rounds, 9);
}
