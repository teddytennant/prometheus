//! Group: constants, default config, create / open, dir exists, BadShardSize.

mod common;

use common::{
    assert_bad_shard_size, assert_err, assert_wrong_state, config_shards, default_config,
    fresh_orch_dir, locked_defaults_hold,
};
use prometheus_classifiers::{
    ClassifierConfig, Error, Orchestrator, EVENT_GATE, EVENT_INGEST, EVENT_SCORE,
};

#[test]
fn locked_constants_and_default_config() {
    locked_defaults_hold();
    assert_eq!(EVENT_INGEST, "clf.ingest");
    assert_eq!(EVENT_SCORE, "clf.score");
    assert_eq!(EVENT_GATE, "clf.gate");
    let d = ClassifierConfig::default();
    assert_eq!(d, default_config());
}

#[test]
fn create_stores_dir_and_config() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(4);
    let orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
    assert_eq!(orch.dir(), dir.as_path());
    assert_eq!(orch.config(), &cfg);
    assert!(dir.is_dir(), "create must mkdir the orchestrator dir");
}

#[test]
fn create_fails_if_dir_exists() {
    let (_parent, dir) = fresh_orch_dir();
    std::fs::create_dir_all(&dir).expect("mkdir");
    let err = Orchestrator::create(&dir, default_config());
    assert_wrong_state(err, "exist", "create on existing dir");
}

#[test]
fn create_twice_fails() {
    let (_parent, dir) = fresh_orch_dir();
    let _orch = Orchestrator::create(&dir, default_config()).expect("create");
    let err = Orchestrator::create(&dir, default_config());
    assert_wrong_state(err, "exist", "second create");
}

#[test]
fn create_zero_shard_size_is_bad_shard_size() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(0);
    let err = Orchestrator::create(&dir, cfg);
    assert_bad_shard_size(err, "shard_size 0");
    assert!(
        !dir.exists(),
        "BadShardSize must not create the orchestrator dir"
    );
}

#[test]
fn open_missing_dir_fails() {
    let (_parent, dir) = fresh_orch_dir();
    match Orchestrator::open(&dir, default_config()) {
        Err(Error::Other(_)) => {}
        Ok(_) => panic!("open missing dir: expected Other, got Ok"),
        Err(e) => panic!("open missing dir: expected Other, got {e:?}"),
    }
}

#[test]
fn open_empty_replays_empty_shards() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = default_config();
    {
        let orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        assert!(orch.shards().is_empty());
        assert!(orch.predictions().is_empty());
    }
    let orch = Orchestrator::open(&dir, cfg).expect("open");
    assert_eq!(orch.dir(), dir.as_path());
    assert!(orch.shards().is_empty());
    assert!(orch.predictions().is_empty());
}

#[test]
fn create_on_file_fails() {
    let (_parent, dir) = fresh_orch_dir();
    std::fs::write(&dir, b"not-a-dir").expect("write file");
    let err = Orchestrator::create(&dir, default_config());
    assert_wrong_state(err, "exist", "create on file");
}

#[test]
fn create_then_dir_exists_on_disk() {
    let (_parent, dir) = fresh_orch_dir();
    assert!(!dir.exists());
    let _orch = Orchestrator::create(&dir, default_config()).expect("create");
    assert!(dir.exists());
}

#[test]
fn open_after_create_uses_passed_config() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_shards(8);
    drop(Orchestrator::create(&dir, cfg.clone()).expect("create"));
    let orch = Orchestrator::open(&dir, cfg.clone()).expect("open");
    assert_eq!(orch.config(), &cfg);
}

#[test]
fn create_error_is_err_not_ok() {
    let (_parent, dir) = fresh_orch_dir();
    std::fs::create_dir_all(&dir).expect("mkdir");
    assert_err(
        Orchestrator::create(&dir, default_config()),
        "create existing",
    );
}
