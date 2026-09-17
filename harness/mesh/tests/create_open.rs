//! Group: Mesh::create / open, create-if-exists fails, open missing fails,
//! freeze-not-set after create+open.

mod common;
mod reference;

use common::{
    assert_already_exists, assert_err, fresh_node_dir, slot_config, trusted_config, unwrap_err,
};
use prometheus_mesh::Mesh;
use reference::RefMesh;

#[test]
fn create_fails_if_dir_exists() {
    let (_parent, dir) = fresh_node_dir();
    std::fs::create_dir_all(&dir).expect("mkdir");
    let cfg = slot_config(0);
    let err = unwrap_err(Mesh::create(&dir, cfg.clone()), "create existing");
    assert_already_exists(&err, "create on existing dir");
    let rerr = unwrap_err(RefMesh::create(&dir, cfg), "ref create existing");
    assert_already_exists(&rerr, "ref create on existing dir");
}

#[test]
fn create_twice_fails() {
    let (_parent, dir) = fresh_node_dir();
    let cfg = slot_config(0);
    let _m = Mesh::create(&dir, cfg.clone()).expect("create");
    let err = unwrap_err(Mesh::create(&dir, cfg), "second create");
    assert_already_exists(&err, "second create");
}

#[test]
fn create_uses_given_dir_and_config() {
    let (_parent, dir) = fresh_node_dir();
    let cfg = trusted_config("alice", "key-alice");
    let m = Mesh::create(&dir, cfg.clone()).expect("create");
    assert_eq!(m.dir(), dir.as_path());
    assert_eq!(m.config().this_id, cfg.this_id);
    assert_eq!(m.config().this_key, cfg.this_key);
    assert!(m.config().trusted);
    assert!(!m.is_frozen());
}

#[test]
fn open_missing_dir_fails() {
    let (_parent, dir) = fresh_node_dir();
    let err = Mesh::open(&dir, slot_config(0));
    assert_err(err, "open missing dir");
}

#[test]
fn create_then_open_not_frozen() {
    let (_parent, dir) = fresh_node_dir();
    let cfg = slot_config(0);
    {
        let m = Mesh::create(&dir, cfg.clone()).expect("create");
        assert!(!m.is_frozen());
        assert_eq!(m.dir(), dir.as_path());
    }
    let m = Mesh::open(&dir, cfg.clone()).expect("open");
    assert_eq!(m.dir(), dir.as_path());
    assert_eq!(m.config().this_id, cfg.this_id);
    assert!(!m.is_frozen());
}
