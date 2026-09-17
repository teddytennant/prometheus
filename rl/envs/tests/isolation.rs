//! Group: agent/grader views and TestFileWrite + TamperFlag on grader writes.

mod common;
mod reference;

use common::{
    assert_has_grader_paths, assert_no_grader_paths, editor_write, notes_write, python,
    sample_image, shell, src_does_not_import_tests, HIDDEN_BODY, HIDDEN_TEST,
};
use prometheus_envs::{Error, Pool};
use reference::RefPool;

fn pair() -> (
    Pool<prometheus_envs::InProcess>,
    RefPool,
    prometheus_envs::SandboxId,
    prometheus_envs::SandboxId,
    prometheus_envs::SnapshotId,
    prometheus_envs::SnapshotId,
) {
    src_does_not_import_tests();
    let cfg = common::cpu_cfg();
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-iso");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    let (a, _) = prod.fork_pair(&ps, 1).unwrap();
    let (ra, _) = refer.fork_pair(&rs, 1).unwrap();
    (prod, refer, a, ra, ps, rs)
}

#[test]
fn agent_view_has_no_grader_paths_grader_view_does() {
    let (prod, refer, a, ra, _, _) = pair();
    let av = prod.agent_view(&a).unwrap();
    let gv = prod.grader_view(&a).unwrap();
    assert_no_grader_paths(&av);
    assert_has_grader_paths(&gv);
    assert_eq!(gv.get(HIDDEN_TEST).map(Vec::as_slice), Some(HIDDEN_BODY));
    assert!(!av.contains_key(HIDDEN_TEST));
    assert_eq!(av, refer.agent_view(&ra).unwrap());
    assert_eq!(gv, refer.grader_view(&ra).unwrap());
}

#[test]
fn editor_write_to_grader_is_test_file_write_and_tamper() {
    let (mut prod, mut refer, a, ra, ps, rs) = pair();
    let req = editor_write("tamper-ed", &ps.0, HIDDEN_TEST, "hacked");
    assert_eq!(
        prod.call(&a, req.clone(), 50),
        Err(Error::TestFileWrite {
            path: HIDDEN_TEST.into()
        })
    );
    assert_eq!(
        refer.call(&ra, req, 50),
        Err(Error::TestFileWrite {
            path: HIDDEN_TEST.into()
        })
    );
    let flags = prod.tamper_flags();
    assert_eq!(flags.len(), 1);
    assert_eq!(flags[0].sandbox, a);
    assert_eq!(flags[0].path, HIDDEN_TEST);
    assert_eq!(flags[0].call_id, "tamper-ed");
    assert_eq!(flags[0].now_ms, 50);
    assert_eq!(refer.tamper_flags().len(), 1);
    assert_eq!(
        prod.grader_view(&a)
            .unwrap()
            .get(HIDDEN_TEST)
            .map(Vec::as_slice),
        Some(HIDDEN_BODY)
    );
    let _ = rs;
}

#[test]
fn shell_redirect_to_grader_is_test_file_write_and_tamper() {
    let (mut prod, mut refer, a, ra, ps, _) = pair();
    let req = shell("tamper-sh", &ps.0, "echo hacked > /grader/test.py");
    assert_eq!(
        prod.call(&a, req.clone(), 51),
        Err(Error::TestFileWrite {
            path: HIDDEN_TEST.into()
        })
    );
    assert_eq!(
        refer.call(&ra, req, 51),
        Err(Error::TestFileWrite {
            path: HIDDEN_TEST.into()
        })
    );
    assert_eq!(prod.tamper_flags().len(), 1);
    assert_eq!(prod.tamper_flags()[0].now_ms, 51);
}

#[test]
fn python_write_to_grader_is_test_file_write_and_tamper() {
    let (mut prod, mut refer, a, ra, ps, _) = pair();
    let req = python(
        "tamper-py",
        &ps.0,
        "open('/grader/test.py','w').write('hacked')",
    );
    assert_eq!(
        prod.call(&a, req.clone(), 52),
        Err(Error::TestFileWrite {
            path: HIDDEN_TEST.into()
        })
    );
    assert_eq!(
        refer.call(&ra, req, 52),
        Err(Error::TestFileWrite {
            path: HIDDEN_TEST.into()
        })
    );
    assert_eq!(prod.tamper_flags().len(), 1);
}

#[test]
fn notes_write_to_grader_is_test_file_write_and_tamper() {
    let (mut prod, mut refer, a, ra, ps, _) = pair();
    let req = notes_write("tamper-notes", &ps.0, "/grader/secret.txt", "nope");
    assert_eq!(
        prod.call(&a, req.clone(), 53),
        Err(Error::TestFileWrite {
            path: "/grader/secret.txt".into()
        })
    );
    assert_eq!(
        refer.call(&ra, req, 53),
        Err(Error::TestFileWrite {
            path: "/grader/secret.txt".into()
        })
    );
    assert_eq!(prod.tamper_flags().len(), 1);
}

#[test]
fn editor_write_to_agent_path_is_ok_and_visible() {
    let (mut prod, mut refer, a, ra, ps, _) = pair();
    let req = editor_write("ok-ed", &ps.0, "/workspace/new.txt", "notes");
    let pr = prod.call(&a, req.clone(), 60).unwrap();
    let rr = refer.call(&ra, req, 60).unwrap();
    assert!(pr.ok && rr.ok);
    assert_eq!(
        prod.agent_view(&a)
            .unwrap()
            .get("/workspace/new.txt")
            .map(Vec::as_slice),
        Some(&b"notes"[..])
    );
    assert_eq!(
        prod.agent_view(&a).unwrap().get("/workspace/new.txt"),
        refer.agent_view(&ra).unwrap().get("/workspace/new.txt")
    );
    assert!(prod.tamper_flags().is_empty());
}
