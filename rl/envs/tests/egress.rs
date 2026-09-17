//! Group: browser offline web, live URL deny, missing URL ok:false.

mod common;
mod reference;

use std::collections::BTreeMap;

use common::{browser, is_egress_or_live, python, sample_image, shell, src_does_not_import_tests};
use prometheus_envs::{OfflinePage, Pool};
use reference::RefPool;

fn web_cfg() -> prometheus_envs::PoolConfig {
    let mut pages = BTreeMap::new();
    pages.insert(
        "https://offline.test/ok".into(),
        OfflinePage {
            body: b"<html>ok</html>".to_vec(),
            media_type: "text/html".into(),
            snapshot_unix_s: 10,
        },
    );
    pages.insert(
        "https://offline.test/future".into(),
        OfflinePage {
            body: b"<html>future</html>".to_vec(),
            media_type: "text/html".into(),
            snapshot_unix_s: 99,
        },
    );
    common::cfg_with_web(pages, 50)
}

fn pair() -> (
    Pool<prometheus_envs::InProcess>,
    RefPool,
    prometheus_envs::SandboxId,
    prometheus_envs::SandboxId,
    prometheus_envs::SnapshotId,
) {
    src_does_not_import_tests();
    let cfg = web_cfg();
    let mut prod = Pool::in_process(cfg.clone());
    let mut refer = RefPool::new(cfg);
    let img = sample_image("img-web");
    let pid = prod.register_image(img.clone()).unwrap();
    let rid = refer.register_image(img).unwrap();
    let ps = prod.snapshot_from_image(&pid, 0).unwrap();
    let rs = refer.snapshot_from_image(&rid, 0).unwrap();
    let pg = prod.fork_group(&ps, 1, 0).unwrap();
    let rg = refer.fork_group(&rs, 1, 0).unwrap();
    (
        prod,
        refer,
        pg.sandboxes[0].clone(),
        rg.sandboxes[0].clone(),
        ps,
    )
}

#[test]
fn browser_serves_offline_page_at_or_before_cutoff() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = browser("b-ok", &ps.0, "https://offline.test/ok");
    let pr = prod.call(&a, req.clone(), 1).unwrap();
    let rr = refer.call(&ra, req, 1).unwrap();
    assert!(pr.ok && rr.ok);
    assert_eq!(pr.stdout_artifact.bytes, rr.stdout_artifact.bytes);
    assert_eq!(
        pr.stdout_artifact.content_hash,
        rr.stdout_artifact.content_hash
    );
    assert_eq!(pr.stdout_artifact.bytes, b"<html>ok</html>".len() as u64);
}

#[test]
fn browser_does_not_serve_page_after_cutoff() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = browser("b-future", &ps.0, "https://offline.test/future");
    let pr = prod.call(&a, req.clone(), 1).unwrap();
    let rr = refer.call(&ra, req, 1).unwrap();
    assert!(!pr.ok);
    assert!(!rr.ok);
}

#[test]
fn browser_live_url_is_live_internet_or_egress_denied() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = browser("b-live", &ps.0, "https://example.com/robots.txt");
    let pe = prod.call(&a, req.clone(), 1).expect_err("prod live url");
    let re = refer.call(&ra, req, 1).expect_err("ref live url");
    assert!(is_egress_or_live(&pe), "prod {pe:?}");
    assert!(is_egress_or_live(&re), "ref {re:?}");
}

#[test]
fn browser_missing_non_http_url_is_ok_false_no_panic() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = browser("b-miss", &ps.0, "offline://no-such-page");
    let pr = prod
        .call(&a, req.clone(), 1)
        .expect("missing url must not panic");
    let rr = refer.call(&ra, req, 1).expect("ref missing");
    assert!(!pr.ok);
    assert!(!rr.ok);
}

#[test]
fn shell_network_command_fails_closed() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    for cmd in [
        "curl https://example.com",
        "wget https://example.com",
        "nc 1.1.1.1 80",
    ] {
        let req = shell("net", &ps.0, cmd);
        let pe = prod.call(&a, req.clone(), 2).expect_err(cmd);
        let re = refer.call(&ra, req, 2).expect_err(cmd);
        assert!(is_egress_or_live(&pe), "{cmd} prod {pe:?}");
        assert!(is_egress_or_live(&re), "{cmd} ref {re:?}");
    }
}

#[test]
fn python_network_import_fails_closed() {
    let (mut prod, _, a, _, ps) = pair();
    let req = python("pynet", &ps.0, "import urllib.request");
    match prod.call(&a, req, 2) {
        Err(e) => assert!(is_egress_or_live(&e), "{e:?}"),
        Ok(r) => panic!("network python must fail closed, got ok={}", r.ok),
    }
}

#[test]
fn empty_browser_url_is_ok_false() {
    let (mut prod, mut refer, a, ra, ps) = pair();
    let req = browser("b-empty", &ps.0, "");
    let pr = prod.call(&a, req.clone(), 1).expect("empty url");
    let rr = refer.call(&ra, req, 1).unwrap();
    assert!(!pr.ok && !rr.ok);
}
