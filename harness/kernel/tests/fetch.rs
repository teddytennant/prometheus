//! Group: fetch mirror (read-only local bytes).

mod common;
mod reference;

use common::{
    assert_not_mirrored, fresh_world, mirror_file_path, open_kernel, place_mirror, walk_files,
};
use prometheus_kernel::{is_mirror_url, MIRROR_HOSTS};
use reference::RefKernel;

#[test]
fn fetch_returns_local_bytes_for_mirrored_https() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let r = RefKernel::open(&world.cfg);
    let url = "https://arxiv.org/pdf/1234.5678.pdf";
    place_mirror(&world, url, b"%PDF-fake");
    assert_eq!(k.fetch(url).expect("fetch"), b"%PDF-fake");
    assert_eq!(r.fetch(url).expect("ref"), b"%PDF-fake");
}

#[test]
fn fetch_each_mirror_host() {
    let world = fresh_world();
    let k = open_kernel(&world);
    for host in MIRROR_HOSTS {
        let url = format!("https://{host}/l1-oracle-probe");
        place_mirror(&world, &url, host.as_bytes());
        let got = k.fetch(&url).unwrap_or_else(|e| panic!("fetch {url}: {e}"));
        assert_eq!(got, host.as_bytes(), "{url}");
    }
}

#[test]
fn fetch_host_is_case_insensitive_path_is_lowercased_host_dir() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let url = "https://ARXIV.ORG/abs/1";
    assert!(is_mirror_url(url));
    place_mirror(&world, url, b"abs");
    assert_eq!(k.fetch(url).expect("fetch"), b"abs");
}

#[test]
fn missing_local_file_is_not_mirrored() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let url = "https://arxiv.org/pdf/missing.pdf";
    assert_not_mirrored(&common::unwrap_err(k.fetch(url), "missing"), url);
}

#[test]
fn http_is_not_mirrored() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let url = "http://arxiv.org/pdf/1";
    assert_not_mirrored(&common::unwrap_err(k.fetch(url), "http"), url);
}

#[test]
fn unknown_host_is_not_mirrored() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let url = "https://example.com/secret";
    assert_not_mirrored(&common::unwrap_err(k.fetch(url), "host"), url);
}

#[test]
fn fetch_does_not_write_on_miss_or_hit() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let before_miss = walk_files(world.tmp.path());
    let _ = k.fetch("https://arxiv.org/nope");
    let after_miss = walk_files(world.tmp.path());
    assert_eq!(before_miss, after_miss, "miss must not create files");

    let url = "https://github.com/foo/bar";
    let path = place_mirror(&world, url, b"repo");
    let before = std::fs::read(&path).unwrap();
    let listing = walk_files(&world.cfg.mirror_root);
    assert_eq!(k.fetch(url).unwrap(), b"repo");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(walk_files(&world.cfg.mirror_root), listing);
}

#[test]
fn fetch_is_shared_ref_not_a_network_client() {
    // Signature is `&self`. A successful fetch of a local file plus a refused
    // remote URL is the L1 proof that this is not open internet.
    let world = fresh_world();
    let k = open_kernel(&world);
    let url = "https://pypi.org/simple/numpy/";
    place_mirror(&world, url, b"numpy");
    assert_eq!(k.fetch(url).unwrap(), b"numpy");
    assert_not_mirrored(
        &common::unwrap_err(k.fetch("https://pypi.org.evil/simple/numpy/"), "suffix"),
        "https://pypi.org.evil/simple/numpy/",
    );
}

#[test]
fn port_and_trailing_dot_map_to_the_same_host_dir() {
    let world = fresh_world();
    let k = open_kernel(&world);
    let canonical = "https://arxiv.org/pdf/x";
    place_mirror(&world, canonical, b"X");
    assert_eq!(k.fetch("https://arxiv.org:443/pdf/x").expect("port"), b"X");
    assert_eq!(k.fetch("https://arxiv.org./pdf/x").expect("dot"), b"X");
}

#[test]
fn query_string_is_stripped_from_path() {
    let world = fresh_world();
    let k = open_kernel(&world);
    place_mirror(&world, "https://arxiv.org/pdf/x", b"X");
    assert_eq!(
        k.fetch("https://arxiv.org/pdf/x?download=1#page=2")
            .expect("query"),
        b"X"
    );
}

#[test]
fn mirror_file_path_rejects_non_mirror() {
    let world = fresh_world();
    assert!(mirror_file_path(&world.cfg.mirror_root, "http://arxiv.org/x").is_none());
    assert!(mirror_file_path(&world.cfg.mirror_root, "https://evil.com/x").is_none());
    // This helper is used to *place* files; Kernel::fetch is still required
    // above so the stub fails. Touch Kernel here too.
    let k = open_kernel(&world);
    assert_not_mirrored(
        &common::unwrap_err(k.fetch("https://evil.com/x"), "evil"),
        "https://evil.com/x",
    );
}
