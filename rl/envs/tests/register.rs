//! Group: `register_image` credential / path / hash rules vs the in-memory reference.

mod common;
mod reference;

use common::{assert_err_eq, sample_image, src_does_not_import_tests};
use prometheus_envs::{Error, Image, Pool, GRADER_ROOT};
use reference::{credential_like_key, hidden_tests_hash, RefPool};

fn pools() -> (Pool<prometheus_envs::InProcess>, RefPool) {
    src_does_not_import_tests();
    let cfg = common::cpu_cfg();
    (Pool::in_process(cfg.clone()), RefPool::new(cfg))
}

#[test]
fn register_image_accepts_clean_sample() {
    let (mut prod, mut refer) = pools();
    let img = sample_image("img-clean");
    let expected_hash = hidden_tests_hash(&img.hidden_tests);
    let a = prod.register_image(img.clone()).expect("prod register");
    let b = refer.register_image(img).expect("ref register");
    assert_eq!(a, b);
    assert_eq!(
        expected_hash,
        "e4cc549b51a37a1aec379f109688700d21900b24db7a8ec13d4e95cf8382d375"
    );
}

#[test]
fn register_image_rejects_credential_like_env_keys() {
    let keys = [
        "API_SECRET",
        "secret",
        "myToken",
        "DB_PASSWORD",
        "CREDENTIAL_FILE",
        "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key",
        "SSH_AUTH_SOCK",
        "ssh_private_key",
    ];
    for k in keys {
        assert!(credential_like_key(k), "{k} must be credential-like");
        let (mut prod, mut refer) = pools();
        let mut img = sample_image("img-cred");
        img.env.insert(k.to_string(), "x".into());
        assert_err_eq(
            prod.register_image(img.clone()),
            refer.register_image(img),
        );
        let (mut prod, _) = pools();
        let mut img = sample_image("img-cred");
        img.env.insert(k.to_string(), "x".into());
        match prod.register_image(img) {
            Err(Error::CredentialInSandbox { name }) => assert_eq!(name, k),
            other => panic!("{k}: expected CredentialInSandbox, got {other:?}"),
        }
    }
}

#[test]
fn register_image_allows_non_credential_env_keys() {
    assert!(!credential_like_key("PATH"));
    assert!(!credential_like_key("HOME"));
    assert!(!credential_like_key("BASE_AWSX"));
    let (mut prod, mut refer) = pools();
    let mut img = sample_image("img-env");
    img.env.insert("PATH".into(), "/bin".into());
    img.env.insert("HOME".into(), "/workspace".into());
    let a = prod.register_image(img.clone()).expect("prod");
    let b = refer.register_image(img).expect("ref");
    assert_eq!(a, b);
}

#[test]
fn register_image_rejects_hidden_tests_outside_grader_root() {
    let (mut prod, mut refer) = pools();
    let mut img = sample_image("img-leak");
    img.hidden_tests
        .insert("/workspace/hidden.py".into(), b"assert 0\n".to_vec());
    assert_err_eq(
        prod.register_image(img.clone()),
        refer.register_image(img),
    );
    let (mut prod, _) = pools();
    let mut img = sample_image("img-leak");
    img.hidden_tests
        .insert("/workspace/hidden.py".into(), b"assert 0\n".to_vec());
    assert_eq!(prod.register_image(img), Err(Error::HiddenTestsVisible));
}

#[test]
fn register_image_rejects_agent_files_under_grader_root() {
    let (mut prod, mut refer) = pools();
    let mut img = sample_image("img-agent-grader");
    img.agent_files
        .insert(format!("{GRADER_ROOT}/owned.py"), b"x".to_vec());
    assert_err_eq(
        prod.register_image(img.clone()),
        refer.register_image(img),
    );
    let (mut prod, _) = pools();
    let mut img = sample_image("img-agent-grader");
    img.agent_files
        .insert(format!("{GRADER_ROOT}/owned.py"), b"x".to_vec());
    assert_eq!(prod.register_image(img), Err(Error::HiddenTestsVisible));
}

#[test]
fn hidden_tests_hash_is_sorted_path_nul_bytes() {
    let (mut prod, mut refer) = pools();
    let mut img = Image::new("img-hash");
    img.hidden_tests.insert("/grader/b".into(), b"y".to_vec());
    img.hidden_tests.insert("/grader/a".into(), b"x".to_vec());
    assert_eq!(
        hidden_tests_hash(&img.hidden_tests),
        "3f3d28c4619574a965fb8f0542be56d5c973ae3ae64993977a9820d8120cd037"
    );
    let a = prod.register_image(img.clone()).expect("prod hash image");
    let b = refer.register_image(img).expect("ref hash image");
    assert_eq!(a, b);
}
