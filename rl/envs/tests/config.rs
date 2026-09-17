//! Group: `PoolConfig::hold`, defaults, and crate constants (implemented on the stub).

use prometheus_envs::{
    EgressPolicy, PoolConfig, FORK_BUDGET_MS, GRADER_ROOT, GRPO_GROUP, SCHEMA_TOOL_REQUEST,
    SCHEMA_TOOL_RESPONSE, SCHEMA_VERSION,
};

#[test]
fn constants_match_spec() {
    assert_eq!(FORK_BUDGET_MS, 200);
    assert_eq!(GRPO_GROUP, 16);
    assert_eq!(GRADER_ROOT, "/grader");
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(SCHEMA_TOOL_REQUEST, "prometheus.env_tool_request");
    assert_eq!(SCHEMA_TOOL_RESPONSE, "prometheus.env_tool_response");
}

#[test]
fn pool_config_default_holds() {
    let c = PoolConfig::default();
    assert_eq!(c.capacity, GRPO_GROUP);
    assert_eq!(c.default_timeout_s, 30);
    assert_eq!(c.egress, EgressPolicy::DenyAll);
    assert!(c.gpu.is_none());
    assert!(c.offline_web.pages.is_empty());
    assert_eq!(c.offline_web.cutoff_unix_s, 0);
    assert!(c.hold());
}

#[test]
fn pool_config_hold_requires_capacity_gt_zero() {
    let zero = PoolConfig {
        capacity: 0,
        ..PoolConfig::default()
    };
    assert!(!zero.hold());
    let one = PoolConfig {
        capacity: 1,
        ..PoolConfig::default()
    };
    assert!(one.hold());
}

#[test]
fn pool_config_hold_requires_timeout_gt_zero() {
    let zero = PoolConfig {
        default_timeout_s: 0,
        ..PoolConfig::default()
    };
    assert!(!zero.hold());
    let one = PoolConfig {
        default_timeout_s: 1,
        ..PoolConfig::default()
    };
    assert!(one.hold());
}

#[test]
fn pool_config_hold_requires_deny_all() {
    let c = PoolConfig::default();
    assert_eq!(c.egress, EgressPolicy::DenyAll);
    assert!(c.hold());
}
