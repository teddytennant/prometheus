//! Oracle tests for `parse_busbw_gbps`.
//!
//! `ref_*` use the independent reference and must pass on the stub crate.
//! `prod_*` call `prometheus_verify_ncshare::parse_busbw_gbps` only and must
//! panic `unimplemented!` until F4 is implemented. Do not `#[should_panic]`.

mod reference;

use prometheus_verify_ncshare::parse_busbw_gbps;
use reference::{
    assert_close, CROSS_OOP_BUSBW, INTRA_AVG_DECOY, INTRA_INPLACE_DECOY, INTRA_OOP_BUSBW,
    NCCL_2NODE, NCCL_HEADER_ONLY, NCCL_INTRA, NCCL_TOO_FEW_COLUMNS,
};

fn mini_table(last_oop_busbw: f64, last_inplace_busbw: f64) -> String {
    format!(
        "\
#                                                       out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
           8             2     float     sum      -1    10.00    0.00    0.01      0    10.00    0.00    0.02      0
        1024           256     float     sum      -1    20.00    0.05    0.10      0    19.00    0.05    0.11      0
   134217728      33554432     float     sum      -1  1000.00   10.00 {last_oop_busbw:>8.4}      0   900.00   11.00 {last_inplace_busbw:>8.4}      0
# Avg bus bandwidth    : 123.456789
"
    )
}

// ---------------------------------------------------------------------------
// Reference
// ---------------------------------------------------------------------------

#[test]
fn ref_parse_busbw_gbps_intra_golden() {
    let got = reference::parse_busbw_gbps(NCCL_INTRA).expect("intra golden");
    assert_close(got, INTRA_OOP_BUSBW);
}

#[test]
fn ref_parse_busbw_gbps_2node_golden() {
    let got = reference::parse_busbw_gbps(NCCL_2NODE).expect("2node golden");
    assert_close(got, CROSS_OOP_BUSBW);
}

#[test]
fn ref_parse_busbw_gbps_goldens_differ() {
    let a = reference::parse_busbw_gbps(NCCL_INTRA).unwrap();
    let b = reference::parse_busbw_gbps(NCCL_2NODE).unwrap();
    assert!(
        (a - b).abs() > 1.0,
        "goldens must disagree on busbw: {a} vs {b}"
    );
}

#[test]
fn ref_parse_busbw_gbps_skips_inplace_and_avg() {
    let got = reference::parse_busbw_gbps(NCCL_INTRA).unwrap();
    assert!((got - INTRA_INPLACE_DECOY).abs() > 1.0);
    assert!((got - INTRA_AVG_DECOY).abs() > 1.0);
}

#[test]
fn ref_parse_busbw_gbps_empty_errors() {
    assert!(reference::parse_busbw_gbps("").is_err());
    assert!(reference::parse_busbw_gbps("   \n\t  ").is_err());
}

#[test]
fn ref_parse_busbw_gbps_malformed_errors() {
    assert!(reference::parse_busbw_gbps("hello world").is_err());
    assert!(reference::parse_busbw_gbps(NCCL_HEADER_ONLY).is_err());
    assert!(reference::parse_busbw_gbps(NCCL_TOO_FEW_COLUMNS).is_err());
}

#[test]
fn ref_parse_busbw_gbps_last_row_not_max_or_first() {
    let table = mini_table(3.5, 99.0);
    let got = reference::parse_busbw_gbps(&table).unwrap();
    assert_close(got, 3.5);
}

#[test]
fn ref_parse_busbw_gbps_property_last_oop() {
    for bw in [0.0_f64, 1.0, 10.5, 52.20, 412.35, 1234.5678] {
        let table = mini_table(bw, bw + 50.0);
        let got = reference::parse_busbw_gbps(&table).expect("property table");
        assert_close(got, bw);
    }
}

#[test]
fn ref_parse_busbw_gbps_crlf() {
    let crlf = NCCL_INTRA.replace('\n', "\r\n");
    let got = reference::parse_busbw_gbps(&crlf).unwrap();
    assert_close(got, INTRA_OOP_BUSBW);
}

// ---------------------------------------------------------------------------
// Production public API — must fail on unimplemented! stub
// ---------------------------------------------------------------------------

#[test]
fn prod_parse_busbw_gbps_intra_golden() {
    let got = parse_busbw_gbps(NCCL_INTRA).expect("intra golden");
    assert_close(got, INTRA_OOP_BUSBW);
    let refer = reference::parse_busbw_gbps(NCCL_INTRA).unwrap();
    assert_close(got, refer);
}

#[test]
fn prod_parse_busbw_gbps_2node_golden() {
    let got = parse_busbw_gbps(NCCL_2NODE).expect("2node golden");
    assert_close(got, CROSS_OOP_BUSBW);
    assert!((got - INTRA_OOP_BUSBW).abs() > 1.0);
}

#[test]
fn prod_parse_busbw_gbps_skips_inplace_and_avg() {
    let got = parse_busbw_gbps(NCCL_INTRA).unwrap();
    assert!((got - INTRA_INPLACE_DECOY).abs() > 1.0);
    assert!((got - INTRA_AVG_DECOY).abs() > 1.0);
}

#[test]
fn prod_parse_busbw_gbps_empty_errors() {
    assert!(parse_busbw_gbps("").is_err());
    assert!(parse_busbw_gbps("   \n\t  ").is_err());
}

#[test]
fn prod_parse_busbw_gbps_malformed_errors() {
    assert!(parse_busbw_gbps("hello world").is_err());
    assert!(parse_busbw_gbps(NCCL_HEADER_ONLY).is_err());
    assert!(parse_busbw_gbps(NCCL_TOO_FEW_COLUMNS).is_err());
}

#[test]
fn prod_parse_busbw_gbps_property_last_oop() {
    for bw in [0.0_f64, 1.0, 10.5, 52.20, 412.35, 1234.5678] {
        let table = mini_table(bw, bw + 50.0);
        let got = parse_busbw_gbps(&table).expect("property table");
        assert_close(got, bw);
    }
}

#[test]
fn prod_parse_busbw_gbps_matches_reference_on_goldens() {
    for stdout in [NCCL_INTRA, NCCL_2NODE] {
        let prod = parse_busbw_gbps(stdout).expect("prod");
        let refer = reference::parse_busbw_gbps(stdout).expect("ref");
        assert_close(prod, refer);
    }
}
