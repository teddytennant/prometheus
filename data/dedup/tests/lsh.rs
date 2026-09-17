//! Group: lsh_band_keys — band keys and config/signature errors.

mod common;
mod reference;

use prometheus_dedup::{lsh_band_keys, minhash_signature, DedupConfig};

#[test]
fn standard_yields_32_keys() {
    let cfg = DedupConfig::standard();
    assert_eq!(cfg.bands, 32);
    assert_eq!(cfg.rows, 4);
    assert_eq!(cfg.num_hashes, cfg.bands * cfg.rows);
    let sig = minhash_signature(common::alphabet(), cfg);
    let keys = lsh_band_keys(&sig, cfg).expect("lsh");
    assert_eq!(keys.len(), 32);
    assert_eq!(keys, reference::lsh_band_keys(&sig, cfg).expect("ref lsh"));
}

#[test]
fn small_config_yields_bands_keys() {
    let cfg = common::cfg_small();
    let sig = minhash_signature("alpha bravo charlie delta echo", cfg);
    let keys = lsh_band_keys(&sig, cfg).expect("lsh");
    assert_eq!(keys.len(), cfg.bands);
}

#[test]
fn identical_signatures_identical_keys() {
    let cfg = DedupConfig::standard();
    let sig = minhash_signature(common::alphabet(), cfg);
    assert_eq!(
        lsh_band_keys(&sig, cfg).unwrap(),
        lsh_band_keys(&sig, cfg).unwrap()
    );
}

#[test]
fn error_if_num_hashes_ne_bands_times_rows() {
    let cfg = DedupConfig {
        shingle_size: 5,
        num_hashes: 10,
        bands: 32,
        rows: 4,
    };
    let sig = vec![1u64; 10];
    common::assert_err_config(lsh_band_keys(&sig, cfg));
}

#[test]
fn error_if_signature_length_ne_num_hashes() {
    let cfg = common::cfg_small();
    let sig = vec![1u64; 7];
    common::assert_err_config(lsh_band_keys(&sig, cfg));
}

#[test]
fn error_if_signature_len_matches_bands_rows_but_not_num_hashes() {
    let cfg = DedupConfig {
        shingle_size: 2,
        num_hashes: 99,
        bands: 4,
        rows: 2,
    };
    let sig = vec![3u64; 8];
    common::assert_err_config(lsh_band_keys(&sig, cfg));
}

#[test]
fn error_if_bands_or_rows_zero() {
    let sig = vec![1u64; 8];
    common::assert_err_config(lsh_band_keys(
        &sig,
        DedupConfig {
            shingle_size: 2,
            num_hashes: 0,
            bands: 0,
            rows: 2,
        },
    ));
    common::assert_err_config(lsh_band_keys(
        &sig,
        DedupConfig {
            shingle_size: 2,
            num_hashes: 0,
            bands: 4,
            rows: 0,
        },
    ));
}

#[test]
fn error_if_shingle_size_zero() {
    let cfg = DedupConfig {
        shingle_size: 0,
        num_hashes: 8,
        bands: 4,
        rows: 2,
    };
    common::assert_err_config(lsh_band_keys(&[1u64; 8], cfg));
}

#[test]
fn overflow_bands_times_rows_is_config_error() {
    let cfg = DedupConfig {
        shingle_size: 5,
        num_hashes: 8,
        bands: usize::MAX,
        rows: 2,
    };
    common::assert_err_config(lsh_band_keys(&[1u64; 8], cfg));
}

#[test]
fn matches_reference_keys_on_known_signature() {
    let cfg = common::cfg_small();
    let sig: Vec<u64> = (0..8).map(|i| i * 1_000_003).collect();
    let got = lsh_band_keys(&sig, cfg).expect("lsh");
    let exp = reference::lsh_band_keys(&sig, cfg).expect("ref");
    assert_eq!(got, exp);
    assert_eq!(got.len(), 4);
}
