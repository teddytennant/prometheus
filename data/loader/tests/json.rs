//! Group: Packing / LoaderState JSON helpers vs F1 schemas and default goldens.

use prometheus_contracts::validate;
use prometheus_loader::{
    LoaderState, Packing, Strategy, SCHEMA_ID_LOADER_STATE, SCHEMA_ID_PACKING, SCHEMA_VERSION,
};
use serde_json::{json, Value};

const PACKING_DEFAULT: &str = include_str!("../../../contracts/goldens/v1/packing.default.json");
const LOADER_STATE_DEFAULT: &str =
    include_str!("../../../contracts/goldens/v1/loader_state.default.json");

fn packing_fixture() -> Packing {
    Packing {
        packing_id: "pretrain-8k-concat".to_string(),
        sequence_length: 8192,
        strategy: Strategy::ConcatEos,
        eos_between_docs: true,
        pad_id: 2,
        drop_short_below: Some(64),
    }
}

fn assert_valid(value: &Value) {
    validate(value).unwrap_or_else(|e| panic!("F1 schema rejected payload: {e}; value={value}"));
}

#[test]
fn packing_to_json_has_envelope_and_validates() {
    let value = packing_fixture().to_json().expect("Packing::to_json");
    assert_eq!(value["schema_id"], SCHEMA_ID_PACKING);
    assert_eq!(value["schema_version"], SCHEMA_VERSION);
    assert_eq!(value["packing_id"], "pretrain-8k-concat");
    assert_eq!(value["sequence_length"], 8192);
    assert_eq!(value["strategy"], "concat_eos");
    assert_eq!(value["eos_between_docs"], true);
    assert_eq!(value["pad_id"], 2);
    assert_eq!(value["drop_short_below"], 64);
    assert_valid(&value);
}

#[test]
fn packing_to_json_omits_drop_short_below_when_none() {
    let packing = Packing {
        packing_id: "no-drop".to_string(),
        sequence_length: 128,
        strategy: Strategy::DocumentMask,
        eos_between_docs: false,
        pad_id: 0,
        drop_short_below: None,
    };
    let value = packing.to_json().expect("to_json");
    assert!(value.get("drop_short_below").is_none());
    assert_eq!(value["strategy"], "document_mask");
    assert_valid(&value);
}

#[test]
fn packing_from_json_round_trips_all_strategies() {
    for strategy in [
        Strategy::ConcatEos,
        Strategy::DocumentMask,
        Strategy::SingleDocument,
    ] {
        let packing = Packing {
            packing_id: "rt".to_string(),
            sequence_length: 32,
            strategy,
            eos_between_docs: true,
            pad_id: 3,
            drop_short_below: Some(1),
        };
        let value = packing.to_json().expect("to_json");
        assert_valid(&value);
        let back = Packing::from_json(&value).expect("from_json");
        assert_eq!(back, packing);
        assert_eq!(back.strategy.as_str(), strategy.as_str());
    }
}

#[test]
fn packing_default_golden_round_trip() {
    let raw: Value = serde_json::from_str(PACKING_DEFAULT).expect("parse packing golden");
    assert_valid(&raw);
    let packing = Packing::from_json(&raw).expect("from_json packing.default.json");
    assert_eq!(packing.packing_id, "pretrain-8k-concat");
    assert_eq!(packing.sequence_length, 8192);
    assert_eq!(packing.strategy, Strategy::ConcatEos);
    assert!(packing.eos_between_docs);
    assert_eq!(packing.pad_id, 2);
    assert_eq!(packing.drop_short_below, Some(64));
    let back = packing.to_json().expect("to_json");
    assert_valid(&back);
    for key in [
        "schema_id",
        "schema_version",
        "packing_id",
        "sequence_length",
        "strategy",
        "eos_between_docs",
        "pad_id",
        "drop_short_below",
    ] {
        assert_eq!(back[key], raw[key], "packing golden field {key}");
    }
}

#[test]
fn packing_from_json_unknown_strategy() {
    let value = json!({
        "schema_id": SCHEMA_ID_PACKING,
        "schema_version": SCHEMA_VERSION,
        "packing_id": "x",
        "sequence_length": 8,
        "strategy": "not_a_strategy",
        "eos_between_docs": false,
        "pad_id": 0
    });
    match Packing::from_json(&value) {
        Err(prometheus_loader::Error::UnknownStrategy(name)) => {
            assert!(name.contains("not_a_strategy"), "got {name}");
        }
        Ok(v) => panic!("expected UnknownStrategy, got Ok({v:?})"),
        Err(other) => panic!("expected UnknownStrategy, got Err({other:?})"),
    }
}

#[test]
fn packing_from_json_rejects_wrong_schema_id() {
    let value = json!({
        "schema_id": "prometheus.nope",
        "schema_version": SCHEMA_VERSION,
        "packing_id": "x",
        "sequence_length": 8,
        "strategy": "concat_eos",
        "eos_between_docs": false,
        "pad_id": 0
    });
    assert!(
        Packing::from_json(&value).is_err(),
        "wrong schema_id must not parse"
    );
}

#[test]
fn packing_from_json_rejects_null() {
    assert!(Packing::from_json(&Value::Null).is_err());
}

#[test]
fn loader_state_to_json_has_envelope_and_validates() {
    let state = LoaderState {
        epoch: 0,
        step: 1024,
        shuffle_seed: 7,
        shard_index: 3,
        shard_offset: 4096,
        consumed_token_count: 8_388_608,
        data_mix_hash: "1f0368a7fbed2400c8f3b14febfddf1631b879b336c784b08ad2619dbdb5ab59"
            .to_string(),
        packing_remainder_hash: "ac760c60d7980c43c3a2c6f6651d125d759c1ecb910a01141e758550414bdc73"
            .to_string(),
    };
    let value = state.to_json().expect("LoaderState::to_json");
    assert_eq!(value["schema_id"], SCHEMA_ID_LOADER_STATE);
    assert_eq!(value["schema_version"], SCHEMA_VERSION);
    assert_eq!(value["epoch"], 0);
    assert_eq!(value["step"], 1024);
    assert_eq!(value["shuffle_seed"], 7);
    assert_eq!(value["shard_index"], 3);
    assert_eq!(value["shard_offset"], 4096);
    assert_eq!(value["consumed_token_count"], 8_388_608);
    assert_valid(&value);
}

#[test]
fn loader_state_default_golden_round_trip() {
    let raw: Value = serde_json::from_str(LOADER_STATE_DEFAULT).expect("parse loader_state golden");
    assert_valid(&raw);
    let state = LoaderState::from_json(&raw).expect("from_json loader_state.default.json");
    assert_eq!(state.epoch, 0);
    assert_eq!(state.step, 1024);
    assert_eq!(state.shuffle_seed, 7);
    assert_eq!(state.shard_index, 3);
    assert_eq!(state.shard_offset, 4096);
    assert_eq!(state.consumed_token_count, 8_388_608);
    assert_eq!(
        state.data_mix_hash,
        "1f0368a7fbed2400c8f3b14febfddf1631b879b336c784b08ad2619dbdb5ab59"
    );
    assert_eq!(
        state.packing_remainder_hash,
        "ac760c60d7980c43c3a2c6f6651d125d759c1ecb910a01141e758550414bdc73"
    );
    let back = state.to_json().expect("to_json");
    assert_valid(&back);
    for key in [
        "schema_id",
        "schema_version",
        "epoch",
        "step",
        "shuffle_seed",
        "shard_index",
        "shard_offset",
        "consumed_token_count",
        "data_mix_hash",
        "packing_remainder_hash",
    ] {
        assert_eq!(back[key], raw[key], "loader_state golden field {key}");
    }
}

#[test]
fn loader_state_from_json_rejects_wrong_schema_id() {
    let mut raw: Value =
        serde_json::from_str(LOADER_STATE_DEFAULT).expect("parse loader_state golden");
    raw["schema_id"] = json!("prometheus.nope");
    assert!(LoaderState::from_json(&raw).is_err());
}

#[test]
fn loader_state_from_json_rejects_null() {
    assert!(LoaderState::from_json(&Value::Null).is_err());
}
