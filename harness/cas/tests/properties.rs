//! Group: property tests vs the in-memory reference (put/get/pin/unpin/gc).

mod common;
mod reference;

use common::{assert_duplicate_pin, assert_not_found, cfg, fresh_store, pin};
use prometheus_cas::{Digest, Store};
use reference::{digest_hex, RefStore};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push((self.next() >> 32) as u8);
        }
        out
    }
}

#[test]
fn random_ops_match_reference() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let mut refer = RefStore::with_default_replicas();
    let mut rng = Lcg(0xC0FFEE);
    let mut known: Vec<(Digest, Vec<u8>)> = Vec::new();
    let mut pin_names: Vec<String> = Vec::new();

    for step in 0..40u32 {
        let op = rng.next() % 5;
        match op {
            0 => {
                let n = (rng.next() % 65) as usize;
                let bytes = rng.bytes(n);
                let d = store.put(&bytes).expect("put");
                let rd = refer.put(&bytes).expect("ref put");
                assert_eq!(d, rd, "step {step} digest");
                assert_eq!(d.0, digest_hex(&bytes));
                assert_eq!(store.get(&d).expect("get"), bytes);
                assert_eq!(
                    store.replica_count(&d).expect("count"),
                    refer.replica_count(&d).expect("ref count")
                );
                known.push((d, bytes));
            }
            1 => {
                if known.is_empty() {
                    continue;
                }
                let idx = (rng.next() as usize) % known.len();
                let d = known[idx].0.clone();
                let name = format!("p{step}");
                match store.pin(&d, pin(&name)) {
                    Ok(()) => {
                        refer.pin(&d, pin(&name)).expect("ref pin");
                        pin_names.push(name);
                    }
                    Err(e) => {
                        let re = refer.pin(&d, pin(&name)).expect_err("ref pin err");
                        match (e, re) {
                            (
                                prometheus_cas::Error::NotFound(a),
                                prometheus_cas::Error::NotFound(b),
                            ) => assert_eq!(a, b),
                            (
                                prometheus_cas::Error::DuplicatePin(a),
                                prometheus_cas::Error::DuplicatePin(b),
                            ) => assert_eq!(a, b),
                            (a, b) => panic!("step {step} pin err mismatch {a:?} vs {b:?}"),
                        }
                    }
                }
            }
            2 => {
                if pin_names.is_empty() {
                    let err = store.unpin(&pin("nope")).expect_err("unpin missing");
                    assert_not_found(&err, "nope");
                    continue;
                }
                let idx = (rng.next() as usize) % pin_names.len();
                let name = pin_names.remove(idx);
                store.unpin(&pin(&name)).expect("unpin");
                refer.unpin(&pin(&name)).expect("ref unpin");
            }
            3 => {
                let report = store.gc().expect("gc");
                let rreport = refer.gc().expect("ref gc");
                assert_eq!(report, rreport, "step {step} gc report");
            }
            _ => {
                if known.is_empty() {
                    continue;
                }
                let idx = (rng.next() as usize) % known.len();
                let (d, bytes) = &known[idx];
                match store.get(d) {
                    Ok(got) => {
                        assert_eq!(got, refer.get(d).expect("ref get"));
                        assert_eq!(&got, bytes);
                    }
                    Err(_) => {
                        refer.get(d).expect_err("ref also missing after gc");
                    }
                }
            }
        }
    }

    for (d, bytes) in &known {
        match store.get(d) {
            Ok(got) => {
                assert_eq!(got, bytes.as_slice());
                assert_eq!(got, refer.get(d).expect("ref get"));
                assert_eq!(
                    store.replica_count(d).expect("count"),
                    refer.replica_count(d).expect("ref count")
                );
            }
            Err(e) => {
                assert_not_found(&e, &d.0);
                assert_not_found(&refer.get(d).expect_err("ref gone"), &d.0);
            }
        }
    }
}

#[test]
fn duplicate_pin_matches_reference() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let mut refer = RefStore::with_default_replicas();
    let d = store.put(b"x").expect("put");
    refer.put(b"x").expect("ref put");
    store.pin(&d, pin("n")).expect("pin");
    refer.pin(&d, pin("n")).expect("ref pin");
    let err = store.pin(&d, pin("n")).expect_err("dup");
    let rerr = refer.pin(&d, pin("n")).expect_err("ref dup");
    assert_duplicate_pin(&err, "n");
    assert_duplicate_pin(&rerr, "n");
}

#[test]
fn get_put_empty_and_nonzero_vs_reference() {
    let h = fresh_store();
    let mut store = Store::create(&h.dir, h.backends.clone(), cfg()).expect("create");
    let mut refer = RefStore::with_default_replicas();
    for bytes in [b"".as_slice(), b"\0".as_slice(), b"zz".as_slice()] {
        let d = store.put(bytes).expect("put");
        let rd = refer.put(bytes).expect("ref put");
        assert_eq!(d, rd);
        assert_eq!(store.get(&d).expect("get"), bytes);
        assert_eq!(refer.get(&d).expect("ref get"), bytes);
    }
}
