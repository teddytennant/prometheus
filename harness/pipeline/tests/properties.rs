//! Group: property tests vs the in-memory reference (same call sequences).

mod common;
mod reference;

use common::{
    assert_same_error, assert_state, cand, default_pipeline_config, default_queue_config,
    good_patch, merge_record, module, pick, planted_patch, reject_all, NOW,
};
use prometheus_pipeline::{is_planted_bad, merge_allowed, review, work_payload, Candidate, Pipeline};
use reference::{ref_is_planted_bad, ref_merge_allowed, ref_review, ref_work_payload, RefPipeline};
use serde_json::json;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
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
fn random_patches_match_is_planted_bad() {
    let mut rng = Lcg(0xC0FFEE);
    for i in 0..80 {
        let n = (rng.next() % 40) as usize;
        let mut bytes = rng.bytes(n);
        if i % 7 == 0 {
            let at = rng.pick(bytes.len() + 1);
            bytes.splice(at..at, prometheus_pipeline::PLANTED_BAD_MARKER.iter().copied());
        }
        if i % 11 == 0 && !bytes.is_empty() {
            bytes.truncate(rng.pick(bytes.len()));
        }
        assert_eq!(
            is_planted_bad(&bytes),
            ref_is_planted_bad(&bytes),
            "patch {i}"
        );
    }
}

#[test]
fn random_review_and_merge_allowed_match_reference() {
    let mut rng = Lcg(0xBAD5EED);
    for step in 0..40u32 {
        let n = rng.pick(5);
        let mut cands = Vec::with_capacity(n);
        let mut tests = Vec::with_capacity(n);
        for i in 0..n {
            let planted = rng.pick(3) == 0;
            let patch = if planted {
                planted_patch()
            } else {
                let n = rng.pick(16);
                rng.bytes(n)
            };
            cands.push(cand(&i.to_string(), &format!("a{i}"), &patch));
            tests.push(rng.pick(2) == 0);
        }
        let pr = review(&cands, &tests);
        let rr = ref_review(&cands, &tests);
        match (pr, rr) {
            (Ok(prometheus_pipeline::Verdict::Pick(a)), Ok(prometheus_pipeline::Verdict::Pick(b))) => {
                assert_eq!(a, b, "step {step} pick")
            }
            (
                Ok(prometheus_pipeline::Verdict::RejectAll { defects: d1 }),
                Ok(prometheus_pipeline::Verdict::RejectAll { defects: d2 }),
            ) => {
                assert!(!d1.is_empty() && !d2.is_empty(), "step {step} defects");
            }
            (Err(prometheus_pipeline::Error::Other(_)), Err(prometheus_pipeline::Error::Other(_))) => {}
            (a, b) => panic!("step {step} review {a:?} vs {b:?}"),
        }
        let verdict = match rng.pick(3) {
            0 if !cands.is_empty() => pick(&cands[rng.pick(cands.len())].id.0),
            1 => pick("missing"),
            _ => reject_all(&["prop"]),
        };
        let pm = merge_allowed(&verdict, &cands, &tests);
        let rm = ref_merge_allowed(&verdict, &cands, &tests);
        match (pm, rm) {
            (Ok(()), Ok(())) => {}
            (Err(a), Err(b)) => assert_same_error(a, b, &format!("step {step} merge")),
            (a, b) => panic!("step {step} merge_allowed {a:?} vs {b:?}"),
        }
    }
}

#[test]
fn random_work_payload_matches_reference() {
    let mut rng = Lcg(42);
    let m = module();
    let roles = ["oracle", "implementer", "reviewer", "pipeline"];
    for step in 0..30u32 {
        let role = roles[rng.pick(roles.len())];
        let extra = match rng.pick(4) {
            0 => json!({ "round": rng.pick(5) as u64 }),
            1 => json!({ "round": rng.pick(4) as u64, "candidate": rng.pick(3).to_string() }),
            2 => json!({ "round": 1, "op": "submit_candidate", "candidate": "0" }),
            _ => json!(rng.pick(9) as u64),
        };
        let prod = work_payload(&m, role, extra.clone());
        let refer = ref_work_payload(&m, role, extra);
        assert_eq!(prod.task_id.0, refer.task_id.0, "step {step} id");
        assert_eq!(prod.payload, refer.payload, "step {step} payload");
    }
}

fn apply_op(
    p: &mut Pipeline,
    r: &mut RefPipeline,
    op: u32,
    rng: &mut Lcg,
    step: u32,
) {
    let ctx = format!("step {step} op {op}");
    match op {
        0 => match (p.start_oracle(NOW), r.start_oracle(NOW)) {
            (Ok(a), Ok(b)) => assert_eq!(a.0, b.0, "{ctx} oracle id"),
            (Err(a), Err(b)) => assert_same_error(a, b, &ctx),
            (a, b) => panic!("{ctx} start_oracle {a:?} vs {b:?}"),
        },
        1 => {
            let passed = rng.pick(4) == 0;
            match (p.record_oracle(passed, NOW), r.record_oracle(passed, NOW)) {
                (Ok(()), Ok(())) => {}
                (Err(a), Err(b)) => assert_same_error(a, b, &ctx),
                (a, b) => panic!("{ctx} record_oracle {a:?} vs {b:?}"),
            }
        }
        2 => match (p.start_implementers(NOW), r.start_implementers(NOW)) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "{ctx} impl ids"),
            (Err(a), Err(b)) => assert_same_error(a, b, &ctx),
            (a, b) => panic!("{ctx} start_implementers {a:?} vs {b:?}"),
        },
        3 => {
            let id = rng.pick(5).to_string();
            let patch = if rng.pick(4) == 0 {
                planted_patch()
            } else {
                good_patch().to_vec()
            };
            let c = cand(&id, "prop", &patch);
            match (
                p.submit_candidate(c.clone(), NOW),
                r.submit_candidate(c, NOW),
            ) {
                (Ok(()), Ok(())) => {}
                (Err(a), Err(b)) => assert_same_error(a, b, &ctx),
                (a, b) => panic!("{ctx} submit {a:?} vs {b:?}"),
            }
        }
        4 => match (p.start_review(NOW), r.start_review(NOW)) {
            (Ok(a), Ok(b)) => assert_eq!(a.0, b.0, "{ctx} review id"),
            (Err(a), Err(b)) => assert_same_error(a, b, &ctx),
            (a, b) => panic!("{ctx} start_review {a:?} vs {b:?}"),
        },
        5 => {
            let v = if rng.pick(2) == 0 {
                pick(&rng.pick(4).to_string())
            } else {
                reject_all(&["prop"])
            };
            match (p.record_verdict(v.clone(), NOW), r.record_verdict(v, NOW)) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "{ctx} verdict stage"),
                (Err(a), Err(b)) => assert_same_error(a, b, &ctx),
                (a, b) => panic!("{ctx} record_verdict {a:?} vs {b:?}"),
            }
        }
        _ => {
            let rec = merge_record(p.module(), rng.pick(4) as u32, rng.pick(3) as u32);
            match (p.record_merge(rec.clone(), NOW), r.record_merge(rec, NOW)) {
                (Ok(()), Ok(())) => {}
                (Err(a), Err(b)) => assert_same_error(a, b, &ctx),
                (a, b) => panic!("{ctx} record_merge {a:?} vs {b:?}"),
            }
        }
    }
}

#[test]
fn random_ops_match_reference() {
    let parent = tempfile::tempdir().expect("tempdir");
    let mut p = Pipeline::create(
        parent.path().join("prod"),
        module(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("prod");
    let mut r = RefPipeline::create(
        parent.path().join("refer"),
        module(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("ref");
    let mut rng = Lcg(0x5EEDF00D);
    for step in 0..50u32 {
        let op = rng.pick(7) as u32;
        apply_op(&mut p, &mut r, op, &mut rng, step);
        assert_state(&p, &r, &format!("after step {step}"));
    }
}

#[test]
fn happy_path_sequence_matches_reference() {
    let parent = tempfile::tempdir().expect("tempdir");
    let mut p = Pipeline::create(
        parent.path().join("prod"),
        module(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("prod");
    let mut r = RefPipeline::create(
        parent.path().join("refer"),
        module(),
        default_pipeline_config(),
        default_queue_config(),
    )
    .expect("ref");
    assert_eq!(p.start_oracle(NOW).unwrap(), r.start_oracle(NOW).unwrap());
    p.record_oracle(false, NOW).unwrap();
    r.record_oracle(false, NOW).unwrap();
    assert_eq!(
        p.start_implementers(NOW).unwrap(),
        r.start_implementers(NOW).unwrap()
    );
    for i in 0..3 {
        let c = Candidate {
            id: prometheus_pipeline::CandidateId(i.to_string()),
            angle: format!("a{i}"),
            patch: good_patch().to_vec(),
        };
        p.submit_candidate(c.clone(), NOW).unwrap();
        r.submit_candidate(c, NOW).unwrap();
    }
    assert_eq!(p.start_review(NOW).unwrap(), r.start_review(NOW).unwrap());
    assert_eq!(
        p.record_verdict(pick("2"), NOW).unwrap(),
        r.record_verdict(pick("2"), NOW).unwrap()
    );
    p.record_merge(merge_record(p.module(), 3, 0), NOW).unwrap();
    r.record_merge(merge_record(r.module(), 3, 0), NOW).unwrap();
    assert_state(&p, &r, "happy path");
    assert_eq!(p.stage(), prometheus_pipeline::Stage::Merged);
}
