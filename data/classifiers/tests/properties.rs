//! Group: property checks of production vs the in-memory reference.

mod common;
mod reference;

use common::{
    assert_close, assert_preds_eq, assert_shards_eq, config_full, doc, err_kind, fresh_orch_dir,
    label, ConstScorer,
};
use prometheus_classifiers::{agreement, LocalScorer, Orchestrator, Scorer, Scores};
use prometheus_extract::ExtractedDocument;
use reference::{ref_agreement, RefLocalScorer, RefOrchestrator};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
    fn f32_01(&mut self) -> f32 {
        (self.next() as f32) / (u64::MAX as f32)
    }
}

fn unique_doc(rng: &mut Lcg, i: u32) -> ExtractedDocument {
    doc(&format!("p{}-{}", i, rng.next()))
}

#[test]
fn random_ingest_shards_vs_ref() {
    let mut rng = Lcg(0xC1A551F1E5);
    for trial in 0..40 {
        let shard = rng.pick(8) + 1;
        let n = rng.pick(17) as u32;
        let (_parent, dir) = fresh_orch_dir();
        let cfg = config_full(shard, 0.5, 0.5, 1.0);
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        let mut refer = RefOrchestrator::new(cfg);
        for i in 0..n {
            let d = unique_doc(&mut rng, i);
            orch.ingest(d.clone()).expect("ingest");
            refer.ingest(d).expect("ref");
        }
        assert_shards_eq(
            &orch.shards(),
            &refer.shards(),
            &format!("trial {trial} shard_size={shard} n={n}"),
        );
    }
}

#[test]
fn random_score_vs_ref() {
    let mut rng = Lcg(0x5C0_5EED);
    for trial in 0..24 {
        let seed = rng.next();
        let n = rng.pick(12) + 1;
        let texts: Vec<String> = (0..n)
            .map(|i| {
                format!(
                    "t{i}-{}-{}",
                    rng.next(),
                    if rng.pick(4) == 0 { "π" } else { "x" }
                )
            })
            .collect();
        let got = LocalScorer { seed }
            .score(&texts, rng.next())
            .expect("prod");
        let exp = RefLocalScorer { seed }
            .score(&texts, rng.next())
            .expect("ref");
        assert_eq!(got.len(), exp.len(), "trial {trial} len");
        for (i, (g, e)) in got.iter().zip(exp.iter()).enumerate() {
            assert_close(g.quality, e.quality, &format!("t{trial} q[{i}]"));
            assert_close(g.safety, e.safety, &format!("t{trial} s[{i}]"));
        }
    }
}

#[test]
fn random_agreement_and_gate_vs_ref() {
    let mut rng = Lcg(0xA6EE);
    for trial in 0..40 {
        let n = rng.pick(8) + 1;
        let mut preds = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            let h = format!("h{i}-{}", rng.next());
            let q = rng.f32_01();
            let s = rng.f32_01();
            preds.push(prometheus_classifiers::Prediction {
                content_hash: h.clone(),
                scores: Scores {
                    quality: q,
                    safety: s,
                },
            });
            if rng.pick(3) != 0 {
                labels.push(label(&h, rng.pick(2) == 0, rng.pick(2) == 0));
            }
        }
        let q_th = 0.5;
        let s_th = 0.5;
        let pg = agreement(&preds, &labels, q_th, s_th);
        let rg = ref_agreement(&preds, &labels, q_th, s_th);
        match (pg, rg) {
            (Ok(a), Ok(b)) => assert_close(a, b, &format!("trial {trial} agree")),
            (Err(e), Err(f)) => assert_eq!(err_kind(&e), err_kind(&f), "trial {trial} err"),
            (a, b) => panic!("trial {trial} mismatch prod={a:?} ref={b:?}"),
        }
    }
}

#[test]
fn mixed_ops_vs_ref() {
    let mut rng = Lcg(0xB3B3);
    for trial in 0..20 {
        let shard = rng.pick(5) + 1;
        let min_ag = if rng.pick(2) == 0 { 1.0 } else { 0.5 };
        let cfg = config_full(shard, 0.5, 0.5, min_ag);
        let (_parent, dir) = fresh_orch_dir();
        let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
        let mut refer = RefOrchestrator::new(cfg.clone());
        let n = rng.pick(10) + 1;
        let mut docs = Vec::new();
        for i in 0..n as u32 {
            let d = unique_doc(&mut rng, i);
            docs.push(d.clone());
            orch.ingest(d.clone()).expect("ingest");
            refer.ingest(d).expect("ref");
        }
        // duplicate of first must fail both sides
        let dup = doc(&format!("dup-{}", rng.next()));
        let mut dup = dup;
        dup.content_hash = docs[0].content_hash.clone();
        let pe = orch.ingest(dup.clone()).expect_err("prod dup");
        let re = refer.ingest(dup).expect_err("ref dup");
        assert_eq!(err_kind(&pe), err_kind(&re), "trial {trial} dup");

        assert_shards_eq(
            &orch.shards(),
            &refer.shards(),
            &format!("trial {trial} shards"),
        );

        let seed = rng.next();
        orch.run(&mut LocalScorer { seed }, 100 + trial as u64)
            .expect("run");
        refer
            .run(&mut RefLocalScorer { seed }, 100 + trial as u64)
            .expect("ref run");
        assert_preds_eq(
            orch.predictions(),
            refer.predictions(),
            &format!("trial {trial} preds"),
        );

        let mut labels = Vec::new();
        for d in &docs {
            if rng.pick(2) == 0 {
                continue;
            }
            let p = refer
                .predictions()
                .iter()
                .find(|p| p.content_hash == d.content_hash)
                .expect("pred");
            let kq = p.scores.quality >= 0.5;
            let ks = p.scores.safety >= 0.5;
            let flip = rng.pick(5) == 0;
            labels.push(label(
                &d.content_hash,
                if flip { !kq } else { kq },
                if flip { !ks } else { ks },
            ));
        }
        let pa = orch.agreement(&labels);
        let ra = refer.agreement(&labels);
        match (&pa, &ra) {
            (Ok(a), Ok(b)) => assert_close(*a, *b, &format!("trial {trial} agree")),
            (Err(e), Err(f)) => assert_eq!(err_kind(e), err_kind(f), "trial {trial} agree err"),
            _ => panic!("trial {trial} agree prod={pa:?} ref={ra:?}"),
        }
        let pg = orch.gate(&labels);
        let rg = refer.gate(&labels);
        match (&pg, &rg) {
            (Ok(()), Ok(())) => {}
            (Err(e), Err(f)) => {
                assert_eq!(err_kind(e), err_kind(f), "trial {trial} gate err");
                if let (
                    prometheus_classifiers::Error::GateFailed { got: ga, min: ma },
                    prometheus_classifiers::Error::GateFailed { got: gb, min: mb },
                ) = (e, f)
                {
                    assert_close(*ga, *gb, &format!("trial {trial} gate got"));
                    assert_close(*ma, *mb, &format!("trial {trial} gate min"));
                }
            }
            _ => panic!("trial {trial} gate prod={pg:?} ref={rg:?}"),
        }
    }
}

#[test]
fn const_scorer_run_vs_ref() {
    let (_parent, dir) = fresh_orch_dir();
    let cfg = config_full(2, 0.4, 0.6, 1.0);
    let mut orch = Orchestrator::create(&dir, cfg.clone()).expect("create");
    let mut refer = RefOrchestrator::new(cfg);
    for i in 0..5 {
        let d = doc(&format!("c{i}"));
        orch.ingest(d.clone()).expect("ingest");
        refer.ingest(d).expect("ref");
    }
    orch.run(
        &mut ConstScorer {
            quality: 0.5,
            safety: 0.5,
        },
        0,
    )
    .expect("run");
    refer
        .run(
            &mut ConstScorer {
                quality: 0.5,
                safety: 0.5,
            },
            0,
        )
        .expect("ref");
    assert_preds_eq(orch.predictions(), refer.predictions(), "const");
}
