//! Group: property tests (hash order-independence, RCI bounds, all held-out
//! slugs, unicode items). Compared to the independent reference.

mod common;
mod reference;

use prometheus_eval_gate::{EvalGate, Subject};

use common::{assert_hash_shape, assert_scores_eq, cfg, fresh_root, HV};
use reference::Item;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
}

fn random_items(rng: &mut Lcg, n: usize) -> Vec<Item> {
    (0..n)
        .map(|i| Item {
            id: format!("id-{:04}", rng.pick(10_000)),
            prompt: format!("p{}-{}", i, rng.next()),
            answer: format!("a{}-{}", i, rng.next()),
            weight_milli: (rng.pick(500) as i64) + 1,
        })
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, mut it)| {
            it.id = format!("id-{i:04}");
            it
        })
        .collect()
}

#[test]
fn hash_independent_of_item_file_order() {
    let mut rng = Lcg(7);
    let items = random_items(&mut rng, 17);
    let h = reference::hash_items(&items);
    let mut reversed = items.clone();
    reversed.reverse();
    assert_eq!(h, reference::hash_items(&reversed));
    let (_tmp, root) = fresh_root();
    reference::install_current(&root, "re_bench", reversed, 1, 0, 1, 0).unwrap();
    let mut answers = std::collections::BTreeMap::new();
    for it in &items {
        answers.insert(it.id.clone(), it.answer.clone());
    }
    reference::install_subject(&root, &Subject::GenomeRev("g".into()), &answers).unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let scores = gate
        .request(Subject::GenomeRev("g".into()), "re_bench", 0)
        .unwrap();
    assert_eq!(scores.suite_hash, h);
    assert_eq!(scores.n_items, 17);
    assert_eq!(
        scores.rci_milli,
        items.iter().map(|i| i.weight_milli).sum::<i64>()
    );
}

#[test]
fn rci_is_sum_of_matching_weights_and_bounded() {
    let (_tmp, root) = fresh_root();
    let items = vec![
        Item {
            id: "x".into(),
            prompt: "px".into(),
            answer: "ax".into(),
            weight_milli: 100,
        },
        Item {
            id: "y".into(),
            prompt: "py".into(),
            answer: "ay".into(),
            weight_milli: 250,
        },
        Item {
            id: "z".into(),
            prompt: "pz".into(),
            answer: "az".into(),
            weight_milli: 3,
        },
    ];
    let total: i64 = items.iter().map(|i| i.weight_milli).sum();
    reference::install_current(&root, "re_bench", items.clone(), 1, 0, 42, 0).unwrap();
    let mut answers = std::collections::BTreeMap::new();
    answers.insert("x".into(), "ax".into());
    answers.insert("y".into(), "NO".into());
    // z missing
    reference::install_subject(&root, &Subject::GenomeRev("mix".into()), &answers).unwrap();
    let cfg = cfg(&root, HV);
    let gate = EvalGate::open(cfg.clone()).unwrap();
    let refer = reference::RefGate::open(cfg).unwrap();
    let subject = Subject::GenomeRev("mix".into());
    let got = gate.request(subject.clone(), "re_bench", 0).unwrap();
    let exp = refer.request(subject, "re_bench", 0).unwrap();
    assert_scores_eq(&got, &exp);
    assert_eq!(got.rci_milli, 100);
    assert!(got.rci_milli >= 0 && got.rci_milli <= total);
    assert_eq!(got.compute_ms, 42);
}

#[test]
fn unicode_item_hashes_and_does_not_leak() {
    let (_tmp, root) = fresh_root();
    let items = vec![Item {
        id: "ü-1".into(),
        prompt: "prompt-日本語-🔬".into(),
        answer: "ans-é".into(),
        weight_milli: 9,
    }];
    let h = reference::hash_items(&items);
    assert_hash_shape(&h);
    reference::install_current(&root, "re_bench", items, 1, 0, 1, 0).unwrap();
    let mut answers = std::collections::BTreeMap::new();
    answers.insert("ü-1".into(), "ans-é".into());
    reference::install_subject(&root, &Subject::Checkpoint("c".into()), &answers).unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let scores = gate
        .request(Subject::Checkpoint("c".into()), "re_bench", 0)
        .unwrap();
    assert_eq!(scores.rci_milli, 9);
    assert_eq!(scores.suite_hash, h);
    let json = serde_json::to_string(&scores).unwrap();
    assert!(!json.contains("日本語"));
    assert!(!json.contains("prompt-"));
    assert!(!json.contains("ans-é"));
}

#[test]
fn exact_match_is_byte_exact_not_trimmed() {
    let (_tmp, root) = fresh_root();
    let items = vec![Item {
        id: "t".into(),
        prompt: "p".into(),
        answer: "yes".into(),
        weight_milli: 5,
    }];
    reference::install_current(&root, "re_bench", items, 1, 0, 1, 0).unwrap();
    let mut answers = std::collections::BTreeMap::new();
    answers.insert("t".into(), "yes ".into());
    reference::install_subject(&root, &Subject::GenomeRev("sp".into()), &answers).unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    let scores = gate
        .request(Subject::GenomeRev("sp".into()), "re_bench", 0)
        .unwrap();
    assert_eq!(scores.rci_milli, 0);
}

#[test]
fn n_items_matches_generation_len() {
    let (_tmp, root) = fresh_root();
    for n in [1usize, 2, 8] {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path();
        let items: Vec<Item> = (0..n)
            .map(|i| Item {
                id: format!("{i}"),
                prompt: format!("p{i}"),
                answer: format!("a{i}"),
                weight_milli: 1,
            })
            .collect();
        reference::install_current(r, "re_bench", items, 1, 0, 1, 0).unwrap();
        reference::install_subject(r, &Subject::GenomeRev("z".into()), &Default::default())
            .unwrap();
        let gate = EvalGate::open(cfg(r, HV)).unwrap();
        let s = gate
            .request(Subject::GenomeRev("z".into()), "re_bench", 0)
            .unwrap();
        assert_eq!(s.n_items as usize, n);
    }
    let _ = root;
}

#[test]
fn golden_item_hash_via_gate() {
    let (_tmp, root) = fresh_root();
    reference::install_current(
        &root,
        "re_bench",
        vec![reference::golden_item()],
        1,
        0,
        0,
        0,
    )
    .unwrap();
    reference::install_subject(&root, &Subject::GenomeRev("g".into()), &Default::default())
        .unwrap();
    let gate = EvalGate::open(cfg(&root, HV)).unwrap();
    assert_eq!(
        gate.suite_hash("re_bench").unwrap(),
        reference::GOLDEN_A_HASH
    );
}
