//! E3 procedural ARC oracle tests (spec 7.1, 10, 15.5).
//!
//! Every call into a stubbed E3 function hits `unimplemented!` today, so
//! `cargo test -p prometheus-synth --offline --test arc` must be red. After E3,
//! production must match `common::arc_ref`.

mod common;

use prometheus_synth::{
    apply_color_perm, apply_dihedral, augment_pair, augment_task, diversity_stats,
    family_from_spec, sample_family_specs, tokenize_grid, Dihedral, DiversityStats, Error,
    FamilyKind, FamilySpec, Grid, GridTokenizer, Pair, ProceduralCorpus, Task, DEFAULT_N_TRAIN,
    MAX_GRID_SIZE, MIN_GRID_SIZE, N_COLORS, N_DIHEDRAL,
};

use common::arc_ref;

const GOLDEN_DIVERSITY: DiversityStats = DiversityStats {
    n_tasks: 2,
    n_families: 2,
    unique_test_inputs: 1,
    unique_test_outputs: 1,
    unique_tasks: 1,
    color_histogram: [12, 4, 4, 0, 0, 0, 0, 0, 0, 0],
    unique_shapes: 2,
    collision_rate: 0.5,
};

const IDENTITY_PERM: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const SWAP01: [u8; 10] = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9];
const CYCLE: [u8; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];

fn grid(rows: &[&[u8]]) -> Grid {
    Grid::try_new(rows.iter().map(|r| r.to_vec()).collect()).unwrap()
}

fn spec(kind: FamilyKind, pairs: &[(&str, i32)]) -> FamilySpec {
    FamilySpec {
        kind,
        params: pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
    }
}

fn kind_specs() -> Vec<FamilySpec> {
    vec![
        spec(FamilyKind::Translate, &[("dx", 1), ("dy", -1), ("bg", 0)]),
        spec(FamilyKind::Recolor, &[("src", 1), ("dst", 2)]),
        spec(FamilyKind::Crop, &[("bg", 0)]),
        spec(FamilyKind::Tile, &[("nx", 2), ("ny", 2)]),
        spec(FamilyKind::Gravity, &[("dir", 0), ("bg", 0)]),
        spec(FamilyKind::Mirror, &[("dihedral", 1)]),
        spec(FamilyKind::Scale, &[("factor", 2)]),
        spec(FamilyKind::Border, &[("color", 9), ("width", 1)]),
    ]
}

fn golden_tasks() -> Vec<Task> {
    vec![
        Task {
            family_id: "translate:bg=0,dx=1,dy=0".into(),
            seed: 7,
            train: vec![Pair {
                input: grid(&[&[1, 0], &[0, 0]]),
                output: grid(&[&[0, 1], &[0, 0]]),
            }],
            test: Pair {
                input: grid(&[&[2]]),
                output: grid(&[&[2]]),
            },
        },
        Task {
            family_id: "mirror:dihedral=0".into(),
            seed: 8,
            train: vec![Pair {
                input: grid(&[&[1, 0], &[0, 0]]),
                output: grid(&[&[0, 1], &[0, 0]]),
            }],
            test: Pair {
                input: grid(&[&[2]]),
                output: grid(&[&[2]]),
            },
        },
    ]
}

struct FakeTok;

impl GridTokenizer for FakeTok {
    fn encode_grid(&self, cells: &[u8]) -> std::result::Result<Vec<u32>, String> {
        Ok(cells.iter().map(|&c| c as u32).collect())
    }
}

struct RecordingTok {
    seen: std::cell::RefCell<Option<Vec<u8>>>,
}

impl GridTokenizer for RecordingTok {
    fn encode_grid(&self, cells: &[u8]) -> std::result::Result<Vec<u32>, String> {
        *self.seen.borrow_mut() = Some(cells.to_vec());
        Ok(cells.iter().map(|&c| 100 + c as u32).collect())
    }
}

struct BoomTok(&'static str);

impl GridTokenizer for BoomTok {
    fn encode_grid(&self, _cells: &[u8]) -> std::result::Result<Vec<u32>, String> {
        Err(self.0.to_string())
    }
}

#[test]
fn constants() {
    assert_eq!(N_COLORS, 10);
    assert_eq!(MIN_GRID_SIZE, 1);
    assert_eq!(MAX_GRID_SIZE, 30);
    assert_eq!(N_DIHEDRAL, 8);
    assert_eq!(DEFAULT_N_TRAIN, 3);
    assert_eq!(Dihedral::ALL.len(), 8);
    assert_eq!(FamilyKind::ALL.len(), 8);
}

#[test]
fn grid_validation_and_flatten() {
    let g = grid(&[&[1, 2], &[3, 4]]);
    assert_eq!(g.rows(), 2);
    assert_eq!(g.cols(), 2);
    assert_eq!(g.flatten(), vec![1, 2, 3, 4]);
    assert_eq!(Grid::try_new(vec![]).unwrap_err(), Error::EmptyGrid);
    assert_eq!(
        Grid::try_new(vec![vec![1, 2], vec![3]]).unwrap_err(),
        Error::JaggedGrid
    );
    assert_eq!(
        Grid::try_new(vec![vec![10]]).unwrap_err(),
        Error::BadColor(10)
    );
}

#[test]
fn family_spec_id_sorts_keys() {
    let s = spec(FamilyKind::Translate, &[("dy", 2), ("bg", 0), ("dx", -1)]);
    assert_eq!(s.family_id(), "translate:bg=0,dx=-1,dy=2");
    assert_eq!(
        spec(FamilyKind::Crop, &[("bg", 7)]).family_id(),
        "crop:bg=7"
    );
}

#[test]
fn dihedral_index_roundtrip() {
    for (i, d) in Dihedral::ALL.iter().copied().enumerate() {
        assert_eq!(d.index(), i as u8);
        assert_eq!(Dihedral::from_index(i as u8).unwrap(), d);
    }
    assert_eq!(Dihedral::from_index(8).unwrap_err(), Error::BadDihedral(8));
}

#[test]
fn rot90_clockwise_golden() {
    let src = grid(&[&[1, 2, 3], &[4, 5, 6]]);
    let got = apply_dihedral(&src, Dihedral::Rot90).unwrap();
    assert_eq!(got, grid(&[&[4, 1], &[5, 2], &[6, 3]]));
}

#[test]
fn all_eight_dihedrals_on_2x3_golden() {
    let src = grid(&[&[1, 2, 3], &[4, 5, 6]]);
    let goldens: [(Dihedral, Grid); 8] = [
        (Dihedral::Identity, grid(&[&[1, 2, 3], &[4, 5, 6]])),
        (Dihedral::Rot90, grid(&[&[4, 1], &[5, 2], &[6, 3]])),
        (Dihedral::Rot180, grid(&[&[6, 5, 4], &[3, 2, 1]])),
        (Dihedral::Rot270, grid(&[&[3, 6], &[2, 5], &[1, 4]])),
        (Dihedral::FlipH, grid(&[&[3, 2, 1], &[6, 5, 4]])),
        (Dihedral::FlipV, grid(&[&[4, 5, 6], &[1, 2, 3]])),
        (Dihedral::Transpose, grid(&[&[1, 4], &[2, 5], &[3, 6]])),
        (Dihedral::AntiTranspose, grid(&[&[6, 3], &[5, 2], &[4, 1]])),
    ];
    for (d, expected) in goldens {
        assert_eq!(apply_dihedral(&src, d).unwrap(), expected);
    }
}

#[test]
fn apply_dihedral_matches_reference_on_several_grids() {
    let col: Vec<Vec<u8>> = (0..MAX_GRID_SIZE as u8).map(|c| vec![c]).collect();
    let mut row = (0..N_COLORS as u8).collect::<Vec<_>>();
    row.extend(std::iter::repeat(0).take(MAX_GRID_SIZE - N_COLORS as usize));
    let grids = vec![
        grid(&[&[7]]),
        grid(&[&[1, 2, 3]]),
        grid(&[&[1], &[2], &[3]]),
        grid(&[&[1, 2, 3], &[4, 5, 6]]),
        grid(&[&[1, 2], &[3, 4]]),
        grid(&[&[0, 1, 2], &[3, 4, 5], &[6, 7, 8]]),
        Grid::try_new(col).unwrap(),
        Grid::try_new(vec![row]).unwrap(),
    ];
    for g in &grids {
        for d in Dihedral::ALL {
            let got = apply_dihedral(g, d).unwrap();
            let expected = arc_ref::apply_dihedral(g, d).unwrap();
            assert_eq!(got, expected);
            Grid::try_new(got.cells.clone()).unwrap();
        }
    }
}

#[test]
fn dihedral_swaps_height_and_width() {
    let src = grid(&[&[1, 2, 3], &[4, 5, 6]]);
    for d in Dihedral::ALL {
        let out = apply_dihedral(&src, d).unwrap();
        let swapped = matches!(
            d,
            Dihedral::Rot90 | Dihedral::Rot270 | Dihedral::Transpose | Dihedral::AntiTranspose
        );
        if swapped {
            assert_eq!((out.rows(), out.cols()), (3, 2));
        } else {
            assert_eq!((out.rows(), out.cols()), (2, 3));
        }
    }
}

#[test]
fn color_perm_identity_and_swap() {
    let src = grid(&[&[0, 1, 2], &[3, 0, 9]]);
    assert_eq!(apply_color_perm(&src, &IDENTITY_PERM).unwrap(), src);
    assert_eq!(
        apply_color_perm(&src, &SWAP01).unwrap(),
        grid(&[&[1, 0, 2], &[3, 1, 9]])
    );
    assert_eq!(
        apply_color_perm(&grid(&[&[0, 9]]), &CYCLE).unwrap(),
        grid(&[&[1, 0]])
    );
}

#[test]
fn color_perm_matches_reference() {
    let src = grid(&[&[0, 1, 2], &[3, 4, 5], &[6, 7, 8]]);
    let rev = [9u8, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    for perm in [&IDENTITY_PERM[..], &SWAP01[..], &CYCLE[..], &rev[..]] {
        assert_eq!(
            apply_color_perm(&src, perm).unwrap(),
            arc_ref::apply_color_perm(&src, perm).unwrap()
        );
    }
}

#[test]
fn bad_color_perm_errors() {
    let src = grid(&[&[1]]);
    let bads: [&[u8]; 4] = [
        &[0, 1],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        &[0, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    ];
    for perm in bads {
        assert_eq!(
            apply_color_perm(&src, perm).unwrap_err(),
            Error::BadPermutation
        );
    }
}

#[test]
fn augment_pair_same_transform_on_both_grids() {
    let pair = Pair {
        input: grid(&[&[1, 2, 3], &[4, 5, 6]]),
        output: grid(&[&[9, 0], &[1, 2]]),
    };
    let got = augment_pair(&pair, Dihedral::Rot90, &SWAP01).unwrap();
    let expected = arc_ref::augment_pair(&pair, Dihedral::Rot90, &SWAP01).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn augment_task_preserves_family_id_and_seed() {
    let task = Task {
        family_id: "recolor:dst=2,src=1".into(),
        seed: 99,
        train: vec![Pair {
            input: grid(&[&[1, 0]]),
            output: grid(&[&[2, 0]]),
        }],
        test: Pair {
            input: grid(&[&[1]]),
            output: grid(&[&[2]]),
        },
    };
    let got = augment_task(&task, Dihedral::FlipH, &CYCLE).unwrap();
    let expected = arc_ref::augment_task(&task, Dihedral::FlipH, &CYCLE).unwrap();
    assert_eq!(got.family_id, "recolor:dst=2,src=1");
    assert_eq!(got.seed, 99);
    assert_eq!(got, expected);
}

#[test]
fn tokenize_grid_fake_and_row_major() {
    let g = grid(&[&[1, 2], &[3, 4]]);
    assert_eq!(tokenize_grid(&FakeTok, &g).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(
        tokenize_grid(&FakeTok, &g).unwrap(),
        arc_ref::tokenize_grid(&FakeTok, &g).unwrap()
    );
    let rec = RecordingTok {
        seen: std::cell::RefCell::new(None),
    };
    assert_eq!(tokenize_grid(&rec, &g).unwrap(), vec![101, 102, 103, 104]);
    assert_eq!(rec.seen.into_inner().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn tokenize_grid_tokenizer_error_becomes_message() {
    let g = grid(&[&[1]]);
    assert_eq!(
        tokenize_grid(&BoomTok(""), &g).unwrap_err(),
        Error::Message(String::new())
    );
    assert_eq!(
        tokenize_grid(&BoomTok("boom"), &g).unwrap_err(),
        Error::Message("boom".into())
    );
}

#[test]
fn family_from_spec_rejects_invalid_params() {
    let bads = [
        spec(FamilyKind::Translate, &[("dx", 1), ("dy", 0)]),
        spec(FamilyKind::Translate, &[("dx", 1), ("dy", 0), ("bg", 10)]),
        spec(FamilyKind::Recolor, &[("src", 1), ("dst", 1)]),
        spec(FamilyKind::Recolor, &[("src", 1), ("dst", 2), ("extra", 0)]),
        spec(FamilyKind::Crop, &[]),
        spec(FamilyKind::Tile, &[("nx", 0), ("ny", 1)]),
        spec(FamilyKind::Tile, &[("nx", 31), ("ny", 1)]),
        spec(FamilyKind::Gravity, &[("dir", 4), ("bg", 0)]),
        spec(FamilyKind::Mirror, &[("dihedral", 8)]),
        spec(FamilyKind::Scale, &[("factor", 4)]),
        spec(FamilyKind::Border, &[("color", 0), ("width", 3)]),
        spec(FamilyKind::Border, &[("color", 10), ("width", 1)]),
    ];
    for s in bads {
        assert!(matches!(family_from_spec(s), Err(Error::UnknownFamily)));
    }
}

#[test]
fn generate_matches_reference_for_all_kinds() {
    for s in kind_specs() {
        let fam = family_from_spec(s.clone()).unwrap();
        assert_eq!(fam.id(), s.family_id());
        assert_eq!(fam.kind(), s.kind);
        let got = fam.generate(0).unwrap();
        let expected = arc_ref::family_from_spec(s.clone())
            .unwrap()
            .generate(0)
            .unwrap();
        assert_eq!(got, expected);
        assert_eq!(got.train.len(), DEFAULT_N_TRAIN as usize);
        assert_eq!(got.seed, 0);
        assert_eq!(got.family_id, s.family_id());
    }
}

#[test]
fn generate_deterministic_and_seed_sensitive() {
    let s = kind_specs().into_iter().next().unwrap();
    let fam = family_from_spec(s.clone()).unwrap();
    let a = fam.generate(123).unwrap();
    let b = fam.generate(123).unwrap();
    let c = fam.generate(124).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a, arc_ref::generate_task(&s, 123, DEFAULT_N_TRAIN).unwrap());
}

#[test]
fn generate_valid_grids_and_held_out_test() {
    for s in kind_specs() {
        let task = family_from_spec(s.clone()).unwrap().generate(7).unwrap();
        assert!(!task.train.is_empty());
        let mut train_keys = Vec::new();
        for pair in &task.train {
            Grid::try_new(pair.input.cells.clone()).unwrap();
            Grid::try_new(pair.output.cells.clone()).unwrap();
            train_keys.push((pair.input.cells.clone(), pair.output.cells.clone()));
            assert_eq!(pair.output, arc_ref::apply_family(&pair.input, &s).unwrap());
        }
        Grid::try_new(task.test.input.cells.clone()).unwrap();
        Grid::try_new(task.test.output.cells.clone()).unwrap();
        let test_key = (
            task.test.input.cells.clone(),
            task.test.output.cells.clone(),
        );
        assert!(!train_keys.contains(&test_key));
        assert_eq!(
            task.test.output,
            arc_ref::apply_family(&task.test.input, &s).unwrap()
        );
    }
}

#[test]
fn translate_wraps_around() {
    let s = spec(FamilyKind::Translate, &[("dx", 1), ("dy", 0), ("bg", 0)]);
    let task = family_from_spec(s.clone()).unwrap().generate(3).unwrap();
    for pair in task.train.iter().chain(std::iter::once(&task.test)) {
        assert_eq!(pair.output, arc_ref::apply_family(&pair.input, &s).unwrap());
    }
}

#[test]
fn crop_pairs_match_bounding_box_or_1x1_bg() {
    let s = spec(FamilyKind::Crop, &[("bg", 0)]);
    let task = family_from_spec(s.clone()).unwrap().generate(11).unwrap();
    for pair in task.train.iter().chain(std::iter::once(&task.test)) {
        assert_eq!(pair.output, arc_ref::apply_family(&pair.input, &s).unwrap());
    }
}

#[test]
fn sample_family_specs_empty_is_empty_list() {
    assert!(sample_family_specs(0, 0).unwrap().is_empty());
    assert!(sample_family_specs(0, 99).unwrap().is_empty());
}

#[test]
fn sample_family_specs_deterministic_distinct_covers_kinds() {
    let a = sample_family_specs(32, 42).unwrap();
    let b = sample_family_specs(32, 42).unwrap();
    let c = sample_family_specs(32, 43).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
    assert_eq!(a, arc_ref::sample_family_specs(32, 42).unwrap());
    let ids: Vec<String> = a.iter().map(|s| s.family_id()).collect();
    let unique: std::collections::HashSet<&String> = ids.iter().collect();
    assert_eq!(unique.len(), 32);
    let n = arc_ref::parameter_grid().len() as u32;
    let full = sample_family_specs(n, 0).unwrap();
    let kinds: std::collections::HashSet<FamilyKind> = full.iter().map(|s| s.kind).collect();
    assert_eq!(kinds.len(), 8);
    let full_ids: std::collections::HashSet<String> = full.iter().map(|s| s.family_id()).collect();
    assert_eq!(full_ids.len(), n as usize);
    assert_eq!(full, arc_ref::sample_family_specs(n, 0).unwrap());
}

#[test]
fn diversity_stats_empty_is_all_zeros() {
    let got = diversity_stats(&[]);
    assert_eq!(
        got,
        DiversityStats {
            n_tasks: 0,
            n_families: 0,
            unique_test_inputs: 0,
            unique_test_outputs: 0,
            unique_tasks: 0,
            color_histogram: [0; 10],
            unique_shapes: 0,
            collision_rate: 0.0,
        }
    );
}

#[test]
fn diversity_stats_hand_golden() {
    let got = diversity_stats(&golden_tasks());
    assert_eq!(got, GOLDEN_DIVERSITY);
    assert_eq!(got, arc_ref::diversity_stats(&golden_tasks()));
}

#[test]
fn diversity_stats_matches_reference_on_generated_tasks() {
    let s = spec(FamilyKind::Recolor, &[("src", 1), ("dst", 2)]);
    let tasks: Vec<Task> = (0..4)
        .map(|seed| arc_ref::generate_task(&s, seed, 1).unwrap())
        .collect();
    assert_eq!(diversity_stats(&tasks), arc_ref::diversity_stats(&tasks));
}

#[test]
fn corpus_sample_empty_n_and_empty_specs() {
    let specs = kind_specs()[..2].to_vec();
    assert_eq!(
        ProceduralCorpus::new(specs.clone(), 0)
            .sample(0)
            .unwrap_err(),
        Error::EmptyTasks
    );
    assert_eq!(
        ProceduralCorpus::new(vec![], 0).sample(0).unwrap_err(),
        Error::EmptyTasks
    );
    assert_eq!(
        ProceduralCorpus::new(vec![], 0).sample(1).unwrap_err(),
        Error::UnknownFamily
    );
    let mut c = ProceduralCorpus::new(specs, 0);
    c.n_train = 0;
    assert_eq!(c.sample(1).unwrap_err(), Error::EmptyTrain);
}

#[test]
fn corpus_sample_matches_reference_and_cycles_specs() {
    let specs = kind_specs()[..3].to_vec();
    let mut corpus = ProceduralCorpus::new(specs.clone(), 5);
    corpus.n_train = 2;
    let got = corpus.sample(5).unwrap();
    let expected = arc_ref::sample_corpus(&specs, 5, 2, 5).unwrap();
    assert_eq!(got, expected);
    assert_eq!(got.len(), 5);
    for (i, task) in got.iter().enumerate() {
        assert_eq!(task.train.len(), 2);
        assert_eq!(task.seed, 5 + i as u64);
        assert_eq!(task.family_id, specs[i % 3].family_id());
    }
    let fam_task = family_from_spec(specs[0].clone())
        .unwrap()
        .generate(5)
        .unwrap();
    assert_eq!(fam_task.train.len(), DEFAULT_N_TRAIN as usize);
    assert_ne!(got[0], fam_task);
}

#[test]
fn corpus_sample_unknown_family_in_bag() {
    let bad = spec(FamilyKind::Scale, &[("factor", 1)]);
    assert_eq!(
        ProceduralCorpus::new(vec![bad], 0).sample(1).unwrap_err(),
        Error::UnknownFamily
    );
}
