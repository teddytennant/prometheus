//! Procedural ARC-like grids (spec 7.1, 10, 15.5 E3).
//!
//! re-arc-style generators: parameterized task families produce input/output
//! grid pairs. Cells are palette indices `0..N_COLORS`. One token per cell
//! via F6 `encode_grid` on the row-major flatten. Training-time augmentation
//! is the 8 dihedral transforms times color permutations. The gate is
//! [`diversity_stats`].
//!
//! Grids are structurally the D2 `Grid` (`cells: Vec<Vec<u8>>`) but this
//! crate does not depend on `prometheus-verifiers`. Tokenization wraps F6
//! behind [`GridTokenizer`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// ARC-AGI cell colors 0-9. Matches F6 `ARC_N_COLORS`.
pub const N_COLORS: u8 = 10;
/// Inclusive minimum rows and cols.
pub const MIN_GRID_SIZE: usize = 1;
/// Inclusive maximum rows and cols. ARC-AGI official max.
pub const MAX_GRID_SIZE: usize = 30;
/// Order of the dihedral group D4.
pub const N_DIHEDRAL: u8 = 8;
/// Default demonstration pairs per task.
pub const DEFAULT_N_TRAIN: u32 = 3;

/// One ARC-style grid. Cell values are palette indices in `0..N_COLORS`.
/// Rectangular, rows and cols in `MIN_GRID_SIZE..=MAX_GRID_SIZE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Grid {
    pub cells: Vec<Vec<u8>>,
}

impl Grid {
    /// Validate and wrap. Empty, jagged, out-of-range size, or a color
    /// outside `0..N_COLORS` is an error.
    pub fn try_new(cells: Vec<Vec<u8>>) -> Result<Self> {
        if cells.is_empty() || cells[0].is_empty() {
            return Err(Error::EmptyGrid);
        }
        let rows = cells.len();
        let cols = cells[0].len();
        if !(MIN_GRID_SIZE..=MAX_GRID_SIZE).contains(&rows)
            || !(MIN_GRID_SIZE..=MAX_GRID_SIZE).contains(&cols)
        {
            return Err(Error::GridSize { rows, cols });
        }
        for row in &cells {
            if row.len() != cols {
                return Err(Error::JaggedGrid);
            }
            for &c in row {
                if c >= N_COLORS {
                    return Err(Error::BadColor(c));
                }
            }
        }
        Ok(Self { cells })
    }

    pub fn rows(&self) -> usize {
        self.cells.len()
    }

    pub fn cols(&self) -> usize {
        self.cells.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Row-major flatten. This is the F6 `encode_grid` input.
    pub fn flatten(&self) -> Vec<u8> {
        self.cells.iter().flatten().copied().collect()
    }
}

/// One input/output pair.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pair {
    pub input: Grid,
    pub output: Grid,
}

/// One generated task. `train` are demonstrations; `test` is held out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub family_id: String,
    pub seed: u64,
    pub train: Vec<Pair>,
    pub test: Pair,
}

/// D4 action. JSON names are snake_case values. Indices 0..8:
/// 0 identity, 1 rot90 CW, 2 rot180, 3 rot270 CW, 4 flip left-right,
/// 5 flip up-down, 6 transpose (main diagonal), 7 anti-transpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dihedral {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    FlipH,
    FlipV,
    Transpose,
    AntiTranspose,
}

impl Dihedral {
    pub const ALL: [Dihedral; 8] = [
        Dihedral::Identity,
        Dihedral::Rot90,
        Dihedral::Rot180,
        Dihedral::Rot270,
        Dihedral::FlipH,
        Dihedral::FlipV,
        Dihedral::Transpose,
        Dihedral::AntiTranspose,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Dihedral::Identity => "identity",
            Dihedral::Rot90 => "rot90",
            Dihedral::Rot180 => "rot180",
            Dihedral::Rot270 => "rot270",
            Dihedral::FlipH => "flip_h",
            Dihedral::FlipV => "flip_v",
            Dihedral::Transpose => "transpose",
            Dihedral::AntiTranspose => "anti_transpose",
        }
    }

    pub fn index(self) -> u8 {
        match self {
            Dihedral::Identity => 0,
            Dihedral::Rot90 => 1,
            Dihedral::Rot180 => 2,
            Dihedral::Rot270 => 3,
            Dihedral::FlipH => 4,
            Dihedral::FlipV => 5,
            Dihedral::Transpose => 6,
            Dihedral::AntiTranspose => 7,
        }
    }

    pub fn from_index(k: u8) -> Result<Self> {
        Dihedral::ALL
            .get(k as usize)
            .copied()
            .ok_or(Error::BadDihedral(k))
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "identity" => Ok(Dihedral::Identity),
            "rot90" => Ok(Dihedral::Rot90),
            "rot180" => Ok(Dihedral::Rot180),
            "rot270" => Ok(Dihedral::Rot270),
            "flip_h" => Ok(Dihedral::FlipH),
            "flip_v" => Ok(Dihedral::FlipV),
            "transpose" => Ok(Dihedral::Transpose),
            "anti_transpose" => Ok(Dihedral::AntiTranspose),
            other => Err(Error::Message(format!("unknown dihedral {other}"))),
        }
    }
}

/// Primitive generator kind. Distinct `(kind, params)` pairs are distinct
/// families. Spec 7.1 wants millions of families via parameterization, not
/// 400 hand-written ARC clones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FamilyKind {
    /// Move the non-background object by `(dx, dy)`. Params: `dx`, `dy`, `bg`.
    Translate,
    /// Map color `src` to `dst`. Params: `src`, `dst`.
    Recolor,
    /// Crop to the bounding box of non-`bg` cells. Params: `bg`.
    Crop,
    /// Repeat the input `nx` by `ny` times. Params: `nx`, `ny`.
    Tile,
    /// Drop non-`bg` cells along `dir` (0 down, 1 up, 2 left, 3 right).
    /// Params: `dir`, `bg`.
    Gravity,
    /// Output is a dihedral of the input. Params: `dihedral` (0..8).
    Mirror,
    /// Nearest-neighbor upscale. Params: `factor` (2 or 3).
    Scale,
    /// Draw a border. Params: `color`, `width` (1 or 2).
    Border,
}

impl FamilyKind {
    pub const ALL: [FamilyKind; 8] = [
        FamilyKind::Translate,
        FamilyKind::Recolor,
        FamilyKind::Crop,
        FamilyKind::Tile,
        FamilyKind::Gravity,
        FamilyKind::Mirror,
        FamilyKind::Scale,
        FamilyKind::Border,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            FamilyKind::Translate => "translate",
            FamilyKind::Recolor => "recolor",
            FamilyKind::Crop => "crop",
            FamilyKind::Tile => "tile",
            FamilyKind::Gravity => "gravity",
            FamilyKind::Mirror => "mirror",
            FamilyKind::Scale => "scale",
            FamilyKind::Border => "border",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "translate" => Ok(FamilyKind::Translate),
            "recolor" => Ok(FamilyKind::Recolor),
            "crop" => Ok(FamilyKind::Crop),
            "tile" => Ok(FamilyKind::Tile),
            "gravity" => Ok(FamilyKind::Gravity),
            "mirror" => Ok(FamilyKind::Mirror),
            "scale" => Ok(FamilyKind::Scale),
            "border" => Ok(FamilyKind::Border),
            other => Err(Error::Message(format!("unknown family {other}"))),
        }
    }
}

/// Parameterized family. `params` keys are documented on [`FamilyKind`].
/// `family_id` is `"{kind}:{k}={v},..."` with keys sorted.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FamilySpec {
    pub kind: FamilyKind,
    pub params: BTreeMap<String, i32>,
}

impl FamilySpec {
    pub fn new(kind: FamilyKind, params: BTreeMap<String, i32>) -> Self {
        Self { kind, params }
    }

    pub fn family_id(&self) -> String {
        let mut parts: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        parts.sort();
        if parts.is_empty() {
            self.kind.as_str().to_string()
        } else {
            format!("{}:{}", self.kind.as_str(), parts.join(","))
        }
    }
}

/// Generator for one family. `generate` is deterministic in `seed`.
pub trait Family {
    fn id(&self) -> &str;
    fn kind(&self) -> FamilyKind;
    fn generate(&self, seed: u64) -> Result<Task>;
}

/// F6 `encode_grid`. Implementors wrap `tokenizer.Tokenizer.encode_grid`.
pub trait GridTokenizer {
    fn encode_grid(&self, cells: &[u8]) -> std::result::Result<Vec<u32>, String>;
}

/// Gate numbers. `collision_rate` is `1 - unique_tasks / n_tasks`, or 0.0
/// when `n_tasks == 0`. Hashes are in-process identity of the grid cells,
/// not a stable digest.
#[derive(Debug, Clone, PartialEq)]
pub struct DiversityStats {
    pub n_tasks: u64,
    pub n_families: u64,
    pub unique_test_inputs: u64,
    pub unique_test_outputs: u64,
    pub unique_tasks: u64,
    pub color_histogram: [u64; 10],
    pub unique_shapes: u64,
    pub collision_rate: f64,
}

/// Apply a D4 action. `rot90` is clockwise. Output size swaps on rot90,
/// rot270, transpose, anti-transpose.
pub fn apply_dihedral(grid: &Grid, dihedral: Dihedral) -> Result<Grid> {
    let _ = (grid, dihedral);
    unimplemented!("E3 apply_dihedral")
}

/// `perm[c]` is the new color of `c`. Must be a permutation of `0..N_COLORS`.
pub fn apply_color_perm(grid: &Grid, perm: &[u8]) -> Result<Grid> {
    let _ = (grid, perm);
    unimplemented!("E3 apply_color_perm")
}

/// Same dihedral and color perm on every grid in the pair.
pub fn augment_pair(pair: &Pair, dihedral: Dihedral, perm: &[u8]) -> Result<Pair> {
    let _ = (pair, dihedral, perm);
    unimplemented!("E3 augment_pair")
}

/// Same dihedral and color perm on every grid in the task. `family_id` and
/// `seed` are unchanged. Augmentation is a training-time view, not a new
/// family.
pub fn augment_task(task: &Task, dihedral: Dihedral, perm: &[u8]) -> Result<Task> {
    let _ = (task, dihedral, perm);
    unimplemented!("E3 augment_task")
}

/// F6 encode of `grid.flatten()`.
pub fn tokenize_grid(tok: &dyn GridTokenizer, grid: &Grid) -> Result<Vec<u32>> {
    let _ = (tok, grid);
    unimplemented!("E3 tokenize_grid")
}

/// Construct a family from a spec. Unknown or invalid params are
/// [`Error::UnknownFamily`].
pub fn family_from_spec(spec: FamilySpec) -> Result<Box<dyn Family>> {
    let _ = spec;
    unimplemented!("E3 family_from_spec")
}

/// `n` distinct specs from the parameter grid, deterministic in `seed`.
pub fn sample_family_specs(n: u32, seed: u64) -> Result<Vec<FamilySpec>> {
    let _ = (n, seed);
    unimplemented!("E3 sample_family_specs")
}

/// Gate: uniqueness and color/shape coverage over `tasks`. Empty is zeros
/// and `collision_rate == 0.0`.
pub fn diversity_stats(tasks: &[Task]) -> DiversityStats {
    let _ = tasks;
    unimplemented!("E3 diversity_stats")
}

/// A bag of families to sample from.
pub struct ProceduralCorpus {
    pub specs: Vec<FamilySpec>,
    pub seed: u64,
    pub n_train: u32,
}

impl ProceduralCorpus {
    pub fn new(specs: Vec<FamilySpec>, seed: u64) -> Self {
        Self {
            specs,
            seed,
            n_train: DEFAULT_N_TRAIN,
        }
    }

    /// `n` tasks, cycling specs, seeds `seed + i`. Empty `specs` is
    /// [`Error::UnknownFamily`]. `n == 0` is [`Error::EmptyTasks`].
    pub fn sample(&self, n: u32) -> Result<Vec<Task>> {
        let _ = n;
        unimplemented!("E3 ProceduralCorpus::sample")
    }
}
