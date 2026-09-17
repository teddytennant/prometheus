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

use std::collections::{BTreeMap, HashSet};

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

const TOO_MANY_SPECS: &str = "requested more family specs than the parameter grid";
const DISTINCT_PAIR_FAIL: &str = "could not sample distinct pairs";
const MAX_SAMPLE_DIM: usize = 8;
const MAX_PAIR_ATTEMPTS: usize = 10000;

type Cells = Vec<Vec<u8>>;
type PairCells = (Cells, Cells);

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_bounded(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

struct SpecFamily {
    spec: FamilySpec,
    id: String,
}

impl Family for SpecFamily {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> FamilyKind {
        self.spec.kind
    }

    fn generate(&self, seed: u64) -> Result<Task> {
        generate_task(&self.spec, seed, DEFAULT_N_TRAIN)
    }
}

/// Apply a D4 action. `rot90` is clockwise. Output size swaps on rot90,
/// rot270, transpose, anti-transpose.
pub fn apply_dihedral(grid: &Grid, dihedral: Dihedral) -> Result<Grid> {
    let src = &grid.cells;
    let h = src.len();
    let w = src[0].len();
    let cells = match dihedral {
        Dihedral::Identity => src.clone(),
        Dihedral::Rot90 => {
            let mut out = vec![vec![0u8; h]; w];
            for r in 0..w {
                for c in 0..h {
                    out[r][c] = src[h - 1 - c][r];
                }
            }
            out
        }
        Dihedral::Rot180 => {
            let mut out = vec![vec![0u8; w]; h];
            for r in 0..h {
                for c in 0..w {
                    out[r][c] = src[h - 1 - r][w - 1 - c];
                }
            }
            out
        }
        Dihedral::Rot270 => {
            let mut out = vec![vec![0u8; h]; w];
            for r in 0..w {
                for c in 0..h {
                    out[r][c] = src[c][w - 1 - r];
                }
            }
            out
        }
        Dihedral::FlipH => {
            let mut out = vec![vec![0u8; w]; h];
            for r in 0..h {
                for c in 0..w {
                    out[r][c] = src[r][w - 1 - c];
                }
            }
            out
        }
        Dihedral::FlipV => {
            let mut out = vec![vec![0u8; w]; h];
            for r in 0..h {
                for c in 0..w {
                    out[r][c] = src[h - 1 - r][c];
                }
            }
            out
        }
        Dihedral::Transpose => {
            let mut out = vec![vec![0u8; h]; w];
            for r in 0..w {
                for c in 0..h {
                    out[r][c] = src[c][r];
                }
            }
            out
        }
        Dihedral::AntiTranspose => {
            let mut out = vec![vec![0u8; h]; w];
            for r in 0..w {
                for c in 0..h {
                    out[r][c] = src[h - 1 - c][w - 1 - r];
                }
            }
            out
        }
    };
    Grid::try_new(cells)
}

/// `perm[c]` is the new color of `c`. Must be a permutation of `0..N_COLORS`.
pub fn apply_color_perm(grid: &Grid, perm: &[u8]) -> Result<Grid> {
    if perm.len() != N_COLORS as usize {
        return Err(Error::BadPermutation);
    }
    let mut seen = [false; 10];
    for &v in perm {
        if v >= N_COLORS || seen[v as usize] {
            return Err(Error::BadPermutation);
        }
        seen[v as usize] = true;
    }
    let cells: Vec<Vec<u8>> = grid
        .cells
        .iter()
        .map(|row| row.iter().map(|&c| perm[c as usize]).collect())
        .collect();
    Grid::try_new(cells)
}

/// Same dihedral and color perm on every grid in the pair.
pub fn augment_pair(pair: &Pair, dihedral: Dihedral, perm: &[u8]) -> Result<Pair> {
    Ok(Pair {
        input: apply_color_perm(&apply_dihedral(&pair.input, dihedral)?, perm)?,
        output: apply_color_perm(&apply_dihedral(&pair.output, dihedral)?, perm)?,
    })
}

/// Same dihedral and color perm on every grid in the task. `family_id` and
/// `seed` are unchanged. Augmentation is a training-time view, not a new
/// family.
pub fn augment_task(task: &Task, dihedral: Dihedral, perm: &[u8]) -> Result<Task> {
    let train = task
        .train
        .iter()
        .map(|p| augment_pair(p, dihedral, perm))
        .collect::<Result<Vec<_>>>()?;
    Ok(Task {
        family_id: task.family_id.clone(),
        seed: task.seed,
        train,
        test: augment_pair(&task.test, dihedral, perm)?,
    })
}

/// F6 encode of `grid.flatten()`.
pub fn tokenize_grid(tok: &dyn GridTokenizer, grid: &Grid) -> Result<Vec<u32>> {
    tok.encode_grid(&grid.flatten()).map_err(Error::Message)
}

/// Construct a family from a spec. Unknown or invalid params are
/// [`Error::UnknownFamily`].
pub fn family_from_spec(spec: FamilySpec) -> Result<Box<dyn Family>> {
    validate_spec(&spec)?;
    let id = spec.family_id();
    Ok(Box::new(SpecFamily { spec, id }))
}

/// `n` distinct specs from the parameter grid, deterministic in `seed`.
pub fn sample_family_specs(n: u32, seed: u64) -> Result<Vec<FamilySpec>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut specs = parameter_grid();
    let n = n as usize;
    if n > specs.len() {
        return Err(Error::Message(TOO_MANY_SPECS.to_string()));
    }
    let mut rng = SplitMix64::new(seed);
    for i in 0..specs.len() {
        let j = i + rng.next_bounded(specs.len() - i);
        specs.swap(i, j);
    }
    specs.truncate(n);
    Ok(specs)
}

/// Gate: uniqueness and color/shape coverage over `tasks`. Empty is zeros
/// and `collision_rate == 0.0`.
pub fn diversity_stats(tasks: &[Task]) -> DiversityStats {
    let n_tasks = tasks.len() as u64;
    let mut families = HashSet::new();
    let mut test_in = HashSet::new();
    let mut test_out = HashSet::new();
    let mut task_keys = HashSet::new();
    let mut hist = [0u64; 10];
    let mut shapes = HashSet::new();

    fn acc(grid: &Grid, hist: &mut [u64; 10], shapes: &mut HashSet<(usize, usize)>) {
        shapes.insert((grid.rows(), grid.cols()));
        for row in &grid.cells {
            for &c in row {
                hist[c as usize] += 1;
            }
        }
    }

    for task in tasks {
        families.insert(task.family_id.clone());
        test_in.insert(task.test.input.cells.clone());
        test_out.insert(task.test.output.cells.clone());
        let train_key: Vec<PairCells> = task
            .train
            .iter()
            .map(|p| (p.input.cells.clone(), p.output.cells.clone()))
            .collect();
        task_keys.insert((
            train_key,
            task.test.input.cells.clone(),
            task.test.output.cells.clone(),
        ));
        for pair in &task.train {
            acc(&pair.input, &mut hist, &mut shapes);
            acc(&pair.output, &mut hist, &mut shapes);
        }
        acc(&task.test.input, &mut hist, &mut shapes);
        acc(&task.test.output, &mut hist, &mut shapes);
    }

    let unique_tasks = task_keys.len() as u64;
    let collision_rate = if n_tasks == 0 {
        0.0
    } else {
        1.0 - unique_tasks as f64 / n_tasks as f64
    };
    DiversityStats {
        n_tasks,
        n_families: families.len() as u64,
        unique_test_inputs: test_in.len() as u64,
        unique_test_outputs: test_out.len() as u64,
        unique_tasks,
        color_histogram: hist,
        unique_shapes: shapes.len() as u64,
        collision_rate,
    }
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
        if n == 0 {
            return Err(Error::EmptyTasks);
        }
        if self.specs.is_empty() {
            return Err(Error::UnknownFamily);
        }
        if self.n_train == 0 {
            return Err(Error::EmptyTrain);
        }
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let spec = &self.specs[i as usize % self.specs.len()];
            out.push(generate_task(
                spec,
                self.seed.wrapping_add(i as u64),
                self.n_train,
            )?);
        }
        Ok(out)
    }
}

fn expect_keys(spec: &FamilySpec, keys: &[&str]) -> Result<()> {
    if spec.params.len() != keys.len() {
        return Err(Error::UnknownFamily);
    }
    for k in keys {
        if !spec.params.contains_key(*k) {
            return Err(Error::UnknownFamily);
        }
    }
    Ok(())
}

fn get_i(spec: &FamilySpec, k: &str) -> Result<i32> {
    spec.params.get(k).copied().ok_or(Error::UnknownFamily)
}

fn color(v: i32) -> Result<u8> {
    if v < 0 || v >= i32::from(N_COLORS) {
        Err(Error::UnknownFamily)
    } else {
        Ok(v as u8)
    }
}

fn validate_spec(spec: &FamilySpec) -> Result<()> {
    match spec.kind {
        FamilyKind::Translate => {
            expect_keys(spec, &["dx", "dy", "bg"])?;
            color(get_i(spec, "bg")?)?;
            Ok(())
        }
        FamilyKind::Recolor => {
            expect_keys(spec, &["src", "dst"])?;
            let src = color(get_i(spec, "src")?)?;
            let dst = color(get_i(spec, "dst")?)?;
            if src == dst {
                Err(Error::UnknownFamily)
            } else {
                Ok(())
            }
        }
        FamilyKind::Crop => {
            expect_keys(spec, &["bg"])?;
            color(get_i(spec, "bg")?)?;
            Ok(())
        }
        FamilyKind::Tile => {
            expect_keys(spec, &["nx", "ny"])?;
            let nx = get_i(spec, "nx")?;
            let ny = get_i(spec, "ny")?;
            if nx < 1 || ny < 1 || nx > MAX_GRID_SIZE as i32 || ny > MAX_GRID_SIZE as i32 {
                Err(Error::UnknownFamily)
            } else {
                Ok(())
            }
        }
        FamilyKind::Gravity => {
            expect_keys(spec, &["dir", "bg"])?;
            let dir = get_i(spec, "dir")?;
            if !(0..=3).contains(&dir) {
                return Err(Error::UnknownFamily);
            }
            color(get_i(spec, "bg")?)?;
            Ok(())
        }
        FamilyKind::Mirror => {
            expect_keys(spec, &["dihedral"])?;
            let d = get_i(spec, "dihedral")?;
            if d < 0 || d >= i32::from(N_DIHEDRAL) {
                Err(Error::UnknownFamily)
            } else {
                Ok(())
            }
        }
        FamilyKind::Scale => {
            expect_keys(spec, &["factor"])?;
            let f = get_i(spec, "factor")?;
            if f == 2 || f == 3 {
                Ok(())
            } else {
                Err(Error::UnknownFamily)
            }
        }
        FamilyKind::Border => {
            expect_keys(spec, &["color", "width"])?;
            color(get_i(spec, "color")?)?;
            let w = get_i(spec, "width")?;
            if w == 1 || w == 2 {
                Ok(())
            } else {
                Err(Error::UnknownFamily)
            }
        }
    }
}

fn wrap(i: i64, n: usize) -> usize {
    let n = n as i64;
    let mut r = i % n;
    if r < 0 {
        r += n;
    }
    r as usize
}

fn apply_family(grid: &Grid, spec: &FamilySpec) -> Result<Grid> {
    validate_spec(spec)?;
    match spec.kind {
        FamilyKind::Translate => translate(
            grid,
            get_i(spec, "dx")?,
            get_i(spec, "dy")?,
            color(get_i(spec, "bg")?)?,
        ),
        FamilyKind::Recolor => recolor(
            grid,
            color(get_i(spec, "src")?)?,
            color(get_i(spec, "dst")?)?,
        ),
        FamilyKind::Crop => crop(grid, color(get_i(spec, "bg")?)?),
        FamilyKind::Tile => tile(grid, get_i(spec, "nx")?, get_i(spec, "ny")?),
        FamilyKind::Gravity => gravity(grid, get_i(spec, "dir")?, color(get_i(spec, "bg")?)?),
        FamilyKind::Mirror => {
            apply_dihedral(grid, Dihedral::from_index(get_i(spec, "dihedral")? as u8)?)
        }
        FamilyKind::Scale => scale(grid, get_i(spec, "factor")?),
        FamilyKind::Border => border(
            grid,
            color(get_i(spec, "color")?)?,
            get_i(spec, "width")? as usize,
        ),
    }
}

fn translate(grid: &Grid, dx: i32, dy: i32, bg: u8) -> Result<Grid> {
    let h = grid.rows();
    let w = grid.cols();
    let mut out = vec![vec![bg; w]; h];
    for r in 0..h {
        for c in 0..w {
            let val = grid.cells[r][c];
            if val != bg {
                out[wrap(r as i64 + i64::from(dy), h)][wrap(c as i64 + i64::from(dx), w)] = val;
            }
        }
    }
    Grid::try_new(out)
}

fn recolor(grid: &Grid, src: u8, dst: u8) -> Result<Grid> {
    let cells = grid
        .cells
        .iter()
        .map(|row| {
            row.iter()
                .map(|&c| if c == src { dst } else { c })
                .collect()
        })
        .collect();
    Grid::try_new(cells)
}

fn crop(grid: &Grid, bg: u8) -> Result<Grid> {
    let mut coords = Vec::new();
    for (r, row) in grid.cells.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            if val != bg {
                coords.push((r, c));
            }
        }
    }
    if coords.is_empty() {
        return Grid::try_new(vec![vec![bg]]);
    }
    let min_r = coords.iter().map(|x| x.0).min().unwrap();
    let max_r = coords.iter().map(|x| x.0).max().unwrap();
    let min_c = coords.iter().map(|x| x.1).min().unwrap();
    let max_c = coords.iter().map(|x| x.1).max().unwrap();
    let cells = grid.cells[min_r..=max_r]
        .iter()
        .map(|row| row[min_c..=max_c].to_vec())
        .collect();
    Grid::try_new(cells)
}

fn tile(grid: &Grid, nx: i32, ny: i32) -> Result<Grid> {
    let nx = nx as usize;
    let ny = ny as usize;
    let h = grid.rows();
    let mut out = Vec::new();
    for _ in 0..ny {
        for r in 0..h {
            let mut row = Vec::new();
            for _ in 0..nx {
                row.extend_from_slice(&grid.cells[r]);
            }
            out.push(row);
        }
    }
    Grid::try_new(out)
}

fn pack_column(grid: &Grid, c: usize, bg: u8, down: bool) -> Vec<u8> {
    let h = grid.rows();
    let objs: Vec<u8> = grid
        .cells
        .iter()
        .filter_map(|row| {
            let v = row[c];
            (v != bg).then_some(v)
        })
        .collect();
    let mut col = vec![bg; h];
    let start = if down { h - objs.len() } else { 0 };
    col[start..start + objs.len()].copy_from_slice(&objs);
    col
}

fn pack_row(row: &[u8], bg: u8, left: bool) -> Vec<u8> {
    let w = row.len();
    let objs: Vec<u8> = row.iter().copied().filter(|&v| v != bg).collect();
    let mut out = vec![bg; w];
    let start = if left { 0 } else { w - objs.len() };
    out[start..start + objs.len()].copy_from_slice(&objs);
    out
}

fn gravity(grid: &Grid, direction: i32, bg: u8) -> Result<Grid> {
    let h = grid.rows();
    let w = grid.cols();
    let cells = match direction {
        0 | 1 => {
            let down = direction == 0;
            let cols: Vec<Vec<u8>> = (0..w).map(|c| pack_column(grid, c, bg, down)).collect();
            (0..h)
                .map(|r| cols.iter().map(|col| col[r]).collect())
                .collect()
        }
        2 => grid
            .cells
            .iter()
            .map(|row| pack_row(row, bg, true))
            .collect(),
        _ => grid
            .cells
            .iter()
            .map(|row| pack_row(row, bg, false))
            .collect(),
    };
    Grid::try_new(cells)
}

fn scale(grid: &Grid, factor: i32) -> Result<Grid> {
    let f = factor as usize;
    let mut out = Vec::new();
    for row in &grid.cells {
        let mut scaled = Vec::new();
        for &c in row {
            for _ in 0..f {
                scaled.push(c);
            }
        }
        for _ in 0..f {
            out.push(scaled.clone());
        }
    }
    Grid::try_new(out)
}

fn border(grid: &Grid, color: u8, width: usize) -> Result<Grid> {
    let h = grid.rows();
    let w = grid.cols();
    let oh = h + 2 * width;
    let ow = w + 2 * width;
    let mut out = vec![vec![color; ow]; oh];
    for r in 0..h {
        for c in 0..w {
            out[r + width][c + width] = grid.cells[r][c];
        }
    }
    Grid::try_new(out)
}

fn max_in_hw(spec: &FamilySpec) -> (usize, usize) {
    match spec.kind {
        FamilyKind::Tile => {
            let nx = *spec.params.get("nx").unwrap() as usize;
            let ny = *spec.params.get("ny").unwrap() as usize;
            (MAX_GRID_SIZE / ny, MAX_GRID_SIZE / nx)
        }
        FamilyKind::Scale => {
            let f = *spec.params.get("factor").unwrap() as usize;
            (MAX_GRID_SIZE / f, MAX_GRID_SIZE / f)
        }
        FamilyKind::Border => {
            let w = *spec.params.get("width").unwrap() as usize;
            (MAX_GRID_SIZE - 2 * w, MAX_GRID_SIZE - 2 * w)
        }
        _ => (MAX_GRID_SIZE, MAX_GRID_SIZE),
    }
}

fn sample_pair(rng: &mut SplitMix64, spec: &FamilySpec) -> Result<Pair> {
    let (mut max_h, mut max_w) = max_in_hw(spec);
    max_h = max_h.min(MAX_SAMPLE_DIM);
    max_w = max_w.min(MAX_SAMPLE_DIM);
    let h = 1 + rng.next_bounded(max_h);
    let w = 1 + rng.next_bounded(max_w);
    let mut cells = vec![vec![0u8; w]; h];
    for row in cells.iter_mut() {
        for cell in row.iter_mut() {
            *cell = rng.next_bounded(N_COLORS as usize) as u8;
        }
    }
    let input = Grid::try_new(cells)?;
    let output = apply_family(&input, spec)?;
    Ok(Pair { input, output })
}

fn generate_task(spec: &FamilySpec, seed: u64, n_train: u32) -> Result<Task> {
    validate_spec(spec)?;
    if n_train == 0 {
        return Err(Error::EmptyTrain);
    }
    let mut rng = SplitMix64::new(seed);
    let need = n_train as usize + 1;
    let mut pairs = Vec::new();
    let mut seen: HashSet<PairCells> = HashSet::new();
    let mut attempts = 0usize;
    while pairs.len() < need {
        attempts += 1;
        if attempts > MAX_PAIR_ATTEMPTS {
            return Err(Error::Message(DISTINCT_PAIR_FAIL.to_string()));
        }
        let pair = sample_pair(&mut rng, spec)?;
        let key = (pair.input.cells.clone(), pair.output.cells.clone());
        if !seen.insert(key) {
            continue;
        }
        pairs.push(pair);
    }
    let test = pairs.pop().unwrap();
    Ok(Task {
        family_id: spec.family_id(),
        seed,
        train: pairs,
        test,
    })
}

fn fs(kind: FamilyKind, pairs: &[(&str, i32)]) -> FamilySpec {
    let params: BTreeMap<String, i32> = pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
    FamilySpec { kind, params }
}

fn parameter_grid() -> Vec<FamilySpec> {
    let mut specs = Vec::new();
    for bg in 0..i32::from(N_COLORS) {
        for dx in -3..=3 {
            for dy in -3..=3 {
                specs.push(fs(
                    FamilyKind::Translate,
                    &[("dx", dx), ("dy", dy), ("bg", bg)],
                ));
            }
        }
    }
    for src in 0..i32::from(N_COLORS) {
        for dst in 0..i32::from(N_COLORS) {
            if src != dst {
                specs.push(fs(FamilyKind::Recolor, &[("src", src), ("dst", dst)]));
            }
        }
    }
    for bg in 0..i32::from(N_COLORS) {
        specs.push(fs(FamilyKind::Crop, &[("bg", bg)]));
    }
    for nx in 1..5 {
        for ny in 1..5 {
            specs.push(fs(FamilyKind::Tile, &[("nx", nx), ("ny", ny)]));
        }
    }
    for bg in 0..i32::from(N_COLORS) {
        for dir in 0..4 {
            specs.push(fs(FamilyKind::Gravity, &[("dir", dir), ("bg", bg)]));
        }
    }
    for d in 0..i32::from(N_DIHEDRAL) {
        specs.push(fs(FamilyKind::Mirror, &[("dihedral", d)]));
    }
    for factor in [2, 3] {
        specs.push(fs(FamilyKind::Scale, &[("factor", factor)]));
    }
    for color in 0..i32::from(N_COLORS) {
        for width in [1, 2] {
            specs.push(fs(
                FamilyKind::Border,
                &[("color", color), ("width", width)],
            ));
        }
    }
    specs
}
