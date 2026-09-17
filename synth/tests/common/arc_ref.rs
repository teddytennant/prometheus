//! Independent slow reference for E3 procedural ARC-like grids.
//! Must match `tests/reference/arc.py`. Production must never import this.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashSet};

use prometheus_synth::{
    Dihedral, DiversityStats, Error, FamilyKind, FamilySpec, Grid, GridTokenizer, Pair, Result,
    Task, DEFAULT_N_TRAIN, MAX_GRID_SIZE, N_COLORS, N_DIHEDRAL,
};

pub const TOO_MANY_SPECS: &str = "requested more family specs than the parameter grid";
pub const DISTINCT_PAIR_FAIL: &str = "could not sample distinct pairs";
pub const MAX_SAMPLE_DIM: usize = 8;
pub const MAX_PAIR_ATTEMPTS: usize = 10000;

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

pub struct RefFamily {
    spec: FamilySpec,
    id: String,
}

impl RefFamily {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn kind(&self) -> FamilyKind {
        self.spec.kind
    }

    pub fn generate(&self, seed: u64) -> Result<Task> {
        generate_task(&self.spec, seed, DEFAULT_N_TRAIN)
    }
}

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

pub fn augment_pair(pair: &Pair, dihedral: Dihedral, perm: &[u8]) -> Result<Pair> {
    Ok(Pair {
        input: apply_color_perm(&apply_dihedral(&pair.input, dihedral)?, perm)?,
        output: apply_color_perm(&apply_dihedral(&pair.output, dihedral)?, perm)?,
    })
}

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

pub fn tokenize_grid(tok: &dyn GridTokenizer, grid: &Grid) -> Result<Vec<u32>> {
    tok.encode_grid(&grid.flatten()).map_err(Error::Message)
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
    if v < 0 || v >= N_COLORS as i32 {
        Err(Error::UnknownFamily)
    } else {
        Ok(v as u8)
    }
}

pub fn validate_spec(spec: &FamilySpec) -> Result<()> {
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
            if d < 0 || d >= N_DIHEDRAL as i32 {
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

pub fn apply_family(grid: &Grid, spec: &FamilySpec) -> Result<Grid> {
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
                out[wrap(r as i64 + dy as i64, h)][wrap(c as i64 + dx as i64, w)] = val;
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

fn gravity(grid: &Grid, direction: i32, bg: u8) -> Result<Grid> {
    let h = grid.rows();
    let w = grid.cols();
    let mut out = vec![vec![bg; w]; h];
    match direction {
        0 => {
            for c in 0..w {
                let objs: Vec<u8> = (0..h)
                    .filter_map(|r| {
                        let v = grid.cells[r][c];
                        if v != bg {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect();
                let start = h - objs.len();
                for (i, val) in objs.into_iter().enumerate() {
                    out[start + i][c] = val;
                }
            }
        }
        1 => {
            for c in 0..w {
                let objs: Vec<u8> = (0..h)
                    .filter_map(|r| {
                        let v = grid.cells[r][c];
                        if v != bg {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect();
                for (i, val) in objs.into_iter().enumerate() {
                    out[i][c] = val;
                }
            }
        }
        2 => {
            for r in 0..h {
                let objs: Vec<u8> = grid.cells[r].iter().copied().filter(|&v| v != bg).collect();
                for (i, val) in objs.into_iter().enumerate() {
                    out[r][i] = val;
                }
            }
        }
        _ => {
            for r in 0..h {
                let objs: Vec<u8> = grid.cells[r].iter().copied().filter(|&v| v != bg).collect();
                let start = w - objs.len();
                for (i, val) in objs.into_iter().enumerate() {
                    out[r][start + i] = val;
                }
            }
        }
    }
    Grid::try_new(out)
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

pub fn family_from_spec(spec: FamilySpec) -> Result<RefFamily> {
    validate_spec(&spec)?;
    let id = spec.family_id();
    Ok(RefFamily { spec, id })
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

pub fn generate_task(spec: &FamilySpec, seed: u64, n_train: u32) -> Result<Task> {
    validate_spec(spec)?;
    if n_train == 0 {
        return Err(Error::EmptyTrain);
    }
    let mut rng = SplitMix64::new(seed);
    let need = n_train as usize + 1;
    let mut pairs = Vec::new();
    let mut seen: HashSet<(Vec<Vec<u8>>, Vec<Vec<u8>>)> = HashSet::new();
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

pub fn parameter_grid() -> Vec<FamilySpec> {
    let mut specs = Vec::new();
    for bg in 0..N_COLORS as i32 {
        for dx in -3..=3 {
            for dy in -3..=3 {
                specs.push(fs(
                    FamilyKind::Translate,
                    &[("dx", dx), ("dy", dy), ("bg", bg)],
                ));
            }
        }
    }
    for src in 0..N_COLORS as i32 {
        for dst in 0..N_COLORS as i32 {
            if src != dst {
                specs.push(fs(FamilyKind::Recolor, &[("src", src), ("dst", dst)]));
            }
        }
    }
    for bg in 0..N_COLORS as i32 {
        specs.push(fs(FamilyKind::Crop, &[("bg", bg)]));
    }
    for nx in 1..5 {
        for ny in 1..5 {
            specs.push(fs(FamilyKind::Tile, &[("nx", nx), ("ny", ny)]));
        }
    }
    for bg in 0..N_COLORS as i32 {
        for dir in 0..4 {
            specs.push(fs(FamilyKind::Gravity, &[("dir", dir), ("bg", bg)]));
        }
    }
    for d in 0..N_DIHEDRAL as i32 {
        specs.push(fs(FamilyKind::Mirror, &[("dihedral", d)]));
    }
    for factor in [2, 3] {
        specs.push(fs(FamilyKind::Scale, &[("factor", factor)]));
    }
    for color in 0..N_COLORS as i32 {
        for width in [1, 2] {
            specs.push(fs(
                FamilyKind::Border,
                &[("color", color), ("width", width)],
            ));
        }
    }
    specs
}

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
        let train_key: Vec<(Vec<Vec<u8>>, Vec<Vec<u8>>)> = task
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

pub fn sample_corpus(specs: &[FamilySpec], seed: u64, n_train: u32, n: u32) -> Result<Vec<Task>> {
    if n == 0 {
        return Err(Error::EmptyTasks);
    }
    if specs.is_empty() {
        return Err(Error::UnknownFamily);
    }
    if n_train == 0 {
        return Err(Error::EmptyTrain);
    }
    let mut out = Vec::new();
    for i in 0..n {
        let spec = &specs[i as usize % specs.len()];
        out.push(generate_task(spec, seed.wrapping_add(i as u64), n_train)?);
    }
    Ok(out)
}
