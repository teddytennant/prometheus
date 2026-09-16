## 7. Data

### 7.1 Pretraining mix (~180T tokens seen)

| Source | Unique tokens | Epochs | Notes |
|---|---|---|---|
| Web, filtered | ~20T | 2 to 3 | classifier-based quality filtering, heavy dedup |
| Code | ~4T | 4 | licensed/permissive, repo-level packing (files in dependency order) |
| Math, science, arXiv | ~1.5T | 4 | LaTeX-preserving extraction |
| Books, papers | licensing-gated | 2 to 4 | only what you have rights to |
| Synthetic rewrites | ~40T | 1 | high-quality docs rephrased in several styles (Kimi K2-style), fact-checked against source |
| Synthetic reasoning | ~10T | 1 | math/code problems with verified solutions |
| Procedural ARC-like grids | ~2T | 1 | re-arc-style generators, millions of task families, 2D tokenization |
| Agentic trajectories | ~1T | 1 | tool use, terminal sessions, verified by outcome |

Mix ratios come from rung 2 mixture ablations and are re-weighted in the decay phase
toward code, math and reasoning.

### 7.2 Pipeline (Rust)

- Extraction: HTML/PDF/LaTeX → text, Rust workers on CPU nodes (not GB200 racks).
- Dedup: exact (hash) + near-duplicate (MinHash LSH) across the whole corpus,
  document and paragraph level.
- Quality and safety classifiers: small models in JAX run in batch on a slice of the
  fleet; Rust owns the sharding and the bookkeeping.
- Decontamination: n-gram and embedding match against every eval in section 11,
  including ARC eval tasks, before anything enters the mix. Fail closed.
- Storage: Arrow/Parquet shards, deterministic global shuffle with a seed per epoch,
  so any step's batch can be reconstructed exactly for spike diagnosis.
- Loader: Rust (PyO3 extension) streams pre-tokenized packed sequences into pinned
  host buffers; the JAX step never waits on data. Loader state is part of the
  checkpoint.
