# Progress

State of the code against `spec/`, as of 2026-09-25 on `48c279c`. Anyone working
on the repo updates this file in the same commit as the change it describes; the
rules are in [AGENTS.md](AGENTS.md).

Short version: about 38k lines of production code against the spec's 320k-640k for
S1 alone, every CPU test green, nothing verified on an H200. Most modules are CPU
stand-ins that nothing else in the repo calls yet. The audit below also found real
bugs in the model, the training step and the RL loss that the tests don't catch,
because the reference oracles in `tests/reference/` copy production's design instead
of deriving from the spec or the papers.

## Tests

Run on `48c279c`, 2026-09-25:

| check | result |
|---|---|
| `python3 .github/ci/checks.py tree` | clean |
| `uv run ruff check .` | clean |
| `uv run pytest` (CI flags) | 2286 passed, 30 skipped, 0 failed. All 30 skips are GPU-only |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace -- -D warnings` | clean |
| `cargo test --workspace` | 2554 passed, 0 failed, 367 test binaries |
| GitHub Actions `cpu` on `48c279c` | success |

Green tests mean the code matches its oracle. They don't mean it matches the spec;
see Known defects.

## H200 stages (spec 16)

| stage | runner touches production code? | ever run | state |
|---|---|---|---|
| V0 NCCL, kvm, NVMe | n/a | jobs 734115 (1 GPU), 734123 (2 GPU, 283 GB/s) passed `check_v0`; the 2x4 job 735855 was cancelled while pending | not verified. `check_v0` doesn't require 2 nodes (`verify/ncshare/src/check.rs:36-57`) |
| V1 JAX vs torch parity | yes (`model`, `train`) | env probe only | not run |
| V2 parallel = single device | imports `parallel` but never calls `ring_attention`, `fsdp_reduce_scatter` or `spmd_circular_pipeline`; one device, no optimizer step | no | runner can't fail on a parallel bug |
| V3 FP8 vs BF16 | yes, but a 128-dim toy, not 0.1-0.5B | no | runner below spec scale |
| V4 fault | no, NumPy re-implementation | no | stand-in |
| V5 rung 0 | no. Fits the same `LAW_A=12, alpha=0.5` law it generates losses from (`prometheus/verify/v5_rung0.py:50,309-326`) | no | passes by construction |
| V6 latent | no, NumPy | no | stand-in |
| V7 RL | no, separate NumPy loss (`prometheus/verify/v7_rl.py:23-27`) | no | stand-in |
| V8 serving | no, NumPy | no | stand-in |
| V9 lab | no, NumPy | no | stand-in |
| V10 72h soak | no. Simulated clock, reports `hours: 72` after seconds (`v10_soak.py:385`) | no | stand-in |

Only `templates/v1.sh` asserts a GPU is present, and `check_exit` records no job id,
device or GPU count. As written, submitting v4-v10 would print PASS. Fix that before
any stage is submitted.

## Coverage by step

LOC is `wc -l` on production `.py`/`.rs` (Rust counts include inline test modules).
"Spec" is the low end of the 15.5 / 17 estimate.

| step | where | LOC / spec | state | biggest gaps |
|---|---|---|---|---|
| F1 contracts | `contracts/`, `crates/contracts` | 818 / 10k | schemas, goldens, validate, codegen | Rust codegen can't read any golden; generated types used nowhere; migration is identity |
| A1 model | `model/` | 2075 / 10k | layout, MLA, linear attn, MoE, MTP, recurrence, latent helpers | see defects; halt head in flight; Stage B not wired to `forward` |
| A2 train | `train/` | 985 / 8k | MuonClip, WSD, AdamW, clip, muP transfer | Muon covers 2.5% of params; trains at r=1 only; no loader or loop |
| A3 kernels | `kernels/` | 1482 / 25k | delta rule with custom_vjp, FP8 fake-quant, EP dispatch/combine, compile cache | no Triton/Pallas/CUDA; not used by `model/` or `train/` |
| A4 parallel | `parallel/` | 1090 / 10k | mesh, stage split, ZeRO-3 math, CP ring, `shard_map` pipeline | FSDP/CP/EP are host-side list ops; no 1F1B or backward through the pipeline; not used by `train/` |
| A5 ckpt | `ckpt/` | 972 / 15k | manifest, content-addressed shards, sync/async save, cross-rack pair | no Python/JAX bridge; not atomic; second replica never read |
| A6 control | `control/src/lib.rs` | 752 / 40k | elastic DP, heal, straggler, SDC, watchdog, spike bookkeeping | not connected to `ckpt` or `train`; rollback only echoes an id |
| A7 rung / I1 scale | `control/src/{rung,scale}.rs` | 550 / 5k | step counter, resume, power-law fit | no actual rung-0 run; fit rejects rung 0 so rungs 0+1 can't produce one |
| B1-B6 data | `data/` | 2209 / 55k | extract, MinHash dedup, 8-gram decontam, loader, mixture | classifiers score by hash; embedding decontam is an error stub; loader has no Parquet, PyO3 or `document_ids` |
| F6 tokenizer | `tokenizer/` | 1197 / 3k | byte-fallback BPE, ARC ids, freeze | quadratic trainer won't reach a 256k vocab; gate compares against raw bytes |
| E1-E4 synth | `synth/` | 3512 / 27k | prompts, rejection sampler, ARC generators, tool loop | fact-check is word overlap; not wired to real F5/D2/D1 |
| C1-C2, I4 sglang-fork | `sglang-fork/` | 903 / 18k | export, layer tables, validation helpers | no SGLang model or forward; bf16 copies fp32; FP8 is int8 |
| F5 serve | `serve/` | 363 / 10k | router shape | `LocalEngine` returns hashed strings; no HTTP client |
| D1 envs | `rl/envs` | 1315 / 40k | pool, snapshot, fork API | "Firecracker" backend is the in-memory engine; shell and Python are string matchers |
| D2 verifiers | `verifiers/` | 1143 / 15k | rational math, grids, market score, test runner | Lean accepts two hardcoded theorems; no mutant generation |
| D3/I6 tasks | `rl/tasks` | 1519 / 80k | 8 factories, solve-rate filter | no mining of real sources |
| I2 loss + coordinator | `rl/loss`, `rl/coordinator` | 1087 / 24k | ratio, clip-higher, dynamic sampling, TIS, overlong, latent ratio | loss math wrong (see defects); plain Python floats, not a JAX loss; not called by `train/` |
| I3 rewards | `rl/rewards` | 554 / 15k | rubric panel, RLCD filters | tampering detector returns caller-set booleans; no efficiency reward |
| I7 multiagent | `rl/multiagent` | 627 / 10k | topologies, roster, credit, distill-back | nothing JAX-side |
| F2/I10 obs | `obs/` | 674 / 20k | metrics, tracer, run registry | in memory only; spike baseline absorbs the spike |
| F3 evals | `evals/` | 374 / 15k | suite list, envelope, n-gram index | `Runner.load_items` fabricates items; no real suite |
| I8 arc | `arc/` | 293 / 10k | D4 augment, 9-op DSL, vote | no TTT, no ARC-AGI-1 eval |
| I9 audit | `audit/` | 251 / 5k | discrete vs latent compare | planted-bug finders fire on healthy input |
| H2-H12 harness | `harness/*` | 6724 / 68k | log, leases, raft, mesh, providers, slurm, cas, ledger, pipeline, chaos, ops | no real network, provider, crypto or process kills; see defects |
| L1-L4 lab v0 | `harness/{kernel,eval-gate,monitors,genome-seed}` | 1986 / 65k | capability API, scores store, freeze file, seed roles | no sandbox; kill switch read by nothing; roles don't match 14.3 |
| F4 verify | `verify/ncshare`, `prometheus/verify` | 5049 / 5k | templates, checkers, runners | see H200 stages |

H1 (wizard-core split) is skipped by decision. S3 (spec 15.6) is out of scope.

## Known defects

Found in the 2026-09-25 audit. Each was confirmed by running a probe or reading the
line; none has a failing test yet. Fix the oracle along with the code: most of
these pass today because `tests/reference/` has the same bug.

### model / train
- `train/__init__.py:267` `classify_param` puts only 2-D weights on Muon. Attention
  projections and experts are 3-D, so on the tiny config 195,584 params get Muon and
  7,506,432 get AdamW. No per-head or per-expert Newton-Schulz.
- `train/__init__.py:902` `train_step` calls `forward(..., r=1)`. Sampled depth, the
  truncated-recurrence cut and every core iteration past the first get no gradient.
- `model/__init__.py:486` `_expert_to_rack` splits experts into exactly `max_racks`
  racks, so node-limited routing never binds and equals plain top-k. The test at
  `tests/test_model.py:522` uses the same map.
- No `router_bias` param; aux-loss-free balancing does nothing (`train/__init__.py:810`).
- `train/__init__.py:909,785` z-loss is logsumexp of last-layer sigmoid probs. The
  correct logit-based value is already in `out.z_loss` and is ignored.
- `train/__init__.py:772` MTP head 0 predicts t+1, same as the main head.
- `train/__init__.py:794` QK-clip needs a `W_k` key MLA doesn't have, so it only
  clips the linear-attention layers, and scales by weight rows, not observed logits.
- `model/__init__.py:749,469` MLA values are the keys; no value projection.
- `model/__init__.py:393-432` Gated DeltaNet is a plain delta rule, beta=1, no gate.
- `model/__init__.py:368` 2D RoPE exists but `forward` uses `arange` positions.
- `model/__init__.py:688-689` decode head is not tied to the embedding.
- `train/__init__.py:461` `transferred_muon_lr` returns its input; init is fixed
  0.02 at every width. With `muon_transfer=True` the step reuses `muon_lr=2e-2`.
- `train/__init__.py:691` `apply_precision` is an FP32 copy and `train_step` never
  calls it.
- `model/__init__.py:578` MoE gathers a (tokens, k, d, hidden) tensor; at rung-0
  size that is hundreds of GB.
- Stale "Not implemented." docstrings at `model/latent.py:767,822,913,1006,1056`.

### RL loss (spec 9.1, 9.2)
- `rl/loss/__init__.py:251,261` `sequence_ratio` is `exp(sum(dlogp))`. GSPO is the
  length-normalized `exp(mean(dlogp))`; over 1k tokens at 0.01 each this is e^10 and
  everything sits at the clip bound.
- `rl/loss/__init__.py:500,510` the ratio and the TIS weight both use rollout
  log-probs. Spec 9.2.1: the ratio is against the trainer's recomputed old-policy
  log-probs; rollout log-probs are only for TIS.
- `rl/loss/__init__.py:516` no pessimistic `min(r*A, clip(r)*A)`.
- `rl/loss/__init__.py:527` total is `pg + latent + overlong` with mixed signs; the
  overlong term has no gradient path. TIS at k=0 returns 1.0 (`:325`).
- `tests/reference/rl_loss.py:12-36` restates the same formulas.

### verification
- V2, V4-V10 runners as described in H200 stages above.
- `verify/ncshare/src/check.rs:36-57` `check_v0` accepts busbw 0 and 1 node.
- `harness/slurm/src/lib.rs:21` `ALLOWED_GPUS=[1,2,4]` and no nodes/ntasks, so V0
  2-node, V2 and V5 can't be submitted through H7.

### harness
- `harness/mesh/src/lib.rs:350-353` freeze signature is `signature == payload`.
- `harness/ops/src/lib.rs:172-184` quorum counts any non-empty bytes as a signature.
- `harness/chaos/src/lib.rs:963-971` the D0 "kill -9s" set an in-memory handle to
  `None` and reopen it. No process is killed.
- `harness/raft/src/inner.rs:408` persist has no fsync.
- `harness/providers/src/lib.rs:315` token refresh is sha256 of the refresh blob.
- `harness/ledger/src/lib.rs:165` `query` splices raw SQL; `:298` embedding is a
  hashed bag of words.
- `harness/pipeline/src/lib.rs:16,119` the planted-bad-patch gate is a substring match.
- `harness/monitors/src/lib.rs:140` `plant()` calls the `trip()` it tests; nothing
  reads the freeze file.

### data, synth, tokenizer
- `data/classifiers/src/lib.rs:359-377` both scorers are FNV hashes.
- `data/decontam/src/lib.rs:163-168` `embedding_match` always errors; items under 8
  words never match (`:133,148`).
- `data/dedup/src/lib.rs:290` one shared paragraph drops the whole document.
- `data/loader/src/lib.rs:120-124` `Batch` has no `document_ids`; padding gets doc id
  0 (`:342`); resume never checks `data_mix_hash`.
- `tokenizer/src/lib.rs:618` `load` never compares the stored `vocab_hash`.
- `synth/src/lib.rs:361-407,428` fact-check is word overlap ("Bob defeated Alice"
  passes against "Alice defeated Bob") and `NotInSource` claims are accepted.

### ckpt, control, contracts, evals, audit, obs, rl
- `ckpt/src/lib.rs:127` dropping a `PendingSave` without `wait` wedges every later
  async save with `SaveInFlight`.
- `ckpt/src/lib.rs:713` restore reads replica 0 only; `:350` `DirStore::put` is a
  plain write, and a `../` checkpoint id escapes the store root.
- `kernels/compile_cache.py:128` two hosts compiling different bytes on the same
  miss raises `CacheError`; key omits jaxlib version and device.
- `control/src/scale.rs:117` fit rejects rung 0.
- `contracts/_generate.py:302-303` Rust codegen types every const/enum as `String`,
  so `schema_version: 1` fails to parse in all goldens.
- `evals/__init__.py:280-333` `load_items` makes up items; `:349` checks eval items
  against the eval index.
- `audit/__init__.py:227` `DECODE_FLIP` fires on healthy logits; `:215`
  `ANSWER_SWAP` fires whenever answers differ.
- `obs/src/lib.rs:501` spiked losses go into the baseline.
- `rl/envs/src/lib.rs:331-395` Firecracker backend checks `/dev/kvm` and runs the
  in-memory engine; fork time is hardcoded 0 (`engine.rs:149`).
- `rl/rewards/src/lib.rs:123-125,470` tampering detector returns caller-set flags;
  `:344` rubric disagreement lowers reward instead of sample weight.

## What remains, in order

1. **Close the false-pass paths.** GPU assert and provenance (job id, `nvidia-smi`,
   GPU count) in every template, checked by `check_exit`. `check_v0` requires 2x4
   and busbw > 0. V4-V10 runners can't write `vN.json` until they drive production
   code. H7 gets nodes, ntasks and 8-GPU requests.
2. **Fix the model and train defects**, oracles first from the spec: Muon on 3-D
   weights, train at sampled r with the truncation window, z-loss, MTP offsets,
   QK-clip on MLA logits, router bias and working node-limited routing, MLA value
   projection, DeltaNet gates.
3. **Fix the RL loss** against the GSPO and DAPO papers, as a JAX loss `train/` calls.
   Rebuild V7 on it.
4. **V0 2x4, then V1** on NCShare.
5. **Wire track A together.** `train/` uses `kernels` and `parallel`; a real sharded
   step V2 can compare to one device; a pytree bridge to `ckpt`; a loader and train
   loop that can run rung 0 for V5.
6. **Data path for rung 0.** Loader with Parquet and `document_ids`, a tokenizer
   trainer that scales and is frozen, real decontam.
7. **Harness crypto and transports.** ed25519 for mesh freeze and ops quorum, real
   process kills for D0, openraft over the network, fsync.
8. Everything else in the coverage table, in 15.5 dependency order.

## In flight

- A1 inference halt head (spec 4.1). Interface `012848a`, oracle `92b0def` (9 tests
  fail on the stubs). Three implementers on `a1-halt-impl-{1,2,3}` have uncommitted
  work from 2026-09-24. No loss trains `halt_w`/`halt_b` yet, so it will be inert
  after merge until one does.
