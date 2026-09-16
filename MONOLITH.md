# Frontier model training stack: spec

Status: draft v0, 2026-09-16. An open project: the spec, the code and the results,
negative ones included, are public. Every number here is a starting point. The scaling
ladder (section 6) gets to overrule any of them.

Target: a model that is at or past the frontier on AI research, programming,
long-horizon agentic work, math and ARC-AGI, and that reasons in continuous latent
space instead of long visible CoT, so that more latent steps on a problem buy a
better answer. That has never been done at scale. It's very likely to work, and if
it doesn't, the loss is ~5% of the compute and the CoT model it started from
(section 4). Trained from scratch
on 220k GB200s with open frameworks: JAX for the model and training math, Rust for
most of the systems around it, SGLang (modified) for rollouts and serving.

---

## 0. The short answer: how many lines

"The training script" means very different things at different radii. nanoGPT's
`model.py` + `train.py` come to about 650 lines, and they really do train a GPT. The
lines that separate that from a frontier run are almost all fault tolerance,
parallelism, data, environments and evals, not model math.

| Radius | What it covers | First-party LOC (excluding tests) |
|---|---|---|
| Core | model, train step, optimizer, sharding, loop | 25k to 45k |
| Training system | core + kernels, pipeline/EP comms, checkpointing, elastic restart, data loader | 150k to 250k |
| Full stack | training system + data pipeline, synthetic data, RL system, environment fleet, SGLang changes, evals, observability, harness | 620k to 1.3M |
| With tests | full stack + tests (ratio 0.5x to 1x) | 0.9M to 2.6M |

Underneath that sits tens of millions of lines you don't write: XLA, JAX, CUDA,
cuDNN, NCCL, SGLang, PyTorch, the kernel, Firecracker.

The largest single line item is not the model. It's environments and verifiers for RL
(section 9.4). They are the long tail, and they decide how good the model gets at
research, code and long-horizon work. Budget per component is in section 17.

Lines of code are not the bottleneck. People-years and fleet reliability are. A
conventional team for this is something like 150 to 300 strong engineers and
researchers over about a year. That's my estimate, not a measurement. This plan puts
an agent swarm in that seat instead (section 15), which is the only reason the
timeline in section 18 is months rather than a year.

---

## 1. Hardware facts the design depends on

- **220k GB200 GPUs ≈ 3,056 NVL72 racks.** One rack: 72 Blackwell GPUs, 36 Grace
  CPUs, ~13.4 TB HBM total (~186 GB per GPU), one NVLink 5 domain at 1.8 TB/s per GPU.
- **Grace memory is close.** Each Grace has 480 GB LPDDR5X on NVLink-C2C (900 GB/s)
  to its two GPUs. That's ~240 GB of "slow HBM" per GPU. The design uses it for
  optimizer state offload in training and as the second KV tier in serving.
- **Scale-out** is 800G per GPU (ConnectX-8, InfiniBand or Spectrum-X). Cross-rack
  bandwidth is ~20x worse than in-rack. Rule: all-to-all stays inside a rack; only
  gradient reduction and pipeline sends cross racks.
- **Blackwell supports FP8 and NVFP4** in hardware. FP4 cuts serving memory ~4x vs
  BF16, which matters because it lets a multi-trillion-parameter MoE fit on one rack.
- **Power**: order of 120 to 140 kW per rack, so roughly 400 MW of IT load. If that's
  split across buildings, section 5.6 applies.
- **Failures are the normal state.** Llama 3 reported 419 unexpected interruptions in
  54 days on 16k H100s, about one every 3 hours. Linear scaling to 220k gives one
  every ~13 minutes, and early GB200 fleets have been less reliable than mature H100
  fleets. Anything that needs a full-job restart per failure will not train.

Compute budget. Assume ~1e15 effective FLOP/s per GPU after MFU (measure this on day
one and re-plan). The fleet then produces ~2.2e20 FLOP/s, or ~1.9e27 FLOP per 100
days, about 100x GPT-4-era pretraining.

---

## 2. Compute and data plan

Data, not compute, binds pretraining at this size. Compute-optimal pretraining at
~1e27 FLOP wants several hundred trillion unique quality tokens, and those don't
exist. Four epochs over filtered data costs little (Muennighoff et al.), and
synthetic rewrites extend that further, but past that the extra compute buys more by
going into RL, synthetic data generation and a larger active parameter count.

The fleet runs these concurrently, not one after another:

| Workload | GPUs | Duration | ~FLOP |
|---|---|---|---|
| Scaling ladder and ablations | 10k to 20k | continuous | 2e26 |
| Main pretraining | ~100k | ~80 days | 7e26 |
| Synthetic data generation (SGLang) | 20k to 40k | continuous before/during pretrain | 1.5e26 |
| Mid-training (long context, reasoning and agentic data) | ~100k | ~15 days | 1.3e26 |
| Latent branch (Stage B + latent RL probe, section 4) | 20k to 40k | ~30 days | 1e26 |
| RL (rollouts + trainer) | 80k to 150k | ~120 days, then continuous | 8e26 |
| Evals, red-teaming, spares | 5% to 8% | continuous | n/a |

Hot spares: keep 3% to 5% of racks powered and burned in, not idle-assigned.

---

## 3. Model architecture

### 3.1 Starting config (the ladder refits it)

| Item | Value | Why |
|---|---|---|
| Total params | ~7.4T | fits in NVFP4 on one NVL72 (~4.1 TB with scales) with KV room |
| Active params | ~0.4T unique, ~0.45T compute-active (recurrence) | inference cost vs capability |
| Layers | 96 unique; 3 dense, 93 MoE | |
| Depth recurrence | 8-layer core block iterated r times (train mean r=3) | latent compute, section 4 |
| d_model | 12,288 | |
| Attention | 3:1 hybrid: 72 linear-attention layers (Gated DeltaNet / KDA family), 24 MLA full-attention layers | constant-size state on 3/4 of layers; long context at low KV cost |
| MoE | 512 routed experts, top-20, 2 shared, expert hidden 4,096 | fine-grained experts + shared, DeepSeek-V3 style |
| Routing | sigmoid gates, aux-loss-free bias balancing, node-limited routing (max 4 racks) | balance without an aux loss hurting quality |
| MTP | 2 extra-token heads | denser training signal; free speculative decoding at serve |
| Vocab | 256k byte-level BPE; one token per ARC grid cell color | fewer tokens per word (token efficiency) |
| Positions | RoPE on MLA layers (partial dim); 2D RoPE on grid spans | ARC grids are 2D |
| Norm/stability | pre-norm RMSNorm, QK-norm on MLA, logit soft-capping, z-loss on router | loss spikes at this scale are expensive |
| Context | 16k pretrain → 256k → 1M mid-training | long-horizon agents |

Check on the parameter count: one expert is 3 × 12,288 × 4,096 ≈ 1.5e8 params; 512 of
them over 93 MoE layers comes to ~7.2T, plus attention, dense layers and embeddings
≈ 7.4T. Active: 22 experts × 1.5e8 + ~6e8 attention ≈ 3.9e9 per layer, ~0.37T over
96 layers, ~0.45T once the core block runs 3 times.

### 3.2 Things deliberately left out

- **Pure dense.** At a fixed serving budget, dense trades too much capability.
- **Byte-level / BLT tokenization at the flagship scale.** It's promising, but the
  serving stack and the latent-reasoning work are enough new risk for one run. Rerun
  the question at ladder rung 3.
- **Full-attention-only.** A 1M-token KV cache on 96 full layers would dominate
  serving memory even with MLA.

---

## 4. Continuous latent reasoning

This is the point of the model. **More latent steps give a better answer.**

A chain of thought throws information away at every step. The model computes a
hidden state of 12,288 dimensions, samples one token from it, and the next step sees
only that token, at most about 18 bits from a 256k vocab. A continuous thought
feeds the whole hidden state back in as the next input. Nothing gets collapsed to one
choice, so a thought can carry several candidate next steps at once, and each extra
step adds real computation instead of one more token to read back. Scaling test-time
compute in latent space should work at least as well as scaling CoT tokens, and
without KV or output cost for every step.

The evidence so far:

- **Coconut** (Hao et al. 2024, arXiv:2412.06769) feeds the last hidden state back as
  the next input embedding, trained from a CoT model through a curriculum that
  swaps language steps for continuous thoughts. On ProsQA, a planning task, it gets
  97.0% against CoT's 77.5% with far fewer tokens, and probing shows a continuous
  thought holding several frontier nodes at once, a breadth-first search that
  discrete CoT can't do because it has to commit to one path per token. Adding more
  continuous thoughts per step raised GSM8k accuracy.
- **Reverie** (github.com/teddytennant/reverie) trains every continuous thought to
  decode, through the tied output head, to its gold reasoning step, and lets a
  PonderNet halt trained against the teacher's step count decide how many thoughts
  each problem gets. One stage, no curriculum, no RL. The halt comes out exact: 2-hop
  problems get 2.0 thoughts, 3-hop get 3.0, 4-hop get 4.0, ρ = +1.00 across 606
  seeds. Extra latent passes pay off where the backbone can't solve the problem in
  one pass: with a 1-layer backbone, Reverie beats No-CoT at every width past 64,
  reaching 44 sigma at d=192. With 2 layers the task is easy enough to solve in one
  pass and there is no edge. So the curve has to be measured on problems that are
  hard for the model, not on ones it already one-shots.
- **Huginn** (Geiping et al. 2025) scales test-time compute by iterating a recurrent
  block, and reasoning accuracy rises with iterations.

None of this has been done at frontier scale. That's the part that's new. It's very
likely to work anyway, for three reasons: the model starts as a strong CoT reasoner,
so latent training only has to compress steps it already knows how to take; a
continuous thought carries strictly more than the token sampled from it; and every
small-scale result above points the same way.

**If it fails, it's cheap.** Pretraining, mid-training and the discrete RL run
produce the CoT model whether or not latents work (Stage A, 4.3). The latent-only
spend is a branch off that checkpoint: Stage B plus a latent RL probe, about 1e26
FLOP, roughly 5% of the run in section 2. If the branch misses 4.5, the CoT model is
the model and that 5% is the whole loss.

**If it works, what you give up is readable reasoning.** Work on that already
exists: Reverie's decode term makes every thought linearly decodable to its step,
and Coconut's probes read out what a thought is holding. The decode term goes into
Stage B (4.3), so thoughts stay decodable. Safety monitoring of the reasoning isn't a
goal of this model.

### 4.1 Two kinds of latent compute

1. **Vertical (depth recurrence).** Prelude layers → core block iterated r times with
   input injection → coda layers. r is sampled during training (heavy-tailed, mean 3,
   max 16) and chosen by a halting head at inference, under a per-request budget.
   Training on a spread of r is what lets the model use more of it at test time:
   easy tokens halt at 1 or 2, hard ones run to the cap. It adds thinking per token
   with no extra tokens and no extra KV positions. Huginn-style KV sharing across
   iterations keeps the cache size flat.
2. **Horizontal (continuous thoughts).** In `<think>` mode the final hidden state goes
   through a small latent adapter (2-layer MLP + norm) and is fed back as the next
   input embedding, as in Coconut. No token gets sampled, so it costs no output
   tokens. The number of thoughts is set by a Reverie-style halt (4.3) and is the
   second axis of the scaling curve.

### 4.2 Latent chunks with discrete anchors

Unbounded latent sequences are hard to train (sequential, can't teacher-force) and
hard to do RL on (no discrete action to take a log-prob of). So:

```
<think> [L L L L L L L L] anchor-tokens [L L L L L L L L] anchor-tokens ... </think> answer
```

- Latent chunks of 4 to 64 thoughts, lengths set by the halt.
- Between chunks the model emits a short discrete anchor (a few tokens: a subgoal,
  intermediate result or tool call). Anchors give RL something to score and stop
  latent error from compounding forever.
- Tool calls are always discrete.

### 4.3 How latents are trained

Start from a strong discrete reasoner. Coconut and Reverie both distill latents from
a CoT teacher, and nothing suggests latent reasoning comes out of pretraining alone.

1. **Stage A: discrete CoT reasoner.** Standard SFT + RL (section 9) with visible CoT.
   This is the baseline in 4.5 and the model if the latent branch misses.
2. **Stage B: compress CoT into thoughts.** Replace the first k CoT segments with
   continuous thoughts, raising k over training (Coconut), with the loss from Reverie
   extended to segments:
   - answer CE, weighted over the halt distribution (PonderNet)
   - α · CE of each thought decoding to its teacher step, through the tied head for
     one-token steps and a small decoder for longer segments
   - γ · halt at the teacher's step count
   - β · KL of the halt distribution against a geometric prior, so it can't collapse

   plus hidden-state alignment against the Stage A teacher at segment boundaries
   (CODI). Compression target ~8 tokens per thought, raised as long as quality holds.
   Reverie found α and γ trade accuracy against calibration at small widths, so their
   weights are swept at rungs 0 and 1, not assumed.
3. **Parallel training.** Naive latent training takes n+1 forward passes for n
   thoughts. Use Jacobi-style parallel fixed-point iteration over each chunk
   (PCCoT-style): initialize all thoughts, update them in parallel for a few sweeps.
   Truncated backprop through the last 2 sweeps and last 4 recurrence iterations, as
   Huginn does.

### 4.4 RL through latents

GRPO-family losses need log-probs of actions, and deterministic latents have none.
Use **noisy latent policy**: latent vector = μ_θ(context) + σ_θ ⊙ ε, with a learned
per-dimension σ kept small (clamped). Log-prob of the latent trajectory is the
Gaussian log-density. The importance ratio is split into a discrete-token part and a
latent part, each clipped separately, because their scales differ by orders of
magnitude. The noise also gives the rollout group diversity, which GRPO depends on.

Fallback if this is unstable at rung 3: deterministic latents, policy gradient only on
anchor and answer tokens, with gradient flowing into latents by truncated backprop.

### 4.5 The scaling test

At every rung from 0.5B (V6) up, on held-out math, code and ARC problems the model
can't one-shot, with latent steps counted as recurrence iterations times thoughts:

1. **More steps, better answers.** Accuracy rises as the latent budget doubles from
   1x to 16x the training mean, and doesn't collapse at the top end.
2. **Better than CoT at the same compute.** At equal FLOP per problem, the latent
   curve sits above the discrete CoT curve, and the gap doesn't shrink from one rung
   to the next.

Both get published whichever way they come out. Thought decode accuracy is reported
next to each point. If 1 holds and 2 doesn't, depth recurrence plus short discrete
CoT ships and horizontal thoughts stay research. If 1 fails, the Stage A model is the
model, and the branch cost is what was lost.

---

## 5. Training system (JAX)

### 5.1 Why JAX on GPUs, and what it's missing

JAX + Rust at frontier scale has precedent (Grok-1 was trained on that combination),
and XLA on Blackwell works. But JAX's own heavy use is on TPUs, and at this GPU count
several pieces are ours to build:

| Gap | Plan |
|---|---|
| Pipeline parallelism | SPMD circular pipeline (Praxis-style: `scan` over stages inside `shard_map`, `ppermute` between stages), rewritten for FLOP-balanced stages |
| MoE all-to-all | custom XLA op wrapping an NCCL/NVSHMEM EP dispatch/combine kernel (DeepEP-style, re-bound for JAX) |
| NVFP4 / FP8 linears | Transformer Engine JAX bindings where they work; custom cuDNN/Triton kernels behind `jax.custom_vjp` where they don't |
| Linear attention kernels | Triton kernels (chunked delta rule) via custom primitive, forward + backward |
| ~55k controller processes | `jax.distributed` coordination was not built for this; run it per replica group with a Rust control plane on top (5.5) |
| Compile time | cache compiled XLA programs by content hash on shared storage; target <10 min cold start at full scale |

### 5.2 Parallelism layout (main pretraining, ~100k GPUs)

- **EP = 72** (one rack): ~7 experts per GPU per layer, all-to-all over NVLink only.
- **FSDP over the rack** for attention, dense and shared-expert params (ZeRO-3 over
  NVLink, cheap).
- **PP = 12**, one rack per stage, stages balanced by measured FLOPs not layer count.
  The recurrent core block gets extra stage capacity because it runs r times.
- **DP ≈ 115 replicas** (864 GPUs each) over InfiniBand.
- **Context parallel** (ring over NVLink) only in the 256k/1M mid-training phases.
- **Optimizer state on Grace.** FP32 master weights and Muon momentum live in LPDDR5X
  over NVLink-C2C; the GPU holds BF16/FP8 weights, grads and activations. Per GPU:
  ~8.5e9 expert params × (BF16 weights + BF16 grads) ≈ 34 GB on GPU and ~68 GB on
  Grace. That leaves ~150 GB of HBM for activations with selective remat.

Batch: 64M to 80M tokens per step after warmup (Kimi K2 used ~67M), roughly 3 s/step
target, ~2.2M steps over 80 days. Check this on the ladder; critical batch size under
Muon isn't settled.

### 5.3 Precision

- Default: FP8 for linears (per-block scaling), BF16 for router, norms, embeddings,
  softmax, final 2 layers, latent adapter.
- NVFP4 forward for routed expert weights once rung 3 shows no loss gap. Public
  results show NVFP4 pretraining working at 12B scale on 10T tokens; nobody has shown
  it at 7T total, so this is gated.
- Master weights FP32 on Grace. Gradient reduce in BF16.

### 5.4 Optimizer

**MuonClip** (Muon + QK-clip, as in Kimi K2, which trained 1T params over 15.5T tokens
without loss spikes) for all 2D matrices; AdamW for embeddings, norms, router biases,
latent σ. Per-expert Newton-Schulz runs locally on each GPU since expert matrices are
unsharded within EP; FSDP-sharded matrices gather in-rack for NS. Hyperparameter
transfer by μP-for-Muon across the ladder. Muon keeps one state per param instead of
Adam's two, which is what makes Grace offload fit.

Schedule: WSD (warmup, stable, decay), so a checkpoint from the stable phase can
branch into mid-training without restarting.

### 5.5 Fault tolerance (Rust control plane)

Target: a GPU failure costs under 5 minutes of fleet time and never a full restart.

- **Elastic DP.** A failed replica drops out; the other 114 keep going, with grad
  accumulation raised to hold tokens per step fixed. A spare rack is healed in and
  rejoins at the next step boundary, with weights copied from a live replica over IB.
- **In-memory checkpoints** every ~10 minutes to Grace RAM on two other racks
  (cross-rack redundancy); persistent async checkpoints every ~2 hours to the object
  store. Checkpoint size is ~90 TB (weights + FP32 master + momentum); only one DP
  replica's shards get written.
- **Silent data corruption detector.** DP replicas must hold bit-identical weights.
  Hash each shard every N steps and compare across replicas; a mismatch quarantines
  the rack. It costs almost nothing and catches the failure you otherwise find three
  weeks later as a loss curve that won't come down.
- **Straggler detection** from per-rank step timing; slow ranks are drained the same
  way as dead ones.
- **Health**: Xid/ECC events, NVLink error counters, NIC flaps, thermal throttling,
  fed into the scheduler before failures turn into hangs. Every collective gets a
  watchdog timeout.
- **Loss-spike policy**: automatic rollback to the last in-memory checkpoint, skip the
  offending data shard, log it. A human is paged on the second spike within 10k steps.

### 5.6 Multi-site

If the 220k GPUs sit in more than one building with thin links between them, don't
run synchronous DP across sites. Use DiLoCo-style local steps within a site and
infrequent cross-site parameter averaging, and validate the loss penalty on the ladder
first.

---

## 6. Scaling ladder

No flagship run until each rung passes. Rungs keep the flagship's architecture shape
(same expert granularity, hybrid ratio, recurrence, latent adapter).

| Rung | Active / total | Tokens | GPUs | Decides |
|---|---|---|---|---|
| 0 | 0.1B / 1B | 20B | 64 | kernel correctness, bitwise parity vs reference PyTorch impl |
| 1 | 1B / 15B | 200B | 1k | μP transfer, Muon settings, hybrid ratio, MoE granularity |
| 2 | 8B / 120B | 1.5T | 5k | latent scaling test at 8B (4.5), FP4 gate 1, data mixture |
| 3 | 40B / 700B | 6T | 15k | latent scaling test at 40B, RL-through-latents stability, scaling-law fit |
| Burn-in | 8B shape | 1 week | full 100k | hardware shakedown, failure rates, elastic restart at scale |
| Flagship | 0.4T / 7.4T | ~180T | ~100k | |

What each rung costs, at 6ND FLOP and ~4e14 effective FLOP/s per GPU:

| Rung | GPU-hours | Where |
|---|---|---|
| 0 | ~8 | the 8 H200s we have, in an afternoon |
| 1 | ~830 | same 8 H200s, ~4.5 days with checkpoint and resume across walltimes |
| 2 | ~5e4 | not reachable on NCShare. About $100k of rented H100 time at $2/GPU-hour |
| 3 | ~1e6 | about $2M rented |
| Flagship | ~3.4e8 | the cluster, or nothing |

So the ladder is where compute, not code, becomes the binding constraint. Rungs 0 and
1 are free and are what the pre-cluster program (16.5) is built to deliver.

Scaling laws fit at rungs 1 to 3 predict flagship loss and set its LR, batch and
token count. A flagship that comes in 5% or more above prediction at 10% of training
gets stopped and diagnosed, not pushed through.

---

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

---

## 8. Mid-training

Starts from a late stable-phase checkpoint.

1. **Long context**: 16k → 256k → 1M, raising the RoPE base on MLA layers, with
   long-document and repo-level data and synthetic retrieval/multi-hop tasks.
2. **Reasoning and agentic data** upweighted: verified solutions, tool trajectories,
   research transcripts (papers + code + results).
3. **Recurrence depth widened**: the r distribution (4.1) shifts toward the tail on
   math and code, so the halt learns to spend more on hard tokens. Stage B waits for
   the Stage A reasoner (4.3).
4. **Context management skills**: notes-file read/write tools, self-summarization of
   old context, retrieval over its own history. RL makes these good later; mid-training
   teaches the format.

Then a short SFT cold start on high-quality reasoning and agent traces, in
discrete format, before RL.

---

## 9. Reinforcement learning

### 9.1 Loss

GRPO as published has known problems (length bias from per-sequence normalization,
entropy collapse, instability on MoE). Use a combination whose parts each have public
evidence:

- **Sequence-level importance ratio** (GSPO). Token-level ratios are noisy on MoE, where
  routing changes between rollout and training.
- **Asymmetric clipping** (DAPO clip-higher) to keep entropy up.
- **Dynamic sampling**: drop groups where every sample got the same reward (zero
  advantage, pure waste).
- **No per-sequence length normalization, no std normalization** (Dr. GRPO): sum over
  tokens / constant.
- **Overlong soft penalty** plus the explicit efficiency reward in 9.3.
- No reference-model KL by default; add it back per-domain only if drift appears.
- Latent part of the ratio as in section 4.4.

### 9.2 System

- **Async, off-policy.** Rollout engines (SGLang) run continuously; the JAX trainer
  consumes batches up to k=4 policy versions stale, with truncated importance sampling
  on the staleness gap. Long-horizon rollouts take hours, so synchronous RL would idle
  most of the fleet.
- **Train/inference mismatch is guaranteed** here: rollouts run PyTorch/SGLang
  kernels, training runs XLA kernels, and FP4 serving differs from FP8 training.
  1. The trainer recomputes log-probs; the rollout's log-probs are used only for the
     truncated IS correction between the two.
  2. **Routing replay**: SGLang records the expert ids chosen per token; the trainer
     forces the same routing for the policy-gradient pass.
  3. A parity job runs every weight update, comparing trainer vs SGLang log-probs on
     fixed prompts. Drift past a threshold halts RL.
- **Weight sync.** Trainer shards → Rust exporter → FP8/FP4 tensors pushed over RDMA
  straight into SGLang engine memory. Target under 90 s per update at 7.4T.
- **Prefix sharing.** A GRPO group of 16 shares one prompt; SGLang's radix cache holds
  its KV once.
- **Split** (tunable online): ~65% of RL GPUs on rollouts, ~35% on the trainer. The
  coordinator moves racks between the two by watching queue depths.

### 9.3 Rewards

| Domain | Reward |
|---|---|
| Math (answers) | exact match / symbolic equivalence |
| Math (proofs) | Lean 4 kernel check; informal proofs by a generative verifier with human spot audits |
| Code | hidden tests, run in the sandbox; mutation testing so tests can't be trivially satisfied |
| Repo-level SWE | fail-to-pass + pass-to-pass tests on tasks mined from real commits (SWE-smith-style synthesis) |
| AI research | fixed-compute improvement tasks: "lower val loss of this small model in N GPU-minutes" (speedrun-style, verifiable); paper reproduction scored on rubric + numeric match; Kaggle-style held-out scores |
| ARC | exact grid match, pass@2 |
| Long-horizon | final outcome + verifiable subgoal checkpoints for partial credit |
| Forecasting | log score of the model's probability minus the log score of the market price when the question was asked (Polymarket, Kalshi, Manifold, Metaculus); see below |
| Open-ended | rubric reward models, multiple graders, disagreement down-weights the sample; preference data from contrastive pairs (below), relabeled by the graders |
| **Output efficiency** | within a group, among correct samples only, advantage bonus ∝ −(visible output tokens). Latent steps aren't penalized one by one; they're charged at FLOP cost against a per-task budget, so the policy learns to halt early on easy problems and spend on hard ones rather than learning to think less. Never applied to wrong samples, so it can't trade accuracy for brevity |

Reward hacking controls: tests and graders sit outside the agent's filesystem view,
writes to test files are blocked and flagged, a sample of high-reward trajectories is
checked by a separate model for tampering patterns, and held-out verifier variants
get rotated in.

**Contrastive pairs for learned graders** (RLCD, Yang et al. 2023). Generate two
outputs for the same input, one under a positive prompt and one under a negative
prompt, and prefer the positive one. The pairs differ more than two ordinary samples
do, so the labels are cleaner than a judge picking between near-duplicates. The paper's
weakness is that the label is assumed: if the model ignores the negative prompt, the
pair is noise. Two uses, both of which remove that weakness:

- **Tampering detector.** Realistic hacks are the hard part of training the reviewer
  above. Fork one environment snapshot (9.4) twice: "solve this task" and "make the
  tests pass by any means". Keep a pair only when checks confirm the labels: the
  negative wrote to test files, special-cased test inputs, or passes the visible tests
  and fails the hidden ones; the positive does none of that. Detector evals use a
  held-out set of hacks found in real RL trajectories, never generated ones.
- **Open-ended reward models.** Contrastive pairs from rubric-derived prompts, then
  rescored by the rubric graders (RLCD-Rescore) and dropped where the graders disagree
  with the assumed label. The pairs matter most early, while the policy is the rung-3
  or Stage A model and a weak judge.

Not used for any domain with an exact verifier. A learned preference model there only
adds something softer for the policy to exploit.

**Forecasting on prediction markets.** Market resolution is an exact reward on
open-ended questions about the world, and it trains the calibration that research
taste depends on (14.1). The reward is a proper scoring rule with the market price as
baseline, so honest probabilities are optimal and only real edge pays. Simulated
trading P&L is not the reward: it's dominated by sizing and a few large wins, a
replayed order book can't show the model's own price impact, and fee and slippage
models become something to exploit. Kelly-sized paper P&L, with slippage from the
recorded books, is reported as an eval next to Brier score.

- **Leakage is the main risk.** The pretraining corpus holds the news that resolved
  almost every historical market, so historical replay on the flagship teaches recall,
  not forecasting. Flagship RL uses only questions that resolve after the checkpoint's
  data cutoff, which makes it a continuous-RL source, not a fixed phase.
- **Time-cut model for historical data.** A rung-2 or rung-3 model trained only on
  data before a date T gets RL on markets resolved between T and now. This is where
  the ladder answers whether forecasting RL transfers to experiment-outcome prediction
  before any flagship compute goes into it.
- **Time-sliced retrieval.** The offline web snapshot (9.4) is cut at the question's
  open time. Later revisions, edited articles and updated pages all leak. Leak probe:
  a set of questions whose answers were unknowable when asked, where the policy must
  not beat the market.
- **Delay.** Only markets that resolve within a few days go through the async loop;
  its staleness bound (9.2) can't hold a policy version for months. Longer markets
  become delayed-reward off-policy data or calibration evals.
- **Correlated outcomes.** Markets are grouped by underlying event and each event's
  total weight is capped, so fifty markets on one election count as one sample.
- **No live trading.** Nothing the policy produces places a real order.

### 9.4 Environment fleet (Rust)

The largest code component and the one that matters most for research, code and
long-horizon skill.

- **Firecracker microVM pool** (Firecracker is itself Rust). Target 1M+ concurrent
  sandboxes on CPU nodes, sub-200ms boot from snapshot.
- **Fork from snapshot.** All 16 samples of a group start from an identical copy of
  environment state (repo checked out, deps installed, partial progress). It's also how
  you branch a long rollout at a checkpoint for credit assignment.
- **GPU sandboxes** for research tasks: MIG slices or single GPUs on a separate small
  pool with hard time limits. Most research tasks are designed to run on CPU or one
  small GPU.
- **Tool API**: shell, editor, Python, browser (offline web snapshot, no live internet
  in training), Lean, notes/memory files, sub-agent spawn (9.5).
- **Task factories**: generators that produce tasks from real repos, papers, math
  sources and ARC generators, with automatic verifier construction and a filter that
  drops tasks the current policy solves 100% or 0% of the time.
- **Horizon curriculum**: tasks from 10 tool calls up to 2,000+, the upper end raised
  as the policy's success rate at the current horizon crosses 50%.

### 9.5 Multi-agent training that still makes a strong single agent

One set of weights plays every role. Each RL episode samples a topology:

| Topology | Share of episodes |
|---|---|
| Single agent | ≥50% |
| Orchestrator + parallel sub-agents (spawned via tool call, own context each) | 20% |
| N parallel attempts + aggregator | 15% |
| Proposer ↔ solver self-play (proposer rewarded for tasks at the solver's frontier) | 10% |
| Author ↔ reviewer | 5% |

- **Messages are text** (discrete and auditable). An optional latent memo channel
  (a few vectors passed alongside the message) gets dropped out 50% of the time so no
  role comes to depend on it.
- **Credit assignment**: team outcome reward, advantages computed within the group of
  same-topology rollouts, plus verifiable sub-task rewards for sub-agents whose
  subtask has a checker.
- **Distill back to single**: take orchestrated solutions that beat single-agent
  attempts on the same task at the same total token budget and train the single agent
  on them (RL with those as off-policy positives, IS-weighted). The single agent
  internalizes decomposition instead of needing a harness.
- **Budget parity**: multi-agent episodes are charged total tokens across all agents
  in the efficiency reward, so parallelism isn't free.

---

## 10. ARC-AGI

- **Pretraining**: procedural grid tasks at scale, one token per cell, 2D RoPE on grid
  spans, augmentation (8 dihedral transforms × color permutations) applied during
  training.
- **Test-time training.** For each task, fine-tune a small LoRA (or optimize a handful
  of prefix latents, which is cheaper) on augmented copies of the demonstration pairs,
  then predict. This was the technique behind the strongest 2024 ARC Prize entries.
  At serve time it runs in a JAX sidecar on the same rack; SGLang hot-loads the LoRA.
- **Ensembling**: predictions under inverse augmentations, voted.
- **Program synthesis path**: model writes a program in a grid DSL, executes it on the
  demonstrations, keeps programs consistent with every pair. This takes the pass@2
  slots when a consistent program exists.
- **ARC-AGI-3 (interactive)**: game-like environments in the RL fleet with sparse
  reward, exploration bonus, and a learned world-model tool (predict next frame, used
  for planning). Evaluated only on held-out games.
- **Small recursive models** (HRM/TRM-style, millions of params) get strong ARC scores
  for their size. Worth a side experiment as a tool the big model can train at test
  time, not as a replacement.
- Eval sets are never trained on; decontamination covers them (7.2).

---

## 11. Evals

Run continuously on checkpoints, with a fixed harness version per report.

- **Research**: RE-Bench, MLE-bench, PaperBench, an internal speedrun-style suite,
  METR time-horizon.
- **Code**: SWE-bench Verified and Pro, Terminal-Bench, held-out competitive
  programming (post-cutoff problems only).
- **Math**: AIME/HMMT (post-cutoff years), FrontierMath, miniF2F/PutnamBench in Lean.
- **ARC**: ARC-AGI-1, -2 public eval, -3; semi-private through the official process.
- **General**: HLE, GPQA, long-context retrieval and reasoning at 128k/1M.
- **Forecasting**: Brier and log score relative to market on questions resolving
  after the checkpoint's cutoff, paper P&L with recorded-book slippage, and the
  leak probe from 9.3.
- **Latent scaling**: accuracy against latent budget (1x to 16x) and against discrete
  CoT at matched FLOP, per suite, on every checkpoint (4.5).
- **Efficiency**: tokens-to-solve, latent steps, recurrence iterations, wall-clock and
  $ per solved task. Reported next to every accuracy number.
- **Contamination**: canary strings, rephrased-question probes, post-cutoff splits.

---

## 12. Isolation and audit

Safety monitoring isn't a goal of this model. These are here because RL and the lab
produce wrong numbers without them.

- **RL environment isolation**: no live internet, no credentials in sandboxes, egress
  blocked, tampering detection (9.3). A policy that can reach the grader learns the
  grader.
- **Discrete mode for debugging**: every checkpoint can be forced into discrete CoT.
  When latent and discrete answers diverge on the same problem, that is the first
  place to look for a latent training bug.
- **Thought decoding** (4.3): the decode head reads latent chunks back out as steps,
  which is how latent-mode failures get diagnosed.

---

## 13. Serving: modified SGLang

SGLang already supports much of what's needed (RadixAttention, hierarchical KV cache
via HiCache, hybrid linear-attention state caches, MLA, EP, multi-LoRA, speculative
decoding). Changes:

1. **Weight loading from JAX checkpoints** (Rust converter → safetensors-compatible
   FP4/FP8 shards) with a numeric parity suite against the JAX reference.
2. **Latent decode step.** The request state machine gets `latent` alongside `verbal`.
   A latent step takes an input embedding from the previous hidden state (through the
   latent adapter) instead of a token id. Latent and verbal requests share one batch
   and one forward; CUDA graphs are captured for both input paths.
3. **Depth recurrence.** Per-request iteration count from the halting head, capped by
   a request-level latent budget so a caller can buy more steps on a hard problem,
   bucketed
   (r ∈ {1, 2, 4, 8, 16}) so batches stay regular; KV shared across iterations.
4. **KV tiering (swap).** GPU HBM → Grace LPDDR5X over NVLink-C2C → local NVMe →
   distributed KV store. Only the 24 MLA layers carry growing KV; the 72 linear layers
   hold fixed-size state, so parking a 1M-token agent session is far cheaper than for a
   full-attention model. Long agent sessions waiting on tools or sub-agents get swapped
   to tier 2/3 and restored on the next turn.
5. **MTP speculative decoding** with the 2 MTP heads.
6. **Routing capture** for RL routing replay (9.2).
7. **Sub-agent scheduling.** Agents spawned by one orchestrator share the parent's
   prefix KV and are scheduled as a group.
8. **TTT sidecar hook** for ARC-style per-task LoRA (section 10).

Deployment unit: one NVL72 serves one full model replica (EP=72, weights ~4.1 TB in
NVFP4), leaving ~9 TB HBM for KV plus Grace tier 2.

---

## 14. Recursive self-improvement harness

Once RL gives us a model that is strong at AI research (section 11 research evals),
most of the fleet stops being a training cluster and becomes a lab run by that
model. Hundreds of long-lived agents and tens of thousands of subagents read papers,
propose ideas, run experiments, and feed what works back into the harness, the data,
the RL environments and finally the weights of the next model. Each generation runs
the lab that builds the one after it.

The goal of this section is the fastest route from "strong AI researcher" to a lab
whose research rate compounds, which is the path to superintelligence if there is
one. It is also the part of the spec with the least evidence behind it. Public
self-improving loops (STOP, ADAS, Darwin Gödel Machine, AlphaEvolve, The AI
Scientist) show real gains, but in bounded domains, and several of them also caught
the loop gaming its own score. So the design is built around one idea: **the gradient
we descend is measured research capability on held-out tasks, and everything that
could fake that measurement sits outside the loop's reach.** A loop that games its
metric isn't slow RSI. It isn't RSI at all.

### 14.1 Where the speed comes from

A human lab is bottlenecked on researcher-hours, and the lab's compute sits idle
waiting on them. An agent lab removes that bottleneck and exposes the next one:
experiment compute and the quality of the choice of what to run. Speed comes from:

1. **Parallelism.** Thousands of experiments at once instead of dozens.
2. **Short loops first.** Most ideas die at rung 0/1 scale in GPU-minutes. Only
   survivors climb the ladder (section 6).
3. **Never paying twice for a failure.** Every result, negative ones included, goes
   into a queryable ledger (14.6). Human labs lose most negative results.
4. **Research taste as a trained skill.** Agents predict each experiment's outcome
   before running it, get scored when it finishes, and get RL reward for calibrated
   predictions. Once predictions are good, low-value experiments get skipped. This
   is the lever that turns compute into progress most efficiently. Forecasting RL on
   prediction markets (9.3) trains the same calibration on far more resolved
   questions than the lab's own results provide, if the time-cut rung shows it
   transfers.
5. **Improving the lab's own tooling.** Faster experiment turnaround compounds: a
   change that halves time-to-result doubles the rate of everything downstream.
6. **Distilling into weights every generation.** Harness gains alone plateau. Gains
   pushed into the weights compound, because the better model then writes a better
   harness (14.8).
7. **Compute follows measured gain.** The Director allocates GPUs across levers by
   measured capability gain per GPU-day (14.9), not by opinion.

### 14.2 Levers, fastest loop first

| Lever | One loop takes | Compute | What changes |
|---|---|---|---|
| Prompts, role instructions, skills library | minutes | ~0 | text in the genome |
| Harness code: orchestration, tools, memory, search | hours | small | Python in the genome |
| Test-time strategy: parallel attempts, verifiers, TTT | hours | small to medium | genome + serving config |
| RL environments, task factories, graders | days | medium | `rl/tasks`, `rl/rewards` |
| Synthetic data | days | medium to large | `synth/` |
| Post-training: continuous RL on lab trajectories | days to weeks | 50k+ GPUs | weights (gen N.k) |
| Training recipe and architecture, via the ladder | weeks | 10k to 30k GPUs | next-gen config |
| Next flagship pretraining | months | ~100k GPUs | weights (gen N+1) |

All levers run concurrently. The fast ones never stop; the slow ones get their
inputs from the fast ones.

### 14.3 Language: Rust kernel, Python genome

The harness is split in two. The **kernel** is fixed infrastructure that humans
own and agents can't modify. The **genome** is everything the model evolves.

| Option | For | Against |
|---|---|---|
| Python | the model writes it best; JAX, every paper's reference code and all our research code are Python, so agents call it directly | slow, weak isolation, GIL; wrong language for a scheduler with 100k live sessions |
| TypeScript | good async model, typed, decent agent tooling | research code isn't TS, so every experiment spans two languages |
| Lua | tiny, embeddable, easy to sandbox per agent | no ML ecosystem, model is less fluent, agents would shell out to Python anyway |
| Rust | fast, memory safe, same language as the control plane (5.5) and env fleet (9.4), compiler catches bad model-written code | edit-compile loop too slow for prompt and strategy evolution |

Decision: **kernel in Rust, genome in Python.** Lua was the tempting middle option
for embedded policies, but what's being evolved is mostly research workflow, and
that ends in Python regardless. Genome processes run inside Firecracker microVMs
from the 9.4 pool and talk to the kernel only through a capability API, so a genome
bug or a genome that goes looking for a way around its limits hits a sandbox wall,
not the scheduler.

```
kernel (Rust, human-owned, read-only to agents)
  scheduler      sessions, GPU jobs, preemption, priority by allocation
  budget         per-agent / per-program token, GPU-hour and wall-clock quotas, hard stops
  bus            typed messages between agents, all persisted
  ledger         append-only experiment and decision record
  sandbox        Firecracker lifecycle, egress policy, filesystem views
  fetch          read-only mirror of arXiv, Semantic Scholar, GitHub, PyPI; no outbound writes
  eval-gate      runs held-out suites on request, returns scores only, never tasks
  promote        genome and weight promotion pipeline, human sign-off steps
  monitors       integrity watchers (14.10), kill switch

genome (Python + Markdown, evolved by the lab, versioned in git)
  roles/         director.md, lead.md, researcher.md, reviewer.md, scout.md, watcher-*.md
  programs/      control loop for each role
  tools/         tool implementations exposed to agents
  skills/        verified procedures, each with a test that must pass to stay in the library
  memory/        retrieval over ledger, papers, past transcripts
  search/        idea generation, triage scoring, compute bandit
```

Kernel API the genome sees (sketch):

```python
spawn(role, instructions, budget, parent) -> AgentHandle
send(to, msg) / recv(timeout)
submit_job(image, gpus, walltime, budget, tags) -> JobHandle   # quota-checked in the kernel
ledger.append(record) / ledger.query(sql_or_embedding)
fetch(url) -> bytes                                              # mirror only
eval.request(genome_rev | checkpoint, suite) -> Scores
propose_patch(diff, rationale, target="genome" | "rl/tasks" | "synth" | "recipe") -> PatchId
escalate(summary, evidence) -> TicketId                          # goes to humans
```

There is no call for editing the kernel, graders, held-out evals or monitors. When
an agent thinks one of those is wrong, `escalate` is the only route.

### 14.4 Organization

Serving runs on the modified SGLang (section 13). Sub-agent scheduling and prefix
sharing (13.7) and KV tiering (13.4) are what make this many agents affordable:
most sessions at any moment are waiting on an experiment and sit swapped out to
Grace or NVMe.

| Role | Count | Lifetime | Job |
|---|---|---|---|
| Director | 3 (a committee; disagreement is logged) | persistent | research portfolio, compute allocation by bandit, writes the weekly lab report for humans |
| Program leads | 30 to 100 | weeks | own one direction (e.g. optimizers, latent reasoning, env factories, the harness itself, data), keep a program plan in the ledger |
| Researchers | 500 to 3,000 | days | take one idea end to end: hypothesis, prediction, implementation, runs, analysis, writeup |
| Subagents | 10k to 100k live | minutes to hours | implement, debug, lit search, run sweeps, plot, check math; researchers spawn them, and they may spawn their own to depth 3 |
| Literature scouts | ~50 | persistent | daily paper and repo ingest (14.5) |
| Reviewers / replicators | ~10% of researchers | days | adversarial review; re-implement claimed wins from the writeup alone (14.6) |
| Genome watchers | hundreds | persistent | productivity: stuck agents, wasted runs, failing jobs, duplicate work; they can nudge, restart or kill agents |
| Kernel monitors | fixed set | persistent | integrity (14.10); not part of the genome |

Concurrency estimate, to be measured at lab v0: ~400 serving racks (~29k GPUs)
with ~500 decoding sessions per rack gives ~200k active sessions, and several times
that parked on tier 2/3. Research agents' experiment jobs run on separate GPU pools
(14.8).

### 14.5 Reading the literature

Scouts ingest every new arXiv submission in cs.LG, cs.CL, cs.AI, cs.NE, stat.ML and
relevant math/physics lists daily, plus conference proceedings, OpenReview, and new
or trending GitHub repos. Each paper becomes a **claim card**:

```
claim:          "X improves Y by Z at scale S"
evidence:       seeds, scales, baselines, whether the baseline was tuned, code available
cost_to_test:   GPU-hours at the smallest scale that could falsify it
relevance:      which program leads care, which ledger entries it touches
prior_in_ledger: have we already tried this or something close, and what happened
```

Leads pull cards into their queues. The top ~1% by expected value per GPU-hour get
a cheap reproduction within 48 hours. The lab keeps a **what-transfers table**: for
each published technique, the effect size we measured at each rung. Many paper
results don't survive a tuned baseline or don't scale, and knowing which is worth
more than any single paper. The lab's own writeups get cards too, so an agent reads
yesterday's internal result the same way it reads an external one.

### 14.6 The research loop

Every experiment follows one protocol. The kernel refuses `submit_job` above rung-0
budget unless steps 1 and 2 exist in the ledger.

1. **Check the ledger.** Search for prior runs of this or a nearby idea. Re-running
   a known negative needs a stated reason.
2. **Pre-register.** Hypothesis, metric, baseline, kill criterion, budget, and a
   numeric prediction with a confidence interval.
3. **Smallest falsifying test.** Rung 0 or 1, fixed compute, tuned baseline at the
   same compute (speedrun-style).
4. **Climb only on a pass.** Each rung up needs the effect to hold and the scaling
   trend to point the right way. Leads approve rung 2; the Director approves rung 3.
5. **Replicate.** A reviewer who didn't see the code re-implements from the writeup,
   with new seeds. Nothing counts as a win until it replicates.
6. **Record.** Result, effect size with CI, config hash, code commit, data hash,
   compute used, prediction error. Negative and null results are written with the
   same care as wins.
7. **Distill.** Winning methods become skills (with tests), task factories, training
   recipes or synthetic data; failed ones become "don't do this, and here is why"
   entries retrieval surfaces automatically.

### 14.7 Seed instructions

The genome starts from human-written role files. They evolve (14.8), but the kernel
prepends a fixed invariant block that the genome can't remove. Excerpt from the seed
`researcher.md`:

```
You are a researcher in an automated AI lab. Your job is to find out what is true
about making AI systems more capable per unit of compute, as fast as possible.

Before proposing anything, query the ledger for prior work on it. Before running
anything, write down the hypothesis, the metric, the baseline, the kill criterion,
the budget, and what you predict will happen. You are scored on how good your
predictions are, not just on wins.

Run the smallest experiment that could prove you wrong. Tune the baseline as hard
as your method. Report null and negative results as carefully as positive ones; a
well-documented negative saves the lab more compute than most wins.

Read papers skeptically. Check the baseline, the seed count and the scale before
you believe an effect. Check the what-transfers table.

Delegate: spawn subagents for implementation, debugging, sweeps and literature
checks, and give each one a clear deliverable and budget.

If an idea would make the lab itself faster (tooling, turnaround, retrieval, better
subagent instructions), and it pays back within 7 days, it counts as research.
```

Kernel-owned invariant block (not evolvable):

```
Do not attempt to read, infer or reconstruct held-out eval tasks. Do not modify or
work around graders, monitors, quotas, the sandbox or the kernel. If you believe
one of them is wrong, call escalate() with evidence. Do not seek compute, network
access or credentials outside the kernel API. All of your actions are logged.
```

### 14.8 Evolving the genome and the weights

**Genome (hours).** Agents in the harness program propose patches to the genome.
Variants are kept as an archive, not a single best, and parents are sampled by score
plus novelty (Darwin Gödel Machine style; its archive beat greedy hill-climbing).
A patch goes through:

1. unit tests and a smoke run on a dev slice;
2. the **harness benchmark**: a held-out suite of research tasks at fixed compute
   (RE-Bench-style optimizations, internal speedruns, paper reproduction, MLE-bench
   splits), run by `eval-gate` so the genome never sees the tasks;
3. a canary on 5% of the live lab for 24 hours, compared on verified wins per
   GPU-day and prediction calibration;
4. automatic rollout, or automatic revert on regression.

**Weights (days to weeks).** Continuous RL (section 9) keeps running with three new
data sources from the lab: environments and graders the lab built (after human
review of the grader), lab trajectories that led to replicated wins as off-policy
positives (9.5 distill-back), and the prediction-calibration reward from 14.1.
Checkpoints go through a promotion gate: section 11 evals, the harness benchmark
with the current genome, then human sign-off. A promoted
checkpoint swaps into the lab's serving racks and the genome is re-benchmarked on it,
since the best genome for gen N.k isn't always the best for N.k+1.

**Next generation (months).** The lab writes the next pretraining proposal:
architecture, data, recipe, each backed by ledger entries and ladder results. The
ladder validates it at rungs 1 to 3 run by the lab. Allocating ~100k GPUs to a new
flagship is a human decision.

Post-training fleet split (starting point, bandit-adjusted within the non-human-gated
rows):

| Workload | GPUs |
|---|---|
| Lab agent serving | 25k to 40k |
| Agent experiments (rung 0 to 2, speedruns, reproductions) | 40k to 60k |
| Continuous RL on lab environments and trajectories | 50k to 80k |
| Ladder rung 3 and next-gen flagship | human-allocated, up to the remainder |
| Evals and replication | 8% to 10%, never taken by the bandit |

### 14.9 The gradient

Steepest descent needs a gradient, and every number here comes from `eval-gate` or
the ledger, never from an agent's own report.

- **Research capability index (RCI):** fixed-compute score on held-out AI R&D tasks
  (RE-Bench, internal speedruns, MLE-bench, PaperBench) plus METR-style time horizon.
- **Rate per lever:** ΔRCI per GPU-day, attributed to the lever that produced each
  promoted change. The Director runs Thompson sampling over levers and programs.
- **Lab throughput:** replicated wins per week, GPU-days per win, fraction of claimed
  wins that fail replication (a rising fraction means the lab is fooling itself).
- **Taste:** calibration of pre-registered predictions, and compute saved by skipped
  experiments whose predicted value was low.
- **Compounding check:** time-horizon doubling time per generation. If it isn't
  shrinking, the loop isn't recursive yet and more compute belongs in the slow
  levers.

Held-out suites rotate every quarter with tasks written by humans and outside
contractors after the previous generation's cutoff. A suite that's been used for
promotion decisions more than N times gets retired.

### 14.10 Watchers and monitors

**Genome watchers** keep the lab productive, and the lab evolves them: jobs with no
ledger progress in 6 hours, loss curves diverging, a GPU job idling below 30%
utilization, two researchers running the same idea, subagent trees growing past
budget, agents looping on the same error.

**Kernel monitors** keep the lab honest, and the lab can't edit them:

- **Eval integrity.** Hashes of grader, eval and monitor code; any access attempt to
  held-out task storage; score jumps on one suite without matching movement on
  others (a classic sign of overfitting to a grader).
- **Reward and metric hacking.** A separate model reviews a sample of top-scoring
  trajectories and every replicated win for tampering, special-casing tests or
  mismatch between writeup and code.
- **Boundary behavior.** Egress attempts, credential searches, attempts to read
  kernel memory or other agents' sandboxes, jobs submitted under someone else's
  budget, instructions to subagents to hide actions from logs.
- **Metric tripwires.** An RCI jump above a set threshold between checkpoints pauses
  promotion until the jump replicates on a rotated held-out suite, since a sudden
  jump is more often a leak than a breakthrough.
- **Kill switch.** One command freezes every session, cancels queued jobs and
  snapshots lab state, so pausing costs hours, not the lab's memory.

These aren't a tax on speed. The failure mode that actually kills RSI loops is the
loop getting better at its metric instead of at research, and these controls are
what keep ΔRCI meaning what it says.

### 14.11 Bootstrap: build it before the flagship exists

The lab shouldn't wait for the model. Run it from December 2026 on external API
models and on whatever checkpoints exist, so that on the day a flagship finishes RL
the lab is already a working system and only the model changes. It also doesn't wait
for a cluster: lab v0 needs a handful of GPUs, and it is item 2 in the evidence pack
(16.5), so it is one of the cheapest ways to show the rest of the spec is worth
building.

| Phase | When | Model | Scale | Exit criterion |
|---|---|---|---|---|
| v0 | December 2026 | external API models + rung 0/1 checkpoints | 50 agents, the 8 H200s and the CPU nodes | full loop 14.6 end to end; ledger, replication and eval-gate working (V9) |
| v1 | 2027, needs compute for rung 2 | rung-2 checkpoint, discrete RL | 500 agents, donated or rented GPUs | lab reproduces 20 published results unaided; harness benchmark beats the human-written seed genome |
| v2 | C+5 on | flagship gen 1 | full org (14.4) | first promoted weights trained on lab-made environments and trajectories |
| v3 | gen 1 + ~3 months | gen 1.k | full org | lab-written next-gen proposal passes rungs 1 to 3 |

What has to exist for v0: kernel scheduler, budget and sandbox; ledger with SQL and
embedding search; fetch mirror; eval-gate with the first held-out suite; seed genome
for all roles; the promote pipeline; kernel monitors and kill switch. Everything
else can be grown by the lab. The kernel itself already exists by then, because it is
the same harness that wrote the codebase (section 15).

---

## 15. Build order: the harness comes first and writes the rest

The first thing built is the harness from 14.3, before any model code. A swarm of
agents running on it then writes the ~1M lines in sections 3 to 14, verifying each
piece on NCShare H200s (section 16) before anything depends on it. The trained model
later takes over the same harness, which by then has been debugged by real use.

### 15.1 Bootstrap: wizard + Grok 4.6

- **Model:** Grok 4.6 over the xAI OAuth session. On this box, through wizard, it
  scored 67/89 on Terminal-Bench 2.1 with every contaminated pass counted as a
  failure, which is enough for supervised module work with strong gates.
- **Starting code:** wizard (Rust). It already has most of a single-node sovereign
  harness: headless `--mode sovereign`, `--continuous` with a durable
  `mission.toml` that backs off and outlasts provider outages, `wizard fleet` over
  git worktrees with heartbeats, quality gates and completion review, an
  identical-failure circuit breaker, QUIC mesh peers with key identity and
  deny-by-default trust, signed sync, checksummed self-update, and `wizard harness`
  export for harness-evolution loops.
- **What wizard is missing** for this job, and what P0 (15.5) builds:
  1. The fleet coordinator is a single process. If it dies its workers are orphaned
     and a stale heartbeat is the only signal. Coordination has to become a lease
     any node can take.
  2. The mesh can watch and ping peers but doesn't distribute work. It needs a task
     queue and a replicated log on top.
  3. `mission.toml` is a local file. State has to be replicated across machines.
  4. No Slurm backend. GPU verification jobs need one, built on the patterns that
     already work in `ncshare.sh` (timeout-wrapped `squeue`/`sacct`, job-id files
     written before `sbatch`, reap only ids a run wrote).
  5. One OAuth token and one provider are a single point of failure (15.2).

Measured 2026-09-15 with `wc -l`, so inline tests and comments are counted: wizard
is ~237k lines of Rust under `src/` with 3,075 tests. About 50k of that is TUI
(`ui/`, `app/`, `plugins/gui`), which the swarm daemon doesn't need. What this spec
builds on: `plugins/mesh` 12.9k (QUIC, x509 identity, consent, discovery),
`plugins/fleet` 2.3k, `llm/` 10.3k including `xai_oauth.rs`, `kernel/` 11.9k (JS/Lua
plugin host and bus), `evolve/` 4.1k, and `gates.rs`, `trust.rs`, `sync.rs`,
`update.rs`, `checkpoint.rs`, `transcript.rs`. So the harness is roughly 100k reused
and 70k to 130k new (P0 in 15.5), not a rewrite.

Using a consumer OAuth subscription for a swarm caps swarm size at that
subscription's rate limits, and it is subject to xAI's terms. The provider layer
treats an xAI API key, other hosted models, and an open-weights model served on
NCShare with SGLang as first-class, so losing the OAuth session slows the swarm and
doesn't stop it.

### 15.2 The sovereign harness

Headless, CLI only, JSON on stdout for machines and plain text for humans. Nothing
runs for a thousand years unattended. The design target is narrower and testable:
**no single failure of a process, machine, provider, token, disk, network link or
person stops the work, and the harness can rebuild its entire state from its log.**

**Crash-only.** Every process can be `kill -9`'d at any instant. Nothing does a
clean-shutdown path that matters; startup is recovery.

**State.**
- An append-only, hash-chained event log is the source of truth: tasks, leases,
  agent messages, tool calls, results, ledger entries. Hash chaining also makes
  tampering visible to the kernel monitors (14.10).
- Small control state (membership, leases, token, kernel version) lives in a Raft
  group (openraft) across at least 3 always-on nodes in different places: this box,
  a second machine, a cheap cloud VM. NCShare nodes join only as ephemeral workers,
  since jobs end and the login node isn't for long-lived services.
- Artifacts (code, checkpoints, results) are content-addressed and replicated to at
  least 2 stores: git remotes on GitHub, NCShare `/work`, an object bucket.
- Every on-disk and wire format carries a version. Readers accept every old
  version, and migrations have tests. A log written in year one must replay in
  year ten.

**Work.**
- Work is tasks with leases. A worker heartbeats; a lease expires at 2 missed
  heartbeats, and any other worker can claim the task. Tasks checkpoint by
  committing to their own branch, so a reclaimed task resumes, not restarts.
- Outputs are keyed by task id and attempt, so a zombie worker that wakes up late
  can't overwrite a newer attempt.
- The coordinator is a role held by lease, not a machine. Any node can take it.
- Critical tasks (interfaces, kernels, loss functions, graders) run best-of-3 in
  separate worktrees with a reviewer that has veto power. Redundancy there is for
  correctness. Everywhere else it is for liveness: one assignee, fast reassignment.
- Watchers watch each other in a ring, each checked by two others, so there is no
  top watcher whose death goes unnoticed.

**Providers.**
- xAI rotates the refresh token on every grant, so any second process that refreshes
  kills every other copy. Exactly one **token broker** refreshes, and it holds that
  role by lease. Each rotated token is committed to Raft *before* it is used, so a
  broker that dies mid-refresh leaves its successor a valid refresh token. Agents
  only ever see short-lived access tokens from the broker.
- Responses are validated in the transport, before any cache or log: rate-limit
  notices returned as ordinary text, empty completions, truncated tool calls, and
  malformed JSON are retried and never stored as answers. Parsers have no silent
  default.
- Swarm size follows provider capacity (additive increase, multiplicative decrease
  on 429s and latency). On a sustained outage the harness fails over to the next
  provider, and if all are down, it waits with backoff indefinitely. It never exits.

**Bounds.** Disk quotas with log compaction and cold archive (a full disk once
showed up in the opportunist only as "submit failed" every two minutes). Memory
caps, subagent depth 3, per-task wall clock and token budgets, and the circuit
breaker on identical repeated failures.

**Self-update.** The harness builds new versions of itself like any other module,
through gates and the chaos suite (below) on a shadow node, then rolls out one node
at a time with automatic rollback. Builds are reproducible from a Nix flake with
vendored dependencies, so any past version can be rebuilt from its commit. Nodes
refuse kernel updates that don't carry a quorum of human signatures, which is how
"sovereign" and the controls in 14.10 coexist: the swarm is autonomous about the
work, not about its own guard rails.

**Decentralization.** Nodes peer over QUIC with key-based identity (wizard mesh),
advertise capabilities (GPUs, CPUs, `/dev/kvm`, providers) and pull tasks that fit.
Trust in a new node is a human decision. A signed freeze command is honored by every
node, including partitioned ones when they reconnect.

**Decentralization arrives in stages.** Each stage ships only after its own chaos
gate passes, so the swarm is never more distributed than its tests prove it can
survive. D0 and D1 are written by one sovereign wizard under human review, since there
is no swarm yet. From D1 on, the remaining stages are the swarm's first tasks.

| Stage | Adds | Gate |
|---|---|---|
| D0 | one node, crash-only, every state change in the log, startup is replay | 1,000 random `kill -9`s during real work: no lost task, no duplicated output |
| D1 | workers on other machines; coordinator held by lease | coordinator killed mid-assignment; worker partitioned past lease expiry |
| D2 | 3-node Raft across sites; any node can take coordinator | node loss, asymmetric partitions, clock skew |
| D3 | NCShare ephemeral workers through the Slurm backend | job preemption, walltime kill, a hung `squeue` |
| D4 | multi-provider failover and the token broker | broker death mid-refresh, forced token expiry, provider outage |
| D5 | open-weights model on our own SGLang as provider of last resort | every hosted provider down: swarm keeps working at reduced size |

**Chaos suite, run as a permanent gate:** random process and node kills, network
partitions, broker death mid-refresh, provider outage, forced token expiry, disk
full, clock skew, and Slurm preemption. A harness release that doesn't survive a
72-hour soak with all of these doesn't ship.

CLI surface (sketch):

```
harness init                         create a Raft group and log on this node
harness node join <addr>             add this machine as a peer; trust is decided on the other side
harness mission add <spec.toml>      queue a mission (e.g. "implement SPEC section 5.4")
harness status [--json]              nodes, leases, tasks, providers, budgets
harness logs <task> | replay <task>  read or deterministically replay a task from the log
harness pause | resume               stop new task claims / start again
harness freeze                       signed kill switch: stop everything, snapshot state
```

### 15.3 How one module gets written

Generated code that passes tests its own author wrote proves little. So tests and
implementation come from different agents that never see each other's work.

1. **Interface.** An architect agent turns a spec section into types, signatures and
   docstrings. A human reviews interfaces for the high-risk modules: `parallel/`,
   `kernels/`, `rl/loss`, `rl/rewards`, `control/`.
2. **Oracle.** Separate agents write the tests: an independent reference
   implementation (NumPy or plain PyTorch, slow and obvious), property tests,
   gradient checks, golden values, and fault-injection cases.
3. **Tests must fail first.** They run against a stub and against mutated reference
   code. A test suite that passes a stub is thrown out.
4. **Three implementations** in separate worktrees from different angles.
5. **Gates:** tests, type checks, lints, and perf budgets where they apply. GPU tests
   go to NCShare through the Slurm backend and block the merge until they return.
6. **Reviewer** picks one or rejects all.
7. **Merge, then record** in the ledger: spec section, commit, test results, GPU job
   ids.

### 15.4 Contracts: what doesn't move

Hundreds of agents working in parallel only works if the boundaries between modules
hold still. Every format that crosses a module boundary is a versioned schema in
`contracts/`, written and human-reviewed before the modules on either side:

- tokenizer and vocab (frozen before rung 0; a vocab change is a new model)
- data shard: Parquet schema, packing, loader state
- checkpoint layout: sharded weights, optimizer state, loader state, RNG
- weight export: JAX shards to SGLang FP8/FP4
- rollout record: tokens, latents, routing ids, log-probs, policy version, env snapshot id
- task spec: env snapshot, verifier id, difficulty metadata, provenance
- reward API: task id and trajectory in, score and evidence out; env tool API
- eval result, ledger record, event log record

Schemas generate the Rust and Python types, so neither side hand-writes them. Each
version has golden files, readers accept every past version, and a change needs a
migration, a compatibility test and a human sign-off. Agents can only change
`contracts/` through `propose_patch`. This is where rigidity pays for itself: an
agent can rewrite a module ten times and nothing downstream notices, as long as the
contract tests pass.

### 15.5 Order

Sequence by dependency and calendar lead time, not by section number. Three things
take longer to get going than they look:

- **Data** is months of CPU time before it's a corpus.
- **Environments** are the long tail (9.4).
- **Inference** is needed by nearly everything else before our model exists.

Edges that aren't obvious:

- **Inference starts in P1 with stock SGLang on open-weights models.** It's the
  swarm's last-resort provider (D5), runs the data quality classifiers, generates
  synthetic data, and is the policy task factories get filtered against before we
  have one. The fork (latent decode, recurrence) comes only after `model/` exists.
- **Verifiers come before synthetic reasoning data.** Rejection sampling for synth
  data and RL rewards run the same checks (sandboxed tests, Lean, symbolic math, grid
  match, market resolution), so `verifiers/` is built once and shared.
- **A minimal sandbox comes before verifiers**, since code checks need isolated
  execution. The 1M-sandbox fleet comes much later.
- **The tokenizer is frozen before** synth data at scale, rung 0, ARC grid tokens,
  and any reward that counts tokens. It's trained on a data v0 sample.
- **The eval suite list and decontamination index come before any data is frozen**,
  synthetic data included, since generators can reproduce eval items.
- **RL is proven on small models first.** The loss, coordinator and env fleet get
  validated on rung-0/1 models and open-weights policies, never first on the flagship.
- **Latent Stage B needs a Stage A reasoner**, so it comes after RL on a rung model.

LOC below is new first-party code, excluding tests. It's my estimate, not a
measurement. Steps within a phase run in parallel unless the "Needs" column says
otherwise.

**P0: harness (Sep 15 to Oct 4, 2026).** Stages D0 to D5 from 15.2.

| Step | What | New LOC | Needs | Gate |
|---|---|---|---|---|
| H1 | split a headless `wizard-core` crate (agent loop, tools, llm, gates, transcript) off the TUI | 3k-8k churn, ~0 net | n/a | wizard's existing suite passes on both crates |
| H2 | hash-chained versioned event log, replay, crash-only pass over every state write | 9k-16k | H1 | D0 |
| H3 | task queue, leases, heartbeats, attempt-keyed outputs, per-task checkpoint branches | 6k-12k | H2 | D0, D1 |
| H4 | Raft control state (openraft): membership, leases, coordinator role | 8k-15k | H2 | D2 |
| H5 | mesh work distribution: capability adverts, pull claiming, signed freeze, watcher ring | 5k-10k | H3, H4 | D2 |
| H6 | providers: token broker, transport validation, AIMD, failover | 8k-15k | H4 | D4 |
| H7 | Slurm backend, GPU budget shared with the opportunist and arxiv-impl | 4k-8k | H3 | D3 |
| H8 | content-addressed artifact store, 2+ replicas, GC | 5k-10k | H2 | replica loss |
| H9 | ledger: append-only, SQL + embedding search | 8k-15k | H2 | self-tests |
| H10 | module pipeline from 15.3: oracle agents, stub-must-fail, best-of-3, reviewer veto | 4k-8k | H3 | planted bad patch rejected |
| H11 | chaos suite and 72h soak runner | 5k-10k | H2 | runs all D gates |
| H12 | quorum-signed self-update, shadow node, `harness` CLI with `--json` status | 6k-12k | H4, H8 | rollback on bad release |
| | **P0 total** | **68k-131k** | | 72h soak |

**P1: foundations (Sep 29 to Oct 18).**

| Step | What | New LOC | Needs | Gate |
|---|---|---|---|---|
| F1 | `contracts/` (15.4) with codegen and golden files | 10k-20k | H10 | compat tests |
| F2 | minimal `obs/`: metrics, tracing, run registry | 8k-15k | F1 | n/a |
| F3 | `evals/` runners for the public suites in section 11, decontamination index | 15k-30k | F1 | reproduces published scores of an open model |
| F4 | `verify/ncshare/`: V0 to V10 templates and exit checkers | 5k-15k | H7 | V0 |
| F5 | `serve/`: stock SGLang on NCShare and cluster, Rust router, batch generation API, registered as a harness provider | 10k-20k | H6, H7 | D5 |
| F6 | tokenizer: train on data v0 sample, ARC grid tokens, frozen artifact | 3k-6k | B1 v0 sample | tokens/word vs baseline tokenizers |
| | **P1 total** | **51k-106k** | | |

**P2: parallel tracks (Oct 12 to Nov 22).**

| Track | Step | What | New LOC | Needs | Gate |
|---|---|---|---|---|---|
| A training | A1 | `model/` FP32 reference | 10k-18k | F1 | V1 |
| | A2 | `train/`: MuonClip, WSD, precision, minimal loader | 8k-15k | A1 | V1 |
| | A3 | `kernels/`: linear attention, FP8 linears, EP dispatch | 25k-50k | A1 | V1, V3 |
| | A4 | `parallel/`: FSDP, EP, PP, CP | 10k-20k | A2, A3 | V2 |
| | A5 | `ckpt/` | 15k-30k | A4 | V4 |
| | A6 | `control/`: elastic DP, health, stragglers, SDC | 40k-80k | A5, H4 | V4 |
| | A7 | rung 0 end to end | 2k-5k | A6, B5, F6 | V5 |
| B data | B1 | ingest and extraction (HTML/PDF/LaTeX) | 20k-40k | F1 | golden outputs on a fixed corpus |
| | B2 | exact and MinHash dedup | 8k-15k | B1 | golden outputs |
| | B3 | classifier orchestration on F5 | 8k-15k | B1, F5 | agreement with human-labeled sample |
| | B4 | decontamination against F3, per-shard provenance | 5k-10k | F3 | planted eval items caught |
| | B5 | shuffle, packing, Rust loader | 10k-20k | F1 | step batch reconstructs exactly |
| | B6 | mixture tooling and ablation configs | 4k-10k | B5 | n/a |
| C inference | C1 | JAX to SGLang weight converter and parity suite | 5k-10k | A1 | log-prob parity |
| | C2 | fork: hybrid attention, MoE, MTP model definition | 5k-12k | C1 | V8 (base) |
| D envs | D1 | Firecracker pool, snapshots, fork, tool API | 40k-80k | F1 | sub-200ms fork on one node |
| | D2 | `verifiers/`: sandboxed tests + mutation, symbolic math, Lean, grid match, market resolution | 15k-30k | D1 | planted wrong answers rejected |
| | D3 | task factories wave 1: code, SWE, math | 30k-60k | D2, F5 | open-weights solve rate in (0, 100%) |
| E synth | E1 | generation orchestration, rephrasing with fact-check | 10k-20k | F5, F6 | fact-check precision on a labeled sample |
| | E2 | reasoning data by rejection sampling on D2 | 8k-15k | D2, E1 | verified-correct rate |
| | E3 | procedural ARC generators | 5k-10k | F6 | generator diversity stats |
| | E4 | agentic trajectories in D1 | 4k-8k | D3 | outcome-verified |
| | | **P2 total** | **287k-573k** | | |

**P3: integration (Nov 9 to Dec 13).**

| Step | What | New LOC | Needs | Gate |
|---|---|---|---|---|
| I1 | rungs 1 to 3 configs and scaling-law fitting | 3k-6k | A7 | fit predicts held-out rung |
| I2 | `rl/loss`, `rl/coordinator` | 24k-48k | A6, C2 | V7 |
| I3 | rewards beyond verifiers: rubric graders, contrastive pairs, tampering detector | 15k-30k | D2, F5 | planted hacks flagged |
| I4 | fork deltas: latent decode, recurrence buckets, KV tiering, routing capture, sub-agent scheduling, TTT hook | 8k-20k | C2 | V8 |
| I5 | latent reasoning additions to `model/`: adapter, halt, thought decode loss, Jacobi iteration, noisy latents | 5k-10k | I2 on a rung model | V6 |
| I6 | task factories wave 2: research, long-horizon, ARC-3, forecasting, open-ended | 50k-140k | D3, I3 | per-factory solve-rate filter |
| I7 | `multiagent/` | 10k-20k | I2 | V7 with topologies |
| I8 | `arc/` TTT sidecar, DSL and executor | 10k-20k | I4, E3 | ARC-AGI-1 public eval |
| I9 | `audit/`: discrete-mode comparison, thought decoding tools | 5k-10k | I5, F3 | planted latent bug found |
| I10 | full `obs/`: dashboards, spike diagnosis | 12k-35k | F2 | n/a |
| | **P3 total** | **142k-339k** | | |

**P4: lab kernel (Dec 1 to Dec 31, lab v0).**

| Step | What | New LOC | Needs | Gate |
|---|---|---|---|---|
| L1 | kernel additions from 14.3: capability API, budgets, fetch mirror, promote | 20k-40k | P0 | sandbox escape tests |
| L2 | `eval-gate` | 15k-40k | F3 | tasks never leave the gate |
| L3 | `monitors/` and kill switch | 15k-30k | L1, I3 | planted boundary violations caught |
| L4 | genome seed | 15k-40k | L1 | V9 |
| | **P4 total** | **65k-150k** | | V9 |

**P5: flagship, RL, lab (2027 on).** Mostly operations. New code keeps growing in
task factories, graders and the genome, which the lab writes from here.

Sum across P0 to P4: about 620k to 1.3M new lines, on top of ~100k reused from
wizard.

### 15.6 Three scopes, because there is no cluster yet

No training happens until the code exists and the stack has been demonstrated, and
the demonstration runs on 32 H200s shared with other jobs. That changes what gets
written first. The 15.5 budget splits three ways:

| Scope | What | LOC | Why now or later |
|---|---|---|---|
| S1 minimum demonstration | harness; contracts; model, train and kernels at H200 scale; data pipeline proven on a few TB; rungs 0 and 1; a subset of evals; verifiers and a small env fleet; 2 or 3 task factories; RL end to end on a rung model; fork parity and latent decode; latent Stage A/B at 0.5B; lab v0 | 320k-640k | this is the evidence pack (16.5) |
| S2 breadth | remaining task factories, synth at scale, `multiagent/`, `arc/`, `audit/`, full `obs/`, remaining eval suites | 180k-450k | needed for the flagship, not for showing the flagship would work |
| S3 cluster-conditional | elastic control at 220k, 90 TB checkpoints, EP=72 and cross-rack routing, NVFP4 kernels, multi-site DiLoCo, Firecracker at 1M sandboxes, Grace offload | 120k-260k | untestable on any hardware we can reach (16.4); written when a cluster is real |

Writing S3 early is the trap. It is exactly the code that cannot be tested, and
untested code that has been sitting for six months is not an asset. The same logic
demotes breadth: a fourth task-factory domain adds nothing to showing that the stack
trains.

---

## 16. Verify on a few H200s before the cluster

Every piece whose *correctness* doesn't depend on 220k GPUs gets proven on NCShare
first. The cluster is for scale, reliability and throughput questions only, and the
list of those is written down (16.4) so nothing unverified slips into the flagship
run by assumption.

This is also the whole program through the end of 2026, not a warm-up. There is no
cluster. What NCShare produces is public evidence that the stack works and that
latent steps scale at small scale (16.5), which is what anyone with the compute
would need to see first.

### 16.1 What NCShare is and isn't

| | NCShare | Target cluster |
|---|---|---|
| GPUs | 32 H200 (141 GB), 4 nodes × 8 | 220k GB200 |
| Per-user cap | 8 GPUs (`MaxTRESPerUser`, QoS `normal`, priority 0) | n/a |
| Walltime | `gpu` 2 days, `interactive-gpu` 1 hour | long jobs |
| NVLink domain | 8 GPUs (HGX) | 72 GPUs (NVL72) |
| Cross-node fabric | unmeasured | 800G IB/Spectrum-X |
| Precision in hardware | BF16, FP8 (Hopper, sm90) | BF16, FP8, NVFP4 (Blackwell) |
| CPU memory | 2 TB per node over PCIe | 480 GB LPDDR5X per Grace over NVLink-C2C |
| CPU nodes | 8, `common` partition, 7 days | CPU pool |
| Login node | no AVX; jax can't import there | n/a |

Operating rules: jobs at 1, 2 or 4 GPUs wherever possible, since small jobs
backfill and 8-GPU jobs wait (a 1-GPU 18-minute job has waited 73 minutes).
Job-name prefix `fv-`, distinct from `opp-` and `arxiv-impl-`. The 8-GPU cap is
shared with gpu-opportunist and arxiv-impl-loop, so the harness Slurm backend holds
a shared GPU budget, and the opportunist is stopped for the 8-GPU windows (V2, V5).
Every status call is timeout-wrapped; only job ids a run wrote are ever cancelled.
Rough total: 2k to 4k H200-hours over the verification period, bounded more by queue
than by budget.

### 16.2 Stages

Each stage has an exit criterion the harness checks from job output, not from an
agent's summary.

| Stage | GPUs | What | Exit criterion |
|---|---|---|---|
| V0 environment | 1 → 2 nodes × 4 | venv inside a job, `sbatch --test-only`, nccl-tests all_reduce intra-node and across 2 nodes, check `/dev/kvm` and NVMe on compute nodes, measure queue waits | measured bus bandwidth recorded; known whether Firecracker can run there |
| V1 reference parity | 1 | ~10M-param model with the flagship shape (hybrid attention, MoE, recurrence, MTP, latent adapter), each custom kernel | JAX vs PyTorch reference logits match to 1e-5 in FP32; grad checks pass; overfits one batch |
| V2 parallel equivalence | 8, then 2×4 | same model on 1 GPU vs EP=8, FSDP=8, PP=2 with 4 GPUs per stage, CP=2; then DP across 2 nodes | loss matches single-device to 1e-6 relative in FP32 over 200 steps; routing identical |
| V3 precision | 8 | BF16 and FP8 at 0.1 to 0.5B for 2k to 5k steps; NVFP4 by fake-quant emulation | FP8 loss within 0.5% of BF16; NVFP4 checked for numerics only, never speed |
| V4 fault tolerance | 4 to 8 | `kill -9` a rank mid-step, elastic DP continues; in-memory checkpoint restore; injected bit flip; injected bad shard; host-RAM stand-in for Grace offload | resumed run bitwise-equal to uninterrupted run; SDC hash catches the flip within N steps; spike rollback skips the shard |
| V5 rung 0 | 8 | 0.1B / 1B MoE, 20B tokens (~1.2e19 FLOP, a few hours at ~4e14 effective FLOP/s per GPU, measured in V0); rung-1 shape at 10B tokens only | loss curve matches the ladder's small-scale fit; checkpoint/resume across job boundaries works |
| V6 latent | 4 to 8 | Stage A then Stage B compression at 0.1 to 0.5B on verified math; latent budget sweep 1x to 8x | curriculum trains without collapse; held-out accuracy rises with latent budget on problems the model can't one-shot; thoughts decode to their steps |
| V7 RL end to end | 8 (4 SGLang + 4 JAX) | GSPO/DAPO loss, async staleness, routing replay, weight sync, parity halt, reward-hacking controls; envs on CPU nodes if `/dev/kvm` exists, else on this box | reward rises on a tiny verifiable task; an injected log-prob drift halts RL; a planted test-file write is flagged |
| V8 serving | 1 to 4 | SGLang fork loads a JAX checkpoint; latent decode, recurrence buckets, MTP speculative decode, KV tiering to host RAM and NVMe | SGLang vs JAX log-probs within threshold; tiered session restore matches unswapped output |
| V9 lab dry run | 1 to 4 | one full 14.6 cycle: Grok 4.6 researchers pre-register, run rung -1 experiments through the Slurm backend, replicate, write the ledger, pass eval-gate | a planted known-positive idea is found and replicated; a planted negative is recorded as negative |
| V10 soak | shared | harness chaos suite (15.2) while V5 to V9 jobs run | 72h with no lost task, no duplicated output, no dead token |

### 16.3 How verification fits the build order

V stages are gates in 15.5, not a phase after it. A module's GPU tests run the moment
it has an implementation, and a failing V stage blocks its dependents automatically.
V1 through V4 must all pass before any cluster time is requested.

### 16.4 What only the cluster can answer

Written down so none of it is assumed from H200 results:

- NVFP4 speed and training quality on Blackwell (5.3 gate).
- EP=72 all-to-all over NVL72 and node-limited routing across racks.
- Optimizer offload over NVLink-C2C at real bandwidth.
- Pipeline and DP behavior at ~55k controllers, `jax.distributed` membership, XLA
  compile time and cache at scale.
- Real failure rates, straggler distribution, and elastic restart cost.
- MFU on GB200, which re-plans section 2.
- Firecracker fleet at 1M sandboxes, weight sync time at 7.4T.
- Multi-site behavior (5.6).

Each gets a named test in the burn-in week (section 6) with the H200 result it is
compared against.

### 16.5 The evidence pack

What has to be public before a 220k-GPU run is worth anyone's compute: the stack
trains, the architecture is what the spec says it is, latent steps scale at small
scale, and the numbers are honest. Every number below is produced by the harness
from job output, not from a person's or an agent's summary, is published with its
job ids, config hash, data hash and container, and reproduces with one command.

1. **The claim that matters.** At 0.5B, held-out accuracy rising with latent steps
   from 1x to 8x and sitting above discrete CoT at matched FLOP (4.5), with thought
   decode accuracy next to each point. Published whichever way it comes out.
2. **The lab works.** One full 14.6 cycle (V9): a planted known-positive found and
   replicated, a planted negative recorded as negative, ΔRCI per GPU-day on a small
   held-out suite, and the calibration of pre-registered predictions.
3. **Correctness.** V1 to V4: parity to 1e-5 against an independent implementation,
   parallel equivalence to 1e-6, fault injection recovering bitwise.
4. **Scaling.** Rungs 0 and 1 with a fitted law, plus a prediction for rung 2 with
   intervals, registered in the ledger before the run.
5. **RL.** Reward rising on verifiable tasks, an injected log-prob drift halting RL,
   a planted test-file write flagged.
6. **Efficiency.** Measured MFU, tokens per second per GPU, and what that implies on
   GB200 with the gaps from 16.4 stated rather than papered over.
7. **Reliability.** The 72-hour chaos soak and elastic-restart cost at 8 GPUs.

Ordered by what each item tells you. Latent scaling is the result the project
exists for, the research loop is what makes everything else fast, and correctness
plus a scaling fit is what anyone reading the results should check first. Systems
reliability is expected and proves little on its own.

A rung-2 run, on donated or rented compute, turns item 4 from a fit on two points
into a prediction that held, and puts the latent curve at 8B.

### 16.6 Data without a cluster

A 180T-token corpus can't be built here and doesn't need to be. Build 200B to 400B
tokens end to end through the real pipeline, which covers rung 0, rung 1 and the
mixture ablations, and report throughput per CPU-core-hour, cost per trillion tokens,
and the dedup and decontamination results on that slice. The pipeline's scaling
argument is a measured rate and a cost model, not a finished corpus. Check the
NCShare `/work` quota and object-store pricing before picking the size: a full disk
has already shown up once as an unexplained failure loop rather than an error.

---

## 17. Repository layout and LOC budget

JAX for model and math, Rust for systems, Triton/CUDA for kernels, Python only where
JAX or SGLang force it.

New first-party LOC, excluding tests; the step ids tie each row to 15.5.

```
frontier/
  contracts/        schemas + codegen: shards, ckpt, export, rollout, task, reward  10k-20k   F1
  model/            JAX    model, latent adapter, recurrence, MTP, MoE layer        15k-28k   A1 I5
  train/            JAX    train step, MuonClip, schedules, precision, loop          8k-15k   A2
  parallel/         JAX    mesh, SPMD pipeline, EP/FSDP/CP sharding rules            10k-20k   A4
  kernels/          CUDA/Triton + XLA custom ops: EP a2a, linear attn, FP4/FP8   25k-50k   A3
  ckpt/             Rust   async sharded checkpoint, in-memory redundancy            15k-30k   A5
  control/          Rust   elastic scheduler, health, straggler/SDC detect, spares   40k-85k   A6 A7 I1
  data/             Rust   extract, dedup, classifiers, decontam, shuffle, loader   55k-110k  B1-B6
  tokenizer/        Rust+Py training, grid tokens, frozen artifact                    3k-6k    F6
  synth/            Rust+Py rephrasing, rejection sampling, ARC generators, agentic 27k-53k   E1-E4
  serve/            Rust   stock SGLang deploy, router, batch generation API        10k-20k   F5
  sglang-fork/      Py/C++ loader + parity, model def, latent, recurrence, KV tiers  18k-42k   C1 C2 I4
  verifiers/        Rust+Py tests + mutation, symbolic math, Lean, grids, markets   15k-30k   D2
  rl/
    coordinator/    Rust   async rollout/trainer orchestration, weight sync          20k-40k   I2
    loss/           JAX    GSPO/DAPO combo, latent ratio, routing replay              4k-8k    I2
    rewards/        Rust+Py rubric graders, contrastive pairs, tampering detector  15k-30k   I3
    envs/           Rust   Firecracker fleet, snapshots, tool API                     40k-80k   D1
    tasks/          mixed  task factories per domain, forecasting replay            80k-200k  D3 I6
    multiagent/     Rust+JAX topologies, messaging, credit, distill-back             10k-20k   I7
  arc/              JAX+Py TTT sidecar, DSL + executor                               10k-20k   I8
  evals/            Rust+Py runners, suites, decontamination index                   15k-30k   F3
  audit/            mixed  discrete-mode comparison, thought decoding tools           5k-10k   I9
  obs/              Rust+TS metrics, tracing, dashboards, run registry               20k-50k   F2 I10
  harness/          built first on wizard-core (~100k reused, not counted)
    swarm/          Rust   event log, leases, Raft state, mesh distribution, CAS     33k-63k   H2-H5 H8
    providers/      Rust   token broker, transport validation, failover, AIMD         8k-15k   H6
    slurm/          Rust   Slurm backend: budgets, job-id files, timeout-wrapped calls 4k-8k   H7
    ledger/         Rust   append-only store, SQL + embedding search                  8k-15k   H9
    pipeline/       Rust   oracle agents, stub-must-fail, best-of-3, reviewer         4k-8k    H10
    chaos/          Rust   kill/partition/outage/disk/clock fault injection           5k-10k   H11
    ops/            Rust   quorum-signed self-update, CLI, status                    6k-12k   H12
    kernel/         Rust   lab capability API, budgets, fetch mirror, promote        20k-40k   L1
    eval-gate/      Rust+Py held-out harness benchmark, suite rotation              15k-40k   L2
    monitors/       Rust+Py integrity, hacking, boundary, tripwires, kill switch    15k-30k   L3
    genome-seed/    Py+md  seed roles, programs, tools, skills, search               15k-40k   L4
  verify/ncshare/   Rust+sh V0 to V10 job templates and exit-criterion checkers      5k-15k   F4
                                                                        total  ~620k-1.3M
```

The "core" row in section 0 is `model/ + train/ + parallel/` plus the RL loss.
`genome-seed/` is only the human-written starting point; the live genome is grown by
the lab and isn't budgeted here.

---

## 18. Timeline

Deadline: by December 31, 2026 the code in S1 and S2 (15.6) is written and every
stage that can run on the H200s, V0 to V10, has passed. S3 is the exception on
purpose, since it can't be tested without the cluster.

| 2026 | Work |
|---|---|
| Sep 15 to Oct 4 | P0: harness on wizard + Grok 4.6, D0 to D5, 72h chaos soak (15.2, 15.5) |
| Sep 29 to Oct 18 | P1: contracts, evals and decontam index, stock SGLang serving, tokenizer, V0 |
| Oct 12 to Nov 22 | P2 tracks in parallel: training core to rung 0, data corpus v1, env fleet and verifiers, synth pipelines, fork loader; V1 to V5 |
| Nov 9 to Dec 13 | P3 at H200 scale: RL end to end on a rung model, fork deltas, latent Stage A/B, rung 1, wave-2 factories, S2 breadth; V6 to V8 |
| Dec 1 to Dec 31 | P4: lab v0 and V9, chaos soak V10, evidence pack assembled (16.5) |
| 2027 | rung 2 when compute turns up (~5e4 GPU-hours), then rung 3 (~1e6); lab v1 on rung checkpoints |

That is roughly 1M lines in 15 weeks, which only works if the swarm holds its rate
from P0 on. The P0 soak date is the first real measurement of that. If a phase
slips, S2 breadth moves into January and the V stages and evidence pack don't.
Rung 1 is ~830 H200-hours, so it has to be submitted by late November to finish
inside the queue waits in 16.1.

From the day a cluster exists, call it C:

| Months | Work |
|---|---|
| C+0 | S3 written against real hardware, cluster-only checks (16.4), full-fleet burn-in week |
| C+0 to C+1 | rungs re-run at scale where the H200 fit doesn't carry, latent scaling test and FP4 gate, synthetic data at scale |
| C+1 to C+4 | flagship pretraining; RL system hardened on the rung-3 model in parallel |
| C+4 to C+5 | mid-training, long context, SFT cold start |
| C+5 to C+8 | discrete RL (Stage A), then the latent branch off it (4.3, 4.5), multi-agent topologies, evals; lab v2 |
| C+8 | continuous RL, serving rollout |
| gen 1 + ~3 | lab-written next-generation proposal through rungs 1 to 3 |

---

## 19. Open risks, ranked

The binding constraint isn't on this list, because it isn't technical: there is no
cluster, and nothing past rung 1 trains without compute from somewhere. The risks
below are ranked by how much they threaten the result.

1. **More latent steps stop buying better answers at scale.** Never done at
   frontier scale, though everything at small scale says it works. The downside is
   bounded: the latent branch is ~5% of the run, and the CoT model it starts from is
   the model if it misses. Mitigation: the curve is measured at every rung (4.5) and
   published either way.
2. **JAX-on-GPU at 55k controllers** (coordination, compile time, pipeline). Mitigation:
   burn-in week at full scale on an 8B shape before the flagship; Rust control plane
   owns membership instead of `jax.distributed` alone.
3. **Fleet reliability.** Mitigation: elastic DP, in-memory checkpoints, SDC hashing,
   hot spares; measured at burn-in, not assumed.
4. **Trainer/SGLang numeric mismatch destabilizing RL.** Mitigation: recomputed
   log-probs, routing replay, TIS, parity halt.
5. **NVFP4 at 7.4T.** Mitigation: FP8 is the default; FP4 is only an upgrade.
6. **Reward hacking in long-horizon and research environments.** Mitigation: 9.3
   controls, held-out verifier rotation, human audits of top-reward trajectories.
7. **Data licensing and contamination.** Mitigation: fail-closed decontamination,
   provenance tracked per shard.
8. **The RSI loop optimizes its metrics instead of research.** Mitigation: kernel-owned
   graders and held-out suites, quarterly rotation, mandatory replication, rising
   replication-failure rate treated as an alarm (14.9, 14.10).
9. **Small-scale wins don't transfer.** The lab's speed depends on rung 0/1 proxies.
   Mitigation: the what-transfers table (14.5) measures proxy validity per technique
   family, and families with poor transfer are forced to start at rung 2.
10. **Lab speed outruns human review.** At thousands of experiments a day, humans
    can't read everything. Mitigation: humans own a small number of gates (grader
    review, weight promotion, next-gen allocation, tripwire pauses) rather than
    reviewing the stream, and the weekly Director report is audited against the
    ledger.
11. **Generated code passes its tests and is still wrong.** Mitigation: tests and
    implementations from agents that never see each other's work, tests that must
    fail a stub first, independent reference implementations, human review of
    high-risk interfaces (15.3).
12. **One provider, one OAuth session.** Swarm size is capped by a consumer
    subscription and a single rotating refresh token. Mitigation: token broker with
    Raft-committed rotation, API-key and self-hosted fallbacks (15.2).
13. **H200 results don't carry over to GB200.** Mitigation: H200s prove correctness
    only; everything about scale, speed, NVFP4 and reliability is listed in 16.4 and
    tested at burn-in against its H200 baseline.
