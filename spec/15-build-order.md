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
