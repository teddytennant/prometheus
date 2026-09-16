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
