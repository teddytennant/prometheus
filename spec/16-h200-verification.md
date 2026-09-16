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
