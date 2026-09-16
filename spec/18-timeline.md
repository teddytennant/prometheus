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
