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
