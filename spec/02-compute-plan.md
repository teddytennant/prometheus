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
