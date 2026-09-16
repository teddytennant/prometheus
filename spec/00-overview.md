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
