# Prometheus

Code for training a frontier model from scratch, written against an open spec. JAX
for the model, Rust for the systems, a modified SGLang for rollouts and serving.
It's a work in progress: the first modules are in and most of the stack isn't.
Labs publish models but not the code that trains them. This is the public version.

The model thinks in latent space instead of writing out a long chain of thought.
Each thought is the whole hidden state fed back in, not one token sampled from it, so
more latent steps give better answers. [Coconut](https://arxiv.org/abs/2412.06769)
and [Reverie](https://github.com/teddytennant/reverie) show it at small scale. It
has never been done at frontier scale. It's very likely to work, since the model
starts as a normal CoT reasoner and only learns to compress steps it already takes.
If it fails, that's about 5% of the compute and you keep the CoT model.

There's no cluster. An agent swarm writes most of the ~1M lines, everything that
doesn't need 220k GPUs gets tested on 32 shared H200s, and the latent scaling curve
gets measured there first. The target is the code done and the H200 tests passing by
the end of 2026.

## What's here so far

- `model/`, the FP32 JAX reference model (A1)
- `data/extract`, HTML, PDF and LaTeX text extraction (B1, Rust)
- `data/loader`, shuffling, packing and a resumable loader (B5, Rust)
- `evals/`, the eval harness and decontamination (F3)
- `contracts/` and `crates/contracts`, the versioned contracts shared by Python and Rust (F1)
- `harness/slurm` and `harness/ledger`, the Slurm backend and experiment ledger (H7, H9)

Nothing has run on an H200 yet. To run the CPU tests:

```
uv run pytest tests -m 'not gpu'
cargo test --workspace
```

## Reading the spec

[MONOLITH.md](MONOLITH.md) is the whole thing in one file. [spec/](spec/) has the
same text split by section. Good places to start:

- [Section 4: latent reasoning](spec/04-latent-reasoning.md), the main idea
- [Section 16: H200 verification](spec/16-h200-verification.md), what gets proven before any cluster
- [Section 18: timeline](spec/18-timeline.md)
- [Section 19: open risks](spec/19-risks.md), what's most likely to be wrong

Edit the files in `spec/`, then rebuild the monolith:

```
awk 'FNR==1 && NR>1 {print ""; print "---"; print ""} {print}' spec/*.md > MONOLITH.md
```
