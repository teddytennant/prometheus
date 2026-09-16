# Prometheus

An open spec, and soon open code, for training a frontier model from scratch. JAX
for the model, Rust for the systems, a modified SGLang for rollouts and serving.
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
gets measured there first. The code is done and the H200 tests pass by the end of
2026.

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
