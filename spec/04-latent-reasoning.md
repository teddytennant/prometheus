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
