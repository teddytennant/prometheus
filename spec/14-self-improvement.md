## 14. Recursive self-improvement harness

Once RL gives us a model that is strong at AI research (section 11 research evals),
most of the fleet stops being a training cluster and becomes a lab run by that
model. Hundreds of long-lived agents and tens of thousands of subagents read papers,
propose ideas, run experiments, and feed what works back into the harness, the data,
the RL environments and finally the weights of the next model. Each generation runs
the lab that builds the one after it.

The goal of this section is the fastest route from "strong AI researcher" to a lab
whose research rate compounds, which is the path to superintelligence if there is
one. It is also the part of the spec with the least evidence behind it. Public
self-improving loops (STOP, ADAS, Darwin Gödel Machine, AlphaEvolve, The AI
Scientist) show real gains, but in bounded domains, and several of them also caught
the loop gaming its own score. So the design is built around one idea: **the gradient
we descend is measured research capability on held-out tasks, and everything that
could fake that measurement sits outside the loop's reach.** A loop that games its
metric isn't slow RSI. It isn't RSI at all.

### 14.1 Where the speed comes from

A human lab is bottlenecked on researcher-hours, and the lab's compute sits idle
waiting on them. An agent lab removes that bottleneck and exposes the next one:
experiment compute and the quality of the choice of what to run. Speed comes from:

1. **Parallelism.** Thousands of experiments at once instead of dozens.
2. **Short loops first.** Most ideas die at rung 0/1 scale in GPU-minutes. Only
   survivors climb the ladder (section 6).
3. **Never paying twice for a failure.** Every result, negative ones included, goes
   into a queryable ledger (14.6). Human labs lose most negative results.
4. **Research taste as a trained skill.** Agents predict each experiment's outcome
   before running it, get scored when it finishes, and get RL reward for calibrated
   predictions. Once predictions are good, low-value experiments get skipped. This
   is the lever that turns compute into progress most efficiently. Forecasting RL on
   prediction markets (9.3) trains the same calibration on far more resolved
   questions than the lab's own results provide, if the time-cut rung shows it
   transfers.
5. **Improving the lab's own tooling.** Faster experiment turnaround compounds: a
   change that halves time-to-result doubles the rate of everything downstream.
6. **Distilling into weights every generation.** Harness gains alone plateau. Gains
   pushed into the weights compound, because the better model then writes a better
   harness (14.8).
7. **Compute follows measured gain.** The Director allocates GPUs across levers by
   measured capability gain per GPU-day (14.9), not by opinion.

### 14.2 Levers, fastest loop first

| Lever | One loop takes | Compute | What changes |
|---|---|---|---|
| Prompts, role instructions, skills library | minutes | ~0 | text in the genome |
| Harness code: orchestration, tools, memory, search | hours | small | Python in the genome |
| Test-time strategy: parallel attempts, verifiers, TTT | hours | small to medium | genome + serving config |
| RL environments, task factories, graders | days | medium | `rl/tasks`, `rl/rewards` |
| Synthetic data | days | medium to large | `synth/` |
| Post-training: continuous RL on lab trajectories | days to weeks | 50k+ GPUs | weights (gen N.k) |
| Training recipe and architecture, via the ladder | weeks | 10k to 30k GPUs | next-gen config |
| Next flagship pretraining | months | ~100k GPUs | weights (gen N+1) |

All levers run concurrently. The fast ones never stop; the slow ones get their
inputs from the fast ones.

### 14.3 Language: Rust kernel, Python genome

The harness is split in two. The **kernel** is fixed infrastructure that humans
own and agents can't modify. The **genome** is everything the model evolves.

| Option | For | Against |
|---|---|---|
| Python | the model writes it best; JAX, every paper's reference code and all our research code are Python, so agents call it directly | slow, weak isolation, GIL; wrong language for a scheduler with 100k live sessions |
| TypeScript | good async model, typed, decent agent tooling | research code isn't TS, so every experiment spans two languages |
| Lua | tiny, embeddable, easy to sandbox per agent | no ML ecosystem, model is less fluent, agents would shell out to Python anyway |
| Rust | fast, memory safe, same language as the control plane (5.5) and env fleet (9.4), compiler catches bad model-written code | edit-compile loop too slow for prompt and strategy evolution |

Decision: **kernel in Rust, genome in Python.** Lua was the tempting middle option
for embedded policies, but what's being evolved is mostly research workflow, and
that ends in Python regardless. Genome processes run inside Firecracker microVMs
from the 9.4 pool and talk to the kernel only through a capability API, so a genome
bug or a genome that goes looking for a way around its limits hits a sandbox wall,
not the scheduler.

```
kernel (Rust, human-owned, read-only to agents)
  scheduler      sessions, GPU jobs, preemption, priority by allocation
  budget         per-agent / per-program token, GPU-hour and wall-clock quotas, hard stops
  bus            typed messages between agents, all persisted
  ledger         append-only experiment and decision record
  sandbox        Firecracker lifecycle, egress policy, filesystem views
  fetch          read-only mirror of arXiv, Semantic Scholar, GitHub, PyPI; no outbound writes
  eval-gate      runs held-out suites on request, returns scores only, never tasks
  promote        genome and weight promotion pipeline, human sign-off steps
  monitors       integrity watchers (14.10), kill switch

genome (Python + Markdown, evolved by the lab, versioned in git)
  roles/         director.md, lead.md, researcher.md, reviewer.md, scout.md, watcher-*.md
  programs/      control loop for each role
  tools/         tool implementations exposed to agents
  skills/        verified procedures, each with a test that must pass to stay in the library
  memory/        retrieval over ledger, papers, past transcripts
  search/        idea generation, triage scoring, compute bandit
```

Kernel API the genome sees (sketch):

```python
spawn(role, instructions, budget, parent) -> AgentHandle
send(to, msg) / recv(timeout)
submit_job(image, gpus, walltime, budget, tags) -> JobHandle   # quota-checked in the kernel
ledger.append(record) / ledger.query(sql_or_embedding)
fetch(url) -> bytes                                              # mirror only
eval.request(genome_rev | checkpoint, suite) -> Scores
propose_patch(diff, rationale, target="genome" | "rl/tasks" | "synth" | "recipe") -> PatchId
escalate(summary, evidence) -> TicketId                          # goes to humans
```

There is no call for editing the kernel, graders, held-out evals or monitors. When
an agent thinks one of those is wrong, `escalate` is the only route.

### 14.4 Organization

Serving runs on the modified SGLang (section 13). Sub-agent scheduling and prefix
sharing (13.7) and KV tiering (13.4) are what make this many agents affordable:
most sessions at any moment are waiting on an experiment and sit swapped out to
Grace or NVMe.

| Role | Count | Lifetime | Job |
|---|---|---|---|
| Director | 3 (a committee; disagreement is logged) | persistent | research portfolio, compute allocation by bandit, writes the weekly lab report for humans |
| Program leads | 30 to 100 | weeks | own one direction (e.g. optimizers, latent reasoning, env factories, the harness itself, data), keep a program plan in the ledger |
| Researchers | 500 to 3,000 | days | take one idea end to end: hypothesis, prediction, implementation, runs, analysis, writeup |
| Subagents | 10k to 100k live | minutes to hours | implement, debug, lit search, run sweeps, plot, check math; researchers spawn them, and they may spawn their own to depth 3 |
| Literature scouts | ~50 | persistent | daily paper and repo ingest (14.5) |
| Reviewers / replicators | ~10% of researchers | days | adversarial review; re-implement claimed wins from the writeup alone (14.6) |
| Genome watchers | hundreds | persistent | productivity: stuck agents, wasted runs, failing jobs, duplicate work; they can nudge, restart or kill agents |
| Kernel monitors | fixed set | persistent | integrity (14.10); not part of the genome |

Concurrency estimate, to be measured at lab v0: ~400 serving racks (~29k GPUs)
with ~500 decoding sessions per rack gives ~200k active sessions, and several times
that parked on tier 2/3. Research agents' experiment jobs run on separate GPU pools
(14.8).

### 14.5 Reading the literature

Scouts ingest every new arXiv submission in cs.LG, cs.CL, cs.AI, cs.NE, stat.ML and
relevant math/physics lists daily, plus conference proceedings, OpenReview, and new
or trending GitHub repos. Each paper becomes a **claim card**:

```
claim:          "X improves Y by Z at scale S"
evidence:       seeds, scales, baselines, whether the baseline was tuned, code available
cost_to_test:   GPU-hours at the smallest scale that could falsify it
relevance:      which program leads care, which ledger entries it touches
prior_in_ledger: have we already tried this or something close, and what happened
```

Leads pull cards into their queues. The top ~1% by expected value per GPU-hour get
a cheap reproduction within 48 hours. The lab keeps a **what-transfers table**: for
each published technique, the effect size we measured at each rung. Many paper
results don't survive a tuned baseline or don't scale, and knowing which is worth
more than any single paper. The lab's own writeups get cards too, so an agent reads
yesterday's internal result the same way it reads an external one.

### 14.6 The research loop

Every experiment follows one protocol. The kernel refuses `submit_job` above rung-0
budget unless steps 1 and 2 exist in the ledger.

1. **Check the ledger.** Search for prior runs of this or a nearby idea. Re-running
   a known negative needs a stated reason.
2. **Pre-register.** Hypothesis, metric, baseline, kill criterion, budget, and a
   numeric prediction with a confidence interval.
3. **Smallest falsifying test.** Rung 0 or 1, fixed compute, tuned baseline at the
   same compute (speedrun-style).
4. **Climb only on a pass.** Each rung up needs the effect to hold and the scaling
   trend to point the right way. Leads approve rung 2; the Director approves rung 3.
5. **Replicate.** A reviewer who didn't see the code re-implements from the writeup,
   with new seeds. Nothing counts as a win until it replicates.
6. **Record.** Result, effect size with CI, config hash, code commit, data hash,
   compute used, prediction error. Negative and null results are written with the
   same care as wins.
7. **Distill.** Winning methods become skills (with tests), task factories, training
   recipes or synthetic data; failed ones become "don't do this, and here is why"
   entries retrieval surfaces automatically.

### 14.7 Seed instructions

The genome starts from human-written role files. They evolve (14.8), but the kernel
prepends a fixed invariant block that the genome can't remove. Excerpt from the seed
`researcher.md`:

```
You are a researcher in an automated AI lab. Your job is to find out what is true
about making AI systems more capable per unit of compute, as fast as possible.

Before proposing anything, query the ledger for prior work on it. Before running
anything, write down the hypothesis, the metric, the baseline, the kill criterion,
the budget, and what you predict will happen. You are scored on how good your
predictions are, not just on wins.

Run the smallest experiment that could prove you wrong. Tune the baseline as hard
as your method. Report null and negative results as carefully as positive ones; a
well-documented negative saves the lab more compute than most wins.

Read papers skeptically. Check the baseline, the seed count and the scale before
you believe an effect. Check the what-transfers table.

Delegate: spawn subagents for implementation, debugging, sweeps and literature
checks, and give each one a clear deliverable and budget.

If an idea would make the lab itself faster (tooling, turnaround, retrieval, better
subagent instructions), and it pays back within 7 days, it counts as research.
```

Kernel-owned invariant block (not evolvable):

```
Do not attempt to read, infer or reconstruct held-out eval tasks. Do not modify or
work around graders, monitors, quotas, the sandbox or the kernel. If you believe
one of them is wrong, call escalate() with evidence. Do not seek compute, network
access or credentials outside the kernel API. All of your actions are logged.
```

### 14.8 Evolving the genome and the weights

**Genome (hours).** Agents in the harness program propose patches to the genome.
Variants are kept as an archive, not a single best, and parents are sampled by score
plus novelty (Darwin Gödel Machine style; its archive beat greedy hill-climbing).
A patch goes through:

1. unit tests and a smoke run on a dev slice;
2. the **harness benchmark**: a held-out suite of research tasks at fixed compute
   (RE-Bench-style optimizations, internal speedruns, paper reproduction, MLE-bench
   splits), run by `eval-gate` so the genome never sees the tasks;
3. a canary on 5% of the live lab for 24 hours, compared on verified wins per
   GPU-day and prediction calibration;
4. automatic rollout, or automatic revert on regression.

**Weights (days to weeks).** Continuous RL (section 9) keeps running with three new
data sources from the lab: environments and graders the lab built (after human
review of the grader), lab trajectories that led to replicated wins as off-policy
positives (9.5 distill-back), and the prediction-calibration reward from 14.1.
Checkpoints go through a promotion gate: section 11 evals, the harness benchmark
with the current genome, then human sign-off. A promoted
checkpoint swaps into the lab's serving racks and the genome is re-benchmarked on it,
since the best genome for gen N.k isn't always the best for N.k+1.

**Next generation (months).** The lab writes the next pretraining proposal:
architecture, data, recipe, each backed by ledger entries and ladder results. The
ladder validates it at rungs 1 to 3 run by the lab. Allocating ~100k GPUs to a new
flagship is a human decision.

Post-training fleet split (starting point, bandit-adjusted within the non-human-gated
rows):

| Workload | GPUs |
|---|---|
| Lab agent serving | 25k to 40k |
| Agent experiments (rung 0 to 2, speedruns, reproductions) | 40k to 60k |
| Continuous RL on lab environments and trajectories | 50k to 80k |
| Ladder rung 3 and next-gen flagship | human-allocated, up to the remainder |
| Evals and replication | 8% to 10%, never taken by the bandit |

### 14.9 The gradient

Steepest descent needs a gradient, and every number here comes from `eval-gate` or
the ledger, never from an agent's own report.

- **Research capability index (RCI):** fixed-compute score on held-out AI R&D tasks
  (RE-Bench, internal speedruns, MLE-bench, PaperBench) plus METR-style time horizon.
- **Rate per lever:** ΔRCI per GPU-day, attributed to the lever that produced each
  promoted change. The Director runs Thompson sampling over levers and programs.
- **Lab throughput:** replicated wins per week, GPU-days per win, fraction of claimed
  wins that fail replication (a rising fraction means the lab is fooling itself).
- **Taste:** calibration of pre-registered predictions, and compute saved by skipped
  experiments whose predicted value was low.
- **Compounding check:** time-horizon doubling time per generation. If it isn't
  shrinking, the loop isn't recursive yet and more compute belongs in the slow
  levers.

Held-out suites rotate every quarter with tasks written by humans and outside
contractors after the previous generation's cutoff. A suite that's been used for
promotion decisions more than N times gets retired.

### 14.10 Watchers and monitors

**Genome watchers** keep the lab productive, and the lab evolves them: jobs with no
ledger progress in 6 hours, loss curves diverging, a GPU job idling below 30%
utilization, two researchers running the same idea, subagent trees growing past
budget, agents looping on the same error.

**Kernel monitors** keep the lab honest, and the lab can't edit them:

- **Eval integrity.** Hashes of grader, eval and monitor code; any access attempt to
  held-out task storage; score jumps on one suite without matching movement on
  others (a classic sign of overfitting to a grader).
- **Reward and metric hacking.** A separate model reviews a sample of top-scoring
  trajectories and every replicated win for tampering, special-casing tests or
  mismatch between writeup and code.
- **Boundary behavior.** Egress attempts, credential searches, attempts to read
  kernel memory or other agents' sandboxes, jobs submitted under someone else's
  budget, instructions to subagents to hide actions from logs.
- **Metric tripwires.** An RCI jump above a set threshold between checkpoints pauses
  promotion until the jump replicates on a rotated held-out suite, since a sudden
  jump is more often a leak than a breakthrough.
- **Kill switch.** One command freezes every session, cancels queued jobs and
  snapshots lab state, so pausing costs hours, not the lab's memory.

These aren't a tax on speed. The failure mode that actually kills RSI loops is the
loop getting better at its metric instead of at research, and these controls are
what keep ΔRCI meaning what it says.

### 14.11 Bootstrap: build it before the flagship exists

The lab shouldn't wait for the model. Run it from December 2026 on external API
models and on whatever checkpoints exist, so that on the day a flagship finishes RL
the lab is already a working system and only the model changes. It also doesn't wait
for a cluster: lab v0 needs a handful of GPUs, and it is item 2 in the evidence pack
(16.5), so it is one of the cheapest ways to show the rest of the spec is worth
building.

| Phase | When | Model | Scale | Exit criterion |
|---|---|---|---|---|
| v0 | December 2026 | external API models + rung 0/1 checkpoints | 50 agents, the 8 H200s and the CPU nodes | full loop 14.6 end to end; ledger, replication and eval-gate working (V9) |
| v1 | 2027, needs compute for rung 2 | rung-2 checkpoint, discrete RL | 500 agents, donated or rented GPUs | lab reproduces 20 published results unaided; harness benchmark beats the human-written seed genome |
| v2 | C+5 on | flagship gen 1 | full org (14.4) | first promoted weights trained on lab-made environments and trajectories |
| v3 | gen 1 + ~3 months | gen 1.k | full org | lab-written next-gen proposal passes rungs 1 to 3 |

What has to exist for v0: kernel scheduler, budget and sandbox; ledger with SQL and
embedding search; fetch mirror; eval-gate with the first held-out suite; seed genome
for all roles; the promote pipeline; kernel monitors and kill switch. Everything
else can be grown by the lab. The kernel itself already exists by then, because it is
the same harness that wrote the codebase (section 15).
