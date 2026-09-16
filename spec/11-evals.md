## 11. Evals

Run continuously on checkpoints, with a fixed harness version per report.

- **Research**: RE-Bench, MLE-bench, PaperBench, an internal speedrun-style suite,
  METR time-horizon.
- **Code**: SWE-bench Verified and Pro, Terminal-Bench, held-out competitive
  programming (post-cutoff problems only).
- **Math**: AIME/HMMT (post-cutoff years), FrontierMath, miniF2F/PutnamBench in Lean.
- **ARC**: ARC-AGI-1, -2 public eval, -3; semi-private through the official process.
- **General**: HLE, GPQA, long-context retrieval and reasoning at 128k/1M.
- **Forecasting**: Brier and log score relative to market on questions resolving
  after the checkpoint's cutoff, paper P&L with recorded-book slippage, and the
  leak probe from 9.3.
- **Latent scaling**: accuracy against latent budget (1x to 16x) and against discrete
  CoT at matched FLOP, per suite, on every checkpoint (4.5).
- **Efficiency**: tokens-to-solve, latent steps, recurrence iterations, wall-clock and
  $ per solved task. Reported next to every accuracy number.
- **Contamination**: canary strings, rephrased-question probes, post-cutoff splits.
