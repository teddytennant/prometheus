## 19. Open risks, ranked

The binding constraint isn't on this list, because it isn't technical: there is no
cluster, and nothing past rung 1 trains without compute from somewhere. The risks
below are ranked by how much they threaten the result.

1. **More latent steps stop buying better answers at scale.** Never done at
   frontier scale, though everything at small scale says it works. The downside is
   bounded: the latent branch is ~5% of the run, and the CoT model it starts from is
   the model if it misses. Mitigation: the curve is measured at every rung (4.5) and
   published either way.
2. **JAX-on-GPU at 55k controllers** (coordination, compile time, pipeline). Mitigation:
   burn-in week at full scale on an 8B shape before the flagship; Rust control plane
   owns membership instead of `jax.distributed` alone.
3. **Fleet reliability.** Mitigation: elastic DP, in-memory checkpoints, SDC hashing,
   hot spares; measured at burn-in, not assumed.
4. **Trainer/SGLang numeric mismatch destabilizing RL.** Mitigation: recomputed
   log-probs, routing replay, TIS, parity halt.
5. **NVFP4 at 7.4T.** Mitigation: FP8 is the default; FP4 is only an upgrade.
6. **Reward hacking in long-horizon and research environments.** Mitigation: 9.3
   controls, held-out verifier rotation, human audits of top-reward trajectories.
7. **Data licensing and contamination.** Mitigation: fail-closed decontamination,
   provenance tracked per shard.
8. **The RSI loop optimizes its metrics instead of research.** Mitigation: kernel-owned
   graders and held-out suites, quarterly rotation, mandatory replication, rising
   replication-failure rate treated as an alarm (14.9, 14.10).
9. **Small-scale wins don't transfer.** The lab's speed depends on rung 0/1 proxies.
   Mitigation: the what-transfers table (14.5) measures proxy validity per technique
   family, and families with poor transfer are forced to start at rung 2.
10. **Lab speed outruns human review.** At thousands of experiments a day, humans
    can't read everything. Mitigation: humans own a small number of gates (grader
    review, weight promotion, next-gen allocation, tripwire pauses) rather than
    reviewing the stream, and the weekly Director report is audited against the
    ledger.
11. **Generated code passes its tests and is still wrong.** Mitigation: tests and
    implementations from agents that never see each other's work, tests that must
    fail a stub first, independent reference implementations, human review of
    high-risk interfaces (15.3).
12. **One provider, one OAuth session.** Swarm size is capped by a consumer
    subscription and a single rotating refresh token. Mitigation: token broker with
    Raft-committed rotation, API-key and self-hosted fallbacks (15.2).
13. **H200 results don't carry over to GB200.** Mitigation: H200s prove correctness
    only; everything about scale, speed, NVFP4 and reliability is listed in 16.4 and
    tested at burn-in against its H200 baseline.
