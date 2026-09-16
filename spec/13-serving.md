## 13. Serving: modified SGLang

SGLang already supports much of what's needed (RadixAttention, hierarchical KV cache
via HiCache, hybrid linear-attention state caches, MLA, EP, multi-LoRA, speculative
decoding). Changes:

1. **Weight loading from JAX checkpoints** (Rust converter → safetensors-compatible
   FP4/FP8 shards) with a numeric parity suite against the JAX reference.
2. **Latent decode step.** The request state machine gets `latent` alongside `verbal`.
   A latent step takes an input embedding from the previous hidden state (through the
   latent adapter) instead of a token id. Latent and verbal requests share one batch
   and one forward; CUDA graphs are captured for both input paths.
3. **Depth recurrence.** Per-request iteration count from the halting head, capped by
   a request-level latent budget so a caller can buy more steps on a hard problem,
   bucketed
   (r ∈ {1, 2, 4, 8, 16}) so batches stay regular; KV shared across iterations.
4. **KV tiering (swap).** GPU HBM → Grace LPDDR5X over NVLink-C2C → local NVMe →
   distributed KV store. Only the 24 MLA layers carry growing KV; the 72 linear layers
   hold fixed-size state, so parking a 1M-token agent session is far cheaper than for a
   full-attention model. Long agent sessions waiting on tools or sub-agents get swapped
   to tier 2/3 and restored on the next turn.
5. **MTP speculative decoding** with the 2 MTP heads.
6. **Routing capture** for RL routing replay (9.2).
7. **Sub-agent scheduling.** Agents spawned by one orchestrator share the parent's
   prefix KV and are scheduled as a group.
8. **TTT sidecar hook** for ARC-style per-task LoRA (section 10).

Deployment unit: one NVL72 serves one full model replica (EP=72, weights ~4.1 TB in
NVFP4), leaving ~9 TB HBM for KV plus Grace tier 2.
