## 3. Model architecture

### 3.1 Starting config (the ladder refits it)

| Item | Value | Why |
|---|---|---|
| Total params | ~7.4T | fits in NVFP4 on one NVL72 (~4.1 TB with scales) with KV room |
| Active params | ~0.4T unique, ~0.45T compute-active (recurrence) | inference cost vs capability |
| Layers | 96 unique; 3 dense, 93 MoE | |
| Depth recurrence | 8-layer core block iterated r times (train mean r=3) | latent compute, section 4 |
| d_model | 12,288 | |
| Attention | 3:1 hybrid: 72 linear-attention layers (Gated DeltaNet / KDA family), 24 MLA full-attention layers | constant-size state on 3/4 of layers; long context at low KV cost |
| MoE | 512 routed experts, top-20, 2 shared, expert hidden 4,096 | fine-grained experts + shared, DeepSeek-V3 style |
| Routing | sigmoid gates, aux-loss-free bias balancing, node-limited routing (max 4 racks) | balance without an aux loss hurting quality |
| MTP | 2 extra-token heads | denser training signal; free speculative decoding at serve |
| Vocab | 256k byte-level BPE; one token per ARC grid cell color | fewer tokens per word (token efficiency) |
| Positions | RoPE on MLA layers (partial dim); 2D RoPE on grid spans | ARC grids are 2D |
| Norm/stability | pre-norm RMSNorm, QK-norm on MLA, logit soft-capping, z-loss on router | loss spikes at this scale are expensive |
| Context | 16k pretrain → 256k → 1M mid-training | long-horizon agents |

Check on the parameter count: one expert is 3 × 12,288 × 4,096 ≈ 1.5e8 params; 512 of
them over 93 MoE layers comes to ~7.2T, plus attention, dense layers and embeddings
≈ 7.4T. Active: 22 experts × 1.5e8 + ~6e8 attention ≈ 3.9e9 per layer, ~0.37T over
96 layers, ~0.45T once the core block runs 3 times.

### 3.2 Things deliberately left out

- **Pure dense.** At a fixed serving budget, dense trades too much capability.
- **Byte-level / BLT tokenization at the flagship scale.** It's promising, but the
  serving stack and the latent-reasoning work are enough new risk for one run. Rerun
  the question at ladder rung 3.
- **Full-attention-only.** A 1M-token KV cache on 96 full layers would dominate
  serving memory even with MLA.
