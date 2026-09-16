## 10. ARC-AGI

- **Pretraining**: procedural grid tasks at scale, one token per cell, 2D RoPE on grid
  spans, augmentation (8 dihedral transforms × color permutations) applied during
  training.
- **Test-time training.** For each task, fine-tune a small LoRA (or optimize a handful
  of prefix latents, which is cheaper) on augmented copies of the demonstration pairs,
  then predict. This was the technique behind the strongest 2024 ARC Prize entries.
  At serve time it runs in a JAX sidecar on the same rack; SGLang hot-loads the LoRA.
- **Ensembling**: predictions under inverse augmentations, voted.
- **Program synthesis path**: model writes a program in a grid DSL, executes it on the
  demonstrations, keeps programs consistent with every pair. This takes the pass@2
  slots when a consistent program exists.
- **ARC-AGI-3 (interactive)**: game-like environments in the RL fleet with sparse
  reward, exploration bonus, and a learned world-model tool (predict next frame, used
  for planning). Evaluated only on held-out games.
- **Small recursive models** (HRM/TRM-style, millions of params) get strong ARC scores
  for their size. Worth a side experiment as a tool the big model can train at test
  time, not as a replacement.
- Eval sets are never trained on; decontamination covers them (7.2).
