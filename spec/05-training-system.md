## 5. Training system (JAX)

### 5.1 Why JAX on GPUs, and what it's missing

JAX + Rust at frontier scale has precedent (Grok-1 was trained on that combination),
and XLA on Blackwell works. But JAX's own heavy use is on TPUs, and at this GPU count
several pieces are ours to build:

| Gap | Plan |
|---|---|
| Pipeline parallelism | SPMD circular pipeline (Praxis-style: `scan` over stages inside `shard_map`, `ppermute` between stages), rewritten for FLOP-balanced stages |
| MoE all-to-all | custom XLA op wrapping an NCCL/NVSHMEM EP dispatch/combine kernel (DeepEP-style, re-bound for JAX) |
| NVFP4 / FP8 linears | Transformer Engine JAX bindings where they work; custom cuDNN/Triton kernels behind `jax.custom_vjp` where they don't |
| Linear attention kernels | Triton kernels (chunked delta rule) via custom primitive, forward + backward |
| ~55k controller processes | `jax.distributed` coordination was not built for this; run it per replica group with a Rust control plane on top (5.5) |
| Compile time | cache compiled XLA programs by content hash on shared storage; target <10 min cold start at full scale |

### 5.2 Parallelism layout (main pretraining, ~100k GPUs)

- **EP = 72** (one rack): ~7 experts per GPU per layer, all-to-all over NVLink only.
- **FSDP over the rack** for attention, dense and shared-expert params (ZeRO-3 over
  NVLink, cheap).
- **PP = 12**, one rack per stage, stages balanced by measured FLOPs not layer count.
  The recurrent core block gets extra stage capacity because it runs r times.
- **DP ≈ 115 replicas** (864 GPUs each) over InfiniBand.
- **Context parallel** (ring over NVLink) only in the 256k/1M mid-training phases.
- **Optimizer state on Grace.** FP32 master weights and Muon momentum live in LPDDR5X
  over NVLink-C2C; the GPU holds BF16/FP8 weights, grads and activations. Per GPU:
  ~8.5e9 expert params × (BF16 weights + BF16 grads) ≈ 34 GB on GPU and ~68 GB on
  Grace. That leaves ~150 GB of HBM for activations with selective remat.

Batch: 64M to 80M tokens per step after warmup (Kimi K2 used ~67M), roughly 3 s/step
target, ~2.2M steps over 80 days. Check this on the ladder; critical batch size under
Muon isn't settled.

### 5.3 Precision

- Default: FP8 for linears (per-block scaling), BF16 for router, norms, embeddings,
  softmax, final 2 layers, latent adapter.
- NVFP4 forward for routed expert weights once rung 3 shows no loss gap. Public
  results show NVFP4 pretraining working at 12B scale on 10T tokens; nobody has shown
  it at 7T total, so this is gated.
- Master weights FP32 on Grace. Gradient reduce in BF16.

### 5.4 Optimizer

**MuonClip** (Muon + QK-clip, as in Kimi K2, which trained 1T params over 15.5T tokens
without loss spikes) for all 2D matrices; AdamW for embeddings, norms, router biases,
latent σ. Per-expert Newton-Schulz runs locally on each GPU since expert matrices are
unsharded within EP; FSDP-sharded matrices gather in-rack for NS. Hyperparameter
transfer by μP-for-Muon across the ladder. Muon keeps one state per param instead of
Adam's two, which is what makes Grace offload fit.

Schedule: WSD (warmup, stable, decay), so a checkpoint from the stable phase can
branch into mid-training without restarting.

### 5.5 Fault tolerance (Rust control plane)

Target: a GPU failure costs under 5 minutes of fleet time and never a full restart.

- **Elastic DP.** A failed replica drops out; the other 114 keep going, with grad
  accumulation raised to hold tokens per step fixed. A spare rack is healed in and
  rejoins at the next step boundary, with weights copied from a live replica over IB.
- **In-memory checkpoints** every ~10 minutes to Grace RAM on two other racks
  (cross-rack redundancy); persistent async checkpoints every ~2 hours to the object
  store. Checkpoint size is ~90 TB (weights + FP32 master + momentum); only one DP
  replica's shards get written.
- **Silent data corruption detector.** DP replicas must hold bit-identical weights.
  Hash each shard every N steps and compare across replicas; a mismatch quarantines
  the rack. It costs almost nothing and catches the failure you otherwise find three
  weeks later as a loss curve that won't come down.
- **Straggler detection** from per-rank step timing; slow ranks are drained the same
  way as dead ones.
- **Health**: Xid/ECC events, NVLink error counters, NIC flaps, thermal throttling,
  fed into the scheduler before failures turn into hangs. Every collective gets a
  watchdog timeout.
- **Loss-spike policy**: automatic rollback to the last in-memory checkpoint, skip the
  offending data shard, log it. A human is paged on the second spike within 10k steps.

### 5.6 Multi-site

If the 220k GPUs sit in more than one building with thin links between them, don't
run synchronous DP across sites. Use DiLoCo-style local steps within a site and
infrequent cross-site parameter averaging, and validate the loss penalty on the ladder
first.
