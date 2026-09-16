## 1. Hardware facts the design depends on

- **220k GB200 GPUs ≈ 3,056 NVL72 racks.** One rack: 72 Blackwell GPUs, 36 Grace
  CPUs, ~13.4 TB HBM total (~186 GB per GPU), one NVLink 5 domain at 1.8 TB/s per GPU.
- **Grace memory is close.** Each Grace has 480 GB LPDDR5X on NVLink-C2C (900 GB/s)
  to its two GPUs. That's ~240 GB of "slow HBM" per GPU. The design uses it for
  optimizer state offload in training and as the second KV tier in serving.
- **Scale-out** is 800G per GPU (ConnectX-8, InfiniBand or Spectrum-X). Cross-rack
  bandwidth is ~20x worse than in-rack. Rule: all-to-all stays inside a rack; only
  gradient reduction and pipeline sends cross racks.
- **Blackwell supports FP8 and NVFP4** in hardware. FP4 cuts serving memory ~4x vs
  BF16, which matters because it lets a multi-trillion-parameter MoE fit on one rack.
- **Power**: order of 120 to 140 kW per rack, so roughly 400 MW of IT load. If that's
  split across buildings, section 5.6 applies.
- **Failures are the normal state.** Llama 3 reported 419 unexpected interruptions in
  54 days on 16k H100s, about one every 3 hours. Linear scaling to 220k gives one
  every ~13 minutes, and early GB200 fleets have been less reliable than mature H100
  fleets. Anything that needs a full-job restart per failure will not train.

Compute budget. Assume ~1e15 effective FLOP/s per GPU after MFU (measure this on day
one and re-plan). The fleet then produces ~2.2e20 FLOP/s, or ~1.9e27 FLOP per 100
days, about 100x GPT-4-era pretraining.
