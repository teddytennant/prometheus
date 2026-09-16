## 8. Mid-training

Starts from a late stable-phase checkpoint.

1. **Long context**: 16k → 256k → 1M, raising the RoPE base on MLA layers, with
   long-document and repo-level data and synthetic retrieval/multi-hop tasks.
2. **Reasoning and agentic data** upweighted: verified solutions, tool trajectories,
   research transcripts (papers + code + results).
3. **Recurrence depth widened**: the r distribution (4.1) shifts toward the tail on
   math and code, so the halt learns to spend more on hard tokens. Stage B waits for
   the Stage A reasoner (4.3).
4. **Context management skills**: notes-file read/write tools, self-summarization of
   old context, retrieval over its own history. RL makes these good later; mid-training
   teaches the format.

Then a short SFT cold start on high-quality reasoning and agent traces, in
discrete format, before RL.
