## 12. Isolation and audit

Safety monitoring isn't a goal of this model. These are here because RL and the lab
produce wrong numbers without them.

- **RL environment isolation**: no live internet, no credentials in sandboxes, egress
  blocked, tampering detection (9.3). A policy that can reach the grader learns the
  grader.
- **Discrete mode for debugging**: every checkpoint can be forced into discrete CoT.
  When latent and discrete answers diverge on the same problem, that is the first
  place to look for a latent training bug.
- **Thought decoding** (4.3): the decode head reads latent chunks back out as steps,
  which is how latent-mode failures get diagnosed.
