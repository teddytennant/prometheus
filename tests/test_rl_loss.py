"""GSPO / DAPO / Dr. GRPO (D1)."""

from __future__ import annotations

import numpy as np

from rl.loss import advantages, drop_zero_advantage_groups, gspo_loss, sequence_logprob


def test_no_length_norm():
    a = sequence_logprob(np.array([[-1.0, -1.0, -1.0]]))
    b = sequence_logprob(np.array([[-1.0]]))
    assert a[0] == -3.0
    assert b[0] == -1.0


def test_drop_uniform_group():
    keep = drop_zero_advantage_groups(np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.5]]))
    assert not keep[0]
    assert keep[1]


def test_gspo_clipped():
    adv = advantages(np.array([1.0, 0.0, 0.0, 0.0]))
    loss = gspo_loss(np.array([0.0, -1, -1, -1]), np.array([-0.2, -1, -1, -1]), adv)
    assert np.isfinite(loss)
