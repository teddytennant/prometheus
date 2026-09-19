"""Independent PyTorch flagship-shape forward for V1 logit parity (spec 16.2).

Not JAX ``model.forward``. Not NumPy. Not imported from ``tests/``.
Production ``prometheus.verify.v1_parity.run_v1`` compares JAX logits to
this module. Gate: finite ``max |a - b|`` <= 1e-5 in FP32 on
``model.tiny_config``. CPU analog stays callable without a GPU; that does
not count as V1 verified.

Torch is the independent stack. Do not re-export NumPy math wrapped in
``torch.tensor``.
"""

from __future__ import annotations

from typing import Any

from model import ModelConfig


def forward(
    tokens: Any,
    params: dict[str, Any],
    config: ModelConfig,
    *,
    r: int | None = None,
) -> Any:
    """Next-token logits, same architecture/params as production ``model.forward``.

    Implementation is PyTorch (CPU is enough). Returns logits of shape
    ``(batch, seq, vocab_size)`` in float32.
    """
    raise NotImplementedError
