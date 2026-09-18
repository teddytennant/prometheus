"""SGLang model definition matching A1 (spec 3.1, 13.1, 15.5 C2).

Hybrid 3:1 linear vs MLA, sigmoid-gated MoE, MTP (2 extra heads), 8-layer
core iterated r times. This is the architecture SGLang loads. Weight
conversion is C1; this module does not convert or load tensors.

Enums and counts match A1 ``model.ModelConfig``. C2 does not import
``model`` so the fork stays standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AttentionKind(StrEnum):
    LINEAR = "linear"
    MLA = "mla"


class FfnKind(StrEnum):
    DENSE = "dense"
    MOE = "moe"


class ModelDefError(ValueError):
    """Definition violates a 3.1 invariant (counts, 3:1 pattern, MTP)."""


LINEAR_TO_MLA = 3


@dataclass(frozen=True)
class SglangLayer:
    """One unique layer. Index 0 is linear attention (spec 3.1)."""

    index: int
    attention: AttentionKind
    ffn: FfnKind
    in_core_block: bool


@dataclass(frozen=True)
class SglangModelDef:
    """Discrete architecture SGLang instantiates.

    ``n_layers`` is unique layers, not unrolled recurrence. The core
    block is the last ``core_block_layers`` unique layers and is
    iterated ``recurrence_max`` times at most.
    """

    d_model: int
    n_layers: int
    n_dense: int
    n_moe: int
    n_linear_attn: int
    n_mla: int
    n_routed_experts: int
    n_shared_experts: int
    top_k: int
    expert_hidden: int
    mtp_heads: int
    core_block_layers: int
    recurrence_train_mean: float
    recurrence_max: int
    vocab_size: int
    max_context: int
    max_racks: int = 4
    layers: tuple[SglangLayer, ...] = ()


def attention_kind(layer_index: int, n_layers: int) -> AttentionKind:
    """3:1 linear:MLA repeating. Index 0 is linear. Raises ModelDefError
    if ``layer_index`` is outside ``[0, n_layers)``.
    """
    raise NotImplementedError("C2 attention_kind")


def ffn_kind(layer_index: int, n_dense: int, n_layers: int) -> FfnKind:
    """First ``n_dense`` unique layers are dense; the rest are MoE."""
    raise NotImplementedError("C2 ffn_kind")


def in_core_block(layer_index: int, n_layers: int, core_block_layers: int) -> bool:
    """True if the unique layer is in the iterated core block."""
    raise NotImplementedError("C2 in_core_block")


def from_counts(
    d_model: int,
    n_layers: int,
    n_dense: int,
    n_moe: int,
    n_linear_attn: int,
    n_mla: int,
    n_routed_experts: int,
    n_shared_experts: int,
    top_k: int,
    expert_hidden: int,
    mtp_heads: int,
    core_block_layers: int,
    recurrence_train_mean: float,
    recurrence_max: int,
    vocab_size: int,
    max_context: int,
    max_racks: int = 4,
) -> SglangModelDef:
    """Build a definition and fill ``layers``. Raises ModelDefError on
    count mismatch (n_dense+n_moe, 3:1 attention, mtp_heads != 2, …).
    """
    raise NotImplementedError("C2 from_counts")


def mtp_head_count(defn: SglangModelDef) -> int:
    """Extra next-token heads. Spec 3.1 is 2."""
    raise NotImplementedError("C2 mtp_head_count")


def layer_table(defn: SglangModelDef) -> tuple[SglangLayer, ...]:
    """``defn.layers`` after validation. Length equals ``n_layers``."""
    raise NotImplementedError("C2 layer_table")
