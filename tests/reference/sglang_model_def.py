"""Independent C2 reference: SGLang discrete model definition (spec 3.1).

Plain Python, slow and obvious. Does not import ``sglang_fork.model_def``
or production ``model``. Enums and the 3:1 / dense-then-MoE / last-core
rules match A1 names and counts; they are re-implemented here.

Index 0 is linear attention. Period is LINEAR_TO_MLA + 1 (four unique
layers: three linear, then one MLA). First ``n_dense`` unique layers are
dense FFN; the rest are MoE. The core block is the last
``core_block_layers`` unique layers. MTP heads must be 2.
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
_PERIOD = LINEAR_TO_MLA + 1
_MTP_HEADS = 2


@dataclass(frozen=True)
class SglangLayer:
    index: int
    attention: AttentionKind
    ffn: FfnKind
    in_core_block: bool


@dataclass(frozen=True)
class SglangModelDef:
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


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ModelDefError(message)


def _in_range(layer_index: int, n_layers: int) -> bool:
    return 0 <= layer_index < n_layers


def attention_kind(layer_index: int, n_layers: int) -> AttentionKind:
    """3:1 linear:MLA repeating. Index 0 is linear."""
    _require(
        _in_range(layer_index, n_layers),
        f"layer_index {layer_index} outside [0, {n_layers})",
    )
    # Position LINEAR_TO_MLA in each period of 4 is MLA; the other three
    # (including index 0) are linear.
    if layer_index % _PERIOD == LINEAR_TO_MLA:
        return AttentionKind.MLA
    return AttentionKind.LINEAR


def ffn_kind(layer_index: int, n_dense: int, n_layers: int) -> FfnKind:
    """First ``n_dense`` unique layers are dense; the rest are MoE."""
    _require(
        _in_range(layer_index, n_layers),
        f"layer_index {layer_index} outside [0, {n_layers})",
    )
    if layer_index < n_dense:
        return FfnKind.DENSE
    return FfnKind.MOE


def in_core_block(layer_index: int, n_layers: int, core_block_layers: int) -> bool:
    """True if the unique layer is in the last ``core_block_layers`` layers."""
    _require(
        _in_range(layer_index, n_layers),
        f"layer_index {layer_index} outside [0, {n_layers})",
    )
    return layer_index >= n_layers - core_block_layers


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
    """Build a definition and fill ``layers``. Raises ModelDefError on mismatch."""
    _require(n_layers >= 1, f"n_layers must be >= 1, got {n_layers}")
    _require(n_dense >= 0, f"n_dense must be >= 0, got {n_dense}")
    _require(n_moe >= 0, f"n_moe must be >= 0, got {n_moe}")
    _require(
        n_dense + n_moe == n_layers,
        f"n_dense + n_moe ({n_dense} + {n_moe}) != n_layers ({n_layers})",
    )
    _require(n_linear_attn >= 0, f"n_linear_attn must be >= 0, got {n_linear_attn}")
    _require(n_mla >= 0, f"n_mla must be >= 0, got {n_mla}")
    _require(
        n_linear_attn + n_mla == n_layers,
        f"n_linear_attn + n_mla ({n_linear_attn} + {n_mla}) != n_layers ({n_layers})",
    )
    _require(
        n_linear_attn == LINEAR_TO_MLA * n_mla,
        f"attention is not {LINEAR_TO_MLA}:1 (n_linear_attn={n_linear_attn}, n_mla={n_mla})",
    )
    _require(
        mtp_heads == _MTP_HEADS,
        f"mtp_heads must be {_MTP_HEADS}, got {mtp_heads}",
    )
    _require(
        1 <= core_block_layers <= n_layers,
        f"core_block_layers {core_block_layers} outside [1, {n_layers}]",
    )
    _require(d_model >= 1, f"d_model must be >= 1, got {d_model}")
    _require(
        n_routed_experts >= 1,
        f"n_routed_experts must be >= 1, got {n_routed_experts}",
    )
    _require(
        n_shared_experts >= 0,
        f"n_shared_experts must be >= 0, got {n_shared_experts}",
    )
    _require(top_k >= 1, f"top_k must be >= 1, got {top_k}")
    _require(
        top_k <= n_routed_experts,
        f"top_k {top_k} > n_routed_experts {n_routed_experts}",
    )
    _require(expert_hidden >= 1, f"expert_hidden must be >= 1, got {expert_hidden}")
    _require(recurrence_max >= 1, f"recurrence_max must be >= 1, got {recurrence_max}")
    _require(
        recurrence_train_mean > 0,
        f"recurrence_train_mean must be > 0, got {recurrence_train_mean}",
    )
    _require(vocab_size >= 1, f"vocab_size must be >= 1, got {vocab_size}")
    _require(max_context >= 1, f"max_context must be >= 1, got {max_context}")
    _require(max_racks >= 1, f"max_racks must be >= 1, got {max_racks}")

    layers = []
    for i in range(n_layers):
        layers.append(
            SglangLayer(
                index=i,
                attention=attention_kind(i, n_layers),
                ffn=ffn_kind(i, n_dense, n_layers),
                in_core_block=in_core_block(i, n_layers, core_block_layers),
            )
        )
    filled = tuple(layers)

    n_lin = sum(1 for layer in filled if layer.attention == AttentionKind.LINEAR)
    n_mla_got = sum(1 for layer in filled if layer.attention == AttentionKind.MLA)
    n_dense_got = sum(1 for layer in filled if layer.ffn == FfnKind.DENSE)
    n_moe_got = sum(1 for layer in filled if layer.ffn == FfnKind.MOE)
    n_core = sum(1 for layer in filled if layer.in_core_block)
    _require(n_lin == n_linear_attn, f"filled linear {n_lin} != {n_linear_attn}")
    _require(n_mla_got == n_mla, f"filled mla {n_mla_got} != {n_mla}")
    _require(n_dense_got == n_dense, f"filled dense {n_dense_got} != {n_dense}")
    _require(n_moe_got == n_moe, f"filled moe {n_moe_got} != {n_moe}")
    _require(
        n_core == core_block_layers,
        f"filled core {n_core} != {core_block_layers}",
    )

    return SglangModelDef(
        d_model=d_model,
        n_layers=n_layers,
        n_dense=n_dense,
        n_moe=n_moe,
        n_linear_attn=n_linear_attn,
        n_mla=n_mla,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        top_k=top_k,
        expert_hidden=expert_hidden,
        mtp_heads=mtp_heads,
        core_block_layers=core_block_layers,
        recurrence_train_mean=recurrence_train_mean,
        recurrence_max=recurrence_max,
        vocab_size=vocab_size,
        max_context=max_context,
        max_racks=max_racks,
        layers=filled,
    )


def mtp_head_count(defn: SglangModelDef) -> int:
    """Extra next-token heads. Spec 3.1 is 2."""
    _require(
        defn.mtp_heads == _MTP_HEADS,
        f"mtp_heads must be {_MTP_HEADS}, got {defn.mtp_heads}",
    )
    return defn.mtp_heads


def layer_table(defn: SglangModelDef) -> tuple[SglangLayer, ...]:
    """``defn.layers`` after validation. Length equals ``n_layers``."""
    _require(
        len(defn.layers) == defn.n_layers,
        f"len(layers) {len(defn.layers)} != n_layers {defn.n_layers}",
    )
    for i, layer in enumerate(defn.layers):
        _require(layer.index == i, f"layers[{i}].index is {layer.index}")
        want_attn = attention_kind(i, defn.n_layers)
        _require(
            layer.attention == want_attn,
            f"layers[{i}].attention {layer.attention} != {want_attn}",
        )
        want_ffn = ffn_kind(i, defn.n_dense, defn.n_layers)
        _require(
            layer.ffn == want_ffn,
            f"layers[{i}].ffn {layer.ffn} != {want_ffn}",
        )
        want_core = in_core_block(i, defn.n_layers, defn.core_block_layers)
        _require(
            layer.in_core_block == want_core,
            f"layers[{i}].in_core_block {layer.in_core_block} != {want_core}",
        )
    return defn.layers
