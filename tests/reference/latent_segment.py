"""Independent reference for spec 4.2 layout and spec 4.3 segment decode.

Numpy math only. Does not call the production functions under test (they
raise). Types come from ``model.latent`` so a test can compare spans by
value. Production must not import this module.
"""

from __future__ import annotations

import math

import numpy as np

from model.latent import (
    LatentConfig,
    LatentError,
    SegmentDecoder,
    SpanKind,
    StageBObjective,
    ThinkSpan,
)

_EPS = 1e-6


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_float(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def plan_think_layout(
    chunk_lengths: tuple[int, ...] | list[int],
    anchor_token_ids: tuple[int, ...] | list[int],
    config: LatentConfig,
) -> tuple[ThinkSpan, ...]:
    if len(chunk_lengths) == 0:
        raise LatentError("chunk_lengths empty")
    for n in chunk_lengths:
        if not _is_int(n) or n < config.chunk_min or n > config.chunk_max:
            raise LatentError("chunk length out of range")
    if len(anchor_token_ids) == 0:
        raise LatentError("anchor empty")
    for tid in anchor_token_ids:
        if not _is_int(tid) or tid < 0:
            raise LatentError("anchor token id")
    anchors = tuple(int(t) for t in anchor_token_ids)
    spans: list[ThinkSpan] = []
    for i, n in enumerate(chunk_lengths):
        spans.append(ThinkSpan(SpanKind.LATENT, int(n), ()))
        if i != len(chunk_lengths) - 1:
            spans.append(ThinkSpan(SpanKind.ANCHOR, 0, anchors))
    return tuple(spans)


def _latent_indexes(layout: tuple[ThinkSpan, ...]) -> list[int]:
    return [i for i, s in enumerate(layout) if s.kind is SpanKind.LATENT]


def _validate_layout(layout: tuple[ThinkSpan, ...]) -> list[int]:
    starts = len(layout) == 0 or layout[0].kind is not SpanKind.LATENT
    ends = len(layout) == 0 or layout[-1].kind is not SpanKind.LATENT
    if starts or ends:
        raise LatentError("layout")
    latents = _latent_indexes(layout)
    if len(latents) == 0:
        raise LatentError("layout")
    for a, b in zip(latents, latents[1:], strict=False):
        gap = layout[a + 1 : b]
        if any(s.kind is SpanKind.LATENT for s in gap):
            raise LatentError("layout")
        anchors = [s for s in gap if s.kind is SpanKind.ANCHOR]
        if len(anchors) != 1:
            raise LatentError("layout")
        # Anchor must come before any tool in the gap.
        kinds = [s.kind for s in gap]
        if SpanKind.TOOL in kinds and kinds.index(SpanKind.ANCHOR) > kinds.index(SpanKind.TOOL):
            raise LatentError("layout")
    return latents


def insert_discrete_tool(
    layout: tuple[ThinkSpan, ...],
    after_chunk: int,
    tool_token_ids: tuple[int, ...] | list[int],
) -> tuple[ThinkSpan, ...]:
    try:
        latents = _validate_layout(layout)
    except LatentError:
        raise
    if not _is_int(after_chunk) or after_chunk < 0 or after_chunk >= len(latents):
        raise LatentError("after_chunk")
    if len(tool_token_ids) == 0:
        raise LatentError("tool empty")
    for tid in tool_token_ids:
        if not _is_int(tid) or tid < 0:
            raise LatentError("tool token id")
    tool = ThinkSpan(SpanKind.TOOL, 0, tuple(int(t) for t in tool_token_ids))
    latent_at = latents[after_chunk]
    if after_chunk == len(latents) - 1:
        insert_at = len(layout)
    else:
        insert_at = latents[after_chunk + 1]
    # Append at the end of the gap: just before the next latent, after
    # anchors and tools already there.
    del latent_at
    out = list(layout)
    out.insert(insert_at, tool)
    return tuple(out)


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _tied(h: np.ndarray, unembed: np.ndarray, final_norm: np.ndarray | None) -> np.ndarray:
    x = h
    if final_norm is not None:
        w = final_norm
        var = np.mean(x * x)
        x = x * (1.0 / np.sqrt(var + _EPS)) * w
    return x @ unembed.T


def decode_thought_segment(
    hidden,
    teacher_len: int,
    unembed,
    decoder: SegmentDecoder | None = None,
    final_norm=None,
):
    if not _is_int(teacher_len) or teacher_len < 1:
        raise LatentError("teacher_len")
    h = _as_np(hidden)
    if h.ndim != 1 or h.shape[0] < 1 or not np.isfinite(h).all():
        raise LatentError("hidden")
    u = _as_np(unembed)
    if u.ndim != 2 or u.shape[0] < 1 or u.shape[1] != h.shape[0] or not np.isfinite(u).all():
        raise LatentError("unembed")
    fn = None
    if final_norm is not None:
        fn = _as_np(final_norm)
        if fn.shape != (h.shape[0],) or not np.isfinite(fn).all():
            raise LatentError("final_norm")
    if teacher_len == 1:
        return _tied(h, u, fn)
    if decoder is None:
        raise LatentError("decoder")
    w_in = _as_np(decoder.w_in)
    w_h = _as_np(decoder.w_h)
    bias = _as_np(decoder.bias)
    d = h.shape[0]
    if w_in.shape != (d, d) or w_h.shape != (d, d) or bias.shape != (d,) or not (
        np.isfinite(w_in).all() and np.isfinite(w_h).all() and np.isfinite(bias).all()
    ):
        raise LatentError("decoder")
    state = np.zeros(d, dtype=np.float64)
    rows = []
    for _ in range(teacher_len):
        state = np.tanh(h @ w_in + state @ w_h + bias)
        rows.append(_tied(state, u, fn))
    return np.stack(rows, axis=0)


def codi_alignment(student_h, teacher_h):
    s = _as_np(student_h)
    if s.ndim != 2 or s.shape[0] < 1 or s.shape[1] < 1 or not np.isfinite(s).all():
        raise LatentError("student_h")
    t = _as_np(teacher_h)
    if t.ndim != 2 or t.shape[0] < 1 or t.shape[1] < 1 or not np.isfinite(t).all():
        raise LatentError("teacher_h")
    if s.shape != t.shape:
        raise LatentError("shape")
    return float(np.mean((s - t) ** 2))


def stage_b_objective(
    halt_logits,
    answer_logits,
    thought_logits,
    teacher_ids,
    thought_mask,
    answer_id,
    teacher_depth,
    config,
    student_h,
    teacher_h,
    *,
    align_weight: float = 1.0,
):
    if not _is_finite_float(align_weight) or float(align_weight) < 0.0:
        raise LatentError("align_weight")
    from model.latent import stage_b_loss

    base = stage_b_loss(
        halt_logits,
        answer_logits,
        thought_logits,
        teacher_ids,
        thought_mask,
        answer_id,
        teacher_depth,
        config,
    )
    l_align = codi_alignment(student_h, teacher_h)
    return StageBObjective(
        stage_b=base,
        l_align=float(l_align),
        total=float(base.total) + float(align_weight) * float(l_align),
    )
