"""Synth package (spec 7.1)."""

from synth.agentic import agentic
from synth.arc_procedural import arc_procedural
from synth.generators import MathItem, code_task, contrastive_pair, eval_expr, math_items
from synth.rejection import rejection_sample
from synth.rephrase import ngram_coverage, rephrase

__all__ = [
    "MathItem",
    "eval_expr",
    "math_items",
    "contrastive_pair",
    "code_task",
    "rephrase",
    "ngram_coverage",
    "rejection_sample",
    "arc_procedural",
    "agentic",
]
