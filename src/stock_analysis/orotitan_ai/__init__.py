"""OroTitan AI modules providing embeddings, decisions, and feedback handling."""
from __future__ import annotations

from .behavior import analyze_behavior, BehaviorAnalysis, BiasSignal, BehaviorIndicator
from .embedding import (
    build_feature_dict,
    embed_tickers,
    encode_inputs,
    encode_snapshot,
    encode_text_optional,
)
from .decision_engine import Decision, decide_for_date as decide
from .feedback import (
    FeedbackItem,
    apply_feedback,
    load_state,
    save_state,
    update_weights,
)
from ..report_orotitan import render_markdown_report

__all__ = [
    "BehaviorAnalysis",
    "BehaviorIndicator",
    "BiasSignal",
    "Decision",
    "FeedbackItem",
    "analyze_behavior",
    "apply_feedback",
    "build_feature_dict",
    "decide",
    "embed_tickers",
    "encode_inputs",
    "encode_snapshot",
    "encode_text_optional",
    "load_state",
    "render_markdown_report",
    "save_state",
    "update_weights",
]
