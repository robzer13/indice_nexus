"""Rendering utilities for the OroTitan cognitive report."""
from __future__ import annotations

"""Rendering helpers for OroTitan cognitive reports."""

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except Exception:  # pragma: no cover - handle missing dependency
    Environment = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import markdown
except Exception:  # pragma: no cover
    markdown = None  # type: ignore

from .orotitan_ai.decision_engine import Decision

logger = logging.getLogger(__name__)
_TEMPLATE_NAME = "report_orotitan.md.j2"


def _get_template_environment() -> Environment | None:
    if Environment is None:
        return None
    templates_dir = Path(__file__).resolve().parents[2] / "templates"
    loader = FileSystemLoader(str(templates_dir))
    return Environment(loader=loader, undefined=StrictUndefined, autoescape=False)


def _serialise_decisions(decisions: Iterable[Decision]) -> List[Dict[str, Any]]:
    serialised: List[Dict[str, Any]] = []
    for decision in decisions:
        payload = asdict(decision)
        payload["date"] = decision.date.isoformat()
        behaviour = payload.pop("behavior", None)
        if behaviour is not None and hasattr(decision.behavior, "dict"):
            payload["behavior"] = decision.behavior.dict()
        serialised.append(payload)
    return serialised


def _summarise_behavior(decisions: Iterable[Decision]) -> Dict[str, Any] | None:
    behaviours = [decision.behavior for decision in decisions if decision.behavior is not None]
    if not behaviours:
        return None
    score = sum(item.behavioral_score for item in behaviours) / len(behaviours)
    adjustment = sum(item.confidence_adjustment for item in behaviours) / len(behaviours)
    top_biases: Dict[str, Dict[str, Any]] = {}
    recommendations: List[str] = []
    for analysis in behaviours:
        for bias in analysis.top_biases:
            if bias.bias_id not in top_biases or bias.score > top_biases[bias.bias_id]["score"]:
                top_biases[bias.bias_id] = {
                    "score": bias.score,
                    "rationale": bias.rationale,
                }
        for action in analysis.recommendations:
            if action not in recommendations:
                recommendations.append(action)
    summary = {
        "score": score,
        "adjustment": adjustment,
        "biases": sorted(
            (
                {
                    "bias_id": bias_id,
                    "score": data["score"],
                    "rationale": data["rationale"],
                }
                for bias_id, data in top_biases.items()
            ),
            key=lambda entry: entry["score"],
            reverse=True,
        )[:3],
        "recommendations": recommendations[:5],
    }
    return summary


def render_markdown_report(
    decisions: Iterable[Decision],
    nexus_weights: Dict[str, float],
    kpis: Dict[str, float],
    *,
    title: str = "OroTitan Cognitive Report",
) -> str:
    """Return the Markdown representation of an OroTitan report."""

    env = _get_template_environment()
    decisions_list = list(decisions)
    as_of = max((decision.date for decision in decisions_list), default=datetime.now(timezone.utc))
    behavior_summary = _summarise_behavior(decisions_list)
    context = {
        "title": title,
        "as_of": as_of.isoformat(),
        "weights": sorted(nexus_weights.items(), key=lambda item: item[0]),
        "decisions": _serialise_decisions(decisions_list),
        "kpis": sorted(kpis.items(), key=lambda item: item[0]),
        "behavior": behavior_summary,
    }

    if env is not None and env.loader.searchpath:  # type: ignore[attr-defined]
        template = env.get_template(_TEMPLATE_NAME)
        return template.render(context)

    logger.warning("Jinja2 not available; generating minimal Markdown report")
    lines = [f"# {title}", f"Generated: {context['as_of']}"]
    lines.append("## Decisions")
    for payload in context["decisions"]:
        lines.append(
            "- {ticker}: {action} score={score:.2f} confidence={confidence:.2f}".format(
                **payload
            )
        )
    lines.append("## Weights")
    lines.extend(f"- {name}: {value:.2f}" for name, value in context["weights"])
    lines.append("## KPIs")
    lines.extend(f"- {name}: {value:.2f}" for name, value in context["kpis"])
    behavior_summary = context.get("behavior")
    if behavior_summary:
        lines.append("## Behavioral Insights")
        lines.append(
            f"Score: {behavior_summary['score']:.1f} | Adjustment: {behavior_summary['adjustment']:.3f}"
        )
        for bias in behavior_summary.get("biases", []):
            lines.append(
                "- {bias_id}: {score:.2f} — {rationale}".format(
                    bias_id=bias["bias_id"], score=bias["score"], rationale=bias["rationale"]
                )
            )
        if behavior_summary.get("recommendations"):
            lines.append("## Self-Coaching Actions")
            lines.extend(f"- {item}" for item in behavior_summary["recommendations"])
    return "\n".join(lines)


def render_orotitan_report(
    decisions: List[Decision],
    nexus_weights: Dict[str, float],
    kpis: Dict[str, float],
    out_dir: Path,
    title: str = "OroTitan Cognitive Report",
) -> Dict[str, Path]:
    """Render Markdown and optional HTML reports summarising OroTitan output."""

    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_body = render_markdown_report(decisions, nexus_weights, kpis, title=title)

    slug = title.lower().replace(" ", "_")
    md_path = out_dir / f"{slug}.md"
    md_path.write_text(markdown_body, encoding="utf-8")
    logger.info("OroTitan Markdown report written", extra={"path": str(md_path)})

    html_path = out_dir / f"{slug}.html"
    if markdown is not None:
        html_path.write_text(markdown.markdown(markdown_body), encoding="utf-8")
    else:
        html_path.write_text(f"<pre>{markdown_body}</pre>", encoding="utf-8")
    logger.info("OroTitan HTML report written", extra={"path": str(html_path)})

    return {"md": md_path, "html": html_path}

