"""Reporting helpers to summarise analysis results."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Dict, Iterable, List

LOGGER = logging.getLogger(__name__)

_FUNDAMENTAL_KEYS = ["pe_ratio", "net_margin_pct", "debt_to_equity", "dividend_yield_pct"]
_SUMMARY_HEADERS = [
    "Ticker",
    "Score",
    "Trend",
    "Momentum",
    "Quality",
    "Risk",
    "AsOf",
    "Dups",
    "OHLC_Anom",
    "Gaps",
    "MissingFundamentals",
]


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN check
        return None
    return numeric


def _extract_count(section: object, key: str) -> int:
    if not isinstance(section, dict):
        return 0
    payload = section.get(key)
    if isinstance(payload, dict):
        value = payload.get("count")
    else:
        value = payload
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _resolve_as_of(payload: Dict[str, object]) -> str:
    score = payload.get("score", {}) if isinstance(payload, dict) else {}
    candidate = None
    if isinstance(score, dict):
        candidate = score.get("as_of")
    if candidate is None and isinstance(payload.get("fundamentals"), dict):
        candidate = payload["fundamentals"].get("as_of")
    if candidate is None:
        prices = payload.get("prices")
        if hasattr(prices, "index") and getattr(prices, "empty", True) is False:
            try:
                candidate = prices.index[-1]
            except Exception:  # pragma: no cover - defensive fallback
                candidate = None
    if isinstance(candidate, datetime):
        return candidate.isoformat()
    if candidate is None:
        return ""
    return str(candidate)


def build_summary_table(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    """Return a list of summary rows aggregating score and quality metrics."""

    rows: List[Dict[str, object]] = []
    for ticker, payload in results.items():
        score = payload.get("score", {}) if isinstance(payload, dict) else {}
        fundamentals = payload.get("fundamentals", {}) if isinstance(payload, dict) else {}
        quality = payload.get("quality", {}) if isinstance(payload, dict) else {}

        if not isinstance(score, dict):
            score = {}
        if not isinstance(fundamentals, dict):
            fundamentals = {}

        missing_fundamentals = sum(1 for key in _FUNDAMENTAL_KEYS if fundamentals.get(key) is None)

        row = {
            "Ticker": ticker,
            "Score": score.get("score"),
            "Trend": score.get("trend"),
            "Momentum": score.get("momentum"),
            "Quality": score.get("quality"),
            "Risk": score.get("risk"),
            "AsOf": _resolve_as_of(payload),
            "Dups": _extract_count(quality, "duplicates"),
            "OHLC_Anom": _extract_count(quality, "ohlc_anomalies"),
            "Gaps": _extract_count(quality, "gaps"),
            "MissingFundamentals": missing_fundamentals,
        }
        rows.append(row)
    return rows


def _score_level(score: float | None) -> str:
    if score is None:
        return "indisponible"
    if score < 40:
        return "faible"
    if score < 70:
        return "modéré"
    return "élevé"


def _format_component(name: str, value: float | None) -> str:
    labels = {
        "trend": "tendance",
        "momentum": "momentum",
        "quality": "qualité",
        "risk": "risque",
    }
    label = labels.get(name, name)
    if value is None:
        return f"{label} indisponible"
    return f"{label} à {value:.2f} pts"


def format_commentary(ticker: str, bundle: Dict[str, object]) -> str:
    """Produce a short textual commentary for a given ticker."""

    score_bundle = bundle.get("score", {}) if isinstance(bundle, dict) else {}
    fundamentals = bundle.get("fundamentals", {}) if isinstance(bundle, dict) else {}
    report_meta = bundle.get("report", {}) if isinstance(bundle, dict) else {}

    if not isinstance(score_bundle, dict):
        score_bundle = {}
    if not isinstance(fundamentals, dict):
        fundamentals = {}
    if not isinstance(report_meta, dict):
        report_meta = {}

    score_value = _safe_float(score_bundle.get("score"))
    level = _score_level(score_value)
    if score_value is None:
        headline = f"{ticker}: score global indisponible pour le moment."
    else:
        headline = f"{ticker}: score global {level} ({score_value:.2f}/100)."

    components = {name: _safe_float(score_bundle.get(name)) for name in ("trend", "momentum", "quality", "risk")}
    valid_components = {name: value for name, value in components.items() if value is not None}
    if valid_components:
        best_component = max(valid_components, key=lambda key: valid_components[key])
        strength = _format_component(best_component, valid_components[best_component])
        strength_sentence = f"Point fort actuel : {strength}."
    else:
        strength_sentence = "Les composantes détaillées ne sont pas disponibles."

    cautions: List[str] = []
    risk_value = components.get("risk")
    if risk_value is not None and risk_value <= 3:
        cautions.append(f"Risque limité à surveiller (score {risk_value:.2f}/10).")

    missing_fundamentals = sum(1 for key in _FUNDAMENTAL_KEYS if fundamentals.get(key) is None)
    if missing_fundamentals:
        cautions.append(f"Données fondamentales incomplètes : {missing_fundamentals} indicateur(s) manquant(s).")

    report_notes = report_meta.get("notes") if isinstance(report_meta.get("notes"), list) else []
    if report_notes:
        cautions.append("Notes supplémentaires : " + ", ".join(str(note) for note in report_notes))

    if not cautions:
        cautions.append("Vigilance de rigueur face aux évolutions de marché.")

    sentences = [headline, strength_sentence]
    sentences.extend(cautions[:2])  # limiter à 4 phrases au total
    return " ".join(sentences[:4])


def _format_value(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        if isinstance(value, str):
            return value
        return "n/a"
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.2f}"


def _render_table(rows: Iterable[Dict[str, object]], headers: List[str]) -> List[str]:
    rows = list(rows)
    if not rows:
        return ["_Aucune donnée disponible._"]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header)
            if header in {"Score", "Trend", "Momentum", "Quality", "Risk"}:
                cells.append(_format_value(value))
            else:
                cells.append(str(value) if value not in (None, "") else "n/a")
        body.append("| " + " | ".join(cells) + " |")
    return [header_line, separator, *body]


def render_markdown(
    results: Dict[str, Dict[str, object]],
    *,
    title: str = "Stock Analysis Report",
    include_charts: bool = True,
    charts_dir: str | None = None,
) -> str:
    """Render a markdown report containing a global summary and per-ticker blocks."""

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: List[str] = [f"# {title}", "", f"_Généré le {timestamp} UTC_", "", "## Synthèse", ""]

    summary_rows = build_summary_table(results)
    lines.extend(_render_table(summary_rows, _SUMMARY_HEADERS))
    lines.append("")

    for ticker, payload in results.items():
        lines.append(f"### {ticker}")
        fundamentals = payload.get("fundamentals", {}) if isinstance(payload, dict) else {}
        if not isinstance(fundamentals, dict):
            fundamentals = {}
        metric_rows = [
            {"Métrique": "P/E", "Valeur": _format_value(fundamentals.get("pe_ratio"))},
            {"Métrique": "Marge nette (%)", "Valeur": _format_value(fundamentals.get("net_margin_pct"))},
            {"Métrique": "Dette/Capitaux propres", "Valeur": _format_value(fundamentals.get("debt_to_equity"))},
            {"Métrique": "Rendement dividende (%)", "Valeur": _format_value(fundamentals.get("dividend_yield_pct"))},
        ]
        lines.extend(_render_table(metric_rows, ["Métrique", "Valeur"]))
        lines.append("")
        lines.append(format_commentary(ticker, payload))
        lines.append("")

        if include_charts:
            report_meta = payload.get("report", {}) if isinstance(payload, dict) else {}
            if isinstance(report_meta, dict):
                if charts_dir and report_meta.get("chart_filename"):
                    chart_path = os.path.join(charts_dir, str(report_meta["chart_filename"]))
                    lines.append(f"![{ticker} chart]({chart_path})")
                    lines.append("")
                else:
                    data_url = report_meta.get("chart_data_url")
                    if isinstance(data_url, str) and data_url:
                        lines.append(f"![{ticker} chart]({data_url})")
                        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_html(markdown_str: str) -> str:
    """Convert markdown to HTML when the `markdown` library is available."""

    try:
        import markdown  # type: ignore
    except Exception:  # pragma: no cover - optional dependency missing
        LOGGER.info("Bibliothèque markdown indisponible, retour du markdown brut")
        return markdown_str
    return markdown.markdown(markdown_str)


__all__ = [
    "build_summary_table",
    "format_commentary",
    "render_markdown",
    "render_html",
]

