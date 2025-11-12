"""Strategic Nexus report generation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .regimes import RegimeAssessment
from .report import build_summary_table, render_html


def _format_float(value: object, *, digits: int = 2, suffix: str = "") -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.{digits}f}{suffix}"


def _render_weights(weights: Mapping[str, float]) -> List[str]:
    lines = ["| Module | Poids |", "| --- | --- |"]
    for key in ("trend", "momentum", "quality", "risk"):
        value = weights.get(key, 0.0)
        lines.append(f"| {key.capitalize()} | {_format_float(value * 100, digits=1, suffix='%')} |")
    return lines


def _top_tickers(summary_rows: Sequence[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    sorted_rows = sorted(
        summary_rows,
        key=lambda row: float(row.get("Score") or 0.0),
        reverse=True,
    )
    return sorted_rows[: max(1, limit)]


def _behavioural_comment(rows: Iterable[Dict[str, object]]) -> str:
    rows = list(rows)
    if not rows:
        return "Aucune donnée comportementale disponible."
    avg_momentum = sum(float(row.get("Momentum") or 0.0) for row in rows) / len(rows)
    avg_risk = sum(float(row.get("Risk") or 0.0) for row in rows) / len(rows)
    avg_quality = sum(float(row.get("Quality") or 0.0) for row in rows) / len(rows)

    parts = []
    if avg_momentum >= 20:
        parts.append("Momentum robuste sur l'échantillon.")
    elif avg_momentum <= 10:
        parts.append("Momentum atone à court terme.")
    if avg_risk <= 5:
        parts.append("Volatilité contrôlée sur les valeurs suivies.")
    else:
        parts.append("Risque accru, privilégier une taille de position maîtrisée.")
    if avg_quality >= 12:
        parts.append("Fondamentaux solides en moyenne.")
    else:
        parts.append("Qualité des fondamentaux à surveiller sur plusieurs dossiers.")
    return " ".join(parts)


def _recommendation(avg_score: float) -> str:
    if avg_score >= 65:
        return "Renforcer"
    if avg_score <= 40:
        return "Alléger"
    return "Surveiller"


def build_markdown(
    results: Dict[str, Dict[str, object]],
    assessment: RegimeAssessment,
    weights: Mapping[str, float],
    *,
    title: str = "Nexus Market Report",
    top_n: int = 10,
) -> str:
    summary_rows = build_summary_table(results)
    top_rows = _top_tickers(summary_rows, top_n)
    avg_score = sum(float(row.get("Score") or 0.0) for row in top_rows) / len(top_rows)

    snapshot = assessment.snapshot
    macro_lines = [
        "| Indicateur | Valeur |",
        "| --- | --- |",
        f"| Régime | {assessment.regime} |",
        f"| VIX | {_format_float(snapshot.vix)} |",
        f"| Inflation (YoY) | {_format_float(snapshot.cpi_yoy, suffix='%')} |",
        f"| Taux 10Y | {_format_float(snapshot.rate_10y, suffix='%')} |",
        f"| Taux 2Y | {_format_float(snapshot.rate_2y, suffix='%')} |",
        f"| Spread crédit | {_format_float(snapshot.credit_spread)} |",
    ]

    lines: List[str] = [f"# {title}"]
    lines.append("")
    lines.append(
        f"_Généré le {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
    )
    lines.append("")
    lines.append("## Contexte macro")
    lines.extend(macro_lines)
    lines.append("")

    lines.append("## Pondérations Nexus")
    lines.extend(_render_weights(weights))
    lines.append("")

    lines.append(f"## Top {len(top_rows)} valeurs")
    lines.extend(_render_top_table(top_rows))
    lines.append("")

    lines.append("## Analyse comportementale")
    lines.append(_behavioural_comment(top_rows))
    lines.append("")

    lines.append("## Recommandation synthétique")
    lines.append(f"Recommandation actuelle : **{_recommendation(avg_score)}**.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_top_table(rows: Sequence[Dict[str, object]]) -> List[str]:
    headers = [
        "Ticker",
        "Score",
        "Trend",
        "Momentum",
        "Quality",
        "Risk",
        "AsOf",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [
            str(row.get("Ticker", "")),
            _format_float(row.get("Score")),
            _format_float(row.get("Trend")),
            _format_float(row.get("Momentum")),
            _format_float(row.get("Quality")),
            _format_float(row.get("Risk")),
            str(row.get("AsOf", "")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def generate_nexus_report(
    results: Dict[str, Dict[str, object]],
    assessment: RegimeAssessment,
    weights: Mapping[str, float],
    *,
    output_dir: Path,
    base_name: str = "Nexus",
    title: str = "Nexus Market Report",
    top_n: int = 10,
    include_html: bool = True,
) -> Dict[str, str | None]:
    """Generate Nexus markdown/HTML reports and return the file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(results, assessment, weights, title=title, top_n=top_n)

    date_stamp = assessment.snapshot.date.strftime("%Y%m%d")
    markdown_path = output_dir / f"{base_name}_{date_stamp}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    html_path: Path | None = None
    if include_html:
        html_content = render_html(markdown)
        html_path = output_dir / f"{base_name}_{date_stamp}.html"
        html_path.write_text(html_content, encoding="utf-8")

    return {
        "markdown": str(markdown_path),
        "html": str(html_path) if html_path else None,
    }


__all__ = ["build_markdown", "generate_nexus_report"]
