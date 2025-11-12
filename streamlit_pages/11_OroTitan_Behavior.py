"""Streamlit page exposing the OroTitan behavioural intelligence layer."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from stock_analysis.orotitan_ai.behavior import analyze_behavior

st.set_page_config(page_title="OroTitan Behaviour", layout="wide")
st.title("OroTitan Behavioral Intelligence")

sidebar = st.sidebar
raw_tickers = sidebar.text_input("Tickers", value="MC.PA TTE.PA ASML")
threshold = sidebar.slider("Behavior threshold", min_value=10, max_value=80, value=30, step=5)
influence = sidebar.slider("Max influence (%)", min_value=5, max_value=25, value=25, step=5)
raw_overrides = sidebar.text_area("Indicator overrides (JSON)", value="{}")
run_button = sidebar.button("Run Decisions")
markdown_button = sidebar.button("Export Markdown")

if not raw_tickers.strip():
    st.info("Enter at least one ticker to analyse behavioural patterns.")
    st.stop()

try:
    overrides = json.loads(raw_overrides) if raw_overrides.strip() else {}
except json.JSONDecodeError as exc:  # pragma: no cover - UI feedback
    st.error(f"Invalid JSON overrides: {exc}")
    overrides = {}

tickers = [token.strip() for token in raw_tickers.replace(",", " ").split() if token.strip()]
if not tickers:
    st.warning("No valid tickers supplied.")
    st.stop()

context: Dict[str, Any] = {}
if isinstance(overrides, dict):
    context["indicator_overrides"] = overrides

if run_button:
    analysis = analyze_behavior(
        tickers,
        context,
        threshold=float(threshold),
        max_influence=float(influence) / 100.0,
        persist="none",
    )
    st.metric("Behavioral Score", f"{analysis.behavioral_score:.1f}", help="0 = sain, 100 = biais marqué")
    st.metric("Confidence Adjustment", f"{analysis.confidence_adjustment:+.3f}")

    if analysis.top_biases:
        table_rows: List[Dict[str, Any]] = []
        for bias in analysis.top_biases:
            table_rows.append(
                {
                    "Bias": bias.bias_id,
                    "Score": round(bias.score, 2),
                    "Rationale": bias.rationale,
                }
            )
        st.subheader("Top Biases")
        st.dataframe(pd.DataFrame(table_rows))

    st.subheader("Self-Coaching Actions")
    for action in analysis.recommendations[:5]:
        st.write(f"- {action}")

    if markdown_button:
        snippet_lines = ["## Behavioral Insights", f"Score: {analysis.behavioral_score:.1f}"]
        for bias in analysis.top_biases:
            snippet_lines.append(f"- {bias.bias_id}: {bias.score:.2f} — {bias.rationale}")
        snippet_lines.append("## Self-Coaching Actions")
        snippet_lines.extend(f"- {action}" for action in analysis.recommendations[:5])
        markdown_body = "\n".join(snippet_lines)
        st.download_button(
            "Download Markdown",
            data=markdown_body,
            file_name="behavior_section.md",
            mime="text/markdown",
        )
else:
    st.info("Configure parameters then click *Run Decisions* to compute behavioural insights.")
