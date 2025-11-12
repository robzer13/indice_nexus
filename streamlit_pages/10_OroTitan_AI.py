"""Streamlit page exposing the OroTitan AI decision engine."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Mapping

import streamlit as st

USE_DIRECT = os.getenv("IMPORT_DIRECT", "1") == "1"
API_URL = os.getenv("API_URL", "http://localhost:8000")

if USE_DIRECT:
    from stock_analysis.cli.orotitan_ai import (
        apply_feedback_events,
        render_markdown_report,
        run_pipeline,
    )
else:  # pragma: no cover - exercised in real deployments
    import urllib.request


def _parse_tickers(raw: str) -> List[str]:
    tokens = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]
    return sorted(set(tokens))


def _parse_weights(text: str) -> Dict[str, float]:
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Les pondérations doivent être un objet JSON.")
    return {str(key): float(value) for key, value in data.items()}


def _call_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
    request = urllib.request.Request(
        url=f"{API_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _run_decisions(tickers: List[str], weights: Dict[str, float], regime: str | None, risk: float, temperature: float) -> Dict[str, Any]:
    if not tickers:
        raise ValueError("Aucun ticker fourni")
    started = time.perf_counter()
    if USE_DIRECT:
        pipeline = run_pipeline(
            tickers,
            regime_override=regime,
            regime_weights=weights if weights else None,
            risk_budget=risk,
            temperature=temperature,
        )
        payload = {
            "decisions": [
                {
                    "ticker": decision.ticker,
                    "decision": decision.action,
                    "score": decision.score,
                    "confidence": decision.confidence,
                    "rationale": decision.rationale,
                    "regime": pipeline.regime,
                    "factors": decision.factors,
                }
                for decision in pipeline.decisions
            ],
            "weights": pipeline.weights,
            "kpis": pipeline.kpis,
            "embeddings": {
                ticker: {ts.isoformat(): vec for ts, vec in mapping.items()}
                for ticker, mapping in pipeline.embeddings.items()
            },
            "run": {
                "started": time.time(),
                "duration_s": round(time.perf_counter() - started, 4),
            },
        }
        return payload
    payload = {
        "tickers": tickers,
        "regime": regime,
        "regime_weights": weights or None,
        "risk_budget": risk,
        "temperature": temperature,
    }
    response = _call_api("/ai/decide", payload)
    return response


def _run_report(tickers: List[str], weights: Dict[str, float], regime: str | None, risk: float, temperature: float) -> Dict[str, Any]:
    if USE_DIRECT:
        pipeline = run_pipeline(
            tickers,
            regime_override=regime,
            regime_weights=weights if weights else None,
            risk_budget=risk,
            temperature=temperature,
        )
        markdown_body = render_markdown_report(
            pipeline.decisions,
            pipeline.weights,
            pipeline.kpis,
        )
        return {"markdown": markdown_body, "weights": pipeline.weights}
    payload = {
        "tickers": tickers,
        "regime": regime,
        "regime_weights": weights or None,
        "risk_budget": risk,
        "temperature": temperature,
    }
    return _call_api("/ai/report", payload)


def _apply_feedback(weights: Dict[str, float], events_text: str) -> Dict[str, Any]:
    if not events_text.strip():
        raise ValueError("Aucun évènement de feedback fourni")
    events = json.loads(events_text)
    if not isinstance(events, list):
        raise ValueError("Le feedback doit être une liste JSON")
    if USE_DIRECT:
        updated, notes = apply_feedback_events(weights, events)
        return {"weights": updated, "notes": notes}
    response = _call_api("/ai/feedback", {"events": events})
    return {"weights": response.get("updated"), "notes": response.get("notes", [])}


def main() -> None:
    st.title("OroTitan AI")
    st.sidebar.header("Paramètres")
    raw_tickers = st.sidebar.text_input("Tickers", value="MC.PA ASML TTE.PA")
    regime = st.sidebar.text_input("Régime", value="")
    weight_text = st.sidebar.text_area("Pondérations (JSON)", value="")
    risk_budget = st.sidebar.slider("Risk budget", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    temperature = st.sidebar.slider("Température", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    run_decisions = st.sidebar.button("Analyser")
    run_report = st.sidebar.button("Générer rapport")

    st.sidebar.markdown("---")
    feedback_text = st.sidebar.text_area("Feedback JSON")
    apply_feedback = st.sidebar.button("Appliquer feedback")

    tickers = _parse_tickers(raw_tickers)
    try:
        weights = _parse_weights(weight_text)
    except ValueError as exc:
        st.error(str(exc))
        weights = {}

    if run_decisions:
        if not tickers:
            st.info("Merci de saisir au moins un ticker.")
        else:
            with st.spinner("Calcul des décisions..."):
                try:
                    payload = _run_decisions(tickers, weights, regime or None, risk_budget, temperature)
                except Exception as exc:  # pragma: no cover - runtime failures only
                    st.error(f"Erreur lors du calcul: {exc}")
                else:
                    decisions = payload.get("decisions", [])
                    if decisions:
                        table = [
                            {
                                "Ticker": entry["ticker"],
                                "Décision": entry["decision"],
                                "Score": round(entry["score"], 3),
                                "Confiance": round(entry.get("confidence", 0.0), 3),
                                "Rationale": entry.get("rationale", "")[:120],
                            }
                            for entry in decisions
                        ]
                        st.dataframe(table)
                        st.download_button(
                            "Télécharger JSON",
                            data=json.dumps(payload, indent=2),
                            file_name="orotitan_decisions.json",
                            mime="application/json",
                        )
                    else:
                        st.warning("Aucune décision générée.")

    if run_report:
        if not tickers:
            st.info("Merci de saisir au moins un ticker.")
        else:
            with st.spinner("Génération du rapport..."):
                try:
                    payload = _run_report(tickers, weights, regime or None, risk_budget, temperature)
                except Exception as exc:  # pragma: no cover
                    st.error(f"Impossible de générer le rapport: {exc}")
                else:
                    st.markdown(payload.get("markdown", ""))
                    st.download_button(
                        "Télécharger le rapport",
                        data=payload.get("markdown", ""),
                        file_name="orotitan_report.md",
                        mime="text/markdown",
                    )

    if apply_feedback:
        try:
            result = _apply_feedback(weights, feedback_text)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - runtime only
            st.error(f"Erreur feedback: {exc}")
        else:
            st.success("Feedback appliqué")
            st.json(result)


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
