"""FastAPI application for stock analysis and OroTitan AI services."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import __version__
from ..orotitan_ai.behavior import analyze_behavior

LOGGER = logging.getLogger("stock_analysis.api")

app = FastAPI(title="Indice Nexus API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegimeWeights(BaseModel):
    weights: Dict[str, float] = Field(default_factory=dict)


class EmbedRequest(BaseModel):
    tickers: List[str]
    regime: Optional[str] = None
    regime_weights: Optional[Dict[str, float]] = None
    as_of: Optional[datetime] = None
    period: str = "1y"
    interval: str = "1d"
    price_column: str = "Close"
    risk_budget: float = 0.1
    temperature: float = 0.0
    behavior: bool = False
    behavior_threshold: Optional[float] = None
    behavior_persist: Optional[str] = None


class DecisionRequest(BaseModel):
    tickers: List[str]
    regime: Optional[str] = None
    regime_weights: Optional[Dict[str, float]] = None
    as_of: Optional[datetime] = None
    period: str = "1y"
    interval: str = "1d"
    price_column: str = "Close"
    risk_budget: float = 0.1
    temperature: float = 0.0
    behavior: bool = False
    behavior_threshold: Optional[float] = None
    behavior_persist: Optional[str] = None


class FeedbackEvent(BaseModel):
    ticker: str
    outcome: Literal["hit", "miss", "neutral"] = "neutral"
    note: Optional[str] = None
    weight: Optional[float] = Field(default=1.0, ge=0)
    horizon_days: Optional[int] = Field(default=5, ge=1)


class FeedbackRequest(BaseModel):
    events: List[FeedbackEvent]


class ReportRequest(BaseModel):
    tickers: List[str]
    regime: Optional[str] = None
    regime_weights: Optional[Dict[str, float]] = None
    include_decisions: bool = True
    include_embeddings: bool = False
    as_of: Optional[datetime] = None
    period: str = "1y"
    interval: str = "1d"
    price_column: str = "Close"
    risk_budget: float = 0.1
    temperature: float = 0.0
    title: Optional[str] = None
    behavior: bool = False
    behavior_threshold: Optional[float] = None
    behavior_persist: Optional[str] = None


class DecisionRecord(BaseModel):
    ticker: str
    decision: str
    score: float
    rationale: Optional[str] = None
    regime: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    behavior: Optional[Dict[str, Any]] = None


class DecisionResponse(BaseModel):
    decisions: List[DecisionRecord]
    run: Dict[str, Any]


class EmbedResponse(BaseModel):
    embeddings: Dict[str, Dict[str, List[float]]]
    run: Dict[str, Any]


class FeedbackResponse(BaseModel):
    updated: int
    notes: List[str]
    run: Dict[str, Any]


class ReportResponse(BaseModel):
    markdown: str
    saved_to: Optional[str] = None
    run: Dict[str, Any]


class BehaviorRequest(BaseModel):
    tickers: List[str]
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class BehaviorResponse(BaseModel):
    behavioral_score: float
    confidence_adjustment: float
    top_biases: List[Dict[str, Any]]
    recommendations: List[str]
    tags: List[str]
    run: Dict[str, Any]


def _runtime_helpers():
    from ..cli import orotitan_ai as orotitan_cli  # local import to keep health light

    return orotitan_cli


def _build_run_payload(started: datetime) -> Dict[str, Any]:
    duration = round((datetime.utcnow() - started).total_seconds(), 4)
    return {
        "started": started.isoformat() + "Z",
        "duration_s": duration,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/ai/behavior/analyze", response_model=BehaviorResponse)
def behavior_analyze(request: BehaviorRequest) -> BehaviorResponse:
    started = datetime.utcnow()
    options = request.options or {}
    threshold = float(options.get("threshold", 30.0))
    max_influence = float(options.get("max_influence", 0.25))
    persist = str(options.get("persist", "none"))
    analysis = analyze_behavior(
        request.tickers,
        request.context or {},
        threshold=threshold,
        max_influence=max_influence,
        persist=persist,
    )
    payload = analysis.dict()
    return BehaviorResponse(
        behavioral_score=payload["behavioral_score"],
        confidence_adjustment=payload["confidence_adjustment"],
        top_biases=payload["top_biases"],
        recommendations=payload["recommendations"],
        tags=payload["tags"],
        run=_build_run_payload(started),
    )


@app.post("/ai/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    started = datetime.utcnow()
    helper = _runtime_helpers()
    try:
        pipeline = helper.run_pipeline(
            request.tickers,
            period=request.period,
            interval=request.interval,
            price_column=request.price_column,
            risk_budget=request.risk_budget,
            temperature=request.temperature,
            regime_override=request.regime,
            regime_weights=request.regime_weights,
            behavior_enabled=request.behavior,
            behavior_threshold=request.behavior_threshold or 30.0,
            behavior_mode=request.behavior_persist or "jsonl",
        )
    except Exception as exc:
        LOGGER.error("Embedding failed", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    embeddings = {
        ticker: {timestamp.isoformat(): vector for timestamp, vector in mapping.items()}
        for ticker, mapping in pipeline.embeddings.items()
    }
    return EmbedResponse(embeddings=embeddings, run=_build_run_payload(started))


@app.post("/ai/decide", response_model=DecisionResponse)
def decide(request: DecisionRequest) -> DecisionResponse:
    started = datetime.utcnow()
    helper = _runtime_helpers()
    try:
        pipeline = helper.run_pipeline(
            request.tickers,
            period=request.period,
            interval=request.interval,
            price_column=request.price_column,
            risk_budget=request.risk_budget,
            temperature=request.temperature,
            regime_override=request.regime,
            regime_weights=request.regime_weights,
            behavior_enabled=request.behavior,
            behavior_threshold=request.behavior_threshold or 30.0,
            behavior_mode=request.behavior_persist or "jsonl",
        )
    except Exception as exc:
        LOGGER.error("Decision generation failed", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    decisions = [
        DecisionRecord(
            ticker=decision.ticker,
            decision=decision.action,
            score=decision.score,
            rationale=decision.rationale,
            regime=pipeline.regime,
            meta={
                "confidence": decision.confidence,
                "size": decision.size,
                "factors": decision.factors,
            },
            behavior=decision.behavior.dict() if decision.behavior is not None else None,
        )
        for decision in pipeline.decisions
    ]
    return DecisionResponse(decisions=decisions, run=_build_run_payload(started))


@app.post("/ai/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    started = datetime.utcnow()
    helper = _runtime_helpers()
    state = helper.load_state()
    weights = state.get("weights", {})
    events = [event.dict() for event in request.events]
    updated, notes = helper.apply_feedback_events(weights, events)
    return FeedbackResponse(updated=len(events), notes=notes, run=_build_run_payload(started))


@app.post("/ai/report", response_model=ReportResponse)
def report(request: ReportRequest) -> ReportResponse:
    started = datetime.utcnow()
    helper = _runtime_helpers()
    try:
        pipeline = helper.run_pipeline(
            request.tickers,
            period=request.period,
            interval=request.interval,
            price_column=request.price_column,
            risk_budget=request.risk_budget,
            temperature=request.temperature,
            regime_override=request.regime,
            regime_weights=request.regime_weights,
            behavior_enabled=request.behavior,
            behavior_threshold=request.behavior_threshold or 30.0,
            behavior_mode=request.behavior_persist or "jsonl",
        )
    except Exception as exc:
        LOGGER.error("Report pipeline failed", exc_info=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    markdown_body = helper.render_markdown_report(
        pipeline.decisions,
        pipeline.weights,
        pipeline.kpis,
        title=request.title or "OroTitan Cognitive Report",
    )
    saved_path: Optional[str] = None
    if os.getenv("SAVE_REPORTS") == "1":
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        result = helper.render_orotitan_report(
            pipeline.decisions,
            pipeline.weights,
            pipeline.kpis,
            reports_dir,
            title=request.title or "OroTitan Cognitive Report",
        )
        saved = result.get("md")
        if saved:
            saved_path = str(saved)
    return ReportResponse(
        markdown=markdown_body,
        saved_to=saved_path,
        run=_build_run_payload(started),
    )
