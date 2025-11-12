"""Rule-based conversion of behavioural indicators into bias scores."""
from __future__ import annotations

from typing import Dict, Iterable, List

from .bias_bank import BiasDefinition, get_bias_definitions
from .schemas import BehaviorIndicator, BiasSignal


def _score_with_threshold(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    span = max(high - low, 1e-6)
    return float((value - low) / span)


def evaluate_biases(indicators: Dict[str, BehaviorIndicator]) -> List[BiasSignal]:
    """Return scored biases using heuristics over the computed indicators."""

    scores: List[BiasSignal] = []
    lookup = indicators

    def indicator(name: str) -> BehaviorIndicator:
        return lookup.get(name, BehaviorIndicator(name=name, value=0.0))

    def build_signal(bias: BiasDefinition, value: float, related: Iterable[str], rationale: str) -> None:
        capped = max(0.0, min(1.0, value))
        if capped <= 0:
            return
        scores.append(
            BiasSignal(
                bias_id=bias.bias_id,
                score=capped,
                indicators=[indicator(name) for name in related],
                rationale=rationale,
            )
        )

    biases = {bias.bias_id: bias for bias in get_bias_definitions()}

    build_signal(
        biases["confirmation"],
        value=max(
            _score_with_threshold(indicator("plan_deviation").value, 0.25, 0.6),
            _score_with_threshold(1.0 - indicator("turnover_ratio").value, 0.4, 0.8),
        ),
        related=["plan_deviation", "turnover_ratio"],
        rationale="Plan rarement remis en question malgré signaux contraires.",
    )

    build_signal(
        biases["recency"],
        value=_score_with_threshold(indicator("turnover_ratio").value, 0.35, 0.9),
        related=["turnover_ratio"],
        rationale="Entrées/sorties fréquentes suggérant un biais de récence.",
    )

    build_signal(
        biases["overconfidence"],
        value=max(
            _score_with_threshold(indicator("position_size_variance").value, 0.3, 0.8),
            _score_with_threshold(indicator("win_loss_asymmetry").value, 0.4, 0.9),
        ),
        related=["position_size_variance", "win_loss_asymmetry"],
        rationale="Variabilité des tailles ou asymétrie gains/pertes élevée.",
    )

    build_signal(
        biases["loss_aversion"],
        value=max(
            _score_with_threshold(indicator("loss_hold_bias").value, 0.2, 0.8),
            _score_with_threshold(indicator("drawdown_streak").value, 0.3, 0.8),
        ),
        related=["loss_hold_bias", "drawdown_streak"],
        rationale="Les pertes sont conservées plus longtemps que les gains.",
    )

    build_signal(
        biases["fomo"],
        value=_score_with_threshold(indicator("chasing_intensity").value, 0.15, 0.6),
        related=["chasing_intensity"],
        rationale="Entrées tardives répétées après un mouvement rapide.",
    )

    build_signal(
        biases["disposition"],
        value=max(
            _score_with_threshold(indicator("loss_hold_bias").value, 0.25, 0.75),
            _score_with_threshold(indicator("plan_deviation").value, 0.3, 0.75),
        ),
        related=["loss_hold_bias", "plan_deviation"],
        rationale="Gestion asymétrique des gains/pertes identifiée.",
    )

    build_signal(
        biases["familiarity"],
        value=_score_with_threshold(indicator("concentration_ratio").value, 0.4, 0.9),
        related=["concentration_ratio"],
        rationale="Portefeuille concentré autour de quelques valeurs.",
    )

    build_signal(
        biases["sunk_cost"],
        value=_score_with_threshold(indicator("add_to_losers").value, 0.2, 0.8),
        related=["add_to_losers"],
        rationale="Renforcement fréquent des positions perdantes.",
    )

    build_signal(
        biases["herding"],
        value=max(
            _score_with_threshold(indicator("volatility_shift").value, 0.25, 0.7),
            _score_with_threshold(indicator("turnover_ratio").value, 0.4, 0.85),
        ),
        related=["volatility_shift", "turnover_ratio"],
        rationale="Synchronisation accrue avec les mouvements de marché collectifs.",
    )

    scores.sort(key=lambda signal: signal.score, reverse=True)
    return scores


__all__ = ["evaluate_biases"]
