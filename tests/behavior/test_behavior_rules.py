from stock_analysis.orotitan_ai.behavior.rules import evaluate_biases
from stock_analysis.orotitan_ai.behavior.schemas import BehaviorIndicator


def _indicator_map(**values: float) -> dict[str, BehaviorIndicator]:
    return {name: BehaviorIndicator(name=name, value=value) for name, value in values.items()}


def test_loss_aversion_trigger() -> None:
    indicators = _indicator_map(loss_hold_bias=0.8, drawdown_streak=0.7)
    signals = evaluate_biases(indicators)
    ids = {signal.bias_id for signal in signals}
    assert "loss_aversion" in ids
    assert "disposition" in ids


def test_overconfidence_and_fomo() -> None:
    indicators = _indicator_map(position_size_variance=0.9, win_loss_asymmetry=0.7, chasing_intensity=0.6)
    signals = evaluate_biases(indicators)
    ids = {signal.bias_id for signal in signals}
    assert "overconfidence" in ids
    assert "fomo" in ids


def test_confirmation_bias_requires_combination() -> None:
    indicators = _indicator_map(plan_deviation=0.7, turnover_ratio=0.1)
    signals = evaluate_biases(indicators)
    scores = {signal.bias_id: signal.score for signal in signals}
    assert scores.get("confirmation", 0.0) >= 0.3
