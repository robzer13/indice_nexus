from stock_analysis.orotitan_ai.behavior.bias_bank import get_bias_definitions
from stock_analysis.orotitan_ai.behavior.scoring import aggregate_scores
from stock_analysis.orotitan_ai.behavior.schemas import BehaviorIndicator, BiasSignal


def _signal(bias_id: str, score: float) -> BiasSignal:
    return BiasSignal(
        bias_id=bias_id,
        score=score,
        indicators=[BehaviorIndicator(name="dummy", value=score)],
        rationale="synthetic",
    )


def test_aggregate_scores_bounds() -> None:
    definitions = get_bias_definitions()
    score, adjustment, tags = aggregate_scores(
        [_signal("loss_aversion", 0.9), _signal("fomo", 0.4)],
        definitions,
        threshold=30.0,
        max_influence=0.25,
    )
    assert 0.0 <= score <= 100.0
    assert -0.25 <= adjustment <= 0.25
    assert tags


def test_low_score_positive_adjustment() -> None:
    definitions = get_bias_definitions()
    score, adjustment, _ = aggregate_scores([_signal("familiarity", 0.05)], definitions)
    assert score < 30.0
    assert adjustment >= 0.0
