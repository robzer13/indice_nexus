"""DataFrame/Series pandas stub semantics required by offline ML API tests."""
import math

from tests import pandas_stub as pd


def test_column_selection_is_not_row_mask() -> None:
    frame = pd.DataFrame({"price": [10.0, 20.0], "signal": [0.0, 1.0]}, index=[1, 2])
    result = frame[["signal"]]
    assert result.columns == ["signal"]
    assert len(result) == 2
    assert result["signal"].tolist() == [0.0, 1.0]


def test_boolean_row_selection_and_dropna_semantics() -> None:
    frame = pd.DataFrame({"left": [1.0, 2.0], "right": [float("nan"), 3.0]})
    assert len(frame[[True, False]]) == 1
    assert len(frame.dropna()) == 1
    assert len(frame.dropna(how="all")) == 2


def test_replace_infinities_and_label_lookup() -> None:
    frame = pd.DataFrame({"price": [float("inf"), 2.0]}, index=[1, 2])
    replaced = frame.replace([float("inf"), float("-inf")], float("nan"))
    assert math.isnan(replaced["price"].tolist()[0])
    series = replaced["price"]
    assert series.loc[[2]].tolist() == [2.0]

def test_pct_change_looks_backward_and_never_erases_valid_tail() -> None:
    series = pd.Series([float(x) for x in range(1, 31)])
    change = series.pct_change(21)
    assert all(math.isnan(x) for x in change.tolist()[:21])
    assert not any(math.isnan(x) for x in change.tolist()[21:])
    assert change.tolist()[-1] == (30 - 9) / 9
