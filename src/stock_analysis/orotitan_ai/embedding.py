"""Signal encoding utilities for the OroTitan AI layer."""
from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable in tests
    _np = None

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    import numpy as np

    ArrayLike = np.ndarray
    MatrixLike = np.ndarray
else:
    ArrayLike = Sequence[float]
    MatrixLike = list[list[float]]
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - guard any runtime issue
    SentenceTransformer = None  # type: ignore[assignment]

_TEXT_MODEL: "SentenceTransformer | None" = None
_DEFAULT_TEXT_DIM = 16
_DEFAULT_PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
_REGIME_ORDER = ("Expansion", "Inflation", "Stress", "Recovery")


def _as_float_sequence(values: Sequence[float]) -> ArrayLike:
    cleaned = [float(value) for value in values]
    if _np is not None:
        return _np.asarray(cleaned, dtype=float)
    return cleaned


def _zeros(length: int) -> ArrayLike:
    if _np is not None:
        return _np.zeros(length, dtype=float)
    return [0.0] * length


def _nan_to_num(values: ArrayLike) -> ArrayLike:
    if _np is not None:
        return _np.nan_to_num(values, copy=False)
    return [0.0 if value != value else float(value) for value in values]  # NaN check


def _ensure_datetime_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):  # pragma: no cover - defensive
        raise ValueError(f"{name} must be indexed by DatetimeIndex")
    return df.sort_index()


def build_feature_dict(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    regimes: pd.Series | pd.DataFrame,
    meta: Dict[str, Any] | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Assemble aligned inputs for OroTitan encoders."""
    prices_aligned = _ensure_datetime_index(prices, "prices").copy()
    prices_aligned.columns = [str(col).lower() for col in prices_aligned.columns]

    features_aligned = _ensure_datetime_index(features, "features").copy()
    regimes_aligned: pd.DataFrame
    if isinstance(regimes, pd.Series):
        regimes_aligned = regimes.to_frame("regime").sort_index()
    else:
        regimes_aligned = _ensure_datetime_index(regimes, "regimes").copy()

    common_index = prices_aligned.index
    common_index = common_index.intersection(features_aligned.index)
    common_index = common_index.intersection(regimes_aligned.index)
    if common_index.empty:
        raise ValueError("No overlapping index between prices, features, and regimes")

    prices_aligned = prices_aligned.loc[common_index]
    features_aligned = features_aligned.loc[common_index]
    regimes_aligned = regimes_aligned.loc[common_index]

    payload: dict[str, pd.Series | pd.DataFrame] = {
        "prices": prices_aligned,
        "features": features_aligned,
        "regimes": regimes_aligned,
    }

    if meta:
        payload["meta"] = pd.Series(meta)  # type: ignore[assignment]

    return payload


def encode_snapshot(
    date: pd.Timestamp,
    feature_dict: dict[str, pd.Series | pd.DataFrame],
    config: dict[str, Any] | None = None,
 ) -> ArrayLike:
    """Encode a single date into a numeric vector suitable for decision engines."""
    config = config or {}
    try:
        prices = feature_dict["prices"]  # type: ignore[assignment]
        features = feature_dict["features"]  # type: ignore[assignment]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError("feature_dict must include 'prices' and 'features'") from exc

    if not isinstance(prices, pd.DataFrame) or not isinstance(features, pd.DataFrame):
        raise TypeError("feature_dict['prices'] and ['features'] must be DataFrames")

    if date not in prices.index or date not in features.index:
        raise KeyError(f"Date {date} missing from prices/features inputs")

    price_columns = config.get("price_columns", _DEFAULT_PRICE_COLUMNS)
    price_row = prices.loc[date]
    price_vector = _as_float_sequence(price_row.reindex(price_columns).astype(float).tolist())

    feature_columns = config.get("feature_columns")
    if feature_columns is None:
        feature_columns = list(features.columns)
    feature_row = features.loc[date, feature_columns]
    if isinstance(feature_row, pd.Series):
        feature_vector = _as_float_sequence(feature_row.astype(float).tolist())
    else:  # pragma: no cover - multiindex columns
        feature_vector = _as_float_sequence(feature_row.astype(float).tolist())

    regimes_df = feature_dict.get("regimes")
    regime_vector = _zeros(len(_REGIME_ORDER))
    if isinstance(regimes_df, (pd.Series, pd.DataFrame)) and date in regimes_df.index:
        regime_value: str | None
        if isinstance(regimes_df, pd.Series):
            regime_value = str(regimes_df.loc[date]) if pd.notna(regimes_df.loc[date]) else None
        else:
            first_col = regimes_df.columns[0]
            regime_value = (
                str(regimes_df.loc[date, first_col])
                if pd.notna(regimes_df.loc[date, first_col])
                else None
            )
        if regime_value:
            try:
                regime_index = _REGIME_ORDER.index(regime_value)
                regime_vector[regime_index] = 1.0
            except ValueError:
                logger.debug("Unknown regime '%s' ignored in encoding", regime_value)
    if _np is not None:
        vector = _np.concatenate([price_vector, feature_vector, regime_vector])
        return _nan_to_num(vector)
    vector_list = list(price_vector) + list(feature_vector) + list(regime_vector)
    return _nan_to_num(vector_list)


def encode_inputs(
    feature_dict: dict[str, pd.Series | pd.DataFrame],
    *,
    dates: Sequence[pd.Timestamp] | None = None,
    config: dict[str, Any] | None = None,
) -> Dict[pd.Timestamp, ArrayLike]:
    """Return a mapping of timestamps to encoded vectors."""

    prices = feature_dict.get("prices")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("feature_dict['prices'] must be a pandas DataFrame")

    available_dates = list(prices.index)
    if not available_dates:
        return {}

    selected_dates: Sequence[pd.Timestamp]
    if dates is None:
        selected_dates = available_dates
    else:
        # ensure timestamps are in the same timezone/index
        lookup = {pd.Timestamp(ts): None for ts in available_dates}
        selected_dates = [pd.Timestamp(ts) for ts in dates if pd.Timestamp(ts) in lookup]

    vectors: Dict[pd.Timestamp, ArrayLike] = {}
    for timestamp in selected_dates:
        try:
            vectors[pd.Timestamp(timestamp)] = encode_snapshot(
                pd.Timestamp(timestamp), feature_dict, config=config
            )
        except KeyError:
            logger.debug("Skipping %s missing from inputs", timestamp)
    return vectors


def embed_tickers(
    datasets: Dict[str, dict[str, pd.Series | pd.DataFrame]],
    *,
    dates: Sequence[pd.Timestamp] | None = None,
    config: dict[str, Any] | None = None,
) -> Dict[str, Dict[pd.Timestamp, ArrayLike]]:
    """Encode multiple tickers using :func:`encode_inputs`."""

    encoded: Dict[str, Dict[pd.Timestamp, ArrayLike]] = {}
    for ticker, feature_dict in datasets.items():
        try:
            encoded[ticker] = encode_inputs(feature_dict, dates=dates, config=config)
        except Exception as exc:  # pragma: no cover - tickers may fail independently
            logger.warning("Embedding failed for %s", ticker, exc_info=exc)
            encoded[ticker] = {}
    return encoded


def encode_text_optional(texts: list[str], model_name: str | None = None) -> MatrixLike:
    """Encode analyst notes if a sentence-transformer is available, else zeros."""
    if not texts:
        if _np is not None:
            return _np.zeros((0, _DEFAULT_TEXT_DIM), dtype=float)
        return []

    if SentenceTransformer is None:
        logger.info("sentence-transformers not available; returning zero vectors")
        if _np is not None:
            return _np.zeros((len(texts), _DEFAULT_TEXT_DIM), dtype=float)
        return [[0.0] * _DEFAULT_TEXT_DIM for _ in texts]

    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        try:
            chosen_model = model_name or "all-MiniLM-L6-v2"
            _TEXT_MODEL = SentenceTransformer(chosen_model)  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - avoid failing without model files
            logger.warning("Falling back to zero embeddings due to error: %s", exc)
            if _np is not None:
                return _np.zeros((len(texts), _DEFAULT_TEXT_DIM), dtype=float)
            return [[0.0] * _DEFAULT_TEXT_DIM for _ in texts]

    try:
        embeddings = _TEXT_MODEL.encode(texts, convert_to_numpy=True)  # type: ignore[union-attr]
        if _np is not None:
            return _np.asarray(embeddings, dtype=float)
        rows = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings  # type: ignore[attr-defined]
        return [[float(value) for value in row] for row in rows]
    except Exception as exc:  # pragma: no cover - guard runtime issues
        logger.warning("sentence-transformers encoding failed: %s", exc)
        if _np is not None:
            return _np.zeros((len(texts), _DEFAULT_TEXT_DIM), dtype=float)
        return [[0.0] * _DEFAULT_TEXT_DIM for _ in texts]

