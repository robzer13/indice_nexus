"""Machine learning helpers for walk-forward evaluation."""
from __future__ import annotations

import logging
import math
import statistics
from typing import Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURES = ["ret_1d", "vol_20", "sma200_gap", "rsi_14", "mom_21"]


def build_model(kind: str):
    """Return a classifier configured for imbalanced financial datasets."""

    name = (kind or "").lower()
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=200,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    if name == "xgb":
        try:  # pragma: no cover - optional dependency path
            from xgboost import XGBClassifier
        except ModuleNotFoundError:  # pragma: no cover - handled by fallback
            LOGGER.warning("xgboost non disponible, utilisation de RandomForest par défaut")
            return build_model("rf")

        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model kind: {kind}")


def _ensure_array(data: pd.DataFrame | pd.Series):
    return data.to_numpy(dtype=float)


def time_cv(X: pd.DataFrame, y: pd.Series, model_kind: str, splits: int = 5) -> Tuple[float, float]:
    """Compute mean/std ROC-AUC across chronological folds."""

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    if splits <= 0:
        raise ValueError("splits must be positive")
    n_samples = len(X)
    if n_samples < max(40, splits * 5):
        raise ValueError("Not enough observations for cross-validation")

    splitter = TimeSeriesSplit(n_splits=min(splits, max(2, n_samples // 40)))
    aucs: list[float] = []

    for train_idx, test_idx in splitter.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            LOGGER.debug("Skipping fold with a single class")
            continue
        model = build_model(model_kind)
        model.fit(_ensure_array(X_train), y_train.to_numpy())
        probs = model.predict_proba(_ensure_array(X_test))[:, 1]
        try:
            aucs.append(float(roc_auc_score(y_test, probs)))
        except ValueError:
            LOGGER.debug("Unable to compute ROC-AUC for this fold")
            continue

    if not aucs:
        raise ValueError("ROC-AUC unavailable (class imbalance or insufficient data)")

    mean_auc = statistics.fmean(aucs)
    std_auc = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
    return float(mean_auc), float(std_auc)


def walk_forward_signals(
    X: pd.DataFrame,
    y: pd.Series,
    retrain_every: int = 60,
    warmup: int = 200,
    model_kind: str = "xgb",
    proba_threshold: float = 0.55,
) -> Tuple[pd.Series, pd.Series]:
    """Generate probability forecasts and binary signals in a walk-forward fashion."""

    if retrain_every <= 0:
        raise ValueError("retrain_every must be positive")
    if warmup <= 0:
        raise ValueError("warmup must be positive")

    index = X.index
    proba = pd.Series(math.nan, index=index, dtype="float64")
    signal = pd.Series(math.nan, index=index, dtype="float64")

    n_samples = len(X)
    if n_samples <= warmup:
        return proba, signal

    for start in range(warmup, n_samples, retrain_every):
        end = min(n_samples, start + retrain_every)
        train_X = X.iloc[:start]
        train_y = y.iloc[:start]
        if train_y.nunique() < 2:
            continue
        model = build_model(model_kind)
        model.fit(_ensure_array(train_X), train_y.to_numpy())
        test_X = X.iloc[start:end]
        preds = model.predict_proba(_ensure_array(test_X))[:, 1]
        proba.iloc[start:end] = preds
        signal.iloc[start:end] = (preds >= proba_threshold).astype(float)

    return proba, signal


def sharpe_sim(price: pd.Series, signal: pd.Series) -> float:
    """Compute an annualised Sharpe ratio for the ML signal."""

    if price.empty or signal.empty:
        return 0.0
    aligned = signal.dropna()
    if aligned.empty:
        return 0.0
    returns = price.pct_change().reindex(aligned.index)
    strat = returns * aligned.shift(1).fillna(0.0)
    strat = strat.dropna()
    if strat.empty:
        return 0.0
    mean = strat.mean()
    std = strat.std(ddof=0)
    if std == 0 or math.isnan(std):
        return 0.0
    return float((mean / std) * math.sqrt(252.0))


def confusion(y_true: pd.Series, y_prob: pd.Series, thr: float = 0.5) -> dict:
    """Return a confusion-matrix dictionary for the given threshold."""

    if y_true.empty or y_prob.empty:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    aligned = y_prob.index.intersection(y_true.index)
    if aligned.empty:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    truth = y_true.reindex(aligned).fillna(0).astype(int)
    preds = (y_prob.reindex(aligned) >= thr).astype(int)
    tn = int(((preds == 0) & (truth == 0)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())
    tp = int(((preds == 1) & (truth == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


__all__ = [
    "FEATURES",
    "build_model",
    "time_cv",
    "walk_forward_signals",
    "sharpe_sim",
    "confusion",
]
