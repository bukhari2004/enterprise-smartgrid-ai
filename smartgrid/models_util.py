"""Model factories: use XGBoost when installed, otherwise scikit-learn gradient boosting (same API)."""
from __future__ import annotations

import numpy as np

try:                                   # optional heavy dependency
    import xgboost as xgb              # type: ignore
    HAS_XGB = True
except Exception:                      # pragma: no cover
    xgb = None
    HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def make_regressor(quantile: float | None = None, n_iter: int = 250, seed: int = 0):
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=n_iter // 2,
                                             learning_rate=0.1, random_state=seed)
    if HAS_XGB:
        return xgb.XGBRegressor(n_estimators=n_iter, max_depth=6, learning_rate=0.08, subsample=0.9,
                                colsample_bytree=0.9, tree_method="hist", random_state=seed, n_jobs=-1)
    return HistGradientBoostingRegressor(max_iter=n_iter, learning_rate=0.08, max_leaf_nodes=48, random_state=seed)


def make_classifier(seed: int = 0):
    if HAS_XGB:
        return xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1)
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=16,
                                          l2_regularization=1.0, random_state=seed)


def backend_name() -> str:
    return "xgboost" if HAS_XGB else "sklearn-histgb"


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-9)) * 100)
