"""JSON-safe conversion of numpy / pandas objects."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def clean(o):
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return clean(o.to_dict())
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


def rnd(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """Round numeric columns only (avoids pandas warning on datetime columns)."""
    return df.round({c: n for c in df.select_dtypes("number").columns})
