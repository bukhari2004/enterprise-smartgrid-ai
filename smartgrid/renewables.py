"""Renewable Energy Intelligence: solar + wind production forecasting, ramp/drop detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models_util import make_regressor
from .simulator import SimData, clear_sky


def _solar_X(w: pd.DataFrame) -> pd.DataFrame:
    hr = np.asarray(w.ts.dt.hour); doy = np.asarray(w.ts.dt.dayofyear)
    return pd.DataFrame({"hour": hr, "doy_sin": np.sin(2 * np.pi * doy / 365), "doy_cos": np.cos(2 * np.pi * doy / 365),
                         "clear_sky": clear_sky(hr, doy), "cloud": w.cloud.to_numpy(), "temp_c": w.temp_c.to_numpy()})


def _wind_X(w: pd.DataFrame) -> pd.DataFrame:
    v = w.wind_ms.to_numpy()
    return pd.DataFrame({"wind_ms": v, "wind_cu": v ** 3, "hour": np.asarray(w.ts.dt.hour)})


class RenewableForecaster:
    def __init__(self, data: SimData):
        self.d = data
        self.solar = self.wind = None
        self.metrics: dict = {}
        w = data.weather
        self.hist = w[w.solar_mw.notna()].copy()
        self.future = w[w.solar_mw.isna()].copy()

    def fit(self, holdout_days: int = 14):
        h = self.hist
        out = {}
        for kind, capcol, X, tgt in (("solar", "solar_cap_mw", _solar_X, "solar_mw"), ("wind", "wind_cap_mw", _wind_X, "wind_mw")):
            sub = h[h[capcol] > 0]
            cut = sub.ts.max() - pd.Timedelta(days=holdout_days)
            tr, te = sub[sub.ts <= cut], sub[sub.ts > cut]
            y_tr = tr[tgt] / tr[capcol]
            m = make_regressor(n_iter=150).fit(X(tr), y_tr)
            pred = np.clip(m.predict(X(te)), 0, 1) * te[capcol]
            ss_res = ((te[tgt] - pred) ** 2).sum(); ss_tot = ((te[tgt] - te[tgt].mean()) ** 2).sum()
            out[kind] = {"r2": round(float(1 - ss_res / ss_tot), 3),
                         "mae_pct_of_capacity": round(float((te[tgt] - pred).abs().mean() / te[capcol].mean() * 100), 2)}
            full = make_regressor(n_iter=150).fit(X(sub), sub[tgt] / sub[capcol])
            setattr(self, kind, full)
        self.metrics = out
        return self

    def forecast(self, horizon: int = 48) -> pd.DataFrame:
        f = self.future.sort_values(["ts", "region_id"])
        ts0 = f.ts.min()
        f = f[f.ts < ts0 + pd.Timedelta(hours=horizon)].copy()
        f["solar_mw"] = np.clip(self.solar.predict(_solar_X(f)), 0, 1) * f.solar_cap_mw
        f["wind_mw"] = np.clip(self.wind.predict(_wind_X(f)), 0, 1) * f.wind_cap_mw
        return f[["ts", "region_id", "solar_mw", "wind_mw"]].reset_index(drop=True)

    def national(self, horizon: int = 48) -> pd.DataFrame:
        g = self.forecast(horizon).groupby("ts", as_index=False)[["solar_mw", "wind_mw"]].sum()
        g["renewable_mw"] = g.solar_mw + g.wind_mw
        return g

    def detect_drops(self, horizon: int = 24, window: int = 3, threshold_pct: float = 12.0) -> pd.DataFrame:
        """Forecast ramp-downs > threshold% of installed renewable capacity within `window` hours
        (excluding the predictable sunset ramp, hours 16-20)."""
        nat = self.national(horizon + window)
        cap = float(self.d.renewables.capacity_mw.sum())
        nat["delta_mw"] = nat.renewable_mw.shift(-window) - nat.renewable_mw
        nat["delta_pct_cap"] = nat.delta_mw / cap * 100
        hr = nat.ts.dt.hour
        bad = nat[(nat.delta_pct_cap < -threshold_pct) & ~hr.between(15, 20)].dropna()
        bad = bad[bad.ts < nat.ts.min() + pd.Timedelta(hours=horizon)]
        return bad.reset_index(drop=True).round({c: 1 for c in bad.columns if c != 'ts'})
