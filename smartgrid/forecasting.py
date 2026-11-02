"""Energy Demand Forecasting Engine.

* Gradient-boosted model (XGBoost if present) on calendar + weather + lag features
* Recursive multi-step forecasting (hourly, up to 7 days) for every region
* Quantile models (P10/P90) give a peak-load uncertainty band
* What-if scenarios (heat-wave +ΔT, emergency surge %)
* Seasonal, segment (residential / commercial / industrial) and surge analytics
* Optional Prophet baseline for benchmarking
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models_util import backend_name, make_regressor, mape
from .simulator import SEGMENTS, SimData

FEATURES = ["hour", "dow", "month", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "is_weekend", "temp_c", "cdd", "hdd",
            "cloud", "region_base", "lag_24", "lag_168", "roll_24"]


def _features(ts: pd.DatetimeIndex, temp, cloud, base, lag24, lag168, roll24) -> pd.DataFrame:
    hr = np.asarray(ts.hour); dow = np.asarray(ts.day_of_week); doy = np.asarray(ts.day_of_year)
    return pd.DataFrame({
        "hour": hr, "dow": dow, "month": np.asarray(ts.month),
        "hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24),
        "doy_sin": np.sin(2 * np.pi * doy / 365), "doy_cos": np.cos(2 * np.pi * doy / 365),
        "is_weekend": (dow >= 5).astype(int), "temp_c": temp, "cdd": np.clip(np.asarray(temp) - 24, 0, None),
        "hdd": np.clip(10 - np.asarray(temp), 0, None), "cloud": cloud, "region_base": base,
        "lag_24": lag24, "lag_168": lag168, "roll_24": roll24})


class DemandForecaster:
    def __init__(self, data: SimData):
        self.d = data
        self.regions = list(data.regions.region_id)
        self.model = None
        self.q10 = None
        self.q90 = None
        self.metrics: dict = {}
        self.base = {}
        w = data.weather.set_index(["region_id", "ts"]).sort_index()
        self._w = {r: w.loc[r] for r in self.regions}
        dem = data.demand.pivot(index="ts", columns="region_id", values="demand_mw").sort_index()
        self.hist = dem[self.regions]
        self.base = {r: float(self.hist[r].mean()) for r in self.regions}

    # ---------------------------------------------------------- training
    def _train_frame(self, upto: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Series]:
        X_parts, y_parts = [], []
        for r in self.regions:
            y = self.hist[r] if upto is None else self.hist[r][self.hist.index < upto]
            w = self._w[r].loc[y.index]
            lag24, lag168 = y.shift(24), y.shift(168)
            roll24 = y.shift(24).rolling(24).mean()
            F = _features(y.index, w.temp_c.to_numpy(), w.cloud.to_numpy(), self.base[r],
                          lag24.to_numpy(), lag168.to_numpy(), roll24.to_numpy())
            F["_y"] = y.to_numpy()
            X_parts.append(F.dropna())
        F = pd.concat(X_parts, ignore_index=True)
        return F[FEATURES], F["_y"]

    def fit(self, holdout_days: int = 7, quantiles: bool = True):
        end = self.hist.index.max() + pd.Timedelta(hours=1)
        cut = end - pd.Timedelta(days=holdout_days)
        # 1) honest back-test: train before the cut, forecast each held-out day from midnight (day-ahead)
        X, y = self._train_frame(upto=cut)
        bt_model = make_regressor().fit(X, y)
        self.model = bt_model
        errs, peak_errs, naive_errs = [], [], []
        for k in range(holdout_days):
            t0 = cut + pd.Timedelta(days=k)
            fc = self._forecast_total(t0, 24)
            actual = self.hist.loc[t0: t0 + pd.Timedelta(hours=23)].sum(axis=1)
            pred = fc.groupby("ts").mw.sum().reindex(actual.index)
            naive = self.hist.sum(axis=1).shift(168).loc[actual.index]
            errs.append(mape(actual, pred)); naive_errs.append(mape(actual, naive))
            peak_errs.append(abs(pred.max() - actual.max()) / actual.max() * 100)
        self.metrics = {"backend": backend_name(), "backtest_days": holdout_days,
                        "day_ahead_mape_pct": round(float(np.mean(errs)), 2),
                        "seasonal_naive_mape_pct": round(float(np.mean(naive_errs)), 2),
                        "peak_error_pct": round(float(np.mean(peak_errs)), 2)}
        # 2) production fit on everything
        X, y = self._train_frame()
        self.model = make_regressor().fit(X, y)
        if quantiles:
            self.q10 = make_regressor(quantile=0.10).fit(X, y)
            self.q90 = make_regressor(quantile=0.90).fit(X, y)
        self.metrics["train_rows"] = int(len(X))
        return self

    # ---------------------------------------------------------- inference
    def _forecast_total(self, start: pd.Timestamp, horizon: int, temp_delta: float = 0.0,
                        surge_pct: float = 0.0, with_band: bool = False) -> pd.DataFrame:
        """Recursive hourly forecast for all regions from `start` (history must cover start-168h)."""
        idx = pd.date_range(start, periods=horizon, freq="h")
        out = []
        series = {r: self.hist[r][self.hist.index < start].copy() for r in self.regions}
        buf = {r: series[r].to_numpy().tolist() for r in self.regions}
        n0 = {r: len(buf[r]) for r in self.regions}
        for step, ts in enumerate(idx):
            rows, bands = [], []
            for r in self.regions:
                b = buf[r]
                w = self._w[r].loc[ts]
                lag24, lag168 = b[-24], b[-168]
                roll24 = float(np.mean(b[-47:-23]))
                rows.append(_features(pd.DatetimeIndex([ts]), [w.temp_c + temp_delta], [w.cloud], self.base[r],
                                      [lag24], [lag168], [roll24]))
            F = pd.concat(rows, ignore_index=True)[FEATURES]
            pred = self.model.predict(F)
            if surge_pct:
                pred = pred * (1 + surge_pct)
            lo = self.q10.predict(F) if (with_band and self.q10 is not None) else pred
            hi = self.q90.predict(F) if (with_band and self.q90 is not None) else pred
            for j, r in enumerate(self.regions):
                buf[r].append(float(pred[j]))
                out.append((ts, r, float(pred[j]), float(min(lo[j], pred[j])), float(max(hi[j], pred[j]))))
        return pd.DataFrame(out, columns=["ts", "region_id", "mw", "p10", "p90"])

    def forecast(self, horizon: int = 48, region: str | None = None, temp_delta: float = 0.0,
                 surge_pct: float = 0.0) -> pd.DataFrame:
        start = self.hist.index.max() + pd.Timedelta(hours=1)
        horizon = int(min(horizon, self.d.cfg.forecast_days * 24))
        fc = self._forecast_total(start, horizon, temp_delta, surge_pct, with_band=True)
        return fc if region is None else fc[fc.region_id == region].reset_index(drop=True)

    def national(self, horizon: int = 48, **kw) -> pd.DataFrame:
        fc = self.forecast(horizon, **kw)
        return fc.groupby("ts", as_index=False)[["mw", "p10", "p90"]].sum()

    def peak(self, horizon: int = 24, **kw) -> dict:
        nat = self.national(horizon, **kw)
        i = nat.mw.idxmax()
        return {"peak_mw": round(nat.mw[i], 0), "peak_time": str(nat.ts[i]), "p10_mw": round(nat.p10[i], 0),
                "p90_mw": round(nat.p90[i], 0), "horizon_h": horizon,
                "installed_firm_mw": float(self.d.plants.capacity_mw.sum()),
                "reserve_margin_pct": round((self.d.plants.capacity_mw.sum() / nat.mw[i] - 1) * 100, 1)}

    # ---------------------------------------------------------- analytics
    def seasonal_profile(self) -> pd.DataFrame:
        df = pd.DataFrame({"mw": self.hist.sum(axis=1).to_numpy(),
                           "year": np.asarray(self.hist.index.year), "month": np.asarray(self.hist.index.month)})
        out = df.groupby(["year", "month"]).mw.agg(avg_mw="mean", peak_mw="max", total_mwh="sum").reset_index()
        out["energy_gwh"] = out.pop("total_mwh") / 1000
        return out.round(1)

    def segment_forecast(self, horizon: int = 24) -> pd.DataFrame:
        """Industrial / commercial / residential split = forecast × recent segment shares per (region, hour, weekend)."""
        seg = self.d.segment_demand
        recent = seg[seg.ts >= seg.ts.max() - pd.Timedelta(days=60)].copy()
        recent["hour"] = recent.ts.dt.hour; recent["wk"] = recent.ts.dt.dayofweek >= 5
        tot = recent.groupby(["ts", "region_id"]).mw.transform("sum")
        recent["share"] = recent.mw / tot
        sh = recent.groupby(["region_id", "hour", "wk", "segment"]).share.mean().reset_index()
        fc = self.forecast(horizon)
        fc["hour"] = fc.ts.dt.hour; fc["wk"] = fc.ts.dt.dayofweek >= 5
        m = fc.merge(sh, on=["region_id", "hour", "wk"])
        m["seg_mw"] = m.mw * m.share
        out = m.groupby(["ts", "segment"], as_index=False).seg_mw.sum(); out["seg_mw"] = out.seg_mw.round(1); return out

    def detect_surges(self, days: int = 30, z: float = 3.0) -> pd.DataFrame:
        """One-step-ahead residuals of the last `days` days; large positive residuals are emergency surges."""
        rows = []
        for r in self.regions:
            y = self.hist[r]
            w = self._w[r].loc[y.index]
            F = _features(y.index, w.temp_c.to_numpy(), w.cloud.to_numpy(), self.base[r], y.shift(24).to_numpy(),
                          y.shift(168).to_numpy(), y.shift(24).rolling(24).mean().to_numpy())
            F.index = y.index
            F = F.iloc[-days * 24:].dropna()
            res = y.loc[F.index] - self.model.predict(F[FEATURES])
            zs = (res - res.mean()) / (res.std() + 1e-9)
            for ts in zs[zs > z].index:
                rows.append(dict(ts=ts, region_id=r, actual_mw=round(y[ts], 1), expected_mw=round(y[ts] - res[ts], 1),
                                 excess_pct=round(res[ts] / (y[ts] - res[ts]) * 100, 1), z=round(zs[ts], 1)))
        return pd.DataFrame(rows).sort_values("z", ascending=False, ignore_index=True) if rows else pd.DataFrame(
            columns=["ts", "region_id", "actual_mw", "expected_mw", "excess_pct", "z"])

    def regional_summary(self, horizon: int = 24) -> pd.DataFrame:
        fc = self.forecast(horizon)
        g = fc.groupby("region_id").mw.agg(peak_mw="max", avg_mw="mean").round(0).reset_index()
        return g.merge(self.d.regions[["region_id", "name"]], on="region_id")

    def feature_importance(self, n: int = 2000) -> pd.DataFrame:
        """Permutation importance on a recent sample (model-agnostic explainability)."""
        X, y = self._train_frame()
        Xs, ys = X.iloc[-n:], y.iloc[-n:]
        base = mape(ys, self.model.predict(Xs))
        rng = np.random.default_rng(0)
        out = []
        for c in FEATURES:
            Xp = Xs.copy(); Xp[c] = rng.permutation(Xp[c].to_numpy())
            out.append((c, mape(ys, self.model.predict(Xp)) - base))
        return pd.DataFrame(out, columns=["feature", "mape_increase_pp"]).sort_values("mape_increase_pp", ascending=False,
                                                                                    ignore_index=True).round(3)

    def explain_forecast(self, ts: pd.Timestamp | None = None) -> dict:
        """Why is the peak what it is? Compare the forecast against a 'reference day' (same hour last week, mean temp)."""
        nat = self.national(24)
        ts = pd.Timestamp(ts) if ts is not None else nat.ts[nat.mw.idxmax()]
        row = nat[nat.ts == ts].iloc[0]
        last_week = float(self.hist.sum(axis=1).get(ts - pd.Timedelta(hours=168), np.nan))
        temps = np.mean([self._w[r].loc[ts].temp_c for r in self.regions])
        ref_t = np.mean([self._w[r].loc[ts - pd.Timedelta(hours=168)].temp_c for r in self.regions])
        return {"time": str(ts), "forecast_mw": round(row.mw, 0), "same_hour_last_week_mw": round(last_week, 0),
                "delta_mw": round(row.mw - last_week, 0), "avg_temp_c": round(float(temps), 1),
                "avg_temp_last_week_c": round(float(ref_t), 1),
                "drivers": [f"hour-of-day {ts.hour}:00 ({'weekend' if ts.dayofweek >= 5 else 'weekday'})",
                            f"temperature {temps:.1f}°C vs {ref_t:.1f}°C a week earlier",
                            "top global features: " + ", ".join(self.feature_importance().feature.head(3))]}


def prophet_baseline(hist_total: pd.Series, horizon: int = 24):   # pragma: no cover - optional dependency
    """Optional Prophet benchmark; returns None when prophet is not installed."""
    try:
        from prophet import Prophet                                  # type: ignore
    except Exception:
        return None
    df = hist_total.reset_index(); df.columns = ["ds", "y"]
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True).fit(df)
    return m.predict(m.make_future_dataframe(horizon, freq="h")).tail(horizon)[["ds", "yhat"]]
