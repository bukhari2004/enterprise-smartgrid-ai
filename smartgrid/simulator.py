"""Synthetic national grid: topology, weather, demand, renewables, meters, transformer health.

The simulator replaces the utility's real data lake so the whole platform is runnable
end-to-end. Ground-truth labels (theft, failures) are stored in `truth_*` frames that are
ONLY used for model evaluation, never as model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import GridConfig

REGION_NAMES = ["North", "South", "East", "West", "Central", "Coastal", "Highland", "Delta"]

PLANT_TEMPLATES = [
    # type, capacity MW, must-run MW, marginal cost $/MWh, tCO2/MWh, ramp MW/h
    ("nuclear", 2400, 1500, 12, 0.00, 150),
    ("hydro", 2200, 0, 8, 0.00, 1200),
    ("coal", 3200, 0, 38, 0.95, 300),
    ("coal", 2800, 0, 42, 1.00, 280),
    ("gas_ccgt", 2600, 0, 58, 0.40, 900),
    ("gas_ccgt", 2000, 0, 62, 0.42, 800),
    ("gas_ocgt", 1500, 0, 110, 0.65, 1400),
    ("oil", 1200, 0, 160, 0.85, 1000),
]

SEGMENTS = ["residential", "commercial", "industrial"]


@dataclass
class SimData:
    cfg: GridConfig
    regions: pd.DataFrame
    plants: pd.DataFrame
    substations: pd.DataFrame
    lines: pd.DataFrame
    transformers: pd.DataFrame
    meters: pd.DataFrame
    renewables: pd.DataFrame
    batteries: pd.DataFrame
    teams: pd.DataFrame
    weather: pd.DataFrame            # hourly, per region, history + forecast horizon
    demand: pd.DataFrame             # hourly, per region (history only)
    segment_demand: pd.DataFrame     # hourly, per region x segment (history only)
    meter_daily: pd.DataFrame        # day, meter_id, kwh   (last N days)
    meter_profile: pd.DataFrame      # meter_id, night_ratio, billed_ratio
    maint_history: pd.DataFrame      # labelled transformer snapshots (training)
    maint_current: pd.DataFrame      # current transformer snapshot (inference)
    incidents: pd.DataFrame
    truth_theft: pd.DataFrame = field(default_factory=pd.DataFrame)
    truth_failure: pd.DataFrame = field(default_factory=pd.DataFrame)
    now: pd.Timestamp = None


# ------------------------------------------------------------------ helpers
def _bump(h, mu, sd):
    d = np.minimum(np.abs(h - mu), 24 - np.abs(h - mu))
    return np.exp(-0.5 * (d / sd) ** 2)


def hourly_profile(kind: str, hour: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    h = hour.astype(float)
    if kind == "residential":
        p = 0.55 + 0.35 * _bump(h, 7.5, 1.6) + 0.65 * _bump(h, 20, 2.4) + 0.10 * _bump(h, 13, 3)
        p = p * np.where(weekend, 1.06, 1.0)
    elif kind == "commercial":
        p = 0.35 + 0.75 * _bump(h, 13, 3.8)
        p = p * np.where(weekend, 0.62, 1.0)
    else:  # industrial: near-flat 3-shift
        p = 0.88 + 0.12 * _bump(h, 11, 5)
        p = p * np.where(weekend, 0.86, 1.0)
    return p


def clear_sky(hour: np.ndarray, doy: np.ndarray) -> np.ndarray:
    day_len = 12 + 2.5 * np.sin(2 * np.pi * (doy - 80) / 365)
    sunrise = 12 - day_len / 2
    x = (hour - sunrise) / day_len
    cs = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)), 0.0)
    return cs * (0.8 + 0.2 * np.sin(2 * np.pi * (doy - 80) / 365))


def wind_power_curve(v: np.ndarray) -> np.ndarray:
    return np.where(v < 3, 0.0, np.where(v < 12, ((v - 3) / 9) ** 3, np.where(v < 25, 1.0, 0.0)))


def _ar1(rng, n, phi, sd):
    e = rng.normal(0, sd, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ------------------------------------------------------------------ simulator
class GridSimulator:
    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # -- topology -------------------------------------------------------
    def _topology(self):
        c, rng = self.cfg, self.rng
        regions = pd.DataFrame({
            "region_id": [f"R{i+1}" for i in range(c.n_regions)],
            "name": REGION_NAMES[: c.n_regions],
            "base_mw": rng.uniform(1000, 2200, c.n_regions).round(0),
            "temp_offset": rng.uniform(-4, 4, c.n_regions).round(1),
            "lat": 30 + rng.uniform(-5, 5, c.n_regions),
            "lon": 70 + rng.uniform(-6, 6, c.n_regions),
        })
        rows = []
        scale = len(PLANT_TEMPLATES) / c.n_plants if c.n_plants > len(PLANT_TEMPLATES) else 1.0
        for i in range(c.n_plants):
            t, cap, mr, cost, em, ramp = PLANT_TEMPLATES[i % len(PLANT_TEMPLATES)]
            rows.append(dict(plant_id=f"P-{i+1:03d}", type=t, capacity_mw=round(cap * scale), must_run_mw=round(mr * scale),
                             cost_usd_mwh=round(cost * rng.uniform(0.95, 1.08), 1), emission_t_mwh=em,
                             ramp_mw_h=round(ramp * scale), region_id=regions.region_id.iloc[i % c.n_regions]))
        plants = pd.DataFrame(rows)
        sub_region = [regions.region_id.iloc[i % c.n_regions] for i in range(c.n_substations)]
        subs = pd.DataFrame({
            "substation_id": [f"S-{i+1:03d}" for i in range(c.n_substations)],
            "region_id": sub_region,
            "voltage_kv": rng.choice([132, 220], c.n_substations, p=[0.7, 0.3]),
            "capacity_mva": rng.uniform(150, 600, c.n_substations).round(0),
            "feeder_km": rng.uniform(5, 60, c.n_substations).round(1),
        })
        subs = subs.merge(regions[["region_id", "lat", "lon"]], on="region_id")
        subs["lat"] += rng.normal(0, 0.8, len(subs))
        subs["lon"] += rng.normal(0, 0.8, len(subs))
        lines = []
        for s in subs.substation_id:
            n_feed = 2 if rng.random() < 0.7 else 1
            for p in rng.choice(plants.plant_id, size=min(n_feed, len(plants)), replace=False):
                lines.append(dict(line_id=f"L-{len(lines)+1:04d}", from_id=p, to_id=s,
                                  capacity_mw=float(rng.uniform(250, 900)), length_km=float(rng.uniform(20, 300))))
        lines = pd.DataFrame(lines)
        sub_ids = subs.substation_id.to_numpy()
        tr = pd.DataFrame({
            "transformer_id": [f"T-{i+1:05d}" for i in range(c.n_transformers)],
            "substation_id": sub_ids[np.arange(c.n_transformers) % c.n_substations],
            "rated_kva": rng.choice([250, 500, 1000, 1600, 2500], c.n_transformers, p=[.2, .3, .3, .15, .05]),
        })
        tr = tr.merge(subs[["substation_id", "region_id", "lat", "lon"]], on="substation_id")
        tr["lat"] += rng.normal(0, 0.1, len(tr)); tr["lon"] += rng.normal(0, 0.1, len(tr))
        n_m = c.n_transformers * c.meters_per_transformer
        m_tr = np.repeat(tr.transformer_id.to_numpy(), c.meters_per_transformer)
        tariff = rng.choice(SEGMENTS, n_m, p=[0.80, 0.15, 0.05])
        base = np.where(tariff == "residential", rng.normal(12, 4, n_m),
                        np.where(tariff == "commercial", rng.normal(60, 20, n_m), rng.normal(600, 180, n_m))).clip(2)
        meters = pd.DataFrame({"meter_id": [f"M-{i+1:07d}" for i in range(n_m)], "transformer_id": m_tr,
                               "consumer_id": [f"C-{i+1:07d}" for i in range(n_m)], "tariff": tariff, "base_kwh_day": base})
        meters = meters.merge(tr[["transformer_id", "region_id", "lat", "lon"]], on="transformer_id")
        meters["lat"] += rng.normal(0, 0.02, len(meters)); meters["lon"] += rng.normal(0, 0.02, len(meters))
        ren = pd.DataFrame(
            [dict(asset_id=f"SOL-{i+1:03d}", kind="solar", capacity_mw=float(rng.uniform(150, 600)),
                  region_id=regions.region_id.iloc[rng.integers(c.n_regions)]) for i in range(c.n_solar)] +
            [dict(asset_id=f"WND-{i+1:03d}", kind="wind", capacity_mw=float(rng.uniform(200, 800)),
                  region_id=regions.region_id.iloc[rng.integers(c.n_regions)]) for i in range(c.n_wind)])
        bat = pd.DataFrame([dict(battery_id=f"BAT-{i+1:03d}", capacity_mwh=float(rng.choice([200, 400, 800])),
                                 power_mw=float(rng.choice([50, 100, 200])), efficiency=0.92,
                                 region_id=regions.region_id.iloc[rng.integers(c.n_regions)]) for i in range(c.n_batteries)])
        teams = pd.DataFrame({"team_id": [f"MT-{i+1}" for i in range(c.n_regions)],
                              "region_id": regions.region_id, "crew_size": rng.integers(6, 15, c.n_regions)})
        return regions, plants, subs, lines, tr, meters, ren, bat, teams

    # -- weather / demand / renewables -----------------------------------
    def _weather_demand(self, regions, ren):
        c, rng = self.cfg, self.rng
        n_hist = c.history_days * 24
        n_all = (c.history_days + c.forecast_days) * 24
        idx = pd.date_range(c.start, periods=n_all, freq="h")
        doy = np.asarray(idx.dayofyear); hr = np.asarray(idx.hour); wk = np.asarray(idx.dayofweek >= 5)
        frames, seg_frames = [], []
        for _, r in regions.iterrows():
            temp = (22 + 11 * np.sin(2 * np.pi * (doy - 105) / 365) + 5 * np.sin(2 * np.pi * (hr - 9) / 24)
                    + r.temp_offset + _ar1(rng, n_all, 0.97, 0.45))
            cloud = np.clip(0.35 + _ar1(rng, n_all, 0.96, 0.08), 0, 1)
            wind = np.clip(6.5 + _ar1(rng, n_all, 0.97, 0.55) + 1.2 * np.sin(2 * np.pi * (hr - 15) / 24), 0, 30)
            w = pd.DataFrame({"ts": idx, "region_id": r.region_id, "temp_c": temp, "cloud": cloud, "wind_ms": wind})
            sol_cap = ren[(ren.region_id == r.region_id) & (ren.kind == "solar")].capacity_mw.sum()
            wnd_cap = ren[(ren.region_id == r.region_id) & (ren.kind == "wind")].capacity_mw.sum()
            sol_cf = clear_sky(hr, doy) * (1 - 0.75 * cloud) * (1 - 0.004 * np.clip(temp - 25, 0, None))
            w["solar_mw"] = np.clip(sol_cap * sol_cf * (1 + rng.normal(0, 0.03, n_all)), 0, None)
            w["wind_mw"] = np.clip(wnd_cap * wind_power_curve(wind) * (1 + rng.normal(0, 0.05, n_all)), 0, wnd_cap)
            w["solar_cap_mw"] = sol_cap; w["wind_cap_mw"] = wnd_cap
            frames.append(w)
            cool = np.clip(temp - 24, 0, None) * 0.035
            heat = np.clip(10 - temp, 0, None) * 0.020
            shares = np.array([0.40, 0.25, 0.35]) * rng.uniform(0.85, 1.15, 3)
            shares /= shares.sum()
            trend = 1 + 0.03 * np.arange(n_all) / (365 * 24)
            surge = np.zeros(n_all)
            for _ in range(max(1, int(12 * c.history_days / 365))):
                s0 = rng.integers(0, n_hist - 8); dur = rng.integers(2, 7)
                surge[s0:s0 + dur] += rng.uniform(0.10, 0.25)
            for k, seg in enumerate(SEGMENTS):
                sens = {"residential": 1.0, "commercial": 0.8, "industrial": 0.1}[seg]
                d = (r.base_mw * shares[k] * hourly_profile(seg, hr, wk) * (1 + sens * (cool + heat)) * trend
                     * (1 + surge * (0.5 if seg == "industrial" else 1.0)) * (1 + rng.normal(0, 0.012, n_all)))
                seg_frames.append(pd.DataFrame({"ts": idx, "region_id": r.region_id, "segment": seg, "mw": d}))
        weather = pd.concat(frames, ignore_index=True)
        seg = pd.concat(seg_frames, ignore_index=True)
        hist_end = idx[n_hist]
        seg = seg[seg.ts < hist_end].reset_index(drop=True)
        demand = seg.groupby(["ts", "region_id"], as_index=False).mw.sum().rename(columns={"mw": "demand_mw"})
        weather.loc[weather.ts >= hist_end, ["solar_mw", "wind_mw"]] = np.nan   # future actuals unknown
        return weather, demand, seg

    # -- meters ---------------------------------------------------------
    def _meters(self, meters, weather, now):
        c, rng = self.cfg, self.rng
        days = pd.date_range(now.normalize() - pd.Timedelta(days=c.meter_days), periods=c.meter_days, freq="D")
        n_m, n_d = len(meters), len(days)
        dtemp = weather.groupby([weather.ts.dt.normalize(), "region_id"]).temp_c.mean().unstack().reindex(days).ffill().bfill()
        t = dtemp[meters.region_id.to_numpy()].to_numpy()
        tariff = meters.tariff.to_numpy()
        seasonal = 1 + 0.025 * np.clip(t - 24, 0, None) * (tariff != "industrial")
        wk = np.array([d.dayofweek >= 5 for d in days])[:, None]
        wkf = np.where(tariff[None, :] == "residential", np.where(wk, 1.08, 1.0), np.where(wk, 0.7, 1.0))
        true = meters.base_kwh_day.to_numpy()[None, :] * seasonal * wkf * rng.lognormal(0, 0.14, (n_d, n_m))
        night = meters.tariff.map({"residential": 0.45, "commercial": 0.25, "industrial": 0.9}).to_numpy() * rng.normal(1, 0.08, n_m)
        n_theft = int(n_m * c.theft_rate)
        theft_idx = rng.choice(n_m, n_theft, replace=False)
        kinds = rng.choice(["tamper", "bypass", "fraud"], n_theft, p=[0.4, 0.35, 0.25])
        read = true.copy()
        truth = pd.DataFrame({"meter_id": meters.meter_id, "is_theft": False, "theft_type": "none", "billing_anomaly": False})
        for i, k in zip(theft_idx, kinds):
            if k == "tamper":
                d0 = rng.integers(n_d // 3, n_d - 8)
                read[d0:, i] *= rng.uniform(0.2, 0.6)
            elif k == "bypass" or tariff[i] == "residential":
                k = "bypass"; read[:, i] *= rng.uniform(0.3, 0.55)
            else:                                   # commercial / industrial fraud
                read[:, i] *= rng.uniform(0.45, 0.7); night[i] *= 1.9
            truth.loc[i, ["is_theft", "theft_type"]] = [True, k]
        billed = 1 + rng.normal(0, 0.02, n_m)
        bill_idx = rng.choice(n_m, max(1, int(n_m * 0.02)), replace=False)
        billed[bill_idx] = np.where(rng.random(len(bill_idx)) < 0.5, rng.uniform(1.25, 1.6, len(bill_idx)),
                                    rng.uniform(0.5, 0.75, len(bill_idx)))
        truth.loc[bill_idx, "billing_anomaly"] = True
        daily = pd.DataFrame(read, index=days, columns=meters.meter_id.to_numpy()).stack().rename("kwh").reset_index()
        daily.columns = ["day", "meter_id", "kwh"]
        true_by_tr = pd.DataFrame(true).T.groupby(meters.transformer_id.to_numpy()).sum().T   # feeder input sees TRUE load
        tr_input = true_by_tr.iloc[-30:].sum() * 1.03
        prof = pd.DataFrame({"meter_id": meters.meter_id, "night_ratio": night, "billed_ratio": billed})
        tr_in = pd.DataFrame({"transformer_id": tr_input.index, "input_kwh_30d": tr_input.to_numpy()})
        return daily, prof, truth, tr_in

    # -- transformer health ------------------------------------------------
    def _maintenance(self, tr, lf_now):
        rng = self.rng

        def snapshot(n, lf=None):
            age = rng.uniform(1, 42, n)
            lf = rng.beta(5, 3.2, n) * 1.15 if lf is None else lf
            faults = rng.poisson(0.25 + age / 40, n)
            since = rng.uniform(0, 1000, n)
            oil = 52 + 38 * lf + 0.25 * age + rng.normal(0, 4, n)
            wind_t = oil + rng.uniform(8, 18, n)
            h2 = rng.lognormal(3.0 + 0.03 * age + 0.3 * faults, 0.5, n)
            c2h2 = np.where(rng.random(n) < 0.04 + 0.01 * faults, rng.lognormal(1.2, 0.6, n), rng.lognormal(-1.8, 0.5, n))
            moist = rng.normal(12 + 0.25 * age, 4, n).clip(2)
            vib = (1.2 + 0.03 * age + 1.5 * np.clip(lf - 0.8, 0, None) + rng.normal(0, 0.3, n)).clip(0.2)
            X = pd.DataFrame(dict(age_years=age, load_factor=lf, oil_temp_c=oil, winding_temp_c=wind_t, dga_h2_ppm=h2,
                                  dga_c2h2_ppm=c2h2, moisture_ppm=moist, vibration_mm_s=vib,
                                  faults_last_year=faults, days_since_maint=since))
            logit = (-7.3 + 0.045 * X.age_years + 3.2 * (X.load_factor - 0.65) + 0.055 * (X.oil_temp_c - 78) +
                     0.004 * X.dga_h2_ppm + 0.9 * (X.dga_c2h2_ppm > 1.0) + 0.0011 * X.days_since_maint +
                     0.45 * X.faults_last_year + 0.02 * X.moisture_ppm + 0.35 * X.vibration_mm_s)
            return X, 1 / (1 + np.exp(-logit))

        parts = []
        for m in range(24):
            X, p = snapshot(len(tr))
            X["transformer_id"] = tr.transformer_id.to_numpy(); X["month"] = m
            X["failed_30d"] = (rng.random(len(tr)) < p).astype(int)
            parts.append(X)
        hist = pd.concat(parts, ignore_index=True)
        X, p = snapshot(len(tr), lf=np.clip(lf_now, 0.15, 1.3))
        X["transformer_id"] = tr.transformer_id.to_numpy()
        truth_fail = pd.DataFrame({"transformer_id": tr.transformer_id, "true_p30": p,
                                   "failed_30d": (rng.random(len(tr)) < p).astype(int)})
        return hist, X, truth_fail

    def _incidents(self, tr, subs, plants):
        rng = self.rng
        kinds = ["transformer_failure", "line_trip", "overload", "voltage_sag", "theft_raid", "plant_trip"]
        rows = []
        for i in range(40):
            k = rng.choice(kinds)
            asset = (rng.choice(tr.transformer_id) if k in ("transformer_failure", "overload", "theft_raid")
                     else rng.choice(subs.substation_id) if k in ("line_trip", "voltage_sag") else rng.choice(plants.plant_id))
            rows.append(dict(incident_id=f"INC-{i+1:04d}", kind=str(k), asset_id=str(asset),
                             start=pd.Timestamp(self.cfg.start) + pd.Timedelta(hours=int(rng.integers(0, self.cfg.history_days * 24))),
                             duration_h=float(rng.uniform(0.5, 18)), customers_affected=int(rng.integers(50, 80000))))
        return pd.DataFrame(rows).sort_values("start", ignore_index=True)

    # -- public ---------------------------------------------------------
    def build(self) -> SimData:
        c = self.cfg
        regions, plants, subs, lines, tr, meters, ren, bat, teams = self._topology()
        weather, demand, seg = self._weather_demand(regions, ren)
        now = pd.Timestamp(c.start) + pd.Timedelta(hours=c.history_days * 24)
        daily, prof, truth_theft, tr_in = self._meters(meters, weather, now)
        rated = tr.set_index("transformer_id").rated_kva
        kwh_tr = daily.merge(meters[["meter_id", "transformer_id"]], on="meter_id").groupby("transformer_id").kwh.sum() / c.meter_days
        lf = (kwh_tr.reindex(tr.transformer_id) / 24 / (rated.reindex(tr.transformer_id) * 0.9) * 11.0).to_numpy()
        hist, cur, truth_fail = self._maintenance(tr, lf)
        tr = tr.merge(tr_in, on="transformer_id", how="left")
        incidents = self._incidents(tr, subs, plants)
        return SimData(cfg=c, regions=regions, plants=plants, substations=subs, lines=lines, transformers=tr,
                       meters=meters, renewables=ren, batteries=bat, teams=teams, weather=weather, demand=demand,
                       segment_demand=seg, meter_daily=daily, meter_profile=prof, maint_history=hist,
                       maint_current=cur, incidents=incidents, truth_theft=truth_theft, truth_failure=truth_fail, now=now)
