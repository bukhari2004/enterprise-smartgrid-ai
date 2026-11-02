"""Predictive Maintenance Engine.

* Transformer failure model (gradient boosting, XGBoost if installed) -> P(fail in 30d), P(7d), expected days to failure
* Local explanations by feature occlusion (each feature replaced by the fleet median) + global permutation importance
* Equipment anomaly monitor (Isolation Forest) for transmission lines, circuit breakers and generators
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from .models_util import backend_name, make_classifier
from .simulator import SimData

FEATS = ["age_years", "load_factor", "oil_temp_c", "winding_temp_c", "dga_h2_ppm", "dga_c2h2_ppm", "moisture_ppm",
         "vibration_mm_s", "faults_last_year", "days_since_maint"]

ACTIONS = {
    "oil_temp_c": "inspect cooling system / radiators and oil level",
    "winding_temp_c": "inspect cooling system and verify winding temperature sensor",
    "dga_h2_ppm": "run full dissolved-gas analysis (DGA) and re-test in 2 weeks",
    "dga_c2h2_ppm": "acetylene detected - possible arcing: schedule urgent internal inspection",
    "moisture_ppm": "oil dehydration / filtration",
    "load_factor": "transfer load to a neighbouring transformer to reduce loading",
    "days_since_maint": "schedule overdue preventive maintenance",
    "vibration_mm_s": "mechanical inspection (core / clamping / tap changer)",
    "faults_last_year": "root-cause review of previous faults",
    "age_years": "include in asset-replacement planning",
}


def tier(p: float) -> str:
    return "critical" if p >= 0.30 else "high" if p >= 0.15 else "medium" if p >= 0.07 else "low"


class PredictiveMaintenance:
    def __init__(self, data: SimData):
        self.d = data
        self.model = None
        self.median = None
        self.metrics: dict = {}
        self.current: pd.DataFrame | None = None

    def fit(self):
        h = self.d.maint_history
        tr, te = h[h.month < 18], h[h.month >= 18]
        m = make_classifier().fit(tr[FEATS], tr.failed_30d)
        p = m.predict_proba(te[FEATS])[:, 1]
        self.metrics = {"backend": backend_name(), "roc_auc": round(float(roc_auc_score(te.failed_30d, p)), 3),
                        "pr_auc": round(float(average_precision_score(te.failed_30d, p)), 3),
                        "base_rate_pct": round(float(te.failed_30d.mean() * 100), 2),
                        "recall_in_top_10pct": round(float(te.failed_30d.to_numpy()[np.argsort(-p)[: max(1, len(p) // 10)]].sum()
                                                           / max(te.failed_30d.sum(), 1)), 3)}
        self.model = make_classifier().fit(h[FEATS], h.failed_30d)
        self.median = h[FEATS].median()
        self.current = self.predict()
        return self

    def predict(self, X: pd.DataFrame | None = None) -> pd.DataFrame:
        X = self.d.maint_current if X is None else X
        p30 = self.model.predict_proba(X[FEATS])[:, 1]
        hazard = -np.log(np.clip(1 - p30, 1e-6, 1)) / 30
        out = pd.DataFrame({"transformer_id": X.transformer_id.to_numpy(), "p_fail_30d": p30,
                            "p_fail_7d": 1 - (1 - p30) ** (7 / 30),
                            "expected_days_to_failure": np.minimum(1 / np.maximum(hazard, 1e-6), 3650).round(0)})
        out["risk_tier"] = [tier(x) for x in p30]
        out = out.merge(self.d.transformers[["transformer_id", "substation_id", "region_id", "rated_kva"]], on="transformer_id")
        return out.sort_values("p_fail_30d", ascending=False, ignore_index=True)

    def explain(self, transformer_id: str, top: int = 4) -> dict:
        row = self.d.maint_current[self.d.maint_current.transformer_id == transformer_id]
        if row.empty:
            raise KeyError(transformer_id)
        base = float(self.model.predict_proba(row[FEATS])[:, 1][0])
        variants = pd.concat([row[FEATS].assign(**{f: self.median[f]}) for f in FEATS], ignore_index=True)
        contrib = base - self.model.predict_proba(variants)[:, 1]
        order = np.argsort(-contrib)
        drivers = [{"feature": FEATS[i], "value": round(float(row[FEATS[i]].iloc[0]), 2),
                    "fleet_median": round(float(self.median[FEATS[i]]), 2), "risk_contribution_pp": round(float(contrib[i] * 100), 2),
                    "action": ACTIONS[FEATS[i]]} for i in order[:top] if contrib[i] > 0]
        team = self.d.teams.set_index("region_id").team_id.get(
            self.d.transformers.set_index("transformer_id").region_id[transformer_id], None)
        return {"transformer_id": transformer_id, "p_fail_30d": round(base, 4), "risk_tier": tier(base), "drivers": drivers,
                "assigned_team": team,
                "summary": (f"{transformer_id} has a {base*100:.1f}% 30-day failure probability ({tier(base)}). Main drivers: "
                            + ", ".join(f"{d['feature']}={d['value']} (+{d['risk_contribution_pp']}pp)" for d in drivers[:3]))}

    def global_importance(self, n: int = 1500) -> pd.DataFrame:
        h = self.d.maint_history.sample(min(n, len(self.d.maint_history)), random_state=0)
        base = roc_auc_score(h.failed_30d, self.model.predict_proba(h[FEATS])[:, 1])
        rng = np.random.default_rng(0); rows = []
        for f in FEATS:
            hp = h.copy(); hp[f] = rng.permutation(hp[f].to_numpy())
            rows.append((f, base - roc_auc_score(h.failed_30d, self.model.predict_proba(hp[FEATS])[:, 1])))
        return pd.DataFrame(rows, columns=["feature", "auc_drop"]).sort_values("auc_drop", ascending=False, ignore_index=True).round(4)

    def evaluate_current(self) -> dict:
        """Compare predictions with the simulator's hidden outcomes (demo evaluation only)."""
        t = self.d.truth_failure.merge(self.current, on="transformer_id")
        if t.failed_30d.sum() == 0:
            return {}
        return {"roc_auc": round(float(roc_auc_score(t.failed_30d, t.p_fail_30d)), 3),
                "actual_failures": int(t.failed_30d.sum()),
                "failures_in_top_20": int(t.sort_values("p_fail_30d", ascending=False).head(20).failed_30d.sum())}


class EquipmentAnomalyMonitor:
    """Isolation-Forest health monitor for lines, breakers and generators (sensor snapshots are simulated)."""

    def __init__(self, data: SimData):
        rng = np.random.default_rng(data.cfg.seed + 11)
        self.d = data
        L = data.lines
        self.frames = {
            "transmission_line": pd.DataFrame({"asset_id": L.line_id, "loading_pct": rng.normal(55, 12, len(L)).clip(5),
                                                "conductor_temp_c": rng.normal(60, 8, len(L)), "sag_m": rng.normal(6, 0.8, len(L)),
                                                "partial_discharge_pc": rng.lognormal(1.0, 0.4, len(L))}),
            "circuit_breaker": pd.DataFrame({"asset_id": [f"CB-{i+1:04d}" for i in range(len(data.substations) * 3)],
                                             "operations": rng.poisson(900, len(data.substations) * 3).astype(float),
                                             "contact_resistance_uohm": rng.normal(60, 8, len(data.substations) * 3),
                                             "trip_time_ms": rng.normal(35, 3, len(data.substations) * 3)}),
            "generator": pd.DataFrame({"asset_id": data.plants.plant_id, "vibration_mm_s": rng.normal(2.5, 0.4, len(data.plants)),
                                       "bearing_temp_c": rng.normal(70, 5, len(data.plants)),
                                       "efficiency_pct": rng.normal(38, 1.5, len(data.plants))}),
        }
        for k in self.frames:                          # inject a few real faults
            f = self.frames[k]; i = rng.integers(len(f)); c = f.columns[1 + rng.integers(len(f.columns) - 1)]
            f.loc[i, c] = f[c].mean() + 5 * f[c].std()

    def scan(self) -> pd.DataFrame:
        rows = []
        for kind, f in self.frames.items():
            X = f.drop(columns="asset_id")
            iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=0).fit(X)
            score = -iso.score_samples(X)
            z = (X - X.mean()) / (X.std() + 1e-9)
            for i in range(len(f)):
                worst = z.iloc[i].abs().idxmax()
                rows.append(dict(kind=kind, asset_id=f.asset_id.iloc[i], anomaly_score=round(float(score[i]), 3),
                                 top_signal=worst, z=round(float(z.iloc[i][worst]), 1)))
        out = pd.DataFrame(rows)
        out["health_pct"] = (100 * (1 - (out.anomaly_score - out.anomaly_score.min()) / (out.anomaly_score.max() - out.anomaly_score.min() + 1e-9))).round(0)
        return out.sort_values("anomaly_score", ascending=False, ignore_index=True)
