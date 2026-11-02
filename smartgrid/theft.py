"""Energy Theft Detection Engine.

Signals: peer deviation, step-change after a point in time (meter tampering), transformer energy-balance gap
(illegal connections / by-pass), abnormal night/day ratio (commercial fraud), billing vs metered mismatch.
Models: Isolation Forest (unsupervised) + gradient boosting trained on past inspection outcomes (semi-supervised),
blended into a single 0-1 suspicion score. Billing anomalies are separated from theft.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from .simulator import SimData

FEATS = ["mean_kwh", "cv", "recent_vs_prior", "min_week_ratio", "zero_days", "peer_z", "local_peer_ratio",
         "night_ratio", "billed_ratio", "tr_gap", "max_drop"]


def _rank01(x):
    return pd.Series(x).rank(pct=True).to_numpy()


class TheftDetector:
    def __init__(self, data: SimData):
        self.d = data
        self.features: pd.DataFrame | None = None
        self.scores: pd.DataFrame | None = None
        self.metrics: dict = {}

    # ------------------------------------------------------------- features
    def build_features(self) -> pd.DataFrame:
        d = self.d
        M = d.meter_daily.pivot(index="day", columns="meter_id", values="kwh")[d.meters.meter_id.to_numpy()]
        meters = d.meters.reset_index(drop=True)
        mean = M.mean()
        n = len(M)
        recent, prior = M.iloc[-14:].mean(), M.iloc[:-14].mean()
        wk = M.rolling(7).mean().dropna()
        a, b = M.iloc[: n // 2].mean(), M.iloc[n // 2:].mean()
        # biggest drop between consecutive 14-day windows
        w14 = M.rolling(14).mean().dropna()
        drops = (w14.shift(14) - w14).iloc[14:]
        max_drop = (drops.max() / (mean + 1e-9)).clip(lower=0)
        F = pd.DataFrame({"meter_id": meters.meter_id.to_numpy(), "tariff": meters.tariff.to_numpy(),
                          "transformer_id": meters.transformer_id.to_numpy()})
        F["mean_kwh"] = mean.to_numpy()
        F["cv"] = (M.std() / (mean + 1e-9)).to_numpy()
        F["recent_vs_prior"] = (recent / (prior + 1e-9)).to_numpy()
        F["min_week_ratio"] = (wk.min() / (mean + 1e-9)).to_numpy()
        F["zero_days"] = (M < 0.5).sum().to_numpy()
        # peer comparison against same-tariff fleet (robust z on log consumption)
        lg = np.log(F.mean_kwh.clip(lower=0.1))
        med = lg.groupby(F.tariff).transform("median"); mad = (lg - med).abs().groupby(F.tariff).transform("median") + 1e-6
        F["peer_z"] = ((lg - med) / (1.4826 * mad)).to_numpy()
        # local: ratio to same-tariff median on the same transformer
        loc_med = F.groupby(["transformer_id", "tariff"]).mean_kwh.transform("median")
        F["local_peer_ratio"] = (F.mean_kwh / (loc_med + 1e-9)).to_numpy()
        F = F.merge(d.meter_profile, on="meter_id")
        # transformer energy balance
        tr = d.transformers.set_index("transformer_id").input_kwh_30d
        sum30 = M.iloc[-30:].sum().groupby(meters.transformer_id.to_numpy()).sum()
        gap = (1 - sum30 / tr.reindex(sum30.index)).rename("tr_gap")
        F["tr_gap"] = F.transformer_id.map(gap).to_numpy()
        F["max_drop"] = max_drop.to_numpy()
        self.features = F
        return F

    # ------------------------------------------------------------- training
    def fit(self, label_fraction: float = 0.35, seed: int = 0):
        F = self.features if self.features is not None else self.build_features()
        truth = self.d.truth_theft.set_index("meter_id").is_theft.reindex(F.meter_id).to_numpy().astype(int)
        rng = np.random.default_rng(seed)
        labelled = rng.random(len(F)) < label_fraction              # meters inspected in the past
        X = F[FEATS]
        self.iso = IsolationForest(n_estimators=200, contamination=self.d.cfg.theft_rate, random_state=seed).fit(X)
        iso_score = -self.iso.score_samples(X)
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=12,
                                                  class_weight="balanced", random_state=seed).fit(X[labelled], truth[labelled])
        clf_p = self.clf.predict_proba(X)[:, 1]
        blend = 0.35 * _rank01(iso_score) + 0.65 * _rank01(clf_p)
        # blend in a cheap, transparent rule score as a sanity floor
        self.scores = pd.DataFrame({"meter_id": F.meter_id, "iso_score": iso_score, "clf_prob": clf_p, "score": blend})
        hold = ~labelled
        self.metrics = {
            "labelled_meters": int(labelled.sum()), "holdout_meters": int(hold.sum()),
            "holdout_roc_auc_blend": round(float(roc_auc_score(truth[hold], blend[hold])), 3),
            "holdout_roc_auc_isolation_forest": round(float(roc_auc_score(truth[hold], iso_score[hold])), 3),
            "holdout_pr_auc_blend": round(float(average_precision_score(truth[hold], blend[hold])), 3),
            "precision_at_top_100_holdout": round(float(truth[hold][np.argsort(-blend[hold])[:100]].mean()), 3),
            "base_rate_pct": round(float(truth.mean() * 100), 2)}
        return self

    # ------------------------------------------------------------- outputs
    def classify(self, r: pd.Series) -> str:
        if abs(r.billed_ratio - 1) > 0.2 and r.peer_z > -1.5 and r.recent_vs_prior > 0.8:
            return "billing_anomaly"
        if r.max_drop > 0.35 and r.recent_vs_prior < 0.75:
            return "meter_tampering"
        if r.tariff != "residential" and r.night_ratio > 1.4 * {"commercial": 0.25, "industrial": 0.9}[r.tariff]:
            return "commercial_fraud"
        if r.tr_gap > 0.08 or r.local_peer_ratio < 0.6:
            return "illegal_connection"
        return "suspicious_pattern"

    def suspects(self, top: int = 50, region: str | None = None, min_score: float = 0.9) -> pd.DataFrame:
        F, S = self.features, self.scores
        df = F.merge(S, on="meter_id").merge(self.d.meters[["meter_id", "region_id", "lat", "lon"]], on="meter_id")
        df = df[df.score >= min_score]
        if region:
            df = df[df.region_id == region]
        df = df.sort_values("score", ascending=False).head(top).copy()
        df["type"] = df.apply(self.classify, axis=1)
        med = F.groupby(["transformer_id", "tariff"]).mean_kwh.median()
        df["est_loss_kwh_day"] = [max(0.0, med.get((r.transformer_id, r.tariff), r.mean_kwh) - r.mean_kwh) for r in df.itertuples()]
        cols = ["meter_id", "transformer_id", "region_id", "tariff", "score", "clf_prob", "type", "mean_kwh",
                "est_loss_kwh_day", "recent_vs_prior", "tr_gap", "billed_ratio"]
        return df[cols].round(3).reset_index(drop=True)

    def heatmap(self, min_score: float = 0.93, cell_deg: float = 0.25) -> pd.DataFrame:
        s = self.suspects(top=100000, min_score=min_score)
        s = s.merge(self.d.meters[["meter_id", "lat", "lon"]], on="meter_id")
        s["cell_lat"] = (s.lat / cell_deg).round() * cell_deg; s["cell_lon"] = (s.lon / cell_deg).round() * cell_deg
        h = s.groupby(["cell_lat", "cell_lon", "region_id"], as_index=False).agg(suspects=("meter_id", "count"),
                                                                              loss_kwh_day=("est_loss_kwh_day", "sum"))
        return h.sort_values("suspects", ascending=False, ignore_index=True).round(2)

    def region_summary(self, min_score: float = 0.93) -> pd.DataFrame:
        s = self.suspects(top=100000, min_score=min_score)
        tot = self.d.meters.groupby("region_id").size().rename("meters")
        g = s.groupby("region_id").agg(suspects=("meter_id", "count"), loss_kwh_day=("est_loss_kwh_day", "sum")).join(tot, how="right").fillna(0)
        g["suspect_rate_pct"] = (g.suspects / g.meters * 100).round(2)
        return g.reset_index().round(1).sort_values("suspect_rate_pct", ascending=False, ignore_index=True)

    def transformer_balance(self, threshold: float = 0.08) -> pd.DataFrame:
        t = self.features.groupby("transformer_id").tr_gap.first().reset_index()
        t = t[t.tr_gap > threshold].sort_values("tr_gap", ascending=False, ignore_index=True)
        t["tr_gap"] = t.tr_gap.round(3)
        return t

    def explain(self, meter_id: str) -> dict:
        r = self.features[self.features.meter_id == meter_id]
        if r.empty:
            raise KeyError(meter_id)
        r = r.iloc[0]; sc = self.scores[self.scores.meter_id == meter_id].iloc[0]
        evidence = []
        if r.recent_vs_prior < 0.75: evidence.append(f"consumption fell to {r.recent_vs_prior*100:.0f}% of its earlier level in the last 14 days")
        if r.max_drop > 0.35: evidence.append("sudden step-change drop consistent with meter tampering")
        if r.local_peer_ratio < 0.6: evidence.append(f"reads only {r.local_peer_ratio*100:.0f}% of similar meters on the same transformer")
        if r.tr_gap > 0.08: evidence.append(f"its transformer shows {r.tr_gap*100:.1f}% unaccounted energy (input minus metered)")
        if r.tariff != "residential" and r.night_ratio > 1.4 * {"commercial": 0.25, "industrial": 0.9}[r.tariff]:
            evidence.append("night-time usage is abnormally high for its tariff class")
        if abs(r.billed_ratio - 1) > 0.2: evidence.append(f"billed energy differs {abs(r.billed_ratio-1)*100:.0f}% from metered energy")
        return {"meter_id": meter_id, "suspicion_score": round(float(sc.score), 3), "model_probability": round(float(sc.clf_prob), 3),
                "type": self.classify(r), "evidence": evidence or ["no strong single signal (score driven by combined pattern)"],
                "features": {k: round(float(r[k]), 3) for k in FEATS}}
