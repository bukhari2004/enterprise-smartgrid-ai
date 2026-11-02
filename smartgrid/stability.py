"""Voltage / frequency stability analytics + load-balancing recommendations (fully explainable physics-style models)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import SimData

V_NORMAL = (0.95, 1.05)
V_WARN = (0.93, 1.07)


class GridStability:
    def __init__(self, data: SimData):
        self.d = data
        rng = np.random.default_rng(data.cfg.seed + 7)
        subs = data.substations.copy()
        subs["util_factor"] = rng.uniform(0.85, 1.20, len(subs))
        subs["cap_bank"] = (np.arange(len(subs)) % 2 == 0)           # every other substation has reactive support
        self.subs = subs
        dem = data.demand.pivot(index="ts", columns="region_id", values="demand_mw")
        self.region_peak = dem.max().to_dict()

    def loading(self, region_demand_mw: dict[str, float]) -> pd.DataFrame:
        s = self.subs.copy()
        s["region_load_pu"] = s.region_id.map(lambda r: region_demand_mw.get(r, 0.0) / self.region_peak[r])
        s["loading"] = s.region_load_pu * 0.80 * s.util_factor
        return s

    # ---------------------------------------------------------------- voltage
    @staticmethod
    def _voltage_terms(loading, feeder_km, ren_ramp, cap_bank):
        a_load = 0.07 * loading ** 2
        a_feed = 0.0005 * feeder_km * loading
        a_ren = 0.04 * ren_ramp
        a_cap = np.where(cap_bank & (loading > 0.6), 0.015, 0.0)
        return a_load, a_feed, a_ren, a_cap

    def voltage_profile(self, region_demand_mw: dict[str, float], ren_ramp_pu: dict[str, float] | None = None) -> pd.DataFrame:
        s = self.loading(region_demand_mw)
        ramp = s.region_id.map(lambda r: (ren_ramp_pu or {}).get(r, 0.0)).to_numpy()
        a_load, a_feed, a_ren, a_cap = self._voltage_terms(s.loading.to_numpy(), s.feeder_km.to_numpy(), ramp, s.cap_bank.to_numpy())
        s["voltage_pu"] = 1.03 - a_load - a_feed - a_ren + a_cap
        s["ren_ramp_pu"] = ramp
        s["c_load"], s["c_feeder"], s["c_renewable_ramp"], s["c_cap_bank"] = -a_load, -a_feed, -a_ren, a_cap
        lo, hi = s.voltage_pu < V_WARN[0], s.voltage_pu < V_NORMAL[0]
        s["status"] = np.where(lo, "critical", np.where(hi, "warning", "normal"))
        s["overload"] = s.loading > 1.0
        return s

    def explain_voltage(self, prof: pd.DataFrame, substation_id: str) -> dict:
        r = prof[prof.substation_id == substation_id].iloc[0]
        contrib = {"heavy loading": r.c_load, "long feeder drop": r.c_feeder,
                   "renewable output ramp": r.c_renewable_ramp, "reactive support (cap bank)": r.c_cap_bank}
        ranked = sorted(contrib.items(), key=lambda kv: kv[1])
        main = ranked[0]
        # counterfactuals
        t = self._voltage_terms(np.array([0.8]), np.array([r.feeder_km]), np.array([r.ren_ramp_pu]), np.array([bool(r.cap_bank)]))
        v_if_load = 1.03 - t[0][0] - t[1][0] - t[2][0] + t[3][0]
        actions = []
        if r.loading > 0.85:
            actions.append("transfer load to neighbouring substations (see /optimization/load-balance)")
        if not r.cap_bank:
            actions.append("switch in / install a capacitor bank or STATCOM for reactive support")
        if r.ren_ramp_pu > 0.2:
            actions.append("dispatch battery storage / smooth renewable ramp with ramp-rate limits")
        actions.append("adjust on-load tap changer (+1 to +2 taps)")
        return {"substation_id": substation_id, "voltage_pu": round(float(r.voltage_pu), 4), "status": r.status,
                "loading_pct": round(float(r.loading) * 100, 1),
                "contributions_pu": {k: round(float(v), 4) for k, v in contrib.items()},
                "primary_cause": main[0],
                "explanation": (f"{substation_id} sits at {r.voltage_pu:.3f} pu ({r.status}). Largest depressing factor: "
                                f"{main[0]} ({main[1]:+.3f} pu). Loading is {r.loading*100:.0f}% of capacity."),
                "counterfactual": f"at 80% loading the voltage would be about {float(v_if_load):.3f} pu",
                "recommended_actions": actions}

    # ---------------------------------------------------------------- frequency
    def frequency_response(self, loss_mw: float, demand_mw: float, spinning_reserve_mw: float | None = None,
                           inertia_h: float = 4.0, f0: float = 50.0, seconds: int = 40) -> dict:
        """Swing-equation simulation of a sudden generation loss with governor droop response."""
        base = demand_mw * 1.2
        reserve = (spinning_reserve_mw if spinning_reserve_mw is not None
                   else float(self.d.plants.capacity_mw.sum()) - demand_mw) * 0.25      # only part of headroom is governor-ready
        dt, D, R, Tg = 0.1, 1.0, 0.05, 6.0
        df = dpm = shed = 0.0; fs = []
        stages = [(49.0, 0.08), (48.8, 0.08), (48.6, 0.08)]        # under-frequency load shedding (UFLS) blocks
        done = set()
        for _ in range(int(seconds / dt)):
            f = f0 * (1 + df)
            for k, (thr, blk) in enumerate(stages):
                if f < thr and k not in done:
                    done.add(k); shed += blk
            dpm_ref = np.clip(-df / R * base, 0, reserve) / base
            dpm += (dpm_ref - dpm) / Tg * dt
            ddf = (dpm - loss_mw / base + shed - D * df) / (2 * inertia_h) * dt
            df += ddf; fs.append(f0 * (1 + df))
        fs = np.array(fs)
        rocof = (fs[2] - fs[0]) / (2 * dt)
        nadir = float(fs.min())
        return {"loss_mw": loss_mw, "nadir_hz": round(nadir, 3), "nadir_time_s": round(float(fs.argmin() * dt), 1),
                "settle_hz": round(float(fs[-1]), 3), "rocof_hz_s": round(float(rocof), 3),
                "under_frequency_load_shedding": bool(shed > 0), "load_shed_mw": round(shed * base),
                "status": "critical" if nadir < 49.0 else "warning" if nadir < 49.5 else "stable",
                "trajectory_hz": [round(float(x), 3) for x in fs[::10]],
                "recommendation": ("increase spinning reserve / fast frequency response (battery) before dispatching this contingency"
                                   if nadir < 49.5 else "contingency is within the frequency security limits")}

    # ---------------------------------------------------------------- load balancing
    def load_balance(self, region_demand_mw: dict[str, float], target: float = 0.80) -> list[dict]:
        s = self.loading(region_demand_mw)
        moves = []
        for rid, g in s.groupby("region_id"):
            over = g[g.loading > target].sort_values("loading", ascending=False)
            under = g[g.loading < target - 0.05].sort_values("loading")
            headroom = {r.substation_id: (target - r.loading) * r.capacity_mva for _, r in under.iterrows()}
            for _, o in over.iterrows():
                excess = (o.loading - target) * o.capacity_mva
                for sid in list(headroom):
                    if excess <= 1e-6:
                        break
                    mv = min(excess, headroom[sid])
                    if mv > 1:
                        moves.append({"from": o.substation_id, "to": sid, "region_id": rid, "transfer_mva": round(float(mv), 1)})
                        headroom[sid] -= mv; excess -= mv
        return moves
