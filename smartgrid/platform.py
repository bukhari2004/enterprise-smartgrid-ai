"""GridPlatform: orchestrates every engine and exposes the operations used by the REST API, copilot and CLI."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import __version__
from .alerts import Alert, AlertCenter
from .audit import AuditLog
from .config import GridConfig, Settings, load_config
from .forecasting import DemandForecaster
from .graph import GridKnowledgeGraph
from .maintenance import EquipmentAnomalyMonitor, PredictiveMaintenance
from .models_util import backend_name
from .optimization import BatteryRL, DispatchOptimizer, merit_price_fn
from .renewables import RenewableForecaster
from .security import Auth
from .simulator import GridSimulator, SimData
from .stability import GridStability
from .storage import Cache, Store
from .streaming import EventGenerator, StreamProcessor, get_bus
from .theft import TheftDetector
from .utils import clean, rnd


class GridPlatform:
    def __init__(self, cfg: GridConfig | None = None, verbose: bool = True, db_url: str | None = None):
        self.cfg = cfg or load_config()
        self.verbose = verbose
        self.timings: dict[str, float] = {}
        self.ready = False
        self._cache: dict = {}
        self._db_url = db_url

    # ------------------------------------------------------------------ lifecycle
    def _t(self, name, fn):
        t = time.time(); r = fn(); self.timings[name] = round(time.time() - t, 2)
        if self.verbose:
            print(f"[init] {name:<22}{self.timings[name]:>6.2f}s", flush=True)
        return r

    def initialize(self) -> "GridPlatform":
        s = Settings
        self.d: SimData = self._t("simulate grid", lambda: GridSimulator(self.cfg).build())
        self.store = Store(self._db_url or s.database_url)
        self.audit = AuditLog(self.store, s.audit_path if (self._db_url or s.database_url) != "sqlite://" else None)
        self.cache = Cache(s.redis_url)
        self.auth = Auth()
        self.alerts = AlertCenter(self.store)
        self.demand = self._t("demand forecaster", lambda: DemandForecaster(self.d).fit())
        self.ren = self._t("renewable forecaster", lambda: RenewableForecaster(self.d).fit())
        self.opt = DispatchOptimizer(self.d)
        self.pm = self._t("predictive maint.", lambda: PredictiveMaintenance(self.d).fit())
        self.equip = EquipmentAnomalyMonitor(self.d)
        self.theft = self._t("theft detector", self._init_theft)
        self.stab = GridStability(self.d)
        self.kg = self._t("knowledge graph", lambda: GridKnowledgeGraph(self.d))
        self.bus = get_bus(s.kafka_bootstrap, s.kafka_topic)
        self.stream = StreamProcessor(self.d, self.alerts)
        self.gen = EventGenerator(self.d)
        self._rl: BatteryRL | None = None
        import threading
        self._rl_lock = threading.Lock()
        self.ready = True
        self.audit.record("system", "platform.initialized", {"version": __version__, "timings": self.timings})
        return self

    # fix: TheftDetector needs build_features before fit, keep a single instance
    def _init_theft(self):
        td = TheftDetector(self.d); td.build_features(); td.fit(); return td

    @property
    def now(self) -> pd.Timestamp:
        return self.d.now

    # ------------------------------------------------------------------ forecasting
    def demand_forecast(self, horizon=48, region=None, temp_delta=0.0, surge_pct=0.0) -> dict:
        key = ("fc", horizon, region, temp_delta, surge_pct)
        if key not in self._cache:
            fc = self.demand.forecast(horizon, region, temp_delta, surge_pct)
            nat = fc.groupby("ts", as_index=False)[["mw", "p10", "p90"]].sum()
            self._cache[key] = {"national": nat, "by_region": fc, "scenario": {"temp_delta_c": temp_delta, "surge_pct": surge_pct}}
        c = self._cache[key]
        nat = c["national"]; i = nat.mw.idxmax()
        return clean({"horizon_h": horizon, "scenario": c["scenario"], "model": self.demand.metrics,
                      "peak": {"mw": nat.mw[i], "time": nat.ts[i], "p10_mw": nat.p10[i], "p90_mw": nat.p90[i]},
                      "national": rnd(nat, 1), "by_region": rnd(c["by_region"], 1) if region else None})

    def forecast_day(self, offset: int = 1) -> dict:
        """offset 0 = today (first forecast day), 1 = tomorrow."""
        nat = pd.DataFrame(self.demand_forecast(48)["national"]); nat["ts"] = pd.to_datetime(nat.ts)
        day = (self.now.normalize() + pd.Timedelta(days=offset))
        d = nat[(nat.ts >= day) & (nat.ts < day + pd.Timedelta(days=1))]
        i = d.mw.idxmax()
        return clean({"date": day.date(), "peak_mw": d.mw[i], "peak_time": d.ts[i], "avg_mw": d.mw.mean(), "min_mw": d.mw.min(),
                      "energy_gwh": d.mw.sum() / 1000, "p10_at_peak": d.p10[i], "p90_at_peak": d.p90[i], "hourly": rnd(d[["ts", "mw"]], 0)})

    def renewable_forecast(self, horizon=48) -> dict:
        nat = self.ren.national(horizon)
        cap = float(self.d.renewables.capacity_mw.sum())
        return clean({"model": self.ren.metrics, "installed_mw": cap, "forecast": rnd(nat, 1),
                      "expected_drops": self.ren.detect_drops(min(horizon, 24)), "peak_renewable_mw": nat.renewable_mw.max()})

    # ------------------------------------------------------------------ optimisation
    def _inputs(self, horizon, temp_delta=0.0, surge_pct=0.0):
        d = pd.DataFrame(self.demand_forecast(horizon, None, temp_delta, surge_pct)["national"])
        r = self.ren.national(horizon)
        return pd.DatetimeIndex(pd.to_datetime(d.ts)), d.mw.to_numpy(), r.renewable_mw.to_numpy()

    def dispatch(self, horizon=24, outage_plants=None, carbon_price=0.0, temp_delta=0.0, surge_pct=0.0, use_battery=True) -> dict:
        ts, dem, ren = self._inputs(horizon, temp_delta, surge_pct)
        res = self.opt.solve(dem, ren, ts, outage_plants, carbon_price, use_battery)
        if res["status"] != "optimal":
            return res
        return clean({"status": "optimal", "summary": res["summary"], "schedule": rnd(res["schedule"], 1)})

    def battery_rl(self, retrain: bool = False) -> dict:
        with self._rl_lock:
            if self._rl is None or retrain:
                dem = self.d.demand.groupby("ts").demand_mw.sum()
                w = self.d.weather[self.d.weather.solar_mw.notna()].groupby("ts")[["solar_mw", "wind_mw"]].sum().sum(axis=1)
                net = (dem - w.reindex(dem.index)).to_numpy()
                n = len(net) // 24 * 24
                price = merit_price_fn(self.d.plants)(net[:n].reshape(-1, 24))
                rl = BatteryRL(cap_mwh=self.opt.batt_cap, power_mw=self.opt.batt_pow, eff=self.opt.eff)
                self._t("battery RL train", lambda: rl.fit(price[:-30], episodes=40000))
                rl.evaluate(price[-30:])
                self._rl = rl
        rl = self._rl
        ts, dem, ren = self._inputs(24)
        p = merit_price_fn(self.d.plants)(dem - ren)
        run = rl.run_day(p, "rl")
        names = {0: "idle", 1: "charge", 2: "discharge"}
        sched = pd.DataFrame({"ts": ts, "price_usd_mwh": p.round(1), "action": [names[a] for a in run["actions"]]})
        return clean({"training_report": rl.report, "tomorrow_schedule": sched, "expected_profit_usd": run["profit"],
                      "battery": {"capacity_mwh": self.opt.batt_cap, "power_mw": self.opt.batt_pow}})

    # ------------------------------------------------------------------ stability
    def _stress_inputs(self, horizon=24, temp_delta=0.0, surge_pct=0.0):
        raw = self.demand.forecast(horizon, None, temp_delta, surge_pct)
        peak = raw.groupby("region_id").mw.max().to_dict()
        rf = self.ren.forecast(horizon)
        cap = self.d.renewables.groupby("region_id").capacity_mw.sum()
        tot = rf.assign(r=rf.solar_mw + rf.wind_mw).pivot(index="ts", columns="region_id", values="r").sort_index()
        ramp = (tot.diff().abs().max() / cap.reindex(tot.columns).replace(0, np.nan)).fillna(0).clip(0, 1).to_dict()
        return peak, ramp

    def voltage(self, temp_delta=0.0, surge_pct=0.0) -> dict:
        peak, ramp = self._stress_inputs(24, temp_delta, surge_pct)
        prof = self.stab.voltage_profile(peak, ramp)
        self._last_voltage = prof
        cols = ["substation_id", "region_id", "loading", "voltage_pu", "status", "overload", "c_load", "c_feeder", "c_renewable_ramp", "c_cap_bank"]
        return clean({"scenario": {"temp_delta_c": temp_delta, "surge_pct": surge_pct},
                      "summary": prof.status.value_counts().to_dict(), "substations": rnd(prof[cols], 4).sort_values("voltage_pu")})

    def explain_voltage(self, substation_id: str | None = None, temp_delta=0.0, surge_pct=0.0) -> dict:
        self.voltage(temp_delta, surge_pct)
        prof = self._last_voltage
        sid = substation_id or prof.sort_values("voltage_pu").substation_id.iloc[0]
        return clean(self.stab.explain_voltage(prof, sid))

    def frequency(self, loss_mw: float | None = None) -> dict:
        nat = pd.DataFrame(self.demand_forecast(24)["national"])
        dem = float(nat.mw.max())
        if loss_mw is None:
            loss_mw = float(self.d.plants.capacity_mw.max()) * 0.5
        spinning = float(self.d.plants.capacity_mw.sum()) - dem
        return clean(self.stab.frequency_response(loss_mw, dem, spinning))

    def load_balance(self, horizon=24) -> dict:
        peak, _ = self._stress_inputs(24)
        moves = self.stab.load_balance(peak)
        plan = self.dispatch(horizon)
        return clean({"substation_transfers": moves, "generation_plan_summary": plan.get("summary"),
                      "note": "transfers bring every substation to <=80% loading where headroom exists"})

    # ------------------------------------------------------------------ maintenance
    def maintenance(self, top=15, horizon_days=7) -> dict:
        cur = self.pm.current.copy()
        gr = self.kg.propagate_risk(dict(zip(cur.transformer_id, cur.p_fail_30d)))
        cur = cur.merge(gr[["transformer_id", "graph_risk"]], on="transformer_id")
        cust = self.d.meters.groupby("transformer_id").size().rename("customers")
        cur = cur.merge(cust, on="transformer_id")
        cur["p_fail_horizon"] = 1 - (1 - cur.p_fail_30d) ** (horizon_days / 30)
        cur["priority"] = cur.p_fail_horizon * cur.customers
        cur = cur.sort_values("p_fail_horizon", ascending=False, ignore_index=True)
        return clean({"horizon_days": horizon_days, "model": self.pm.metrics, "validation_on_current_fleet": self.pm.evaluate_current(),
                      "tiers": cur.risk_tier.value_counts().to_dict(), "top": rnd(cur.head(top), 4),
                      "global_importance": self.pm.global_importance().head(6)})

    # ------------------------------------------------------------------ theft
    def theft_overview(self) -> dict:
        td = self.theft
        return clean({"model": td.metrics, "regions": td.region_summary(), "heatmap": td.heatmap().head(30),
                      "transformer_balance_flags": td.transformer_balance().head(10)})

    # ------------------------------------------------------------------ outage simulation
    def simulate_plant_outage(self, plant_id: str | None = None, hours: int = 24, temp_delta: float = 0.0) -> dict:
        ts, dem, ren = self._inputs(hours, temp_delta)
        base = self.opt.solve(dem, ren, ts)
        if plant_id is None:
            out_mw = base["schedule"][list(self.d.plants.plant_id)].max()
            plant_id = out_mw.idxmax()
        if plant_id not in set(self.d.plants.plant_id):
            raise KeyError(plant_id)
        scen = self.opt.solve(dem, ren, ts, outage_plants=[plant_id])
        lost = float(base["schedule"][plant_id].max())
        freq = self.stab.frequency_response(lost, float(dem.max()), float(self.d.plants.capacity_mw.sum() - dem.max()))
        impact = self.kg.impact(plant_id)
        b, s = base["summary"], scen["summary"]
        actions = []
        if s["unserved_mwh"] > 0:
            actions.append(f"activate demand response / rolling load shed plan: {s['unserved_mwh']:.0f} MWh would be unserved")
        if s["peak_price_usd_mwh"] > 1.5 * b["peak_price_usd_mwh"]:
            actions.append("peaking units (OCGT/oil) become marginal - pre-commit them and secure fuel")
        if freq["status"] != "stable":
            actions.append("frequency nadir is outside secure limits - increase spinning reserve / battery fast response before the contingency")
        if not impact["n_minus_1_secure"]:
            actions.append(f"{len(impact['islanded_substations'])} substations lose all supply - pre-arrange switching to alternate feeds")
        if not actions:
            actions.append("system remains secure: dispatch re-optimises onto remaining units")
        return clean({"plant": plant_id, "plant_type": self.d.plants.set_index("plant_id").type[plant_id], "scheduled_output_lost_mw": lost,
                      "horizon_h": hours, "baseline": b, "scenario": s,
                      "extra_fuel_cost_usd": s["fuel_cost_usd"] - b["fuel_cost_usd"], "extra_emissions_tco2": s["emissions_tco2"] - b["emissions_tco2"],
                      "frequency_event": {k: freq[k] for k in ("nadir_hz", "rocof_hz_s", "status", "load_shed_mw")},
                      "topology_impact": impact, "recommended_actions": actions,
                      "price_profile": rnd(pd.DataFrame({"ts": ts, "baseline": base["schedule"].price_usd_mwh.to_numpy(),
                                                          "outage": scen["schedule"].price_usd_mwh.to_numpy()}), 1)})

    # ------------------------------------------------------------------ alerts + dashboard
    def scan_alerts(self) -> dict:
        new = []
        def emit(a: Alert):
            r = self.alerts.emit(a)
            if r: new.append(r)
        for r in self.pm.current.head(15).itertuples():
            if r.p_fail_30d >= 0.30:
                emit(Alert("TRANSFORMER_FAILURE_RISK", "critical" if r.p_fail_30d >= 0.5 else "high", r.transformer_id,
                           f"{r.transformer_id}: {r.p_fail_30d*100:.0f}% failure probability within 30 days", r.region_id,
                           {"p_fail_30d": round(r.p_fail_30d, 3), "p_fail_7d": round(r.p_fail_7d, 3)}))
        v = self.voltage()["substations"]
        for r in v:
            if r["status"] == "critical":
                emit(Alert("VOLTAGE_INSTABILITY", "critical", r["substation_id"], f"voltage {r['voltage_pu']:.3f} pu", r["region_id"], r))
            elif r["status"] == "warning":
                emit(Alert("VOLTAGE_INSTABILITY", "warning", r["substation_id"], f"voltage {r['voltage_pu']:.3f} pu", r["region_id"], {"voltage_pu": r["voltage_pu"]}))
            if r["overload"]:
                emit(Alert("GRID_OVERLOAD", "high", r["substation_id"], f"loading {r['loading']*100:.0f}%", r["region_id"], {"loading": r["loading"]}))
        for r in self.theft.suspects(top=10, min_score=0.99).itertuples():
            emit(Alert("ENERGY_THEFT", "warning", r.meter_id, f"{r.type.replace('_', ' ')} suspected (score {r.score:.2f})", r.region_id,
                       {"score": r.score, "est_loss_kwh_day": r.est_loss_kwh_day}))
        for r in self.ren.detect_drops(24).itertuples():
            emit(Alert("RENEWABLE_ENERGY_DROP", "warning", "RENEWABLES", f"forecast ramp-down of {abs(r.delta_mw):.0f} MW within 3h at {r.ts:%H:%M}", "",
                       {"delta_mw": r.delta_mw}))
        pk = self.demand.peak(24)
        if pk["reserve_margin_pct"] < 15:
            emit(Alert("RESERVE_MARGIN_LOW", "high", "GRID", f"reserve margin only {pk['reserve_margin_pct']}% at forecast peak", "", pk))
        return clean({"new_alerts": len(new), "open_by_type": self.alerts.counts()})

    def health_score(self) -> dict:
        pk = self.demand.peak(24)
        reserve = float(np.clip((pk["reserve_margin_pct"] - 5) / 25, 0, 1) * 100)
        v = self.voltage()["summary"]; ns = max(sum(v.values()), 1)
        volt = 100 * (v.get("normal", 0) + 0.5 * v.get("warning", 0)) / ns
        tiers = self.pm.current.risk_tier.value_counts(); nt = max(len(self.pm.current), 1)
        equip = float(np.clip(100 - 300 * (tiers.get("critical", 0) + 0.5 * tiers.get("high", 0)) / nt, 0, 100))
        crit = sum(1 for a in self.alerts.alerts if a.status == "open" and a.severity == "critical")
        alerts = float(np.clip(100 - 4 * crit, 0, 100))
        trate = len(self.theft.suspects(top=100000, min_score=0.95)) / len(self.d.meters)
        theft = float(np.clip(100 - 2000 * trate, 0, 100))
        score = 0.25 * reserve + 0.25 * volt + 0.25 * equip + 0.15 * alerts + 0.10 * theft
        label = "good" if score >= 80 else "fair" if score >= 60 else "poor"
        return {"score": round(score, 1), "label": label,
                "components": {"reserve_margin": round(reserve, 1), "voltage_security": round(volt, 1), "equipment_health": round(equip, 1),
                               "alert_load": round(alerts, 1), "revenue_protection": round(theft, 1)},
                "weights": {"reserve_margin": .25, "voltage_security": .25, "equipment_health": .25, "alert_load": .15, "revenue_protection": .10}}

    def recommendations(self) -> list[str]:
        recs = []
        top = self.pm.current.iloc[0]
        recs.append(f"Inspect transformer {top.transformer_id}: {top.p_fail_30d*100:.0f}% failure risk in 30 days ({top.expected_days_to_failure:.0f} days expected).")
        mv = self.stab.load_balance(self._stress_inputs(24)[0])
        if mv:
            m = mv[0]; recs.append(f"Transfer {m['transfer_mva']} MVA from {m['from']} to {m['to']} to relieve loading.")
        reg = self.theft.region_summary().iloc[0]
        recs.append(f"Prioritise revenue-protection inspections in region {reg.region_id} ({reg.suspect_rate_pct}% of meters flagged).")
        drops = self.ren.detect_drops(24)
        if len(drops):
            recs.append(f"Pre-charge batteries before {pd.Timestamp(drops.ts.iloc[0]):%H:%M}: renewables ramp down {abs(drops.delta_mw.iloc[0]):.0f} MW.")
        recs.append("Schedule EV charging into the overnight valley to flatten tomorrow's peak (see /bonus/ev-charging).")
        return recs

    def dashboard(self) -> dict:
        pk = self.demand.peak(24)
        nat = pd.DataFrame(self.demand_forecast(48)["national"])
        ren = self.ren.national(48)
        cap = float(self.d.renewables.capacity_mw.sum())
        merged = nat.merge(ren, left_on=pd.to_datetime(nat.ts), right_on="ts", how="left", suffixes=("", "_r"))
        ru = float(ren.renewable_mw.mean() / cap * 100)
        self.scan_alerts()
        plan = self.dispatch(24)
        return clean({"generated_at": self.now, "grid_health": self.health_score(), "demand_peak_24h": pk,
                      "demand_forecast": rnd(nat, 0), "renewable_forecast": rnd(ren, 0),
                      "renewable_utilization_pct": round(ru, 1), "renewable_share_pct": plan["summary"]["renewable_share_pct"],
                      "equipment_health": self.pm.current.risk_tier.value_counts().to_dict(),
                      "outage_risk": {"transformers_p7d_gt_10pct": int((self.pm.current.p_fail_7d > 0.10).sum()),
                                      "expected_failures_7d": round(float(self.pm.current.p_fail_7d.sum()), 1)},
                      "theft_heatmap": self.theft.heatmap().head(40), "theft_regions": self.theft.region_summary(),
                      "alerts": self.alerts.list(limit=25), "alert_counts": self.alerts.counts(),
                      "dispatch_summary": plan["summary"], "recommendations": self.recommendations(),
                      "voltage_summary": self.voltage()["summary"], "stream": self.stream.metrics()})

    # ------------------------------------------------------------------ streaming
    def ingest(self, events: list[dict]) -> dict:
        self.bus.publish(Settings.kafka_topic, events)
        return clean(self.stream.process_dicts(self.bus.consume(100000)))

    def simulate_stream(self, n_events=100000, anomaly_rate=0.002, batch=50000) -> dict:
        done = 0; alerts = 0; t = time.perf_counter()
        while done < n_events:
            r = self.stream.process_batch(self.gen.batch(min(batch, n_events - done), anomaly_rate)); done += r["events"]; alerts += r["alerts"]
        el = time.perf_counter() - t
        return clean({"events": done, "seconds": el, "events_per_sec": done / el, "alerts": alerts, "metrics": self.stream.metrics()})

    # ------------------------------------------------------------------ bonus
    def self_heal(self, asset_id: str) -> dict:
        from . import bonus
        return bonus.self_healing(self, asset_id)

    def copilot(self):
        if not hasattr(self, "_copilot"):
            from .copilot import Copilot
            self._copilot = Copilot(self)
        return self._copilot

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        d = self.d
        return clean({"version": __version__, "ml_backend": backend_name(), "scale": {"plants": len(d.plants), "substations": len(d.substations),
                      "transformers": len(d.transformers), "meters": len(d.meters), "renewable_assets": len(d.renewables),
                      "history_days": self.cfg.history_days}, "bus": self.bus.name, "cache": self.cache.backend,
                      "db": "postgres" if self.store.pg else "sqlite", "auth_dev_keys": self.auth.dev_defaults, "init_seconds": self.timings,
                      "simulated_now": self.now})
