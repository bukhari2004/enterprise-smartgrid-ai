"""Run:  python -m unittest discover -s tests -v      (uses the small 'tiny' grid, ~10 s)"""
import os
import unittest

os.environ["SMARTGRID_SCALE"] = "tiny"
os.environ["DATABASE_URL"] = "sqlite://"
os.environ["APP_ENV"] = "dev"

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

from smartgrid import bonus                                # noqa: E402
from smartgrid.alerts import Alert, AlertCenter            # noqa: E402
from smartgrid.audit import AuditLog                       # noqa: E402
from smartgrid.optimization import BatteryRL, battery_lp_profit   # noqa: E402
from smartgrid.platform import GridPlatform                # noqa: E402
from smartgrid.security import Auth                        # noqa: E402
from smartgrid.storage import Store                        # noqa: E402
from smartgrid.streaming import METER, TR_TEMP, StreamProcessor   # noqa: E402

P = None


def setUpModule():
    global P
    P = GridPlatform(db_url="sqlite://", verbose=False).initialize()


class TestSimulatorAndForecast(unittest.TestCase):
    def test_shapes_and_truth(self):
        d = P.d
        self.assertEqual(len(d.meters), len(d.transformers) * d.cfg.meters_per_transformer)
        self.assertGreater(d.truth_theft.is_theft.sum(), 0)
        self.assertFalse(d.demand.demand_mw.isna().any())

    def test_demand_model_beats_naive(self):
        m = P.demand.metrics
        self.assertLess(m["day_ahead_mape_pct"], 3.0)
        self.assertLess(m["day_ahead_mape_pct"], m["seasonal_naive_mape_pct"])

    def test_forecast_scenarios(self):
        base = P.demand.peak(24)["peak_mw"]
        hot = P.demand.peak(24, temp_delta=8)["peak_mw"]
        surge = P.demand.peak(24, surge_pct=0.2)["peak_mw"]
        self.assertGreater(hot, base * 0.999)
        self.assertAlmostEqual(surge / base, 1.2, delta=0.03)

    def test_renewable_model(self):
        self.assertGreater(P.ren.metrics["solar"]["r2"], 0.9)
        self.assertGreater(P.ren.metrics["wind"]["r2"], 0.9)


class TestOptimization(unittest.TestCase):
    def test_power_balance(self):
        ts, dem, ren = P._inputs(24)
        r = P.opt.solve(dem, ren, ts)
        s = r["schedule"]; gen = s[list(P.d.plants.plant_id)].sum(axis=1)
        lhs = gen + s.renewable_used_mw + s.battery_discharge_mw - s.battery_charge_mw + s.unserved_mw
        np.testing.assert_allclose(lhs, s.demand_mw, atol=1e-3 * dem.max())     # dump is 0 in normal conditions
        self.assertLessEqual(s.battery_soc_mwh.max(), P.opt.batt_cap + 1e-6)

    def test_outage_costs_more_and_extreme_sheds_load(self):
        ts, dem, ren = P._inputs(24)
        base = P.opt.solve(dem, ren, ts)["summary"]
        out = P.opt.solve(dem, ren, ts, outage_plants=["P-003"])["summary"]
        self.assertGreaterEqual(out["fuel_cost_usd"], base["fuel_cost_usd"] - 1)
        ex = P.opt.solve(dem, ren, ts, outage_plants=list(P.d.plants.plant_id[2:]))["summary"]
        self.assertGreater(ex["unserved_mwh"], 0)

    def test_carbon_price_reduces_emissions(self):
        ts, dem, ren = P._inputs(24)
        e0 = P.opt.solve(dem, ren, ts)["summary"]["emissions_tco2"]
        e1 = P.opt.solve(dem, ren, ts, carbon_price=100)["summary"]["emissions_tco2"]
        self.assertLessEqual(e1, e0)

    def test_rl_never_beats_perfect_foresight(self):
        rng = np.random.default_rng(0)
        days = 60 + 40 * np.sin(np.linspace(0, 2 * np.pi, 24))[None, :] + rng.normal(0, 3, (80, 24))
        rl = BatteryRL(400, 100, 0.92, seed=1).fit(days[:60], episodes=6000)
        rep = rl.evaluate(days[60:])
        self.assertLessEqual(rep["rl"], rep["lp_optimal"] + 1)
        self.assertGreater(rep["rl"], 0)
        self.assertGreater(battery_lp_profit(days[0], 400, 100, 0.92, 200), 0)


class TestMaintenanceTheft(unittest.TestCase):
    def test_failure_model(self):
        self.assertGreater(P.pm.metrics["roc_auc"], 0.7)
        top = P.pm.current.transformer_id.iloc[0]
        ex = P.pm.explain(top)
        self.assertTrue(ex["drivers"])
        self.assertTrue(all(dr["risk_contribution_pp"] > 0 for dr in ex["drivers"]))

    def test_theft_detector(self):
        self.assertGreater(P.theft.metrics["holdout_roc_auc_blend"], 0.85)
        s = P.theft.suspects(top=100000, min_score=0.93)
        truth = P.d.truth_theft.set_index("meter_id").loc[s.meter_id]
        self.assertGreater(truth.is_theft.mean(), 0.5)

    def test_theft_explain(self):
        mid = P.theft.suspects(top=1).meter_id.iloc[0]
        self.assertIn("evidence", P.theft.explain(mid))


class TestGraphStability(unittest.TestCase):
    def test_impact_logic(self):
        g = P.kg
        sub = P.d.substations.substation_id.iloc[0]
        self.assertGreater(g.impact(sub)["customers_affected"], 0)
        every_plant = set(P.d.plants.plant_id)
        self.assertEqual(set(g.islanded_substations(every_plant)), set(P.d.substations.substation_id))
        self.assertEqual(g.islanded_substations(set()), [])

    def test_risk_propagation(self):
        t0 = P.d.transformers.transformer_id.iloc[0]
        r = P.kg.propagate_risk({t0: 1.0}).set_index("transformer_id")
        self.assertLess(r.graph_risk[t0], 1.0)
        sib = P.d.transformers[(P.d.transformers.substation_id == P.d.transformers.substation_id.iloc[0])].transformer_id.iloc[1]
        self.assertGreater(r.graph_risk[sib], 0)

    def test_frequency_monotonic(self):
        a = P.stab.frequency_response(300, 9000)["nadir_hz"]
        b = P.stab.frequency_response(1200, 9000)["nadir_hz"]
        self.assertGreater(a, b)

    def test_voltage_explanation_sums(self):
        v = P.voltage()
        sid = v["substations"][0]["substation_id"]
        ex = P.explain_voltage(sid)
        total = 1.03 + sum(ex["contributions_pu"].values())
        self.assertAlmostEqual(total, ex["voltage_pu"], places=3)


class TestStreamingAlertsAudit(unittest.TestCase):
    def test_outage_and_hot_transformer_alerts(self):
        ac = AlertCenter(); sp = StreamProcessor(P.d, ac)
        m = np.flatnonzero(sp.meter_tr == 0)[:6]
        b = {"etype": np.r_[np.full(len(m), METER), [TR_TEMP]], "asset": np.r_[m, [0]], "value": np.r_[np.zeros(len(m)), [110.0]], "ts": np.zeros(len(m) + 1)}
        sp.process_batch(b)
        kinds = {a.type for a in ac.alerts}
        self.assertIn("POWER_OUTAGE", kinds); self.assertIn("TRANSFORMER_FAILURE_RISK", kinds)
        n = len(ac.alerts); sp.process_batch(b)
        self.assertEqual(len(ac.alerts), n, "duplicates must be suppressed during cooldown")

    def test_alert_escalation_bypasses_cooldown(self):
        ac = AlertCenter()
        self.assertIsNotNone(ac.emit(Alert("GRID_OVERLOAD", "warning", "S-1", "x")))
        self.assertIsNone(ac.emit(Alert("GRID_OVERLOAD", "warning", "S-1", "x")))
        self.assertIsNotNone(ac.emit(Alert("GRID_OVERLOAD", "critical", "S-1", "x")))

    def test_audit_chain_detects_tampering(self):
        s = Store("sqlite://"); a = AuditLog(s)
        for i in range(5):
            a.record("u", "act", {"i": i})
        self.assertTrue(a.verify()["valid"])
        s.execute("UPDATE audit SET detail='{\"i\": 99}' WHERE detail='{\"i\": 2}'")
        self.assertFalse(a.verify()["valid"])

    def test_rbac(self):
        au = Auth("k1:viewer,k2:admin", "prod")
        self.assertTrue(au.allowed("k2", "operator")); self.assertFalse(au.allowed("k1", "operator")); self.assertFalse(au.allowed("zzz", "viewer"))
        self.assertEqual(Auth("", "prod").keys, {})                 # no default keys outside dev


class TestCopilotBonus(unittest.TestCase):
    def test_required_questions_route_correctly(self):
        c = P.copilot()
        expected = {"Which transformer is likely to fail next week?": "transformer_failure", "Forecast tomorrow's electricity demand.": "demand_forecast",
                    "Show regions with energy theft.": "energy_theft", "Recommend optimal load balancing.": "load_balance",
                    "Simulate a major power plant outage.": "plant_outage", "Explain why voltage instability occurred.": "voltage_explain"}
        for q, intent in expected.items():
            r = c.ask(q)
            self.assertEqual(r["intent"], intent, q)
            self.assertTrue(r["answer"])
        self.assertEqual(c.ask("what is the best pizza recipe")["intent"], "unknown")

    def test_bonus(self):
        ev = bonus.ev_charging(P)
        self.assertGreaterEqual(ev["peak_reduction_mw"], 0)
        self.assertLess(bonus.carbon_forecast(P, 24, 100)["carbon_price_scenario"]["tco2"], bonus.carbon_forecast(P, 24)["total_tco2"] + 1)
        self.assertIn("switching_plan", P.self_heal(P.d.substations.substation_id.iloc[0]))

    def test_dashboard_payload(self):
        d = P.dashboard()
        for k in ("grid_health", "demand_forecast", "alerts", "recommendations", "theft_heatmap"):
            self.assertIn(k, d)
        self.assertTrue(0 <= d["grid_health"]["score"] <= 100)


try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api import main
        main.platform = P
        cls.c = TestClient(main.app)

    def test_auth(self):
        self.assertEqual(self.c.get("/api/v1/info").status_code, 401)
        self.assertEqual(self.c.get("/api/v1/info", headers={"X-API-Key": "demo-viewer"}).status_code, 200)
        self.assertEqual(self.c.get("/api/v1/audit/verify", headers={"X-API-Key": "demo-operator"}).status_code, 403)

    def test_endpoints(self):
        h = {"X-API-Key": "demo-operator"}
        for path in ("/api/v1/forecast/demand?horizon=24", "/api/v1/maintenance/risk", "/api/v1/theft/overview", "/api/v1/graph/stats", "/api/v1/dashboard"):
            self.assertEqual(self.c.get(path, headers=h).status_code, 200, path)
        r = self.c.post("/api/v1/copilot/ask", json={"question": "Show regions with energy theft."}, headers=h)
        self.assertEqual(r.json()["intent"], "energy_theft")
        self.assertEqual(self.c.get("/api/v1/graph/nodes/NOPE", headers=h).status_code, 404)


if __name__ == "__main__":
    unittest.main()
