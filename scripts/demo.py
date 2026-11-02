"""End-to-end demo without any server:  python -m scripts.demo
Prints the key outputs of every module (and the evaluation metrics used in the report)."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("DATABASE_URL", "sqlite://")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartgrid import bonus                      # noqa: E402
from smartgrid.platform import GridPlatform      # noqa: E402


def hr(t): print("\n" + "=" * 8, t, "=" * 8)


def main():
    p = GridPlatform().initialize()
    hr("SCALE / BACKENDS"); print(json.dumps({k: p.info()[k] for k in ("scale", "ml_backend", "bus", "cache", "db")}, indent=1))
    hr("DEMAND FORECAST"); print(p.demand.metrics); print(p.forecast_day(1)["peak_mw"], "MW peak tomorrow")
    hr("RENEWABLES"); print(p.ren.metrics)
    hr("LP DISPATCH (24h)"); print(p.dispatch(24)["summary"])
    hr("RL BATTERY"); print(p.battery_rl()["training_report"])
    hr("PREDICTIVE MAINTENANCE"); print(p.pm.metrics); print(p.pm.evaluate_current())
    hr("THEFT DETECTION"); print(p.theft.metrics)
    hr("GRAPH"); print(p.kg.stats())
    hr("STREAM THROUGHPUT"); r = p.simulate_stream(1_000_000); print({k: r[k] for k in ("events", "events_per_sec")})
    hr("COPILOT")
    for q in ["Which transformer is likely to fail next week?", "Forecast tomorrow's electricity demand.", "Show regions with energy theft.",
              "Recommend optimal load balancing.", "Simulate a major power plant outage.", "Explain why voltage instability occurred."]:
        print(f"\n> {q}\n{p.copilot().ask(q)['answer']}")
    hr("BONUS"); print("EV peak reduction %:", round(bonus.ev_charging(p)["peak_reduction_pct"], 1),
                       "| carbon @ $80/t reduction %:", bonus.carbon_forecast(p, 48, 80)["carbon_price_scenario"]["reduction_pct"])
    hr("AUDIT CHAIN"); print(p.audit.verify())


if __name__ == "__main__":
    main()
