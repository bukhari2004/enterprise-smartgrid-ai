"""AI Grid Copilot.

Pipeline:  question -> intent router (TF-IDF cosine over exemplar questions + keyword boosts)
                    -> entity extraction (asset ids, regions, horizon)
                    -> grounded tool call on the platform (forecast / maintenance / theft / optimisation / simulation / graph)
                    -> answer composer.

The answer text is always built from tool output (no hallucinated numbers). If ANTHROPIC_API_KEY is set, an LLM rewrites
the grounded facts into a concise operator briefing; the structured `data` returned to the caller is unchanged.
A LangGraph version of the same pipeline is provided in `build_langgraph` (optional dependency).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import bonus
from .config import Settings
from .utils import clean

INTENTS = {
    "transformer_failure": ["which transformer is likely to fail next week", "which transformers will fail soon", "predict transformer failures",
                            "equipment at risk of failure", "what needs maintenance first", "show maintenance priorities"],
    "demand_forecast": ["forecast tomorrow's electricity demand", "what will demand be tomorrow", "predict peak load today",
                        "electricity demand forecast", "what is the expected peak load", "how much power will we need"],
    "energy_theft": ["show regions with energy theft", "where is electricity being stolen", "which areas have the most theft",
                     "list suspicious meters", "meter tampering hotspots", "energy fraud overview"],
    "load_balance": ["recommend optimal load balancing", "how should we balance the load", "optimise power distribution",
                     "which substations are overloaded and what to do", "best dispatch plan for tomorrow"],
    "plant_outage": ["simulate a major power plant outage", "what if a power plant trips", "simulate loss of a generator",
                     "power plant failure scenario", "what happens if plant P-003 goes down"],
    "voltage_explain": ["explain why voltage instability occurred", "why is voltage low", "voltage problem root cause",
                        "what is causing the voltage drop", "explain voltage sag"],
    "renewables": ["how much solar and wind will we have", "renewable energy forecast", "will wind output drop", "renewable generation tomorrow"],
    "carbon": ["what are the carbon emissions", "co2 forecast", "how green is the grid tomorrow", "emission intensity"],
    "battery": ["how should we charge and discharge the batteries", "battery storage schedule", "battery arbitrage"],
    "grid_health": ["how healthy is the grid", "give me a grid status summary", "overall grid health score", "status report"],
    "alerts": ["show active alerts", "what alarms are open", "any critical alerts"],
    "impact": ["what happens if this substation fails", "who loses power if transformer fails", "impact of asset failure", "customers affected by outage"],
    "asset_info": ["tell me about this asset", "show details for transformer", "explain this meter", "describe substation"],
    "frequency": ["what happens to frequency if we lose generation", "frequency stability", "under frequency load shedding"],
}
RUNBOOKS = {
    "transformer_overheat": "Transformer overheating: verify cooling fans and oil level, reduce loading via switching, take DGA sample, plan outage if oil temp stays above 95C.",
    "voltage_sag": "Voltage sag: check tap changer position, switch in capacitor banks, transfer load to neighbouring substation, consider battery reactive support.",
    "frequency_event": "Frequency event: activate spinning reserve and fast frequency response, confirm UFLS settings, restore load in stages once frequency exceeds 49.8 Hz.",
    "theft_inspection": "Theft inspection: prioritise by score and estimated loss, verify seals and CT wiring on site, record evidence, follow regulatory notice process.",
    "plant_trip": "Plant trip: confirm lost MW, start fast units, request interchange, activate demand response, update operating reserve forecast.",
}


class Copilot:
    def __init__(self, platform):
        self.p = platform
        self.labels, texts = [], []
        for k, v in INTENTS.items():
            for q in v:
                self.labels.append(k); texts.append(q)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(texts + list(RUNBOOKS.values()))
        self.M = self.vec.transform(texts)
        self.rb_keys = list(RUNBOOKS); self.RB = self.vec.transform(RUNBOOKS.values())
        self.keywords = {"transformer_failure": ["transformer", "fail", "maintenance"], "demand_forecast": ["demand", "load", "forecast", "peak"],
                         "energy_theft": ["theft", "steal", "stolen", "tamper", "fraud"], "load_balance": ["balance", "balancing", "dispatch", "overload"],
                         "plant_outage": ["outage", "trip", "plant"], "voltage_explain": ["voltage", "sag"], "renewables": ["solar", "wind", "renewable"],
                         "carbon": ["carbon", "co2", "emission"], "battery": ["battery", "batteries", "storage"], "grid_health": ["health", "status", "summary"],
                         "alerts": ["alert", "alarm"], "impact": ["impact", "affected", "loses", "lose power"], "frequency": ["frequency", "hz"]}

    # ---------------------------------------------------------------- routing
    def route(self, q: str) -> tuple[str, float]:
        ql = q.lower()
        sims = cosine_similarity(self.vec.transform([q]), self.M)[0]
        score = {k: 0.0 for k in INTENTS}
        for lab, s in zip(self.labels, sims):
            score[lab] = max(score[lab], float(s))
        for k, words in self.keywords.items():
            score[k] += 0.12 * sum(w in ql for w in words)
        if re.search(r"\b(?:T-\d{5}|M-\d{7})\b", q) and not re.search(r"fail|risk", ql):
            score["asset_info"] += 0.4
        if re.search(r"\b(?:S-\d{3}|T-\d{5}|P-\d{3})\b", q) and re.search(r"fail|goes down|trip|lose|outage|what if", ql):
            score["impact"] += 0.35
        best = max(score, key=score.get)
        hits = sum(w in ql for w in self.keywords.get(best, []))
        sim = max((float(x) for lab, x in zip(self.labels, sims) if lab == best), default=0.0)
        self._last = (hits, sim)
        return best, round(score[best], 3)

    @staticmethod
    def entities(q: str) -> dict:
        e = {}
        for key, pat in (("plant", r"\bP-\d{3}\b"), ("substation", r"\bS-\d{3}\b"), ("transformer", r"\bT-\d{5}\b"), ("meter", r"\bM-\d{7}\b"),
                         ("region", r"\bR\d\b")):
            m = re.search(pat, q, re.I)
            if m: e[key] = m.group(0).upper()
        ql = q.lower()
        e["when"] = "tomorrow" if "tomorrow" in ql else "today" if "today" in ql else "next24"
        m = re.search(r"(\d+)\s*(day|week|hour)", ql)
        if m:
            n = int(m.group(1)); e["horizon_days"] = n * (7 if m.group(2) == "week" else 1 if m.group(2) == "day" else 0) or 1
        elif "week" in ql:
            e["horizon_days"] = 7
        return e

    # ---------------------------------------------------------------- tools
    def _tool(self, intent: str, q: str, e: dict) -> dict:
        p = self.p
        if intent == "transformer_failure":
            days = e.get("horizon_days", 7)
            m = p.maintenance(top=5, horizon_days=days); top = m["top"]
            ex = p.pm.explain(top[0]["transformer_id"])
            lines = [f"{r['transformer_id']} ({r['region_id']}): {r['p_fail_horizon']*100:.1f}% chance of failure in {days} days, "
                     f"{r['customers']} customers, tier {r['risk_tier']}" for r in top]
            return {"headline": f"Top transformer risks over the next {days} days:", "bullets": lines,
                    "why": ex["summary"], "actions": [d["action"] for d in ex["drivers"][:3]], "data": m}
        if intent == "demand_forecast":
            off = 1 if e["when"] == "tomorrow" else 0
            f = p.forecast_day(off); pk = p.demand.peak(24)
            return {"headline": f"Demand forecast for {f['date']}: peak {f['peak_mw']:.0f} MW at {pd.Timestamp(f['peak_time']):%H:%M}, "
                                f"average {f['avg_mw']:.0f} MW, daily energy {f['energy_gwh']:.0f} GWh.",
                    "bullets": [f"80% interval at peak: {f['p10_at_peak']:.0f}-{f['p90_at_peak']:.0f} MW",
                                f"model day-ahead error in back-test: {p.demand.metrics['day_ahead_mape_pct']}% MAPE "
                                f"(seasonal-naive: {p.demand.metrics['seasonal_naive_mape_pct']}%)"],
                    "why": "; ".join(p.demand.explain_forecast()["drivers"][:2]), "data": f}
        if intent == "energy_theft":
            o = p.theft_overview(); r = o["regions"][:3]
            return {"headline": "Regions with the highest share of suspected energy theft:",
                    "bullets": [f"{x['region_id']}: {x['suspect_rate_pct']}% of meters flagged, ~{x['loss_kwh_day']:.0f} kWh/day unbilled" for x in r],
                    "why": f"Isolation-Forest + supervised blend, holdout ROC-AUC {o['model']['holdout_roc_auc_blend']}",
                    "actions": ["inspect highest-score meters first (GET /theft/suspects)"], "data": o}
        if intent == "load_balance":
            lb = p.load_balance(); s = lb["generation_plan_summary"]
            mv = lb["substation_transfers"][:4]
            return {"headline": f"Optimal 24h dispatch costs ${s['fuel_cost_usd']:,.0f} with {s['renewable_share_pct']}% renewables and "
                                f"{s['unserved_mwh']:.0f} MWh unserved.",
                    "bullets": [f"move {m['transfer_mva']} MVA from {m['from']} to {m['to']}" for m in mv] or ["no substation exceeds 80% loading in the forecast - no transfers needed"],
                    "why": "multi-period LP minimises fuel cost subject to ramp limits, must-run units and battery constraints", "data": lb}
        if intent == "plant_outage":
            o = p.simulate_plant_outage(e.get("plant"))
            return {"headline": f"Losing {o['plant']} ({o['plant_type']}, {o['scheduled_output_lost_mw']:.0f} MW scheduled) adds "
                                f"${o['extra_fuel_cost_usd']:,.0f} fuel cost over {o['horizon_h']}h (emissions change {o['extra_emissions_tco2']:+,.0f} tCO2).",
                    "bullets": [f"frequency nadir {o['frequency_event']['nadir_hz']} Hz ({o['frequency_event']['status']}), UFLS {o['frequency_event']['load_shed_mw']:.0f} MW",
                                f"unserved energy after re-dispatch: {o['scenario']['unserved_mwh']:.0f} MWh",
                                f"customers islanded by topology: {o['topology_impact']['customers_affected']}"],
                    "actions": o["recommended_actions"], "data": o}
        if intent == "voltage_explain":
            ex = p.explain_voltage(e.get("substation"))
            note = ""
            if ex["status"] == "normal":
                ex2 = p.explain_voltage(e.get("substation"), temp_delta=8.0, surge_pct=0.45)
                note = (f"No instability in the current forecast (lowest voltage {ex['voltage_pu']} pu). Explained instead for a stress case "
                        f"(+8°C heat-wave and a 45% emergency demand surge): {ex2['substation_id']} falls to {ex2['voltage_pu']} pu.")
                ex = ex2
            return {"headline": ex["explanation"], "bullets": [f"{k}: {v:+.3f} pu" for k, v in ex["contributions_pu"].items()],
                    "why": ex["counterfactual"] + (". " + note if note else ""), "actions": ex["recommended_actions"], "data": ex}
        if intent == "renewables":
            r = p.renewable_forecast(24)
            return {"headline": f"Renewables peak at {r['peak_renewable_mw']:.0f} MW (installed {r['installed_mw']:.0f} MW) in the next 24h.",
                    "bullets": [f"expected ramp-down {abs(x['delta_mw']):.0f} MW at {pd.Timestamp(x['ts']):%H:%M}" for x in r["expected_drops"][:3]] or ["no abnormal ramp-downs expected"],
                    "why": f"model R2 solar {r['model']['solar']['r2']}, wind {r['model']['wind']['r2']}", "data": r}
        if intent == "carbon":
            c = bonus.carbon_forecast(p, 48)
            return {"headline": f"Forecast grid carbon intensity {c['avg_intensity_g_kwh']:.0f} gCO2/kWh over 48h ({c['total_tco2']:,.0f} tCO2).",
                    "bullets": [f"cleanest hour {c['cleanest_hour']}", f"dirtiest hour {c['dirtiest_hour']}"], "data": c}
        if intent == "battery":
            b = p.battery_rl(); acts = pd.DataFrame(b["tomorrow_schedule"])
            return {"headline": f"RL battery policy expects ${b['expected_profit_usd']:,.0f} arbitrage profit tomorrow.",
                    "bullets": [f"charge at {', '.join(pd.to_datetime(acts[acts.action=='charge'].ts).dt.strftime('%H:%M')[:6])}",
                                f"discharge at {', '.join(pd.to_datetime(acts[acts.action=='discharge'].ts).dt.strftime('%H:%M')[:6])}"],
                    "why": f"agent earns {b['training_report']['rl_pct_of_optimal']}% of the perfect-foresight optimum on held-out days", "data": b}
        if intent == "grid_health":
            h = p.health_score()
            return {"headline": f"Grid health score {h['score']}/100 ({h['label']}).", "bullets": [f"{k}: {v}" for k, v in h["components"].items()],
                    "actions": p.recommendations()[:3], "data": h}
        if intent == "alerts":
            p.scan_alerts(); al = p.alerts.list(limit=8)
            return {"headline": f"{len(p.alerts.list(limit=10000))} open alerts; most severe:", "bullets": [f"[{a['severity']}] {a['type']} {a['asset_id']}: {a['message']}" for a in al],
                    "data": {"counts": p.alerts.counts(), "top": al}}
        if intent == "impact":
            aid = e.get("substation") or e.get("transformer") or e.get("plant")
            if not aid:
                return {"headline": "Which asset? Mention an id such as S-004, T-00012 or P-003.", "data": {}}
            im = p.kg.impact(aid)
            return {"headline": f"If {aid} fails: {im['customers_affected']} customers lose supply ({im['industrial_customers']} industrial).",
                    "bullets": [f"islanded substations: {', '.join(im['islanded_substations']) or 'none'}", im["note"] or ("no alternate feed available" if im["customers_affected"] else "")],
                    "actions": [s for s in p.self_heal(aid)["switching_plan"]] if hasattr(p, "self_heal") else [], "data": im}
        if intent == "asset_info":
            aid = e.get("transformer") or e.get("meter") or e.get("substation") or e.get("plant")
            if not aid:
                return {"headline": "Which asset? Mention an id such as T-00012, M-0000123, S-004 or P-003.", "data": {}}
            info = p.kg.node(aid); extra = {}
            if aid.startswith("T-"):
                extra = p.pm.explain(aid)
            elif aid.startswith("M-"):
                extra = p.theft.explain(aid)
            return {"headline": extra.get("summary") or f"{aid} is a {info['type']}.", "bullets": extra.get("evidence", []) or [f"{k} {', '.join(map(str, v[:4]))}" for k, v in info["relations"].items()],
                    "data": {"graph": info, "analysis": extra}}
        if intent == "frequency":
            f = p.frequency()
            return {"headline": f"Losing {f['loss_mw']:.0f} MW gives a frequency nadir of {f['nadir_hz']} Hz ({f['status']}).",
                    "bullets": [f"RoCoF {f['rocof_hz_s']} Hz/s", f"load shed {f['load_shed_mw']:.0f} MW"], "actions": [f["recommendation"]], "data": f}
        return {"headline": "I can't answer that yet.", "data": {}}

    # ---------------------------------------------------------------- answer
    def ask(self, question: str, user: str = "anonymous") -> dict:
        q = question.strip()
        if re.search(r"\b(runbook|procedure|how (do|to) (i|we) (handle|respond))\b", q.lower()):
            sims = cosine_similarity(self.vec.transform([q]), self.RB)[0]
            k = self.rb_keys[int(sims.argmax())]
            out = {"intent": "runbook", "confidence": round(float(sims.max()), 3), "answer": RUNBOOKS[k], "data": {"runbook": k}}
            self.p.audit.record(user, "copilot.ask", {"q": q, "intent": "runbook"}); return out
        intent, conf = self.route(q)
        hits, sim = self._last
        has_id = bool(re.search(r"\b(?:P-\d{3}|S-\d{3}|T-\d{5}|M-\d{7})\b", q))
        if conf < 0.18 or (hits == 0 and sim < 0.45 and not has_id):
            out = {"intent": "unknown", "confidence": conf, "answer": "I can forecast demand, rank transformer failure risk, find energy theft, optimise dispatch, "
                   "simulate plant outages, explain voltage instability, forecast renewables/carbon, and look up any asset. Try one of those.",
                   "data": {}, "examples": [v[0] for v in INTENTS.values()][:6]}
            self.p.audit.record(user, "copilot.ask", {"q": q, "intent": "unknown"}); return out
        ents = self.entities(q)
        try:
            r = self._tool(intent, q, ents)
        except KeyError as exc:
            r = {"headline": f"I couldn't find asset {exc}.", "data": {}}
        text = r["headline"]
        for k, title in (("bullets", None), ):
            if r.get("bullets"):
                text += "\n" + "\n".join(f"• {b}" for b in r["bullets"] if b)
        if r.get("why"):
            text += f"\nWhy: {r['why']}"
        if r.get("actions"):
            text += "\nRecommended: " + "; ".join(r["actions"][:3])
        out = {"intent": intent, "confidence": conf, "entities": ents, "answer": text, "llm_used": False, "data": clean(r.get("data", {}))}
        polished = self._llm_polish(q, text)
        if polished:
            out["answer"], out["llm_used"] = polished, True
        self.p.audit.record(user, "copilot.ask", {"q": q, "intent": intent, "confidence": conf})
        return out

    def _llm_polish(self, question: str, facts: str) -> str | None:                    # pragma: no cover - needs API key
        if not Settings.llm_api_key:
            return None
        try:
            import requests
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=20,
                              headers={"x-api-key": Settings.llm_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                              json={"model": Settings.llm_model, "max_tokens": 400,
                                    "system": "You are a grid operations copilot. Rewrite the verified facts as a concise operator briefing. "
                                              "Use ONLY the numbers and assets given; never invent data.",
                                    "messages": [{"role": "user", "content": f"Question: {question}\n\nVerified facts:\n{facts}"}]})
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json()["content"]).strip() or None
        except Exception:
            return None


def build_langgraph(copilot: Copilot):                                                   # pragma: no cover - optional dependency
    """Same pipeline as a LangGraph state machine (route -> tool -> compose). Requires `pip install langgraph`."""
    from typing import TypedDict
    from langgraph.graph import END, StateGraph

    class S(TypedDict, total=False):
        question: str; intent: str; confidence: float; entities: dict; result: dict; answer: str

    def route(s): i, c = copilot.route(s["question"]); return {"intent": i, "confidence": c, "entities": copilot.entities(s["question"])}
    def tool(s): return {"result": copilot._tool(s["intent"], s["question"], s["entities"])}
    def compose(s): return {"answer": s["result"]["headline"]}
    g = StateGraph(S); g.add_node("route", route); g.add_node("tool", tool); g.add_node("compose", compose)
    g.set_entry_point("route"); g.add_edge("route", "tool"); g.add_edge("tool", "compose"); g.add_edge("compose", END)
    return g.compile()
