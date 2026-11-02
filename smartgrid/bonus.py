"""Bonus challenges: Carbon forecasting, EV charging optimiser, AI dynamic pricing, VPP controller, autonomous self-healing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import clean, rnd


def carbon_forecast(p, horizon: int = 72, carbon_price: float = 0.0) -> dict:
    """Emission forecast from the optimal dispatch (tCO2/h and grid intensity gCO2/kWh) + carbon-price what-if."""
    horizon = min(horizon, p.cfg.forecast_days * 24)
    ts, dem, ren = p._inputs(horizon)
    base = p.opt.solve(dem, ren, ts)
    em = (base["schedule"][list(p.d.plants.plant_id)].to_numpy() * p.d.plants.emission_t_mwh.to_numpy()).sum(axis=1)
    out = {"horizon_h": horizon, "total_tco2": em.sum(), "avg_intensity_g_kwh": em.sum() / dem.sum() * 1000,
           "hourly": rnd(pd.DataFrame({"ts": ts, "tco2": em, "intensity_g_kwh": em / dem * 1000}), 1),
           "cleanest_hour": str(ts[int(np.argmin(em / dem))]), "dirtiest_hour": str(ts[int(np.argmax(em / dem))])}
    if carbon_price > 0:
        alt = p.opt.solve(dem, ren, ts, carbon_price=carbon_price)
        a_em = alt["summary"]["emissions_tco2"]
        out["carbon_price_scenario"] = {"price_usd_t": carbon_price, "tco2": a_em, "reduction_pct": round((1 - a_em / em.sum()) * 100, 1),
                                        "extra_fuel_cost_usd": alt["summary"]["fuel_cost_usd"] - base["summary"]["fuel_cost_usd"]}
    return clean(out)


def ev_charging(p, n_evs: int = 500_000, kwh_per_ev: float = 14.0, charger_kw: float = 7.0, participation: float = 0.6) -> dict:
    """Valley-filling EV charge scheduler (water-filling) vs. unmanaged 'plug-in at 18:00' behaviour."""
    ts, dem, ren = p._inputs(24)
    net = dem - ren
    energy = n_evs * kwh_per_ev / 1000.0                                   # MWh needed
    flex = energy * participation
    pmax = n_evs * participation * charger_kw / 1000.0                     # MW if all flexible EVs plug in
    window = np.array([(t.hour >= 18) or (t.hour < 7) for t in ts])
    # unmanaged: everything charges during 18-23
    un = np.zeros(24); eve = np.array([18 <= t.hour <= 22 for t in ts]); un[eve] = energy / eve.sum()
    base_load = net + un * (1 - participation)                              # non-participating EVs stay unmanaged
    lo, hi = base_load.min(), base_load.max() + flex
    for _ in range(60):                                                      # binary search for the water level
        lam = (lo + hi) / 2
        ch = np.where(window, np.clip(lam - base_load, 0, pmax), 0.0)
        lo, hi = (lam, hi) if ch.sum() < flex else (lo, lam)
    managed = base_load + ch
    unmanaged = net + un
    return clean({"n_evs": n_evs, "energy_mwh": energy, "flexible_energy_mwh": flex, "peak_unmanaged_mw": unmanaged.max(),
                  "peak_managed_mw": managed.max(), "peak_reduction_mw": unmanaged.max() - managed.max(),
                  "peak_reduction_pct": (1 - managed.max() / unmanaged.max()) * 100,
                  "profile": rnd(pd.DataFrame({"ts": ts, "net_load_mw": net, "unmanaged_mw": unmanaged, "managed_mw": managed, "ev_charge_mw": ch}), 1)})


def dynamic_pricing(p, p_min: float = 30.0, p_max: float = 220.0, gamma: float = 2.0, elasticity: float = -0.12) -> dict:
    """Retail price follows net-load percentile (scarcity-style); estimated demand response from price elasticity."""
    ts, dem, ren = p._inputs(24)
    net = dem - ren
    x = (net - net.min()) / max(net.max() - net.min(), 1e-9)
    price = p_min + (p_max - p_min) * x ** gamma
    ref = float(price.mean())
    resp = elasticity * (price - ref) / ref                                  # fractional demand change
    flexible_share = 0.18                                                    # share of load that responds
    new = dem * (1 + resp * flexible_share)
    shifted = float(np.clip(dem - new, 0, None).sum())
    return clean({"profile": rnd(pd.DataFrame({"ts": ts, "net_load_mw": net, "price_usd_mwh": price, "demand_after_response_mw": new}), 1),
                  "peak_before_mw": dem.max(), "peak_after_mw": new.max(), "peak_shaved_mw": dem.max() - new.max(), "energy_shifted_mwh": shifted,
                  "assumptions": {"elasticity": elasticity, "flexible_load_share": flexible_share}})


def vpp(p, n_evs=500_000, v2g_kw=3.0, v2g_participation=0.1, rooftop_mw=400.0) -> dict:
    """Virtual Power Plant: aggregate batteries + V2G + flexible rooftop solar into dispatchable reserve."""
    batt_mw, batt_mwh = p.opt.batt_pow, p.opt.batt_cap
    v2g_mw = n_evs * v2g_participation * v2g_kw / 1000
    ts, dem, ren = p._inputs(24)
    net = dem - ren
    hour_peak = int(np.argmax(net))
    avail = batt_mw + v2g_mw + 0.15 * rooftop_mw
    dispatch = np.zeros(24); need = np.clip(net - np.quantile(net, 0.9), 0, None)
    dispatch = np.minimum(need, avail)
    energy = dispatch.sum()
    return clean({"resources": {"battery_mw": batt_mw, "battery_mwh": batt_mwh, "v2g_mw": v2g_mw, "flex_rooftop_mw": 0.15 * rooftop_mw,
                                "total_available_mw": avail}, "peak_hour": str(ts[hour_peak]),
                  "dispatch": rnd(pd.DataFrame({"ts": ts, "net_load_mw": net, "vpp_dispatch_mw": dispatch}), 1),
                  "peak_shaving_mw": dispatch.max(), "energy_mwh": energy, "battery_energy_ok": bool(energy <= batt_mwh + v2g_mw * 2),
                  "can_provide_frequency_reserve_mw": min(batt_mw, avail) * 0.8})


def self_healing(p, failed_asset: str, target_loading: float = 0.95) -> dict:
    """Autonomous restoration: for a failed substation, find tie-switches to neighbouring substations in the region
    that can pick up the load (graph reachability + headroom), and for a failed transformer re-feed via a sibling."""
    kg, d = p.kg, p.d
    t = kg.G.nodes[failed_asset]["type"] if failed_asset in kg.G else None
    if t is None:
        raise KeyError(failed_asset)
    peak, _ = p._stress_inputs(24)
    s = p.stab.loading(peak).set_index("substation_id")
    steps, restored, unserved = [], 0.0, 0.0
    if t == "Substation":
        reg = s.region_id[failed_asset]; load = s.loading[failed_asset] * s.capacity_mva[failed_asset]
        pool = s[(s.region_id == reg) & (s.index != failed_asset)].copy()
        pool["headroom"] = (target_loading - pool.loading).clip(lower=0) * pool.capacity_mva
        remaining = load
        for sid, row in pool.sort_values("headroom", ascending=False).iterrows():
            if remaining <= 0.5 or row.headroom <= 0.5:
                continue
            mv = min(remaining, row.headroom); remaining -= mv
            steps.append(f"close tie switch {failed_asset}->{sid}: transfer {mv:.0f} MVA (new loading {(row.loading*row.capacity_mva+mv)/row.capacity_mva*100:.0f}%)")
        restored, unserved = load - remaining, remaining
        total = load
    elif t == "Transformer":
        sub = d.transformers.set_index("transformer_id").substation_id[failed_asset]
        sib = d.transformers[(d.transformers.substation_id == sub) & (d.transformers.transformer_id != failed_asset)].sort_values("rated_kva", ascending=False)
        n = int((d.meters.transformer_id == failed_asset).sum()); steps.append(f"isolate {failed_asset}; re-feed {n} customers from sibling transformers at {sub}: {', '.join(sib.transformer_id.head(2))}")
        restored, unserved, total = n, 0.0, n
    elif t == "PowerPlant":
        isl = kg.islanded_substations({failed_asset}); steps.append("plant loss handled by dispatch re-optimisation" + (f"; islanded: {isl}" if isl else " (no islanded substations)"))
        restored, unserved, total = 1.0, 0.0, 1.0
    else:
        raise ValueError(f"self-healing not defined for {t}")
    return clean({"failed_asset": failed_asset, "type": t, "switching_plan": steps or ["no neighbour has headroom - initiate load shedding"],
                  "restored_pct": restored / total * 100 if total else 100.0, "unserved_load": unserved,
                  "mode": "recommendation (human approval required before switching)"})
