"""Smart Grid Optimization Engine.

1. `DispatchOptimizer`  - multi-period LP (HiGHS): economic dispatch + load balancing + renewable allocation +
                          battery charge/discharge + ramp limits + carbon price + unserved-energy slack.
                          Dual variables of the balance constraint = system marginal price ($/MWh).
2. `BatteryRL`          - Reinforcement-learning battery arbitrage agent (tabular Q-learning, pure NumPy) that
                          learns from historical days and is benchmarked against rule-based and LP-optimal policies.
                          An optional Stable-Baselines3 PPO trainer is included for when gymnasium + SB3 are installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .simulator import SimData


# ===================================================================== LP dispatch
class DispatchOptimizer:
    def __init__(self, data: SimData):
        self.d = data
        self.plants = data.plants.reset_index(drop=True)
        self.batt_cap = float(data.batteries.capacity_mwh.sum())
        self.batt_pow = float(data.batteries.power_mw.sum())
        self.eff = float(data.batteries.efficiency.mean())

    def solve(self, demand: np.ndarray, ren_avail: np.ndarray, ts: pd.DatetimeIndex | None = None,
              outage_plants: list[str] | None = None, carbon_price: float = 0.0, use_battery: bool = True,
              soc0_frac: float = 0.5, derate: dict[str, float] | None = None) -> dict:
        demand = np.asarray(demand, float); ren_avail = np.asarray(ren_avail, float)
        T, P = len(demand), len(self.plants)
        outage = set(outage_plants or [])
        derate = derate or {}
        # variable layout: gen[P*T] | ren[T] | chg[T] | dis[T] | soc[T] | uns[T] | dump[T]
        g0, r0, c0, d0, s0, u0, x0 = 0, P * T, P * T + T, P * T + 2 * T, P * T + 3 * T, P * T + 4 * T, P * T + 5 * T
        n = P * T + 6 * T
        cost = np.zeros(n)
        lb = np.zeros(n); ub = np.full(n, np.inf)
        for p, pl in self.plants.iterrows():
            avail = 0.0 if pl.plant_id in outage else pl.capacity_mw * derate.get(pl.plant_id, 1.0)
            mr = 0.0 if pl.plant_id in outage else min(pl.must_run_mw, avail)
            for t in range(T):
                cost[g0 + p * T + t] = pl.cost_usd_mwh + carbon_price * pl.emission_t_mwh
                lb[g0 + p * T + t], ub[g0 + p * T + t] = mr, avail
        ub[r0:r0 + T] = ren_avail
        cost[c0:c0 + T] = cost[d0:d0 + T] = 1.0                    # tiny wear cost on battery throughput
        bp = self.batt_pow if use_battery else 0.0
        bc = self.batt_cap if use_battery else 0.0
        ub[c0:c0 + T] = bp; ub[d0:d0 + T] = bp; ub[s0:s0 + T] = bc
        cost[u0:u0 + T] = 10000.0                                    # value of lost load
        cost[x0:x0 + T] = 50.0                                       # over-generation penalty
        ec = ed = np.sqrt(self.eff)
        A_eq = lil_matrix((2 * T, n)); b_eq = np.zeros(2 * T)
        for t in range(T):
            for p in range(P):
                A_eq[t, g0 + p * T + t] = 1
            A_eq[t, r0 + t] = 1; A_eq[t, d0 + t] = 1; A_eq[t, c0 + t] = -1; A_eq[t, u0 + t] = 1; A_eq[t, x0 + t] = -1
            b_eq[t] = demand[t]
            row = T + t                                              # soc[t] - soc[t-1] - ec*chg + dis/ed = 0
            A_eq[row, s0 + t] = 1; A_eq[row, c0 + t] = -ec; A_eq[row, d0 + t] = 1 / ed
            if t > 0:
                A_eq[row, s0 + t - 1] = -1
            else:
                b_eq[row] = soc0_frac * bc
        lb[s0 + T - 1] = soc0_frac * bc                              # end the day with at least the starting charge
        ub[s0 + T - 1] = max(ub[s0 + T - 1], lb[s0 + T - 1])
        rows = []
        A_ub = lil_matrix((2 * P * max(T - 1, 0), n)); b_ub = np.zeros(2 * P * max(T - 1, 0))
        k = 0
        for p, pl in self.plants.iterrows():
            for t in range(1, T):
                A_ub[k, g0 + p * T + t] = 1; A_ub[k, g0 + p * T + t - 1] = -1; b_ub[k] = pl.ramp_mw_h; k += 1
                A_ub[k, g0 + p * T + t] = -1; A_ub[k, g0 + p * T + t - 1] = 1; b_ub[k] = pl.ramp_mw_h; k += 1
        res = linprog(cost, A_ub=A_ub.tocsr() if k else None, b_ub=b_ub if k else None, A_eq=A_eq.tocsr(), b_eq=b_eq,
                      bounds=list(zip(lb, ub)), method="highs")
        if res.status != 0:
            return {"status": "infeasible", "message": res.message}
        x = res.x
        sched = pd.DataFrame({"hour": np.arange(T) if ts is None else ts})
        for p, pl in self.plants.iterrows():
            sched[pl.plant_id] = x[g0 + p * T: g0 + (p + 1) * T]
        sched["demand_mw"] = demand
        sched["renewable_used_mw"] = x[r0:r0 + T]
        sched["renewable_curtailed_mw"] = ren_avail - x[r0:r0 + T]
        sched["battery_charge_mw"] = x[c0:c0 + T]; sched["battery_discharge_mw"] = x[d0:d0 + T]
        sched["battery_soc_mwh"] = x[s0:s0 + T]
        sched["unserved_mw"] = x[u0:u0 + T]
        sched["price_usd_mwh"] = res.eqlin.marginals[:T] if res.eqlin is not None else np.nan
        gen = sched[list(self.plants.plant_id)]
        emis = float((gen.to_numpy() * self.plants.emission_t_mwh.to_numpy()[None, :]).sum())
        fuel = float((gen.to_numpy() * self.plants.cost_usd_mwh.to_numpy()[None, :]).sum())
        thermal_mwh = float(gen.to_numpy().sum())
        summary = {"status": "optimal", "fuel_cost_usd": round(fuel), "emissions_tco2": round(emis),
                   "unserved_mwh": round(float(sched.unserved_mw.sum()), 1),
                   "renewable_share_pct": round(float(sched.renewable_used_mw.sum() / demand.sum() * 100), 1),
                   "curtailed_mwh": round(float(sched.renewable_curtailed_mw.sum()), 1),
                   "avg_price_usd_mwh": round(float(np.nanmean(sched.price_usd_mwh)), 1),
                   "peak_price_usd_mwh": round(float(np.nanmax(sched.price_usd_mwh)), 1),
                   "thermal_mwh": round(thermal_mwh)}
        return {"status": "optimal", "summary": summary, "schedule": sched}


# ===================================================================== merit-order price proxy
def merit_price_fn(plants: pd.DataFrame):
    order = plants.sort_values("cost_usd_mwh")
    cum = order.capacity_mw.cumsum().to_numpy(); cst = order.cost_usd_mwh.to_numpy()

    def price(net_mw):
        net = np.asarray(net_mw, float)
        i = np.searchsorted(cum, net, side="left").clip(0, len(cst) - 1)
        scarcity = 250.0 * (net / cum[-1]) ** 2                       # scarcity / uplift adder grows with utilisation
        return cst[i] + scarcity
    return price


def battery_lp_profit(prices: np.ndarray, cap: float, power: float, eff: float, soc0: float) -> float:
    """Perfect-foresight arbitrage optimum for one battery (upper bound used to benchmark the RL agent)."""
    T = len(prices); ec = np.sqrt(eff)
    c = np.concatenate([prices + 1.0, -prices + 1.0, np.zeros(T)])   # chg, dis, soc  (minimise cost)
    A = lil_matrix((T, 3 * T)); b = np.zeros(T)
    for t in range(T):
        A[t, 2 * T + t] = 1; A[t, t] = -ec; A[t, T + t] = 1 / ec
        if t > 0:
            A[t, 2 * T + t - 1] = -1
        else:
            b[t] = soc0
    bounds = [(0, power)] * (2 * T) + [(0, cap)] * (T - 1) + [(soc0, cap)]
    res = linprog(c, A_eq=A.tocsr(), b_eq=b, bounds=bounds, method="highs")
    return float(-res.fun) if res.status == 0 else 0.0


# ===================================================================== RL battery agent
class BatteryRL:
    """Tabular Q-learning on (hour, SoC bucket, price bucket). Actions: 0 idle, 1 charge, 2 discharge."""

    def __init__(self, cap_mwh=400.0, power_mw=100.0, eff=0.92, n_soc=21, n_price=8, seed=0):
        self.cap, self.power, self.eff = cap_mwh, power_mw, eff
        self.n_soc, self.n_price = n_soc, n_price
        self.Q = np.zeros((24, n_soc, n_price, 3))
        self.edges = None
        self.rng = np.random.default_rng(seed)
        self.report: dict = {}

    # -- environment ----------------------------------------------------
    def _bucket_price(self, p):
        return int(np.searchsorted(self.edges, p))

    def _step(self, soc, action):
        ec = np.sqrt(self.eff)
        if action == 1:
            e = min(self.power, (self.cap - soc) / ec); return soc + ec * e, -e          # buys e MWh
        if action == 2:
            e = min(self.power, soc * ec); return soc - e / ec, e                           # sells e MWh
        return soc, 0.0

    def _soc_b(self, soc):
        return int(round(soc / self.cap * (self.n_soc - 1)))

    def fit(self, day_prices: np.ndarray, episodes: int = 40000, alpha=0.1, gamma=1.0):
        """day_prices: (n_days, 24) price curves ($/MWh) used as training experience."""
        self.edges = np.quantile(day_prices, np.linspace(0, 1, self.n_price + 1)[1:-1])
        mean_price = float(day_prices.mean())
        self.lo, self.hi = np.quantile(day_prices, 0.35), np.quantile(day_prices, 0.65)   # fair rule baseline (no foresight)
        for ep in range(episodes):
            eps = max(0.05, 1.0 - ep / (0.7 * episodes))
            prices = day_prices[self.rng.integers(len(day_prices))]
            soc = self.cap * 0.5
            for h in range(24):
                s = (h, self._soc_b(soc), self._bucket_price(prices[h]))
                a = self.rng.integers(3) if self.rng.random() < eps else int(np.argmax(self.Q[s]))
                soc2, mwh = self._step(soc, a)
                r = prices[h] * mwh - 1.0 * abs(mwh)                     # revenue - degradation cost
                if h == 23:
                    r += (soc2 - self.cap * 0.5) * mean_price * 0.9      # terminal value of stored energy
                    target = r
                else:
                    s2 = (h + 1, self._soc_b(soc2), self._bucket_price(prices[h + 1]))
                    target = r + gamma * self.Q[s2].max()
                self.Q[s][a] += alpha * (target - self.Q[s][a])
                soc = soc2
        return self

    def run_day(self, prices: np.ndarray, policy="rl") -> dict:
        soc = self.cap * 0.5; profit = 0.0; acts = []
        lo, hi = self.lo, self.hi
        for h in range(24):
            if policy == "rl":
                a = int(np.argmax(self.Q[h, self._soc_b(soc), self._bucket_price(prices[h])]))
            elif policy == "rule":
                a = 1 if prices[h] <= lo else 2 if prices[h] >= hi else 0
            else:
                a = 0
            soc, mwh = self._step(soc, a)
            profit += prices[h] * mwh - 1.0 * abs(mwh); acts.append(a)
        profit += (soc - self.cap * 0.5) * float(np.mean(prices)) * 0.9
        return {"profit": profit, "actions": acts}

    def evaluate(self, test_days: np.ndarray) -> dict:
        res = {k: np.mean([self.run_day(d, k)["profit"] for d in test_days]) for k in ("idle", "rule", "rl")}
        res["lp_optimal"] = float(np.mean([battery_lp_profit(d, self.cap, self.power, self.eff, self.cap * 0.5)
                                           for d in test_days]))
        res = {k: round(float(v), 0) for k, v in res.items()}
        res["rl_vs_rule_pct"] = round((res["rl"] - res["rule"]) / max(abs(res["rule"]), 1) * 100, 1)
        res["rl_pct_of_optimal"] = round(res["rl"] / max(res["lp_optimal"], 1) * 100, 1)
        self.report = {"metric": "avg daily arbitrage profit (USD) on held-out days", "test_days": len(test_days),
                       "note": "rule = fixed price thresholds (no foresight); lp_optimal = perfect-foresight upper bound", **res}
        return self.report


def train_ppo_battery(day_prices: np.ndarray, steps: int = 50000):   # pragma: no cover - optional deps
    """Stable-Baselines3 PPO version of the battery agent (requires gymnasium + stable-baselines3)."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
    except Exception:
        return None

    class Env(gym.Env):
        def __init__(self):
            self.agent = BatteryRL(); self.observation_space = spaces.Box(0, 1, (3,), np.float32)
            self.action_space = spaces.Discrete(3); self.mx = float(day_prices.max())

        def reset(self, seed=None, options=None):
            super().reset(seed=seed); self.p = day_prices[np.random.randint(len(day_prices))]; self.h = 0
            self.soc = self.agent.cap * 0.5; return self._obs(), {}

        def _obs(self):
            return np.array([self.h / 24, self.soc / self.agent.cap, self.p[min(self.h, 23)] / self.mx], np.float32)

        def step(self, a):
            self.soc, mwh = self.agent._step(self.soc, int(a)); r = self.p[self.h] * mwh - 2 * abs(mwh)
            self.h += 1; return self._obs(), float(r) / 100, self.h >= 24, False, {}

    return PPO("MlpPolicy", Env(), verbose=0).learn(steps)
