"""Event-driven ingestion: Kafka (production) or in-memory bus (demo/tests) + vectorised stream analytics.

`StreamProcessor` handles micro-batches with NumPy so a single worker sustains >100k events/s; the case-study target of
100M events/day is ~1,160 events/s on average (~10k/s at 10x peak), so one worker already has a large margin and
Kafka partitions + consumer groups give horizontal scaling.
"""
from __future__ import annotations

import json
import time
from collections import deque

import numpy as np

from .alerts import Alert, AlertCenter
from .simulator import SimData

# event type codes
METER, TR_TEMP, SUB_VOLT, FREQ, RENEW = 0, 1, 2, 3, 4
TYPE_NAMES = {METER: "meter", TR_TEMP: "transformer_temp", SUB_VOLT: "substation_voltage", FREQ: "frequency", RENEW: "renewable_output"}


# ----------------------------------------------------------------- bus
class InMemoryBus:
    name = "memory"

    def __init__(self, maxlen: int = 1_000_000):
        self.q: deque = deque(maxlen=maxlen)

    def publish(self, topic: str, events: list[dict]) -> int:
        self.q.extend(events); return len(events)

    def consume(self, max_events: int = 10000, timeout_s: float = 0.0) -> list[dict]:
        out = []
        while self.q and len(out) < max_events:
            out.append(self.q.popleft())
        return out


class KafkaBus:                                                       # pragma: no cover - needs a broker
    name = "kafka"

    def __init__(self, bootstrap: str, topic: str, group: str = "smartgrid-processors"):
        from kafka import KafkaConsumer, KafkaProducer                # type: ignore
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode(),
                                      linger_ms=20, compression_type="gzip")
        self.consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap, group_id=group, auto_offset_reset="latest",
                                      value_deserializer=lambda b: json.loads(b.decode()), enable_auto_commit=True)

    def publish(self, topic: str, events: list[dict]) -> int:
        for e in events:
            self.producer.send(topic or self.topic, e, key=str(e.get("id", "")).encode())
        self.producer.flush(); return len(events)

    def consume(self, max_events: int = 10000, timeout_s: float = 0.5) -> list[dict]:
        batches = self.consumer.poll(timeout_ms=int(timeout_s * 1000), max_records=max_events)
        return [m.value for recs in batches.values() for m in recs]


def get_bus(bootstrap: str = "", topic: str = "grid.events"):
    if bootstrap:
        try:
            return KafkaBus(bootstrap, topic)
        except Exception as exc:                                       # pragma: no cover
            print(f"[stream] Kafka unavailable ({exc}); falling back to in-memory bus")
    return InMemoryBus()


# ----------------------------------------------------------------- generator
class EventGenerator:
    """Synthetic telemetry: columnar batches (fast) and dict events (for Kafka/REST)."""

    def __init__(self, data: SimData, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        self.n_meters, self.n_tr = len(data.meters), len(data.transformers)
        self.n_sub, self.n_ren = len(data.substations), len(data.renewables)
        self.mix = np.array([0.70, 0.15, 0.10, 0.01, 0.04])

    def batch(self, n: int, anomaly_rate: float = 0.002) -> dict[str, np.ndarray]:
        r = self.rng
        etype = r.choice(5, n, p=self.mix)
        asset = np.zeros(n, dtype=np.int64); val = np.zeros(n)
        m = etype == METER; asset[m] = r.integers(0, self.n_meters, m.sum()); val[m] = r.lognormal(-1.5, 0.5, m.sum())
        m = etype == TR_TEMP; asset[m] = r.integers(0, self.n_tr, m.sum()); val[m] = r.normal(68, 6, m.sum())
        m = etype == SUB_VOLT; asset[m] = r.integers(0, self.n_sub, m.sum()); val[m] = r.normal(0.99, 0.012, m.sum())
        m = etype == FREQ; val[m] = r.normal(50.0, 0.03, m.sum())
        m = etype == RENEW; asset[m] = r.integers(0, self.n_ren, m.sum()); val[m] = r.uniform(80, 120, m.sum())
        bad = r.random(n) < anomaly_rate                                  # inject anomalies
        val = np.where(bad & (etype == TR_TEMP), r.uniform(96, 120, n), val)
        val = np.where(bad & (etype == SUB_VOLT), r.uniform(0.86, 0.925, n), val)
        val = np.where(bad & (etype == FREQ), r.choice([49.2, 50.8], n), val)
        val = np.where(bad & (etype == METER), 0.0, val)
        val = np.where(bad & (etype == RENEW), r.uniform(5, 30, n), val)
        return {"etype": etype, "asset": asset, "value": val, "ts": np.full(n, time.time())}

    @staticmethod
    def to_dicts(b: dict[str, np.ndarray]) -> list[dict]:
        return [{"type": int(t), "asset": int(a), "value": float(v), "ts": float(s)}
                for t, a, v, s in zip(b["etype"], b["asset"], b["value"], b["ts"])]


# ----------------------------------------------------------------- processor
class StreamProcessor:
    def __init__(self, data: SimData, alerts: AlertCenter | None = None):
        self.d = data
        self.alerts = alerts
        n_m, n_t, n_s, n_r = len(data.meters), len(data.transformers), len(data.substations), len(data.renewables)
        self.meter_ewma = np.full(n_m, np.nan); self.tr_temp = np.zeros(n_t); self.sub_volt = np.ones(n_s)
        self.ren_ewma = np.full(n_r, np.nan); self.freq = 50.0
        self.meter_tr = data.meters.transformer_id.map({t: i for i, t in enumerate(data.transformers.transformer_id)}).to_numpy()
        self.tr_ids = data.transformers.transformer_id.to_numpy(); self.sub_ids = data.substations.substation_id.to_numpy()
        self.ren_ids = data.renewables.asset_id.to_numpy(); self.meter_ids = data.meters.meter_id.to_numpy()
        self.meters_per_tr = np.bincount(self.meter_tr, minlength=n_t)
        self.tr_region = data.transformers.region_id.to_numpy(); self.sub_region = data.substations.region_id.to_numpy()
        self.total = 0; self.batches = 0; self.alert_count = 0; self.lat_ms: deque = deque(maxlen=2000)
        self.started = time.time()
        self.counts = {n: 0 for n in TYPE_NAMES.values()}

    def process_batch(self, b: dict[str, np.ndarray], max_alerts: int = 50) -> dict:
        t0 = time.perf_counter()
        et, asset, val = b["etype"], b["asset"], b["value"]
        found: list[Alert] = []

        m = et == METER
        if m.any():
            a, v = asset[m], val[m]
            prev = self.meter_ewma[a]
            drop = (v < 0.2 * prev) & (prev > 0.05)
            self.meter_ewma[a] = np.where(np.isnan(prev), v, 0.95 * prev + 0.05 * v)
            zero = v <= 0.0
            if zero.any():                                                  # outage = many meters of one transformer at zero
                trs, cnt = np.unique(self.meter_tr[a[zero]], return_counts=True)
                for t, c in zip(trs, cnt):
                    if c >= max(2, 0.05 * self.meters_per_tr[t]):
                        found.append(Alert("POWER_OUTAGE", "high", str(self.tr_ids[t]), f"{c} meters reading zero on {self.tr_ids[t]}",
                                           str(self.tr_region[t]), {"zero_meters": int(c)}))
            for i in np.flatnonzero(drop & ~zero)[:5]:
                found.append(Alert("ENERGY_THEFT", "warning", str(self.meter_ids[a[i]]), "sudden consumption drop (possible tampering)",
                                   evidence={"reading": round(float(v[i]), 3), "baseline": round(float(prev[i]), 3)}))
        m = et == TR_TEMP
        if m.any():
            a, v = asset[m], val[m]; self.tr_temp[a] = v
            for i in np.flatnonzero(v > 95)[:10]:
                found.append(Alert("TRANSFORMER_FAILURE_RISK", "critical", str(self.tr_ids[a[i]]), f"oil temperature {v[i]:.0f}°C",
                                   str(self.tr_region[a[i]]), {"oil_temp_c": round(float(v[i]), 1)}))
        m = et == SUB_VOLT
        if m.any():
            a, v = asset[m], val[m]; self.sub_volt[a] = v
            for i in np.flatnonzero((v < 0.93) | (v > 1.07))[:10]:
                found.append(Alert("VOLTAGE_INSTABILITY", "critical" if v[i] < 0.90 else "high", str(self.sub_ids[a[i]]),
                                   f"voltage {v[i]:.3f} pu", str(self.sub_region[a[i]]), {"voltage_pu": round(float(v[i]), 3)}))
        m = et == FREQ
        if m.any():
            v = val[m]; self.freq = float(v[-1])
            if (v < 49.5).any() or (v > 50.5).any():
                bad = float(v[np.argmax(np.abs(v - 50))])
                found.append(Alert("FREQUENCY_DEVIATION", "critical" if abs(bad - 50) > 0.8 else "high", "GRID", f"frequency {bad:.2f} Hz",
                                   evidence={"hz": round(bad, 2)}))
        m = et == RENEW
        if m.any():
            a, v = asset[m], val[m]; prev = self.ren_ewma[a]
            drop = (v < 0.6 * prev) & ~np.isnan(prev)
            self.ren_ewma[a] = np.where(np.isnan(prev), v, 0.9 * prev + 0.1 * v)
            for i in np.flatnonzero(drop)[:5]:
                found.append(Alert("RENEWABLE_ENERGY_DROP", "warning", str(self.ren_ids[a[i]]),
                                   f"output fell to {v[i]:.0f} MW from ~{prev[i]:.0f} MW", evidence={"mw": round(float(v[i]), 1)}))
        emitted = []
        for al in found[:max_alerts]:
            if self.alerts is None or self.alerts.emit(al):
                emitted.append(al)
        for k, name in TYPE_NAMES.items():
            self.counts[name] += int((et == k).sum())
        dt = (time.perf_counter() - t0) * 1000
        self.total += len(et); self.batches += 1; self.alert_count += len(emitted); self.lat_ms.append(dt)
        return {"events": int(len(et)), "alerts": len(emitted), "latency_ms": round(dt, 2)}

    def process_dicts(self, events: list[dict]) -> dict:
        if not events:
            return {"events": 0, "alerts": 0, "latency_ms": 0.0}
        b = {"etype": np.array([e["type"] for e in events]), "asset": np.array([e.get("asset", 0) for e in events]),
             "value": np.array([e["value"] for e in events], float), "ts": np.array([e.get("ts", time.time()) for e in events])}
        return self.process_batch(b)

    def metrics(self) -> dict:
        lat = np.array(self.lat_ms) if self.lat_ms else np.array([0.0])
        el = max(time.time() - self.started, 1e-9)
        return {"events_processed": self.total, "batches": self.batches, "alerts_emitted": self.alert_count,
                "batch_latency_ms_p50": round(float(np.percentile(lat, 50)), 2), "batch_latency_ms_p99": round(float(np.percentile(lat, 99)), 2),
                "uptime_s": round(el, 1), "by_type": self.counts, "latest_frequency_hz": round(self.freq, 3)}
