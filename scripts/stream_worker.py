"""Kafka consumer worker: python -m scripts.stream_worker   (falls back to a self-feeding in-memory loop without Kafka)."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smartgrid.alerts import AlertCenter                    # noqa: E402
from smartgrid.config import Settings, load_config          # noqa: E402
from smartgrid.simulator import GridSimulator               # noqa: E402
from smartgrid.storage import Store                         # noqa: E402
from smartgrid.streaming import EventGenerator, StreamProcessor, get_bus   # noqa: E402


def main():
    d = GridSimulator(load_config()).build()                 # topology must match the producer (same seed/scale)
    alerts = AlertCenter(Store(Settings.database_url)); sp = StreamProcessor(d, alerts)
    bus = get_bus(Settings.kafka_bootstrap, Settings.kafka_topic); gen = EventGenerator(d)
    print(f"[worker] bus={bus.name}", flush=True); last = time.time()
    while True:
        if bus.name == "memory":                              # demo mode: produce + consume locally
            bus.publish("t", gen.to_dicts(gen.batch(5000))); time.sleep(0.2)
        r = sp.process_dicts(bus.consume(50000, timeout_s=0.5))
        if r["events"] and time.time() - last > 5:
            print(f"[worker] {sp.metrics()}", flush=True); last = time.time()


if __name__ == "__main__":
    main()
