"""Synthetic telemetry producer: python -m scripts.stream_producer --rate 20000 (events/s)"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smartgrid.config import Settings, load_config          # noqa: E402
from smartgrid.simulator import GridSimulator               # noqa: E402
from smartgrid.streaming import EventGenerator, get_bus     # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rate", type=int, default=20000); ap.add_argument("--seconds", type=int, default=0)
    a = ap.parse_args()
    d = GridSimulator(load_config()).build(); gen = EventGenerator(d); bus = get_bus(Settings.kafka_bootstrap, Settings.kafka_topic)
    t0 = time.time(); sent = 0
    while not a.seconds or time.time() - t0 < a.seconds:
        t = time.time(); sent += bus.publish(Settings.kafka_topic, gen.to_dicts(gen.batch(a.rate)))
        time.sleep(max(0, 1 - (time.time() - t)))
        print(f"[producer] sent={sent:,}", flush=True)


if __name__ == "__main__":
    main()
