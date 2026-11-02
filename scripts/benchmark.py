"""Throughput / latency benchmark for the non-functional requirements:  python -m scripts.benchmark [events]"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("DATABASE_URL", "sqlite://")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                                # noqa: E402

from smartgrid.alerts import AlertCenter          # noqa: E402
from smartgrid.config import load_config          # noqa: E402
from smartgrid.simulator import GridSimulator     # noqa: E402
from smartgrid.streaming import EventGenerator, StreamProcessor   # noqa: E402


def main(total=5_000_000, batch=50_000):
    d = GridSimulator(load_config()).build()
    gen, sp = EventGenerator(d), StreamProcessor(d, AlertCenter())
    batches = [gen.batch(batch) for _ in range(10)]                # pre-generate: measure processing only
    done, t0, lat = 0, time.perf_counter(), []
    while done < total:
        b = batches[(done // batch) % len(batches)]
        t = time.perf_counter(); sp.process_batch(b); lat.append((time.perf_counter() - t) * 1000); done += batch
    el = time.perf_counter() - t0
    lat = np.array(lat)
    print(f"processed {done:,} events in {el:.2f}s = {done/el:,.0f} events/s (single core, processing only)")
    print(f"=> {done/el*86400/1e6:,.0f} M events/day per core;  requirement: 100 M/day (~1,160/s avg)")
    print(f"batch ({batch:,} events) latency p50={np.percentile(lat,50):.1f} ms  p99={np.percentile(lat,99):.1f} ms  (requirement: < 1000 ms)")
    print(f"per-event alerting latency (micro-batch of 1,000): ", end="")
    small = gen.batch(1000); ts = []
    for _ in range(200):
        t = time.perf_counter(); sp.process_batch(small); ts.append((time.perf_counter() - t) * 1000)
    print(f"p50={np.percentile(ts,50):.2f} ms p99={np.percentile(ts,99):.2f} ms")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000)
