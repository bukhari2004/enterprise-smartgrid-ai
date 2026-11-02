# Enterprise Smart Grid Intelligence & Energy Optimization Platform
**Ezitech EEF case study AI-238** — demand forecasting, grid optimisation, predictive maintenance, energy-theft detection,
grid knowledge graph, real-time stream analytics, and an AI operations copilot, behind a secured REST API and a web dashboard.

> **Data note.** The real provider's data (35 plants, 750 substations, 120k transformers, 18M meters) is not available, so the
> platform ships with a **grid simulator** (`smartgrid/simulator.py`) that generates topology, weather, demand, renewables,
> smart-meter readings, transformer sensors and *hidden* ground truth for evaluation. Every model is trained on simulated data —
> metrics below show the pipeline works end-to-end, **not** real-world accuracy. Swap the simulator for your data loaders (see "Going to real data").

## Quick start (no Docker, no external services)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # add requirements-full.txt for XGBoost/Kafka/Neo4j/Redis/Postgres/SB3
python -m scripts.demo                     # runs every module, prints results (~40 s)
uvicorn api.main:app --port 8000           # API + dashboard  ->  http://localhost:8000   (docs: /docs)
python -m unittest discover -s tests -v    # 24 tests
python -m scripts.benchmark                # streaming throughput / latency
```
Dev API keys (only when `APP_ENV=dev` and `API_KEYS` is empty): `demo-viewer`, `demo-operator`, `demo-admin` — send as header `X-API-Key`.
Production: `APP_ENV=prod API_KEYS="key1:admin,key2:operator,key3:viewer"`.

Full stack: `docker compose up --build` (API, Kafka worker, PostgreSQL, Redis, Kafka, Neo4j). Every external service is optional —
the code falls back to SQLite / in-memory cache / in-memory bus / NetworkX when a service is not configured.

```bash
curl -H "X-API-Key: demo-operator" localhost:8000/api/v1/forecast/day/1
curl -H "X-API-Key: demo-operator" -X POST localhost:8000/api/v1/copilot/ask -H 'content-type: application/json' \
     -d '{"question":"Which transformer is likely to fail next week?"}'
curl -H "X-API-Key: demo-operator" -X POST localhost:8000/api/v1/simulate/plant-outage -H 'content-type: application/json' -d '{"plant_id":"P-003"}'
```

## Requirements coverage
| Requirement | Where | How |
|---|---|---|
| Demand forecasting (hourly, regional, peak, seasonal, industrial, surges) | `forecasting.py` | Gradient boosting (XGBoost if installed) + calendar/weather/lag features, recursive multi-step, P10/P90 quantile models, what-if (`temp_delta`, `surge_pct`), segment split, residual-z surge detector, permutation importance. Prophet baseline hook. |
| Renewable intelligence (solar/wind forecast, drops) | `renewables.py` | Weather → capacity-factor models; ramp-down detector (excludes predictable sunset ramp). |
| Optimisation (dispatch, balancing, renewable allocation, storage) | `optimization.py` | Multi-period **LP (HiGHS)**: economic dispatch + ramp limits + must-run + renewable curtailment + battery + carbon price + unserved-energy slack; duals give marginal price. |
| Reinforcement learning | `optimization.py` | Tabular Q-learning battery agent (NumPy) benchmarked vs. fixed-threshold rule and perfect-foresight LP; PPO (Stable-Baselines3) trainer included, optional. |
| Voltage / frequency stability | `stability.py` | Explainable additive voltage model (loading, feeder drop, renewable ramp, cap bank); swing-equation frequency simulation with governor droop and UFLS; load-balance transfer recommender. |
| Predictive maintenance | `maintenance.py` | Failure-probability classifier (30d/7d, expected days), per-asset occlusion explanations + recommended actions, Isolation-Forest monitor for lines/breakers/generators. |
| Energy-theft detection | `theft.py` | Peer deviation, step-change, transformer energy balance, night ratio, billing mismatch → Isolation Forest + supervised blend; typed (tampering / bypass / commercial fraud / billing anomaly), heatmap, explanations. |
| Knowledge graph | `graph.py` | NetworkX graph of 10 node types; N-1 aware impact/islanding analysis, criticality ranking, GNN-style risk propagation; Cypher export + Neo4j loader. |
| Real-time ingestion & alerts | `streaming.py`, `alerts.py` | Kafka (or in-memory) bus, vectorised micro-batch processor, de-duplicated alert engine (8 alert types, escalation bypasses cooldown). |
| AI copilot | `copilot.py` | Intent router (TF-IDF) → grounded tool calls → answer; optional LLM rewrite; runbook retrieval; LangGraph version. |
| Explainability | all engines | Every prediction has a `/explain` path (voltage contributions, failure drivers, theft evidence, forecast drivers). |
| Security & audit | `security.py`, `audit.py` | API-key RBAC (viewer/operator/admin), hash-chained tamper-evident audit log (DB + JSONL) with `/audit/verify`. |
| Bonus | `bonus.py` | Carbon forecasting + carbon-price what-if, EV charging optimiser, dynamic pricing, VPP, autonomous self-healing (recommendation mode). |

## Measured results (simulated data, `python -m scripts.demo`, sklearn backend)
| Module | Result |
|---|---|
| Demand, day-ahead (7 held-out days) | **0.63 % MAPE** vs 1.29 % seasonal-naive; peak error 0.98 % |
| Solar / wind forecast (14 held-out days) | R² 0.999 / 0.997 |
| LP dispatch (24 h) | optimal in ~0.1 s; 0 MWh unserved; reproduces plant-outage and carbon-price scenarios |
| RL battery | +3.6 % vs. fixed-threshold rule, **65 % of perfect-foresight optimum** (honest: modest) |
| Transformer failure | ROC-AUC 0.80 (time-split), 0.95 on current fleet; 13 of 20 true failures in top-20 |
| Theft | hold-out ROC-AUC **0.95** (IF alone 0.85); ~73 % precision at the 0.93 score cut, billing anomalies rarely mislabeled (2.4 %) |
| Streaming | ~20 M events/s vectorised rules (columnar), ~2.3 M/s from dict events, single core, excl. Kafka/network. Target: 100 M/day ≈ 1.2 k/s avg. Batch latency p99 ≈ 3 ms per 50k events |

## Architecture
See `docs/architecture.md` (Mermaid diagram, data flow, scaling plan) and `docs/deployment.md`.
```
smartgrid/   simulator · forecasting · renewables · optimization · stability · maintenance · theft · graph
             streaming · alerts · storage · audit · security · copilot · bonus · platform (orchestrator)
api/main.py  FastAPI gateway (48 routes)           dashboard/index.html  executive dashboard (Chart.js)
scripts/     demo · benchmark · stream_worker · stream_producer       tests/   24 unit/integration tests
```

## Going to real data
1. Implement loaders returning the same frames as `SimData` (`demand`, `weather`, `meter_daily`, `maint_history`, topology tables) — e.g. from the data lake/Postgres.
2. Replace the hidden-truth evaluation with real inspection outcomes and failure logs; retrain (`GridPlatform.initialize()` trains everything at start-up — move to scheduled jobs and load from `models/` in production).
3. Partition Kafka by `substation_id`, run N workers in one consumer group; store meter time-series in TimescaleDB/ClickHouse; keep this API stateless behind a load balancer.
4. Scale notes: theft features are computed per-meter with pandas — move to Spark/Dask for 18 M meters; the knowledge graph should live in Neo4j (`/graph/export-neo4j`), not in process memory.

## Known limitations (read before relying on it)
- All data is synthetic; accuracy numbers are optimistic by construction (the simulator's patterns are simpler than a real grid).
- The grid is modelled as a single "copper plate" in dispatch (no power-flow / line-limit constraints); voltage and frequency models are simplified, explainable approximations, not a substitute for AC power-flow / dynamic simulation tools.
- Self-healing and dispatch outputs are **recommendations** requiring operator approval.
- In the build environment (no internet) the FastAPI app, Docker/compose files, Kafka, Neo4j, Redis, PostgreSQL, XGBoost, SB3 and LangGraph integrations could **not** be executed. The route handlers were smoke-tested against a stub of FastAPI and all core logic is covered by the unit tests; the integrations are written to documented APIs and guarded with fallbacks, but run `docker compose up` and `pytest`-style checks once on your machine.
