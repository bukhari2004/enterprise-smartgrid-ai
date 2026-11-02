# REST API (all under /api/v1 unless noted; header `X-API-Key`)

| Method | Path | Note |
|---|---|---|
| GET | `/` |  |
| GET | `/health` |  |
| GET | `/api/v1/info` |  |
| GET | `/api/v1/dashboard` | Executive dashboard payload |
| GET | `/api/v1/forecast/demand` |  |
| GET | `/api/v1/forecast/peak` |  |
| GET | `/api/v1/forecast/day/{offset}` | offset 0=today, 1=tomorrow |
| GET | `/api/v1/forecast/seasonal` |  |
| GET | `/api/v1/forecast/segments` | Industrial / commercial / residential demand |
| GET | `/api/v1/forecast/regions` |  |
| GET | `/api/v1/forecast/surges` | Detected emergency demand surges (last 30 days) |
| GET | `/api/v1/forecast/explain` |  |
| GET | `/api/v1/renewables/forecast` |  |
| GET | `/api/v1/optimization/dispatch` | Multi-period LP dispatch (cost, renewables, battery) |
| GET | `/api/v1/optimization/battery-rl` | Reinforcement-learning battery policy |
| GET | `/api/v1/optimization/load-balance` |  |
| GET | `/api/v1/stability/voltage` |  |
| GET | `/api/v1/stability/voltage/{substation_id}/explain` |  |
| GET | `/api/v1/stability/frequency` |  |
| GET | `/api/v1/maintenance/risk` |  |
| GET | `/api/v1/maintenance/transformers/{transformer_id}/explain` |  |
| GET | `/api/v1/maintenance/equipment` | Lines, breakers, generators anomaly scan |
| GET | `/api/v1/theft/overview` |  |
| GET | `/api/v1/theft/suspects` |  |
| GET | `/api/v1/theft/heatmap` |  |
| GET | `/api/v1/theft/meters/{meter_id}/explain` |  |
| GET | `/api/v1/graph/stats` |  |
| GET | `/api/v1/graph/nodes/{node_id}` |  |
| GET | `/api/v1/graph/impact/{asset_id}` | Customers affected if the asset fails (N-1 aware) |
| GET | `/api/v1/graph/criticality` |  |
| GET | `/api/v1/graph/path` |  |
| GET | `/api/v1/graph/cypher` |  |
| POST | `/api/v1/graph/export-neo4j` |  |
| GET | `/api/v1/alerts` |  |
| POST | `/api/v1/alerts/scan` |  |
| POST | `/api/v1/alerts/{alert_id}/ack` |  |
| POST | `/api/v1/copilot/ask` | Ask the AI Grid Copilot |
| POST | `/api/v1/simulate/plant-outage` |  |
| POST | `/api/v1/events/ingest` | Ingest a batch of grid events (REST -> bus -> processor) |
| POST | `/api/v1/stream/simulate` | Generate + process synthetic telemetry (throughput test) |
| GET | `/api/v1/stream/metrics` |  |
| GET | `/api/v1/bonus/carbon` |  |
| GET | `/api/v1/bonus/ev-charging` |  |
| GET | `/api/v1/bonus/pricing` |  |
| GET | `/api/v1/bonus/vpp` |  |
| GET | `/api/v1/bonus/self-healing/{asset_id}` |  |
| GET | `/api/v1/audit/recent` |  |
| GET | `/api/v1/audit/verify` | Verify the tamper-evident hash chain |
