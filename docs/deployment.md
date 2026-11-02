# Deployment guide
**Local:** `pip install -r requirements.txt && uvicorn api.main:app --port 8000` (start-up trains all models, ~12 s on the default grid; the RL agent trains lazily on first use, ~10 s).
**Docker:** `docker compose up --build`; add the synthetic producer with `docker compose --profile demo up producer`; scale consumers with `--scale worker=4`.
**Scale presets:** `SMARTGRID_SCALE=tiny|demo|large` (large ≈ 48 k meters, 35 plants, 120 substations).
**Security checklist:** `APP_ENV=prod`, set `API_KEYS`, put the API behind TLS (reverse proxy), rotate keys, restrict `/audit/*` and `/graph/export-neo4j` to admin (already enforced), mount `data/` on persistent storage, back up PostgreSQL.
**Observability:** `/health`, `/api/v1/stream/metrics`, `X-Process-Time-ms` header on every response, `/api/v1/audit/verify`.
**Config:** see `.env.example`.
