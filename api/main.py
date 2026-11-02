"""REST API gateway (FastAPI). Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000

Auth:   send header  X-API-Key: <key>   (roles viewer < operator < admin, see README / .env.example)
Docs:   http://localhost:8000/docs
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from smartgrid import bonus
from smartgrid.config import Settings
from smartgrid.platform import GridPlatform
from smartgrid.security import ROLE_LEVEL
from smartgrid.utils import clean

platform = GridPlatform()
DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    platform.initialize()
    yield


app = FastAPI(title="Enterprise Smart Grid Intelligence & Energy Optimization Platform", version="1.0.0", lifespan=lifespan,
              description="Ezitech AI-238 - forecasting, optimisation, predictive maintenance, theft detection, knowledge graph, AI copilot.")


# ----------------------------------------------------------------------------- security
def require(role: str):
    def dep(request: Request, x_api_key: str | None = Header(default=None)) -> str:
        if not platform.ready:
            raise HTTPException(503, "platform is still initialising")
        r = platform.auth.role_for(x_api_key)
        if r is None:
            platform.audit.record("anonymous", "auth.denied", {"path": request.url.path})
            raise HTTPException(401, "missing or invalid X-API-Key")
        if ROLE_LEVEL[r] < ROLE_LEVEL[role]:
            platform.audit.record(f"{r}:{x_api_key[:4]}", "auth.forbidden", {"path": request.url.path, "needs": role})
            raise HTTPException(403, f"requires role '{role}'")
        request.state.actor = f"{r}:{(x_api_key or '')[:4]}"
        return request.state.actor
    return dep


viewer, operator, admin = require("viewer"), require("operator"), require("admin")


@app.middleware("http")
async def audit_and_timing(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    resp.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    if platform.ready and (request.method != "GET" or "/simulate" in request.url.path):
        platform.audit.record(getattr(request.state, "actor", "unknown"), f"{request.method} {request.url.path}",
                              {"status": resp.status_code, "ms": round(ms, 1)})
    return resp


@app.exception_handler(KeyError)
async def key_error(_, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})


@app.exception_handler(ValueError)
async def value_error(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ----------------------------------------------------------------------------- models
class CopilotRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class OutageRequest(BaseModel):
    plant_id: str | None = None
    hours: int = Field(default=24, ge=1, le=72)
    temp_delta: float = Field(default=0.0, ge=-10, le=15)


class IngestRequest(BaseModel):
    events: list[dict] = Field(max_length=100000, description="[{type:0..4, asset:int, value:float, ts:float}]")


class StreamSimRequest(BaseModel):
    n_events: int = Field(default=100000, ge=1, le=5_000_000)
    anomaly_rate: float = Field(default=0.002, ge=0, le=0.5)


class Neo4jExport(BaseModel):
    uri: str | None = None
    user: str | None = None
    password: str | None = None


# ----------------------------------------------------------------------------- root / health
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(DASHBOARD) if DASHBOARD.exists() else PlainTextResponse("Smart Grid API - see /docs")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ready" if platform.ready else "starting", "version": "1.0.0"}


@app.get("/api/v1/info", tags=["system"])
def info(_: str = Depends(viewer)):
    return platform.info()


@app.get("/api/v1/dashboard", tags=["dashboard"], summary="Executive dashboard payload")
def dashboard(_: str = Depends(viewer)):
    return platform.dashboard()


# ----------------------------------------------------------------------------- forecasting
@app.get("/api/v1/forecast/demand", tags=["forecasting"])
def forecast_demand(horizon: int = Query(48, ge=1, le=168), region: str | None = None, temp_delta: float = Query(0.0, ge=-10, le=15),
                    surge_pct: float = Query(0.0, ge=0, le=1.0), _: str = Depends(viewer)):
    return platform.demand_forecast(horizon, region, temp_delta, surge_pct)


@app.get("/api/v1/forecast/peak", tags=["forecasting"])
def forecast_peak(horizon: int = Query(24, ge=1, le=168), temp_delta: float = 0.0, _: str = Depends(viewer)):
    return clean(platform.demand.peak(horizon, temp_delta=temp_delta))


@app.get("/api/v1/forecast/day/{offset}", tags=["forecasting"], summary="offset 0=today, 1=tomorrow")
def forecast_day(offset: int, _: str = Depends(viewer)):
    if not 0 <= offset <= 5:
        raise ValueError("offset must be 0..5")
    return platform.forecast_day(offset)


@app.get("/api/v1/forecast/seasonal", tags=["forecasting"])
def forecast_seasonal(_: str = Depends(viewer)):
    return clean(platform.demand.seasonal_profile())


@app.get("/api/v1/forecast/segments", tags=["forecasting"], summary="Industrial / commercial / residential demand")
def forecast_segments(horizon: int = Query(24, ge=1, le=72), _: str = Depends(viewer)):
    return clean(platform.demand.segment_forecast(horizon))


@app.get("/api/v1/forecast/regions", tags=["forecasting"])
def forecast_regions(horizon: int = Query(24, ge=1, le=72), _: str = Depends(viewer)):
    return clean(platform.demand.regional_summary(horizon))


@app.get("/api/v1/forecast/surges", tags=["forecasting"], summary="Detected emergency demand surges (last 30 days)")
def forecast_surges(_: str = Depends(viewer)):
    return clean(platform.demand.detect_surges().head(50))


@app.get("/api/v1/forecast/explain", tags=["forecasting", "explainability"])
def forecast_explain(_: str = Depends(viewer)):
    return clean({"peak_explanation": platform.demand.explain_forecast(), "feature_importance": platform.demand.feature_importance()})


@app.get("/api/v1/renewables/forecast", tags=["renewables"])
def renewables_forecast(horizon: int = Query(48, ge=1, le=168), _: str = Depends(viewer)):
    return platform.renewable_forecast(horizon)


# ----------------------------------------------------------------------------- optimisation
@app.get("/api/v1/optimization/dispatch", tags=["optimization"], summary="Multi-period LP dispatch (cost, renewables, battery)")
def opt_dispatch(horizon: int = Query(24, ge=1, le=72), carbon_price: float = Query(0.0, ge=0, le=500), outage: str = "",
                 temp_delta: float = 0.0, surge_pct: float = Query(0.0, ge=0, le=1), use_battery: bool = True, _: str = Depends(operator)):
    return platform.dispatch(horizon, [x for x in outage.split(",") if x], carbon_price, temp_delta, surge_pct, use_battery)


@app.get("/api/v1/optimization/battery-rl", tags=["optimization"], summary="Reinforcement-learning battery policy")
def opt_battery(retrain: bool = False, _: str = Depends(operator)):
    return platform.battery_rl(retrain)


@app.get("/api/v1/optimization/load-balance", tags=["optimization"])
def opt_load_balance(_: str = Depends(operator)):
    return platform.load_balance()


# ----------------------------------------------------------------------------- stability
@app.get("/api/v1/stability/voltage", tags=["stability"])
def stab_voltage(temp_delta: float = 0.0, surge_pct: float = Query(0.0, ge=0, le=1), _: str = Depends(viewer)):
    return platform.voltage(temp_delta, surge_pct)


@app.get("/api/v1/stability/voltage/{substation_id}/explain", tags=["stability", "explainability"])
def stab_voltage_explain(substation_id: str, temp_delta: float = 0.0, surge_pct: float = Query(0.0, ge=0, le=1), _: str = Depends(viewer)):
    return platform.explain_voltage(substation_id, temp_delta, surge_pct)


@app.get("/api/v1/stability/frequency", tags=["stability"])
def stab_frequency(loss_mw: float | None = Query(None, gt=0), _: str = Depends(viewer)):
    return platform.frequency(loss_mw)


# ----------------------------------------------------------------------------- maintenance
@app.get("/api/v1/maintenance/risk", tags=["maintenance"])
def maint_risk(top: int = Query(15, ge=1, le=200), horizon_days: int = Query(7, ge=1, le=90), _: str = Depends(viewer)):
    return platform.maintenance(top, horizon_days)


@app.get("/api/v1/maintenance/transformers/{transformer_id}/explain", tags=["maintenance", "explainability"])
def maint_explain(transformer_id: str, _: str = Depends(viewer)):
    return clean(platform.pm.explain(transformer_id))


@app.get("/api/v1/maintenance/equipment", tags=["maintenance"], summary="Lines, breakers, generators anomaly scan")
def maint_equipment(top: int = Query(15, ge=1, le=100), _: str = Depends(viewer)):
    return clean(platform.equip.scan().head(top))


# ----------------------------------------------------------------------------- theft
@app.get("/api/v1/theft/overview", tags=["theft"])
def theft_overview(_: str = Depends(viewer)):
    return platform.theft_overview()


@app.get("/api/v1/theft/suspects", tags=["theft"])
def theft_suspects(top: int = Query(50, ge=1, le=1000), region: str | None = None, min_score: float = Query(0.9, ge=0, le=1), _: str = Depends(operator)):
    return clean(platform.theft.suspects(top, region, min_score))


@app.get("/api/v1/theft/heatmap", tags=["theft"])
def theft_heatmap(_: str = Depends(viewer)):
    return clean(platform.theft.heatmap())


@app.get("/api/v1/theft/meters/{meter_id}/explain", tags=["theft", "explainability"])
def theft_explain(meter_id: str, _: str = Depends(operator)):
    return clean(platform.theft.explain(meter_id))


# ----------------------------------------------------------------------------- knowledge graph
@app.get("/api/v1/graph/stats", tags=["graph"])
def graph_stats(_: str = Depends(viewer)):
    return clean(platform.kg.stats())


@app.get("/api/v1/graph/nodes/{node_id}", tags=["graph"])
def graph_node(node_id: str, _: str = Depends(viewer)):
    return clean(platform.kg.node(node_id))


@app.get("/api/v1/graph/impact/{asset_id}", tags=["graph"], summary="Customers affected if the asset fails (N-1 aware)")
def graph_impact(asset_id: str, _: str = Depends(viewer)):
    return clean(platform.kg.impact(asset_id))


@app.get("/api/v1/graph/criticality", tags=["graph"])
def graph_criticality(top: int = Query(10, ge=1, le=100), _: str = Depends(viewer)):
    return clean(platform.kg.criticality(top))


@app.get("/api/v1/graph/path", tags=["graph"])
def graph_path(source: str, target: str, _: str = Depends(viewer)):
    return {"path": platform.kg.shortest_path(source, target)}


@app.get("/api/v1/graph/cypher", tags=["graph"], response_class=PlainTextResponse)
def graph_cypher(max_nodes: int = Query(200, ge=1, le=5000), _: str = Depends(viewer)):
    return platform.kg.to_cypher(max_nodes)


@app.post("/api/v1/graph/export-neo4j", tags=["graph"])
def graph_export(req: Neo4jExport, _: str = Depends(admin)):
    uri, user, pw = req.uri or Settings.neo4j_uri, req.user or Settings.neo4j_user, req.password or Settings.neo4j_password
    if not uri:
        raise ValueError("NEO4J_URI not configured")
    return platform.kg.export_neo4j(uri, user, pw)


# ----------------------------------------------------------------------------- alerts
@app.get("/api/v1/alerts", tags=["alerts"])
def alerts_list(type: str | None = None, severity: str | None = None, status: str | None = "open", limit: int = Query(100, ge=1, le=1000),
                _: str = Depends(viewer)):
    return clean({"counts": platform.alerts.counts(), "alerts": platform.alerts.list(type, severity, status, limit)})


@app.post("/api/v1/alerts/scan", tags=["alerts"])
def alerts_scan(_: str = Depends(operator)):
    return platform.scan_alerts()


@app.post("/api/v1/alerts/{alert_id}/ack", tags=["alerts"])
def alerts_ack(alert_id: str, _: str = Depends(operator)):
    if not platform.alerts.acknowledge(alert_id):
        raise KeyError(alert_id)
    return {"acknowledged": alert_id}


# ----------------------------------------------------------------------------- copilot / simulation
@app.post("/api/v1/copilot/ask", tags=["copilot"], summary="Ask the AI Grid Copilot")
def copilot_ask(req: CopilotRequest, actor: str = Depends(operator)):
    return platform.copilot().ask(req.question, actor)


@app.post("/api/v1/simulate/plant-outage", tags=["simulation"])
def simulate_outage(req: OutageRequest, _: str = Depends(operator)):
    return platform.simulate_plant_outage(req.plant_id, req.hours, req.temp_delta)


# ----------------------------------------------------------------------------- streaming
@app.post("/api/v1/events/ingest", tags=["streaming"], summary="Ingest a batch of grid events (REST -> bus -> processor)")
def events_ingest(req: IngestRequest, _: str = Depends(operator)):
    return platform.ingest(req.events)


@app.post("/api/v1/stream/simulate", tags=["streaming"], summary="Generate + process synthetic telemetry (throughput test)")
def stream_simulate(req: StreamSimRequest, _: str = Depends(operator)):
    return platform.simulate_stream(req.n_events, req.anomaly_rate)


@app.get("/api/v1/stream/metrics", tags=["streaming"])
def stream_metrics(_: str = Depends(viewer)):
    return clean(platform.stream.metrics())


# ----------------------------------------------------------------------------- bonus
@app.get("/api/v1/bonus/carbon", tags=["bonus"])
def bonus_carbon(horizon: int = Query(72, ge=1, le=168), carbon_price: float = Query(0.0, ge=0, le=500), _: str = Depends(viewer)):
    return bonus.carbon_forecast(platform, horizon, carbon_price)


@app.get("/api/v1/bonus/ev-charging", tags=["bonus"])
def bonus_ev(n_evs: int = Query(500000, ge=1000, le=20_000_000), participation: float = Query(0.6, ge=0, le=1), _: str = Depends(viewer)):
    return bonus.ev_charging(platform, n_evs=n_evs, participation=participation)


@app.get("/api/v1/bonus/pricing", tags=["bonus"])
def bonus_pricing(_: str = Depends(viewer)):
    return bonus.dynamic_pricing(platform)


@app.get("/api/v1/bonus/vpp", tags=["bonus"])
def bonus_vpp(_: str = Depends(viewer)):
    return bonus.vpp(platform)


@app.get("/api/v1/bonus/self-healing/{asset_id}", tags=["bonus"])
def bonus_self_heal(asset_id: str, _: str = Depends(operator)):
    return platform.self_heal(asset_id)


# ----------------------------------------------------------------------------- audit
@app.get("/api/v1/audit/recent", tags=["audit"])
def audit_recent(n: int = Query(50, ge=1, le=500), _: str = Depends(admin)):
    return clean(platform.audit.recent(n))


@app.get("/api/v1/audit/verify", tags=["audit"], summary="Verify the tamper-evident hash chain")
def audit_verify(_: str = Depends(admin)):
    return platform.audit.verify()
