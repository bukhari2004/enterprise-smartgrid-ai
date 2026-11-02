"""Central configuration. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GridConfig:
    """Size of the simulated grid. The *real* provider in the case study has
    35 plants / 18M meters / 750 substations / 120k transformers; the same code
    runs at any scale, presets below keep the demo fast on a laptop."""

    seed: int = 42
    n_regions: int = 6
    n_plants: int = 8
    n_substations: int = 24
    n_transformers: int = 150
    meters_per_transformer: int = 20
    n_solar: int = 8
    n_wind: int = 6
    n_batteries: int = 4
    history_days: int = 365
    forecast_days: int = 7
    meter_days: int = 60
    theft_rate: float = 0.04
    start: str = "2025-10-31 00:00"   # history ends 2026-10-30 23:00 ("now" of the demo)


PRESETS = {
    "tiny": dict(n_plants=8, n_substations=12, n_transformers=60, meters_per_transformer=15, history_days=240),
    "demo": dict(),
    "large": dict(n_plants=35, n_substations=120, n_transformers=1200, meters_per_transformer=40,
                  n_solar=40, n_wind=30, n_batteries=20),
}


def load_config() -> GridConfig:
    preset = os.getenv("SMARTGRID_SCALE", "demo").lower()
    cfg = GridConfig(**PRESETS.get(preset, {}))
    cfg.seed = int(os.getenv("SMARTGRID_SEED", cfg.seed))
    return cfg


class Settings:
    """Runtime (non-simulation) settings."""

    app_env = os.getenv("APP_ENV", "dev")
    database_url = os.getenv("DATABASE_URL", "sqlite:///data/smartgrid.db")
    redis_url = os.getenv("REDIS_URL", "")
    kafka_bootstrap = os.getenv("KAFKA_BOOTSTRAP", "")
    kafka_topic = os.getenv("KAFKA_TOPIC", "grid.events")
    neo4j_uri = os.getenv("NEO4J_URI", "")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    api_keys = os.getenv("API_KEYS", "")        # "key:role,key2:role"
    audit_path = os.getenv("AUDIT_PATH", "data/audit.log.jsonl")
    llm_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    model_dir = os.getenv("MODEL_DIR", "models")
