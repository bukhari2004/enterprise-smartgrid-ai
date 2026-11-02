"""Persistence (PostgreSQL in production, SQLite fallback) and caching (Redis, in-memory fallback)."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, ts TEXT, type TEXT, severity TEXT, asset_id TEXT,
       region_id TEXT, message TEXT, evidence TEXT, status TEXT)""",
    """CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, ts TEXT, kind TEXT, asset_id TEXT, payload TEXT)""",
    """CREATE TABLE IF NOT EXISTS audit (id TEXT PRIMARY KEY, ts TEXT, actor TEXT, action TEXT, detail TEXT,
       prev_hash TEXT, hash TEXT)""",
]


class Store:
    """Tiny SQL wrapper. `sqlite:///path` or `postgresql://user:pass@host/db` (needs psycopg2)."""

    def __init__(self, url: str):
        self.url = url
        self.lock = threading.RLock()
        self.pg = url.startswith("postgres")
        if self.pg:                                                        # pragma: no cover
            import psycopg2                                                # type: ignore
            self.conn = psycopg2.connect(url); self.conn.autocommit = True
        else:
            path = url.replace("sqlite:///", "") if url != "sqlite://" else ":memory:"
            if path != ":memory:":
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False)
        for ddl in SCHEMA:
            self.execute(ddl)

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.pg else sql

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.lock:
            cur = self.conn.cursor(); cur.execute(self._sql(sql), params)
            if not self.pg:
                self.conn.commit()
            cur.close()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.lock:
            cur = self.conn.cursor(); cur.execute(self._sql(sql), params)
            cols = [c[0] for c in cur.description]; rows = [dict(zip(cols, r)) for r in cur.fetchall()]; cur.close()
            return rows


class Cache:
    """Redis if REDIS_URL is reachable, otherwise a TTL dict. Values are JSON-serialised."""

    def __init__(self, url: str = ""):
        self.r = None
        self.mem: dict[str, tuple[float, str]] = {}
        if url:                                                            # pragma: no cover
            try:
                import redis                                               # type: ignore
                self.r = redis.Redis.from_url(url); self.r.ping()
            except Exception:
                self.r = None

    @property
    def backend(self) -> str:
        return "redis" if self.r is not None else "memory"

    def get(self, key: str) -> Any | None:
        if self.r is not None:                                             # pragma: no cover
            v = self.r.get(key); return json.loads(v) if v else None
        hit = self.mem.get(key)
        if hit and hit[0] > time.time():
            return json.loads(hit[1])
        return None

    def set(self, key: str, value: Any, ttl: int = 60) -> None:
        s = json.dumps(value, default=str)
        if self.r is not None:                                             # pragma: no cover
            self.r.setex(key, ttl, s)
        else:
            self.mem[key] = (time.time() + ttl, s)
