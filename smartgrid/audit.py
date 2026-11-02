"""Tamper-evident audit log: every record carries sha256(prev_hash + payload) - a hash chain stored in the DB
and mirrored to an append-only JSONL file."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone

from .storage import Store


class AuditLog:
    def __init__(self, store: Store, path: str | None = None):
        self.store, self.path, self.lock = store, path, threading.Lock()
        rows = store.query("SELECT hash FROM audit ORDER BY ts DESC, id DESC LIMIT 1")
        self.prev = rows[0]["hash"] if rows else "GENESIS"
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @staticmethod
    def _h(prev: str, ts: str, actor: str, action: str, detail: str) -> str:
        return hashlib.sha256(f"{prev}|{ts}|{actor}|{action}|{detail}".encode()).hexdigest()

    def record(self, actor: str, action: str, detail: dict | str = "") -> str:
        d = detail if isinstance(detail, str) else json.dumps(detail, default=str, sort_keys=True)
        with self.lock:
            ts = datetime.now(timezone.utc).isoformat()
            h = self._h(self.prev, ts, actor, action, d)
            rid = f"{ts}-{uuid.uuid4().hex[:8]}"
            self.store.execute("INSERT INTO audit (id, ts, actor, action, detail, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
                               (rid, ts, actor, action, d, self.prev, h))
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps({"ts": ts, "actor": actor, "action": action, "detail": d, "prev": self.prev, "hash": h}) + "\n")
            self.prev = h
            return h

    def recent(self, n: int = 50) -> list[dict]:
        return self.store.query("SELECT ts, actor, action, detail, hash FROM audit ORDER BY ts DESC LIMIT ?", (n,))

    def verify(self) -> dict:
        rows = self.store.query("SELECT ts, actor, action, detail, prev_hash, hash FROM audit ORDER BY ts ASC, id ASC")
        prev = "GENESIS"
        for i, r in enumerate(rows):
            if r["prev_hash"] != prev or self._h(prev, r["ts"], r["actor"], r["action"], r["detail"]) != r["hash"]:
                return {"valid": False, "broken_at": i, "records": len(rows)}
            prev = r["hash"]
        return {"valid": True, "records": len(rows)}
