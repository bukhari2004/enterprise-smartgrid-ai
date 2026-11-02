"""Alert Center: rule engine over model outputs + live stream, with de-duplication, severities and persistence."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .storage import Store

ALERT_TYPES = ["TRANSFORMER_FAILURE_RISK", "POWER_OUTAGE", "GRID_OVERLOAD", "VOLTAGE_INSTABILITY", "ENERGY_THEFT",
               "RENEWABLE_ENERGY_DROP", "FREQUENCY_DEVIATION", "RESERVE_MARGIN_LOW"]
SEV_RANK = {"info": 0, "warning": 1, "high": 2, "critical": 3}


@dataclass
class Alert:
    type: str
    severity: str
    asset_id: str
    message: str
    region_id: str = ""
    evidence: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    status: str = "open"


class AlertCenter:
    def __init__(self, store: Store | None = None, cooldown_s: int = 900):
        self.store = store
        self.cooldown_s = cooldown_s
        self.alerts: list[Alert] = []
        self._last: dict[tuple[str, str], tuple[float, int]] = {}
        self.lock = threading.Lock()

    def emit(self, alert: Alert) -> Alert | None:
        """Returns None when suppressed (same type+asset inside the cooldown unless severity escalated)."""
        import time
        key = (alert.type, alert.asset_id); now = time.time(); rank = SEV_RANK[alert.severity]
        with self.lock:
            last = self._last.get(key)
            if last and now - last[0] < self.cooldown_s and rank <= last[1]:
                return None
            self._last[key] = (now, rank)
            self.alerts.append(alert)
            if len(self.alerts) > 5000:
                self.alerts = self.alerts[-5000:]
        if self.store:
            self.store.execute("INSERT INTO alerts (id, ts, type, severity, asset_id, region_id, message, evidence, status) "
                               "VALUES (?,?,?,?,?,?,?,?,?)", (alert.id, alert.ts, alert.type, alert.severity, alert.asset_id,
                                                              alert.region_id, alert.message, json.dumps(alert.evidence, default=str), alert.status))
        return alert

    def list(self, type: str | None = None, severity: str | None = None, status: str | None = "open", limit: int = 100) -> list[dict]:
        out = [a for a in reversed(self.alerts) if (not type or a.type == type) and (not severity or a.severity == severity)
               and (not status or a.status == status)]
        out.sort(key=lambda a: -SEV_RANK[a.severity])
        return [asdict(a) for a in out[:limit]]

    def acknowledge(self, alert_id: str) -> bool:
        for a in self.alerts:
            if a.id == alert_id:
                a.status = "acknowledged"
                if self.store:
                    self.store.execute("UPDATE alerts SET status=? WHERE id=?", ("acknowledged", alert_id))
                return True
        return False

    def counts(self) -> dict:
        c: dict = {}
        for a in self.alerts:
            if a.status == "open":
                c[a.type] = c.get(a.type, 0) + 1
        return c
