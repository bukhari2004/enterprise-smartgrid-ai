"""API-key authentication with role-based access control (viewer < operator < admin)."""
from __future__ import annotations

from .config import Settings

ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3}
DEV_KEYS = {"demo-viewer": "viewer", "demo-operator": "operator", "demo-admin": "admin"}


class Auth:
    def __init__(self, spec: str | None = None, env: str | None = None):
        spec = Settings.api_keys if spec is None else spec
        env = Settings.app_env if env is None else env
        self.keys = {}
        for part in filter(None, (p.strip() for p in spec.split(","))):
            k, _, role = part.partition(":")
            if role in ROLE_LEVEL:
                self.keys[k] = role
        if not self.keys and env == "dev":
            self.keys = dict(DEV_KEYS)                    # convenient defaults - NEVER use in production
        self.dev_defaults = self.keys == DEV_KEYS

    def role_for(self, key: str | None) -> str | None:
        return self.keys.get(key or "")

    def allowed(self, key: str | None, minimum: str) -> bool:
        r = self.role_for(key)
        return r is not None and ROLE_LEVEL[r] >= ROLE_LEVEL[minimum]
