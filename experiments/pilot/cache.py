"""
Pilot Cache
===========

Persistent SQLite cache of API calls, keyed by a hash of the endpoint and request body.
A rerun with the same requests reads every response from here and spends nothing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def request_key(endpoint: str, body: dict[str, Any]) -> str:
    """Stable hash of an endpoint and request body (key order does not matter)."""
    canonical = json.dumps(
        {"endpoint": endpoint, "body": body}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class CachedCall:
    """A stored API call."""

    key: str
    endpoint: str
    model: str
    response: dict[str, Any]
    latency_ms: float
    cost_usd: float
    created_at: str


class CallCache:
    """SQLite-backed cache. Safe to share between coroutines in one process."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                key TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                model TEXT NOT NULL,
                request TEXT NOT NULL,
                response TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                cost_usd REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> CachedCall | None:
        """Return the stored call for key, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT key, endpoint, model, response, latency_ms, cost_usd, created_at "
                "FROM calls WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return CachedCall(
            key=row[0],
            endpoint=row[1],
            model=row[2],
            response=json.loads(row[3]),
            latency_ms=row[4],
            cost_usd=row[5],
            created_at=row[6],
        )

    def put(
        self,
        key: str,
        endpoint: str,
        model: str,
        request: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Store a completed call."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO calls "
                "(key, endpoint, model, request, response, latency_ms, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    endpoint,
                    model,
                    json.dumps(request, sort_keys=True),
                    json.dumps(response),
                    latency_ms,
                    cost_usd,
                ),
            )
            self._conn.commit()

    def total_cost(self) -> float:
        """Sum of the cost of every stored call: all live spend recorded in this file."""
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls").fetchone()
        return float(row[0])

    def __len__(self) -> int:
        """Number of stored calls."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0])

    def close(self) -> None:
        """Close the database."""
        with self._lock:
            self._conn.close()
