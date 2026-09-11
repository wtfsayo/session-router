"""Session state store — in-memory with optional sqlite persistence."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Optional

from .types import ModelCacheState, SessionPhase, SessionState


class SessionStore:
    def __init__(self, path: Optional[str] = None, ttl_seconds: float = 86400):
        self._mem: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self.ttl = ttl_seconds
        self._db: Optional[sqlite3.Connection] = None
        if path:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS sessions "
                "(id TEXT PRIMARY KEY, data TEXT, updated REAL)")

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            s = self._mem.get(session_id)
            if s is None and self._db is not None:
                row = self._db.execute(
                    "SELECT data FROM sessions WHERE id=?",
                    (session_id,)).fetchone()
                if row:
                    s = self._deserialize(session_id, json.loads(row[0]))
            if s is None:
                s = SessionState(session_id=session_id)
            self._mem[session_id] = s
            return s

    def put(self, s: SessionState) -> None:
        with self._lock:
            self._mem[s.session_id] = s
            if self._db is not None:
                self._db.execute(
                    "INSERT OR REPLACE INTO sessions VALUES (?,?,?)",
                    (s.session_id, json.dumps(self._serialize(s)), time.time()))
                self._db.commit()

    def gc(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        with self._lock:
            dead = [k for k, s in self._mem.items()
                    if now - s.last_active_ts > self.ttl]
            for k in dead:
                del self._mem[k]
            if self._db is not None and dead:
                self._db.executemany("DELETE FROM sessions WHERE id=?",
                                     [(k,) for k in dead])
                self._db.commit()
            return len(dead)

    @staticmethod
    def _serialize(s: SessionState) -> dict:
        return {
            "incumbent_model": s.incumbent_model,
            "turn_count": s.turn_count,
            "history_tokens_est": s.history_tokens_est,
            "phase": s.phase.value,
            "switch_count": s.switch_count,
            "last_category": s.last_category,
            "escalated": s.escalated,
            "created_ts": s.created_ts,
            "last_active_ts": s.last_active_ts,
            "cache": {k: vars(v) for k, v in s.cache.items()},
            "inter_arrival_gaps": s.inter_arrival_gaps[-64:],
            "pii_restore_map": s.pii_restore_map,
        }

    @staticmethod
    def _deserialize(sid: str, d: dict) -> SessionState:
        s = SessionState(session_id=sid)
        s.incumbent_model = d.get("incumbent_model")
        s.turn_count = d.get("turn_count", 0)
        s.history_tokens_est = d.get("history_tokens_est", 0)
        s.phase = SessionPhase(d.get("phase", "normal"))
        s.switch_count = d.get("switch_count", 0)
        s.last_category = d.get("last_category")
        s.escalated = d.get("escalated", False)
        s.created_ts = d.get("created_ts", time.time())
        s.last_active_ts = d.get("last_active_ts", time.time())
        s.cache = {k: ModelCacheState(**v) for k, v in d.get("cache", {}).items()}
        s.inter_arrival_gaps = d.get("inter_arrival_gaps", [])
        s.pii_restore_map = d.get("pii_restore_map", {})
        return s
