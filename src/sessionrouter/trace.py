"""Decision trace log — sqlite. Schema follows the router-replay pattern:
enough per-decision state to shadow-replay a different policy later
(incl. candidate scores so a soft policy can be reconstructed)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid


class TraceLog:
    def __init__(self, path: str = "router_trace.db"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                trace_id TEXT PRIMARY KEY,
                ts REAL, session_id TEXT, turn INTEGER,
                category TEXT, privacy_tier TEXT,
                chosen TEXT, reason TEXT, stayed INTEGER,
                scores_json TEXT, switch_cost REAL,
                gate_json TEXT, pii_kinds TEXT,
                usage_json TEXT, latency_ms REAL
            )""")

    def log(self, session_id: str, turn: int, decision, usage=None,
            latency_ms: float = 0.0) -> str:
        tid = uuid.uuid4().hex[:16]
        with self._lock:
            self._db.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tid, time.time(), session_id, turn, decision.scores and
                 decision.gate.get("category", ""), decision.privacy_tier.value,
                 decision.model, decision.reason, int(decision.stayed),
                 json.dumps(decision.scores), decision.switch_cost,
                 json.dumps(decision.gate),
                 ",".join(sorted({f.kind for f in decision.pii_findings})),
                 json.dumps(vars(usage)) if usage else "{}",
                 latency_ms))
            self._db.commit()
        return tid
