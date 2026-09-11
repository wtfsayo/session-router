"""OpenAI-compatible HTTP front-end (stdlib only).

POST /v1/chat/completions  — model:"auto" or a specific model name;
                             session via "metadata": {"session_id": ...}
                             or X-Session-Id header.
GET  /v1/router/session/{id}/keepalive — keep-warm advice per model
GET  /healthz
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .keepalive import KeepAlive
from .types import Message


def make_handler(router, keepalive: KeepAlive):
    class H(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                return self._json(200, {"ok": True})
            if self.path.startswith("/v1/router/session/") and \
                    self.path.endswith("/keepalive"):
                sid = self.path.split("/")[4]
                s = router.store.get(sid)
                out = {}
                for m in router.models.values():
                    keep, d = keepalive.should_keep_warm(s, m)
                    out[m.name] = {"keep_warm": keep, **d}
                return self._json(200, out)
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"])))
            except Exception:
                return self._json(400, {"error": "bad json"})
            msgs = [Message(role=m.get("role", "user"),
                            content=m.get("content") or "",
                            name=m.get("name"),
                            tool_call_id=m.get("tool_call_id"))
                    for m in body.get("messages", [])]
            sid = (body.get("metadata") or {}).get("session_id") \
                or self.headers.get("X-Session-Id", "")
            t0 = time.time()
            resp, dec = router.handle(
                msgs, session_id=sid,
                wants_json=(body.get("response_format") or {})
                .get("type") == "json_object",
                wants_tools=bool(body.get("tools")))
            out = {
                "id": f"chatcmpl-{dec.trace_id or 'x'}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": resp.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.text,
                                **({"tool_calls": resp.tool_calls}
                                   if resp.tool_calls else {})},
                    "finish_reason": resp.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": resp.usage.input_tokens,
                    "completion_tokens": resp.usage.output_tokens,
                    "cached_tokens": resp.usage.cached_input_tokens,
                },
                "_router": {
                    "model": dec.model, "reason": dec.reason,
                    "privacy_tier": dec.privacy_tier.value,
                    "sanitized": dec.sanitized,
                    "escalated_after_reject": dec.escalated_after_reject,
                    "scores": dec.scores, "gate": dec.gate,
                    "pii_kinds": sorted({f.kind for f in dec.pii_findings}),
                    "trace_id": dec.trace_id,
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                },
            }
            return self._json(200, out)

        def log_message(self, *a):  # quiet
            pass

    return H


def serve(router, host: str = "127.0.0.1", port: int = 8400) -> None:
    ka = KeepAlive()
    srv = ThreadingHTTPServer((host, port), make_handler(router, ka))
    print(f"session-router listening on http://{host}:{port}")
    srv.serve_forever()
