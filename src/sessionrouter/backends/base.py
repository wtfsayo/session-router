from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from ..types import BackendResponse, Message, ModelSpec


class Backend(Protocol):
    def generate(self, spec: ModelSpec, messages: list[Message],
                 *, want_logprobs: bool = False, session_key: str = "",
                 max_tokens: int = 1024) -> BackendResponse:
        ...


def post_json(url: str, payload: dict, headers: dict | None = None,
              timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def to_openai_messages(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        d: dict = {"role": m.role, "content": m.content}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out
