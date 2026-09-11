"""Local backends: Ollama and llama.cpp server.

KV/session notes:
- Ollama: keep_alive controls model residency; internal prefix reuse while
  runner is alive. No disk KV persistence -> time stickiness to keep_alive TTL.
- llama.cpp server: --slot-save-path enables /slots/{id}/save|restore —
  the best primitive for holding a session warm while it's escalated.
  We pass id_slot + cache_prompt when configured.
"""
from __future__ import annotations

from ..types import (BackendResponse, Message, ModelSpec, TokenLogprob,
                     TokenUsage)
from .base import post_json, to_openai_messages


class OllamaBackend:
    def __init__(self, base_url: str = "http://localhost:11434",
                 keep_alive: str = "10m"):
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive

    def generate(self, spec: ModelSpec, messages: list[Message], *,
                 want_logprobs: bool = False, session_key: str = "",
                 max_tokens: int = 1024) -> BackendResponse:
        payload = {
            "model": spec.name,
            "messages": to_openai_messages(messages),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": max_tokens},
        }
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        r = post_json(f"{self.base_url}/api/chat", payload)
        msg = r.get("message", {})
        lps = [TokenLogprob(t.get("token", ""), t.get("logprob", 0.0),
                            [(x.get("token", ""), x.get("logprob", 0.0))
                             for x in t.get("top_logprobs", [])])
               for t in (msg.get("logprobs") or [])]
        return BackendResponse(
            text=msg.get("content", ""), model=spec.name,
            usage=TokenUsage(
                input_tokens=r.get("prompt_eval_count", 0),
                output_tokens=r.get("eval_count", 0),
                # Ollama doesn't report cache hits; infer from prompt_eval
                # vs expected history if needed upstream
            ),
            logprobs=lps,
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=r.get("done_reason", "stop"),
            raw=r)


class LlamaCppBackend:
    """llama.cpp llama-server (OpenAI-compatible + slot persistence)."""

    def __init__(self, base_url: str = "http://localhost:8080",
                 slot_id: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.slot_id = slot_id

    def generate(self, spec: ModelSpec, messages: list[Message], *,
                 want_logprobs: bool = False, session_key: str = "",
                 max_tokens: int = 1024) -> BackendResponse:
        payload = {
            "messages": to_openai_messages(messages),
            "max_tokens": max_tokens,
            "cache_prompt": True,
        }
        if self.slot_id is not None:
            payload["id_slot"] = self.slot_id
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        r = post_json(f"{self.base_url}/v1/chat/completions", payload)
        ch = (r.get("choices") or [{}])[0]
        lps = []
        for t in ((ch.get("logprobs") or {}).get("content") or []):
            lps.append(TokenLogprob(
                t.get("token", ""), t.get("logprob", 0.0),
                [(x.get("token", ""), x.get("logprob", 0.0))
                 for x in t.get("top_logprobs", [])]))
        u = r.get("usage") or {}
        return BackendResponse(
            text=(ch.get("message") or {}).get("content", ""), model=spec.name,
            usage=TokenUsage(u.get("prompt_tokens", 0),
                             u.get("completion_tokens", 0),
                             (u.get("prompt_tokens_details") or {})
                             .get("cached_tokens", 0)),
            logprobs=lps,
            tool_calls=(ch.get("message") or {}).get("tool_calls") or [],
            finish_reason=ch.get("finish_reason", "stop"), raw=r)

    def save_slot(self, slot_id: int, path: str) -> None:
        post_json(f"{self.base_url}/slots/{slot_id}",
                  {"action": "save", "filename": path})

    def restore_slot(self, slot_id: int, path: str) -> None:
        post_json(f"{self.base_url}/slots/{slot_id}",
                  {"action": "restore", "filename": path})
