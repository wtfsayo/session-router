"""Cloud backends: OpenAI-compatible and Anthropic.

Cache handling:
- OpenAI-compat: pass prompt_cache_key / user for node stickiness; read
  usage.prompt_tokens_details.cached_tokens. Works for OpenAI, OpenRouter,
  Together, Fireworks, vLLM endpoints.
- Anthropic: set a cache_control breakpoint at the last cacheable block
  (end of system or last-but-one user message — the stable prefix boundary).
  Read usage.cache_read_input_tokens / cache_creation_input_tokens.
"""
from __future__ import annotations

from ..types import (BackendResponse, Message, ModelSpec, TokenUsage)
from .base import post_json, to_openai_messages


class OpenAICompatBackend:
    def __init__(self, base_url: str = "https://api.openai.com/v1",
                 api_key: str = "", sticky_header: str = "prompt_cache_key"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.sticky_header = sticky_header

    def generate(self, spec: ModelSpec, messages: list[Message], *,
                 want_logprobs: bool = False, session_key: str = "",
                 max_tokens: int = 1024) -> BackendResponse:
        payload = {
            "model": spec.name,
            "messages": to_openai_messages(messages),
            "max_tokens": max_tokens,
        }
        if session_key:
            payload[self.sticky_header] = session_key
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = post_json(f"{self.base_url}/chat/completions", payload, headers)
        ch = (r.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        lps = [TokenLogprobLP(t) for t in
               ((ch.get("logprobs") or {}).get("content") or [])]
        u = r.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        return BackendResponse(
            text=msg.get("content", ""), model=spec.name,
            usage=TokenUsage(u.get("prompt_tokens", 0),
                             u.get("completion_tokens", 0),
                             det.get("cached_tokens", 0),
                             u.get("cache_write_tokens", 0)),
            logprobs=lps,
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=ch.get("finish_reason", "stop"), raw=r)


def TokenLogprobLP(t: dict):
    from ..types import TokenLogprob
    return TokenLogprob(t.get("token", ""), t.get("logprob", 0.0),
                        [(x.get("token", ""), x.get("logprob", 0.0))
                         for x in t.get("top_logprobs", [])])


class AnthropicBackend:
    """Anthropic Messages API with prompt-caching breakpoints."""

    def __init__(self, base_url: str = "https://api.anthropic.com",
                 api_key: str = "", cache_ttl: str = "5m"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cache_ttl = cache_ttl  # "5m" | "1h"

    def generate(self, spec: ModelSpec, messages: list[Message], *,
                 want_logprobs: bool = False, session_key: str = "",
                 max_tokens: int = 1024) -> BackendResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        conv = [m for m in messages if m.role != "system"]
        anth_msgs = [{"role": ("user" if m.role == "tool" else m.role),
                      "content": m.content} for m in conv]

        # stable-prefix boundary: cache system + all but the last user turn
        cache_break = max(0, len(anth_msgs) - 1)
        for i, m in enumerate(anth_msgs):
            block = {"type": "text", "text": m["content"]}
            if i == cache_break - 1 or (i == 0 and cache_break == 0):
                block["cache_control"] = {
                    "type": "ephemeral",
                    **({"ttl": "1h"} if self.cache_ttl == "1h" else {})}
            m["content"] = [block]

        payload: dict = {
            "model": spec.name, "max_tokens": max_tokens,
            "messages": anth_msgs,
        }
        if system_parts:
            payload["system"] = [
                {"type": "text", "text": "\n".join(system_parts),
                 "cache_control": {"type": "ephemeral",
                                   **({"ttl": "1h"} if self.cache_ttl == "1h" else {})}}]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "extended-cache-ttl-2025-04-11",
        }
        r = post_json(f"{self.base_url}/v1/messages", payload, headers)
        text = "".join(b.get("text", "") for b in r.get("content", [])
                       if b.get("type") == "text")
        u = r.get("usage") or {}
        return BackendResponse(
            text=text, model=spec.name,
            usage=TokenUsage(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                cached_input_tokens=u.get("cache_read_input_tokens", 0),
                cache_write_tokens=u.get("cache_creation_input_tokens", 0)),
            finish_reason=r.get("stop_reason", "stop"), raw=r)
