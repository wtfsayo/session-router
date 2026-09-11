"""Simulated end-to-end run — exercises privacy gate, escalation,
stickiness, idle reset, and cache bookkeeping without live models."""
from __future__ import annotations

import time

from .policy import PolicyConfig
from .router import Router
from .session import SessionStore
from .types import (BackendResponse, Message, ModelSpec, Pricing,
                    TokenLogprob, TokenUsage)


class FakeLocal:
    """Weak local model: answers trivially, gets 'hard' things wrong,
    reports low confidence on them via logprobs."""

    def generate(self, spec, messages, *, want_logprobs=False,
                 session_key="", max_tokens=1024):
        last = messages[-1].content.lower()
        if "prove" in last or "optimize" in last:
            text = "I think the answer is probably 42."
            lps = [TokenLogprob(t, -0.15) for t in text.split()[:6]] + \
                  [TokenLogprob(t, -3.5) for t in text.split()[6:]]
        else:
            text = f"local answer to: {messages[-1].content[:40]}"
            lps = [TokenLogprob(t, -0.2) for t in text.split()]
        toks_in = sum(len(m.content) for m in messages) // 4
        return BackendResponse(
            text=text, model=spec.name, logprobs=lps,
            usage=TokenUsage(toks_in, len(text) // 4),
            finish_reason="stop")


class FakeCloud:
    """Cloud model: reports cache hits once it's seen the session prefix."""

    def __init__(self):
        self.seen: set[str] = set()
        self.seen_texts: list[str] = []

    def generate(self, spec, messages, *, want_logprobs=False,
                 session_key="", max_tokens=1024):
        toks_in = sum(len(m.content) for m in messages) // 4
        self.seen_texts.append(messages[-1].content)
        cached = 0
        if session_key in self.seen and toks_in > spec.pricing.cache_min_tokens:
            cached = toks_in - 64  # all but the newest turn
        self.seen.add(session_key)
        return BackendResponse(
            text="detailed cloud answer with reasoning", model=spec.name,
            usage=TokenUsage(toks_in, 180, cached_input_tokens=cached),
            finish_reason="stop")


def run_demo() -> None:
    models = [
        ModelSpec("qwen3-4b", tier="local", backend="local",
                  pricing=Pricing(0.0, 0.0), quality=0.7),
        ModelSpec("claude-sonnet", tier="cloud", backend="cloud",
                  pricing=Pricing(3.0, 15.0,
                                  cached_input_per_mtok=0.30,
                                  cache_write_per_mtok=3.75,
                                  cache_ttl_seconds=300),
                  quality=0.97),
    ]
    cloud = FakeCloud()
    r = Router(models, {"local": FakeLocal(), "cloud": cloud},
               cfg=PolicyConfig(quality_weight=0.3, switch_margin=0.005),
               store=SessionStore())
    sid = "demo"

    # long system prompt -> history clears the provider's cache floor,
    # so cache reads/writes are observable on later turns
    history: list[Message] = [
        Message("system", "You are an assistant. " + "Policy text. " * 500)]

    turns = [
        ("What is the capital of France?", "easy -> local"),
        ("Prove that the square root of 2 is irrational, step by step.",
         "hard -> local tries, gate escalates to cloud"),
        ("Thanks! And what's 2+2?", "easy but stays cloud — sticky escalation"),
        ("My card is 4242 4242 4242 4242, store it.",
         "PII STRICT -> forced local"),
        ("What is the largest ocean?",
         "easy, but transcript holds a card -> still STRICT local"),
    ]
    for text, note in turns:
        history.append(Message(role="user", content=text))
        resp, dec = r.handle(list(history), session_id=sid)
        history.append(Message(role="assistant", content=resp.text))
        s = r.store.get(sid)
        print(f"\n--- {note}")
        print(f"  tier={dec.privacy_tier.value} model={dec.model} "
              f"reason={dec.reason} sanitized={dec.sanitized} "
              f"esc_reject={dec.escalated_after_reject}")
        print(f"  scores={ {k: round(v,4) for k,v in dec.scores.items()} }")
        print(f"  gate={dec.gate}")
        print(f"  answer={resp.text[:60]!r} cached_in={resp.usage.cached_input_tokens}")

    print("\n--- cloud actually saw (last user msg per call):")
    for t in cloud.seen_texts:
        print(f"  {t[:70]!r}")


if __name__ == "__main__":
    run_demo()
