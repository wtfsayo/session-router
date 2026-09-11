"""Replay backend: serves precomputed RouterBench outcomes so the real
Router pipeline (privacy -> policy -> gate -> cache bookkeeping) runs
against benchmark data with no live inference."""
from __future__ import annotations

from sessionrouter.types import (BackendResponse, Message, ModelSpec,
                                 TokenLogprob, TokenUsage)


def synth_logprobs(score: float, n: int = 24):
    """Synthetic token logprobs consistent with a quality outcome:
    correct -> confident tail; wrong -> low-confidence answer span."""
    if score >= 0.99:
        return [TokenLogprob("t", -0.2) for _ in range(n)]
    if score <= 0.01:
        return [TokenLogprob("t", -0.1)] * (n - 6) + \
               [TokenLogprob("t", -6.5)] * 6
    return [TokenLogprob("t", -0.4)] * (n - 4) + \
           [TokenLogprob("t", -1.6)] * 4


class ReplayBackend:
    """Backend serving recorded (response, score) per prompt per side."""

    def __init__(self, outcomes: dict, out_tokens: int = 200):
        # outcomes[prompt_text] = {tier: (text, score)}
        self.outcomes = outcomes
        self.out_tokens = out_tokens

    def generate(self, spec: ModelSpec, messages: list[Message], *,
                 want_logprobs=False, session_key="", max_tokens=1024):
        last = messages[-1].content
        row = self.outcomes.get(last)
        if row is None:
            return BackendResponse(text="", model=spec.name,
                                   usage=TokenUsage(0, 0), finish_reason="stop")
        text, score = row[spec.tier]
        in_toks = sum(len(m.content) for m in messages) // 4
        return BackendResponse(
            text=text, model=spec.name,
            usage=TokenUsage(in_toks, self.out_tokens),
            logprobs=synth_logprobs(score) if want_logprobs else [],
            finish_reason="stop")


class CostBook:
    """Simulator-side cost accounting: provider-style cache pricing applied
    to append-only session transcripts."""

    def __init__(self, in_price, out_price, write_price, read_price,
                 ttl_s=300.0, min_prefix=1024):
        self.in_p, self.out_p = in_price / 1e6, out_price / 1e6
        self.write_p, self.read_p = write_price / 1e6, read_price / 1e6
        self.ttl, self.min_prefix = ttl_s, min_prefix
        self.warm: dict = {}   # (session, model) -> (last_ts, prefix_tokens)

    def charge(self, session_id: str, model: str, prefix_tokens: int,
               in_tokens: int, out_tokens: int, now: float) -> dict:
        """Append-only transcript: warm prefix = min(previous, current)."""
        key = (session_id, model)
        warm = 0
        prev = self.warm.get(key)
        if prev and now - prev[0] <= self.ttl:
            warm = min(prev[1], prefix_tokens)
            if warm < self.min_prefix:
                warm = 0
        cold = in_tokens - warm
        cost = cold * self.write_p + warm * self.read_p + out_tokens * self.out_p
        self.warm[key] = (now, prefix_tokens + in_tokens)
        return {"cost": cost, "warm": warm, "cold": cold}
