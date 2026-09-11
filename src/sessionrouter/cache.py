"""Prompt-cache economics: warmth model, hit probability, switch cost.

Derived from the provider survey (see README):
- Anthropic/Bedrock: sliding TTL, refreshed on hit. Deterministic inside an
  active session, dead after ~TTL idle.
- OpenAI: machine-local in-memory, ~5-10min idle decay, hard cap ~1h from last
  use; needs prompt_cache_key for node stickiness.
- DeepSeek: hours-to-days disk cache.
- Local engines: free but memory-bound; llama.cpp slot save/restore, Ollama
  keep_alive.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .types import ModelCacheState, ModelSpec, Pricing


def cache_checkout_cost(p: Pricing) -> float:
    """The '$/Mtok delta' between a cold write and a warm read — SAAR's
    'cached-input checkout' term. 0 when no cache pricing exists."""
    if p.cache_write_per_mtok is None:
        return p.input_per_mtok
    return max(p.cache_write_per_mtok - (p.cached_input_per_mtok or 0.0), 0.0)


def cost_pressure(spec: ModelSpec, max_cost: float) -> float:
    """Normalize a model's checkout delta to [0,1] against the pool."""
    if max_cost <= 0:
        return 0.5
    c = cache_checkout_cost(spec.pricing)
    return min(c / max_cost, 1.0) if c > 0 else 0.5


def p_alive(p: Pricing, dt_seconds: float, observed_hit: bool) -> float:
    """Probability the provider still holds our prefix.

    Sliding-TTL providers (Anthropic, Bedrock): alive iff dt < TTL (hits refresh).
    Hard-cap providers (OpenAI): decay from TTL to hard cap.
    Best-effort providers (long TTL): slow decay.
    """
    if dt_seconds < 0:
        return 0.0
    if p.cache_ttl_hard_cap is None:
        # sliding TTL — deterministic while active
        if dt_seconds <= p.cache_ttl_seconds:
            return 1.0 if observed_hit or p.cache_ttl_seconds >= 300 else 0.9
        return 0.0
    # hard-cap style: certain until ttl, then linear decay to cap
    if dt_seconds <= p.cache_ttl_seconds:
        return 1.0
    if dt_seconds >= p.cache_ttl_hard_cap:
        return 0.0
    frac = (dt_seconds - p.cache_ttl_seconds) / (p.cache_ttl_hard_cap - p.cache_ttl_seconds)
    return max(0.0, 1.0 - frac)


def p_hit(
    spec: ModelSpec,
    cache_state: ModelCacheState | None,
    now: float,
    prefix_stable: bool = True,
    prefix_tokens: int = 0,
) -> float:
    """P(provider cache hit for this session on this model next turn)."""
    if prefix_tokens < spec.pricing.cache_min_tokens:
        return 0.0
    if not prefix_stable:
        return 0.0
    if cache_state is None or cache_state.cached_prefix_tokens <= 0:
        return 0.0
    dt = now - cache_state.last_seen_ts
    return p_alive(spec.pricing, dt, cache_state.observed_hit)


def warm_tokens(spec: ModelSpec, state: ModelCacheState | None, now: float) -> int:
    """Tokens we expect to come back as cache-read if we stay on this model."""
    if state is None:
        return 0
    if p_hit(spec, state, now, prefix_tokens=state.cached_prefix_tokens) > 0.5:
        return state.cached_prefix_tokens
    return 0


def turn_cost(spec: ModelSpec, history_tokens: int, new_tokens: int,
              est_output_tokens: int, cache_state: ModelCacheState | None,
              now: float) -> float:
    """Expected $ cost of serving the next turn on this model."""
    p = spec.pricing
    warm = min(history_tokens, warm_tokens(spec, cache_state, now))
    cold = history_tokens - warm
    write_price = p.cache_write_per_mtok or p.input_per_mtok
    read_price = p.cached_input_per_mtok if p.cached_input_per_mtok is not None else p.input_per_mtok
    return (cold * write_price + warm * read_price + new_tokens * p.input_per_mtok
            + est_output_tokens * p.output_per_mtok) / 1e6


def switch_cost(current: ModelSpec, candidate: ModelSpec,
                history_tokens: int,
                cand_state: ModelCacheState | None,
                cur_state: ModelCacheState | None,
                now: float) -> float:
    """Marginal $ cost of switching current -> candidate THIS turn vs staying.

    positive => switching pays a premium now.
    = H·W(B)·P(B)  (cold prefill at B, minus B's own warm portion)
      − H·R(A)·P(A) (the cheap read A would have gotten)
    """
    p_c, p_a = candidate.pricing, current.pricing
    cand_warm = min(history_tokens, warm_tokens(candidate, cand_state, now))
    cold_at_b = history_tokens - cand_warm
    write_b = p_c.cache_write_per_mtok or p_c.input_per_mtok
    cost_b = (cold_at_b * write_b + cand_warm *
              (p_c.cached_input_per_mtok or p_c.input_per_mtok)) / 1e6
    cur_warm = min(history_tokens, warm_tokens(current, cur_state, now))
    cost_a = (cur_warm * (p_a.cached_input_per_mtok or p_a.input_per_mtok)) / 1e6
    return cost_b - cost_a


def keepalive_breakeven_seconds(spec: ModelSpec, ping_tokens: int,
                                history_tokens: int) -> float:
    """Seconds of keep-alive pinging that equal one cold re-prefill.
    keep warm iff ping_cost < re_prefill_cost * P(return next window)."""
    p = spec.pricing
    read = p.cached_input_per_mtok or p.input_per_mtok
    write = p.cache_write_per_mtok or p.input_per_mtok
    ping = (history_tokens * read + ping_tokens * p.input_per_mtok) / 1e6
    reprefill = history_tokens * write / 1e6
    if ping <= 0:
        return float("inf")
    # number of TTL windows we can afford = reprefill/ping, times window length
    return (reprefill / ping) * p.cache_ttl_seconds
