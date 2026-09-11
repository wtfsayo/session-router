"""Keep-alive / cache-warmth decisions — hazard-rule on return gaps.

Keep a session's cache warm iff:
    ping_cost < reprefill_cost * P(session returns within next TTL window)

P(return) comes from the session's own inter-arrival-gap histogram,
falling back to a population prior when data is sparse (the
"Serverless in the Wild" recipe).
"""
from __future__ import annotations

import time

from .cache import keepalive_breakeven_seconds
from .types import ModelSpec, SessionState


class KeepAlive:
    def __init__(self, population_median_gap: float = 120.0,
                 min_samples: int = 4):
        self.population_median = population_median_gap
        self.min_samples = min_samples

    def p_return_within(self, session: SessionState, window_s: float) -> float:
        """P(next turn arrives within window_s), given elapsed idle time."""
        gaps = session.inter_arrival_gaps
        if len(gaps) < self.min_samples:
            # exponential prior with population median
            return 1.0 - pow(2.0, -window_s / max(self.population_median, 1.0))
        # empirical survival: fraction of observed gaps <= window
        return sum(1 for g in gaps if g <= window_s) / len(gaps)

    def should_keep_warm(self, session: SessionState, spec: ModelSpec,
                         now: float | None = None) -> tuple[bool, dict]:
        now = now or time.time()
        ttl = spec.pricing.cache_ttl_seconds
        p_ret = self.p_return_within(session, ttl)
        # cost of one keep-alive ping vs one cold re-prefill
        ping = ((session.history_tokens_est
                 * (spec.pricing.cached_input_per_mtok
                    or spec.pricing.input_per_mtok))
                / 1e6) or 1e-9
        reprefill = (session.history_tokens_est
                     * (spec.pricing.cache_write_per_mtok
                        or spec.pricing.input_per_mtok)) / 1e6
        keep = ping < reprefill * p_ret
        return keep, {"p_return": round(p_ret, 3), "ping_cost": ping,
                      "reprefill_cost": reprefill,
                      "breakeven_s": keepalive_breakeven_seconds(
                          spec, 32, session.history_tokens_est)}
