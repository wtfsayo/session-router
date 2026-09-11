"""Core types for the session-aware router."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class PrivacyTier(str, Enum):
    """How a request may be treated w.r.t. cloud egress.

    STRICT: may never leave the box — hard gate, router doesn't even see cloud.
    REDACT: PII is detected and substituted with placeholders before cloud egress;
            the restore map never leaves local state.
    OPEN:   no PII detected (or explicitly waived) — free to route anywhere.
    """
    STRICT = "strict"
    REDACT = "redact"
    OPEN = "open"


class SessionPhase(str, Enum):
    NORMAL = "normal"
    TOOL_LOOP = "tool_loop"          # a tool call is pending its result
    PROVIDER_STATE = "provider_state"  # carries non-portable provider continuation state
    IDLE_RESET = "idle_reset"        # idled past boundary — reselection allowed
    DRIFT_RESET = "drift_reset"      # task category changed — reselection allowed


@dataclass
class Message:
    role: str                      # system | user | assistant | tool
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclass
class Pricing:
    """Per-model pricing, $ per 1M tokens. See cache economics notes:
    write_mult ~1.25 (Anthropic), read_mult ~0.1."""
    input_per_mtok: float
    output_per_mtok: float = 0.0
    cached_input_per_mtok: Optional[float] = None   # read price; None = no cache discount
    cache_write_per_mtok: Optional[float] = None    # write price; None = free/automatic
    cache_ttl_seconds: float = 300.0                # sliding TTL (Anthropic-style)
    cache_min_tokens: int = 1024                    # minimum cacheable prefix
    cache_ttl_hard_cap: Optional[float] = None      # OpenAI-style ~1h cap; None = pure sliding
    requires_sticky_key: bool = False               # needs session key to hit warm node


@dataclass
class ModelSpec:
    name: str
    tier: str                        # "local" | "cloud"
    backend: str                     # backend adapter name: "ollama" | "llamacpp" | "openai_compat" | "anthropic"
    pricing: Pricing
    quality: float = 1.0             # expected quality in [0,1] for this pool
    max_context: int = 8192
    cost_per_mtok_estimate: float = 0.0  # shorthand for quick comparisons
    extra: dict = field(default_factory=dict)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0     # observed cache reads (provider-reported)
    cache_write_tokens: int = 0      # observed cache writes


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top_logprobs: list = field(default_factory=list)  # [(token, logprob)]


@dataclass
class BackendResponse:
    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    logprobs: list = field(default_factory=list)       # per-token, if requested
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


@dataclass
class ModelCacheState:
    """Per-(session, model) cache bookkeeping."""
    last_seen_ts: float = 0.0
    cached_prefix_tokens: int = 0    # longest prefix known warm on this model
    observed_hit: bool = False       # did the last turn report a cache hit


@dataclass
class SessionState:
    session_id: str
    incumbent_model: Optional[str] = None
    turn_count: int = 0
    history_tokens_est: int = 0
    phase: SessionPhase = SessionPhase.NORMAL
    switch_count: int = 0
    last_category: Optional[str] = None
    escalated: bool = False          # sticky: once escalated to cloud, stay until reset
    created_ts: float = field(default_factory=time.time)
    last_active_ts: float = field(default_factory=time.time)
    cache: dict = field(default_factory=dict)          # model -> ModelCacheState
    inter_arrival_gaps: list = field(default_factory=list)  # seconds between session turns
    pii_restore_map: dict = field(default_factory=dict)     # placeholder -> original (local only)
    metadata: dict = field(default_factory=dict)


@dataclass
class RouteDecision:
    model: str
    reason: str
    privacy_tier: PrivacyTier
    scores: dict = field(default_factory=dict)         # candidate -> adjusted score
    switch_cost: float = 0.0
    stayed: bool = False
    escalated_after_reject: bool = False
    pii_findings: list = field(default_factory=list)
    gate: dict = field(default_factory=dict)           # cascade-gate diagnostics
    sanitized: bool = False
    trace_id: str = ""


class BackendError(Exception):
    pass


class PrivacyBlock(BackendError):
    """Raised when a STRICT request would have to leave the box."""
    pass
