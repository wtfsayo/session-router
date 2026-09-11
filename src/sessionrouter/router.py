"""Router orchestrator — wires privacy gate, session policy, backends,
cascade gate, cache bookkeeping, and tracing into one handle() call.

Flow per turn:
  1. session lookup + inter-arrival bookkeeping
  2. PII scan on new user text -> privacy tier (STRICT | REDACT | OPEN)
  3. complexity score -> per-model quality deficit
  4. policy.decide(): locks -> resets -> sticky escalation -> margin/WFA
  5. dispatch (sanitized payload if REDACT; local slot/session key)
  6. cascade gate on local responses -> escalate on failure (unless STRICT)
  7. update cache warmth + session state; de-sanitize; trace
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from . import gate as gatemod
from . import pii as piimod
from . import policy as policymod
from .cache import p_hit
from .features import categorize, default_scorer
from .session import SessionStore
from .trace import TraceLog
from .types import (BackendResponse, Message, ModelCacheState, ModelSpec,
                    PrivacyTier, RouteDecision, SessionPhase)


class Router:
    def __init__(self, models: list[ModelSpec], backends: dict,
                 cfg: policymod.PolicyConfig | None = None,
                 store: SessionStore | None = None,
                 tracer: TraceLog | None = None,
                 scorer=None,
                 strict_pii_kinds: set[str] | None = None,
                 gate_cfg: gatemod.GateConfig | None = None,
                 enable_gate: bool = True,
                 enable_pii: bool = True):
        self.models = {m.name: m for m in models}
        self.backends = backends               # backend_name -> Backend
        self.cfg = cfg or policymod.PolicyConfig()
        self.store = store or SessionStore()
        self.tracer = tracer
        self.scorer = scorer or default_scorer()
        self.sanitizer = piimod.Sanitizer()
        self.strict_pii_kinds = strict_pii_kinds
        self.gate_cfg = gate_cfg or gatemod.GateConfig()
        self.enable_gate = enable_gate
        self.enable_pii = enable_pii

    # -- main entry ---------------------------------------------------------

    def handle(self, messages: list[Message], session_id: str = "",
               wants_json: bool = False, wants_tools: bool = False
               ) -> tuple[BackendResponse, RouteDecision]:
        session_id = session_id or uuid.uuid4().hex
        now = time.time()
        session = self.store.get(session_id)

        # inter-arrival bookkeeping (feeds keep-alive hazard model)
        if session.turn_count > 0:
            gap = now - session.last_active_ts
            if gap > 0:
                session.inter_arrival_gaps.append(gap)
        # note: last_active_ts updated AFTER decide() so idle detection works

        # ---- privacy scan on new user content ----
        if self.enable_pii:
            new_user_text = " ".join(
                m.content for m in messages if m.role == "user")
            findings = piimod.detect(new_user_text)
            tier = piimod.classify_tier(findings, self.strict_pii_kinds)
        else:
            findings, tier = [], PrivacyTier.OPEN

        # sanitize the egress payload once; placeholders are session-stable
        if tier != PrivacyTier.OPEN:
            egress = piimod.sanitize_messages(self.sanitizer, session_id,
                                            messages)
        else:
            egress = messages

        # ---- score & decide ----
        last_user = next((m.content for m in reversed(messages)
                          if m.role == "user"), "")
        category = categorize(last_user)
        p_local = self.scorer.local_sufficiency(messages, category)
        tier_suff = getattr(self.scorer, "tier_sufficiency", None)
        if tier_suff is not None:
            # multi-tier scorer: quality_deficit[m] = P(need stronger than m's tier)
            probs = tier_suff(messages)
            qd = {m.name: 1.0 - probs[m.extra.get("suff_tier", 0)]
                  for m in self.models.values()}
        else:
            # binary: local models carry (1 - p_local); cloud residual small
            qd = {m.name: ((1.0 - p_local) if m.tier == "local"
                           else (1.0 - p_local) * 0.15)
                  for m in self.models.values()}

        decision = policymod.decide(
            session, messages, list(self.models.values()), qd, tier,
            self.cfg, now=now, pii_findings=findings, category=category)

        spec = self.models[decision.model]
        resp = self._call(spec, egress, session_id,
                          want_logprobs=(spec.tier == "local"))
        decision.gate["category"] = category
        decision.gate["p_local"] = round(p_local, 3)

        # ---- cascade gate on local responses ----
        if (self.enable_gate and spec.tier == "local"
                and session.phase != SessionPhase.TOOL_LOOP):
            gc = gatemod.GateConfig(expects_json=wants_json,
                                    expects_tool_call=wants_tools,
                                    **{k: v for k, v in vars(self.gate_cfg).items()
                                       if k in ("mean_logprob_floor",
                                                "min_logprob_tripwire",
                                                "min_answer_chars",
                                                "max_answer_chars",
                                                "max_repetition")})
            verdict, gdiag = gatemod.check(resp, messages, gc)
            decision.gate.update(gdiag)
            if verdict == gatemod.Verdict.ESCALATE:
                cloud = self._pick_cloud(session, feasible_tier=tier)
                if cloud is not None:
                    resp2 = self._call(cloud, egress, session_id)
                    session.escalated = True
                    session.incumbent_model = cloud.name
                    session.switch_count += 1
                    decision.escalated_after_reject = True
                    decision.model = cloud.name
                    decision.reason += "+escalated_on_gate"
                    spec, resp = cloud, resp2
                else:
                    decision.gate["blocked"] = "privacy"

        # ---- bookkeeping ----
        session.last_active_ts = now
        session.turn_count += 1
        session.history_tokens_est += (resp.usage.input_tokens
                                       + resp.usage.output_tokens)
        self._update_cache_state(session, spec, resp, now)
        if resp.tool_calls:
            session.phase = SessionPhase.TOOL_LOOP
        elif session.phase in (SessionPhase.TOOL_LOOP,
                               SessionPhase.IDLE_RESET,
                               SessionPhase.DRIFT_RESET):
            session.phase = SessionPhase.NORMAL

        if decision.sanitized or tier != PrivacyTier.OPEN:
            resp.text = self.sanitizer.de_sanitize(session_id, resp.text)
            decision.sanitized = tier != PrivacyTier.OPEN

        self.store.put(session)
        if self.tracer:
            decision.trace_id = self.tracer.log(
                session_id, session.turn_count, decision, resp.usage)
        return resp, decision

    # -- internals ----------------------------------------------------------

    def _call(self, spec: ModelSpec, messages: list[Message],
              session_key: str, want_logprobs: bool = False) -> BackendResponse:
        backend = self.backends[spec.backend]
        return backend.generate(spec, messages, want_logprobs=want_logprobs,
                                session_key=session_key)

    def _pick_cloud(self, session, feasible_tier: PrivacyTier) -> Optional[ModelSpec]:
        """Best cloud model to escalate to. None when STRICT."""
        if feasible_tier == PrivacyTier.STRICT:
            return None
        clouds = [m for m in self.models.values() if m.tier == "cloud"]
        if not clouds:
            return None
        # prefer incumbent if already cloud (keep the warm cache)
        inc = session.incumbent_model
        if inc in {m.name for m in clouds}:
            return self.models[inc]
        return max(clouds, key=lambda m: m.quality)

    def _update_cache_state(self, session, spec: ModelSpec,
                            resp: BackendResponse, now: float) -> None:
        st = session.cache.setdefault(spec.name, ModelCacheState())
        st.last_seen_ts = now
        hit = resp.usage.cached_input_tokens > 0
        st.observed_hit = hit
        st.cached_prefix_tokens = (
            resp.usage.cached_input_tokens + resp.usage.cache_write_tokens
            if (hit or resp.usage.cache_write_tokens)
            else max(0, resp.usage.input_tokens))
