"""Session policy layer — the SAAR/WFA hybrid gate.

Order of checks (correctness before cost):
  1. Privacy hard gate   — STRICT tier restricts candidates to local pool
  2. Hard locks          — active tool loop or non-portable provider state
                           => hold incumbent unconditionally
  3. Reset boundaries    — idle timeout or category drift => free reselection
  4. Escalation stickiness — once escalated to cloud, stay until reset
  5. Decision rule       — "margin" (greedy + switch penalty, SAAR-style)
                           or "wfa" (work function, provably 2n-1 competitive)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import cache as cachemod
from .types import (Message, ModelSpec, PrivacyTier, RouteDecision,
                    SessionPhase, SessionState)
from .workfn import WorkFunction
from .features import category_group, is_observation_turn


@dataclass
class PolicyConfig:
    idle_reset_seconds: float = 1800.0       # idle session => free reselection
    switch_margin: float = 0.01              # $ challenger must beat incumbent by
    stay_bias: float = 0.0                   # extra $ credit to incumbent
    escalation_sticky: bool = True           # once on cloud, hold until reset
    switch_history_penalty: float = 0.002    # $ per recent switch (anti-oscillation)
    switch_history_window: int = 8
    # $ per unit of expected quality deficit. Calibrated: a cloud call runs
    # ~$0.003–0.03/turn, so values ≳0.2 collapse to always-cloud on
    # benchmark traffic; 0.05 keeps local for p_local ≳ 0.7.
    quality_weight: float = 0.05
    cloud_call_overhead: float = 0.003       # fixed $ proxy for network hop/TTFT
    est_output_tokens: int = 512
    decision_mode: str = "margin"            # "margin" | "wfa" | "satisfice"
    satisfice_tau: float = 0.7               # satisfice: lowest tier with P(suffice) >= tau
    agentic_hold: bool = True                # hold incumbent on observation turns


def detect_tool_loop(messages: list[Message]) -> bool:
    """A tool result just arrived or a tool call is outstanding."""
    if messages and messages[-1].role == "tool":
        return True
    for m in reversed(messages):
        if m.role == "assistant":
            return bool(m.metadata_tool_calls) if hasattr(m, "metadata_tool_calls") else False
        if m.role in ("user", "tool"):
            break
    return False


def effective_history_tokens(session: SessionState, messages: list[Message]) -> int:
    est = sum(len(m.content) for m in messages) // 4
    return max(est, session.history_tokens_est)


def decide(
    session: SessionState,
    messages: list[Message],
    candidates: list[ModelSpec],
    quality_deficit: dict[str, float],   # model -> [0,1]
    tier: PrivacyTier,
    cfg: PolicyConfig,
    now: float | None = None,
    pii_findings: list | None = None,
    category: str = "general",
) -> RouteDecision:
    now = now or time.time()
    history_tokens = effective_history_tokens(session, messages)
    new_tokens = max(1, len(messages[-1].content) // 4) if messages else 1

    # ---- privacy hard gate ----
    feasible = [m for m in candidates
                if not (tier == PrivacyTier.STRICT and m.tier == "cloud")]
    if not feasible:
        feasible = [m for m in candidates if m.tier == "local"]
    if not feasible:
        raise RuntimeError("no feasible model under privacy tier")

    incumbent = session.incumbent_model
    cur = next((m for m in candidates if m.name == incumbent), None)

    # ---- reset boundaries ----
    idle_expired = (now - session.last_active_ts) > cfg.idle_reset_seconds
    drift = (session.last_category is not None
             and category_group(category) != category_group(session.last_category)
             and session.last_category != "general")
    if idle_expired:
        session.phase = SessionPhase.IDLE_RESET
        session.escalated = False
    elif drift:
        session.phase = SessionPhase.DRIFT_RESET
        session.escalated = False
    escalated_idle_drift = idle_expired or drift
    last_user = next((m.content for m in reversed(messages)
                      if m.role == "user"), "")

    # ---- hard locks ----
    if cur is not None:
        # a genuinely outstanding tool call in a live session (phase set by
        # the router after a response carrying tool_calls) is non-portable.
        # A transcript merely *ending* on a tool result is portable — the
        # next call may legally switch — so it falls through to the softer
        # agentic hold below instead of a hard lock.
        if session.phase == SessionPhase.TOOL_LOOP:
            return RouteDecision(model=cur.name, reason="hard_lock:tool_loop",
                                 privacy_tier=tier, stayed=True,
                                 pii_findings=pii_findings or [])
        if session.phase == SessionPhase.PROVIDER_STATE:
            return RouteDecision(model=cur.name, reason="hard_lock:provider_state",
                                 privacy_tier=tier, stayed=True,
                                 pii_findings=pii_findings or [])
        # mid-agentic-loop steps whose transcripts encode observations as
        # user-role messages (no tool roles) — hold the incumbent; only
        # instruction boundaries get a fresh routing decision. In satisfice
        # mode the hold yields if the incumbent no longer clears tau —
        # under-routing wastes the whole chain.
        if cfg.agentic_hold and session.phase == SessionPhase.NORMAL \
                and not escalated_idle_drift \
                and is_observation_turn(messages, last_user):
            hold_ok = (cfg.decision_mode != "satisfice"
                       or quality_deficit.get(cur.name, 1.0)
                       <= 1.0 - cfg.satisfice_tau)
            if hold_ok:
                return RouteDecision(model=cur.name, reason="agentic:hold",
                                     privacy_tier=tier, stayed=True,
                                     pii_findings=pii_findings or [])
        # sticky escalation — same sufficiency guard in satisfice mode
        if (cfg.escalation_sticky and session.escalated
                and session.phase == SessionPhase.NORMAL
                and cur in feasible):
            hold_ok = (cfg.decision_mode != "satisfice"
                       or quality_deficit.get(cur.name, 1.0)
                       <= 1.0 - cfg.satisfice_tau)
            if hold_ok:
                return RouteDecision(model=cur.name, reason="sticky:escalated",
                                     privacy_tier=tier, stayed=True,
                                     pii_findings=pii_findings or [])
        if cur not in feasible:  # incumbent lost to privacy gate
            best_local = min(feasible, key=lambda m: m.pricing.input_per_mtok)
            session.incumbent_model = best_local.name
            session.switch_count += 1
            return RouteDecision(model=best_local.name,
                                 reason="privacy_gate:forced_local",
                                 privacy_tier=tier,
                                 pii_findings=pii_findings or [])

    # ---- candidate economics ----
    max_checkout = max((cachemod.cache_checkout_cost(m.pricing) for m in feasible),
                       default=0.0)
    scores: dict[str, float] = {}
    for m in feasible:
        st = session.cache.get(m.name)
        c = cachemod.turn_cost(m, history_tokens, new_tokens,
                               cfg.est_output_tokens, st, now)
        c += cfg.quality_weight * quality_deficit.get(m.name, 0.0)
        if m.tier == "cloud":
            c += cfg.cloud_call_overhead
        if cur is not None and m.name != cur.name:
            sw = cachemod.switch_cost(cur, m, history_tokens,
                                      st, session.cache.get(cur.name), now)
            penalty = 0.0 if (idle_expired or drift) else max(sw, 0.0)
            penalty += 0.0 if (idle_expired or drift) else (
                cfg.switch_history_penalty
                * min(session.switch_count, cfg.switch_history_window))
            penalty += cfg.switch_margin
            c += penalty
        elif cur is not None and m.name == cur.name:
            c -= cfg.stay_bias
        scores[m.name] = -c   # higher = better

    if cfg.decision_mode == "wfa" and len(feasible) > 1:
        # WFA over feasible models, state tracked per session
        wfa = session.metadata.setdefault("_wfa_state", None)
        idx = {m.name: i for i, m in enumerate(feasible)}
        if wfa is None or wfa.get("names") != [m.name for m in feasible]:
            wfa = {"wf": WorkFunction(len(feasible)),
                   "names": [m.name for m in feasible],
                   "state_idx": idx.get(incumbent, 0)}
            wfa["wf"].state = wfa["state_idx"]
            session.metadata["_wfa_state"] = wfa
        costs = [-scores[n] for n in wfa["names"]]
        dmat = [[0.0] * len(feasible) for _ in feasible]
        for i, mi in enumerate(feasible):
            for j, mj in enumerate(feasible):
                if i != j:
                    dmat[i][j] = cachemod.switch_cost(
                        mi, mj, history_tokens,
                        session.cache.get(mj.name),
                        session.cache.get(mi.name), now)
        chosen_idx = wfa["wf"].step(costs, dmat)
        chosen = wfa["names"][chosen_idx]
        wfa["state_idx"] = chosen_idx
        stayed = chosen == incumbent
        reason = "wfa:stay" if stayed else "wfa:switch"
        return _commit(session, chosen, reason, tier, scores,
                       pii_findings, feasible, category)

    if cfg.decision_mode == "satisfice" and len(feasible) > 1:
        # lowest capability tier that still clears P(suffice) >= tau.
        # correct under failure-asymmetric metrics (under-routing wastes
        # the whole chain) where $-margin under-protects.
        ordered = sorted(
            feasible,
            key=lambda m: (m.extra.get("suff_tier", 99),
                           m.pricing.output_per_mtok))
        chosen = ordered[-1].name
        for m in ordered:
            if quality_deficit.get(m.name, 1.0) <= 1.0 - cfg.satisfice_tau:
                chosen = m.name
                break
        stayed = chosen == incumbent
        reason = "satisfice:stay" if stayed else "satisfice:switch"
        if incumbent is None:
            reason = "satisfice:initial"
        return _commit(session, chosen, reason, tier, scores,
                       pii_findings, feasible, category)

    chosen_name = max(feasible, key=lambda m: scores[m.name]).name
    stayed = chosen_name == incumbent
    reason = "margin:stay" if stayed else "margin:switch"
    if incumbent is None:
        reason = "margin:initial"
    return _commit(session, chosen_name, reason, tier, scores,
                   pii_findings, feasible, category)


def _commit(session, chosen, reason, tier, scores, pii_findings,
            feasible, category) -> RouteDecision:
    spec = next(m for m in feasible if m.name == chosen)
    switched = session.incumbent_model is not None and chosen != session.incumbent_model
    if switched:
        session.switch_count += 1
    if session.incumbent_model != chosen:
        session.incumbent_model = chosen
    if spec.tier == "cloud":
        session.escalated = True
    session.last_category = category
    return RouteDecision(model=chosen, reason=reason, privacy_tier=tier,
                         scores=scores, stayed=not switched,
                         pii_findings=pii_findings or [])
