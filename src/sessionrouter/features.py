"""Query complexity scoring — the per-turn 'does local suffice?' estimator.

Ships a zero-dependency heuristic scorer. The `Scorer` protocol is the seam
for a trained head (ModernBERT/embedding+MLP) once bootstrap labels exist —
train on cloud-labeled outcomes per the RoRF/HybridLLM recipe.
"""
from __future__ import annotations

import re
from typing import Protocol

from .types import Message


class Scorer(Protocol):
    def local_sufficiency(self, messages: list[Message], category: str) -> float:
        """P(local model produces an acceptable answer) in [0,1]."""
        ...


_CATEGORIES = {
    "coding": re.compile(
        r"```|\b(def|function|class|import|traceback|syntax|compile|regex|"
        r"refactor|debug|implement|algorithm)\b", re.I),
    "math": re.compile(
        r"\b(compute|solve|integral|derivative|proof|equation|probability|"
        r"\d+\s*[\+\-\*/\^]\s*\d+|modular|matrix)\b", re.I),
    "reasoning": re.compile(
        r"\b(step[- ]by[- ]step|prove|deduce|analyze|compare|evaluate|"
        r"trade-?off|why does|explain why|reason)\b", re.I),
    "creative": re.compile(
        r"\b(write|draft|poem|story|essay|rewrite|paraphrase|tone)\b", re.I),
    "factual": re.compile(
        r"\b(who|what|when|where|which|define|capital|list|name)\b", re.I),
}


def categorize(text: str) -> str:
    for cat, rx in _CATEGORIES.items():
        if rx.search(text):
            return cat
    return "general"


# Coarse groupings — drift resets only fire across groups, so
# math→reasoning (same technical task) doesn't bust an escalation.
CATEGORY_GROUP = {
    "coding": "technical", "math": "technical", "reasoning": "technical",
    "creative": "creative",
    "factual": "factual", "general": "factual",
}


def category_group(cat: str) -> str:
    return CATEGORY_GROUP.get(cat, "factual")


CAT_INDEX = {"coding": 0, "math": 1, "reasoning": 2, "creative": 3,
             "factual": 4, "general": 5}


_OBSERVATION = re.compile(
    r"(traceback|diff --git|@@|-{10,}|\bexit code\b|<pr_description>|"
    r"\bobservation\b|^stdout:|^stderr:)", re.I | re.M)


def is_agentic_step(messages: list[Message], last_user: str) -> bool:
    """Session contains tool traffic or this turn carries an observation —
    used as a structural FEATURE for scoring."""
    return any(m.role == "tool" for m in messages) or \
        is_observation_turn(messages, last_user)


def is_observation_turn(messages: list[Message], last_user: str) -> bool:
    """The CURRENT step is an observation/pasted artifact rather than a
    fresh instruction — mid-loop steps like this should hold the incumbent
    model rather than re-route on observation content."""
    if messages and messages[-1].role == "tool":
        return True
    # an observation can only exist after the conversation has started;
    # a first-turn blob containing a traceback/PR description is still an
    # instruction and must be scored for difficulty
    if len(messages) <= 3:
        return False
    if _OBSERVATION.search(last_user):
        return True
    # long pasted blob after earlier turns = context, not instruction
    return len(last_user) > 4000


def extract_features(messages: list[Message]) -> list[float]:
    """Feature vector for a trained head. Shared by the heuristic scorer
    and bench/trainable variants."""
    import math
    last_user = next((m.content for m in reversed(messages)
                      if m.role == "user"), "")
    cat = categorize(last_user)
    n_chars = len(last_user)
    n_turns = sum(1 for m in messages if m.role == "user")
    n_hard = len(HeuristicScorer._HARD_MARKERS.findall(last_user))
    n_steps = len(HeuristicScorer._MULTISTEP.findall(last_user))
    has_spec = 1.0 if HeuristicScorer._SPEC.search(last_user) else 0.0
    has_code = 1.0 if "```" in last_user else 0.0
    onehot = [0.0] * len(CAT_INDEX)
    onehot[CAT_INDEX.get(cat, 5)] = 1.0
    n_msgs = len(messages)
    n_tool = sum(1 for m in messages if m.role == "tool")
    n_asst = sum(1 for m in messages if m.role == "assistant")
    total_chars = sum(len(m.content) for m in messages)
    agentic = 1.0 if is_agentic_step(messages, last_user) else 0.0
    return [math.log1p(n_chars), float(n_turns), float(min(n_hard, 5)),
            float(min(n_steps, 6)), has_spec, has_code,
            math.log1p(total_chars), float(n_msgs), float(n_tool),
            float(n_asst), agentic] + onehot


class HeuristicScorer:
    """Feature-based P(local suffices). Hand-tuned; replace with a trained
    scorer via the Scorer protocol when labeled data exists.

    Signals (per literature): length is a weak/bad proxy, multi-step markers
    and domain (math/code) matter more; presence of tools forces cloud-tier
    handling only if the local model lacks tool support (handled upstream).
    """

    # per-category prior that a small model suffices
    PRIOR = {"coding": 0.45, "math": 0.45, "reasoning": 0.55,
             "creative": 0.75, "factual": 0.80, "general": 0.80}

    _HARD_MARKERS = re.compile(
        r"\b(prove|theorem|optimize|formally|complexity analysis|amortized|"
        r"threadsafe|concurrency|distributed|cryptograph|differential)\b", re.I)
    _MULTISTEP = re.compile(
        r"\b(first|then|next|finally|step \d|1\.|2\.|3\.)\b", re.I)
    _SPEC = re.compile(
        r"\b(exactly|must|constraints?|requirements?|specification|schema|"
        r"as json|table format)\b", re.I)

    def local_sufficiency(self, messages: list[Message], category: str = "") -> float:
        last_user = next((m.content for m in reversed(messages)
                          if m.role == "user"), "")
        cat = category or categorize(last_user)
        p = self.PRIOR.get(cat, 0.8)

        # score the CURRENT turn, not the whole transcript — long sessions
        # must not make every trivial follow-up look hard
        n_chars = len(last_user)
        n_turns = sum(1 for m in messages if m.role == "user")
        n_hard = len(self._HARD_MARKERS.findall(last_user))
        n_steps = len(self._MULTISTEP.findall(last_user))
        has_spec = bool(self._SPEC.search(last_user))
        has_code = "```" in last_user

        if is_observation_turn(messages, last_user):
            # observation/tool-loop step: marker words ('must', 'then',
            # 'prove') live inside the pasted artifact, not the task, and
            # step-level reasoning is usually mechanical — skip content
            # penalties entirely and apply an agentic prior instead
            p = self.PRIOR.get(cat, 0.8) + 0.10
            return min(max(p, 0.02), 0.98)

        # bounded logit-style adjustments
        adj = 0.0
        adj -= 0.18 * min(n_hard, 3)
        adj -= 0.10 * min(n_steps, 4)
        adj -= 0.12 if has_spec else 0.0
        adj -= 0.10 if has_code and cat == "coding" else 0.0
        adj -= 0.08 if n_chars > 4000 else 0.0          # very long asks
        adj += 0.08 if n_chars < 200 else 0.0           # short & simple
        adj += 0.05 if n_turns <= 1 else 0.0
        p = min(max(p + adj, 0.02), 0.98)
        # trivial-ask floor: short, no difficulty signals -> near-certain local
        if n_chars < 120 and n_hard == 0 and n_steps == 0 and not has_spec \
                and not has_code:
            p = max(p, 0.97)
        return p


def default_scorer() -> HeuristicScorer:
    return HeuristicScorer()
