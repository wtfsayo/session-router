"""Accept/escalate gate for the local-first cascade.

Tiered: hard validators (near-free, ~100% precision on catastrophic failure)
then token-confidence on the answer span. Never gates on self-verification —
small models are confidently wrong.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum

from .types import BackendResponse, Message


class Verdict(str, Enum):
    ACCEPT = "accept"
    ESCALATE = "escalate"


_REFUSAL = re.compile(
    r"\b(I (?:can(?:not|'t)|won't|am unable to)|as an AI|"
    r"I'm not able to|I do not have)\b", re.I)
_HEDGE = re.compile(r"\b(I think|probably|maybe|might be|not sure)\b", re.I)


def _repetition_ratio(text: str, n: int = 4) -> float:
    toks = text.split()
    if len(toks) < 2 * n:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _answer_span_logprob(resp: BackendResponse) -> tuple[float, float]:
    """(mean_logprob, min_logprob) over the tail of the generation —
    CoT tokens inflate confidence when wrong, so we score only the
    final-answer region (last ~25% of tokens, min 8)."""
    lps = resp.logprobs
    if not lps:
        return 0.0, 0.0
    k = max(8, len(lps) // 4)
    span = [t.logprob for t in lps[-k:]]
    return sum(span) / len(span), min(span)


@dataclass
class GateConfig:
    mean_logprob_floor: float = -1.0     # escalate below this (fit on your traffic)
    min_logprob_tripwire: float = -6.0   # single very-low-prob answer token
    min_answer_chars: int = 1
    max_answer_chars: int = 32768
    max_repetition: float = 0.55
    expects_json: bool = False
    expects_tool_call: bool = False


def check(resp: BackendResponse, messages: list[Message],
          cfg: GateConfig | None = None) -> tuple[Verdict, dict]:
    cfg = cfg or GateConfig()
    diag: dict = {"stage": "validators"}
    text = resp.text.strip()

    # --- Stage A: hard validators ---
    if resp.finish_reason == "length" or len(text) >= cfg.max_answer_chars:
        return Verdict.ESCALATE, {**diag, "reason": "truncated"}
    if len(text) < cfg.min_answer_chars and not resp.tool_calls:
        return Verdict.ESCALATE, {**diag, "reason": "empty"}
    rep = _repetition_ratio(text)
    if rep > cfg.max_repetition:
        return Verdict.ESCALATE, {**diag, "reason": f"repetition={rep:.2f}"}
    if _REFUSAL.search(text) and len(text) < 200:
        return Verdict.ESCALATE, {**diag, "reason": "refusal"}
    if cfg.expects_json:
        try:
            json.loads(text)
        except Exception:
            return Verdict.ESCALATE, {**diag, "reason": "invalid_json"}
    if cfg.expects_tool_call and not resp.tool_calls:
        return Verdict.ESCALATE, {**diag, "reason": "missing_tool_call"}
    if resp.tool_calls:
        for tc in resp.tool_calls:
            args = tc.get("function", {}).get("arguments", "")
            try:
                json.loads(args) if isinstance(args, str) else args
            except Exception:
                return Verdict.ESCALATE, {**diag, "reason": "invalid_tool_args"}

    # --- Stage B: answer-span token confidence ---
    if resp.logprobs:
        mean_lp, min_lp = _answer_span_logprob(resp)
        diag.update(mean_logprob=round(mean_lp, 3), min_logprob=round(min_lp, 3))
        diag["stage"] = "confidence"
        if min_lp < cfg.min_logprob_tripwire:
            return Verdict.ESCALATE, {**diag, "reason": f"min_logprob={min_lp:.2f}"}
        if mean_lp < cfg.mean_logprob_floor:
            return Verdict.ESCALATE, {**diag, "reason": f"mean_logprob={mean_lp:.2f}"}

    return Verdict.ACCEPT, diag
