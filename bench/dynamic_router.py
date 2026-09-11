"""TwinRouterBench dynamic-track adapter: exposes our trained tier scorer
through the swerouter Router protocol (select(ctx) -> RouterDecision).

Run from the trbenv venv:
  python -m miniswerouter.cli run \
      --router-import sessionrouter_bench.dynamic_router:SessionRouterAdapter \
      --router-arg scorer_path=/path/to/artifacts/tier_scorer.pkl \
      --router-arg tau=0.7 \
      --router-label sessionrouter-tau07 \
      --output-dir runs/sr --limit 5
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

# allow running as a loose module: --router-import accepts a file path too in
# some harness versions; fall back gracefully on relative imports
try:
    from sessionrouter.types import Message
    from sessionrouter.features import extract_features
except ImportError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from sessionrouter.types import Message
    from sessionrouter.features import extract_features

from scipy.sparse import hstack, csr_matrix
from swerouter.router import RouterContext, RouterDecision

TIER_TO_MODEL = {
    "low": "deepseek/deepseek-v3.2",
    "mid": "minimax/minimax-m2.7",
    "mid_high": "google/gemini-3-flash-preview",
    "high": "anthropic/claude-opus-4.6",
}
TIERS = ["low", "mid", "mid_high", "high"]


def _step_text(msgs: list[Message]) -> str:
    users = [m.content for m in msgs if m.role == "user"]
    first = users[0][:3000] if users else ""
    last = users[-1][:3000] if users else ""
    sys_m = next((m.content for m in msgs if m.role == "system"), "")
    parts = [sys_m[:1500], first]
    if last != first:
        parts.append("[TURN]\n" + last)
    return "\n".join(p for p in parts if p)


class SessionRouterAdapter:
    """tau-satisficing tier router. Picks the lowest tier whose cumulative
    P(gold <= t) >= tau; maps tier -> concrete model via the locked pool map.
    """

    def __init__(self, scorer_path: str, tau: float = 0.7,
                 tier_map_path: str | None = None):
        with open(scorer_path, "rb") as f:
            blob = pickle.load(f)
        self.m = blob["model"]
        self.tau = float(tau)
        if tier_map_path:
            with open(tier_map_path) as f:
                self.t2m = json.load(f)["map"]
        else:
            self.t2m = TIER_TO_MODEL

    def _suff(self, messages: tuple) -> np.ndarray:
        msgs = []
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                c = "".join(b.get("text", "") for b in c
                            if isinstance(b, dict))
            role = m.get("role", "user")
            msgs.append(Message(role if role in ("system", "user",
                                                 "assistant", "tool")
                                else "user", str(c)))
        X = hstack([self.m["tf"].transform([_step_text(msgs)]),
                    csr_matrix(self.m["sc"].transform(
                        np.array([extract_features(msgs)])))])
        return np.cumsum(self.m["clf"].predict_proba(X)[0])

    def select(self, ctx: RouterContext) -> RouterDecision:
        suff = self._suff(ctx.messages)
        tier = 3
        for t in range(3):
            if suff[t] >= self.tau:
                tier = t
                break
        model_id = self.t2m[TIERS[tier]]
        if model_id not in ctx.available_models:
            model_id = ctx.available_models[-1]
        return RouterDecision(
            model_id=model_id,
            rationale=f"tier={TIERS[tier]} suff="
                      f"{[round(float(x), 2) for x in suff]}")
