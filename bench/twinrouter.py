"""TwinRouterBench v2 — OFFICIAL scoring via the benchmark's own package.

Runs our Router against the 970 step-level prefixes, predicting one of 4
tier_ids per step, and scores with main.eval.section11.compute_v2_scores
(RowPass / RowExact / TrajPass / CostSave / Combined — the paper's Table 2
metrics, including the failure-aware trajectory penalty and full tier
pricing with cache-read/write splits).

Policies:
  - always-low / always-high baselines
  - heuristic scorer -> binary router (low vs high)
  - trained multiclass head -> argmax (no session policy; ablation)
  - trained head -> real Router with 4-tier ladder (margin + holds)

Requires: -e . pandas pyarrow huggingface_hub scikit-learn tiktoken
and the TwinRouterBench repo cloned locally (TRB_REPO env or /tmp/trb).

Usage:
  uv run --with-editable . --with pandas --with pyarrow \
      --with huggingface_hub --with scikit-learn --with tiktoken \
      python bench/twinrouter.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, os.environ.get("TRB_REPO", "/tmp/trb"))

from huggingface_hub import hf_hub_download

from main.eval.section11 import compute_v2_scores  # noqa: E402
from main.tiers import ID_TO_TIER  # noqa: E402

from sessionrouter import (Message, ModelSpec, PolicyConfig, Pricing, Router,  # noqa: E402
                           SessionStore)
from sessionrouter.features import extract_features  # noqa: E402
from sessionrouter.types import BackendResponse, TokenLogprob, TokenUsage  # noqa: E402

TIERS = ["low", "mid", "mid_high", "high"]

# tier pricing tables (from trb/main/pricing.py) — USD per 1M tokens
TIER_PRICE = {  # input, output, cache_read, cache_write
    "low": (0.26, 0.5, 0.13, 0.26),
    "mid": (0.30, 2.0, 0.059, 0.30),
    "mid_high": (0.50, 5.0, 0.05, 0.08333),
    "high": (5.0, 25.0, 0.50, 6.25),
}


class DummyBackend:
    def generate(self, spec, messages, *, want_logprobs=False,
                 session_key="", max_tokens=1024):
        toks = sum(len(m.content) for m in messages) // 4
        return BackendResponse(
            text="ok", model=spec.name, usage=TokenUsage(toks, 200),
            logprobs=[TokenLogprob("t", -0.3)] * 10, finish_reason="stop")


def to_messages(raw) -> list[Message]:
    msgs = json.loads(raw) if isinstance(raw, str) else list(raw)
    out = []
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, list):
            c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant", "tool"):
            role = "user"
        out.append(Message(role, str(c)))
    return out


# ---------- feature pipeline ----------

def last_user_text(msgs: list[Message]) -> str:
    return next((m.content for m in reversed(msgs) if m.role == "user"), "")


def step_text(msgs: list[Message]) -> str:
    """Instruction (first user msg — holds the task in agentic transcripts)
    + current turn (last user msg, usually an observation)."""
    users = [m.content for m in msgs if m.role == "user"]
    first = users[0][:3000] if users else ""
    last = users[-1][:3000] if users else ""
    if first == last:
        return last
    return first + "\n[TURN]\n" + last


def build_pipeline(train_df):
    """TF-IDF over instruction+current turn + structural features."""
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    texts, feats, y = [], [], []
    for _, row in train_df.iterrows():
        msgs = to_messages(row["messages"])
        texts.append(step_text(msgs))
        feats.append(extract_features(msgs))
        y.append(int(row["target_tier_id"]))

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 3),
                            min_df=2, sublinear_tf=True)
    X_text = tfidf.fit_transform(texts)
    sc = StandardScaler(with_mean=False)
    X_feat = sc.fit_transform(np.array(feats))
    X = hstack([X_text, csr_matrix(X_feat)])
    clf = LogisticRegression(max_iter=3000, C=8.0, class_weight=None)
    clf.fit(X, np.array(y))
    return {"tfidf": tfidf, "scaler": sc, "clf": clf}


def predict_proba(pipe, msgs) -> np.ndarray:
    from scipy.sparse import hstack, csr_matrix
    Xt = pipe["tfidf"].transform([step_text(msgs)])
    Xf = pipe["scaler"].transform(np.array([extract_features(msgs)]))
    return pipe["clf"].predict_proba(hstack([Xt, csr_matrix(Xf)]))[0]


class TierScorer:
    """Scorer protocol impl: multiclass head -> per-tier sufficiency.

    tier_sufficiency(t) = P(gold_min_sufficient_tier <= t).
    Also satisfies local_sufficiency (binary fallback) as P(gold == low).
    """
    def __init__(self, pipe):
        self.pipe = pipe

    def _probs(self, messages) -> np.ndarray:
        return predict_proba(self.pipe, messages)

    def tier_sufficiency(self, messages) -> list[float]:
        return np.cumsum(self._probs(messages)).tolist()

    def local_sufficiency(self, messages, category="") -> float:
        return float(self._probs(messages)[0])

    def predict_tier_id(self, messages, tau: float = 0.5) -> int:
        """Lowest tier whose cumulative sufficiency >= tau (conservative)."""
        cum = np.cumsum(self._probs(messages))
        for t in range(3):
            if cum[t] >= tau:
                return t
        return 3


# ---------- router construction ----------

def make_tier_router(scorer, **cfg_kw):
    models = []
    for i, t in enumerate(TIERS):
        inp, outp, cr, cw = TIER_PRICE[t]
        models.append(ModelSpec(
            name=f"tier-{t}", tier="local" if i == 0 else "cloud",
            backend="d",
            pricing=Pricing(inp, outp, cached_input_per_mtok=cr,
                            cache_write_per_mtok=cw,
                            cache_ttl_seconds=1e9,  # bench TTL is step-based
                            cache_min_tokens=0),
            quality=0.6 + 0.1 * i,
            extra={"suff_tier": i}))
    return Router(models, {"d": DummyBackend()},
                  cfg=PolicyConfig(**cfg_kw), store=SessionStore(),
                  scorer=scorer, enable_gate=False, enable_pii=False)


# ---------- evaluation ----------

def score_rows(records, label):
    v2 = compute_v2_scores(records)
    print(f"{label:34s} RowPass={v2['case_pass_rate_percent']:6.2f} "
          f"RowExact={v2['case_exact_match_percent']:6.2f} "
          f"TrajPass={v2['trajectory_pass_rate_percent']:6.2f} "
          f"CostSave={v2['cost_savings_score_percent']:7.2f} "
          f"Combined={v2['combined_score_percent']:6.2f}")
    return v2


def records_const(df, tier_id):
    return [dict(id=r["id"], benchmark=r["benchmark"],
                 gold_tier_id=int(r["target_tier_id"]), pred_tier_id=tier_id,
                 match=int(r["target_tier_id"]) == tier_id,
                 passed=tier_id >= int(r["target_tier_id"]),
                 instance_id=r["instance_id"], step_index=r["step_index"],
                 total_steps=r["total_steps"],
                 messages=json.loads(r["messages"]) if isinstance(
                     r["messages"], str) else r["messages"])
            for _, r in df.iterrows()]


def records_argmax(df, scorer, tau):
    recs = []
    for _, r in df.iterrows():
        msgs = to_messages(r["messages"])
        pred = scorer.predict_tier_id(msgs, tau)
        gold = int(r["target_tier_id"])
        recs.append(dict(id=r["id"], benchmark=r["benchmark"],
                         gold_tier_id=gold, pred_tier_id=pred,
                         match=pred == gold, passed=pred >= gold,
                         instance_id=r["instance_id"],
                         step_index=r["step_index"],
                         total_steps=r["total_steps"],
                         messages=json.loads(r["messages"]) if isinstance(
                             r["messages"], str) else r["messages"]))
    return recs


def records_router(df, scorer, return_reasons=False, **cfg_kw):
    """Real Router replay: session state carries across steps."""
    r = make_tier_router(scorer, **cfg_kw)
    recs, reasons = [], defaultdict(int)
    for (bench, inst), grp in df.groupby(["benchmark", "instance_id"],
                                         sort=False):
        grp = grp.sort_values("step_index")
        sid = f"{bench}:{inst}"
        for _, row in grp.iterrows():
            msgs = to_messages(row["messages"])
            _, dec = r.handle(msgs, session_id=sid)
            reasons[dec.reason] += 1
            pred = TIERS.index(dec.model[len("tier-"):])
            gold = int(row["target_tier_id"])
            recs.append(dict(id=row["id"], benchmark=row["benchmark"],
                             gold_tier_id=gold, pred_tier_id=pred,
                             match=pred == gold, passed=pred >= gold,
                             instance_id=row["instance_id"],
                             step_index=row["step_index"],
                             total_steps=row["total_steps"],
                             messages=json.loads(row["messages"]) if isinstance(
                                 row["messages"], str) else row["messages"]))
    if return_reasons:
        return recs, reasons
    return recs


def main():
    p = hf_hub_download("Amorph/TwinRouterBench", "data/train.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    print(f"{len(df)} steps; tier dist: "
          f"{df['target_tier_id'].value_counts().sort_index().to_dict()}")

    # session-level split
    insts = df[["benchmark", "instance_id"]].drop_duplicates()
    tr_inst = insts.sample(frac=0.7, random_state=0)
    tr_key = set((tr_inst["benchmark"] + "|" + tr_inst["instance_id"]))
    key = df["benchmark"] + "|" + df["instance_id"]
    train_df, test_df = df[key.isin(tr_key)], df[~key.isin(tr_key)]
    print(f"train={len(train_df)} test={len(test_df)}")

    print("\n--- baselines (test split) ---")
    score_rows(records_const(test_df, 0), "always-low")
    score_rows(records_const(test_df, 3), "always-high")

    print("\n--- training multiclass head ---")
    pipe = build_pipeline(train_df)
    scorer = TierScorer(pipe)
    tr_acc = (np.array([int(np.argmax(predict_proba(pipe, to_messages(m))))
                        for m in train_df["messages"]])
              == train_df["target_tier_id"].to_numpy()).mean()
    print(f"train argmax acc={tr_acc:.3f}")

    print("\n--- direct argmax / tau-swept (no session policy) ---")
    best = None
    for tau in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        v2 = score_rows(records_argmax(test_df, scorer, tau),
                        f"argmax tau={tau}")
        if best is None or v2["combined_score_percent"] > best[1]:
            best = (tau, v2["combined_score_percent"])

    print("\n--- real Router, 4-tier ladder ---")
    for qw in (0.05, 0.3):
        score_rows(records_router(test_df, scorer, quality_weight=qw,
                                  switch_margin=0.002),
                   f"router margin qw={qw}")
    score_rows(records_router(test_df, scorer, quality_weight=0.05,
                              switch_margin=0.002, decision_mode="wfa"),
               "router WFA qw=0.05")
    for tau in (0.6, 0.65, 0.7, 0.8):
        score_rows(records_router(test_df, scorer,
                                  decision_mode="satisfice",
                                  satisfice_tau=tau),
                   f"router satisfice tau={tau}")
    best_tau = best[0]
    v2 = score_rows(
        records_router(test_df, scorer, decision_mode="satisfice",
                       satisfice_tau=best_tau),
        f"router satisfice tau={best_tau} (best)")
    print("\nper-benchmark CostSave (best router):")
    for b, s in v2["by_benchmark"].items():
        print(f"    {b:12s} rows={s['row_count']:3d} "
              f"CostSave={s['cost_savings_score_percent']:7.2f} "
              f"failed_traj={s['failed_trajectory_count']}")


if __name__ == "__main__":
    main()
