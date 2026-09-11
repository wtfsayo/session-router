"""Hillclimb harness for TwinRouterBench. Every variant evaluated on the
held-out session split is logged to bench/experiments.sqlite.

A variant = (feature spec, classifier spec, decision spec). Reproducible:
same seed, same session-level split as bench/twinrouter.py.

Usage:
  uv run --with-editable . --with pandas --with pyarrow \
      --with huggingface_hub --with scikit-learn --with tiktoken \
      --with scipy python bench/hillclimb.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
sys.path.insert(0, os.environ.get("TRB_REPO", "/tmp/trb"))

from huggingface_hub import hf_hub_download
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from main.eval.section11 import compute_v2_scores  # noqa: E402
from sessionrouter.features import extract_features  # noqa: E402
from twinrouter import to_messages, TIERS  # noqa: E402

DB = "bench/experiments.sqlite"


# ---------------- data ----------------

def load_split():
    p = hf_hub_download("Amorph/TwinRouterBench", "data/train.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    insts = df[["benchmark", "instance_id"]].drop_duplicates()
    tr = insts.sample(frac=0.7, random_state=0)
    tr_key = set(tr["benchmark"] + "|" + tr["instance_id"])
    key = df["benchmark"] + "|" + df["instance_id"]
    return df[key.isin(tr_key)], df[~key.isin(tr_key)]


# ---------------- features ----------------

def step_text(msgs, with_sys=True, with_asst=False, prev_user=False):
    users = [m.content for m in msgs if m.role == "user"]
    first = users[0][:3000] if users else ""
    last = users[-1][:3000] if users else ""
    parts = []
    if with_sys:
        sys_m = next((m.content for m in msgs if m.role == "system"), "")
        parts.append(sys_m[:1500])
    parts.append(first)
    if prev_user and len(users) >= 2 and users[-2] != last:
        parts.append("[PREV]\n" + users[-2][:1500])
    if last != first:
        parts.append("[TURN]\n" + last)
    if with_asst:
        a = next((m.content for m in reversed(msgs)
                  if m.role == "assistant"), "")
        if a:
            parts.append("[ASST]\n" + a[:2000])
    return "\n".join(p for p in parts if p)


@dataclass
class Variant:
    name: str
    # features
    max_features: int = 20000
    ngram: int = 3
    with_sys: bool = True
    with_asst: bool = False
    with_pos: bool = False     # step_index/total_steps features
    C: float = 8.0
    head: str = "multiclass"   # "multiclass" | "per-tier-binary" | "ensemble"
    n_ensemble: int = 1
    analyzer: str = "word"
    prev_user: bool = False    # include second-to-last user turn
    calibrate: bool = False
    # decision
    rule: str = "tau"          # "tau" | "ecost"
    tau: float = 0.7
    fail_cost: float = 2.0     # $ estimate for ecost rule
    switch_pen: float = 0.0    # $ per-tier-jump penalty in ecost


def _feats(v: Variant, msgs, row=None) -> np.ndarray:
    f = extract_features(msgs)
    if v.with_pos and row is not None:
        f = f + [float(row.get("step_index", 1)),
                 float(row.get("total_steps", 1)),
                 float(row.get("step_index", 1))
                 / max(1.0, float(row.get("total_steps", 1)))]
    return np.array(f)


def build(v: Variant, train_df):
    texts = [step_text(to_messages(m), v.with_sys, v.with_asst, v.prev_user)
             for m in train_df["messages"]]
    feats = np.array([_feats(v, to_messages(r["messages"]), r)
                      for _, r in train_df.iterrows()])
    y = train_df["target_tier_id"].to_numpy()
    tf = TfidfVectorizer(max_features=v.max_features,
                         ngram_range=(1, v.ngram), min_df=2,
                         sublinear_tf=True, analyzer=v.analyzer)
    sc = StandardScaler(with_mean=False)
    X = hstack([tf.fit_transform(texts), csr_matrix(sc.fit_transform(feats))])
    if v.head == "ensemble":
        from sklearn.ensemble import BaggingClassifier
        base = LogisticRegression(max_iter=2000, C=v.C)
        clf = BaggingClassifier(base, n_estimators=v.n_ensemble,
                                max_samples=0.7, random_state=0)
        clf.fit(X, y)
        return {"tf": tf, "sc": sc, "clf": clf}
    if v.head == "multiclass":
        clf = LogisticRegression(max_iter=3000, C=v.C)
        if v.calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            clf = CalibratedClassifierCV(clf, cv=3)
        clf.fit(X, y)
        return {"tf": tf, "sc": sc, "clf": clf}
    # per-tier binary heads: P(gold <= t) for t in 0..2
    heads = [LogisticRegression(max_iter=3000, C=v.C).fit(X, (y <= t).astype(int))
             for t in range(3)]
    return {"tf": tf, "sc": sc, "heads": heads}


# ---------------- decision rules ----------------

TIER_STEP_COST = {  # rough per-step $ (in+out at typical sizes) for ecost
    0: 0.002, 1: 0.004, 2: 0.010, 3: 0.09,
}


def predict_tier(suff: np.ndarray, v: Variant,
                 prev_tier: int | None = None) -> int:
    if v.rule == "tau":
        taus = v.tau if isinstance(v.tau, (list, tuple)) else (v.tau,) * 3
        for t in range(3):
            if suff[t] >= taus[t]:
                return t
        return 3
    # ecost: argmin_t [ step_cost(t) + P(gold>t) * fail_cost + jump*switch_pen ]
    best, best_t = float("inf"), 3
    for t in range(4):
        c = TIER_STEP_COST[t] + (1.0 - suff[t]) * v.fail_cost
        if prev_tier is not None and t != prev_tier:
            c += v.switch_pen
        if c < best:
            best, best_t = c, t
    return best_t


def suff_probs(model, v: Variant, msgs, row=None) -> np.ndarray:
    """P(gold <= t) for t=0..3 (cumsum of class probs or direct binaries)."""
    X = hstack([model["tf"].transform(
                [step_text(msgs, v.with_sys, v.with_asst, v.prev_user)]),
                csr_matrix(model["sc"].transform(
                    np.array([_feats(v, msgs, row)])))])
    if "clf" in model:
        return np.cumsum(model["clf"].predict_proba(X)[0])
    p = np.array([h.predict_proba(X)[0][1] for h in model["heads"]])
    p = np.maximum.accumulate(np.clip(p, 0, 1))   # enforce monotone
    return np.concatenate([p, [1.0]])


def evaluate(v: Variant, model, test_df, per_session=True) -> dict:
    recs = []
    for (b, inst), grp in test_df.groupby(["benchmark", "instance_id"],
                                         sort=False):
        grp = grp.sort_values("step_index")
        prev = None
        for _, r in grp.iterrows():
            suff = suff_probs(model, v, to_messages(r["messages"]), r)
            pred = predict_tier(suff, v, prev)
            prev = pred if per_session else None
            gold = int(r["target_tier_id"])
            recs.append(dict(id=r["id"], benchmark=r["benchmark"],
                             gold_tier_id=gold, pred_tier_id=pred,
                             match=pred == gold, passed=pred >= gold,
                             instance_id=r["instance_id"],
                             step_index=r["step_index"],
                             total_steps=r["total_steps"],
                             messages=json.loads(r["messages"]) if isinstance(
                                 r["messages"], str) else r["messages"]))
    return compute_v2_scores(recs)


# ---------------- experiment log ----------------

def log(db: sqlite3.Connection, v: Variant, s: dict):
    db.execute(
        "INSERT INTO runs (ts, variant, config, rowpass, rowexact, trajpass,"
        " costsave, combined) VALUES (?,?,?,?,?,?,?,?)",
        (time.time(), v.name, json.dumps(asdict(v)),
         s["case_pass_rate_percent"], s["case_exact_match_percent"],
         s["trajectory_pass_rate_percent"],
         s["cost_savings_score_percent"], s["combined_score_percent"]))
    db.commit()


def show(v: Variant, s: dict):
    print(f"{v.name:42s} RP={s['case_pass_rate_percent']:6.2f} "
          f"RE={s['case_exact_match_percent']:6.2f} "
          f"TP={s['trajectory_pass_rate_percent']:6.2f} "
          f"CS={s['cost_savings_score_percent']:7.2f} "
          f"COMB={s['combined_score_percent']:6.2f}")


def main():
    os.makedirs("bench", exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("""CREATE TABLE IF NOT EXISTS runs
                  (ts REAL, variant TEXT, config TEXT, rowpass REAL,
                   rowexact REAL, trajpass REAL, costsave REAL,
                   combined REAL)""")
    train_df, test_df = load_split()

    variants = [
        Variant("base tau=.7"),
        Variant("taus=.7/.5/.5", tau=[0.7, 0.5, 0.5]),
        Variant("taus=.75/.6/.5", tau=[0.75, 0.6, 0.5]),
        Variant("taus=.8/.6/.5", tau=[0.8, 0.6, 0.5]),
        Variant("taus=.7/.4/.4", tau=[0.7, 0.4, 0.4]),
        Variant("taus=.75/.75/.75", tau=[0.75, 0.75, 0.75]),
        Variant("taus=.85/.7/.5", tau=[0.85, 0.7, 0.5]),
    ]
    for v in variants:
        model = build(v, train_df)
        s = evaluate(v, model, test_df)
        show(v, s)
        log(db, v, s)


if __name__ == "__main__":
    main()
