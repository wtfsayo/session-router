"""Train a persistable P(local-suffices) head on TwinRouterBench
(step-level session routing labels) and optionally RouterBench
(per-model outcome matrix). Produces artifacts/scorer.pkl loadable via
`session-router serve --scorer artifacts/scorer.pkl`.

Usage:
  uv run --with pandas --with pyarrow --with huggingface_hub \
      --with scikit-learn python bench/train_scorer.py [--out artifacts/scorer.pkl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from sessionrouter.features import extract_features
from sessionrouter.trained import save_scorer
from sessionrouter.types import Message

TIER_MAP = {"low": 1, "mid": 0, "mid_high": 0, "high": 0}  # 1 = local


def to_messages(raw) -> list[Message]:
    msgs = json.loads(raw) if isinstance(raw, str) else list(raw)
    out = []
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, list):
            c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
        role = m.get("role", "user")
        out.append(Message(role if role in ("system", "user", "assistant",
                                            "tool") else "user", str(c)))
    return out


def load_twin() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Amorph/TwinRouterBench", "data/train.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    return df[df["target_tier"].isin(TIER_MAP)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/scorer.pkl")
    ap.add_argument("--routerbench-pkl", default="",
                    help="optional: merge RouterBench outcome rows")
    args = ap.parse_args()

    df = load_twin()
    X, y, groups = [], [], []
    for _, row in df.iterrows():
        X.append(extract_features(to_messages(row["messages"])))
        y.append(TIER_MAP[row["target_tier"]])
        groups.append(f"{row['benchmark']}|{row['instance_id']}")

    # optional: RouterBench local-vs-gpt4 outcomes, single-turn sessions
    if args.routerbench_pkl:
        rb = pd.read_pickle(args.routerbench_pkl)
        LOCAL = "mistralai/mistral-7b-chat"
        rb = rb.dropna(subset=[LOCAL])
        rb["pt"] = rb["prompt"].apply(
            lambda p: "\n".join(p) if isinstance(p, list) else str(p))
        for i, row in rb.iterrows():
            X.append(extract_features(
                [Message("user", row["pt"])]))
            y.append(1 if float(row[LOCAL]) >= 0.5 else 0)
            groups.append(f"rb|{i}")

    X, y = np.array(X), np.array(y)
    groups = np.array(groups)

    # group split: no session leaks across train/test
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
    print(f"train={len(tr)} test={len(te)} "
          f"acc={clf.score(X[te], y[te]):.3f} base_rate={y.mean():.2f}")

    # final: fit on everything
    clf.fit(X, y)
    path = save_scorer(clf, args.out, feature_dim=X.shape[1],
                       meta={"trained_on": "twinrouterbench"
                             + ("+routerbench" if args.routerbench_pkl else ""),
                             "n": int(len(y)), "local_rate": float(y.mean())})
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
