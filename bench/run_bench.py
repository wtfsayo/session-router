"""RouterBench evaluation + synthetic session simulation.

Part A — single-turn routing on held-out data:
  policies: always-local, always-cloud, oracle, margin router with
  heuristic scorer, margin router with a trained LR scorer
  (features -> P(local correct)), each swept over quality_weight.

Part B — synthetic sessions (the novel part):
  Sessions of K turns sampled from held-out prompts; a system prompt
  lifts history past the provider cache floor. CostBook applies
  Anthropic-style write/read pricing on append-only transcripts.
  Compares: always-local, always-cloud, per-turn margin (no stickiness),
  session-aware margin, session-aware WFA, session-aware + gate cascade.

Usage: uv run --with pandas --with scikit-learn python bench/run_bench.py
"""
from __future__ import annotations

import argparse
import random
import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "bench")

from sessionrouter import (Message, ModelSpec, PolicyConfig, Pricing, Router,
                           SessionStore)
from sessionrouter.features import extract_features
from replay import CostBook, ReplayBackend

LOCAL = "mistralai/mistral-7b-chat"
CLOUD = "gpt-4-1106-preview"

# Anthropic-style cache economics for the cloud tier
CLOUD_PRICING = dict(in_p=3.0, out_p=15.0, write_p=3.75, read_p=0.30)


def load(pkl: str, n: int, seed: int = 0):
    df = pd.read_pickle(pkl)
    df = df.dropna(subset=[LOCAL, CLOUD])
    df["prompt_text"] = df["prompt"].apply(
        lambda p: "\n".join(p) if isinstance(p, list) else str(p))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if n and len(df) > n:
        df = df.iloc[:n]
    return df


def build_outcomes(df: pd.DataFrame) -> dict:
    out = {}
    rl = df.get(f"{LOCAL}|model_response")
    rc = df.get(f"{CLOUD}|model_response")
    for i, row in df.iterrows():
        out[row["prompt_text"]] = {
            "local": (str(rl[i])[:400] if rl is not None else "",
                      float(row[LOCAL])),
            "cloud": (str(rc[i])[:400] if rc is not None else "",
                      float(row[CLOUD])),
        }
    return out


class LRScorer:
    """Trained logistic-regression head over extract_features.
    Fits the Scorer protocol used by Router."""

    def __init__(self, clf):
        self.clf = clf

    def local_sufficiency(self, messages, category: str = "") -> float:
        return float(self.clf.predict_proba([extract_features(messages)])[0][1])


def train_scorer(df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    X, y = [], []
    for _, row in df.iterrows():
        X.append(extract_features([Message("user", row["prompt_text"])]))
        y.append(1 if float(row[LOCAL]) >= 0.5 else 0)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(np.array(X), np.array(y))
    acc = clf.score(np.array(X), np.array(y))
    return LRScorer(clf), acc


def make_router(outcomes, scorer=None, enable_gate=True, **cfg_kw):
    models = [
        ModelSpec("local-7b", tier="local", backend="replay",
                  pricing=Pricing(0.0, 0.0), quality=0.7),
        ModelSpec("cloud-gpt4", tier="cloud", backend="replay",
                  pricing=Pricing(
                      CLOUD_PRICING["in_p"], CLOUD_PRICING["out_p"],
                      cached_input_per_mtok=CLOUD_PRICING["read_p"],
                      cache_write_per_mtok=CLOUD_PRICING["write_p"],
                      cache_ttl_seconds=300.0),
                  quality=0.97),
    ]
    return Router(models, {"replay": ReplayBackend(outcomes)},
                  cfg=PolicyConfig(**cfg_kw), store=SessionStore(),
                  scorer=scorer, enable_gate=enable_gate)


# ---------- Part A ----------

def eval_router(df, outcomes, scorer, qw, enable_gate):
    r = make_router(outcomes, scorer=scorer, quality_weight=qw,
                    escalation_sticky=False, enable_gate=enable_gate)
    import uuid
    res = []
    for _, row in df.iterrows():
        resp, dec = r.handle([Message("user", row["prompt_text"])],
                             session_id=uuid.uuid4().hex)
        is_cloud = dec.model.endswith("gpt4")
        score = float(row[CLOUD]) if is_cloud else float(row[LOCAL])
        cost = float(row[f"{CLOUD}|total_cost"]) if is_cloud else 0.0
        res.append((is_cloud, score, cost))
    return res


def summarize(res):
    q = sum(x[1] for x in res) / len(res)
    c = sum(x[2] for x in res)
    pc = sum(1 for x in res if x[0]) / len(res)
    return q, c, pc


def part_a(test, outcomes, scorer):
    print("\n=== Part A: single-turn routing (held-out) ===")
    ql = float(test[LOCAL].mean())
    qc = float(test[CLOUD].mean())
    cc = float(test[f"{CLOUD}|total_cost"].sum())
    qo = float(test[[LOCAL, CLOUD]].max(axis=1).mean())
    co = sum(0.0 if float(r[LOCAL]) >= float(r[CLOUD])
             else float(r[f"{CLOUD}|total_cost"])
             for _, r in test.iterrows())
    print(f"always-local     quality={ql:.4f} cost=$0.00")
    print(f"always-cloud     quality={qc:.4f} cost=${cc:.2f}")
    print(f"oracle           quality={qo:.4f} cost=${co:.2f}")

    for tag, sc in (("heuristic", None), ("trained-LR", scorer)):
        for qw in (0.005, 0.02, 0.05, 0.2, 0.5):
            res = eval_router(test, outcomes, sc, qw, enable_gate=False)
            q, c, pc = summarize(res)
            pgr = (q - ql) / max(qc - ql, 1e-9)
            print(f"router/{tag:10s} qw={qw:<5} quality={q:.4f} "
                  f"cost=${c:7.2f} cloud={pc:4.0%} PGR={pgr:6.3f}")
        # cascade variant: gate on
        res = eval_router(test, outcomes, sc, 0.05, enable_gate=True)
        q, c, pc = summarize(res)
        pgr = (q - ql) / max(qc - ql, 1e-9)
        print(f"router/{tag:10s} qw=0.05 +gate quality={q:.4f} "
              f"cost=${c:7.2f} cloud={pc:4.0%} PGR={pgr:6.3f}")


# ---------- Part B ----------

SYSTEM = Message("system", "You are a helpful assistant. " + "x " * 3000)


def run_sessions(sessions, outcomes, scorer=None, enable_gate=True,
                 turn_gap_s=60.0, label="", **cfg_kw):
    r = make_router(outcomes, scorer=scorer, enable_gate=enable_gate, **cfg_kw)
    book = CostBook(in_price=CLOUD_PRICING["in_p"],
                    out_price=CLOUD_PRICING["out_p"],
                    write_price=CLOUD_PRICING["write_p"],
                    read_price=CLOUD_PRICING["read_p"])
    tot_cost, cached_in, in_tok_total = 0.0, 0, 0
    scores, switches, cloud_calls = [], 0, 0
    for si, prompts in enumerate(sessions):
        sid = f"sess{si}"
        hist = [SYSTEM]
        last_model, t = None, 0.0
        for prompt in prompts:
            hist.append(Message("user", prompt))
            s = r.store.get(sid)
            s.last_active_ts = t
            resp, dec = r.handle(list(hist), session_id=sid)
            if dec.model.endswith("gpt4"):
                prefix = sum(len(m.content) for m in hist[:-1]) // 4
                in_toks = sum(len(m.content) for m in hist) // 4
                ch = book.charge(sid, dec.model, prefix, in_toks, 200, t)
                tot_cost += ch["cost"]
                cached_in += ch["warm"]
                in_tok_total += in_toks
                cloud_calls += 1
                scores.append(outcomes[prompt]["cloud"][1])
            else:
                scores.append(outcomes[prompt]["local"][1])
            if last_model is not None and dec.model != last_model:
                switches += 1
            last_model = dec.model
            hist.append(Message("assistant", resp.text))
            t += turn_gap_s
    n = sum(len(s) for s in sessions)
    print(f"{label:30s} quality={sum(scores)/n:.4f} cost=${tot_cost:7.2f} "
          f"cloud%={cloud_calls/n:3.0%} switches/sess={switches/len(sessions):5.2f} "
          f"cache-read%={cached_in/max(in_tok_total,1):3.0%}")


def part_b(test, outcomes, scorer):
    print("\n=== Part B: synthetic sessions — 6 turns, 60s gaps ===")
    rng = random.Random(1)
    pool = test["prompt_text"].tolist()
    sessions = [rng.sample(pool, 6) for _ in range(300)]

    run_sessions(sessions, outcomes, scorer, enable_gate=False,
                 quality_weight=0.0, label="always-local")
    run_sessions(sessions, outcomes, scorer, enable_gate=False,
                 quality_weight=1e9, label="always-cloud")
    run_sessions(sessions, outcomes, scorer, enable_gate=False,
                 quality_weight=0.05, switch_margin=0.0,
                 escalation_sticky=False, switch_history_penalty=0.0,
                 label="naive per-turn margin")
    run_sessions(sessions, outcomes, scorer, enable_gate=False,
                 quality_weight=0.05, switch_margin=0.005,
                 escalation_sticky=True,
                 label="session-aware margin")
    run_sessions(sessions, outcomes, scorer, enable_gate=False,
                 quality_weight=0.05, decision_mode="wfa",
                 escalation_sticky=True,
                 label="session-aware WFA")
    run_sessions(sessions, outcomes, scorer, enable_gate=True,
                 quality_weight=0.05, switch_margin=0.005,
                 escalation_sticky=True,
                 label="session-aware + cascade gate")


def main():
    import glob
    default_pkl = next(iter(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--withmartian--routerbench/"
        "snapshots/*/routerbench_0shot.pkl"))), None)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=default_pkl,
                    help="path to routerbench_0shot.pkl (auto-detected from "
                    "the HF cache; otherwise pass explicitly)")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()

    df = load(args.pkl, args.n)
    half = len(df) // 2
    train, test = df.iloc[:half], df.iloc[half:]
    print(f"loaded {len(df)} prompts (train={len(train)} test={len(test)})")

    outcomes = build_outcomes(test)
    scorer, acc = train_scorer(train)
    print(f"trained LR scorer: train-acc={acc:.3f}")

    if not args.skip_a:
        part_a(test, outcomes, scorer)
    if not args.skip_b:
        part_b(test, outcomes, scorer)


if __name__ == "__main__":
    main()
