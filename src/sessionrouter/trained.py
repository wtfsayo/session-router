"""Trained scorer artifact: persist/load a fitted probability head
(e.g. sklearn LogisticRegression over `extract_features`) so Router can
run a real P(local suffices) model instead of the heuristic default.

Train offline — see bench/train_scorer.py — then:

    scorer = load_scorer("artifacts/scorer.pkl")
    Router(models, backends, scorer=scorer)
"""
from __future__ import annotations

import pickle
from pathlib import Path

from .features import extract_features
from .types import Message

FORMAT_VERSION = 1


class TrainedScorer:
    """Wraps any estimator with predict_proba over extract_features."""

    def __init__(self, clf, feature_dim: int = 17):
        self.clf = clf
        self.feature_dim = feature_dim

    def local_sufficiency(self, messages: list[Message],
                          category: str = "") -> float:
        x = extract_features(messages)
        if len(x) != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch: scorer expects {self.feature_dim}, "
                f"extract_features produced {len(x)} — retrain the artifact")
        return float(self.clf.predict_proba([x])[0][1])


def save_scorer(clf, path: str | Path,
                feature_dim: int = 17, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"version": FORMAT_VERSION, "clf": clf,
                     "feature_dim": feature_dim, "meta": meta or {}}, f)
    return path


def load_scorer(path: str | Path) -> TrainedScorer:
    with Path(path).open("rb") as f:
        blob = pickle.load(f)
    if blob.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported scorer artifact version "
                         f"{blob.get('version')}")
    return TrainedScorer(blob["clf"], blob.get("feature_dim", 17))
