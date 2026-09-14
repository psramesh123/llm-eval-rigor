"""
System 1: TF-IDF + logistic regression.
"""

from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from .base import Prediction, System


class TfidfLogReg(System):
    name = "tfidf_lr"

    def __init__(self, C: float = 10.0, seed: int = 20260903):
        self.C = C
        self.seed = seed
        self.pipe = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2),
                    sublinear_tf=True, min_df=1)),
                ("char", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5),
                    sublinear_tf=True, min_df=2)),
            ])),
            ("clf", LogisticRegression(
                C=C, max_iter=2000, random_state=seed)),
        ])

    def config(self) -> dict:
        return {
            "C": self.C,
            "seed": self.seed,
            "features": "word 1-2gram + char_wb 3-5gram, sublinear tf",
            "classifier": "LogisticRegression",
        }

    def fit(self, texts, labels) -> None:
        self.pipe.fit(texts, labels)

    def predict(self, ids, texts) -> list[Prediction]:
        # Batch for speed, then attribute mean latency per example.
        t0 = time.perf_counter()
        labels = self.pipe.predict(texts)
        per_example_ms = (time.perf_counter() - t0) * 1000 / len(texts)
        return [
            Prediction(eid, str(lab), per_example_ms)
            for eid, lab in zip(ids, labels)
        ]
