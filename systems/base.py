
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Prediction:
    example_id: str
    predicted_label: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class System(ABC):
    """Base class. Subclasses set `name` and implement fit() and predict()."""

    name: str = "unnamed"

    def config(self) -> dict:
        """Anything that would change results. Stored with the run."""
        return {}

    def fit(self, texts: list[str], labels: list[str]) -> None:
        """Optional. LLM systems typically no-op."""
        return None

    @abstractmethod
    def predict(self, ids: list[str], texts: list[str]) -> list[Prediction]:
        ...

    # -- helper for systems that genuinely work one example at a time --
    def _predict_looped(self, ids, texts, fn) -> list[Prediction]:
        out = []
        for eid, text in zip(ids, texts):
            t0 = time.perf_counter()
            label = fn(text)
            out.append(Prediction(eid, label, (time.perf_counter() - t0) * 1000))
        return out


def run_and_save(
    system: System,
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    results_dir: str | Path = "results",
) -> pd.DataFrame:
    """Fit, predict, join to truth, persist per-example rows. Returns the frame."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if train_df is not None:
        system.fit(train_df.text.tolist(), train_df.category.tolist())

    preds = system.predict(eval_df.example_id.tolist(), eval_df.text.tolist())
    pred_df = pd.DataFrame([asdict(p) for p in preds])

    df = eval_df.merge(pred_df, on="example_id", validate="one_to_one")
    assert len(df) == len(eval_df), "prediction/truth join lost rows"

    df = df.rename(columns={"category": "true_label"})
    df["correct"] = (df.predicted_label == df.true_label).astype(int)
    df["system"] = system.name

    cols = ["system", "example_id", "text", "true_label", "predicted_label",
            "correct", "latency_ms", "input_tokens", "output_tokens"]
    df[cols].to_csv(results_dir / f"{system.name}.csv", index=False)

    (results_dir / f"{system.name}.config.json").write_text(
        json.dumps({"system": system.name, "n": len(df), **system.config()}, indent=2)
    )
    return df[cols]
