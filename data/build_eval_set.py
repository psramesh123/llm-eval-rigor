"""
Build frozen evaluation and development sets from Banking77.

Design decisions (documented deliberately):
  - EVAL set (6/class = 462) is frozen and used ONLY for final reported runs.
  - DEV set (3/class = 231) is for prompt iteration, hyperparameter choices,
    retrieval-k selection. Disjoint from EVAL.
  - Both drawn from the official test split so the training split stays
    untouched for the classical and embedding systems.
  - Fixed seed. Sets are written to disk and committed so every run is
    reproducible and comparable.

Why small n: the statistical layer of this project exists because small
evaluation sets are where naive CLT-based error bars fail.
A 462-example set is realistic for a specialised benchmark and makes the
uncertainty quantification necessary rather than decorative.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

SEED = 20260903
EVAL_PER_CLASS = 6
DEV_PER_CLASS = 3

RAW = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in ("train", "test"):
        local = data_dir / f"{split}.csv"
        if not local.exists():
            print(f"downloading {split}.csv ...")
            urllib.request.urlretrieve(f"{RAW}/{split}.csv", local)
        frames[split] = pd.read_csv(local)
    return frames["train"], frames["test"]


def sanity_checks(train: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, df in (("train", train), ("test", test)):
        assert list(df.columns) == ["text", "category"], (name, df.columns)
        assert df.notna().all().all(), f"nulls in {name}"
    assert train.category.nunique() == test.category.nunique() == 77, "expected 77 classes"
    assert test.text.duplicated().sum() == 0, "duplicate texts in test"
    counts = test.category.value_counts()
    assert counts.min() == counts.max() == 40, "test split is no longer balanced"
    print(f"checks passed | train={len(train)} test={len(test)} classes=77 balanced=40/class")


def shuffle(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def stratified_split(test: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take EVAL_PER_CLASS then DEV_PER_CLASS from each class, disjoint."""
    eval_rows, dev_rows = [], []
    for _, g in shuffle(test, seed).groupby("category"):
        eval_rows.append(g.iloc[:EVAL_PER_CLASS])
        dev_rows.append(g.iloc[EVAL_PER_CLASS:EVAL_PER_CLASS + DEV_PER_CLASS])
    return shuffle(pd.concat(eval_rows), seed), shuffle(pd.concat(dev_rows), seed)


def add_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Stable example_id derived from the text, so IDs survive reordering."""
    df = df.assign(example_id=[
        f"{prefix}_{hashlib.sha1(t.encode()).hexdigest()[:10]}" for t in df.text
    ])
    assert df.example_id.duplicated().sum() == 0, "id collision"
    return df[["example_id", "text", "category"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_raw(data_dir)
    sanity_checks(train, test)

    eval_df, dev_df = stratified_split(test, SEED)
    eval_df, dev_df = add_ids(eval_df, "ev"), add_ids(dev_df, "dv")

    labels = sorted(train.category.unique())
    assert set(eval_df.text).isdisjoint(dev_df.text), "eval/dev overlap"
    for name, df, per_class in (("eval", eval_df, EVAL_PER_CLASS), ("dev", dev_df, DEV_PER_CLASS)):
        counts = df.category.value_counts().to_dict()
        assert counts == dict.fromkeys(labels, per_class), f"{name} not balanced"

    eval_df.to_csv(out_dir / "eval_set.csv", index=False)
    dev_df.to_csv(out_dir / "dev_set.csv", index=False)
    train.to_csv(out_dir / "train_set.csv", index=False)

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    manifest = {
        "seed": SEED,
        "eval_n": len(eval_df),
        "dev_n": len(dev_df),
        "train_n": len(train),
        "n_classes": len(labels),
        "eval_per_class": EVAL_PER_CLASS,
        "dev_per_class": DEV_PER_CLASS,
        "eval_sha1": hashlib.sha1(
            "".join(sorted(eval_df.example_id)).encode()
        ).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {out_dir}/eval_set.csv, dev_set.csv, train_set.csv, labels.json, manifest.json")


if __name__ == "__main__":
    main()
