"""
Build four disjoint data pools from Banking77.

  TRAIN (~9,000)  fitting TF-IDF and fine-tuning the transformer
  VAL   (~1,000)  epoch selection, early stopping, hyperparameters
  DEV   (385)     prompt iteration, retrieval-k selection
  EVAL  (2,695)   the reported number; final runs only

The separation is the integrity of the project. Rules, ordered by how badly
violating them would matter:

  1. Nothing is ever tuned on EVAL. Not a prompt, not an epoch count.
  2. Transformer epoch selection uses VAL, carved from TRAIN. Never DEV or EVAL.
  3. Prompt and retrieval-k iteration uses DEV only.
  4. Every system is evaluated on the identical EVAL set, with per-example
     results joined on example_id.

EVAL is deliberately the full remainder of the official test split rather than a
subsample. Sample size is the object of study here, not a constraint: the
headline experiment subsamples EVAL to measure how reliably a benchmark of size
n recovers the full-data ranking. Withholding data to make a difference
detectable would be engineering the result.

Fixed seed. All pools written to disk and committed so results are reproducible.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

SEED = 20260903
DEV_PER_CLASS = 5      # EVAL takes the remaining 35/class
VAL_FRACTION = 0.10    # stratified holdout from TRAIN for epoch selection

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


def split_test(test: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DEV takes DEV_PER_CLASS from each class; EVAL takes everything else."""
    dev_rows, eval_rows = [], []
    for _, g in shuffle(test, seed).groupby("category"):
        dev_rows.append(g.iloc[:DEV_PER_CLASS])
        eval_rows.append(g.iloc[DEV_PER_CLASS:])
    return shuffle(pd.concat(eval_rows), seed), shuffle(pd.concat(dev_rows), seed)


def split_train(train: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified VAL holdout from TRAIN, for epoch selection only."""
    train_rows, val_rows = [], []
    for _, g in shuffle(train, seed).groupby("category"):
        n_val = max(1, round(len(g) * VAL_FRACTION))
        val_rows.append(g.iloc[:n_val])
        train_rows.append(g.iloc[n_val:])
    return shuffle(pd.concat(train_rows), seed), shuffle(pd.concat(val_rows), seed)


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

    train_full, test = load_raw(data_dir)
    sanity_checks(train_full, test)

    eval_df, dev_df = split_test(test, SEED)
    train_df, val_df = split_train(train_full, SEED)

    eval_df, dev_df = add_ids(eval_df, "ev"), add_ids(dev_df, "dv")
    train_df, val_df = add_ids(train_df, "tr"), add_ids(val_df, "va")

    labels = sorted(train_full.category.unique())

    # every pool disjoint from every other
    pools = {"train": train_df, "val": val_df, "dev": dev_df, "eval": eval_df}
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(pools[a].text).isdisjoint(pools[b].text), f"{a}/{b} overlap"

    # dev and eval balanced by construction; train/val only need full coverage
    for name, per_class in (("dev", DEV_PER_CLASS), ("eval", 40 - DEV_PER_CLASS)):
        counts = pools[name].category.value_counts().to_dict()
        assert counts == dict.fromkeys(labels, per_class), f"{name} not balanced"
    for name in ("train", "val"):
        assert pools[name].category.nunique() == 77, f"{name} missing classes"

    for name, df in pools.items():
        df.to_csv(out_dir / f"{name}_set.csv", index=False)

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    manifest = {
        "seed": SEED,
        "n_classes": len(labels),
        **{f"{k}_n": len(v) for k, v in pools.items()},
        "dev_per_class": DEV_PER_CLASS,
        "eval_per_class": 40 - DEV_PER_CLASS,
        "val_fraction": VAL_FRACTION,
        "eval_sha1": hashlib.sha1(
            "".join(sorted(eval_df.example_id)).encode()
        ).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(f"\nwrote train/val/dev/eval sets + labels.json + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
