"""Run one system against the frozen eval set and persist per-example results."""
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from systems.base import run_and_save
from systems.tfidf_lr import TfidfLogReg

REGISTRY = {"tfidf_lr": TfidfLogReg}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("system", choices=list(REGISTRY))
    p.add_argument("--split", default="eval", choices=["eval", "dev"])
    args = p.parse_args()

    eval_df = pd.read_csv(f"data/processed/{args.split}_set.csv")
    train_df = pd.read_csv("data/processed/train_set.csv")

    sysobj = REGISTRY[args.system]()
    df = run_and_save(sysobj, eval_df, train_df)

    acc = accuracy_score(df.true_label, df.predicted_label)
    f1 = f1_score(df.true_label, df.predicted_label, average="macro")
    print(f"\n{sysobj.name} on {args.split} (n={len(df)})")
    print(f"  accuracy   {acc:.4f}")
    print(f"  macro-F1   {f1:.4f}")
    print(f"  mean latency {df.latency_ms.mean():.2f} ms/example")
    print(f"  wrote results/{sysobj.name}.csv")

if __name__ == "__main__":
    main()
