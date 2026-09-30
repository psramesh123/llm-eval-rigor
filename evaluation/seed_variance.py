"""
Training-seed variance, and what it does to a reported comparison.

A significance test on a single checkpoint conditions on that checkpoint. It
answers: given THIS trained model, does it beat the baseline across the
population of examples? The inference is over examples, with n = |eval| and
n = 1 model.

The claim usually being made is about the METHOD: does fine-tuning this
architecture beat the baseline on this task? That is a claim about the
distribution of models the training procedure produces, and a single
checkpoint is one draw from it. Reporting the draw as the parameter treats a
random effect as fixed.

This module retrains under several seeds with everything else held constant,
scores each checkpoint on EVAL, and decomposes the uncertainty:

    sigma^2_examples   shrinks like 1/n as the evaluation set grows
    sigma^2_training   does not shrink with n at all

The second component is why "collect more evaluation data" has a floor. Past
some n, the dominant uncertainty is which model you happened to train, and no
quantity of additional evaluation examples reduces it.

Per-seed results are written through the same run_and_save() path every other
system uses, so they are readable by significance.load_results() and carry the
usual per-run config sidecar.

Usage
    python -m evaluation.seed_variance --seeds 11 22 33 44 55
    python -m evaluation.seed_variance --report-only      # reuses scored runs
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from evaluation.significance import ALPHA, load_results
from systems.base import run_and_save

SEED_DIR = Path("artifacts/seeds")
Z = norm.ppf(1 - ALPHA / 2)


def _r(x: float) -> float:
    """Round for reporting. float() keeps numpy scalars out of the JSON and repr."""
    return float(round(x, 4))


def _banner(text: str) -> None:
    print(f"\n{'=' * len(text)}\n{text}\n{'=' * len(text)}")


# ---------------------------------------------------------------- training --
def train_seeds(seeds: list[int], force: bool = False) -> None:
    """Retrain once per seed. Only the seed varies."""
    from systems.transformer import has_checkpoint, train

    for s in seeds:
        ckpt = SEED_DIR / f"seed_{s}"
        if has_checkpoint(ckpt) and not force:
            print(f"seed {s}: checkpoint exists, skipping (use --force to retrain)")
            continue
        _banner(f"seed {s}")
        train(seed=s, ckpt=ckpt)


def score_seeds(seeds: list[int], split: str = "eval") -> None:
    """Score every seed checkpoint on the split, persisting per-example results."""
    import torch
    from systems.transformer import TransformerClassifier, get_device, has_checkpoint

    eval_df = pd.read_csv(f"data/processed/{split}_set.csv")

    for s in seeds:
        ckpt = SEED_DIR / f"seed_{s}"
        if not has_checkpoint(ckpt):
            print(f"seed {s}: no complete checkpoint, skipping")
            continue
        sysobj = TransformerClassifier(ckpt=ckpt)
        sysobj.name = f"seed_{s}"
        df = run_and_save(sysobj, eval_df, results_dir=SEED_DIR, split=split)
        print(f"seed {s}: {split} accuracy {df.correct.mean():.4f}")

        # Release before loading the next 266MB checkpoint, so the peak stays
        # at one model rather than two, and hand the cache back to the driver.
        # torch.cuda / torch.mps expose empty_cache(); torch.cpu does not.
        del sysobj, df
        backend = getattr(torch, get_device(), None)
        if hasattr(backend, "empty_cache"):
            backend.empty_cache()


# ---------------------------------------------------------------- analysis --
def variance_components(accs: np.ndarray, n_eval: int,
                        baseline_acc: float) -> dict:
    """Split uncertainty into example-sampling and training-seed components."""
    mean_acc = accs.mean()
    sd_seed = accs.std(ddof=1)                       # across training runs
    delta = mean_acc - baseline_acc

    # Example-sampling SE for one checkpoint, binomial approximation.
    se_examples = sqrt(mean_acc * (1 - mean_acc) / n_eval)
    se_both = sqrt(se_examples**2 + sd_seed**2)      # model treated as a draw

    # n at which example-sampling SE falls below the training-seed SD:
    # beyond this, more evaluation data cannot reduce the dominant uncertainty.
    n_crossover = (mean_acc * (1 - mean_acc) / sd_seed**2) if sd_seed > 0 else np.inf

    p_cond, ci_cond = _inference(delta, se_examples)
    p_both, ci_both = _inference(delta, se_both)
    return {
        "n_seeds": len(accs),
        "seed_accuracies": accs.round(4).tolist(),
        "mean_accuracy": _r(mean_acc),
        "sd_across_seeds": _r(sd_seed),
        "range_across_seeds": _r(accs.max() - accs.min()),
        "baseline_accuracy": float(baseline_acc),
        "delta_vs_baseline": _r(delta),
        "se_examples": _r(se_examples),
        "se_examples_and_training": _r(se_both),
        "p_conditioning_on_checkpoint": p_cond,
        "p_treating_model_as_draw": p_both,
        "ci_conditional": ci_cond,
        "ci_unconditional": ci_both,
        "n_crossover": None if np.isinf(n_crossover) else int(round(n_crossover)),
    }


def _inference(delta: float, se: float) -> tuple[float, list[float]]:
    """Two-sided p-value and CI for an estimate with the given standard error."""
    return _r(2 * norm.sf(abs(delta / se))), [_r(delta - Z * se), _r(delta + Z * se)]


def report(res: dict, n_eval: int) -> None:
    _banner(f"TRAINING-SEED VARIANCE ({res['n_seeds']} seeds, n_eval = {n_eval:,})")
    print(f"  accuracies        {res['seed_accuracies']}")
    print(f"  mean              {res['mean_accuracy']:.4f}")
    print(f"  SD across seeds   {res['sd_across_seeds']:.4f}")
    print(f"  range             {res['range_across_seeds']:.4f}")

    print(f"\nvs baseline ({res['baseline_accuracy']:.4f})")
    print(f"  delta                        {res['delta_vs_baseline']:+.4f}")
    print(f"  SE, example sampling only    {res['se_examples']:.4f}")
    print(f"  SE, + training variance      {res['se_examples_and_training']:.4f}")

    pct = f"{100 * (1 - ALPHA):.0f}%"
    print(f"\n  conditioning on one checkpoint: "
          f"{pct} CI {res['ci_conditional']}  p = {res['p_conditioning_on_checkpoint']:.4f}")
    print(f"  model treated as a draw:        "
          f"{pct} CI {res['ci_unconditional']}  p = {res['p_treating_model_as_draw']:.4f}")

    changed = ((res["p_conditioning_on_checkpoint"] < ALPHA)
               != (res["p_treating_model_as_draw"] < ALPHA))
    print(f"\n  >> the conclusion {'CHANGES' if changed else 'is unchanged'} "
          f"once training variance is included")

    if res["n_crossover"]:
        print(f"\n  example-sampling SE drops below the training-seed SD at "
              f"n ~ {res['n_crossover']:,}.")
        print("  Beyond that, more evaluation data cannot reduce the dominant uncertainty.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--baseline", default="tfidf_lr")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        SEED_DIR.mkdir(parents=True, exist_ok=True)
        train_seeds(args.seeds, force=args.force)
        score_seeds(args.seeds, args.split)

    try:
        mat = load_results(args.split, results_dir=SEED_DIR)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{e}\n\nRun without --report-only to train and score the seeds first.")

    base = load_results(args.split, systems=[args.baseline])
    n_eval = len(base)
    res = variance_components(mat.mean().to_numpy(), n_eval,
                              float(base[args.baseline].mean()))

    report(res, n_eval)
    out = Path("results/analysis") / f"seed_variance_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "baseline": args.baseline,
                               "n_eval": n_eval, **res}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
