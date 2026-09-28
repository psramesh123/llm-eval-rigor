"""
Paired significance testing across systems.

Every system is scored on the identical EVAL set, so comparisons are PAIRED.

Three methods, each chosen deliberately:

  McNemar (exact binomial)
      The correct test for paired binary outcomes. Only the DISCORDANT cells
      carry information: cases where one system was right and the other wrong.
      The exact binomial form is used rather than the chi-square approximation,
      because discordant counts here are small enough that the approximation
      is unreliable.

  Paired bootstrap
      Resamples examples with replacement and recomputes the paired accuracy
      difference each time. Used instead of a normal approximation because
      CLT-based intervals underestimate uncertainty on benchmarks of this size
      (arXiv:2503.01747).

  Benjamini-Hochberg
      k systems give k(k-1)/2 comparisons. At alpha = 0.05 the chance of at
      least one false positive across them is well above 5%, so p-values are
      reported both raw and FDR-adjusted.

Effect size is reported as the raw accuracy difference (most interpretable)
alongside the McNemar odds ratio b/c (the natural paired measure).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

SEED = 20260903
N_BOOT = 10_000
ALPHA = 0.05


def load_results(split: str = "eval",
                 results_dir: str | Path = "results",
                 data_dir: str | Path = "data/processed",
                 systems: list[str] | None = None) -> pd.DataFrame:
    """Join every system's per-example results into one correctness matrix."""
    results_dir, data_dir = Path(results_dir) / split, Path(data_dir)
    split_file = data_dir / f"{split}_set.csv"

    if not split_file.exists():
        raise FileNotFoundError(
            f"{split_file} not found. Run data/build_eval_set.py first.")
    expected = set(pd.read_csv(split_file).example_id)

    files = sorted(results_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no results in {results_dir}. Run: python run.py <system> --split {split}")

    frames, problems = {}, []
    for f in files:
        name = f.stem
        if systems and name not in systems:
            continue
        d = pd.read_csv(f)
        got = set(d.example_id)
        if got != expected:
            problems.append(
                f"  {name}: {len(got):,} examples, expected {len(expected):,} "
                f"({len(expected - got):,} missing, {len(got - expected):,} unexpected)")
            continue
        if d.example_id.duplicated().any():
            problems.append(f"  {name}: duplicate example_ids")
            continue
        frames[name] = d.set_index("example_id")["correct"]

    if problems:
        raise ValueError(
            f"results do not match the {split} split as defined in {split_file}:\n"
            + "\n".join(problems)
            + "\n\nRe-run those systems, or rebuild the split if it changed.")
    if not frames:
        raise ValueError(f"no usable results in {results_dir}")

    mat = pd.DataFrame(frames)
    assert not mat.isna().any().any(), "unreachable: ids verified above"
    return mat.astype(int)

def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar. a, b are 0/1 correctness vectors, aligned by example."""
    b_only = int(((a == 0) & (b == 1)).sum())   # b right, a wrong
    a_only = int(((a == 1) & (b == 0)).sum())   # a right, b wrong
    n_disc = a_only + b_only
    if n_disc:
        odds_ratio = b_only / a_only if a_only else np.inf
        p_value = float(binomtest(b_only, n_disc, 0.5).pvalue)
    else:
        odds_ratio, p_value = np.nan, 1.0       # ratio undefined, no evidence
    return {
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": n_disc,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                     alpha: float = ALPHA, seed: int = SEED) -> tuple[float, float]:
    """Percentile CI on the paired accuracy difference (b - a).

    The per-example difference only ever takes the values -1, 0 and +1, so a
    resample's mean depends solely on how many of each it draws. Drawing those
    counts from a multinomial is an exact reformulation of resampling examples
    one at a time, without materialising an (n_boot, n) index matrix.
    """
    rng = np.random.default_rng(seed)
    diff = b - a                                  # per-example paired difference
    n = len(diff)
    values, counts = np.unique(diff, return_counts=True)
    boot = rng.multinomial(n, counts / n, size=n_boot) @ values / n
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compare_all(mat: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """All pairwise comparisons, BH-adjusted."""
    if len(mat.columns) < 2:
        raise ValueError(
            f"need >= 2 systems to compare, found {list(mat.columns)}")
    rows = []
    for x, y in combinations(mat.columns, 2):
        a, b = mat[x].to_numpy(), mat[y].to_numpy()
        mc = mcnemar(a, b)
        ci_low, ci_high = paired_bootstrap(a, b, n_boot=n_boot)
        acc_a, acc_b = float(a.mean()), float(b.mean())
        rows.append({
            "system_a": x, "system_b": y,
            "acc_a": acc_a, "acc_b": acc_b,
            "delta": acc_b - acc_a,
            "ci_low": ci_low, "ci_high": ci_high,
            "a_only": mc["a_only"], "b_only": mc["b_only"],
            "n_discordant": mc["n_discordant"],
            "odds_ratio": mc["odds_ratio"],
            "p_raw": mc["p_value"],
        })
    df = pd.DataFrame(rows)
    df["p_adj"] = false_discovery_control(df.p_raw.to_numpy(), method="bh")
    df["significant"] = df.p_adj < ALPHA
    return df


def report(mat: pd.DataFrame) -> pd.DataFrame:
    print(f"n = {len(mat):,} examples, {len(mat.columns)} systems\n")
    print("Accuracy")
    for s, acc in mat.mean().sort_values(ascending=False).items():
        print(f"  {s:16}{acc:.4f}")

    df = compare_all(mat)
    print(f"\nPairwise comparisons ({len(df)} pairs, Benjamini-Hochberg adjusted)\n")
    hdr = (f"{'comparison':34}{'delta':>9}{f'{100 * (1 - ALPHA):.0f}% CI':>20}"
           f"{'disc':>7}{'OR':>7}{'p_raw':>9}{'p_adj':>9}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for _, r in df.iterrows():
        label = f"{r.system_a[:15]} vs {r.system_b[:15]}"
        ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        orv = f"{r.odds_ratio:.2f}" if np.isfinite(r.odds_ratio) else str(r.odds_ratio)
        verdict = "significant" if r.significant else "not significant"
        print(f"{label:34}{r.delta:+9.4f}{ci:>20}"
              f"{r.n_discordant:>7}{orv:>7}{r.p_raw:>9.4f}{r.p_adj:>9.4f}  {verdict}")
    return df


if __name__ == "__main__":
    # Written outside results/ so a later load_results() glob cannot read it back
    # in as if it were a system.
    out = Path("results/analysis/comparisons.csv")
    df = report(load_results())
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
