#!/usr/bin/env python
"""Sample size for the paired TOST at +/-0.05, behind the paper's "114 to 204 query pairs".

The paper's Limitations, first paragraph: "an equivalence test at +/-0.05 needs at
least 114 to 204 query pairs for 80% power". Until this script, the figure lived
only in the 2026-09-23 power-analysis audit, so it was the one number in the
paper that no script regenerated.

**Input.** The run-to-run variation of the invalid rate on the hosted endpoint:
for each model, the published run against its replication on the same 50
queries, paired by query (ITT, as in `aggregate_arms.py`, which loads and
filters the rows). The SD of those per-query differences is the sigma of a
future paired test.

**Formula.** For a paired TOST at margin delta, one-sided alpha, and a true
difference of zero, n = (z_{1-alpha} + z_{1-beta/2})^2 sigma^2 / delta^2. The
beta/2 is required because equivalence rejects two one-sided nulls (the
2026-09-23 audit corrected an earlier z_{1-beta}, which gave 85 and 149).

**Check.** The formula is confirmed by simulation: paired TOSTs with
t-critical values at the computed n, with a true difference of zero. These
are floors; any true offset raises n.

Usage:

    PYTHONPATH=src python scripts/power_tost.py
    PYTHONPATH=src python scripts/power_tost.py --sims 20000
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("aggregate_arms", ROOT / "scripts" / "aggregate_arms.py")
agg = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_arms"] = agg
_spec.loader.exec_module(agg)

# (label, published run, replication run, model filter). The published runs
# predate the `arm` column, so rows are selected by model.
PAIRS = [
    ("LLaMA 3.1-70B", "eval_20260827_arm1-2_llama70b-8b_100q_topup", "final_20260922_arm1_llama70b_50q_fp8", "70b"),
    ("LLaMA 3.1-8B", "eval_20260827_arm1-2_llama70b-8b_100q_topup", "final_20260922_arm2_llama8b_50q_fp8", "8b"),
]


def paired_sd(published: str, replication: str, model: str) -> tuple[float, float, int]:
    a = agg.by_query(agg.load(ROOT / "outputs" / published, model=model), agg.itt)
    b = agg.by_query(agg.load(ROOT / "outputs" / replication, model=model), agg.itt)
    common = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in common]
    return st.stdev(diffs), st.fmean(diffs), len(common)


def required_n(sigma: float, margin: float, alpha: float, power: float) -> float:
    z = stats.norm.ppf
    return (z(1 - alpha) + z(1 - (1 - power) / 2)) ** 2 * sigma**2 / margin**2


def simulated_power(sigma: float, n: int, margin: float, alpha: float, sims: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, sigma, size=(sims, n))
    mean, se = d.mean(axis=1), d.std(axis=1, ddof=1) / math.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha, n - 1)
    return float(np.mean(((mean + margin) / se > t_crit) & ((margin - mean) / se > t_crit)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--sims", type=int, default=20000, help="simulated TOSTs per model (0 to skip)")
    args = ap.parse_args()

    print(f"Paired TOST, margin +/-{args.margin}, alpha {args.alpha} one-sided, power {args.power}, true difference 0\n")
    print(f"{'model':<15} {'pairs':>5} {'mean diff':>10} {'SD':>8} {'n (exact)':>10} {'n needed':>9} {'sim. power at n':>16}")
    for label, published, replication, model in PAIRS:
        sd, mean, pairs = paired_sd(published, replication, model)
        exact = required_n(sd, args.margin, args.alpha, args.power)
        n = math.ceil(exact)
        sim = simulated_power(sd, n, args.margin, args.alpha, args.sims, seed=34) if args.sims else float("nan")
        print(f"{label:<15} {pairs:>5} {mean:>+10.4f} {sd:>8.4f} {exact:>10.1f} {n:>9d} {sim:>16.3f}")
    print("\nThese are floors: a true difference other than zero raises the required n.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
