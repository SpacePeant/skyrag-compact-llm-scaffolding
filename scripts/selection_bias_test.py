#!/usr/bin/env python
"""Does the schema decline the queries the unconstrained baseline finds hard?

Section 4.7.2's selection-bias test, previously computed by hand (Welch
p = 0.7041 on `eval_20260908_arm2-3_condition_c_50q_r3_resumed`) and never
scripted, so it could not be re-run when the stamped injection cells reached
three passes.

Definition, as recorded in the 2026-09-11 metric audit,
item 19: split the queries by whether Arm 3 went silent (emitted no keyword) on
**at least one** pass, against queries it answered on **every** pass. Compare
Arm 2's per-query As-Treated invention rate (passes averaged, mutes skipped)
between the two groups with a Welch t-test. If the schema declined the queries
the baseline fails, the declined group would carry the higher Arm 2 rate.

A non-significant result is an absence of evidence for selection, not evidence
of its absence (DEC-11), and is printed as such.

    uv run python scripts/selection_bias_test.py \
        --arm2 outputs/run_a --arm3 outputs/run_b
    uv run python scripts/selection_bias_test.py \
        --arm2 outputs/mixed_run --arm2-select 2 --arm3 outputs/mixed_run --arm3-select 3
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics as st
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("aggregate_arms", _ROOT / "scripts" / "aggregate_arms.py")
agg = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_arms"] = agg
_SPEC.loader.exec_module(agg)


def pooled(dirs: list[Path], arm: str | None) -> "agg.Cell":
    """Several single-pass lanes as one cell, re-keyed so repeats stay distinct."""
    if len(dirs) == 1:
        return agg.load(dirs[0], arm=arm)
    rows = {
        (query_id, i + 1): row
        for i, d in enumerate(dirs)
        for (query_id, _rep), row in agg.load(d, arm=arm).rows.items()
    }
    return agg.Cell(f"pooled[{len(dirs)}]", rows)


def main(argv: list[str] | None = None) -> int:
    from scipy.stats import ttest_ind

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm2", type=Path, nargs="+", required=True)
    ap.add_argument("--arm3", type=Path, nargs="+", required=True)
    ap.add_argument("--arm2-select", default=None, help="arm filter within the Arm 2 run(s)")
    ap.add_argument("--arm3-select", default=None, help="arm filter within the Arm 3 run(s)")
    a = ap.parse_args(argv)

    base = pooled(a.arm2, a.arm2_select)
    schema = pooled(a.arm3, a.arm3_select)

    muted_any: set[str] = set()
    seen: set[str] = set()
    for (query_id, _rep), row in schema.rows.items():
        seen.add(query_id)
        if not (row.get("keywords_emitted") or 0):
            muted_any.add(query_id)
    answered_all = seen - muted_any

    base_rate = agg.by_query(base, agg.both_spoke)
    g_muted = [base_rate[q] for q in sorted(muted_any) if q in base_rate]
    g_answered = [base_rate[q] for q in sorted(answered_all) if q in base_rate]

    passes = len({rep for _, rep in schema.rows})
    print(f"Arm 3 passes: {passes}; queries: {len(seen)}")
    print(f"  Arm 3 silent on >=1 pass : n={len(g_muted):3d}  Arm 2 As-Treated mean="
          + (f"{st.mean(g_muted):.4f}" if g_muted else "--"))
    print(f"  Arm 3 answered every pass: n={len(g_answered):3d}  Arm 2 As-Treated mean="
          + (f"{st.mean(g_answered):.4f}" if g_answered else "--"))
    if len(g_muted) < 2 or len(g_answered) < 2:
        print("  too few queries in a group for a Welch test")
        return 0
    t, p = ttest_ind(g_muted, g_answered, equal_var=False)
    diff = st.mean(g_muted) - st.mean(g_answered)
    print(f"  difference (silent - answered) = {diff:+.4f}   Welch t={t:.3f}  p={p:.4f}")
    print("  reading: " + (
        "the declined queries are HARDER for the baseline -- selection is a live explanation"
        if p < 0.05 and diff > 0 else
        "no evidence of selection; an absence of evidence, not evidence of absence"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
