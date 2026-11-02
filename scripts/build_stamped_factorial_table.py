#!/usr/bin/env python
"""Emit Table 6 from the STAMPED factorial, one payload shape across all six cells.

WHY. The manuscript's Table 6 crosses context scaffolding with format
scaffolding, but its static rows are `v2` runs (Level 1 retrieves 24.9, which is
`arm3_A_spec_only_20260831`) while its Condition C rows are `v5` (its own note 2
says so). The context axis is therefore confounded with the payload shape, and
the table's context effect is context scaffolding AND the v2->v5 change,
inseparably. Section 6.4 already flags the payload as load-bearing and names the
replication that would settle it; the 2026-09-23 factorial IS that replication.

This script builds the table from those cells only, so every figure in it shares
a payload, a query set, a provider and a serving precision. It assembles nothing
by hand: every number comes from `aggregate_arms` metric functions.
"""

from __future__ import annotations

import importlib.util
import statistics as st
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("aggregate_arms", _ROOT / "scripts" / "aggregate_arms.py")
agg = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_arms"] = agg
_spec.loader.exec_module(agg)

STAMP = "fact_20260923_v5fp8"
ROWS = [
    ("Static Top-1,000 Window", "Level 0: Unassisted (Arm 2)", "s_l0"),
    ("", "Level 1: Bare Schema Spec (Arm 3)", "s_l1"),
    ("", "Level 2: Schema + Exemplar (Arm 3)", "s_l2"),
    ("Dynamic Candidate Injection", "Level 0: Unassisted (Arm 2)", "i_l0"),
    ("", "Level 1: Bare Schema Spec (Arm 3)", "i_l1"),
    ("", "Level 2: Schema + Exemplar (Arm 3)", "i_l2"),
]


def _dirs(cell: str) -> list[Path]:
    out = []
    for rep in ("r1", "r2", "r3"):
        for suffix in ("", "_resumed"):
            p = _ROOT / "outputs" / f"{STAMP}_{cell}_{rep}{suffix}"
            if (p / "benchmark_eval.json").exists():
                out.append(p)
                break
    return out


def _retrieved(row: dict) -> float:
    return float(row.get("retrieved_count") or 0)


def _per_query_mean(cells: list, fn) -> float:
    """Mean over queries of each query's mean over passes.

    This table first pooled every row of the three passes into one mean. That
    weights a query by how many of its passes count, so under As-Treated, where
    silent runs drop out, a query silent in two passes weighed a third of one
    that always answered, and under ITT the one unmeasured row (sci01, i_l2 r3)
    shifted the weights too. Tables 4 and 5 average per query
    (`aggregate_arms.by_query`), so Table 6 disagreed with them on the same
    cell: 0.6502 against 0.6477 for the static Level 0 As-Treated rate, the
    discrepancy the 2026-10-21 pre-submission review (A4) points to. Each directory is
    one pass, so `by_query` on it gives one value per query, and the passes are
    averaged here with undefined values skipped, as `by_query` does.
    """
    per_query: dict[str, list[float]] = {}
    for c in cells:
        for q, v in agg.by_query(c, fn).items():
            per_query.setdefault(q, []).append(v)
    return st.fmean(st.fmean(vs) for vs in per_query.values())


def main() -> int:
    # Six columns, per the manuscript anti-pattern guide: comparative tables are
    # capped at 4-6 so a 6.5-inch printable width does not word-wrap the data
    # columns to nothing. Survivorship deliberately lives in the companion token
    # table rather than here -- a `Silent Pipelines` column would restate that
    # table's `Surviving Cohort` in different arithmetic, which is the redundant
    # column the same guide warns against.
    print("| Context Scaffolding | Format Scaffolding Level | Invalid Keyword Rate (ITT) | "
          "Invalid Keyword Rate (As-Treated, Macro) | Mean Courses Retrieved | Passes |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")

    silence: list[tuple[str, str, int, int]] = []
    for ctx, level, cell in ROWS:
        dirs = _dirs(cell)
        if not dirs:
            print(f"| **{ctx}** | {level} | -- | -- | -- | 0 |")
            continue
        cells = [agg.load(d) for d in dirs]
        label = f"**{ctx}**" if ctx else ""
        print(f"| {label} | {level} | **{_per_query_mean(cells, agg.itt)*100:.2f}%** | "
              f"{_per_query_mean(cells, agg.both_spoke)*100:.2f}% | "
              f"{_per_query_mean(cells, _retrieved):.1f} | {len(dirs)} |")
        silence.append((ctx or silence[-1][0], level, sum(c.mutes() for c in cells), sum(c.n for c in cells)))

    print()
    print("All six cells: `--sweep-version v5`, `--serving-precision fp8`, provider "
          "`deepinfra`, `benchmarks/queries_v1.jsonl`, temperature 0. One payload "
          "shape across the whole table, so the context axis is not confounded with "
          "the payload version.")
    print("Every figure is a mean over queries: each query's passes are averaged "
          "first, as in Tables 4 and 5, so a query counts once however many of its "
          "passes were measured or answered.")

    # Usable output, the third of the three measures (the pre-submission review, A3).
    # Counted over runs, one query in one pass, because a silent run is an event
    # of a run: a query silent in one pass of three is answered in the other two.
    print()
    print("| Context Scaffolding | Format Scaffolding Level | Silent Runs | Usable-Output Rate (runs) |")
    print("| :--- | :--- | :---: | :---: |")
    for ctx, level, mutes, n in silence:
        print(f"| {ctx} | {level} | {mutes} of {n} | {(n - mutes) / n * 100:.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
