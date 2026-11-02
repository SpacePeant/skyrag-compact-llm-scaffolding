#!/usr/bin/env python
"""Stitch the query chunks of ONE pass back into one run directory.

`merge_lane_repeats.py` handles a *repeat* split (every lane runs the same
queries) and refuses a *query* split by design. This is the other half: lanes
that each ran a disjoint subset of the queries for the same pass, launched that
way only to cut wall-clock when the provider's per-query latency, not the
machine, is the bottleneck.

It refuses anything that is not unambiguously one pass split by query:

* every chunk must be one cell with the same treatment, judged by the same
  `TREATMENT_FIELDS` and rules `merge_lane_repeats.py` applies;
* chunks must not share a query, and no chunk may carry more than one repeat;
* with `--expect`, the union must be exactly the query set of that file.

Rows are copied verbatim. The output is one pass, ready to be stacked with its
sibling passes by `merge_lane_repeats.py`.

    python scripts/concat_query_chunks.py outputs/x_r1_q1 outputs/x_r1_q2 \
        --expect benchmarks/queries_v1.jsonl --output outputs/x_r1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("merge_lane_repeats", _HERE / "merge_lane_repeats.py")
mlr = importlib.util.module_from_spec(_spec)
sys.modules["merge_lane_repeats"] = mlr
_spec.loader.exec_module(mlr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chunks", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--expect", type=Path, help="query file whose ids the union must equal")
    args = ap.parse_args(argv)
    if len(args.chunks) < 2:
        raise SystemExit("give at least two chunks; one chunk is already a run directory.")

    lanes = [(p, mlr.load_lane(p)) for p in args.chunks]
    reference = None
    for path, rows in lanes:
        stamp = mlr.treatment_of(rows)
        if reference is None:
            reference = (path, stamp)
            continue
        for field in mlr.TREATMENT_FIELDS:
            mine, theirs = reference[1][field], stamp[field]
            if mine is mlr.UNRECORDED or theirs is mlr.UNRECORDED or mine != theirs:
                raise SystemExit(
                    f"chunks disagree on {field!r}: {reference[0].name}={mine!r}, "
                    f"{path.name}={theirs!r}. Chunks of one pass must be one cell."
                )

    seen: dict[str, Path] = {}
    for path, rows in lanes:
        repeats = {mlr._repeat_of(r) for r in rows}
        if len(repeats) != 1:
            raise SystemExit(f"{path.name} carries repeats {sorted(repeats)}; a chunk is one pass.")
        for r in rows:
            q = str(r["query_id"])
            if q in seen and seen[q] != path:
                raise SystemExit(f"query {q} is in both {seen[q].name} and {path.name}; not a query split.")
            seen[q] = path

    if args.expect:
        want = {json.loads(line)["id"] for line in args.expect.read_text(encoding="utf-8").splitlines() if line.strip()}
        missing, extra = want - set(seen), set(seen) - want
        if missing or extra:
            raise SystemExit(f"union does not match {args.expect.name}: missing {sorted(missing)}, extra {sorted(extra)}")

    merged = [r for _, rows in lanes for r in rows]
    mlr.write(args.output, merged)
    print(f"Wrote {args.output} ({len(merged)} rows from {len(lanes)} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
