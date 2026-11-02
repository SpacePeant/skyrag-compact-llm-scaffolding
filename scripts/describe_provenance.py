#!/usr/bin/env python
"""Tabulate the run conditions stamped on every exported row, one line per run folder.

For each `outputs/<run>/benchmark_eval.json` it reports the rows, distinct
queries and passes, and every distinct value of the stamps a reader needs to
know what ran: model, provider, declared precision, sweep version, arm, rewrite
mode, candidate injection and limit, ranker, candidate source and run dates.
Missing stamps are printed as `(not stamped)`, never filled in.

Usage:

    python scripts/describe_provenance.py                 # Markdown table to stdout
    python scripts/describe_provenance.py --gaps-only     # only runs with a missing stamp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# (column header, row field). Model stamps are folded into one column below.
FIELDS = [
    ("Provider", "provider"),
    ("Precision", "serving_precision"),
    ("Sweep", "sweep_version"),
    ("Arm", "arm"),
    ("Rewrite", "expansion_mode"),
    ("Injection", "candidate_injection"),
    ("Limit", "candidate_limit"),
    ("Ranker", "candidate_ranker"),
    ("Candidate source", "candidate_source"),
]
REQUIRED = ("provider", "serving_precision")
MISSING = "(not stamped)"


def _values(rows: list[dict], field: str) -> str:
    present = [r for r in rows if field in r]
    if not present:
        return MISSING
    vals = sorted({str(r[field]) if r[field] not in (None, "") else "" for r in present})
    shown = ["blank" if v == "" else v for v in vals]
    if len(present) < len(rows):
        shown.append(MISSING)
    return ", ".join(shown)


def describe(path: Path) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(data, list) and data and isinstance(data[0], dict) and "query" in data[0]):
        return None
    models = sorted({str(r.get(k)) for r in data for k in ("model", "keyword_model", "generation_model") if r.get(k)})
    passes = sorted({str(r.get("lane_original_repeat") or r.get("repeat") or "") for r in data} - {""})
    dates = sorted({str(r.get("run_timestamp", ""))[:10] for r in data} - {""})
    out = {
        "Run": str(path.parent.relative_to(OUTPUTS)).replace("\\", "/"),
        "Rows": str(len(data)),
        "Queries": str(len({r.get("query_id") or r.get("query") for r in data})),
        "Passes": ", ".join(passes) or MISSING,
        "Models": ", ".join(m.split("/")[-1] for m in models) or MISSING,
    }
    for header, field in FIELDS:
        out[header] = _values(data, field)
    out["Dates"] = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else (dates[0] if dates else MISSING)
    out["_gap"] = any(MISSING in _values(data, f) or _values(data, f) == "blank" for f in REQUIRED)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gaps-only", action="store_true", help="only runs missing a provider or precision stamp")
    args = ap.parse_args()

    runs = [d for p in sorted(OUTPUTS.rglob("benchmark_eval.json")) if (d := describe(p))]
    if args.gaps_only:
        runs = [r for r in runs if r["_gap"]]
    headers = ["Run", "Rows", "Queries", "Passes", "Models"] + [h for h, _ in FIELDS] + ["Dates"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(" :--- " for _ in headers) + "|")
    for r in runs:
        print("| " + " | ".join(f"`{r[h]}`" if h == "Run" else r[h] for h in headers) + " |")
    print(f"\n{len(runs)} run folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
