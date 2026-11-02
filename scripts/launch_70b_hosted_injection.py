#!/usr/bin/env python
"""Hosted LLaMA 3.1-70B under candidate injection, for pre-submission review item A1.

WHY. The paper compares LLaMA 3.1-8B *with* candidate injection and the short
rewrite (JSON) against LLaMA 3.1-70B *without* them. The 2026-10-21
pre-submission review (A1) asks for the 70B with the same scaffolding, "using
comparable queries and serving conditions". The 2026-10-20 Colab run answered
the first half (int4 on an A100: 0.0136 with injection and the short rewrite),
but not the second: its unassisted static baseline was 0.2220 against 0.3857
hosted, so int4 on vLLM and fp8 on DeepInfra are not interchangeable. These two
cells put the 70B on the provider and precision every 8B cell in the paper ran
on.

WHAT. Two cells, three passes each, LLaMA 3.1-70B for all three stages:

    i_a2     Arm 2, candidate injection, the arm's own (long) rewrite
             pairs with fact_20260923_v5fp8_i_l0          (8B: 0.3860)
    i_a2_s1  Arm 2, candidate injection, short rewrite (JSON)
             pairs with s1dec_20261001_v5fp8_i_a2_s1      (8B: 0.1803)

COMMAND LINE. Identical to `launch_stage1_decomp_hosted.py` (and so to
`launch_e5_factorial_repeats.sh`) except `-m`, so each cell pairs query by query
with its 8B counterpart. `-m` routes rewrite, selection and generation to the
70B (`pipelines/eval.py`), so the `.env` keyword and expansion models, which
are 8B, do not apply. Each pass is its own lane (`-r 1`), merged afterwards
with `merge_lane_repeats.py`. Pass 1 of both cells runs before any pass 2, so
a run cut short still leaves each cell one complete pass.

AUDIT. A lane counts only if its CSV has 50 rows, its log never printed the
embedder refusal, and every row carries the model, provider, precision,
injection settings and rewrite it was asked for.

Usage, from the repo root (needs OPENROUTER_API_KEY in .env):

    python scripts/launch_70b_hosted_injection.py --dry-run   # print the commands, spend nothing
    python scripts/launch_70b_hosted_injection.py --smoke     # one query per cell, about a cent
    python scripts/launch_70b_hosted_injection.py             # the run: 2 cells x 3 passes
    python scripts/launch_70b_hosted_injection.py --lanes 6   # all six lanes at once

Rerunning skips lanes whose output directory already exists, so an interrupted
night resumes where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = "c70h_v5fp8"
MODEL = "meta-llama/llama-3.1-70b-instruct"
PROVIDER = "deepinfra"
PRECISION = "fp8"
COMMON = [
    "-b", "openrouter", "-P", "skyrag",
    "--serving-precision", PRECISION, "--sweep-version", "v5",
    "-m", MODEL,
]
INJ = ["--candidate-injection", "--candidate-limit", "100"]
# (cell, arm, --expansion-mode or None, the expansion_mode rows must carry).
# i_a2 passes no --expansion-mode, exactly as its 8B pair did, so the arm's
# default rewrite applies. The current code stamps that default as `plain`, as
# the 2026-10-20 Colab 70B lanes show; the 8B lanes predate the column.
CELLS = [
    ("i_a2", "2", None, "plain"),
    ("i_a2_s1", "2", "schema", "schema"),
]
PASSES = 3
EXPECTED_ROWS = 50

ENV = {
    **os.environ,
    "SKYRAG_OPENROUTER_PROVIDER": PROVIDER,
    "PYTHONPATH": str(ROOT / "src"),
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def jobs(smoke: bool):
    queries = "benchmarks/queries_1q_smoke.jsonl" if smoke else "benchmarks/queries_v1.jsonl"
    passes = 1 if smoke else PASSES
    for rep in range(1, passes + 1):
        for cell, arm, mode, stamp_mode in CELLS:
            name = f"{cell}_smoke" if smoke else f"{cell}_r{rep}"
            args = ["-Q", queries, *COMMON, *INJ, "--arm", arm]
            if mode:
                args += ["--expansion-mode", mode]
            yield name, stamp_mode, args


def audit(name: str, stamp_mode: str, expected_rows: int) -> str:
    out = ROOT / "outputs" / f"{STAMP}_{name}"
    log = ROOT / "outputs" / f"lane_{STAMP}_{name}.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    if "NOT VALID AS RESULTS" in text.replace("\r", "\n"):
        return "INVALID (embedder fallback; delete the output directory and rerun)"
    csv_path = out / "benchmark_eval.csv"
    if not csv_path.exists():
        return "FAILED (no benchmark_eval.csv)"
    csv.field_size_limit(10**8)
    rows = list(csv.DictReader(io.open(csv_path, encoding="utf-8")))
    expect = {
        "model": MODEL,
        "keyword_model": MODEL,
        "provider": PROVIDER,
        "serving_precision": PRECISION,
        "candidate_injection": "True",
        "candidate_limit": "100",
        "expansion_mode": stamp_mode,
    }
    for col, want in expect.items():
        seen = {r.get(col, "<absent>") for r in rows}
        if seen != {want}:
            return f"WRONG {col} (rows carry {sorted(seen)}, expected {want!r})"
    if len(rows) != expected_rows:
        return f"SHORT ({len(rows)}/{expected_rows} rows)"
    failed = sum(r.get("status") in ("cell_failed", "arm_failed") for r in rows)
    return f"OK ({len(rows)} rows, {failed} failed)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument("--smoke", action="store_true", help="one query per cell, to check the setup")
    ap.add_argument("--lanes", type=int, default=3,
                    help="lanes at once (default 3, the WinError 1455 ceiling on the laptop)")
    ap.add_argument("--only", default="",
                    help="comma-separated lane names to run (e.g. i_a2_cq_r3), to split one run "
                         "across machines; the others are left alone")
    args = ap.parse_args()

    expected_rows = 1 if args.smoke else EXPECTED_ROWS
    queue = list(jobs(args.smoke))
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        unknown = wanted - {name for name, _, _ in queue}
        if unknown:
            raise SystemExit(f"unknown lane(s): {sorted(unknown)}")
        queue = [j for j in queue if j[0] in wanted]
    if args.dry_run:
        for name, _, a in queue:
            print(f"SKYRAG_OPENROUTER_PROVIDER={PROVIDER} uv run skyrag eval {' '.join(a)} "
                  f"-o outputs/{STAMP}_{name}/")
        return 0

    running: dict[str, tuple[subprocess.Popen, str, float]] = {}
    results: dict[str, str] = {}
    while queue or running:
        while queue and len(running) < args.lanes:
            name, stamp_mode, a = queue.pop(0)
            out = ROOT / "outputs" / f"{STAMP}_{name}"
            if out.exists():
                results[name] = audit(name, stamp_mode, expected_rows)
                print(f"  SKIP {name}: {out.name} exists -> {results[name]}", flush=True)
                continue
            log = open(ROOT / "outputs" / f"lane_{STAMP}_{name}.log", "w", encoding="utf-8")
            cmd = [sys.executable, "-m", "skyrag.cli", "eval", *a, "-o", f"outputs/{STAMP}_{name}/"]
            print(f"[{time.strftime('%H:%M:%S')}] launch {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
            running[name] = (proc, stamp_mode, time.time())
            # Staggered: loading the corpus and the embedder is the memory peak.
            if queue:
                time.sleep(90)
        for name, (proc, stamp_mode, t0) in list(running.items()):
            if proc.poll() is not None:
                del running[name]
                results[name] = audit(name, stamp_mode, expected_rows)
                print(f"[{time.strftime('%H:%M:%S')}] done {name} exit={proc.returncode} "
                      f"after {(time.time() - t0) / 3600:.2f} h: {results[name]}", flush=True)
        time.sleep(30)

    print("\nSUMMARY")
    for name, verdict in results.items():
        print(f"  {name}: {verdict}")
    ok = all(v.startswith("OK") for v in results.values())
    if ok and not args.smoke:
        print("\nNext: merge the passes, then compare with the 8B cells.")
        for cell, *_ in CELLS:
            lanes = " ".join(f"outputs/{STAMP}_{cell}_r{k}" for k in range(1, PASSES + 1))
            print(f"  python scripts/merge_lane_repeats.py {lanes} --output outputs/{STAMP}_{cell}_m3")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
