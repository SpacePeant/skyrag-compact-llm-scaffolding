#!/usr/bin/env python
"""Hosted LLaMA 3.1-8B with the candidate list held fixed, for pre-submission review item A7.

WHY. Under candidate injection the candidates are ranked against the Stage 1
rewrite, the same text the selector then reads. So when the short rewrite (JSON)
lowered invention from 0.3860 to 0.1803, it had changed two things at once: the
text the selector reads and the 100 candidates it is shown. The review (A7)
asks that the two paths be separated, or the claim narrowed.

WHAT. The same two cells, with candidates ranked against the learner's original
query (`--candidate-source query`), so both see one list per query and differ
only in the rewrite:

    i_a2_cq     Arm 2, default (long) rewrite   pairs with fact_20260923_v5fp8_i_l0
    i_a2_s1_cq  Arm 2, short rewrite (JSON)     pairs with s1dec_20261001_v5fp8_i_a2_s1

Reading: if i_a2_s1_cq still invents clearly less than i_a2_cq, the rewrite
works through the text the selector reads. If the gap closes, it worked through
the candidate list.

COMMAND LINE. `launch_stage1_decomp_hosted.py`'s, plus `--candidate-source
query`. Each pass is its own lane (`-r 1`); pass 1 of both cells runs first.

Usage, from the repo root:

    python scripts/launch_a7_fixed_candidates_hosted.py --dry-run
    python scripts/launch_a7_fixed_candidates_hosted.py --smoke
    python scripts/launch_a7_fixed_candidates_hosted.py [--lanes N]
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
STAMP = "a7fix_v5fp8"
MODEL = "meta-llama/llama-3.1-8b-instruct"
PROVIDER = "deepinfra"
PRECISION = "fp8"
COMMON = [
    "-b", "openrouter", "-P", "skyrag",
    "--serving-precision", PRECISION, "--sweep-version", "v5",
    "-m", MODEL,
    "--candidate-injection", "--candidate-limit", "100",
    "--candidate-source", "query",
]
# (cell, arm, --expansion-mode or None, the expansion_mode rows must carry)
CELLS = [
    ("i_a2_cq", "2", None, "plain"),
    ("i_a2_s1_cq", "2", "schema", "schema"),
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
    for rep in range(1, (1 if smoke else PASSES) + 1):
        for cell, arm, mode, stamp_mode in CELLS:
            name = f"{cell}_smoke" if smoke else f"{cell}_r{rep}"
            args = ["-Q", queries, *COMMON, "--arm", arm]
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
        "candidate_source": "query",
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
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lanes", type=int, default=3)
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
            # A lane moved from another machine part-way: its rows so far sit in
            # `<lane>_partial/`, and `--resume` carries them and runs only the rest.
            partial = ROOT / "outputs" / f"{STAMP}_{name}_partial" / "benchmark_eval.csv"
            if partial.exists():
                cmd += ["--resume", str(partial)]
                print(f"  {name}: resuming from {partial.parent.name}", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] launch {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
            running[name] = (proc, stamp_mode, time.time())
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
        print("\nNext: merge the passes.")
        for cell, *_ in CELLS:
            lanes = " ".join(f"outputs/{STAMP}_{cell}_r{k}" for k in range(1, PASSES + 1))
            print(f"  python scripts/merge_lane_repeats.py {lanes} --output outputs/{STAMP}_{cell}_m3")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
