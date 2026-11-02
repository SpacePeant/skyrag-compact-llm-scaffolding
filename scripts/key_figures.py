#!/usr/bin/env python
"""Every number the paper's rewritten sections cite, regenerated from artifacts.

WHY. Sections 4.4, 4.7.1, 5.1, 6.4 and 7 and the abstract were revised on
2026-09-29 to report the stamped factorial instead of the mixed-payload table
that preceded it. Those revisions cite paired effect sizes and p-values inline.
Hand-copying them on each new pass is exactly the manual table assembly the
reporting standard forbids, and it is how a figure comes to appear in prose that
no artifact supports.

So this prints them, labelled by the section that quotes each one. Run it after
any new pass and reconcile the manuscript against the output.

It assembles nothing itself: every figure comes from `merge_lane_repeats` (for
pooling passes) and `aggregate_arms` (for the paired tests), the same two tools
the manuscript's other numbers come from.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("aggregate_arms", _ROOT / "scripts" / "aggregate_arms.py")
agg = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_arms"] = agg
_spec.loader.exec_module(agg)

STAMP = "fact_20260923_v5fp8"
WANTED = ("invention_itt", "invention_both_spoke", "retrieved", "keywords_emitted",
          "NOT SHOWN EQUIVALENT", "EQUIVALENT")


def _passes(cell: str) -> list[Path]:
    out = []
    for rep in ("r1", "r2", "r3"):
        for suffix in ("", "_resumed"):
            p = _ROOT / "outputs" / f"{STAMP}_{cell}_{rep}{suffix}"
            if (p / "benchmark_eval.json").exists():
                out.append(p)
                break
    return out


def _merge(cell: str, workdir: Path) -> Path | None:
    lanes = _passes(cell)
    if not lanes:
        return None
    out = workdir / f"m_{cell}"
    if len(lanes) == 1:
        shutil.copytree(lanes[0], out)
        return out
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "merge_lane_repeats.py"),
         *[str(p) for p in lanes], "--output", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  !! merge refused for {cell}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[:160]}")
        return None
    return out


ARM1 = _ROOT / "outputs" / "arm1_merged_70b_v5fp8_4pass_v2"
QWEN = _ROOT / "outputs" / "qwen_20260929_arm2-3_v5_static_50q_phala_r3"
QWEN_INJ = _ROOT / "outputs" / "qwen_20260930_arm2-3_v5_inject_50q_phala_r3"


def _contrast(a_dir: Path, b_dir: Path, label: str, a_arm: str = "", b_arm: str = "") -> None:
    a_spec = f":{a_arm}" if a_arm else ""
    b_spec = f":{b_arm}" if b_arm else ""
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "aggregate_arms.py"),
         "--compare", f"A={a_dir.relative_to(_ROOT).as_posix()}{a_spec}",
         "--compare", f"B={b_dir.relative_to(_ROOT).as_posix()}{b_spec}"],
        capture_output=True, text=True, cwd=str(_ROOT),
    )
    print(f"\n### {label}")
    if r.returncode != 0:
        print("  FAILED:", r.stderr.strip()[:300])
        return
    for line in r.stdout.splitlines():
        s = line.strip()
        if s.startswith("matched on query_id") or any(s.startswith(w) for w in WANTED):
            print("   ", s)


def main() -> int:
    # `aggregate_arms --compare LABEL=path` splits `path` on ':' to parse an
    # optional arm spec, so an absolute Windows path is torn apart at the drive
    # letter. Stage the merges inside the repo and pass repo-relative paths.
    work = _ROOT / "outputs" / "scratch_p1figs"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        print("# Key figures, regenerated\n")
        for cell in ("s_l0", "s_l1", "s_l2", "i_l0", "i_l1", "i_l2"):
            n = len(_passes(cell))
            print(f"  {cell}: {n} pass(es)")

        merged = {c: _merge(c, work) for c in ("s_l0", "s_l1", "i_l0", "i_l1")}
        if merged["s_l0"] and merged["s_l1"]:
            _contrast(merged["s_l0"], merged["s_l1"],
                      "STATIC L0 -> L1  (quoted in 4.7.1 item 1&2, 5.1, 6.4, 7, abstract)")
        if merged["i_l0"] and merged["i_l1"]:
            _contrast(merged["i_l0"], merged["i_l1"],
                      "INJECT L0 -> L1  (quoted in 4.7.1 item 1&2, 6.4, 7, abstract)")

        # Table 4 / Sections 4.1-4.3, rebased 2026-09-29 onto matched flat-payload
        # cells. Arm 1 is the stamped 4-pass 70B, never the August 0.2851.
        if merged["s_l0"]:
            _contrast(merged["s_l0"], ARM1, "8B A2 -> 70B A1  (quoted in 4.1, abstract, 7)")
            _contrast(merged["s_l0"], QWEN, "8B A2 -> Qwen A2  (quoted in 4.2)", b_arm="2")
        _contrast(QWEN, ARM1, "Qwen A2 -> 70B A1  (quoted in 4.2)", a_arm="2")
        _contrast(QWEN, QWEN, "Qwen A2 -> Qwen A3  (quoted in 4.3, Table 5, 6.4, 7, abstract)", a_arm="2", b_arm="3")
        _contrast(ARM1, QWEN, "70B A1 -> Qwen A3  (quoted in 4.3, 7)", b_arm="3")
        # Section 4.7.2's selection-bias test on the stamped injection cells.
        lanes = {c: _passes(c) for c in ("i_l0", "i_l1")}
        if all(lanes.values()):
            r = subprocess.run(
                [sys.executable, str(_ROOT / "scripts" / "selection_bias_test.py"),
                 "--arm2", *map(str, lanes["i_l0"]), "--arm3", *map(str, lanes["i_l1"])],
                capture_output=True, text=True)
            print("\n### Selection-bias test, stamped injection L0 vs L1  (quoted in 4.7.2)")
            print("\n".join("    " + line for line in r.stdout.strip().splitlines()))
        # Qwen under candidate injection (2026-09-30), Section 4.7.5.
        _contrast(QWEN_INJ, QWEN_INJ, "Qwen inject A2 -> A3  (quoted in 4.7.5, 7, abstract)", a_arm="2", b_arm="3")
        _contrast(QWEN, QWEN_INJ, "Qwen A2 static -> inject  (quoted in 4.7.5)", a_arm="2", b_arm="2")
        _contrast(QWEN, QWEN_INJ, "Qwen A3 static -> inject  (quoted in 4.7.5)", a_arm="3", b_arm="3")
        _contrast(ARM1, QWEN_INJ, "70B A1 -> Qwen inject A3  (quoted in 4.7.5, 7)", b_arm="3")

        print("\n### Table 6 (regenerate and paste over the table in the manuscript)")
        r = subprocess.run([sys.executable, str(_ROOT / "scripts" / "build_stamped_factorial_table.py")],
                           capture_output=True, text=True)
        print(r.stdout.rstrip() if r.returncode == 0 else r.stderr[:300])

        print("\n### Interaction equivalence (quoted in 7)")
        print("  run: scripts/analyze_factorial.py --repeats-per-cell <n> --cell ...")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
