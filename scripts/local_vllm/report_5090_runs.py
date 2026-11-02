"""One report of every number in the paper measured on the lab RTX 5090 (local vLLM).

    python scripts/local_vllm/report_5090_runs.py [out_dir]

Writes `5090_summary.md` and `5090_cells.csv` (default: `outputs/local_vllm_5090_summary/`).

Every figure is computed from the run folders through `scripts/aggregate_arms.py`
(its loader, per-query averaging, bootstrap interval, Wilcoxon test and effect
size), so the report cannot drift from the numbers the paper quotes. Nothing is
typed in by hand. Local cells are never pooled with hosted ones; the hosted rows
are printed beside them for orientation only.

Runs covered (all full pipeline, flat payload unless stated, 50 queries, three passes):

    2026-10-01  decomposition       s1dec5090_{llama,qwen}_{s,i}_{a2,a2_sw,a2_s1,a3_p1,a3}
    2026-10-02  Run A, array        merged_s1decA_20261002/{llama,qwen}_{a3v2p1,a3v2,a2v2s1}
    2026-10-06  Qwen precision      merged_qprec_20261006/{fp8,bf16}_{s,i}_{...}
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_arms as agg  # noqa: E402

OUT = ROOT / "outputs"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "local_vllm_5090_summary"

COND = {
    "a2": "Arm 2, no scaffolding",
    "a2_sw": "Arm 2 + rewrite wording only",
    "a2_s1": "Arm 2 + rewrite (wording and schema)",
    "a3_p1": "Arm 3 with the baseline rewrite (selection schema only)",
    "a3": "Arm 3, full schema arm",
    "a3v2p1": "Arm 3 array payload, baseline rewrite",
    "a3v2": "Arm 3 array payload, own rewrite",
    "a2v2s1": "Arm 2 + the array arm's rewrite",
}
CTX = {"s": "Static window", "i": "Candidate injection"}
ORDER = ["a2", "a2_sw", "a2_s1", "a3_p1", "a3"]
MODEL = {"llama": "LLaMA 3.1-8B", "qwen": "Qwen 2.5-7B"}

lines: list[str] = []
csv_rows: list[dict] = []


def emit(s: str = "") -> None:
    lines.append(s)


def cell_stats(path: Path, **kw) -> dict:
    cell = agg.load(path, **kw)
    itt = list(agg.by_query(cell, agg.itt).values())
    spoke = list(agg.by_query(cell, agg.both_spoke).values())
    retrieved = list(agg.by_query(cell, agg.METRICS["retrieved"]).values())
    ci = agg.bootstrap_ci(itt)
    return {
        "cell": cell,
        "itt": st.mean(itt), "lo": ci[0], "hi": ci[1],
        "as_treated": st.mean(spoke) if spoke else float("nan"), "n_as_treated": len(spoke),
        "retrieved": st.mean(retrieved),
        "silent": cell.mutes(), "rows": cell.n, "queries": len(itt), "passes": agg._repeats_per_cell(cell),
    }


def paired(a: dict, b: dict, fn=agg.itt) -> dict | None:
    """The same computation as `aggregate_arms.compare`, returned instead of printed."""
    av, bv = agg.by_query(a["cell"], fn), agg.by_query(b["cell"], fn)
    xs, ys = [], []
    for q in sorted(set(av) & set(bv)):
        if av[q] is not None and bv[q] is not None:
            xs.append(av[q]); ys.append(bv[q])
    if len(xs) < 2:
        return None
    diffs = [y - x for x, y in zip(xs, ys)]
    ci = agg.bootstrap_ci(diffs)
    p = agg.wilcoxon(xs, ys).pvalue if agg.wilcoxon and any(diffs) else float("nan")
    return {"from": st.mean(xs), "to": st.mean(ys), "delta": st.mean(diffs), "lo": ci[0], "hi": ci[1], "p": p, "n": len(xs)}


def fmt_p(p: float) -> str:
    return "n/a" if p != p else ("< 0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}")


def cell_table(title: str, rows: list[tuple[str, str, str, str, dict]]) -> None:
    """rows: (run, model and serving, context, condition, stats)."""
    emit(f"### {title}\n")
    emit("| Model and serving | Context | Condition | Invalid rate, all queries [95% CI] | Invalid rate, answered queries | Courses retrieved | Silent runs | Passes |")
    emit("|---|---|---|---|---|---|---|---|")
    for run, model, ctx, cond, s in rows:
        emit(f"| {model} | {ctx} | {cond} | **{s['itt']:.4f}** [{s['lo']:.3f}, {s['hi']:.3f}] | {s['as_treated']:.4f} (n = {s['n_as_treated']}) "
             f"| {s['retrieved']:.1f} | {s['silent']}/{s['rows']} | {s['passes']} |")
        csv_rows.append({"run": run, "model_serving": model, "context": ctx, "condition": cond,
                         "itt_invalid_rate": round(s["itt"], 4), "itt_ci_low": round(s["lo"], 4), "itt_ci_high": round(s["hi"], 4),
                         "as_treated_invalid_rate": round(s["as_treated"], 4), "as_treated_queries": s["n_as_treated"],
                         "courses_retrieved": round(s["retrieved"], 1), "silent_runs": s["silent"], "rows": s["rows"],
                         "queries": s["queries"], "passes": s["passes"]})
    emit()


def contrast_table(title: str, note: str, rows: list[tuple[str, str, dict, dict]]) -> None:
    emit(f"### {title}\n")
    if note:
        emit(note + "\n")
    emit("| Setting | Contrast | From | To | Difference [95% CI] | p | Queries | Courses retrieved |")
    emit("|---|---|---|---|---|---|---|---|")
    for setting, label, a, b in rows:
        r = paired(a, b)
        emit(f"| {setting} | {label} | {r['from']:.4f} | {r['to']:.4f} | **{r['delta']:+.4f}** [{r['lo']:+.4f}, {r['hi']:+.4f}] | {fmt_p(r['p'])} | {r['n']} "
             f"| {a['retrieved']:.1f} to {b['retrieved']:.1f} |")
    emit()


# ------------------------------------------------------------------ load every cell
dec = {(m, c, k): cell_stats(OUT / f"s1dec5090_{m}_{c}_{k}") for m in MODEL for c in CTX for k in ORDER}
runA = {(m, k): cell_stats(OUT / "merged_s1decA_20261002" / f"{m}_{k}") for m in MODEL for k in ("a3v2p1", "a3v2", "a2v2s1")}
prec = {(p, c, k): cell_stats(OUT / "merged_qprec_20261006" / f"{p}_{c}_{k}") for p in ("fp8", "bf16") for c in CTX for k in ORDER}
hosted = {(c, k): cell_stats(OUT / "merged_s1dec_20261001" / f"{c}_{k}") for c in CTX for k in ORDER}
big = cell_stats(OUT / "arm1_merged_70b_v5fp8_4pass_v2", arm="1")
hq = {("s", "2"): cell_stats(OUT / "qwen_20260929_arm2-3_v5_static_50q_phala_r3", arm="2"),
      ("s", "3"): cell_stats(OUT / "qwen_20260929_arm2-3_v5_static_50q_phala_r3", arm="3"),
      ("i", "2"): cell_stats(OUT / "qwen_20260930_arm2-3_v5_inject_50q_phala_r3", arm="2"),
      ("i", "3"): cell_stats(OUT / "qwen_20260930_arm2-3_v5_inject_50q_phala_r3", arm="3")}
SERVE = {"llama": "LLaMA 3.1-8B, local fp8", "qwen": "Qwen 2.5-7B, local bf16"}

# ------------------------------------------------------------------ report
emit("# Every result measured on the lab RTX 5090\n")
emit("Generated by `scripts/local_vllm/report_5090_runs.py` from the run folders. No figure is typed by hand.\n")
emit("## How to read this\n")
emit("* **Invalid rate** is the share of the keywords a model selected that do not exist in the course catalogue. Lower is better.")
emit("* **All queries** counts a query the model failed to answer as a complete failure (rate 1.0). **Answered queries** leaves those out.")
emit("* **Silent runs** are runs where the model produced no usable keyword, so nothing was retrieved.")
emit("* Every cell is the full pipeline on 50 learner queries. Passes are averaged per query before any interval or test, so n is queries, not runs.")
emit("* Local cells are **not pooled** with hosted ones. The hosted rows in Section 5 are for orientation only.")
emit("* The model served at 70B size does not fit this card (32 GB), so there is no local 70B cell.\n")

emit("## 1. Runs\n")
emit("| Date | Run | Models and serving | Cells | Purpose |")
emit("|---|---|---|---|---|")
emit("| 2026-10-01 | Decomposition | LLaMA 3.1-8B fp8, Qwen 2.5-7B bf16 | 5 conditions x 2 contexts x 2 models | Separate the query rewrite from the selection schema |")
emit("| 2026-10-02 | Run A, array payload | LLaMA 3.1-8B fp8, Qwen 2.5-7B bf16 | 3 conditions x 2 models, static window | Test payload shape with the rewrite held fixed |")
emit("| 2026-10-06 | Qwen precision | Qwen 2.5-7B fp8 and bf16, one session | 5 conditions x 2 contexts x 2 precisions | Check whether serving precision moves Qwen |")
emit()
emit("Earlier pilots on the same machine scored keyword selection alone, one pass (`outputs/local_vllm_20261001/`). They are superseded by the full-pipeline runs above and are not repeated here.\n")

emit("## 2. Decomposition, 2026-10-01\n")
cell_table("2.1 Cells", [("decomposition", SERVE[m], CTX[c], COND[k], dec[m, c, k]) for m in MODEL for c in CTX for k in ORDER])
contrast_table("2.2 What each part of the schema arm does, against Arm 2", "Invalid rate over all queries. A negative difference means fewer invented keywords.",
               [(f"{SERVE[m]}, {CTX[c].lower()}", lab, dec[m, c, "a2"], dec[m, c, k])
                for m in MODEL for c in CTX
                for k, lab in (("a2_s1", "Rewrite alone"), ("a2_sw", "Rewrite wording alone"), ("a3_p1", "Selection schema alone"), ("a3", "Full schema arm"))])

emit("## 3. Run A, array payload, 2026-10-02 (static window)\n")
cell_table("3.1 Cells", [("runA", SERVE[m], CTX["s"], COND[k], runA[m, k]) for m in MODEL for k in ("a3v2p1", "a3v2", "a2v2s1")])
contrast_table("3.2 Contrasts", "The flat-payload partners are the decomposition cells of Section 2.",
               [(SERVE[m], lab, a, b) for m in MODEL for lab, a, b in (
                   ("Payload, rewrite held fixed: flat to array", dec[m, "s", "a3_p1"], runA[m, "a3v2p1"]),
                   ("Array arm as specified, against the flat arm", dec[m, "s", "a3"], runA[m, "a3v2"]),
                   ("Arm 2 with the array arm's rewrite, against Arm 2", dec[m, "s", "a2"], runA[m, "a2v2s1"]))])

emit("## 4. Qwen precision, 2026-10-06 (fp8 and bf16 in one session)\n")
cell_table("4.1 Cells", [("qwen_precision", f"Qwen 2.5-7B, local {p}", CTX[c], COND[k], prec[p, c, k]) for p in ("bf16", "fp8") for c in CTX for k in ORDER])
contrast_table("4.2 Precision: bf16 to fp8, same condition", "The first three rows of each context are the pre-registered core cells.",
               [(CTX[c], COND[k], prec["bf16", c, k], prec["fp8", c, k]) for c in CTX for k in ("a2", "a2_s1", "a3", "a2_sw", "a3_p1")])
contrast_table("4.3 Session check: bf16 on 2026-10-01 to bf16 on 2026-10-06", "Same model, same precision, same machine, five days apart. This is the size of ordinary run-to-run variation.",
               [(CTX[c], COND[k], dec["qwen", c, k], prec["bf16", c, k]) for c in CTX for k in ORDER])
contrast_table("4.4 The decomposition at each precision, against Arm 2", "",
               [(f"Qwen {p}, {CTX[c].lower()}", lab, prec[p, c, "a2"], prec[p, c, k])
                for p in ("bf16", "fp8") for c in CTX
                for k, lab in (("a2_s1", "Rewrite alone"), ("a2_sw", "Rewrite wording alone"), ("a3_p1", "Selection schema alone"), ("a3", "Full schema arm"))])

emit("## 5. Hosted cells, for orientation only\n")
emit("These are the cells the paper reports. They ran on hosted providers, not on the 5090, and are never pooled with the local cells above.\n")
cell_table("5.1 Hosted reference cells",
           [("hosted", "LLaMA 3.1-70B, DeepInfra fp8", CTX["s"], "Arm 1, no scaffolding", big)]
           + [("hosted", "LLaMA 3.1-8B, DeepInfra fp8", CTX[c], COND[k], hosted[c, k]) for c in CTX for k in ORDER]
           + [("hosted", "Qwen 2.5-7B, Phala (precision undeclared)", CTX[c], COND["a2" if a == "2" else "a3"], hq[c, a]) for c in CTX for a in ("2", "3")])

emit("## 6. The rewrite alone, everywhere it was measured\n")
emit("The paper's main finding is that changing the query rewrite alone lowers invention. This table collects that one contrast (Arm 2 to Arm 2 with the rewrite) from every setting.\n")
emit("| Setting | Context | Arm 2 | With the rewrite | Difference [95% CI] | p |")
emit("|---|---|---|---|---|---|")
for setting, src in (("LLaMA 3.1-8B, hosted fp8", lambda c, k: hosted[c, k]),
                     ("LLaMA 3.1-8B, local fp8", lambda c, k: dec["llama", c, k]),
                     ("Qwen 2.5-7B, local bf16, 1 Oct", lambda c, k: dec["qwen", c, k]),
                     ("Qwen 2.5-7B, local bf16, 6 Oct", lambda c, k: prec["bf16", c, k]),
                     ("Qwen 2.5-7B, local fp8, 6 Oct", lambda c, k: prec["fp8", c, k])):
    for c in CTX:
        r = paired(src(c, "a2"), src(c, "a2_s1"))
        emit(f"| {setting} | {CTX[c]} | {r['from']:.4f} | {r['to']:.4f} | **{r['delta']:+.4f}** [{r['lo']:+.4f}, {r['hi']:+.4f}] | {fmt_p(r['p'])} |")
emit()

DEST.mkdir(parents=True, exist_ok=True)
(DEST / "5090_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
with open(DEST / "5090_cells.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
    w.writeheader(); w.writerows(csv_rows)
print(DEST / "5090_summary.md")
print(DEST / "5090_cells.csv", f"({len(csv_rows)} cells)")
