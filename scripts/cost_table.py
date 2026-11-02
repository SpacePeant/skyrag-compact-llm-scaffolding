"""Table 8 (A8): token cost per query by stage, per attempted query and per answered run.

    python scripts/cost_table.py [out_dir]

`cost_and_answer.py` prices only the runs whose three-stage token total is complete, so a
run that failed at keyword selection, and never reached the learning-path prompt, is left out
although its rewrite and selection calls were paid for. This table counts what each run spent:

* **Tokens by stage.** Rewrite (`expansion_*`), keyword selection (`selection_*`) and the
  learning-path prompt (`prompt_tokens`, `completion_tokens`). A stage that did not run spent
  nothing and counts as 0; a stage that ran with no usage recorded is counted and reported, never
  silently zeroed.
* **Cost per attempted query.** Every run, unanswered or not, at the list prices of Table 3.
* **Cost per answered run.** The cell's total cost divided by its answered runs (a run that
  emitted at least one keyword), so a condition that leaves runs unanswered pays for them here.

Cells are loaded through `aggregate_arms.load`; per-query values average the passes with
`aggregate_arms.by_query` before the mean over queries, as for every other figure in the paper.
Prices are list prices, not invoices, and token cost is not total cost.
"""
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_arms as agg  # noqa: E402

OUT = ROOT / "outputs"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "cost_table"

# USD per million tokens (input, output). OpenRouter list prices retrieved 2026-09-30, manuscript Table 3.
PRICE = {"70b": (0.40, 0.40), "8b": (0.02, 0.04), "qwen": (0.10, 0.20)}
DEC = OUT / "merged_s1dec_20261001"
QS = OUT / "qwen_20260929_arm2-3_v5_static_50q_phala_r3"
QI = OUT / "qwen_20260930_arm2-3_v5_inject_50q_phala_r3"
CELLS = [
    ("LLaMA 3.1-70B", "Static", "Unassisted", "70b", OUT / "arm1_merged_70b_v5fp8_4pass_v2", {"arm": "1"}),
    ("LLaMA 3.1-70B", "Injection", "Baseline", "70b", OUT / "c70h_v5fp8_i_a2_m3", {}),
    ("LLaMA 3.1-70B", "Injection", "Short rewrite (JSON)", "70b", OUT / "c70h_v5fp8_i_a2_s1_m3", {}),
    ("LLaMA 3.1-8B", "Static", "Baseline", "8b", DEC / "s_a2", {}),
    ("LLaMA 3.1-8B", "Static", "Short rewrite (JSON)", "8b", DEC / "s_a2_s1", {}),
    ("LLaMA 3.1-8B", "Static", "Level 1 schema", "8b", DEC / "s_a3", {}),
    ("LLaMA 3.1-8B", "Injection", "Baseline", "8b", DEC / "i_a2", {}),
    ("LLaMA 3.1-8B", "Injection", "Short rewrite (JSON)", "8b", DEC / "i_a2_s1", {}),
    ("LLaMA 3.1-8B", "Injection", "Level 1 schema", "8b", DEC / "i_a3", {}),
    ("Qwen 2.5-7B", "Static", "Baseline", "qwen", QS, {"arm": "2"}),
    ("Qwen 2.5-7B", "Static", "Level 1 schema", "qwen", QS, {"arm": "3"}),
    ("Qwen 2.5-7B", "Injection", "Baseline", "qwen", QI, {"arm": "2"}),
    ("Qwen 2.5-7B", "Injection", "Level 1 schema", "qwen", QI, {"arm": "3"}),
]
STAGES = {
    "rewrite": ("expansion_prompt_tokens", "expansion_completion_tokens"),
    "selection": ("selection_prompt_tokens", "selection_completion_tokens"),
    "path": ("prompt_tokens", "completion_tokens"),
}


def _num(row, key):
    v = row.get(key)
    if v is None or v == "":
        return None
    return float(v)


def stage(row, name):
    """(input, output) tokens of one stage; (0, 0) if the stage did not run, None if unrecorded."""
    p, c = (_num(row, k) for k in STAGES[name])
    if p is None and c is None:
        return None
    return (p or 0.0, c or 0.0)


def answered(row):
    return bool(row.get("keywords_emitted") or 0)


def main() -> int:
    lines = ["# Table 8: token cost by stage, per attempted query and per answered run", "",
             "Passes averaged per query, then over queries. List prices of Table 3 (OpenRouter, 2026-09-30).", "",
             "| Model | Context | Condition | Runs | Answered | Rewrite tokens | Selection tokens | Path tokens "
             "| USD per 1,000 attempted queries | USD per 1,000 answered runs | Unrecorded stages |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, ctx, cond, key, path, kw in CELLS:
        cell = agg.load(path, **kw)
        pin, pout = PRICE[key]
        rows = list(cell.rows.values())
        unrecorded = {"rewrite": 0, "selection": 0, "path": 0}
        for r in rows:
            for name in STAGES:
                # The path stage legitimately does not run after an empty retrieval.
                if stage(r, name) is None and not (name == "path" and r.get("status") == "empty_retrieval"):
                    unrecorded[name] += 1

        def tok(name):
            return lambda r: sum(stage(r, name) or (0.0, 0.0))

        def cost(r):
            total = 0.0
            for name in STAGES:
                p, c = stage(r, name) or (0.0, 0.0)
                total += p * pin + c * pout
            return total / 1e6

        per_stage = {n: st.mean(agg.by_query(cell, tok(n)).values()) for n in STAGES}
        attempted = st.mean(agg.by_query(cell, cost).values()) * 1000
        n_ans = sum(1 for r in rows if answered(r))
        per_answered = sum(cost(r) for r in rows) / n_ans * 1000 if n_ans else float("nan")
        miss = ", ".join(f"{k} {v}" for k, v in unrecorded.items() if v) or "none"
        lines.append(f"| {model} | {ctx} | {cond} | {len(rows)} | {n_ans} | {per_stage['rewrite']:,.0f} "
                     f"| {per_stage['selection']:,.0f} | {per_stage['path']:,.0f} | {attempted:.3f} "
                     f"| {per_answered:.3f} | {miss} |")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "cost_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
