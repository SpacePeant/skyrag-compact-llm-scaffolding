"""Measured cost per query and what reaches the final answer, per hosted cell.

    python scripts/cost_and_answer.py [out_dir]

Two things the manuscript states only as list prices or not at all:

* **Cost per query.** `pipeline_prompt_tokens` and `pipeline_completion_tokens` (all three model
  stages) times the list prices in the manuscript's Table 1. A row whose token total is incomplete
  has no cost and is left out, with the count printed, never treated as zero.
* **The final answer.** `grounding_accuracy` (share of cited course identifiers that exist) and
  `cited_count`. Grounding is undefined on a run that cited nothing; that cohort is printed.

Cells are loaded through `aggregate_arms.load`, passes are averaged per query with
`aggregate_arms.by_query` before any mean or interval, and intervals are `bootstrap_ci` over
queries, as for every other figure in the paper. Prices are list prices, not invoices.
"""
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_arms as agg  # noqa: E402

OUT = ROOT / "outputs"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "cost_and_answer"

# USD per million tokens (input, output). OpenRouter list prices retrieved 2026-09-30, manuscript Table 1.
PRICE = {"70b": (0.40, 0.40), "8b": (0.02, 0.04), "qwen": (0.10, 0.20)}
DEC = OUT / "merged_s1dec_20261001"
CELLS = [
    ("LLaMA 3.1-70B, Arm 1, static window", "70b", OUT / "arm1_merged_70b_v5fp8_4pass_v2", {"arm": "1"}),
    ("LLaMA 3.1-8B, Arm 2, static window", "8b", DEC / "s_a2", {}),
    ("LLaMA 3.1-8B, rewrite only, static window", "8b", DEC / "s_a2_s1", {}),
    ("LLaMA 3.1-8B, Arm 3, static window", "8b", DEC / "s_a3", {}),
    ("LLaMA 3.1-8B, Arm 2, candidate injection", "8b", DEC / "i_a2", {}),
    ("LLaMA 3.1-8B, rewrite only, candidate injection", "8b", DEC / "i_a2_s1", {}),
    ("LLaMA 3.1-8B, Arm 3, candidate injection", "8b", DEC / "i_a3", {}),
    ("Qwen 2.5-7B, Arm 2, static window", "qwen", OUT / "qwen_20260929_arm2-3_v5_static_50q_phala_r3", {"arm": "2"}),
    ("Qwen 2.5-7B, Arm 3, static window", "qwen", OUT / "qwen_20260929_arm2-3_v5_static_50q_phala_r3", {"arm": "3"}),
    ("Qwen 2.5-7B, Arm 2, candidate injection", "qwen", OUT / "qwen_20260930_arm2-3_v5_inject_50q_phala_r3", {"arm": "2"}),
    ("Qwen 2.5-7B, Arm 3, candidate injection", "qwen", OUT / "qwen_20260930_arm2-3_v5_inject_50q_phala_r3", {"arm": "3"}),
]


def cost_fn(price):
    def cost(row):
        p, c = row.get("pipeline_prompt_tokens"), row.get("pipeline_completion_tokens")
        if p is None or c is None or not row.get("pipeline_tokens_complete"):
            return None
        return (float(p) * price[0] + float(c) * price[1]) / 1e6
    return cost


def tokens(row):
    return float(row["pipeline_total_tokens"]) if row.get("pipeline_tokens_complete") and row.get("pipeline_total_tokens") is not None else None


def mean_ci(values):
    lo, hi = agg.bootstrap_ci(values)
    return st.mean(values), lo, hi


def main() -> int:
    lines = ["# Cost per query and the final answer, hosted cells", "",
             "Passes averaged per query, intervals bootstrapped over queries. Prices are the list prices of the manuscript's Table 1.", "",
             "| Cell | Runs | Runs with a token total | Tokens per query | Cost per 1,000 queries, USD [95% CI] | Grounding (queries scored) | Courses cited | Runs citing nothing |",
             "|---|---|---|---|---|---|---|---|"]
    loaded = {}
    for label, key, path, kw in CELLS:
        cell = agg.load(path, **kw)
        cost = agg.by_query(cell, cost_fn(PRICE[key]))
        loaded[label] = cost
        tok = list(agg.by_query(cell, tokens).values())
        ground = list(agg.by_query(cell, agg.grounding).values())
        cited = list(agg.by_query(cell, agg.METRICS["cited"]).values())
        rows = list(cell.rows.values())
        with_tokens = sum(1 for r in rows if tokens(r) is not None)
        nothing = sum(1 for r in rows if not (r.get("cited_count") or 0))
        m, lo, hi = mean_ci([v * 1000 for v in cost.values()])
        lines.append(f"| {label} | {cell.n} | {with_tokens} | {st.mean(tok):,.0f} | {m:.3f} [{lo:.3f}, {hi:.3f}] "
                     f"| {st.mean(ground):.4f} ({len(ground)}) | {st.mean(cited):.1f} | {nothing} |")

    lines += ["", "## Cost of the 70B against each LLaMA 3.1-8B cell, paired on the query", "",
              "| 8B cell | 70B, USD per 1,000 | 8B, USD per 1,000 | Ratio of means | Queries |", "|---|---|---|---|---|"]
    big = loaded[CELLS[0][0]]
    for label, key, _, _ in CELLS[1:]:
        if key != "8b":
            continue
        ids = sorted(set(big) & set(loaded[label]))
        a, b = st.mean(big[q] for q in ids) * 1000, st.mean(loaded[label][q] for q in ids) * 1000
        lines.append(f"| {label} | {a:.3f} | {b:.3f} | {a / b:.1f} | {len(ids)} |")

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "cost_and_answer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
