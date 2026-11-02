"""Mechanism check for the Stage 1 decomposition: are invented keywords copied from the rewrite?

For every row that exports both `expanded_query` (added 2026-10-01) and
`hallucinated_keyword_terms` (read from the JSON artifact, which holds the raw
rewrite; the merged CSV does not JSON-encode it), counts the invented keywords that appear verbatim
(case-folded, whitespace-normalised) in the rewrite the selector read, and the
rewrite's length in words. Rows from before the column existed carry no rewrite
and are skipped, so the factorial's reference cells cannot be included.

    python scripts/local_vllm/copy_share_decomp.py
"""
import csv
import glob
import io
import json
import os
import re
import statistics
import sys
from pathlib import Path

# Anchored on this file, not the working directory: run from anywhere else, the
# relative paths found no cell and the script printed an empty table with exit 0.
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

csv.field_size_limit(10**8)
CELLS = {
    "hosted LLaMA": "outputs/merged_s1dec_20261001/{ctx}_{cell}",
    "5090 LLaMA": "outputs/s1dec5090_llama_{ctx}_{cell}",
    "5090 Qwen": "outputs/s1dec5090_qwen_{ctx}_{cell}",
}
ORDER = ["a2", "a2_sw", "a2_s1", "a3_p1", "a3"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


print("| setup | context | cell | rewrite words (median) | invented copied from rewrite |")
print("|---|---|---|---|---|")
found = 0
for setup, pattern in CELLS.items():
    for ctx in ("s", "i"):
        for cell in ORDER:
            path = os.path.join(pattern.format(ctx=ctx, cell=cell), "benchmark_eval.json")
            if not os.path.exists(path):
                continue
            rows = [r for r in json.load(io.open(path, encoding="utf-8")) if r.get("query_id")]
            if not rows or "expanded_query" not in rows[0]:
                continue
            words, inv, copied = [], 0, 0
            for r in rows:
                if r.get("status") == "cell_failed":
                    continue
                text = norm(r.get("expanded_query") or "")
                words.append(len(text.split()))
                for k in filter(None, (r.get("hallucinated_keyword_terms") or "").split(";")):
                    inv += 1
                    copied += norm(k) in text
            ctx_name = "static" if ctx == "s" else "injection"
            share = f"{copied}/{inv} = {copied / inv:.1%}" if inv else "-"
            print(f"| {setup} | {ctx_name} | {cell} | {statistics.median(words):.0f} | {share} |")
            found += 1

if not found:
    sys.exit(f"no decomposition cell with an expanded_query column found under {ROOT / 'outputs'}")
