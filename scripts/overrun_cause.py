#!/usr/bin/env python
"""Why keyword-selection calls fail under the schema: budget overrun or cap violation.

Rebuilds `outputs/merged_s1decA_20261002/overrun_cause.txt` from the exported
rows. A selection call (one platform of one row) fails by one of two routes:

* **Budget overrun**: the reply is cut off after exceeding its completion budget
  (`platforms_lost_detail` reads `<platform>:reply_overran_cap`). The entries the
  model had already written are counted in the retained cut-off text,
  `selection_cutoff_raw`: `"keyword":` items for the array payload, `"term": score`
  pairs for the flat one.
* **Cap violation**: the reply completes but fails validation
  (`selection_violation_detail`, whose message states the item count).

Usage:

    python scripts/overrun_cause.py
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_arms as agg  # noqa: E402

CAP = 40
O = "outputs"
CELLS = [
    ("hosted LLaMA array, baseline rewrite", f"{O}/merged_s1decA_20261002/hosted_a3v2p1"),
    ("hosted LLaMA array, own rewrite", f"{O}/merged_s1decA_20261002/hosted_a3v2"),
    ("hosted LLaMA flat, baseline rewrite", f"{O}/merged_s1dec_20261001/s_a3_p1"),
    ("5090 LLaMA array, baseline rewrite", f"{O}/merged_s1decA_20261002/llama_a3v2p1"),
    ("5090 LLaMA array, own rewrite", f"{O}/merged_s1decA_20261002/llama_a3v2"),
    ("5090 LLaMA flat, baseline rewrite", f"{O}/s1dec5090_llama_s_a3_p1"),
    ("5090 LLaMA flat, own rewrite", f"{O}/s1dec5090_llama_s_a3"),
    ("5090 Qwen array, baseline rewrite", f"{O}/merged_s1decA_20261002/qwen_a3v2p1"),
    ("5090 Qwen array, own rewrite", f"{O}/merged_s1decA_20261002/qwen_a3v2"),
    ("5090 Qwen flat, baseline rewrite", f"{O}/s1dec5090_qwen_s_a3_p1"),
    ("5090 Qwen flat, own rewrite", f"{O}/s1dec5090_qwen_s_a3"),
]
ARRAY_ENTRY = re.compile(r'"keyword"\s*:')
FLAT_ENTRY = re.compile(r'"(?:[^"\\]|\\.)*"\s*:\s*-?[0-9.]')
ITEMS = re.compile(r"has (\d+) (?:items|properties)")


def _median(xs: list[float]) -> str:
    return f"{st.median(xs)}" if xs else "-"


def describe(path: Path) -> list[str] | None:
    file = path / "benchmark_eval.json"
    if not file.exists():
        return None
    rows = json.loads(file.read_text(encoding="utf-8"))
    silent = agg.load(path).mutes()  # the same silent-run count aggregate_arms prints
    over_n, over_entries = 0, []
    viol_n, viol_items = 0, []
    for r in rows:
        cut = json.loads(r["selection_cutoff_raw"]) if r.get("selection_cutoff_raw") else {}
        for part in filter(None, (r.get("platforms_lost_detail") or "").split(";")):
            platform, _, reason = part.strip().partition(":")
            if reason == "reply_overran_cap":
                over_n += 1
                text = cut.get(platform, "")
                pattern = ARRAY_ENTRY if '"keyword"' in text else FLAT_ENTRY
                over_entries.append(len(pattern.findall(text)))
        detail = json.loads(r["selection_violation_detail"]) if r.get("selection_violation_detail") else {}
        for messages in detail.values():
            viol_n += 1
            counts = [int(m) for msg in messages for m in ITEMS.findall(msg)]
            if counts:  # a violation of another kind states no item count
                viol_items.append(max(counts))
    return [
        f"{silent}/{len(rows)}",
        str(over_n),
        f"{sum(e > CAP for e in over_entries)}/{over_n}",
        _median(over_entries),
        str(viol_n),
        str(sum(i > CAP for i in viol_items)),
        _median(viol_items),
    ]


def main() -> int:
    print("| cell | silent runs | overrun calls | of those, >40 entries already written | median entries at cut-off "
          "| violation calls | of those, over the 40 cap | median items |")
    print("|---|---|---|---|---|---|---|---|")
    for label, rel in CELLS:
        cols = describe(ROOT / rel)
        if cols is None:
            sys.exit(f"missing run: {rel}")
        print(f"| {label} | " + " | ".join(cols) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
