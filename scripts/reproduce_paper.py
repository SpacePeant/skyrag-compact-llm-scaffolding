#!/usr/bin/env python
"""Check every number in the manuscript against the run exports.

    python scripts/reproduce_paper.py [--manuscript PATH] [--only PREFIX] [--verbose]

Rebuilt 2026-11-02 for the JIOS revision (task 20). The earlier version checked
twelve claims from a September draft against runs the paper no longer reports.

Every claim is one of three kinds, and the report says which:

* **RECOMPUTE.** Recalculated here from `benchmark_eval.json` through
  `aggregate_arms` (passes averaged per query, then over queries). Covers every
  cell of Tables 4 to 8 and the paired contrasts the text reports: average gap,
  bootstrap interval, d_z and Wilcoxon p.
* **TRACE.** The printed value must appear in a named analysis output that an
  earlier script wrote (Holm-adjusted families, local RTX 5090 runs, the
  diagnostic pass, the floor ranker, the constrained-decoding pilot). The
  check is that the paper quotes the artifact, not a recomputation.
* **SOURCE.** A design constant (cap, window, prices, resamples) that must
  appear in the code or config that sets it.

A value that differs is DRIFT, never silently replaced. Finally every number in
the manuscript body is matched against the claims; one that no claim covers is
listed as UNCOVERED, so a clean report cannot hide an unchecked figure.

The claim values are written into this script, so the claim check needs only
the shipped `outputs/`. The coverage step needs the manuscript itself, which is
not in this repository while it is under review: pass it with `--manuscript`;
without it the step is skipped and the report says so.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("aggregate_arms", ROOT / "scripts" / "aggregate_arms.py")
agg = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_arms"] = agg
_spec.loader.exec_module(agg)

from scipy.stats import ttest_ind, wilcoxon  # noqa: E402
from skyrag.stats import bootstrap_ci, cohens_dz, tost_paired  # noqa: E402

O = ROOT / "outputs"
DEFAULT_MS = ROOT / "manuscript.md"  # not shipped; pass --manuscript

# --------------------------------------------------------------------------- cells
_cache: dict[str, list] = {}


def _fact(cell: str) -> list:
    out = []
    for rep in ("r1", "r2", "r3"):
        for suf in ("", "_resumed"):
            p = O / f"fact_20260923_v5fp8_{cell}_{rep}{suf}"
            if (p / "benchmark_eval.json").exists():
                out.append(agg.load(p))
                break
    return out


def _one(path: str, **kw) -> Callable[[], list]:
    return lambda: [agg.load(O / path, **kw)]


DEC = "merged_s1dec_20261001/"
QS, QI = "qwen_20260929_arm2-3_v5_static_50q_phala_r3", "qwen_20260930_arm2-3_v5_inject_50q_phala_r3"
CELLS: dict[str, Callable[[], list]] = {
    "70b": _one("arm1_merged_70b_v5fp8_4pass_v2", arm="1"),
    "70b_i_long": _one("c70h_v5fp8_i_a2_m3"),
    "70b_i_short": _one("c70h_v5fp8_i_a2_s1_m3"),
    **{f"8b_{c}": _one(DEC + c) for c in ("s_a2", "s_a2_sw", "s_a2_s1", "s_a3_p1", "s_a3",
                                          "i_a2", "i_a2_sw", "i_a2_s1", "i_a3_p1", "i_a3")},
    **{f"8b_f_{c}": (lambda c=c: _fact(c)) for c in ("s_l0", "s_l1", "s_l2", "i_l0", "i_l1", "i_l2")},
    "q_s_a2": _one(QS, arm="2"), "q_s_a3": _one(QS, arm="3"),
    "q_i_a2": _one(QI, arm="2"), "q_i_a3": _one(QI, arm="3"),
    "8b_cq_long": _one("a7fix_v5fp8_i_a2_cq_m3"), "8b_cq_short": _one("a7fix_v5fp8_i_a2_s1_cq_m3"),
    # Array-of-objects payload, one pass (Table 5 lower half).
    "8b_arr_a2": _one("eval_20260827_arm1-2_llama70b-8b_100q_topup", model="8b"),
    "8b_arr_a3": _one("arm3_A_spec_only_20260831"),
    "q_arr_a2": _one("sweep_v2_20260902_arm2-3_qwen7b_50q", arm="2"),
    "q_arr_a3": _one("sweep_v2_20260902_arm2-3_qwen7b_50q", arm="3"),
    "70b_old": _one("eval_20260827_arm1-2_llama70b-8b_100q_topup", model="70b"),
    # Array payload with the long rewrite (selection schema only), three passes.
    "8b_arr_a3p1": _one("merged_s1decA_20261002/hosted_a3v2p1"),
    "8b_arr_a3_r3": _one("merged_s1decA_20261002/hosted_a3v2"),
    # Floor ranker, later session, short rewrite (JSON).
    "8b_floor_lex": _one("merged_floor_20261007/hosted_llama_a2_s1_lexical"),
    "8b_floor": _one("merged_floor_20261007/hosted_llama_a2_s1_floor"),
    # Local RTX 5090 decomposition cells (never pooled with hosted ones).
    **{f"loc_{m}_{c}": _one(f"s1dec5090_{m}_{c}") for m in ("llama", "qwen")
       for c in ("s_a2", "s_a2_sw", "s_a2_s1", "s_a3_p1", "s_a3", "i_a2", "i_a2_sw", "i_a2_s1", "i_a3_p1", "i_a3")},    # Qwen at fp8, the same precision as local LLaMA (precision session, 2026-10-06).
    "loc_qwen_fp8_s_a2": _one("merged_qprec_20261006/fp8_s_a2"),
    "loc_qwen_fp8_i_a2": _one("merged_qprec_20261006/fp8_i_a2"),
}


def cells(name: str) -> list:
    if name not in _cache:
        _cache[name] = CELLS[name]()
    return _cache[name]


FN = {
    "itt": agg.itt, "as": agg.both_spoke, "ret": agg.METRICS["retrieved"],
    "kw": agg.METRICS["keywords_emitted"], "nm": agg.near_miss_rate,
}


def per_query(name: str, metric: str) -> dict[str, float]:
    """Each query's mean over passes; a pass that leaves the metric undefined is skipped."""
    d: dict[str, list[float]] = {}
    for c in cells(name):
        for q, v in agg.by_query(c, FN[metric]).items():
            d.setdefault(q, []).append(v)
    return {q: st.fmean(v) for q, v in d.items()}


def mean(name: str, metric: str) -> float:
    return st.fmean(per_query(name, metric).values())


def runs(name: str) -> int:
    return sum(c.n for c in cells(name))


def unanswered(name: str) -> int:
    return sum(c.mutes() for c in cells(name))


def never_answered(name: str) -> int:
    pq = per_query(name, "as")
    allq = per_query(name, "itt")
    return sum(1 for q in allq if q not in pq)


def paired(a: str, b: str, metric: str) -> dict[str, float]:
    """b minus a over the queries both define; the paper's 'average gap'."""
    pa, pb = per_query(a, metric), per_query(b, metric)
    ids = sorted(set(pa) & set(pb))
    xs, ys = [pa[q] for q in ids], [pb[q] for q in ids]
    diffs = [y - x for x, y in zip(xs, ys)]
    lo, hi = bootstrap_ci(diffs)
    p = wilcoxon(xs, ys).pvalue if any(diffs) else 1.0
    dz = cohens_dz(xs, ys)
    # The paper prints an interval as magnitudes ("an average drop of 0.2057 (95% CI 0.1527 to 0.2571)").
    return {"n": len(ids), "delta": st.fmean(diffs), "lo": min(abs(lo), abs(hi)), "hi": max(abs(lo), abs(hi)), "p": p,
            "dz": abs(dz) if dz is not None else float("nan"), "a": st.fmean(xs), "b": st.fmean(ys)}


def tost_p(a: str, b: str) -> float:
    pa, pb = per_query(a, "itt"), per_query(b, "itt")
    ids = sorted(set(pa) & set(pb))
    return tost_paired([pa[q] for q in ids], [pb[q] for q in ids], margin=0.05, repeats_per_cell=3,
                       margin_provenance="+/-0.05 fixed before any repeat run (Measures and Analysis)").p_value


_pc: dict[tuple, dict] = {}


def P(a: str, b: str, metric: str = "itt") -> dict[str, float]:
    k = (a, b, metric)
    if k not in _pc:
        _pc[k] = paired(a, b, metric)
    return _pc[k]


# --------------------------------------------------------------------------- claims
@dataclass
class Claim:
    id: str
    where: str
    printed: str            # exactly as printed; several values separated by " | "
    value: Callable[[], float] | None = None   # RECOMPUTE
    trace: str | None = None                    # TRACE / SOURCE: file relative to ROOT
    kind: str = "RECOMPUTE"
    note: str = ""


def _num(s: str) -> float:
    s = s.replace("−", "-").replace(",", "").strip()
    pct = s.endswith("%")
    v = float(s.rstrip("%"))
    return v / 100 if pct else v


def _tol(s: str) -> float:
    s = s.replace(",", "").rstrip("%")
    dec = len(s.split(".")[1]) if "." in s else 0
    t = 0.5 * 10 ** -dec + 1e-9
    return t / 100 if s != s and False else t


def check_value(printed: str, actual: float) -> bool:
    printed = printed.strip()
    if printed.startswith("<"):
        return actual < _num(printed[1:])
    if printed.startswith("≥") or printed.startswith(">="):
        return actual >= _num(printed.lstrip("≥>="))
    pct = printed.endswith("%")
    target = _num(printed)
    tol = _tol(printed) / (100 if pct else 1)
    return abs(abs(actual) - abs(target)) <= tol


R = lambda f: f  # noqa: E731  readability marker for lambdas below


def table4() -> list[Claim]:
    rows = [("70b", "0.3857", "96.6", "0/200"), ("70b_i_long", "0.0457", "149.3", "0/150"),
            ("70b_i_short", "0.0373", "99.8", "0/150"),
            ("8b_s_a2", "0.6666", "78.9", "7/150"), ("8b_s_a2_sw", "0.5772", "99.6", "7/150"),
            ("8b_s_a2_s1", "0.5252", "125.0", "3/150"), ("8b_s_a3_p1", "0.7087", "36.6", "42/150"),
            ("8b_s_a3", "0.5552", "69.6", "26/150"), ("8b_f_s_l2", "0.6763", "48.3", "32/150"),
            ("8b_i_a2", "0.3860", "149.9", "4/150"), ("8b_i_a2_sw", "0.2586", "156.9", "2/150"),
            ("8b_i_a2_s1", "0.1803", "150.4", "3/150"), ("8b_i_a3_p1", "0.5350", "61.6", "35/150"),
            ("8b_i_a3", "0.3812", "59.3", "32/150"), ("8b_f_i_l2", "0.3111", "89.8", "15/149"),
            ("q_s_a2", "0.4629", "60.0", "0/150"), ("q_s_a3", "0.2805", "106.0", "2/148"),
            ("q_i_a2", "0.2439", "81.5", "0/145"), ("q_i_a3", "0.0821", "77.6", "0/150")]
    out = []
    for c, itt, ret, un in rows:
        out.append(Claim(f"T4-{c}-itt", "Table 4", itt, lambda c=c: mean(c, "itt")))
        out.append(Claim(f"T4-{c}-ret", "Table 4", ret, lambda c=c: mean(c, "ret")))
        u, n = un.split("/")
        out.append(Claim(f"T4-{c}-un", "Table 4", u, lambda c=c: unanswered(c)))
        out.append(Claim(f"T4-{c}-n", "Table 4", n, lambda c=c: runs(c)))
    return out


def table5() -> list[Claim]:
    out = []
    spec = [("flat-8b", "8b_s_a2", "8b_s_a3", [("itt", "0.6666", "0.5552", "50", "0.0035", "0.44"),
                                                ("as", "0.6477", "0.4691", "50", "<0.0001", "0.68")]),
            ("flat-q", "q_s_a2", "q_s_a3", [("itt", "0.4629", "0.2805", "50", "<0.0001", "0.84"),
                                            ("as", "0.4629", "0.2768", "50", "<0.0001", "0.88")]),
            ("arr-8b", "8b_arr_a2", "8b_arr_a3", [("itt", "0.6902", "0.8305", "50", "0.0058", "0.44"),
                                                  ("as", "0.6764", "0.6587", "22", "0.80", "0.07")]),
            ("arr-q", "q_arr_a2", "q_arr_a3", [("itt", "0.4734", "0.5164", "50", "0.23", "0.17"),
                                               ("as", "0.4797", "0.4856", "47", "0.59", "0.03")])]
    for tag, a, b, rows in spec:
        for m, va, vb, n, p, dz in rows:
            k = f"T5-{tag}-{m}"
            out += [Claim(k + "-a", "Table 5", va, lambda a=a, b=b, m=m: P(a, b, m)["a"]),
                    Claim(k + "-b", "Table 5", vb, lambda a=a, b=b, m=m: P(a, b, m)["b"]),
                    Claim(k + "-n", "Table 5", n, lambda a=a, b=b, m=m: P(a, b, m)["n"]),
                    Claim(k + "-p", "Table 5", p, lambda a=a, b=b, m=m: P(a, b, m)["p"]),
                    Claim(k + "-dz", "Table 5", dz, lambda a=a, b=b, m=m: P(a, b, m)["dz"])]
    return out


def table6() -> list[Claim]:
    rows = [("8b_f_s_l0", "0.6666", "0.6477", "78.9"), ("8b_f_s_l1", "0.5552", "0.4691", "69.6"),
            ("8b_f_s_l2", "0.6763", "0.5871", "48.3"), ("8b_f_i_l0", "0.3860", "0.3701", "149.9"),
            ("8b_f_i_l1", "0.3812", "0.2022", "59.3"), ("8b_f_i_l2", "0.3111", "0.2312", "89.8")]
    out = []
    for c, itt, a, ret in rows:
        out += [Claim(f"T6-{c}-itt", "Table 6", itt, lambda c=c: mean(c, "itt")),
                Claim(f"T6-{c}-as", "Table 6", a, lambda c=c: mean(c, "as")),
                Claim(f"T6-{c}-ret", "Table 6", ret, lambda c=c: mean(c, "ret"))]
    return out


def table7() -> list[Claim]:
    a, b = "8b_f_i_l0", "8b_f_i_l1"
    return [
        Claim("T7-as-a", "Table 7", "0.3750", lambda: P(a, b, "as")["a"]),
        Claim("T7-as-b", "Table 7", "0.2022", lambda: P(a, b, "as")["b"]),
        Claim("T7-as-n", "Table 7", "49", lambda: P(a, b, "as")["n"]),
        Claim("T7-as-d", "Table 7", "−0.1728", lambda: P(a, b, "as")["delta"]),
        Claim("T7-as-p", "Table 7", "<0.0001", lambda: P(a, b, "as")["p"]),
        Claim("T7-as-dz", "Table 7", "0.70", lambda: P(a, b, "as")["dz"]),
        Claim("T7-un-a", "Table 7", "2.7%", lambda: unanswered(a) / runs(a)),
        Claim("T7-un-b", "Table 7", "21.3%", lambda: unanswered(b) / runs(b)),
        Claim("T7-un-d", "Table 7", "18.7", lambda: 100 * (unanswered(b) / runs(b) - unanswered(a) / runs(a))),
        Claim("T7-itt-d", "Table 7", "−0.0048", lambda: P(a, b)["delta"]),
        Claim("T7-itt-p", "Table 7", "0.80", lambda: P(a, b)["p"]),
        Claim("T7-itt-dz", "Table 7", "0.02", lambda: P(a, b)["dz"]),
        Claim("T7-ret-d", "Table 7", "−90.6", lambda: P(a, b, "ret")["delta"]),
        Claim("T7-ret-p", "Table 7", "<0.0001", lambda: P(a, b, "ret")["p"]),
        Claim("T7-ret-dz", "Table 7", "1.01", lambda: P(a, b, "ret")["dz"]),
        Claim("T7-kw-a", "Table 7", "67.9", lambda: P(a, b, "kw")["a"]),
        Claim("T7-kw-b", "Table 7", "22.2", lambda: P(a, b, "kw")["b"]),
        Claim("T7-kw-d", "Table 7", "−45.7", lambda: P(a, b, "kw")["delta"]),
        Claim("T7-kw-p", "Table 7", "<0.0001", lambda: P(a, b, "kw")["p"]),
        Claim("T7-kw-dz", "Table 7", "1.82", lambda: P(a, b, "kw")["dz"]),
        Claim("T7-tost", "Schema's Gain, para 2", "0.15", lambda: tost_p(a, b)),
    ]


def table8() -> list[Claim]:
    import cost_table as ct
    printed = {
        ("LLaMA 3.1-70B", "Static", "Unassisted"): ("591", "8,214", "7,864", "6.668", "6.668"),
        ("LLaMA 3.1-70B", "Injection", "Baseline"): ("541", "2,336", "12,871", "6.300", "6.300"),
        ("LLaMA 3.1-70B", "Injection", "Short rewrite (JSON)"): ("600", "2,314", "9,270", "4.873", "4.873"),
        ("LLaMA 3.1-8B", "Static", "Baseline"): ("590", "9,240", "6,833", "0.404", "0.423"),
        ("LLaMA 3.1-8B", "Injection", "Baseline"): ("590", "3,254", "14,168", "0.416", "0.428"),
        ("LLaMA 3.1-8B", "Injection", "Short rewrite (JSON)"): ("637", "3,815", "14,666", "0.446", "0.455"),
        ("LLaMA 3.1-8B", "Injection", "Level 1 schema"): ("481", "4,290", "6,349", "0.283", "0.360"),
        ("Qwen 2.5-7B", "Injection", "Level 1 schema"): ("495", "3,036", "7,298", "1.173", "1.173"),
    }
    out = []
    computed: dict[tuple, tuple] = {}

    def compute():
        if computed:
            return computed
        for model, ctx, cond, key, path, kw in ct.CELLS:
            if (model, ctx, cond) not in printed:
                continue
            cell = agg.load(path, **kw)
            pin, pout = ct.PRICE[key]
            rows = list(cell.rows.values())

            def cost(r):
                return sum((ct.stage(r, n) or (0.0, 0.0))[0] * pin + (ct.stage(r, n) or (0.0, 0.0))[1] * pout
                           for n in ct.STAGES) / 1e6
            toks = [st.fmean(agg.by_query(cell, lambda r, n=n: sum(ct.stage(r, n) or (0.0, 0.0))).values())
                    for n in ("rewrite", "selection", "path")]
            att = st.fmean(agg.by_query(cell, cost).values()) * 1000
            n_ans = sum(1 for r in rows if ct.answered(r))
            computed[(model, ctx, cond)] = (*toks, att, sum(cost(r) for r in rows) / n_ans * 1000)
        return computed
    for k, vals in printed.items():
        for i, v in enumerate(vals):
            out.append(Claim(f"T8-{k[0][-3:]}-{k[1][:3]}-{k[2][:5]}-{i}", "Table 8", v,
                             lambda k=k, i=i: compute()[k][i]))
    return out


def text() -> list[Claim]:
    c = Claim
    T = lambda id, where, printed, f, note="": c(id, where, printed, trace=f, kind="TRACE", note=note)  # noqa: E731
    S = lambda id, where, printed, f, note="": c(id, where, printed, trace=f, kind="SOURCE", note=note)  # noqa: E731
    return [
        # Scale gap
        c("SG-ratio", "Scale Gap; Abstract; Conclusions", "1.73", lambda: mean("8b_s_a2", "itt") / mean("70b", "itt")),
        c("SG-d", "Scale Gap", "0.2809", lambda: P("8b_s_a2", "70b")["delta"]),
        c("SG-lo", "Scale Gap", "0.2245", lambda: P("8b_s_a2", "70b")["lo"]),
        c("SG-hi", "Scale Gap", "0.3365", lambda: P("8b_s_a2", "70b")["hi"]),
        c("SG-dz", "Scale Gap", "1.38", lambda: P("8b_s_a2", "70b")["dz"]),
        c("SG-p", "Scale Gap", "<0.0001", lambda: P("8b_s_a2", "70b")["p"]),
        c("SG-never", "Scale Gap: no query unanswered on every pass", "0",
          lambda: never_answered("70b") + never_answered("8b_s_a2")),
        c("SG-as-ratio", "Scale Gap", "1.68", lambda: mean("8b_s_a2", "as") / mean("70b", "as")),
        c("SG-nm70", "Scale Gap", "2.13%", lambda: P("70b", "8b_s_a2", "nm")["a"]),
        c("SG-nm8", "Scale Gap", "1.72%", lambda: P("70b", "8b_s_a2", "nm")["b"]),
        c("SG-nm-p", "Scale Gap", "0.41", lambda: P("70b", "8b_s_a2", "nm")["p"]),
        c("SG-gap028", "Scale Gap", "0.28", lambda: P("8b_s_a2", "70b")["delta"]),
        c("SG-q-p1", "Scale Gap: Qwen vs LLaMA 8B", "<0.05", lambda: P("8b_s_a2", "q_s_a2")["p"]),
        c("SG-q-p2", "Scale Gap: Qwen vs 70B", "<0.05", lambda: P("70b", "q_s_a2")["p"]),
        c("SG-72.5", "Scale Gap; Implications; Conclusions", "72.5%",
          lambda: (mean("8b_s_a2", "itt") - mean("q_s_a2", "itt")) / (mean("8b_s_a2", "itt") - mean("70b", "itt"))),
        c("SG-local-s", "Scale Gap: local Qwen vs LLaMA, static", "0.0002", lambda: P("loc_llama_s_a2", "loc_qwen_fp8_s_a2")["p"]),
        c("SG-local-i", "Scale Gap: local Qwen vs LLaMA, injection", "0.87", lambda: P("loc_llama_i_a2", "loc_qwen_fp8_i_a2")["p"]),
        # Level 1 helps
        c("L1-018", "Level 1 Helps; Abstract", "0.18", lambda: P("8b_s_a2", "8b_s_a3", "as")["delta"]),
        c("L1-019", "Level 1 Helps", "0.19", lambda: P("q_s_a2", "q_s_a3", "as")["delta"]),
        c("L1-ret12", "Level 1 Helps; Factorial", "12%", lambda: 1 - mean("8b_s_a3", "ret") / mean("8b_s_a2", "ret")),
        c("L1-ret-p", "Level 1 Helps; Factorial", "0.31", lambda: P("8b_s_a2", "8b_s_a3", "ret")["p"]),
        c("L1-q-ret-p", "Level 1 Helps", "<0.0001", lambda: P("q_s_a2", "q_s_a3", "ret")["p"]),
        c("L1-arr-24", "Level 1 Helps: array, unanswered queries", "24", lambda: unanswered("8b_arr_a3")),
        c("L1-arr-63", "Level 1 Helps: array retrieval", "63%", lambda: 1 - mean("8b_arr_a3", "ret") / mean("8b_arr_a2", "ret")),
        c("L1-arr-41", "Level 1 Helps: array retrieval", "41%", lambda: 1 - mean("q_arr_a3", "ret") / mean("q_arr_a2", "ret")),
        c("L1-arr-p8", "Level 1 Helps: array retrieval", "<0.001", lambda: P("8b_arr_a2", "8b_arr_a3", "ret")["p"]),
        c("L1-arr-pq", "Level 1 Helps: array retrieval", "<0.001", lambda: P("q_arr_a2", "q_arr_a3", "ret")["p"]),
        c("L1-prov", "Level 1 Helps: invented keywords flagged as copied (lower bound)", "492 | 694",
          lambda: _provenance()[:2], note="invented minus every keyword flagged as recalled"),
        c("L1-prov-pct", "Level 1 Helps: 70.9% of invented flagged as copied", "70.9%",
          lambda: _provenance()[0] / _provenance()[1]),
        c("L1-two-thirds", "Level 1 Helps: two thirds of emitted keywords invented", "67%",
          lambda: _provenance()[1] / _provenance()[2], note="printed as 'about 67%' (two thirds)"),
        # Factorial
        c("F-s", "Factorial", "0.1785", lambda: P("8b_f_s_l0", "8b_f_s_l1", "as")["delta"]),
        c("F-i", "Factorial", "0.1728", lambda: P("8b_f_i_l0", "8b_f_i_l1", "as")["delta"]),
        c("F-i-p", "Factorial", "<0.0001", lambda: P("8b_f_i_l0", "8b_f_i_l1", "as")["p"]),
        c("F-ret60", "Factorial", "60%", lambda: 1 - mean("8b_f_i_l1", "ret") / mean("8b_f_i_l0", "ret")),
        c("F-un-s0", "Factorial", "4.7%", lambda: unanswered("8b_f_s_l0") / runs("8b_f_s_l0")),
        c("F-un-s1", "Factorial", "17.3%", lambda: unanswered("8b_f_s_l1") / runs("8b_f_s_l1")),
        c("F-itt-p", "Factorial; Table 5", "0.0035", lambda: P("8b_f_s_l0", "8b_f_s_l1")["p"]),
        c("F-l2-s-p", "Factorial: Level 2 vs Level 1, static", "0.0035", lambda: P("8b_f_s_l1", "8b_f_s_l2", "as")["p"]),
        c("F-l2-i-p", "Factorial: Level 2 vs Level 1, injection", "0.3105", lambda: P("8b_f_i_l1", "8b_f_i_l2", "as")["p"]),
        T("F-holm", "Factorial: Holm-adjusted", "0.0059 | 0.026", "outputs/factorial_r3_invention_20260930.txt"),
        # Schema's gain
        c("G-welch", "Schema's Gain: Welch", "0.90", lambda: _welch()[2]),
        c("G-welch-a", "Schema's Gain: Welch", "0.3669", lambda: _welch()[0]),
        c("G-welch-b", "Schema's Gain: Welch", "0.3729", lambda: _welch()[1]),
        c("G-welch-n", "Schema's Gain: Welch", "23 | 27", lambda: _welch()[3]),
        c("G-diag", "Schema's Gain: diagnostic, median keywords in replies over the cap", "96", lambda: _diag()[0]),
        c("G-diag-min", "Schema's Gain: every parsed violating reply over 40", "41", lambda: _diag()[1],
          note="smallest count among parsed violations; 7 replies held no parseable JSON object"),
        c("G-q-p", "Schema's Gain: Qwen", "<0.0001", lambda: P("q_i_a2", "q_i_a3")["p"]),
        c("G-q-ret", "Schema's Gain: Qwen retrieval", "0.78", lambda: P("q_i_a2", "q_i_a3", "ret")["p"]),
        # Rewrite
        T("RW-holm", "Rewrite: Holm-adjusted p", "0.66 | 0.014 | 0.006 | 0.115 | 0.028",
          "outputs/merged_s1dec_20261001/holm_hosted.txt"),
        c("RW-i-d", "Rewrite; Implications", "0.2057", lambda: P("8b_i_a2", "8b_i_a2_s1")["delta"]),
        c("RW-i-lo", "Rewrite", "0.1527", lambda: P("8b_i_a2", "8b_i_a2_s1")["lo"]),
        c("RW-i-hi", "Rewrite", "0.2571", lambda: P("8b_i_a2", "8b_i_a2_s1")["hi"]),
        c("RW-i-dz", "Rewrite", "1.10", lambda: P("8b_i_a2", "8b_i_a2_s1")["dz"]),
        c("RW-s-ret", "Rewrite: static retrieval", "0.0002", lambda: P("8b_s_a2", "8b_s_a2_s1", "ret")["p"]),
        c("RW-i-ret", "Rewrite: injection retrieval", "0.48", lambda: P("8b_i_a2", "8b_i_a2_s1", "ret")["p"]),
        c("RW-cq-d", "Rewrite: fixed candidates", "0.2795", lambda: P("8b_cq_long", "8b_cq_short")["delta"]),
        c("RW-cq-lo", "Rewrite: fixed candidates", "0.2243", lambda: P("8b_cq_long", "8b_cq_short")["lo"]),
        c("RW-cq-hi", "Rewrite: fixed candidates", "0.3347", lambda: P("8b_cq_long", "8b_cq_short")["hi"]),
        c("RW-cq-dz", "Rewrite: fixed candidates", "1.40", lambda: P("8b_cq_long", "8b_cq_short")["dz"]),
        c("RW-cq-p", "Rewrite: fixed candidates", "<0.0001", lambda: P("8b_cq_long", "8b_cq_short")["p"]),
        c("RW-cq-a", "Rewrite: fixed candidates", "0.4419", lambda: mean("8b_cq_long", "itt")),
        c("RW-cq-b", "Rewrite: fixed candidates", "0.1624", lambda: mean("8b_cq_short", "itt")),
        c("RW-70-d", "Rewrite: vs 70B", "0.2054", lambda: P("70b", "8b_i_a2_s1")["delta"]),
        c("RW-70-lo", "Rewrite: vs 70B", "0.1488", lambda: P("70b", "8b_i_a2_s1")["lo"]),
        c("RW-70-hi", "Rewrite: vs 70B", "0.2601", lambda: P("70b", "8b_i_a2_s1")["hi"]),
        c("RW-70-dz", "Rewrite: vs 70B", "1.02", lambda: P("70b", "8b_i_a2_s1")["dz"]),
        c("RW-70-p", "Rewrite: vs 70B", "<0.0001", lambda: P("70b", "8b_i_a2_s1")["p"]),
        c("RW-sel-i", "Rewrite; Implications", "0.1490", lambda: P("8b_i_a2", "8b_i_a3_p1")["delta"]),
        c("RW-sel-as", "Rewrite", "0.0724", lambda: P("8b_s_a2", "8b_s_a3_p1", "as")["delta"]),
        c("RW-sel-as-p", "Rewrite", "0.0029", lambda: P("8b_s_a2", "8b_s_a3_p1", "as")["p"]),
        c("RW-sel-asi-p", "Rewrite", "0.83", lambda: P("8b_i_a2", "8b_i_a3_p1", "as")["p"]),
        c("RW-63s", "Rewrite: plain-text share of the rewrite's effect, static", "63%", lambda: _share("s", "a2_sw")),
        c("RW-62i", "Rewrite: plain-text share of the rewrite's effect, injection", "62%", lambda: _share("i", "a2_sw")),
        T("RW-copy", "Rewrite: copying", "104 | 31 | 67.9% | 18.5%", "outputs/merged_s1dec_20261001/copy_share.txt"),
        T("RW-arr", "Rewrite: array payload", "0.8372 | 0.0001 | 0.58",
          "outputs/merged_s1decA_20261002/compare_hosted.txt"),
        c("RW-arr-47", "Rewrite: array payload", "47%",
          lambda: (mean("8b_arr_a3p1", "itt") - mean("8b_s_a3_p1", "itt")) / (mean("8b_arr_a3", "itt") - mean("8b_s_a3", "itt"))),
        T("RW-overrun", "Rewrite: array failed calls", "94 | 95 | 126 | 128",
          "outputs/merged_s1decA_20261002/overrun_cause.txt", "row 'hosted LLaMA array, own rewrite'"),
        c("RW-overrun-sum", "Rewrite: array failed calls", "220 | 223", lambda: _overrun(),
          note="220 = 94 cap violations + 126 cut-offs over the cap; 223 = 95 + 128 failed calls"),
        T("RW-local", "Rewrite: local replication", "0.0021", "outputs/local_vllm_5090_summary/5090_summary.md"),
        T("RW-local-un", "Rewrite: local unanswered", "15", "outputs/local_vllm_5090_summary/5090_summary.md"),
        # 70B
        c("70-d", "70B", "0.3400", lambda: P("70b", "70b_i_long")["delta"]),
        c("70-lo", "70B", "0.2991", lambda: P("70b", "70b_i_long")["lo"]),
        c("70-hi", "70B", "0.3830", lambda: P("70b", "70b_i_long")["hi"]),
        c("70-dz", "70B", "2.23", lambda: P("70b", "70b_i_long")["dz"]),
        c("70-short-d", "70B; Discussion", "0.0085", lambda: P("70b_i_long", "70b_i_short")["delta"]),
        c("70-short-p", "70B", "0.046", lambda: P("70b_i_long", "70b_i_short")["p"]),
        c("70-vs8-long-p", "70B vs 8B, long rewrite", "<0.0001", lambda: P("8b_i_a2", "70b_i_long")["p"]),
        c("70-vs8-short-p", "70B vs 8B, short rewrite", "<0.0001", lambda: P("8b_i_a2_s1", "70b_i_short")["p"]),
        c("70-ret-p", "70B: short rewrite retrieval", "<0.0001", lambda: P("70b_i_long", "70b_i_short", "ret")["p"]),
        # Limitations
        c("LIM-old70", "Limitations: hosted drift", "0.2851", lambda: mean("70b_old", "itt")),
        T("LIM-prec", "Limitations: precision", "0.0429", "outputs/local_vllm_5090_summary/5090_summary.md"),
        c("LIM-floor-p", "Limitations: second ranker, invalid rate", "0.17", lambda: P("8b_floor_lex", "8b_floor")["p"]),
        c("LIM-floor-ret", "Limitations: second ranker, retrieval", "<0.0001", lambda: P("8b_floor_lex", "8b_floor", "ret")["p"]),
        c("LIM-floor-un", "Limitations: second ranker, unanswered", "7", lambda: unanswered("8b_floor")),
        c("LIM-floor-un0", "Limitations: second ranker, unanswered", "1", lambda: unanswered("8b_floor_lex")),
        T("LIM-pilot", "Limitations: decoder pilot", "0.4895 | 0.4510=0.451",
          "outputs/local_vllm_20261001/4070ti_pilot/stage2_summary.md"),
        T("LIM-cited", "Limitations: cited share", "0.9865", "outputs/cost_and_answer/cost_and_answer.md"),
        c("LIM-tost-n", "Limitations: equivalence sample size", "115 | 205", lambda: _power()),
        c("LIM-edx", "Limitations; Appendix", "26", lambda: len(__import__("json").loads(
            (ROOT / "benchmarks" / "edx_published_omissions.json").read_text(encoding="utf-8")))),
        # Corpus and benchmark
        c("COR-queries", "Corpus: 50 queries", "50", lambda: len(_queries())),
        c("COR-domains", "Corpus: queries per domain (8, 6, 5, 5, 5, 5, 5, 4, 4, 3)", "0",
          lambda: sum(abs(a - b) for a, b in zip(sorted(_domain_counts().values(), reverse=True),
                                                  [8, 6, 5, 5, 5, 5, 5, 4, 4, 3])),
          note="0 means the ten printed domain counts equal the query file's"),
        c("COR-ai", "Corpus: five original, 45 AI-assisted", "45", lambda: len(_queries()) - 5,
          note="the five original queries are q01 to q05"),
        Claim("COR-check", "Corpus: authors' reading of the 50 queries", "49 | 47 | 29", kind="EXEMPT",
              note="the authors' judgement of each query, not a computation"),
        T("COR-25", "Corpus: edX records tagged locally", "25", "CORPUS.md||project_knowledge/research_directions/corpus_provenance_and_licensing.md"),
        S("DES-100", "Experimental Design; Schema's Gain", "100", "src/skyrag/selection.py"),
        c("INT-ratio", "Introduction; Table 3: 10 to 20 times cheaper", "10 | 20", lambda: _price_ratio()),
        c("SRV-q-ratio", "Table 3: Qwen 2 to 4 times cheaper", "0", lambda: abs(_price_ratio("qwen")[0] - 2) + abs(_price_ratio("qwen")[1] - 4),
          note="0 means the printed 2x to 4x equals the list prices"),
        Claim("SRV-date", "Serving: prices retrieved 30 September 2026", "30", trace="scripts/cost_table.py",
              kind="SOURCE", note="date of the list prices, stated in the script's PRICE comment"),
        T("MEA-14", "Measures: 14 paired contrasts in the decomposition family", "14", "outputs/merged_s1dec_20261001/holm_hosted.txt"),
        S("MEA-itt1", "Measures: an unanswered run scores 1.0", "1.0", "scripts/aggregate_arms.py"),
        Claim("MEA-conv", "Measures: 95% level, d_z 0.8 'large', the 1.4 illustration; '70-billion-parameter'", "95% | 0.8 | 1.4 | 70", kind="EXEMPT",
              note="conventions and a worked illustration, not results"),
        c("SG-2pct", "Scale Gap; Conclusions: near-misses about 2%", "2%",
          lambda: st.fmean([P("70b", "8b_s_a2", "nm")["a"], P("70b", "8b_s_a2", "nm")["b"]]),
          note="printed as 'about 2%'"),
        T("RW-local-01", "Rewrite: local short rewrite lowers the rate in all four cells, all p < 0.01", "0.01",
          "outputs/local_vllm_5090_summary/5090_summary.md"),
        c("DIS-37s", "Discussion: the rewrite schema's share of the rewrite's gain, static", "37%", lambda: 1 - _share("s", "a2_sw")),
        c("DIS-38i", "Discussion: the rewrite schema's share of the rewrite's gain, injection", "38%", lambda: 1 - _share("i", "a2_sw")),
        Claim("IMP-70", "Implications: at least 70% flagged as copied", "70%", kind="EXEMPT",
              note="floor of the 70.9% checked in L1-prov"),
        T("LIM-80", "Limitations: 80% power", "80%=0.80", "scripts/power_tost.py"),
        T("LIM-25", "Limitations: pilot of 25 queries", "25", "outputs/local_vllm_20261001/4070ti_pilot/stage2_summary.md"),
        # Method constants and corpus
        S("SYS-cap", "System and Settings", "40", "src/skyrag/config.py"),
        S("SYS-window", "System and Settings", "1,000 | 1000", "src/skyrag/config.py"),
        Claim("SYS-85", "System and Settings", "85", trace="(not in this repository)",
              kind="CARRIED", note="mean rendered course length; measured in the 10-14 audit, no tracked output recomputes it"),
        S("SYS-600", "System and Settings", "600", "src/skyrag/config.py"),
        S("SYS-801", "System and Settings", "801", "src/skyrag/config.py"),
        S("SYS-81", "System and Settings", "81%", "src/skyrag/config.py"),
        S("MEA-boot", "Measures", "10,000=10_000", "src/skyrag/stats.py"),
        S("SRV-price", "Table 3; Table 8", "0.40 | 0.02 | 0.04 | 0.10 | 0.20", "scripts/cost_table.py"),
        S("SRV-ctx70", "Table 3; Serving", "131,072", "src/skyrag/config.py"),
        Claim("SRV-ctxq", "Table 3; Serving", "32,768", trace="(not in this repository)",
              kind="CARRIED", note="Qwen 2.5-7B context window as the provider lists it; external"),
        T("COR-counts", "Corpus", "10,092 | 6,641 | 3,451 | 1,952 | 29.4% | 0.72%",
          "CORPUS.md||project_knowledge/research_directions/corpus_provenance_and_licensing.md"),
        Claim("COR-orig", "Corpus: the original study's 13,373 courses", "13,373",
              trace="docs/papers/From_Queries_to_Courses_extracted.txt", kind="CARRIED",
              note="from the original SKYRAG paper (Soekamto et al., 2025)"),
        c("SRV-local-23", "Serving: local Qwen over context", "23 | 1,500", lambda: _local_excluded()),
    ]


def _share(ctx: str, step: str) -> float:
    """Share of the short rewrite (JSON)'s drop from the baseline already reached by `step`."""
    base, full = mean(f"8b_{ctx}_a2", "itt"), mean(f"8b_{ctx}_a2_s1", "itt")
    return (base - mean(f"8b_{ctx}_{step}", "itt")) / (base - full)


def _queries() -> list[dict]:
    import json
    path = ROOT / "benchmarks" / "queries_v1.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _domain_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for q in _queries():
        out[q["domain"]] = out.get(q["domain"], 0) + 1
    return out


def _price_ratio(model: str = "8b") -> tuple[float, float]:
    import cost_table as ct
    big, small = ct.PRICE["70b"], ct.PRICE[model]
    lo, hi = sorted((big[0] / small[0], big[1] / small[1]))
    return lo, hi


def _provenance() -> tuple[int, int, int]:
    """(lower bound flagged as copied, invented, emitted) for LLaMA 3.1-8B under the array schema."""
    rows = [r for c in cells("8b_arr_a3") for r in c.rows.values()]
    emitted = sum(r.get("keywords_emitted") or 0 for r in rows)
    invented = sum(r.get("hallucinated_keywords") or 0 for r in rows)
    recalled = sum(r.get("keywords_claimed_recalled") or 0 for r in rows)
    return invented - recalled, invented, emitted


def _diag() -> tuple[float, int]:
    """Median and minimum keyword count among diagnostic replies rejected for the cap."""
    import glob
    import json
    counts = []
    for f in glob.glob(str(O / "diag_20260930_violog" / "i_l1_shard*" / "benchmark_eval.json")):
        for r in json.loads(Path(f).read_text(encoding="utf-8")):
            d = r.get("selection_violation_detail")
            for msgs in (json.loads(d).values() if d else []):
                for m in msgs:
                    k = re.search(r"has (\d+) properties, maximum is 40", m)
                    if k:
                        counts.append(int(k.group(1)))
    return st.median(counts), min(counts)


def _overrun():
    body = (O / "merged_s1decA_20261002" / "overrun_cause.txt").read_text(encoding="utf-8")
    row = next(ln for ln in body.splitlines() if ln.startswith("| hosted LLaMA array, own rewrite"))
    cols = [c.strip() for c in row.strip().strip("|").split("|")]
    over_cut, cut = (int(x) for x in cols[3].split("/"))
    viol, viol_cap = int(cols[5]), int(cols[6])
    return over_cut + viol_cap, cut + viol


def _power():
    import power_tost as pw
    ns = []
    for _label, published, replication, model in pw.PAIRS:
        sd, _m, _pairs = pw.paired_sd(published, replication, model)
        ns.append(math.ceil(pw.required_n(sd, 0.05, 0.05, 0.80)))
    return tuple(sorted(ns))


def _local_excluded():
    names = [f"loc_qwen_{c}" for c in ("s_a2", "s_a2_sw", "s_a2_s1", "s_a3_p1", "s_a3",
                                       "i_a2", "i_a2_sw", "i_a2_s1", "i_a3_p1", "i_a3")]
    planned = 150 * len(names)
    return planned - sum(runs(n) for n in names), planned


def _welch():
    """Baseline invalid rate on the queries the schema left unanswered at least once, against the rest."""
    silenced = set()
    for cell in cells("8b_f_i_l1"):
        for (q, _), r in cell.rows.items():
            if not (r.get("keywords_emitted") or 0):
                silenced.add(q)
    base = per_query("8b_f_i_l0", "as")  # As-Treated, as selection_bias_test.py
    a = [v for q, v in base.items() if q in silenced]
    b = [v for q, v in base.items() if q not in silenced]
    return st.fmean(a), st.fmean(b), ttest_ind(a, b, equal_var=False).pvalue, (len(a), len(b))


def _rounds_to(found: str, printed: str) -> bool:
    try:
        x = float(found.replace(",", "").replace("_", ""))
    except ValueError:
        return False
    dec = len(printed.split(".")[1])
    return round(abs(x), dec) == round(abs(_num(printed)), dec)


def _trace_ok(claim: Claim) -> tuple[bool, str]:
    path = next((ROOT / c for c in claim.trace.split("||") if (ROOT / c).exists()), None)
    if path is None:
        return False, f"missing {claim.trace}"
    body = path.read_text(encoding="utf-8", errors="replace")
    nums = re.findall(r"-?\d[\d,_]*\.?\d*", body)
    missing = []
    for v in (v.strip() for v in claim.printed.split("|")):
        needle = v.split("=")[-1]
        if needle in body or needle.replace(",", "") in body:
            continue
        if "." in needle and any(_rounds_to(n, needle) for n in nums):
            continue
        missing.append(v)
    return (not missing), ("not found: " + ", ".join(missing)) if missing else ""


# --------------------------------------------------------------------------- coverage
_IGNORE = re.compile(r"^(3\.1-70B|3\.1-8B|2\.5-7B|3\.1|2\.5|70B|8B|7B|\d{4})$")


def manuscript_numbers(md: str) -> list[tuple[str, str]]:
    md = md.split("# References")[0]
    md = re.sub(r"\[@[^\]]*\]|@[A-Za-z]+\d{4}[A-Za-z]*|\]\([^)]*\)", "", md)
    out, section = [], "front matter"
    for line in md.splitlines():
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        for m in re.finditer(r"(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?%?(?![\dB])", line):
            tok = m.group(0).lstrip("+").rstrip(",")
            if _IGNORE.match(tok) or re.fullmatch(r"\d", tok):
                continue
            out.append((section, tok))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", type=Path, default=DEFAULT_MS)
    ap.add_argument("--only", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    claims = [c for c in table4() + table5() + table6() + table7() + table8() + text() if c.id.startswith(a.only)]
    counts = {"MATCH": 0, "DRIFT": 0, "MISSING": 0}
    covered: set[str] = set()
    for cl in claims:
        for v in cl.printed.split("|"):
            v = v.strip().lstrip("<≥").replace("−", "-")
            covered.add(v.split("=")[0].lstrip("-"))
        if cl.kind == "EXEMPT":
            counts["MATCH"] += 0
            if a.verbose:
                print(f"  EXEMPT   {cl.id:24s} {cl.printed} ({cl.note})")
            continue
        if cl.kind == "RECOMPUTE":
            try:
                actual = cl.value()
            except Exception as exc:  # a missing run is reported, never skipped
                counts["MISSING"] += 1
                print(f"  MISSING  {cl.id:24s} {cl.where}: {exc}")
                continue
            if isinstance(actual, tuple):
                ok = all(check_value(p.strip(), x) for p, x in zip(cl.printed.split("|"), actual))
            else:
                ok = (abs(actual - _num(cl.printed)) <= 0.05 if cl.note.startswith("printed as 'about")
                      else check_value(cl.printed, actual))
            status = "MATCH" if ok else "DRIFT"
            counts[status] += 1
            if status != "MATCH" or a.verbose:
                print(f"  {status:8s} {cl.id:24s} printed {cl.printed:10s} computed {actual}  ({cl.where})")
        else:
            ok, why = _trace_ok(cl)
            if cl.kind == "CARRIED":
                counts["CARRIED"] = counts.get("CARRIED", 0) + 1
                print(f"  CARRIED  {cl.id:24s} {cl.printed}: {cl.note}")
                continue
            status = "MATCH" if ok else "DRIFT"
            counts[status] += 1
            if status != "MATCH" or a.verbose:
                print(f"  {status:8s} {cl.id:24s} [{cl.kind}] {cl.printed} in {cl.trace} {why}")
    by_kind = {}
    for cl in claims:
        by_kind[cl.kind] = by_kind.get(cl.kind, 0) + 1
    exempt = sum(1 for cl in claims if cl.kind == "EXEMPT")
    print(f"\n{len(claims)} claims ({', '.join(f'{k} {v}' for k, v in by_kind.items())}): "
          f"{counts['MATCH']} match, {counts['DRIFT']} drift, {counts['MISSING']} missing, "
          f"{counts.get('CARRIED', 0)} carried, {exempt} exempt")
    if a.only:
        return 0 if not (counts["DRIFT"] or counts["MISSING"]) else 1
    if not a.manuscript.exists():
        print("\nCoverage step skipped: no manuscript (pass --manuscript PATH to list "
              "numbers no claim covers).")
        return 0 if not (counts["DRIFT"] or counts["MISSING"]) else 1
    md = a.manuscript.read_text(encoding="utf-8")
    unc = [(s, t) for s, t in manuscript_numbers(md) if t.replace("−", "-").lstrip("-") not in covered]
    print(f"\nUNCOVERED numbers in the manuscript: {len(unc)}")
    for s, t in unc:
        print(f"  {s[:45]:45s} {t}")
    return 0 if not (counts["DRIFT"] or counts["MISSING"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
