#!/usr/bin/env python
"""CB6. N run directories -> one table, with denominators and intervals attached.

Every Arm 1/2/3 table produced so far was assembled by hand from raw JSON,
including the ones in this session. That method has already published wrong
numbers once (the V2 handover), and it does not survive the scale the leg is
heading for: 4-5 models x 2 arms x 3 repeats is ~30 run directories.

What this fixes is **reproducibility, not correctness**. The hand-computed
figures in the task log are right; they are simply not regenerable by anyone
who does not re-derive the script. Numbers a reviewer cannot reproduce from the
repository are numbers the paper cannot defend.

Design commitments, each inherited from a decision this project already made:

* **ITT is primary** (CB3). Every query is scored; a mute counts as total
  failure at 1.0. The both-spoke figure is reported alongside and never without
  its denominator, because the treatment selects that subset itself.
* **No bare means** (X3). Every mean carries a bootstrap interval and an n.
* **Equivalence is tested, never inferred** (DEC-11). A non-significant
  difference is printed as an absence, and only `tost_paired` may call anything
  equivalent -- with its margin provenance printed beside the verdict.
* **Denominators are printed automatically**, so a subset can never be quoted as
  if it were the whole set.

Usage:

    python scripts/aggregate_arms.py outputs/run_a outputs/run_b ...
    python scripts/aggregate_arms.py --compare "A=arm3_A_...:3" "B=arm3_B_...:3" ...
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.stats import (  # noqa: E402
    M2_1_EQUIVALENCE_MARGIN,
    M2_1_MARGIN_PROVENANCE,
    bootstrap_ci,
    cohens_dz,
    tost_paired,
)

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover - scipy is a dev dependency here
    wilcoxon = None


# Statuses that mean the cell was never measured, as opposed to measured and
# empty. Deliberately NOT included: `partial_retrieval` (982 rows corpus-wide)
# and `empty_retrieval` (306), which are legitimate outcomes -- the pipeline ran
# and found little or nothing, which is a result. Treating those as failures
# would discard a fifth of every artifact.
FAILED_STATUSES = frozenset({"cell_failed", "arm_failed"})


# --- metrics -----------------------------------------------------------------
#
# Each returns None where the quantity is undefined for that row, so a mean can
# never be taken over a cell that never produced the measurement. `itt` is the
# exception by design: an unanswered query is not missing data, it is a failure,
# and scoring it 1.0 is the whole point of CB3.


def itt(row: dict[str, Any]) -> float:
    emitted = row.get("keywords_emitted") or 0
    if not emitted:
        return 1.0
    return (row.get("hallucinated_keywords") or 0) / emitted


def both_spoke(row: dict[str, Any]) -> float | None:
    emitted = row.get("keywords_emitted") or 0
    if not emitted:
        return None
    return (row.get("hallucinated_keywords") or 0) / emitted


def near_miss_rate(row: dict[str, Any]) -> float | None:
    """M2.2 as the glossary specifies it: near-misses over keywords emitted.

    `None` on a mute, not 1.0. The ITT convention in `itt` above scores an
    unanswered query as total *invention* failure because inventing is the
    thing M2.1 measures and silence is its worst case. M2.2 has no such
    reading: a model that emitted nothing did not mis-transcribe anything, and
    charging it 1.0 would assert a transcription failure that did not occur.
    So this is a both-spoke metric by construction, and its printed `n` says
    over how many rows it is defined.
    """
    emitted = row.get("keywords_emitted") or 0
    if not emitted:
        return None
    return (row.get("near_miss_keywords") or 0) / emitted


def out_of_window_rate(row: dict[str, Any]) -> float | None:
    """M2.3 as specified: out-of-window over keywords emitted.

    **Not arm-comparable whenever invalid rates differ.** Dividing by
    everything emitted dilutes an arm's real keywords in proportion to how
    much it invented, so the arm that invents more is flattered. Exported and
    computed because it is what the glossary defines; paired against another
    arm only with `out_of_window_conditional` beside it.
    """
    emitted = row.get("keywords_emitted") or 0
    if not emitted:
        return None
    return (row.get("out_of_window_keywords") or 0) / emitted


def out_of_window_conditional(row: dict[str, Any]) -> float | None:
    """R24(a). Out-of-window over the keywords that are REAL in the corpus.

    The question capability 1 actually asks: of the terms the model named that
    the corpus contains, what fraction lay outside the window it was shown?
    This is the arm-comparable form, and on 2026-08-20 it reversed the
    specified form's conclusion (0.126 vs 0.144 became 0.381 vs 0.204).

    Reads `keywords_real` where the row carries a usable value and
    reconstructs it otherwise, by the identity stated in the `keywords_real`
    docstring: `kept + out_of_window + removed_by_cleaning`.

    **Two reasons the stamp is not simply trusted.** The column arrived on
    2026-08-21, so older artifacts lack it entirely. And
    `eval_20260827_arm1-2_llama70b-8b_100q_topup` -- the Arm 1 vs Arm 2
    baseline Paper #1's F1 is built on -- stamps it **0 on 82 of its 100
    rows** while those same rows report up to 43 kept keywords. `real < kept`
    is arithmetically impossible, since the real bucket contains the kept one,
    so such a value is a stamping failure rather than a measurement and is
    repaired here. Trusted stamps are never second-guessed: the reconstruction
    fires only when the value is absent or impossible, and
    `conditional_repairs` counts every row it touched so a run can never be
    quoted without knowing how much of its denominator was rebuilt.
    """
    kept = row.get("keywords_kept") or 0
    real = row.get("keywords_real")
    if real is None or real < kept:
        real = (
            kept
            + (row.get("out_of_window_keywords") or 0)
            + (row.get("removed_by_cleaning_keywords") or 0)
        )
    if not real:
        return None
    return (row.get("out_of_window_keywords") or 0) / real


def conditional_repairs(rows: Sequence[dict[str, Any]]) -> int:
    """How many rows `out_of_window_conditional` had to rebuild a denominator for.

    Printed beside the metric rather than logged, because a conditional rate
    computed over a repaired denominator is a different provenance claim from
    one computed over a stamped denominator, and only the count distinguishes
    them in the output.
    """
    return sum(
        1
        for r in rows
        if r.get("keywords_real") is None
        or (r.get("keywords_real") or 0) < (r.get("keywords_kept") or 0)
    )


def grounding(row: dict[str, Any]) -> float | None:
    """M3. Passed through as exported, including its nullability.

    `grounding_accuracy` is already `None` -- not 1.0 -- on a row that cited
    nothing, because a run recommending no courses has no grounding to
    measure and scoring it perfect rewards producing nothing. That decision
    belongs to the export; this only declines to undo it. The printed `n` is
    therefore the scored cohort and will differ from `cell.n` on any arm that
    mutes, which is exactly the denominator a grounding figure must carry.
    """
    value = row.get("grounding_accuracy")
    return None if value is None else float(value)


def citation_coverage(row: dict[str, Any]) -> float | None:
    """M3.2. Cited over retrieved -- how much of the context was used.

    The necessary companion to `grounding`, which is silent about it: a row
    citing 1 of 82 retrieved courses scores 1.0 grounding and has used almost
    nothing. `None` when nothing was retrieved; a ratio with a zero
    denominator is absent, not zero.
    """
    value = row.get("citation_coverage")
    if value is not None:
        return float(value)
    retrieved = row.get("retrieved_count") or 0
    if not retrieved:
        return None
    return (row.get("cited_count") or 0) / retrieved


def _plain(field: str) -> Callable[[dict[str, Any]], float]:
    def get(row: dict[str, Any]) -> float:
        return float(row.get(field) or 0)

    return get


# Ordered so that a metric and the companion that stops it being misread sit
# adjacent in the printout: the two M2.3 forms together, grounding immediately
# above the coverage figure that qualifies it.
METRICS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "invention_itt": itt,
    "invention_both_spoke": both_spoke,
    "near_miss_rate": near_miss_rate,
    "out_of_window_rate": out_of_window_rate,
    "out_of_window_conditional": out_of_window_conditional,
    "grounding": grounding,
    "citation_coverage": citation_coverage,
    "retrieved": _plain("retrieved_count"),
    "keywords_emitted": _plain("keywords_emitted"),
    "cited": _plain("cited_count"),
}

# Metrics whose printed mean must never be read without the one beside it.
# `describe` prints the warning under the pair rather than leaving it to the
# reader's memory, because the specified M2.3 form reverses capability 1's
# conclusion and a grounding mean over a cited-nothing cohort is meaningless.
PAIRED_READING = {
    "out_of_window_rate": (
        "read with out_of_window_conditional -- the specified form is not "
        "arm-comparable when invalid rates differ (R24a)"
    ),
    "grounding": (
        "read with citation_coverage -- grounding is 1.0 on a row that cited "
        "one course of eighty-two (R24b)"
    ),
}

COUNTS = {
    "overrun": "selection_overrun",
    "schema_violation": "selection_schema_violation",
    "platforms_lost": "platforms_lost",
    "near_miss": "near_miss_keywords",
    "out_of_window": "out_of_window_keywords",
}


@dataclass
class Cell:
    """One (run, arm) group of rows, keyed for pairing by (query_id, repeat).

    The repeat is part of the key, not decoration. Keyed on `query_id` alone a
    `-r 3` run silently collapses to its LAST pass -- two thirds of the rows
    discarded with no warning, while `_repeats_per_cell` still reports 3 and so
    drops the single-pass caveat from the TOST verdict. That is a green check
    measuring something other than what it names.
    """

    label: str
    rows: dict[tuple[str, int], dict[str, Any]]

    @property
    def n(self) -> int:
        return len(self.rows)

    def stamp(self) -> str:
        def uniq(field: str) -> str:
            vals = {str(r.get(field) or "") for r in self.rows.values()}
            return "/".join(sorted(v for v in vals if v)) or "-"

        inj_vals = {str(bool(r.get("candidate_injection"))) for r in self.rows.values() if "candidate_injection" in r}
        inj_str = "/".join(sorted(inj_vals)) if inj_vals else "-"

        return (
            f"model={uniq('keyword_model').split('/')[-1]} arm={uniq('arm')} "
            f"sweep={uniq('sweep_version')} variant={uniq('schema_prompt_variant')} "
            f"inject={inj_str}"
        )

    def mutes(self) -> int:
        return sum(1 for r in self.rows.values() if not (r.get("keywords_emitted") or 0))


def load(
    path: Path,
    arm: str | None = None,
    model: str | None = None,
    injected: bool | None = None,
) -> Cell:
    """One run directory, optionally narrowed to a single arm, model, and/or candidate injection.

    The model filter is a substring match on `keyword_model`, because the early
    Arm 1/2 runs predate the `arm` column and stamp it empty -- their only
    distinguishing field is the model ID. Without it those runs cannot enter a
    paired comparison at all, which is how the fp8-vs-bf16 contrast came to be
    computed by hand the first time.
    """
    data = json.loads((path / "benchmark_eval.json").read_text(encoding="utf-8"))

    def keep(r: dict[str, Any]) -> bool:
        if not r.get("query_id"):
            return False
        # C20. A cell the transport never delivered is not a measurement.
        #
        # `cell_failed` / `arm_failed` rows are written with
        # `keywords_emitted = 0`, and `itt` above scores anything that emitted
        # nothing as 1.0 -- so without this filter an HTTP 429 is indistinguish-
        # able from a model that chose to stay silent, and `Cell.mutes()` counts
        # it as one. Those are claims about different things: silence is a
        # property of the treatment, a 429 is a property of the afternoon.
        #
        # Found when all three E5 r3 static lanes died on upstream congestion at
        # DeepInfra. It was already live: `arm1_merged_70b_v5fp8_4pass` reported
        # ITT 0.3914 over n=200 with 2 mutes, and both "mutes" were 429s. The
        # corrected figure is 0.3852 over n=198 with ZERO mutes -- the ITT moves
        # -0.0061, but the sentence "the 70B sometimes declines" had no evidence
        # behind it at all.
        #
        # **Absent is unknown, never failed.** A row with no `status` key predates
        # the column and is kept, because dropping it would be the mirror-image
        # error: a silent corpus-wide loss of every run made before the field
        # existed.
        if r.get("status") in FAILED_STATUSES:
            return False
        if arm is not None and str(r.get("arm")) != str(arm):
            return False
        if model is not None and model not in str(r.get("keyword_model") or ""):
            return False
        if injected is not None:
            # **Absent is unknown, never False.** `candidate_injection` was
            # added on 2026-09-08; every run before it omits the key entirely,
            # and `bool(None)` would silently answer "not injected" for a row
            # that never recorded whether it was. That is the same error
            # `serving_precision` exports `""` to avoid -- a row that does not
            # say is not a row that ran at the default. An unstamped row
            # therefore matches neither `injected=True` nor `injected=False`,
            # and the empty-result message below names the reason.
            if "candidate_injection" not in r or r.get("candidate_injection") is None:
                return False
            if bool(r.get("candidate_injection")) != injected:
                return False
        return True

    kept = [r for r in data if keep(r)]

    # C20. Say so when transport failures were dropped. A silently shrinking `n`
    # is its own defect: it turns "we could not measure 2 of these" into "we
    # measured 198", and nothing downstream would ever ask the difference.
    dropped = [
        r for r in data
        if r.get("query_id") and r.get("status") in FAILED_STATUSES
    ]
    if dropped:
        shown = ", ".join(sorted({str(r.get("query_id")) for r in dropped})[:6])
        more = "" if len(dropped) <= 6 else f", +{len(dropped) - 6} more"
        print(
            f"  note: {len(dropped)} row(s) in {path.name} were never measured "
            f"(status in {sorted(FAILED_STATUSES)}) and are excluded from n: "
            f"{shown}{more}"
        )

    rows = {(r["query_id"], int(r.get("repeat") or 1)): r for r in kept}
    if not rows:
        message = f"no rows in {path} for arm={arm!r} model={model!r} injected={injected!r}"
        if injected is not None and not any(
            "candidate_injection" in r and r.get("candidate_injection") is not None
            for r in data
            if r.get("query_id")
        ):
            message += (
                "\n  This run does not stamp `candidate_injection` at all -- it "
                "predates the 2026-09-08 provenance fix. Whether it injected "
                "candidates is not recorded in the artifact, so it cannot be "
                "filtered on. Omit the filter, or take the answer from the run's "
                "task log rather than from the export."
            )
        raise SystemExit(message)
    if len(rows) != len(kept):
        raise SystemExit(
            f"{path}: {len(kept)} rows collapse to {len(rows)} (query_id, repeat) "
            "keys -- duplicate cells that are not distinguished by the `repeat` "
            "column. Refusing to guess which row wins."
        )
    inj_tag = "[injected]" if injected is True else ("[noinject]" if injected is False else "")
    tag = "".join(t for t in (f"[arm{arm}]" if arm else "",
                              f"[{model}]" if model else "",
                              inj_tag) if t)
    return Cell(f"{path.name}{tag}", rows)


# --- reporting ---------------------------------------------------------------


def describe(cell: Cell) -> None:
    print(f"\n### {cell.label}")
    print(f"    {cell.stamp()}   n={cell.n}   mutes={cell.mutes()}")
    # A multi-pass run is described per QUERY, exactly as `compare` pairs it.
    # Bootstrapping k passes of N queries as kN rows treats repeats of one query
    # as independent draws and narrows every interval by roughly sqrt(k) -- the
    # same pseudo-replication the paired path already refuses.
    multi_pass = len({rep for _, rep in cell.rows}) > 1
    n_queries = len({q for q, _ in cell.rows})
    if multi_pass:
        print(f"    multi-pass: passes averaged per query; CI and n are over QUERIES ({n_queries}), not rows")
    for name, fn in METRICS.items():
        if multi_pass:
            vals = list(by_query(cell, fn).values())
            denom = n_queries
        else:
            vals = [v for v in (fn(r) for r in cell.rows.values()) if v is not None]
            denom = cell.n
        if not vals:
            print(f"    {name:26s} -- no rows define this metric")
            continue
        ci = bootstrap_ci(vals)
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "(n<2)"
        # The denominator is printed for every metric, not only the ones where
        # it currently differs, so a reader never has to check whether it does.
        print(f"    {name:26s} mean={st.mean(vals):9.4f}  95% CI={ci_text:22s} n={len(vals)}/{denom}")
        if name in PAIRED_READING:
            print(f"    {'':26s} ^ {PAIRED_READING[name]}")
        if name == "out_of_window_conditional":
            repaired = conditional_repairs(list(cell.rows.values()))
            if repaired:
                print(
                    f"    {'':26s} ^ denominator rebuilt on {repaired}/{cell.n} rows "
                    "(keywords_real absent or below keywords_kept)"
                )
    for name, field in COUNTS.items():
        print(f"    {name:26s} total={sum(int(r.get(field) or 0) for r in cell.rows.values())}")
    _repeat_spread(cell)


def _repeat_spread(cell: Cell) -> None:
    """Per-pass means of the primary metric, and the spread between them.

    This is the question a `-r N` run exists to answer: repeats shrink
    run-to-run noise and do nothing to between-query heterogeneity, so the
    spread here -- not the SD of the paired differences -- says whether more
    passes would narrow anything.
    """
    passes: dict[int, list[float]] = {}
    for (_, rep), row in cell.rows.items():
        passes.setdefault(rep, []).append(itt(row))
    if len(passes) < 2:
        return
    means = {rep: st.mean(v) for rep, v in sorted(passes.items())}
    spread = max(means.values()) - min(means.values())
    print(f"    {'invention_itt by pass':22s} "
          + "  ".join(f"r{rep}={m:.4f}(n={len(passes[rep])})" for rep, m in means.items()))
    print(f"    {'  spread across passes':22s} {spread:.4f}"
          + (f"   sd={st.stdev(list(means.values())):.4f}" if len(means) > 2 else ""))


def by_query(cell: "Cell", fn: Callable[[dict[str, Any]], float | None]) -> dict[str, float]:
    """One value per query, averaging a metric over that query's repeat passes.

    `Cell.rows` is keyed by `(query_id, repeat)` so CB6 cannot recur -- but that
    key must not reach the paired tests. Pairing on it treats three passes over
    fifty queries as **150 independent observations**, tripling n while adding
    almost no information: repeats of one query at temperature 0 are near copies,
    so the standard error falls by roughly sqrt(3) and every p-value comes out
    smaller than the evidence supports. On the Condition C repeat sweep that is
    p=0.3516 (n=150) against the correct p=0.7297 (n=50) -- a different verdict,
    not a rounding difference.

    Averaging the passes first is the correct use of repeats: it reduces
    measurement error per query without inventing observations. Identified in
    prose 2026-09-07 (`lane-b-v5-recovers-retrieval-and-repeats-are-cancelled`
    Section 2, "Error 1 -- the n=150 TOST was pseudo-replicated"); the correction
    reached the code 2026-09-08.

    Passes where the metric is undefined (`fn` returns None -- a mute under
    `invention_both_spoke`, say) are skipped, not counted as zero. A query whose
    every pass is undefined is absent from the result and so drops out of the
    pairing rather than contributing a fabricated value.

    Single-pass runs are unaffected: a mean over one value is that value, so every
    previously reported one-repeat figure is reproduced exactly.
    """
    acc: dict[str, list[float]] = {}
    for (qid, _repeat), row in cell.rows.items():
        v = fn(row)
        if v is None:
            continue
        acc.setdefault(qid, []).append(v)
    return {q: st.mean(v) for q, v in acc.items()}


def compare(a: Cell, b: Cell, *, margin: float, provenance: str) -> None:
    ra, rb = _repeats_per_cell(a), _repeats_per_cell(b)
    ids = sorted({q for q, _ in a.rows} & {q for q, _ in b.rows})
    print(f"\n### PAIRED  {a.label}  ->  {b.label}")
    print(f"    matched on query_id: n={len(ids)}  (of {a.n} and {b.n} rows)")
    if max(ra, rb) > 1:
        print(f"    repeats averaged per query before pairing: {ra} and {rb} "
              f"passes -- n is QUERIES, not rows")
    if not ids:
        print("    nothing to compare")
        return
    for name, fn in METRICS.items():
        av, bv = by_query(a, fn), by_query(b, fn)
        xs, ys = [], []
        for q in ids:
            x, y = av.get(q), bv.get(q)
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if len(xs) < 2:
            print(f"    {name:22s} n={len(xs)} -- too few paired observations")
            continue
        diffs = [y - x for x, y in zip(xs, ys)]
        ci = bootstrap_ci(diffs)
        p = wilcoxon(xs, ys).pvalue if wilcoxon and any(d for d in diffs) else float("nan")
        dz = cohens_dz(xs, ys)
        ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "(n<2)"
        # dz is undefined when the paired differences have no spread; printing
        # "n/a" beats crashing on a comparison whose means are still meaningful.
        dz_text = f"{dz:+.3f}" if dz is not None else "  n/a"
        print(
            f"    {name:22s} {st.mean(xs):8.4f} -> {st.mean(ys):8.4f}  "
            f"delta={st.mean(diffs):+.4f} {ci_text:22s} dz={dz_text} p={p:.4f} n={len(xs)}"
        )
    _equivalence(a, b, ids, margin=margin, provenance=provenance)


def _equivalence(a: Cell, b: Cell, ids: Sequence[str], *, margin: float, provenance: str) -> None:
    """TOST on the primary metric only.

    Deliberately not run on every metric. TOST answers "is this difference
    small enough to be uninteresting", which is only a question worth asking
    where a null is being claimed. Running it on a significant difference
    invites reading a failed equivalence test as a second significance test.
    """
    # Averaged per query, exactly as `compare` does. A TOST over pooled passes is
    # the pseudo-replication described in `by_query`, and TOST is the output most
    # likely to be quoted as a verdict.
    av, bv = by_query(a, itt), by_query(b, itt)
    paired = [q for q in ids if q in av and q in bv]
    xs = [av[q] for q in paired]
    ys = [bv[q] for q in paired]
    repeats = _repeats_per_cell(a) or 1
    result = tost_paired(
        xs, ys, margin=margin, margin_provenance=provenance, repeats_per_cell=repeats
    )
    print(f"\n    TOST on invention_itt, margin +/-{result.margin}")
    verdict = "EQUIVALENT" if result.equivalent else "NOT SHOWN EQUIVALENT"
    print(f"      {verdict}   p={result.p_value:.4f} (lower {result.p_lower:.4f}, upper {result.p_upper:.4f})")
    print(f"      mean difference {result.mean_difference:+.4f} over n={result.n_pairs} pairs")
    if not result.equivalent:
        print("      NOTE: this is an absence, not equivalence. Report as "
              "'did not close the gap', never 'is equivalent to'.")
    for caveat in result.caveats:
        print(f"      CAVEAT: {caveat}")
    print(f"      margin provenance: {result.margin_provenance}")


def _repeats_per_cell(cell: Cell) -> int:
    """How many passes produced this cell, read from the rows themselves.

    Passed to `tost_paired` so a single-pass run cannot yield a bare
    `equivalent=True`: with one draw per cell the paired differences carry query
    heterogeneity and run-to-run noise inseparably, and this instrument is known
    to exhibit the latter.
    """
    repeats = {int(r.get("repeat") or 1) for r in cell.rows.values()}
    return max(repeats) if repeats else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories to describe")
    ap.add_argument(
        "--compare",
        action="append",
        default=[],
        help="LABEL=path[:arm | :arm=N,model=SUBSTR], given twice or more; consecutive pairs are compared",
    )
    ap.add_argument("--margin", type=float, default=M2_1_EQUIVALENCE_MARGIN)
    ap.add_argument("--margin-provenance", default=M2_1_MARGIN_PROVENANCE)
    args = ap.parse_args()

    if not args.runs and not args.compare:
        ap.error("give run directories, --compare pairs, or both")

    for run in args.runs:
        describe(load(Path(run)))

    cells = []
    for spec in args.compare:
        label, _, target = spec.partition("=")
        path, _, sel = target.partition(":")
        arm = model = None
        injected = None
        if "=" in sel:
            # `arm=3,model=qwen,inject=true` -- needed once a run directory holds more than
            # one model, or holds rows whose `arm` column predates the field.
            for part in sel.split(","):
                k, _, v = part.partition("=")
                k_clean = k.strip()
                v_clean = v.strip()
                if k_clean == "arm":
                    arm = v_clean
                elif k_clean == "model":
                    model = v_clean
                elif k_clean in ("injected", "inject", "candidate_injection"):
                    injected = v_clean.lower() in ("1", "true", "yes", "t", "y")
                else:
                    raise SystemExit(f"unknown selector {k!r} in {spec!r}")
        elif sel:
            arm = sel
        cell = load(Path(path), arm=arm, model=model, injected=injected)
        cell.label = label or cell.label
        cells.append(cell)
    for first, second in zip(cells[::2], cells[1::2]):
        compare(first, second, margin=args.margin, provenance=args.margin_provenance)
    if len(cells) % 2:
        print(f"\n### {cells[-1].label} (unpaired)")
        describe(cells[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
