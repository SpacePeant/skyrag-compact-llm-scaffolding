#!/usr/bin/env python
r"""Pre-registered analysis of the 2x3 factorial. Written before the data lands.

Tonight's lane roster generates up to 600 factorial rows and **nothing in this
repo reads a 2x3**. Every Arm 2/3 comparison so far has been one cell against
one other cell, which is a t-test wearing a factorial's clothes: it cannot say
whether the two axes interact, which is the only question a factorial exists to
answer.

**This is written against the design, not the numbers.** That is what makes it
pre-registered, and it is the whole point of writing it at 23:47 while the
lanes are still queued: a test chosen after seeing the data is a test chosen to
pass.

---

## The design

Two axes, crossed, **fully within-subject** -- every cell runs the same 50
benchmark queries:

* **Axis 1, context scaffolding**: `static` (top-1,000 window) vs `inject`
  (query-matched candidate keywords).
* **Axis 2, format scaffolding**: `L0` unassisted, `L1` bare schema,
  `L2` schema + worked exemplar.

Because every query appears in every cell, all tests below are **paired**, and
every one of them is computed on **complete cases** -- the queries present in
all cells that test touches -- with that n printed beside it.

## The tests

All nonparametric, matching the Wilcoxon commitment in Section 3.5. The two
axes get different tests because they have different numbers of levels.

1. **Main effect of context** (2 levels): per query, average across the three
   format levels within each context, then paired Wilcoxon.
2. **Main effect of format** (3 levels): per query, average across the two
   contexts, then **Friedman** across L0/L1/L2, with Holm-corrected pairwise
   Wilcoxon post-hocs.
3. **Interaction** (2 df, so two orthogonal contrasts). Each is a
   difference-of-differences computed *per query* and Wilcoxon-tested against
   zero, Holm-corrected across the two:

       I1 = (inject.L1 - inject.L0) - (static.L1 - static.L0)
       I2 = (inject.L2 - inject.L0) - (static.L2 - static.L0)

   A difference-of-differences per query is the interaction contrast itself, so
   this needs no distributional assumption and no cell-variance homogeneity --
   neither of which this pipeline has ever been shown to satisfy.

## What is reported, and why it is reported twice

* **ITT and As-Treated are computed separately and never pooled**, per
  `AUDIT_DISCREPANCY_LEDGER.md` 3.1. They have different cohorts: a muted query
  scores 1.0 under ITT and is *absent* under As-Treated. Their n are printed
  separately because they are not the same n.
* **Macro and micro are both printed and both labelled**, per ledger 3.2. The
  published 42.2% drop is a pooled micro-average; the query-level macro is
  51.2%. A factorial table that does not say which it is repeats exactly the
  defect the ledger exists to record.
* **A missing or empty cell is `None`, never `0.0`.** Round 2 may not finish and
  `inject x L2` may not exist at all (T0). Any test touching an absent cell is
  reported as not computable, naming the cell -- a factorial that quietly drops
  a cell reports a 2x2 under a 2x3's name.

## Cell assignment, and why it is not inferred

**Cells are assigned explicitly and then verified against the row stamps.**
Auto-discovery from provenance alone is refused for any row that does not stamp
`candidate_injection`, because for the historical runs it is not merely missing
-- it is *actively ambiguous*:

    outputs/arm3_A_spec_only_20260831        arm=3 variant=spec_only inject=<unstamped>
    outputs/eval_20260907_arm2-3_condition_c_50q (arm 3)
                                             arm=3 variant=spec_only inject=<unstamped>

Those two rows carry **identical stamps and sit in opposite cells of axis 1**.
The first is static, the second is Condition C; only the directory name says so.
Inferring `static` from an absent `candidate_injection` would silently merge a
static cell into an injected one -- the "absent is never the default" rule, with
a live instance behind it. Tonight's lanes stamp the field, so discovery works
for them; the refusal protects the historical runs from being swept in.

Note also that `schema_prompt_variant` **conflates the two axes**:
`spec_plus_candidate_injection` names a format level (L1) and a context
(inject) in one string. Format level is therefore derived from arm plus
variant, and context only from `candidate_injection`.

## Usage

    python scripts/analyze_factorial.py \
        --cell static:L0=outputs/fact_static_L0_r1,outputs/fact_static_L0_r2 \
        --cell static:L1=outputs/fact_static_L1_r1 \
        ... \
        --metric invention

    python scripts/analyze_factorial.py --discover outputs/fact_*   # stamped runs only

Repeats may be passed either as several directories for one cell, or as one
directory already merged by `merge_lane_repeats.py`. Both are averaged per
query before any test -- pairing on `(query_id, repeat)` would turn 50 queries
x 3 passes into 150 "independent" observations and shrink the standard error by
about sqrt(3).
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics as st
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skyrag.stats import tost_paired  # noqa: E402  (after the sys.path insert above)

ROOT = Path(__file__).resolve().parent.parent

# Imported, never edited: Session A holds `aggregate_arms.py` tonight, and the
# metric definitions must not fork. `plot_figures.py` consumes it the
# same way.
_SPEC = importlib.util.spec_from_file_location("aggregate_arms", ROOT / "scripts" / "aggregate_arms.py")
agg = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_arms"] = agg
_SPEC.loader.exec_module(agg)

try:
    from scipy.stats import friedmanchisquare, wilcoxon
except ImportError:  # pragma: no cover - scipy is a dev dependency here
    wilcoxon = friedmanchisquare = None

CONTEXTS = ("static", "inject")
LEVELS = ("L0", "L1", "L2")
CELLS = tuple((c, l) for c in CONTEXTS for l in LEVELS)

# Format level is arm plus variant. `spec_plus_candidate_injection` is a bare
# schema that happens to have been run under injection -- the string names both
# axes, so only its format half is read here.
L1_VARIANTS = {"spec_only", "spec_plus_candidate_injection"}
L2_VARIANTS = {"spec_plus_exemplar", "spec_plus_exemplar_candidate_injection"}


class Ambiguous(Exception):
    """A row that cannot be placed in a cell from what it stamps."""


# --- cell assignment ---------------------------------------------------------


def level_of(row: dict[str, Any]) -> str:
    arm = str(row.get("arm") or "")
    variant = str(row.get("schema_prompt_variant") or "")
    if arm == "2":
        if variant:
            raise Ambiguous(f"arm 2 row carries schema_prompt_variant={variant!r}; L0 is the unschema'd level")
        return "L0"
    if arm == "3":
        if variant in L1_VARIANTS:
            return "L1"
        if variant in L2_VARIANTS:
            return "L2"
        raise Ambiguous(f"arm 3 row has unrecognised schema_prompt_variant={variant!r}")
    raise Ambiguous(f"arm={arm!r} is not a factorial arm (expected 2 or 3)")


def context_of(row: dict[str, Any]) -> str:
    """Context axis, from `candidate_injection` only.

    **An unstamped row is refused, not defaulted to static.** Historical runs
    omit the field, and two rows with identical stamps sit in opposite cells of
    this axis -- see the module docstring. `bool(None)` answering "static" here
    is the C10 defect with a factorial's blast radius.
    """
    if "candidate_injection" not in row or row.get("candidate_injection") is None:
        raise Ambiguous(
            "row does not stamp `candidate_injection`, so its context axis is "
            "unrecorded. It predates the 2026-09-08 provenance fix; assign it "
            "with an explicit --cell rather than --discover."
        )
    return "inject" if bool(row.get("candidate_injection")) else "static"


@dataclass
class Cell:
    """One factorial cell: per-query values, already averaged across repeats."""

    context: str
    level: str
    sources: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.context, self.level)

    @property
    def label(self) -> str:
        return f"{self.context}.{self.level}"

    def per_query(self, fn: Callable[[dict[str, Any]], float | None]) -> dict[str, float]:
        """Mean within a query across its repeats. `None` contributes nothing.

        A metric that is `None` for every repeat of a query leaves that query
        out of the returned map entirely -- which is what makes the As-Treated
        cohort smaller than the ITT one, rather than silently zero-filled.
        """
        buckets: dict[str, list[float]] = {}
        for row in self.rows:
            value = fn(row)
            if value is not None:
                buckets.setdefault(str(row["query_id"]), []).append(value)
        return {q: st.fmean(v) for q, v in buckets.items()}

    def micro(self, numerator: str, denominator: str) -> float | None:
        """Pooled ratio over all rows. `None` when the denominator is zero.

        The micro form weights a query that emitted 90 keywords nine times as
        heavily as one that emitted 10. It is reported *beside* the macro form,
        never instead of it (ledger 3.2).
        """
        num = sum(r.get(numerator) or 0 for r in self.rows)
        den = sum(r.get(denominator) or 0 for r in self.rows)
        return None if not den else num / den

    def mutes(self) -> int:
        return sum(1 for r in self.rows if not (r.get("keywords_emitted") or 0))


def load_cells(
    explicit: dict[tuple[str, str], list[Path]],
    discover: list[Path],
    model: str | None = None,
) -> tuple[dict[tuple[str, str], Cell], list[str]]:
    """Build the six cells. Returns the cells and a list of notes for the report."""
    cells = {key: Cell(*key) for key in CELLS}
    notes: list[str] = []

    for key, paths in explicit.items():
        for path in paths:
            rows = _rows(path)
            if model:
                # One directory routinely holds two models (the Arm 1/2 runs)
                # and they are different cells. Substring match on
                # `keyword_model`, the same rule `aggregate_arms.load` uses,
                # because the early runs stamp `arm` as '' and the model string
                # is the only thing that separates them.
                rows = [r for r in rows if model in str(r.get("keyword_model") or "")]
                if not rows:
                    raise SystemExit(f"{path}: no rows match --model {model!r}.")

            # The assignment is the caller's, but it is checked wherever the
            # row can speak for itself. A row that *states* a context or level
            # contradicting its --cell is a mis-specified lane, and analysing
            # it anyway would report the wrong cell under the right name.
            for row in rows:
                try:
                    stamped = context_of(row)
                except Ambiguous:
                    continue
                if stamped != key[0]:
                    raise SystemExit(
                        f"{path}: assigned to {key[0]}.{key[1]} but its rows stamp "
                        f"candidate_injection={row.get('candidate_injection')!r}, i.e. "
                        f"{stamped}. Refusing to file a row under a context it "
                        "contradicts."
                    )
                break

            # A directory may hold several format levels -- an `--arm 2,3`
            # sweep is two cells in one file. Keep the rows whose level matches
            # the declaration, and say how many were left behind: dropping them
            # silently is how `inject.L0` and `inject.L1` first came back with
            # identical numbers, both having swallowed the whole two-arm run.
            kept, dropped = [], 0
            for row in rows:
                try:
                    level = level_of(row)
                except Ambiguous:
                    # The row cannot say which level it is (the Arm 1/2 runs
                    # stamp `arm` as ''), so the explicit declaration stands.
                    kept.append(row)
                    continue
                if level == key[1]:
                    kept.append(row)
                else:
                    dropped += 1
            if not kept:
                raise SystemExit(
                    f"{path}: no rows are {key[1]} (dropped {dropped} of another level). "
                    "Check the --cell assignment."
                )
            cells[key].rows.extend(kept)
            suffix = f" [{len(kept)} rows" + (f", {dropped} of another level skipped]" if dropped else "]")
            cells[key].sources.append(path.name + suffix)

    for path in discover:
        rows = _rows(path)
        if model:
            rows = [r for r in rows if model in str(r.get("keyword_model") or "")]
            if not rows:
                raise SystemExit(f"{path}: no rows match --model {model!r}.")
        placed: dict[tuple[str, str], int] = {}
        for row in rows:
            try:
                key = (context_of(row), level_of(row))
            except Ambiguous as exc:
                raise SystemExit(
                    f"{path}: cannot place a row by its stamps -- {exc}\n"
                    "  Use --cell to assign this run explicitly."
                )
            cells[key].rows.append(row)
            placed[key] = placed.get(key, 0) + 1
        for key, n in placed.items():
            cells[key].sources.append(f"{path.name}({n})")
        if len(placed) > 1:
            notes.append(f"{path.name} spans {len(placed)} cells: {', '.join(f'{c}.{l}' for c, l in placed)}")

    return cells, notes


def _rows(path: Path) -> list[dict[str, Any]]:
    import json

    artifact = path / "benchmark_eval.json"
    if not artifact.exists():
        raise SystemExit(f"{path}: no benchmark_eval.json. Not a run directory.")
    rows = [r for r in json.loads(artifact.read_text(encoding="utf-8")) if r.get("query_id")]
    if not rows:
        raise SystemExit(f"{path}: no rows carry a query_id.")
    return rows


# --- tests -------------------------------------------------------------------


@dataclass
class Result:
    name: str
    n: int | None
    statistic: float | None
    p: float | None
    effect: float | None = None
    note: str = ""

    @property
    def computable(self) -> bool:
        return self.p is not None


def _paired(a: dict[str, float], b: dict[str, float]) -> tuple[list[float], list[float], list[str]]:
    ids = sorted(set(a) & set(b))
    return [a[q] for q in ids], [b[q] for q in ids], ids


def _dz(diffs: list[float]) -> float | None:
    """Paired Cohen's d, signed first-minus-second, as `aggregate_arms` signs it."""
    if len(diffs) < 2:
        return None
    sd = st.stdev(diffs)
    return None if not sd else st.fmean(diffs) / sd


def _wilcoxon(xs: list[float], ys: list[float], name: str) -> Result:
    if wilcoxon is None:
        return Result(name, len(xs), None, None, note="scipy unavailable")
    if len(xs) < 2:
        return Result(name, len(xs), None, None, note="fewer than 2 paired observations")
    diffs = [x - y for x, y in zip(xs, ys)]
    if not any(diffs):
        # Every pair identical. Wilcoxon is undefined here, and reporting p=1.0
        # would read as "tested and found no difference" rather than "there is
        # nothing to test".
        return Result(name, len(xs), None, None, note="all paired differences are zero")
    result = wilcoxon(xs, ys)
    return Result(name, len(xs), float(result.statistic), float(result.pvalue), _dz(diffs))


def holm(results: list[Result]) -> list[Result]:
    """Holm-Bonferroni over the computable results, in place of no correction.

    Only the computable ones enter the family: a test that could not run is not
    a test that passed, and counting it would make the correction *weaker* for
    the tests that did run.
    """
    live = [r for r in results if r.computable]
    order = sorted(range(len(live)), key=lambda i: live[i].p)
    m = len(live)
    previous = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * live[idx].p)
        adjusted = max(adjusted, previous)
        previous = adjusted
        live[idx].note = (live[idx].note + " " if live[idx].note else "") + f"Holm p={adjusted:.4g}"
    return results


def main_effect_context(cells: dict[tuple[str, str], Cell], fn) -> Result:
    """Average the three format levels within each context, then pair."""
    per_context: dict[str, dict[str, float]] = {}
    for context in CONTEXTS:
        maps = []
        for level in LEVELS:
            cell = cells[(context, level)]
            if not cell.rows:
                return Result("context (static vs inject)", None, None, None,
                              note=f"cell {context}.{level} is empty")
            maps.append(cell.per_query(fn))
        ids = set.intersection(*(set(m) for m in maps))
        per_context[context] = {q: st.fmean(m[q] for m in maps) for q in ids}
    xs, ys, _ = _paired(per_context["static"], per_context["inject"])
    return _wilcoxon(xs, ys, "context (static vs inject)")


def main_effect_format(cells: dict[tuple[str, str], Cell], fn) -> tuple[Result, list[Result]]:
    """Friedman across the three levels, then Holm-corrected pairwise Wilcoxon."""
    per_level: dict[str, dict[str, float]] = {}
    for level in LEVELS:
        maps = []
        for context in CONTEXTS:
            cell = cells[(context, level)]
            if not cell.rows:
                return (Result("format (L0/L1/L2)", None, None, None,
                               note=f"cell {context}.{level} is empty"), [])
            maps.append(cell.per_query(fn))
        ids = set.intersection(*(set(m) for m in maps))
        per_level[level] = {q: st.fmean(m[q] for m in maps) for q in ids}

    ids = sorted(set.intersection(*(set(per_level[l]) for l in LEVELS)))
    if len(ids) < 2 or friedmanchisquare is None:
        omnibus = Result("format (L0/L1/L2)", len(ids), None, None,
                         note="fewer than 2 complete cases" if len(ids) < 2 else "scipy unavailable")
    else:
        columns = [[per_level[l][q] for q in ids] for l in LEVELS]
        # "No variation across levels" means each *query* scored the same at
        # every level -- so compare across the row, not down the column. The
        # first draft tested `len(set(column)) == 1`, which asks whether every
        # query scored alike *within* a level. That is a perfectly ordinary
        # thing for a tight cell to do, and it made a real three-level
        # difference report as nothing to test.
        if all(len(set(row)) == 1 for row in zip(*columns)):
            omnibus = Result("format (L0/L1/L2)", len(ids), None, None, note="no variation across levels")
        else:
            stat, p = friedmanchisquare(*columns)
            omnibus = Result("format (L0/L1/L2)", len(ids), float(stat), float(p))

    pairwise = []
    for i, a in enumerate(LEVELS):
        for b in LEVELS[i + 1:]:
            xs, ys, _ = _paired(per_level[a], per_level[b])
            pairwise.append(_wilcoxon(xs, ys, f"  {a} vs {b}"))
    holm(pairwise)
    return omnibus, pairwise


def interaction(cells: dict[tuple[str, str], Cell], fn) -> list[Result]:
    """Two orthogonal difference-of-differences contrasts, Holm-corrected."""
    results = []
    for level in ("L1", "L2"):
        needed = [("static", "L0"), ("static", level), ("inject", "L0"), ("inject", level)]
        missing = [f"{c}.{l}" for c, l in needed if not cells[(c, l)].rows]
        name = f"interaction: context x ({level} - L0)"
        if missing:
            results.append(Result(name, None, None, None, note=f"needs {', '.join(missing)}"))
            continue
        maps = {(c, l): cells[(c, l)].per_query(fn) for c, l in needed}
        ids = sorted(set.intersection(*(set(m) for m in maps.values())))
        diffs = [
            (maps[("inject", level)][q] - maps[("inject", "L0")][q])
            - (maps[("static", level)][q] - maps[("static", "L0")][q])
            for q in ids
        ]
        results.append(_wilcoxon(diffs, [0.0] * len(diffs), name))
        _INTERACTION_DIFFS[name] = diffs
    holm(results)
    return results


# Populated by `interaction` so the equivalence pass can reuse the exact
# difference-of-differences the Wilcoxon saw, rather than recomputing it and
# risking the two answering slightly different questions.
_INTERACTION_DIFFS: dict[str, list[float]] = {}


def equivalence(margin: float, provenance: str, repeats_per_cell: int) -> None:
    """DEC-11. "Not significant" is not "no interaction".

    The factorial's headline claim is that context and format scaffolding do not
    interact. Every test above can do is fail to reject; a null p is equally
    consistent with additivity and with an underpowered design, and DEC-11 exists
    precisely to forbid reading the first from the second.

    This is not a test chosen after seeing the data. DEC-11 pre-registered the
    +/-0.05 margin; what was missing was anything in this script that ran it, so
    the claim the whole 12-lane factorial was commissioned to support could not
    actually be made from the script's own output.

    `tost_paired` requires `repeats_per_cell` for a reason worth restating here:
    at one pass per cell the paired differences carry query heterogeneity and
    run-to-run noise inseparably, and this instrument is known to show the
    latter -- cell means moved up to 0.06 between passes 1 and 2. A one-pass
    `equivalent=True` is therefore not quotable, and the caveat travels attached
    to the result.
    """
    if not _INTERACTION_DIFFS:
        return
    print()
    print(f"### Equivalence of the interaction contrasts (TOST, margin +/-{margin})")
    print()
    for name, diffs in _INTERACTION_DIFFS.items():
        if len(diffs) < 2:
            print(f"  {name:44} NOT COMPUTABLE -- {len(diffs)} complete case(s)")
            continue
        result = tost_paired(
            diffs,
            [0.0] * len(diffs),
            margin=margin,
            margin_provenance=provenance,
            repeats_per_cell=repeats_per_cell,
        )
        verdict = "EQUIVALENT" if result.equivalent else "not shown equivalent"
        print(f"  {name:44} n={len(diffs):3}  {verdict}")
        print(f"      mean difference    {result.mean_difference:+.5f}   "
              f"margin +/-{result.margin}   alpha {result.alpha}")
        print(f"      p_lower {result.p_lower:.4g}   p_upper {result.p_upper:.4g}   "
              f"p {result.p_value:.4g}   n_pairs {result.n_pairs}")
        for caveat in result.caveats:
            print(f"      ^ {caveat}")
    print()
    print(f"  Margin provenance: {provenance}")
    print("  A non-significant Wilcoxon above is NOT evidence of additivity; only an")
    print("  EQUIVALENT verdict here is, and only at the margin named.")


# --- reporting ---------------------------------------------------------------

METRICS = {
    "invention": (agg.itt, agg.both_spoke, "hallucinated_keywords", "keywords_emitted"),
    "retrieved": (agg.METRICS["retrieved"], agg.METRICS["retrieved"], None, None),
    "cited": (agg.METRICS["cited"], agg.METRICS["cited"], None, None),
}


def describe(cells: dict[tuple[str, str], Cell], metric: str) -> None:
    itt_fn, at_fn, num, den = METRICS[metric]
    print(f"\n## Cell descriptives — {metric}\n")
    micro_note = " (macro = mean of per-query rates; micro = pooled ratio)" if num else ""
    print(f"  {'cell':14} {'rows':>5} {'ITT n':>6} {'ITT macro':>10} {'AT n':>5} {'AT macro':>9} {'micro':>8} {'mutes':>6}")
    for key in CELLS:
        cell = cells[key]
        if not cell.rows:
            print(f"  {cell.label:14} {'-':>5} {'-':>6} {'-':>10} {'-':>5} {'-':>9} {'-':>8} {'-':>6}   EMPTY")
            continue
        itt = cell.per_query(itt_fn)
        at = cell.per_query(at_fn)
        micro = cell.micro(num, den) if num else None
        print(
            f"  {cell.label:14} {len(cell.rows):5} {len(itt):6} {st.fmean(itt.values()):10.4f} "
            f"{len(at):5} {(st.fmean(at.values()) if at else float('nan')):9.4f} "
            f"{(f'{micro:.4f}' if micro is not None else '-'):>8} {cell.mutes():6}"
        )
    print(f"\n  ITT and As-Treated are different cohorts and are never pooled (ledger 3.1).{micro_note}")


def report(results: list[Result], title: str) -> None:
    print(f"\n### {title}\n")
    for r in results:
        if not r.computable:
            print(f"  {r.name:44} NOT COMPUTABLE — {r.note}")
            continue
        effect = f"dz={r.effect:+.3f}" if r.effect is not None else "dz=-"
        print(f"  {r.name:44} n={r.n:3}  p={r.p:.4g}  {effect}  {r.note}")


def analyse(cells: dict[tuple[str, str], Cell], metric: str, basis: str) -> list[Result]:
    itt_fn, at_fn, _, _ = METRICS[metric]
    fn = itt_fn if basis == "ITT" else at_fn
    print(f"\n## {basis} — {metric}")
    results = [main_effect_context(cells, fn)]
    report(results, f"Main effect: context [{basis}]")
    omnibus, pairwise = main_effect_format(cells, fn)
    report([omnibus] + pairwise, f"Main effect: format [{basis}]")
    inter = interaction(cells, fn)
    report(inter, f"Interaction [{basis}]")
    return results + [omnibus] + pairwise + inter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", action="append", default=[], metavar="CTX:LEVEL=DIR[,DIR...]",
                        help="assign run directories to a cell explicitly, e.g. static:L1=outputs/a,outputs/b")
    parser.add_argument("--discover", nargs="*", type=Path, default=[],
                        help="place runs by their stamps; refuses any row that does not stamp candidate_injection")
    parser.add_argument("--model", default=None,
                        help="substring match on keyword_model, to slice a run directory that holds "
                             "more than one model (the Arm 1/2 runs stamp arm as '')")
    parser.add_argument("--metric", default="invention", choices=sorted(METRICS),
                        help="which measure to analyse (default: invention)")
    parser.add_argument("--equivalence-margin", type=float, default=0.05,
                        help="TOST margin for the interaction contrasts (DEC-11 "
                             "pre-registered this at 0.05). Pass 0 to skip.")
    parser.add_argument("--equivalence-provenance",
                        default="DEC-11, pre-registered before the 2026-09-23 "
                                "factorial landed: 0.05 invention rate is the "
                                "smallest difference judged pedagogically "
                                "meaningful for this measure.",
                        help="Why that margin. Required by tost_paired and "
                             "recorded in the artifact.")
    parser.add_argument("--repeats-per-cell", type=int, default=1,
                        help="Passes per cell. NOT used in the arithmetic -- it "
                             "attaches the single-pass caveat, so a one-pass "
                             "run cannot print a bare EQUIVALENT.")
    args = parser.parse_args(argv)

    explicit: dict[tuple[str, str], list[Path]] = {}
    for spec in args.cell:
        if "=" not in spec or ":" not in spec.split("=")[0]:
            raise SystemExit(f"--cell {spec!r} is malformed; expected CTX:LEVEL=DIR[,DIR...]")
        where, paths = spec.split("=", 1)
        context, level = where.split(":", 1)
        if context not in CONTEXTS or level not in LEVELS:
            raise SystemExit(f"--cell {spec!r}: context must be one of {CONTEXTS}, level one of {LEVELS}")
        explicit.setdefault((context, level), []).extend(Path(p) for p in paths.split(",") if p)

    if not explicit and not args.discover:
        raise SystemExit("nothing to analyse: pass --cell and/or --discover.")

    cells, notes = load_cells(explicit, args.discover, args.model)

    print("# 2x3 factorial analysis")
    if args.model:
        print(f"\n  model filter: {args.model!r}")
    print("\n## Cells\n")
    for key in CELLS:
        cell = cells[key]
        print(f"  {cell.label:14} {'<empty>' if not cell.rows else ', '.join(cell.sources)}")
    for note in notes:
        print(f"  note: {note}")

    empty = [f"{c}.{l}" for c, l in CELLS if not cells[(c, l)].rows]
    if empty:
        print(f"\n  ⚠ {len(empty)} cell(s) empty: {', '.join(empty)}. Tests touching them "
              "report NOT COMPUTABLE rather than a number.")

    describe(cells, args.metric)
    for basis in ("ITT", "AT"):
        analyse(cells, args.metric, basis)
        if args.equivalence_margin > 0:
            equivalence(
                args.equivalence_margin,
                args.equivalence_provenance,
                args.repeats_per_cell,
            )
        _INTERACTION_DIFFS.clear()

    print("\n---\nPre-registered: tests fixed before the data landed. "
          "Wilcoxon per Section 3.5; Friedman for the 3-level axis; Holm within each family.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
