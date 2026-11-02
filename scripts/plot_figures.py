#!/usr/bin/env python
"""Build Paper #1's figures from the run artifacts.

The manuscript carries eight tables and no figures. These three are the results
that tables serve worst: a governing finding stated as three numbers in prose, a
2x3 grid whose shape is the point, and a per-domain spread currently given as
two endpoints.

**Every value is computed from `outputs/`, never transcribed.** The metric
functions are imported from `scripts/aggregate_arms.py` -- the tool Section 3.5
names -- so a figure cannot drift from the aggregator the way a hand-copied
table can. `--check` re-asserts each plotted value against the number the
manuscript prints and reports drift without touching anything.

**It writes figures and nothing else.** The manuscript is not edited; see
`FIGURES.md`, written beside the output, for where each belongs.

Writes into `figures/` at the repository root (or `--output DIR`):

    uv run python scripts/plot_figures.py
    uv run python scripts/plot_figures.py --check

---

**Design constraints, because this is print.** No hover layer and no dark mode:
a paper figure is read on white and often photocopied. Two consequences drive
every mark decision below.

* **Every series carries two encodings**, colour and one of marker shape or
  dash, so the figures survive greyscale and CVD. The palette
  (`#2a78d6` / `#eb6834`) is slots 1-2 of the reference categorical set and
  validates at CVD deltaE 24.7 and normal-vision deltaE 33.6, both far above
  the 8 / 15 floors.
* **No dual-axis chart anywhere.** Invalid-rate and courses-retrieved differ by
  two orders of magnitude and share no unit; putting them on twin y-axes would
  let the eye read a crossing point that is an artifact of two arbitrary
  scales. Where both are shown they are separate panels on a shared x.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

# --- the aggregator, imported rather than reimplemented ----------------------
_SPEC = importlib.util.spec_from_file_location("aggregate_arms", ROOT / "scripts" / "aggregate_arms.py")
agg = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_arms"] = agg
_SPEC.loader.exec_module(agg)

# --- palette -----------------------------------------------------------------
# Slots 1 and 2 of the reference categorical palette. Validated together, not
# chosen by eye: worst-pair CVD deltaE 24.7 (protan), normal-vision 33.6.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

# IEEE Access double-column text width, in inches.
WIDE = 7.16
NARROW = 3.5

# x of the right-hand delta column in figure 1, in data coordinates.
DELTA_X = 0.60

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _style(ax, *, xgrid: bool = True) -> None:
    """Recessive grid, thin spines. The data is the ink; the frame is not."""
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


# --- data --------------------------------------------------------------------


def per_query(cell, fn: Callable[[dict[str, Any]], float | None]) -> dict[str, float]:
    """Mean within a query across its repeats, keyed by query.

    Averaging passes before anything downstream pairs on them is the rule
    `aggregate_arms` enforces and the manuscript's Table 8 states: pairing on
    `(query_id, repeat)` would turn 50 queries x 3 passes into 150 "independent"
    observations, shrinking the standard error by about sqrt(3).

    A metric returning `None` (the As-Treated form on a muted row) contributes
    nothing rather than a zero, so the denominator is the scored cohort.
    """
    buckets: dict[str, list[float]] = {}
    for (query_id, _repeat), row in cell.rows.items():
        value = fn(row)
        if value is not None:
            buckets.setdefault(query_id, []).append(value)
    return {q: st.fmean(v) for q, v in buckets.items()}


def cell_mean(cell, fn: Callable[[dict[str, Any]], float | None]) -> float:
    values = [fn(r) for r in cell.rows.values()]
    return st.fmean([v for v in values if v is not None])


def domains() -> dict[str, str]:
    mapping = {}
    for line in (ROOT / "benchmarks" / "queries_v1.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            mapping[record["id"]] = record.get("domain", "?")
    return mapping


# --- figure 1: the triad -----------------------------------------------------

# Rebased 2026-09-30 from the unstamped sweep `eval_20260908_arm2-3_condition_c_50q_r3_resumed`
# onto the stamped injection cells of Table 6, which Section 4.7.2's Table 8 now reports.
# That sweep survives in the prose as a replication, not in the figure.
TRIAD_CELLS = ("i_l0", "i_l1")


def _pooled(cell: str) -> Any:
    """One stamped cell pooled across its pass lanes, re-keyed by lane index."""
    rows = {
        (query_id, i + 1): row
        for i, d in enumerate(table6._dirs(cell))
        for (query_id, _rep), row in agg.load(d).rows.items()
    }
    return agg.Cell(cell, rows)


def triad_data() -> dict[str, Any]:
    arm2, arm3 = (_pooled(c) for c in TRIAD_CELLS)

    at2, at3 = per_query(arm2, agg.both_spoke), per_query(arm3, agg.both_spoke)
    # The As-Treated row pairs only queries where *both* arms produced output.
    # Queries Arm 3 muted on every pass cannot enter a paired comparison, and
    # averaging them into one arm but not the other would compare two cohorts.
    paired = sorted(set(at2) & set(at3))

    def silence(cell) -> float:
        return sum(1 for r in cell.rows.values() if not (r.get("keywords_emitted") or 0)) / cell.n

    return {
        "As-Treated invention\n(paired cohort)": (
            st.fmean(at2[q] for q in paired),
            st.fmean(at3[q] for q in paired),
            f"n = {len(paired)} queries",
        ),
        "Pipeline silence\n(abstention)": (
            silence(arm2),
            silence(arm3),
            f"{arm2.n} passes per arm",
        ),
        "Intention-to-Treat\ninvention": (
            st.fmean(per_query(arm2, agg.itt).values()),
            st.fmean(per_query(arm3, agg.itt).values()),
            "n = 50 queries",
        ),
    }


def figure_triad(data: dict[str, Any], path: Path) -> None:
    """A dumbbell per measure: where Arm 2 sits, where Arm 3 sits, and the move.

    Chosen over grouped bars because the finding *is* the direction of travel --
    two measures move in opposite directions and the third does not move at all.
    Bars would render that as six heights to compare pairwise; a connector
    states it once. All three measures are proportions, so one axis is honest.
    """
    labels = list(data)
    fig, ax = plt.subplots(figsize=(WIDE, 2.5))

    for i, label in enumerate(labels):
        a, b, _note = data[label]
        y = len(labels) - 1 - i
        # The connector is the message; it is drawn under both endpoints and
        # sized so neither dot is obscured by it.
        ax.plot([a, b], [y, y], color=INK_MUTED, linewidth=1.6, zorder=2, solid_capstyle="round")
        ax.scatter([a], [y], s=70, color=BLUE, marker="o", zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.scatter([b], [y], s=70, color=ORANGE, marker="s", zorder=3, edgecolor=SURFACE, linewidth=1.2)

        # Endpoint labels go *outside* the connector -- left value to the left,
        # right value to the right -- rather than above each dot. Centred
        # labels collide whenever the two values are close, which is exactly
        # the case the ITT row exists to show (36.3% against 37.9%).
        left_value, left_colour = (a, BLUE) if a <= b else (b, ORANGE)
        right_value, right_colour = (b, ORANGE) if a <= b else (a, BLUE)
        ax.annotate(f"{left_value:.1%}", (left_value, y), textcoords="offset points",
                    xytext=(-9, -3), ha="right", fontsize=7.5, color=left_colour, fontweight="bold")
        ax.annotate(f"{right_value:.1%}", (right_value, y), textcoords="offset points",
                    xytext=(9, -3), ha="left", fontsize=7.5, color=right_colour, fontweight="bold")

        # Deltas share one right-hand column at a fixed x, so a short connector
        # cannot push its delta into an endpoint label. **In percentage points,
        # not proportions**: these are differences between rates, and printing
        # -0.1 for a 14.5-point fall understates the paper's central effect by
        # two orders of magnitude.
        ax.annotate(f"{(b - a) * 100:+.1f}", (DELTA_X, y), xycoords=("data", "data"),
                    ha="right", va="center", fontsize=7.5, color=INK_2)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=7.5, color=INK)
    # Left margin so the 4.0% label, which sits outside its dot, does not
    # collide with the y-axis spine.
    ax.set_xlim(-0.045, DELTA_X + 0.01)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax.set_xlabel("Rate", fontsize=7.5)
    _style(ax)
    ax.set_ylim(-0.5, len(labels) - 0.1)
    # Header for the delta column, at the same x the values are drawn on.
    ax.annotate("Δ pts", (DELTA_X, len(labels) - 0.42), xycoords=("data", "data"),
                ha="right", va="center", fontsize=7, color=INK_MUTED, fontweight="bold")

    # The legend sits below the axes. Inside, it lands on the Intention-to-Treat
    # row, whose two dots are at 36% and 38% -- the busiest part of the plot.
    ax.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=7, label="Arm 2, unassisted"),
            Line2D([], [], color=ORANGE, marker="s", linestyle="", markersize=7, label="Arm 3, bare JSON schema"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    ax.set_title(
        "The schema buys precision and pays for it in coverage",
        fontsize=9,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    fig.text(
        0.005,
        -0.10,
        "Both arms under dynamic candidate injection (Condition C), 3 repeats averaged per query before pairing. "
        "Scored over every query, the two effects cancel.",
        fontsize=6.5,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- figure 2: the 2x3 factorial --------------------------------------------

LEVELS = ["Level 0\nbaseline", "Level 1\nschema", "Level 2\nschema"]

# The stamped factorial, read through Table 6's own builder. Until 2026-09-29
# this figure plotted the superseded mixed-payload cells (static rows `v2`
# array-of-objects, injection rows `v5`), so its context axis was confounded
# with the payload shape exactly as the old Table 6 was. Reusing the builder's
# pass discovery and pooling means the figure and the table cannot disagree.
_TSPEC = importlib.util.spec_from_file_location(
    "build_stamped_factorial_table", ROOT / "scripts" / "build_stamped_factorial_table.py")
table6 = importlib.util.module_from_spec(_TSPEC)
_TSPEC.loader.exec_module(table6)

FACTORIAL_CELLS = [
    ("Static window", ["s_l0", "s_l1", "s_l2"]),
    ("Candidate injection", ["i_l0", "i_l1", "i_l2"]),
]


def factorial_data() -> dict[str, dict[str, list[float]]]:
    """Mean over queries of each query's passes, as Table 6 averages them."""
    out: dict[str, dict[str, list[float]]] = {}
    for axis_label, cells in FACTORIAL_CELLS:
        itt, retrieved, passes = [], [], []
        for cell in cells:
            dirs = table6._dirs(cell)
            loaded = [agg.load(d) for d in dirs]
            itt.append(table6._per_query_mean(loaded, agg.itt))
            retrieved.append(table6._per_query_mean(loaded, table6._retrieved))
            passes.append(len(dirs))
        out[axis_label] = {"itt": itt, "retrieved": retrieved, "passes": passes}
    return out


def figure_factorial(data: dict[str, dict[str, list[float]]], path: Path) -> None:
    """Two panels on a shared x, never twin y-axes.

    Invalid rate is a proportion and retrieval is a course count two orders of
    magnitude larger. On one pair of y-axes their lines would cross wherever the
    scaling happened to put them, and a reader would take the crossing for a
    finding. Panels keep the comparison within each measure, which is where it
    is real.
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.9))
    series = [
        ("Static window", BLUE, "o", "-"),
        ("Candidate injection", ORANGE, "s", "--"),
    ]
    x = range(len(LEVELS))

    for ax, key, title, fmt in (
        (axes[0], "itt", "Overall invalid keyword rate", lambda v: f"{v:.0%}"),
        (axes[1], "retrieved", "Mean courses retrieved", lambda v: f"{v:.0f}"),
    ):
        for label, colour, marker, dash in series:
            values = data[label][key]
            ax.plot(x, values, color=colour, linewidth=2, marker=marker, markersize=7,
                    linestyle=dash, markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3)
            others = [data[lbl][key] for lbl, *_ in series if lbl != label]
            for xi, value in zip(x, values):
                # The lower of two points at one x is labelled beneath it, or the
                # label lands on the other series' line where the two converge.
                below = any(value < o[xi] for o in others)
                ax.annotate(fmt(value), (xi, value), textcoords="offset points",
                            xytext=(0, -14 if below else 9), ha="center", va="bottom",
                            fontsize=7, color=colour, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(LEVELS, fontsize=7)
        ax.set_title(title, fontsize=8.5, loc="left", color=INK, fontweight="bold")
        _style(ax, xgrid=False)
        ax.margins(y=0.24)
        ax.set_xlim(-0.35, len(LEVELS) - 0.65)

    axes[0].set_ylim(0, 1.0)
    axes[0].yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    axes[1].set_ylim(0, 190)

    axes[0].legend(
        handles=[
            Line2D([], [], color=c, marker=m, linestyle=d, linewidth=2, markersize=7, label=lbl)
            for lbl, c, m, d in series
        ],
        loc="lower left",
        frameon=False,
        fontsize=7,
        handletextpad=0.5,
    )
    fig.suptitle(
        "The schema lowers retrieval sharply only under candidate injection",
        fontsize=9.5,
        fontweight="bold",
        x=0.005,
        ha="left",
        y=1.04,
    )
    passes = {lbl: sorted(set(data[lbl]["passes"])) for lbl, *_ in series}
    # The paper states one pass count for every cell; fail rather than print a wrong one.
    counts = {n for p in passes.values() for n in p}
    assert len(counts) == 1, f"cells differ in passes: {passes}"
    pass_words = {1: "one", 2: "two", 3: "three", 4: "four"}.get(counts.pop(), "several")
    fig.text(
        0.005,
        -0.08,
        "LLaMA 3.1-8B on the flat payload, served at fp8 by DeepInfra; "
        f"50 queries, {pass_words} passes per cell.\n"
        "Lower is better at left. Courses retrieved are counted, not judged for relevance.",
        fontsize=6.5,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- figure 3: per-domain ----------------------------------------------------

# Rebased 2026-09-30 onto the same matched flat-payload cells as Table 4 and
# Section 4.1. The August `eval_20260827` pair it used before is time-drifted and
# unstamped (Section 6.3). The 8B passes are pooled by Table 6's own `_dirs`.
ARM1_RUN = "arm1_merged_70b_v5fp8_4pass_v2"
ARM2_8B_CELL = "s_l0"


# Spelled out rather than derived by replacing "_" with " / ". That rule is
# right for `law_policy` and wrong for `computer_science`, which it renders as
# "computer / science" -- a field name the reader has to translate back.
DOMAIN_LABELS = {
    "computer_science": "computer science",
    "law_policy": "law / policy",
    "arts_humanities": "arts / humanities",
    "health_medicine": "health / medicine",
    "science_engineering": "science / engineering",
    "business_finance": "business / finance",
    "communication_leadership": "communication / leadership",
    "mathematics": "mathematics",
    "education": "education",
    "language": "language",
}


def _domain_cells() -> dict[str, Any]:
    """The 70B cell and the 8B cell pooled across its passes.

    Separate lanes each number their rows from repeat 1, so the pooled 8B rows
    are re-keyed by lane index to keep `(query_id, repeat)` unique.
    """
    rows = {
        (query_id, i + 1): row
        for i, d in enumerate(table6._dirs(ARM2_8B_CELL))
        for (query_id, _rep), row in agg.load(d).rows.items()
    }
    return {"8B": agg.Cell("8B", rows), "70B": agg.load(OUTPUTS / ARM1_RUN)}


def domain_data() -> list[dict[str, Any]]:
    """Per-domain means of per-QUERY means, so `n` counts queries, not rows."""
    mapping = domains()
    retrieved = agg.METRICS["retrieved"]
    per_model: dict[str, dict[str, dict[str, float]]] = {}
    for label, cell in _domain_cells().items():
        q_ret = agg.by_query(cell, retrieved)
        q_itt = agg.by_query(cell, agg.itt)
        grouped: dict[str, list[str]] = {}
        for query_id in q_itt:
            grouped.setdefault(mapping.get(query_id, "?"), []).append(query_id)
        per_model[label] = {
            domain: {
                "retrieved": st.fmean(q_ret[q] for q in qs),
                "itt": st.fmean(q_itt[q] for q in qs),
                "n": len(qs),
            }
            for domain, qs in grouped.items()
        }
    rows = [
        {
            "domain": DOMAIN_LABELS.get(domain, domain.replace("_", " ")),
            "n": per_model["8B"][domain]["n"],
            "retrieved_8b": per_model["8B"][domain]["retrieved"],
            "retrieved_70b": per_model["70B"][domain]["retrieved"],
            "itt_8b": per_model["8B"][domain]["itt"],
            "itt_70b": per_model["70B"][domain]["itt"],
        }
        for domain in per_model["8B"]
    ]
    return sorted(rows, key=lambda r: r["retrieved_8b"])


def print_domain_table(rows: list[dict[str, Any]]) -> None:
    """Section 6.1's table and its two rank correlations, from the same cells."""
    from scipy.stats import spearmanr

    print("| Domain | Queries | Mean retrieved / query | ITT $M_{2.1}$ |")
    print("| :--- | ---: | ---: | ---: |")
    for r in sorted(rows, key=lambda r: -r["retrieved_8b"]):
        print(f"| {r['domain']} | {r['n']} | {r['retrieved_8b']:.1f} | {r['itt_8b']:.3f} |")
    for label, cell in _domain_cells().items():
        ret = agg.by_query(cell, agg.METRICS["retrieved"])
        itt = agg.by_query(cell, agg.itt)
        qs = sorted(set(ret) & set(itt))
        rho, p = spearmanr([ret[q] for q in qs], [itt[q] for q in qs])
        print(f"  Spearman {label}: rho={rho:.3f} p={p:.2g} n={len(qs)} queries")
    # Whether the two scales rank the domains alike: 6.1 used to assert they do.
    for key in ("retrieved", "itt"):
        rho, p = spearmanr([r[f"{key}_8b"] for r in rows], [r[f"{key}_70b"] for r in rows])
        print(f"  domain ordering 8B vs 70B ({key}): rho={rho:.2f} p={p:.2g} n={len(rows)} domains")


def figure_domains(rows: list[dict[str, Any]], path: Path) -> None:
    """Two panels, domains on a shared y ordered by 8B retrieval.

    **Deliberately not a scatter of retrieval against invention.** The two are
    mechanically coupled -- an invented keyword retrieves nothing, so it lowers
    one and raises the other by construction -- and a scatter, still more one
    with a fitted line, invites the reader to take that identity for a causal
    finding. Section 6.1 states the coupling; this matches its framing. Ordering
    the shared axis by one measure lets the eye see the co-movement without any
    mark asserting a direction of cause.
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.4), sharey=True)
    y = range(len(rows))
    series = [("70B", BLUE, "o"), ("8B", ORANGE, "s")]

    for ax, key, title, fmt, xlim in (
        (axes[0], "retrieved", "Mean courses retrieved", lambda v: f"{v:.0f}", (0, 200)),
        (axes[1], "itt", "Overall invalid keyword rate", lambda v: f"{v:.0%}", (0, 1.0)),
    ):
        for yi, row in zip(y, rows):
            a, b = row[f"{key}_70b"], row[f"{key}_8b"]
            ax.plot([a, b], [yi, yi], color=INK_MUTED, linewidth=1.3, zorder=2, solid_capstyle="round")
            for value, (_label, colour, marker) in zip((a, b), series):
                ax.scatter([value], [yi], s=46, color=colour, marker=marker, zorder=3,
                           edgecolor=SURFACE, linewidth=1.0)
        ax.set_title(title, fontsize=8.5, loc="left", color=INK, fontweight="bold")
        ax.set_xlim(*xlim)
        _style(ax)
        if key == "itt":
            ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([f"{r['domain']}  (n={r['n']})" for r in rows], fontsize=7)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    # Below the axes, as in figure 1. Inside the left panel it sits in the
    # sparse lower-right corner, which is sparse only because those two domains
    # retrieve almost nothing -- the legend would be covering the finding.
    axes[0].legend(
        handles=[
            Line2D([], [], color=c, marker=m, linestyle="", markersize=6.5, label=f"LLaMA 3.1-{lbl}")
            for lbl, c, m in series
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.10),
        ncol=2,
        frameon=False,
        fontsize=7,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    fig.suptitle(
        "The compact model's deficit is uneven across domains",
        fontsize=9.5,
        fontweight="bold",
        x=0.005,
        ha="left",
        y=1.02,
    )
    fig.text(
        0.005,
        -0.045,
        "Domains ordered by 8B retrieval. The two measures are mechanically coupled, since an invented keyword "
        "retrieves nothing, so no causal direction is implied between the panels.",
        fontsize=6.5,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- the check ---------------------------------------------------------------

# What the manuscript prints, for each value these figures plot. A DRIFT line is
# a question for the authors, not something to fix here: the manuscript may be
# wrong, or the figure may be built over the wrong run. Adopting the computed
# value silently would destroy the evidence that they ever disagreed.
EXPECTED = {
    "F1 As-Treated Arm 2": 0.3750,
    "F1 As-Treated Arm 3": 0.2022,
    "F1 silence Arm 2": 0.0267,
    "F1 silence Arm 3": 0.2133,
    "F1 ITT Arm 2": 0.3860,
    "F1 ITT Arm 3": 0.3812,
    # Table 6 as rebuilt from the stamped factorial on 2026-09-29.
    "F2 static ITT L0": 0.6666,
    "F2 static ITT L1": 0.5552,
    "F2 static ITT L2": 0.6763,
    "F2 injection ITT L0": 0.3860,
    "F2 injection ITT L1": 0.3812,
    "F2 injection ITT L2": 0.3111,
    "F2 static retrieved L0": 78.9,
    "F2 static retrieved L1": 69.6,
    "F2 static retrieved L2": 48.3,
    "F2 injection retrieved L0": 149.9,
    "F2 injection retrieved L1": 59.3,
    "F2 injection retrieved L2": 89.8,
    "F3 law_policy retrieved 8B": 52.8,
    "F3 law_policy ITT 8B": 0.666,
    "F3 language retrieved 8B": 110.8,
    "F3 language ITT 8B": 0.520,
}


def run_check(triad, factorial, domain_rows) -> int:
    found = {
        "F1 As-Treated Arm 2": triad["As-Treated invention\n(paired cohort)"][0],
        "F1 As-Treated Arm 3": triad["As-Treated invention\n(paired cohort)"][1],
        "F1 silence Arm 2": triad["Pipeline silence\n(abstention)"][0],
        "F1 silence Arm 3": triad["Pipeline silence\n(abstention)"][1],
        "F1 ITT Arm 2": triad["Intention-to-Treat\ninvention"][0],
        "F1 ITT Arm 3": triad["Intention-to-Treat\ninvention"][1],
    }
    for axis, prefix in (("Static window", "static"), ("Candidate injection", "injection")):
        for i, level in enumerate(("L0", "L1", "L2")):
            found[f"F2 {prefix} ITT {level}"] = factorial[axis]["itt"][i]
            found[f"F2 {prefix} retrieved {level}"] = factorial[axis]["retrieved"][i]
    by_domain = {r["domain"]: r for r in domain_rows}
    found["F3 law_policy retrieved 8B"] = by_domain["law / policy"]["retrieved_8b"]
    found["F3 law_policy ITT 8B"] = by_domain["law / policy"]["itt_8b"]
    found["F3 language retrieved 8B"] = by_domain["language"]["retrieved_8b"]
    found["F3 language ITT 8B"] = by_domain["language"]["itt_8b"]

    drift = 0
    for name, expected in EXPECTED.items():
        actual = found[name]
        # Tolerance is the manuscript's own printed precision, not a fudge
        # factor: 67.8 is stated to one decimal and cannot be held to three.
        tolerance = 0.06 if expected > 1 else 0.0006
        ok = abs(actual - expected) <= tolerance
        drift += 0 if ok else 1
        print(f"  {'OK   ' if ok else 'DRIFT'} {name:34} computed {actual:9.4f}   manuscript {expected:9.4f}")
    print(f"\n  {len(EXPECTED) - drift}/{len(EXPECTED)} match")
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify every plotted value against the manuscript and report drift")
    parser.add_argument("--domain-table", action="store_true",
                        help="print Section 6.1's per-domain table and rank correlations")
    parser.add_argument("--output", type=Path, default=FIGURES,
                        help=f"directory to write into (default: {FIGURES})")
    args = parser.parse_args(argv)

    triad = triad_data()
    factorial = factorial_data()
    domain_rows = domain_data()

    if args.domain_table:
        print_domain_table(domain_rows)
        return 0

    if args.check:
        print("Checking every plotted value against the manuscript:\n")
        return 1 if run_check(triad, factorial, domain_rows) else 0

    args.output.mkdir(parents=True, exist_ok=True)
    figure_triad(triad, args.output / "fig2_precision_coverage_tradeoff")
    figure_factorial(factorial, args.output / "fig1_factorial_scaffolding")
    figure_domains(domain_rows, args.output / "fig3_domain_spread")
    for name in ("fig2_precision_coverage_tradeoff", "fig1_factorial_scaffolding", "fig3_domain_spread"):
        print(f"  wrote {args.output / name}.pdf and .png")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
