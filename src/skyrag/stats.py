"""X3 / R6 / R7 -- the statistical layer: variance, intervals, effect size, TOST.

`summarize_metrics` computes means and sums. Nothing in it groups by arm,
computes a variance, or produces an interval, so every claim in the results
table has been unsupportable for a structural reason rather than an arithmetic
one. This module is the layer those claims need.

**The code here is unblocked; its inputs are not.** Repeats are deferred (D9)
and their count is DEC-3, so the within-arm variance functions below will return
`None` -- loudly, with a reason -- on today's single-pass data. That is the
correct behaviour and it is why they exist now: a variance term that silently
falls back to "between-query spread" would let a single-pass sweep produce a
confident-looking interval, and the instrument is known to move by up to 77% on
recommendation-set overlap between two runs of the *same* arm.

---

## R6 / DEC-11 -- why `margin` has no default

The research question says the assisted 8B **matches** the 70B. A
non-significant difference is not evidence of equivalence; it is the single most
common way a "our small model matches the big one" claim fails review. TOST is
the test that can actually support the claim -- but only if its equivalence
boundary is fixed **before anyone sees a result**. Choosing it afterwards makes
the claim unfalsifiable, and no amount of correct arithmetic repairs that.

So `tost` requires both the margin and a `margin_provenance` string saying where
and when it was declared. The string is not decoration: DEC-11's risk is
entirely about *timing*, timing is not recoverable from the number itself, and a
margin whose origin is undocumented is indistinguishable from one reverse-fitted
to the data. The module cannot verify the claim -- but it can refuse to let the
margin travel without one, so the question is asked at the point of use.

## R7 -- effect size

With n = 50 queries a trivial difference can be significant. Cohen's d is what
says whether it matters, and most venues expect it beside a p-value.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, stdev

from scipy import stats as _scipy_stats

__all__ = [
    "ArmAggregate",
    "TOSTResult",
    "aggregate_by_arm",
    "bootstrap_ci",
    "cohens_d",
    "cohens_dz",
    "tost_paired",
    "O1_EQUIVALENCE_MARGIN",
    "O1_MARGIN_PROVENANCE",
    "within_arm_variance",
]


# --------------------------------------------------------------------------
# X3 -- grouping, variance, intervals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmAggregate:
    """One arm's central tendency, spread, and interval.

    `within_arm_sd` is separate from `sd` on purpose and the distinction is the
    whole point of this type. `sd` is the spread of the per-query values, which
    mostly measures *how much queries differ from each other*.
    `within_arm_sd` is the spread of repeated runs of the same cell, which is
    the instrument's own noise -- and it is the term an equivalence claim
    actually consumes. Reporting the first where the second is meant overstates
    precision by however much the queries happen to be homogeneous.
    """

    arm: str
    n: int
    mean: float
    sd: float | None
    ci_low: float | None
    ci_high: float | None
    within_arm_sd: float | None
    within_arm_note: str


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 34,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval. Seeded, so a reported interval reproduces.

    Non-parametric by choice: O1 scores are bounded 1-5 ordinal-ish values whose
    normality is not an assumption this data earns -- the same reasoning that
    put Wilcoxon rather than a t-test in `compare_arms.py`.

    `None` below two values: an interval around a single observation is not a
    narrow interval, it is an absent one, and returning `(x, x)` would render as
    perfect precision in a table.
    """
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    pool = list(values)
    n = len(pool)
    means = []
    for _ in range(resamples):
        means.append(mean(rng.choices(pool, k=n)))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    lo = means[int(tail * resamples)]
    hi = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return round(lo, 4), round(hi, 4)


def within_arm_variance(
    repeats_by_cell: dict[str, Sequence[float]],
) -> tuple[float | None, str]:
    """The instrument's own noise: spread across repeated runs of the same cell.

    Returns `(sd, note)`. The note is never empty and is meant to be printed:
    when this returns `None` the reason matters more than the value, because the
    tempting next step -- substituting between-query spread -- produces a number
    that looks like the right one.

    `repeats_by_cell` maps a cell key (arm+query) to that cell's repeated
    measurements. Cells with a single measurement contribute nothing; that is
    the D9/DEC-3 state today, and it is why this returns `None` rather than
    pooling something else.
    """
    usable = {k: list(v) for k, v in repeats_by_cell.items() if len(v) >= 2}
    if not usable:
        return None, (
            "no cell was run more than once, so the instrument's own variance is "
            "unmeasured. Repeats are deferred (D9) and their count is DEC-3. "
            "Do NOT substitute between-query spread: it measures how much "
            "queries differ, not how much a rerun of the same query differs, "
            "and this instrument has returned different recommendation sets on "
            "3 of 8 within-process pairs at pk=7."
        )
    # Pooled within-cell standard deviation.
    numerator = sum(
        sum((x - mean(vals)) ** 2 for x in vals) for vals in usable.values()
    )
    dof = sum(len(vals) - 1 for vals in usable.values())
    if dof <= 0:
        return None, "no degrees of freedom for a within-cell estimate"
    sd = (numerator / dof) ** 0.5
    return round(sd, 4), (
        f"pooled across {len(usable)} cell(s) with repeats, {dof} degree(s) of freedom"
    )


def aggregate_by_arm(
    values_by_arm: dict[str, Sequence[float]],
    *,
    repeats_by_cell: dict[str, dict[str, Sequence[float]]] | None = None,
    confidence: float = 0.95,
    seed: int = 34,
) -> dict[str, ArmAggregate]:
    """X3. Group by arm; report spread and interval, never a bare mean.

    `repeats_by_cell` is per-arm: `{arm: {cell_key: [measurements]}}`. Omitted
    or empty, `within_arm_sd` comes back `None` with the reason attached rather
    than silently absent.
    """
    out: dict[str, ArmAggregate] = {}
    for arm, values in values_by_arm.items():
        vals = list(values)
        if not vals:
            continue
        ci = bootstrap_ci(vals, confidence=confidence, seed=seed)
        wsd, note = within_arm_variance((repeats_by_cell or {}).get(arm, {}))
        out[arm] = ArmAggregate(
            arm=arm,
            n=len(vals),
            mean=round(mean(vals), 4),
            sd=round(stdev(vals), 4) if len(vals) > 1 else None,
            ci_low=ci[0] if ci else None,
            ci_high=ci[1] if ci else None,
            within_arm_sd=wsd,
            within_arm_note=note,
        )
    return out


# --------------------------------------------------------------------------
# R7 -- effect size
# --------------------------------------------------------------------------


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Paired effect size: mean difference over the SD of the differences.

    This is the form already in `compare_arms.py`, restated here so the
    statistical layer has one home. `d_z` and independent-samples `d` are
    different quantities and the subscript is not decoration. Which is larger
    depends on how strongly the pairs correlate, and the crossover is exact:
    with comparable within-arm spreads, sd(diff)^2 = 2s^2(1 - r), so

        d_z = d / sqrt(2(1 - r))    and therefore    d_z > d  iff  r > 0.5.

    Highly correlated arms -- which two arms answering the same 50 queries
    generally are -- give d_z well above d; weakly correlated ones give d_z
    *below* it. So reporting one under the other's name does not merely
    overstate the effect, it misstates it in a direction that depends on data
    nobody re-derives at reading time. Report the one you computed, named.

    `None` when the differences have no spread: every pair differing by exactly
    the same amount gives a division by zero, which is a degenerate input rather
    than an infinite effect.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    diff = [x - y for x, y in zip(a, b)]
    sd = stdev(diff)
    return round(mean(diff) / sd, 4) if sd else None


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Independent-samples effect size, pooled SD. Use when arms are unpaired.

    Kept beside `cohens_dz` so the choice is explicit at the call site. The arms
    in this project pair on `query_id`, so `cohens_dz` is nearly always the right
    one; this exists for the cells that failed on one arm and not the other,
    where pairing drops the row entirely.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    sa, sb = stdev(a), stdev(b)
    pooled_var = ((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2)
    if pooled_var <= 0:
        return None
    return round((mean(a) - mean(b)) / pooled_var**0.5, 4)


# --------------------------------------------------------------------------
# R6 -- TOST
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TOSTResult:
    equivalent: bool
    p_value: float             # max of the two one-sided p-values
    p_lower: float
    p_upper: float
    mean_difference: float
    margin: float
    alpha: float
    n_pairs: int
    margin_provenance: str
    #: Non-empty whenever something about the inputs weakens the verdict. Read
    #: it before quoting `equivalent`.
    caveats: tuple[str, ...] = ()


# --- DEC-11 / commitment C12 -------------------------------------------------
# The equivalence margin, settled 2026-08-27 **before Arm 3 produced a single
# row**. That timestamp is the load-bearing part: a margin fixed at any
# value before the data is defensible; the same value chosen after seeing the gap
# is not, and no correct arithmetic afterwards repairs it.
#
# Units: mean **O1 rubric points on the anchored 1-5 scale**, applied to the
# BETWEEN-ARM DIFFERENCE (DEC-12 -- absolute scores are not reportable).
#
# These are constants, not defaults. `tost_paired` still requires both arguments
# explicitly: a caller studying some other quantity must state its own margin
# rather than inherit O1's by omission, which is the mistake a default invites.
O1_EQUIVALENCE_MARGIN: float = 0.25

O1_MARGIN_PROVENANCE: str = (
    "DEC-11 / commitment C12, fixed 2026-08-27 before any Arm 3 row existed. "
    "Delta = 0.25 mean O1 rubric points on the anchored 1-5 scale, applied to the "
    "between-arm difference. Grounds: each rubric integer carries a described "
    "behaviour (pedagogical_coherence runs from 'unrelated courses with no "
    "through-line' to 'each course builds on the last toward a stated goal'), so "
    "one anchor step is the smallest difference the rubric is built to describe; "
    "0.25 is a quarter of that, and two paths separated by less share an anchor "
    "description. NOT '5% of the scale' -- a scale-relative percentage is a "
    "convention, not a pedagogical judgment. The tight end of the defensible "
    "range was taken deliberately: fp8 measures effects at roughly half their "
    "bf16 size (commitment 3.3e), which biases toward declaring equivalence, and "
    "a loose margin on an attenuating instrument is two thumbs on one side."
)


# --- M2.1 equivalence margin, Arms 1/2/3 thesis leg --------------------------
# A SECOND margin, for a different quantity. O1's above is rubric points; this is
# the invalid keyword rate, a proportion. The comment above is explicit that a
# caller studying another quantity must state its own rather than inherit O1's,
# and this is that statement.
#
# Units: **proportion of emitted keywords that are absent from the corpus**,
# applied to the paired between-condition difference under ITT.
M2_1_EQUIVALENCE_MARGIN: float = 0.05

M2_1_MARGIN_PROVENANCE: str = (
    "Fixed 2026-09-04, before any repeat run and before any TOST on M2.1 was "
    "computed. Delta = 0.05 on the invalid keyword rate, applied to the paired "
    "between-condition difference under intention-to-treat. Grounds: one-eighth "
    "of the measured Arm 1 - Arm 2 gap. Llama 3.1-70B scores 0.2851 and Llama "
    "3.1-8B 0.6902 under ITT (n=50 each), a gap of 0.4050; an intervention "
    "moving invention by less than 0.05 has closed under 12.3% of the distance "
    "to the larger model, which cannot justify adoption -- still less given the "
    "same intervention costs 40-60% of retrieval in both families tested. "
    "HONESTY NOTE, weaker than O1's provenance and stated rather than hidden: "
    "this margin was fixed AFTER conditions A and B had run, so the observed "
    "A-vs-B difference (-0.0270) was known when it was chosen. Two things "
    "defend it. It is derived from the Arm 1 - Arm 2 gap, which is independent "
    "of the A-vs-B contrast it will be applied to. And it is a margin the "
    "current data FAILS -- the paired 95% CI is [-0.1204, +0.0673], outside "
    "both +/-0.05 and +/-0.10 -- so it cannot have been chosen to manufacture "
    "an equivalence verdict. +/-0.15 would have delivered one and was rejected "
    "for that reason."
)


def tost_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    margin: float,
    margin_provenance: str,
    alpha: float = 0.05,
    repeats_per_cell: int = 1,
) -> TOSTResult:
    """Two one-sided tests for equivalence, paired. R6.

    Equivalence is declared when the mean difference is significantly inside
    `(-margin, +margin)` -- both one-sided tests reject -- which is a positive
    claim, unlike a non-significant difference.

    **`margin` and `margin_provenance` are required and have no defaults
    (DEC-11).** The margin must come from what would matter pedagogically -- how
    much worse an O1 score may be before the path is meaningfully worse -- not
    from what the data later makes achievable. This function cannot check when
    the margin was chosen; requiring its provenance makes the question
    unavoidable at the point of use, and puts the answer in the artifact.

    `repeats_per_cell` is not used in the arithmetic. It is required so that a
    single-pass run cannot produce a bare `equivalent=True`: with one draw per
    cell the paired differences carry query heterogeneity *and* run-to-run
    noise, inseparably, and this instrument is known to exhibit the latter. The
    caveat is attached to the result rather than raised, because the test is
    still the right one to run -- its verdict just is not yet quotable.
    """
    if not margin_provenance.strip():
        raise ValueError(
            "margin_provenance is required (DEC-11). State where and when the "
            "equivalence boundary was declared. A margin chosen after seeing "
            "results makes the equivalence claim unfalsifiable, and the number "
            "alone cannot show which happened."
        )
    if margin <= 0:
        raise ValueError(
            f"margin={margin}: an equivalence boundary must be positive. Zero "
            "asks whether two arms are exactly identical, which nothing can show."
        )
    if len(a) != len(b):
        raise ValueError(f"unpaired inputs: {len(a)} vs {len(b)}")
    if len(a) < 2:
        raise ValueError("TOST needs at least two pairs")

    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    md = mean(diff)
    sd = stdev(diff)
    if sd == 0:
        raise ValueError(
            "the paired differences have zero variance; TOST has no standard "
            "error to work with. This is a degenerate input, not equivalence."
        )
    se = sd / (n**0.5)
    df = n - 1

    # H01: mu <= -margin  (reject -> mu is above the lower bound)
    t_lower = (md + margin) / se
    p_lower = float(_scipy_stats.t.sf(t_lower, df))
    # H02: mu >= +margin  (reject -> mu is below the upper bound)
    t_upper = (md - margin) / se
    p_upper = float(_scipy_stats.t.cdf(t_upper, df))

    p = max(p_lower, p_upper)

    caveats: list[str] = []
    if repeats_per_cell < 2:
        caveats.append(
            "repeats_per_cell=1: the paired differences conflate query "
            "heterogeneity with run-to-run noise, and this instrument has "
            "returned different recommendation sets across repeated runs of the "
            "same cell. The verdict is not quotable until DEC-3 sets a repeat "
            "count and L5 runs it."
        )
    if n < 20:
        caveats.append(
            f"n={n} pairs: TOST is underpowered at this size, so a "
            "'not equivalent' verdict here is at least as likely to be missing "
            "power as a real difference."
        )

    return TOSTResult(
        equivalent=p < alpha,
        p_value=round(p, 6),
        p_lower=round(p_lower, 6),
        p_upper=round(p_upper, 6),
        mean_difference=round(md, 4),
        margin=margin,
        alpha=alpha,
        n_pairs=n,
        margin_provenance=margin_provenance,
        caveats=tuple(caveats),
    )
