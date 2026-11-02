"""X3 / R6 / R7 -- the statistical layer.

Cross-reference:
- src/skyrag/stats.py
- project_knowledge/task_log/backlog_2026-08-20_events_open-decisions.md
  (DEC-3, DEC-11)

TOST is verified against a hand-computed example rather than against the shape
of its own implementation: the two one-sided p-values are recomputed here from
the t distribution directly, so a sign error or a swapped tail fails.

The tests that matter most are the refusals. R6's risk is not that TOST is
computed wrongly -- it is that an equivalence verdict gets quoted from a
single-pass run whose variance term conflates two things, or from a margin
chosen after the results were visible. Neither of those is arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import correlation, mean, stdev

import pytest
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.stats import (
    aggregate_by_arm,
    bootstrap_ci,
    cohens_d,
    cohens_dz,
    tost_paired,
    within_arm_variance,
)

PROV = "declared in methods_section_commitments.md before any arm ran (test)"


# --------------------------------------------------------------------------
# X3 -- variance and intervals
# --------------------------------------------------------------------------


def test_within_arm_variance_is_none_and_says_why_on_single_pass_data():
    """**The load-bearing test of this module.**

    Today every cell has exactly one measurement. The tempting fallback is to
    use between-query spread as "the variance", which produces a plausible
    number that answers a different question. This asserts the function refuses,
    and that the refusal explains itself -- the note is what stops the next
    person reaching for the substitute.
    """
    sd, note = within_arm_variance({"arm1-q01": [4.0], "arm1-q02": [3.0]})
    assert sd is None
    assert "no cell was run more than once" in note
    assert "Do NOT substitute between-query spread" in note
    assert "DEC-3" in note


def test_within_arm_variance_pools_across_cells_once_repeats_exist():
    """Two cells, three repeats each. Pooled SD with 4 degrees of freedom."""
    repeats = {"q01": [4.0, 4.0, 4.0], "q02": [3.0, 4.0, 5.0]}
    sd, note = within_arm_variance(repeats)
    # Cell 1 contributes 0; cell 2 contributes sum of squares = 2, dof = 4.
    assert sd == pytest.approx((2 / 4) ** 0.5, abs=1e-4)
    assert "4 degree(s) of freedom" in note


def test_within_arm_variance_ignores_cells_that_were_run_once():
    """A single-measurement cell contributes no information about spread, and
    counting it as a zero-variance cell would drag the pooled estimate down --
    reporting the instrument as more stable the more incomplete the run was."""
    sd_mixed, _ = within_arm_variance({"q01": [3.0, 5.0], "q02": [4.0]})
    sd_alone, _ = within_arm_variance({"q01": [3.0, 5.0]})
    assert sd_mixed == sd_alone


def test_a_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    """Twenty varied values, not eight.

    The first version of this test used a pool of eight and asserted that two
    different seeds give different endpoints. On a small discrete pool the
    percentile endpoints land on the same values regardless of seed, so the
    assertion failed on correct code -- a flaky test dressed as a real property.
    The pool is now large and varied enough that the seed genuinely moves the
    endpoints, which is what makes the reproducibility half meaningful.
    """
    values = [
        3.0, 4.0, 5.0, 4.0, 3.5, 4.5, 5.0, 2.5, 1.0, 2.0,
        4.75, 3.25, 5.0, 2.0, 4.0, 3.0, 1.5, 4.25, 2.75, 3.75,
    ]
    first = bootstrap_ci(values, seed=34)
    second = bootstrap_ci(values, seed=34)
    different = bootstrap_ci(values, seed=99)

    assert first == second, "a reported interval must reproduce"
    assert first != different, "the seed must actually drive the resampling"
    assert first[0] < mean(values) < first[1]


def test_an_interval_around_one_observation_is_absent_not_narrow():
    """(x, x) would render in a table as perfect precision."""
    assert bootstrap_ci([4.0]) is None
    assert bootstrap_ci([]) is None


def test_aggregate_by_arm_never_reports_a_bare_mean():
    agg = aggregate_by_arm({"1": [4.0, 4.5, 3.5, 4.0], "2": [3.0, 3.5, 2.5, 3.0]})
    for arm in ("1", "2"):
        a = agg[arm]
        assert a.sd is not None
        assert a.ci_low is not None and a.ci_high is not None
        # And the within-arm term is absent-with-a-reason, not silently missing.
        assert a.within_arm_sd is None
        assert a.within_arm_note


# --------------------------------------------------------------------------
# R7 -- effect size
# --------------------------------------------------------------------------


def test_cohens_dz_matches_its_definition():
    a = [5.0, 4.0, 4.0, 3.0, 5.0]
    b = [3.0, 3.0, 2.0, 3.0, 4.0]
    diff = [x - y for x, y in zip(a, b)]
    assert cohens_dz(a, b) == pytest.approx(mean(diff) / stdev(diff), abs=1e-4)


def test_dz_and_d_cross_over_at_a_pairwise_correlation_of_one_half():
    """The subscript is not decoration, and which one is larger is not fixed.

    With comparable within-arm spreads, sd(diff)^2 = 2s^2(1 - r), so
    d_z = d / sqrt(2(1 - r)) and d_z exceeds d exactly when r > 0.5. Both
    directions are asserted because the tempting summary -- "d_z is the larger
    one" -- is false, and a test written from that summary passes on half the
    inputs and hides the other half. This is not a hypothetical: the first
    version of this test asserted exactly that and failed on the low-correlation
    case below.
    """
    low_a, low_b = [5.0, 4.0, 4.0, 3.0, 5.0], [3.0, 3.0, 2.0, 3.0, 4.0]
    high_a, high_b = [5.0, 4.0, 3.0, 2.0, 1.0], [4.1, 3.0, 2.1, 1.0, 0.1]

    assert correlation(low_a, low_b) < 0.5
    assert abs(cohens_dz(low_a, low_b)) < abs(cohens_d(low_a, low_b))

    assert correlation(high_a, high_b) > 0.5
    assert abs(cohens_dz(high_a, high_b)) > abs(cohens_d(high_a, high_b))


def test_effect_sizes_are_none_on_degenerate_input():
    assert cohens_dz([1.0, 2.0], [2.0, 3.0]) is None   # zero spread in diffs
    assert cohens_dz([1.0], [2.0]) is None             # too few
    assert cohens_dz([1.0, 2.0], [1.0]) is None        # unpaired
    assert cohens_d([1.0, 1.0], [1.0, 1.0]) is None    # no pooled variance


# --------------------------------------------------------------------------
# R6 -- TOST
# --------------------------------------------------------------------------


def test_tost_p_values_match_a_direct_recomputation():
    """Both tails recomputed from the t distribution independently of the
    implementation, so a sign error or a swapped tail fails here."""
    a = [4.0, 4.2, 3.8, 4.1, 3.9, 4.0, 4.3, 3.7, 4.1, 4.0]
    b = [4.1, 4.0, 3.9, 4.0, 4.0, 3.9, 4.2, 3.8, 4.0, 4.1]
    margin = 0.5

    res = tost_paired(a, b, margin=margin, margin_provenance=PROV, repeats_per_cell=3)

    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    se = stdev(diff) / (n**0.5)
    expected_lower = float(scipy_stats.t.sf((mean(diff) + margin) / se, n - 1))
    expected_upper = float(scipy_stats.t.cdf((mean(diff) - margin) / se, n - 1))

    assert res.p_lower == pytest.approx(expected_lower, abs=1e-6)
    assert res.p_upper == pytest.approx(expected_upper, abs=1e-6)
    assert res.p_value == pytest.approx(max(expected_lower, expected_upper), abs=1e-6)


def test_tost_declares_equivalence_for_a_genuinely_tiny_difference():
    a = [4.0, 4.1, 3.9, 4.0, 4.05, 3.95, 4.0, 4.1, 3.9, 4.0] * 3
    b = [4.0, 4.0, 4.0, 4.05, 4.0, 4.0, 3.95, 4.05, 3.95, 4.0] * 3
    res = tost_paired(a, b, margin=0.5, margin_provenance=PROV, repeats_per_cell=3)
    assert res.equivalent is True
    assert res.caveats == ()


def test_a_large_difference_is_not_declared_equivalent():
    a = [5.0, 5.0, 4.8, 5.0, 4.9, 5.0, 4.7, 5.0, 4.9, 5.0] * 3
    b = [2.0, 2.1, 2.0, 1.9, 2.0, 2.2, 2.0, 1.8, 2.1, 2.0] * 3
    res = tost_paired(a, b, margin=0.5, margin_provenance=PROV, repeats_per_cell=3)
    assert res.equivalent is False


def test_non_significant_difference_is_not_equivalence():
    """**The failure mode R6 exists to prevent**, asserted directly.

    These two arms differ by a lot on noisy data: a paired t-test does not
    reject, and the tempting reading is "no significant difference, so they
    match". TOST says the opposite -- the data cannot place the difference
    inside the margin -- and that disagreement is the entire reason the test
    was specified.
    """
    a = [5.0, 1.0, 5.0, 1.0, 5.0, 1.0, 4.0, 2.0, 5.0, 1.0]
    b = [1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 2.0, 4.0, 1.0, 5.0]

    _, ttest_p = scipy_stats.ttest_rel(a, b)
    assert ttest_p > 0.05, "precondition: the difference is not significant"

    res = tost_paired(a, b, margin=0.5, margin_provenance=PROV, repeats_per_cell=3)
    assert res.equivalent is False


def test_a_margin_without_provenance_is_refused():
    """DEC-11's risk is timing, and timing is not recoverable from the number.
    The module cannot verify the claim; it can refuse to let the margin travel
    without one."""
    with pytest.raises(ValueError, match="margin_provenance is required"):
        tost_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.1], margin=0.5, margin_provenance="")
    with pytest.raises(ValueError, match="margin_provenance is required"):
        tost_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.1], margin=0.5, margin_provenance="   ")


def test_there_is_no_default_margin():
    with pytest.raises(TypeError):
        tost_paired([1.0, 2.0], [1.0, 2.1], margin_provenance=PROV)  # type: ignore[call-arg]


def test_a_zero_or_negative_margin_is_refused():
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="must be positive"):
            tost_paired(
                [1.0, 2.0, 3.0], [1.0, 2.0, 3.1], margin=bad, margin_provenance=PROV
            )


def test_single_pass_data_attaches_the_variance_caveat_to_the_verdict():
    """DEC-3 and DEC-11 are not independent, and this is where that bites.

    With one draw per cell the paired differences carry query heterogeneity and
    run-to-run noise inseparably. The test still runs -- it is the right test --
    but `equivalent` must not be quotable without the caveat travelling beside
    it, so the caveat is on the result object rather than in a docstring.
    """
    a = [4.0, 4.1, 3.9, 4.0, 4.05, 3.95, 4.0, 4.1, 3.9, 4.0] * 3
    b = [4.0, 4.0, 4.0, 4.05, 4.0, 4.0, 3.95, 4.05, 3.95, 4.0] * 3
    res = tost_paired(a, b, margin=0.5, margin_provenance=PROV, repeats_per_cell=1)
    assert res.equivalent is True
    assert any("repeats_per_cell=1" in c for c in res.caveats)
    assert any("DEC-3" in c for c in res.caveats)


def test_small_samples_are_flagged_as_underpowered():
    a = [4.0, 4.1, 3.9, 4.0, 4.05]
    b = [4.0, 4.0, 4.0, 4.05, 4.0]
    res = tost_paired(a, b, margin=0.5, margin_provenance=PROV, repeats_per_cell=3)
    assert any("underpowered" in c for c in res.caveats)


def test_zero_variance_differences_are_a_degenerate_input_not_equivalence():
    """Two identical arms give no standard error. Returning `equivalent=True`
    would be the most flattering possible reading of a broken input."""
    with pytest.raises(ValueError, match="degenerate input, not equivalence"):
        tost_paired(
            [4.0, 4.0, 4.0], [4.0, 4.0, 4.0], margin=0.5, margin_provenance=PROV
        )
