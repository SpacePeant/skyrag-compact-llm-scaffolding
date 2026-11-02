"""The metrics `aggregate_arms.py` must be able to compute, and their denominators.

Section 3.5 of Paper #1 names this script as the authority for every reported
statistic. Until 2026-09-22 its `METRICS` table held five entries -- invention
under two mute conventions, retrieved, emitted, cited -- so M2.2, M2.3 and
grounding were reachable only through `COUNTS`, which prints a bare total with
no interval, no `n` and no paired test. F1's near-miss result (1.54% vs 1.41%,
p = 0.99, n = 43) therefore could not be reproduced by the tool the manuscript
points a reviewer at, and no grounding figure could be either.

What these pin is not the values -- those belong to the artifacts -- but the
three decisions that make the values readable:

* every rate declines to invent a denominator it does not have (`None`, never
  0.0, which reads as the opposite finding),
* M2.3's conditional form survives an artifact whose `keywords_real` column is
  wrong, and says so rather than silently shrinking its `n`,
* the mute conventions differ per metric on purpose: 1.0 for invention, `None`
  for transcription.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_arms", Path(__file__).resolve().parents[1] / "scripts" / "aggregate_arms.py"
)
agg = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_arms"] = agg
_SPEC.loader.exec_module(agg)


def _row(**overrides):
    """A row that is internally consistent before any override is applied."""
    row = dict(
        query_id="q01",
        repeat=1,
        keywords_emitted=50,
        keywords_kept=40,
        hallucinated_keywords=6,
        near_miss_keywords=4,
        out_of_window_keywords=5,
        removed_by_cleaning_keywords=0,
        keywords_real=45,
        retrieved_count=100,
        cited_count=10,
        grounding_accuracy=0.9,
        citation_coverage=0.1,
    )
    row.update(overrides)
    return row


# --- the metrics exist at all -------------------------------------------------


def test_the_metric_table_covers_every_family_the_manuscript_reports():
    """M2.2, both M2.3 forms and grounding are pairable, not just countable.

    A name in `COUNTS` is summed and printed; a name in `METRICS` gets a
    bootstrap interval, an `n`, and enters `compare`'s Wilcoxon. Only the
    second is a statistic a paper can cite.
    """
    for name in (
        "near_miss_rate",
        "out_of_window_rate",
        "out_of_window_conditional",
        "grounding",
        "citation_coverage",
    ):
        assert name in agg.METRICS, f"{name} is not a pairable metric"


def test_the_two_m23_forms_are_flagged_as_a_pair():
    """The specified form may not be quoted arm-against-arm alone (R24a).

    It reverses capability 1's conclusion: on the 2026-08-20 sweep it read
    0.126 (8B) vs 0.144 (70B) -- the smaller model apparently better at
    long-context recall -- while the conditional form read 0.381 vs 0.204.
    """
    assert "out_of_window_rate" in agg.PAIRED_READING
    assert "grounding" in agg.PAIRED_READING


# --- denominators are never invented -----------------------------------------


def test_every_rate_is_none_on_a_mute_rather_than_zero():
    """A mute has no transcription rate, and 0.0 asserts one that did not occur.

    `itt` is the deliberate exception: silence is invention's worst case, so it
    scores 1.0. Nothing else in the table may follow it.
    """
    mute = _row(keywords_emitted=0, keywords_kept=0, keywords_real=0)
    assert agg.near_miss_rate(mute) is None
    assert agg.out_of_window_rate(mute) is None
    assert agg.out_of_window_conditional(mute) is None
    assert agg.itt(mute) == 1.0
    assert agg.both_spoke(mute) is None


def test_grounding_passes_through_a_null_instead_of_scoring_it_perfect():
    """A run that cited nothing has no grounding to measure.

    The export already encodes this as `None`; the metric's only job is to
    decline to undo it. Scoring it 1.0 would reward producing nothing and
    inflate the mean by exactly the rows that failed.
    """
    assert agg.grounding(_row(grounding_accuracy=None)) is None
    assert agg.grounding(_row(grounding_accuracy=0.75)) == 0.75


def test_citation_coverage_is_none_when_nothing_was_retrieved():
    """A ratio with a zero denominator is absent, not zero."""
    assert agg.citation_coverage(_row(citation_coverage=None, retrieved_count=0)) is None


def test_citation_coverage_is_derivable_on_artifacts_that_predate_the_column():
    """Older runs carry `cited_count` and `retrieved_count` but no ratio."""
    row = _row(citation_coverage=None, retrieved_count=80, cited_count=20)
    assert agg.citation_coverage(row) == 0.25


# --- the conditional denominator survives a bad artifact ---------------------


def test_the_conditional_form_trusts_a_usable_stamp():
    """A stamped `keywords_real` is authoritative when it is possible."""
    assert agg.out_of_window_conditional(_row()) == 5 / 45


def test_the_conditional_form_repairs_a_denominator_below_kept():
    """`real < kept` is arithmetically impossible: the real bucket contains kept.

    `eval_20260827_arm1-2_llama70b-8b_100q_topup` -- the Arm 1 vs Arm 2
    baseline F1 rests on -- stamps `keywords_real = 0` on 82 of its 100 rows
    while those rows report up to 43 kept keywords. Trusting the stamp drops
    the paired comparison from n=42 to n=2, which is a metric silently
    measuring nothing.
    """
    broken = _row(keywords_real=0)
    assert agg.out_of_window_conditional(broken) == 5 / 45


def test_the_conditional_form_reconstructs_when_the_column_is_absent():
    """The column arrived 2026-08-21; runs before it have no key at all."""
    old = _row()
    del old["keywords_real"]
    assert agg.out_of_window_conditional(old) == 5 / 45


def test_repairs_are_counted_so_a_rebuilt_denominator_cannot_be_quoted_silently():
    """A rate over a repaired denominator is a different provenance claim.

    The count is what distinguishes it in the output, so it is reported rather
    than logged.
    """
    rows = [_row(), _row(keywords_real=0), _row(keywords_real=None)]
    assert agg.conditional_repairs(rows) == 2
    assert agg.conditional_repairs([_row(), _row()]) == 0


def test_load_filters_by_candidate_injection(tmp_path):
    import json
    data = [
        {"query_id": "q1", "arm": "2", "candidate_injection": True, "repeat": 1, "keywords_emitted": 10},
        {"query_id": "q2", "arm": "2", "candidate_injection": False, "repeat": 1, "keywords_emitted": 10},
    ]
    (tmp_path / "benchmark_eval.json").write_text(json.dumps(data), encoding="utf-8")
    cell_all = agg.load(tmp_path, arm="2")
    assert cell_all.n == 2
    cell_inj = agg.load(tmp_path, arm="2", injected=True)
    assert cell_inj.n == 1
    assert "q1" in [k[0] for k in cell_inj.rows]
    cell_noinj = agg.load(tmp_path, arm="2", injected=False)
    assert cell_noinj.n == 1
    assert "q2" in [k[0] for k in cell_noinj.rows]



def test_an_unstamped_run_matches_neither_injection_filter(tmp_path):
    """Absent `candidate_injection` is unknown, never False.

    The column arrived 2026-09-08. `bool(None)` is `False`, so a naive filter
    answers "not injected" for every row of every run that predates it --
    `eval_20260827_arm1-2_llama70b-8b_100q_topup` omits the key on all 100
    rows and would come back labelled `[noinject]`, asserting a treatment
    status the artifact never recorded. That is the error `serving_precision`
    exports `""` to avoid.
    """
    import json

    import pytest

    data = [
        {"query_id": "q1", "arm": "2", "repeat": 1, "keywords_emitted": 10},
        {"query_id": "q2", "arm": "2", "repeat": 1, "keywords_emitted": 10},
    ]
    (tmp_path / "benchmark_eval.json").write_text(json.dumps(data), encoding="utf-8")

    assert agg.load(tmp_path, arm="2").n == 2
    for wanted in (True, False):
        with pytest.raises(SystemExit) as excinfo:
            agg.load(tmp_path, arm="2", injected=wanted)
        # The message has to say WHY, or the next reader assumes the run is
        # simply empty and reaches for a different artifact.
        assert "does not stamp" in str(excinfo.value)


def test_an_explicit_false_still_filters_normally(tmp_path):
    """A stamped False is a measurement and must keep working."""
    import json

    data = [
        {"query_id": "q1", "arm": "2", "candidate_injection": True, "repeat": 1},
        {"query_id": "q2", "arm": "2", "candidate_injection": False, "repeat": 1},
    ]
    (tmp_path / "benchmark_eval.json").write_text(json.dumps(data), encoding="utf-8")

    assert [k[0] for k in agg.load(tmp_path, arm="2", injected=False).rows] == ["q2"]
    assert [k[0] for k in agg.load(tmp_path, arm="2", injected=True).rows] == ["q1"]
