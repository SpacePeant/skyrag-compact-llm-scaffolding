"""The eval CLI's per-cell results summary.

It used to be one wide pandas row that wrapped in any terminal and glued adjacent
values together (`0.45710.1407` was retrieval_score_mean and _min). What these pin
is that every column the old table carried is still printed, that each value is
under a header naming its cell and status, and that a missing value reads `n/a`.
"""

from __future__ import annotations

import math

import pandas as pd

from skyrag.cli import (
    RESULT_SUMMARY_GROUPS,
    RESULT_SUMMARY_MAX_COLUMNS,
    format_results_summary,
)

COLUMNS = [c for _, cols in RESULT_SUMMARY_GROUPS for c in cols]


def _row(**overrides):
    row = {c: 0 for c in COLUMNS}
    row.update(
        model="meta-llama/llama-3.1-8b-instruct", pipeline="skyrag", status="ok",
        arm="2", query_id="q01", repeat=1,
        elapsed_seconds=44.97258, generation_seconds=10.103803,
        grounding_accuracy=1.0, citation_coverage=0.1087,
        retrieval_score_mean=0.4571, retrieval_score_min=0.1407,
        selection_rank_mean=8.9, keywords_emitted=55.0,
    )
    row.update(overrides)
    return row


def test_every_old_column_is_still_printed():
    out = format_results_summary(pd.DataFrame([_row()]))
    for column in COLUMNS:
        assert column in out
    # The 25 columns of the old wide table, minus model/pipeline/status,
    # which moved into the cell header.
    assert len(COLUMNS) == 22


def test_adjacent_values_are_on_separate_lines():
    out = format_results_summary(pd.DataFrame([_row()]))
    assert "0.45710.1407" not in out
    lines = {line.split()[0]: line.split()[-1] for line in out.splitlines()
             if line.startswith("    ")}
    assert lines["retrieval_score_mean"] == "0.4571"
    assert lines["retrieval_score_min"] == "0.1407"


def test_formats_by_column_kind():
    out = format_results_summary(pd.DataFrame([_row()]))
    lines = {line.split()[0]: line.split()[-1] for line in out.splitlines()
             if line.startswith("    ")}
    assert lines["grounding_accuracy"] == "1.0000"  # a score, not a count
    assert lines["keywords_emitted"] == "55"        # a count stored as float
    assert lines["elapsed_seconds"] == "44.97"
    assert lines["selection_rank_mean"] == "8.9"


def test_header_names_cell_and_status():
    out = format_results_summary(pd.DataFrame([
        _row(),
        _row(arm="1", model="meta-llama/llama-3.1-70b-instruct",
             status="cell_failed", grounding_accuracy=math.nan),
    ]))
    head = "\n".join(out.splitlines()[:3])
    assert "r1 q01 arm 2" in head and "r1 q01 arm 1" in head
    assert "llama-3.1-70b-instruct" in head
    assert "skyrag -> cell_failed" in head
    grounding = next(l for l in out.splitlines() if l.split()[0] == "grounding_accuracy")
    assert grounding.split()[1:] == ["1.0000", "n/a"]


def test_large_sweep_points_to_csv_instead_of_a_table():
    rows = [_row(query_id=f"q{i:02d}") for i in range(RESULT_SUMMARY_MAX_COLUMNS + 1)]
    out = format_results_summary(pd.DataFrame(rows))
    assert "CSV" in out and "grounding_accuracy" not in out
