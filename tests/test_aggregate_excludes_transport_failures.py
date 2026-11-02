"""A transport failure is not silence, and must not be scored as a mute.

C20, found 2026-09-29. All three E5 r3 static lanes died on HTTP 429 upstream
congestion at DeepInfra. The harness handled it correctly: it stamped each
abandoned cell `status=cell_failed`, put the error in `failure_detail`, and left
`grounding_accuracy` EMPTY rather than 0.0 -- absent is not default.

`aggregate_arms` then undid all of that. Its `keep()` filters on `query_id`,
`arm`, `keyword_model` and `candidate_injection`, and never looks at `status`.
A `cell_failed` row carries `keywords_emitted = 0`, so:

    def itt(row):
        emitted = row.get("keywords_emitted") or 0
        if not emitted:
            return 1.0

An HTTP error is scored as total invention failure, and `Cell.mutes()` counts it
as a query the model declined to answer. Those are different claims about
different things: silence is a property of the treatment, a 429 is a property of
the afternoon.

This is not hypothetical. `arm1_merged_70b_v5fp8_4pass` -- the artifact behind
slide 9's Arm 1 column -- reports ITT 0.3914 over n=200 with 2 mutes. Both
"mutes" are 429s. Excluding them gives 0.3852 over n=198 with **zero** mutes.
The ITT moves only -0.0061, but the qualitative claim inverts: the 70B never
went silent on this query set.
"""

import importlib.util
import json
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_arms", Path(__file__).resolve().parents[1] / "scripts" / "aggregate_arms.py"
)
agg = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_arms"] = agg
_SPEC.loader.exec_module(agg)


def _row(query_id, status, emitted, hallucinated):
    return {
        "query_id": query_id,
        "repeat": 1,
        "status": status,
        "keywords_emitted": emitted,
        "hallucinated_keywords": hallucinated,
        "retrieved_count": 10 if status == "ok" else 0,
        "cited_count": 1 if status == "ok" else 0,
    }


def _run(tmp_path, rows):
    (tmp_path / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")
    return tmp_path


def test_cell_failed_row_is_not_loaded_as_a_measurement(tmp_path):
    """The 429 row must not enter the cell at all."""
    cell = agg.load(_run(tmp_path, [
        _row("q01", "ok", 10, 2),
        _row("q02", "ok", 10, 4),
        _row("q03", "cell_failed", 0, 0),   # HTTP 429, nothing was measured
    ]))
    assert cell.n == 2, (
        f"a transport failure was loaded as a measured row (n={cell.n}, expected 2)"
    )


def test_transport_failure_is_not_counted_as_a_mute(tmp_path):
    """`mutes()` must report queries the MODEL declined, not ones the network did."""
    cell = agg.load(_run(tmp_path, [
        _row("q01", "ok", 10, 2),
        _row("q02", "ok", 0, 0),            # a real mute: the model emitted nothing
        _row("q03", "cell_failed", 0, 0),   # not a mute: the request never landed
    ]))
    assert cell.mutes() == 1, (
        f"mutes()={cell.mutes()}, expected 1 -- a 429 is being counted as silence"
    )


def test_transport_failure_does_not_inflate_itt(tmp_path):
    """ITT over two clean rows is (0.2 + 0.4)/2 = 0.3, with or without the 429."""
    rows = [_row("q01", "ok", 10, 2), _row("q02", "ok", 10, 4)]

    clean_dir = tmp_path / "a"
    clean_dir.mkdir()
    clean = agg.load(_run(clean_dir, [*rows]))

    dirty_dir = tmp_path / "b"
    dirty_dir.mkdir()
    dirty = agg.load(_run(dirty_dir, [*rows, _row("q03", "cell_failed", 0, 0)]))

    clean_itt = sum(agg.itt(r) for r in clean.rows.values()) / clean.n
    dirty_itt = sum(agg.itt(r) for r in dirty.rows.values()) / dirty.n
    assert dirty_itt == clean_itt, (
        f"ITT {dirty_itt:.4f} with the 429 vs {clean_itt:.4f} without -- "
        "the transport failure is being scored 1.0"
    )


def test_arm_failed_is_excluded_on_the_same_grounds(tmp_path):
    """`arm_failed` is the same class of event and gets the same treatment."""
    cell = agg.load(_run(tmp_path, [
        _row("q01", "ok", 10, 2),
        _row("q02", "arm_failed", 0, 0),
    ]))
    assert cell.n == 1, f"an arm_failed row was loaded as a measurement (n={cell.n})"


def test_rows_without_a_status_field_are_still_kept(tmp_path):
    """Absent is unknown, never failed.

    Runs predating the `status` column must not vanish from the corpus, which is
    the mirror-image error of the one this module exists to prevent.
    """
    cell = agg.load(_run(tmp_path, [
        {"query_id": "q01", "repeat": 1, "keywords_emitted": 10, "hallucinated_keywords": 2},
        {"query_id": "q02", "repeat": 1, "keywords_emitted": 0, "hallucinated_keywords": 0},
    ]))
    assert cell.n == 2, f"unstamped rows were dropped (n={cell.n}, expected 2)"
    assert cell.mutes() == 1
