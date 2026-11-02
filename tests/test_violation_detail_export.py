"""The reasons and raw reply behind a selection schema violation reach the row.

Added 2026-09-30 with `selection_violation_detail` / `selection_violation_raw`.
Before them only the count was exported, so which rule LLaMA 3.1-8B broke on
61% of injected Arm 3 runs could not be read back from any artifact.
"""

from __future__ import annotations

import json

from skyrag.pipelines.base import PipelineResult
from skyrag.pipelines.eval import _violation_json
from skyrag.selection import OUTCOME_OK, OUTCOME_SCHEMA_VIOLATION, KeywordSelection


def _result(*selections: KeywordSelection) -> PipelineResult:
    return PipelineResult(pipeline="skyrag", query="q", learning_path="", selections=list(selections))


def test_only_violated_selections_are_recorded_by_platform():
    bad = KeywordSelection(platform="coursera", keywords={}, raw_response='{"platform": "coursera"}',
                           outcome=OUTCOME_SCHEMA_VIOLATION,
                           schema_violations=["$: missing required property 'keywords'"])
    good = KeywordSelection(platform="edx", keywords={"python": 0.9}, raw_response="{...}", outcome=OUTCOME_OK)
    res = _result(bad, good)
    assert json.loads(_violation_json(res, lambda s: s.schema_violations)) == {
        "coursera": ["$: missing required property 'keywords'"]}
    assert json.loads(_violation_json(res, lambda s: s.raw_response)) == {"coursera": '{"platform": "coursera"}'}


def test_no_violation_leaves_the_columns_empty():
    ok = KeywordSelection(platform="edx", keywords={"python": 0.9}, outcome=OUTCOME_OK)
    assert _violation_json(_result(ok), lambda s: s.schema_violations) == ""
    assert _violation_json(_result(), lambda s: s.raw_response) == ""


def test_cutoff_replies_are_recorded_separately_c16():
    from skyrag.selection import OUTCOME_OVERRUN

    cut = KeywordSelection(platform="coursera", keywords={}, raw_response='{"platform": "coursera", "keywords": {"a": 0.9,',
                           outcome=OUTCOME_OVERRUN)
    res = _result(cut)
    picked = _violation_json(res, lambda s: s.raw_response, (OUTCOME_OVERRUN,))
    assert json.loads(picked) == {"coursera": '{"platform": "coursera", "keywords": {"a": 0.9,'}
    # an overrun is not a schema violation, so the violation columns stay empty
    assert _violation_json(res, lambda s: s.raw_response) == ""
