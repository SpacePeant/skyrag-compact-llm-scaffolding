"""The list that turns a full edX rebuild into the catalogue the study measured.

A full rebuild gives 3,477 edX courses and the published study measured 3,451.
`scripts/prepare_corpus.py --as-published` removes the 26 listed in
`benchmarks/edx_published_omissions.json`. The first two tests are offline. The
third needs `mine/raw/` and `documents/`, which are not in git, and skips
without them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("prepare_corpus", REPO / "scripts" / "prepare_corpus.py")
pc = importlib.util.module_from_spec(_spec)
sys.modules["prepare_corpus"] = pc
_spec.loader.exec_module(pc)


def _omissions() -> list[dict]:
    return json.loads(pc.EDX_OMISSIONS.read_text(encoding="utf-8"))


def test_the_list_accounts_for_the_whole_gap():
    rows = _omissions()
    assert len(rows) == pc.REBUILD_COUNT["edx"] - pc.SHIPPED_COUNT["edx"] == 26
    assert len({row["url"] for row in rows}) == 26
    assert all(row["url"].startswith("https://www.edx.org/") and row["course_title"] for row in rows)


def test_a_listed_course_missing_from_the_rebuild_is_refused():
    records = [{"url": row["url"]} for row in _omissions()][:-1] + [{"url": "https://www.edx.org/other"}]
    with pytest.raises(SystemExit):
        pc.drop_published_omissions(records)


def test_full_rebuild_minus_the_list_is_the_published_catalogue():
    raw = pc.RAW / pc.EDX_JSONL
    published = pc.DOCUMENTS / "EdxDataset-Clean.json"
    if not (raw.exists() and published.exists()):
        pytest.skip("mine/raw/ and documents/ are not in git")
    rebuilt = pc.drop_published_omissions(pc.clean_edx(raw))
    assert rebuilt == json.loads(published.read_text(encoding="utf-8"))
