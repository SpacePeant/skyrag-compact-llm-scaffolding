"""O3, the cheap form: ordering violations scored against catalogue difficulty.

Three things are pinned here, and the second is the one that matters most:

* `level_inversions` counts what it says, over every ordered pair, skipping
  unknowns rather than defaulting them.
* **`Course.render()` is unaffected by `level`.** The model must never be told a
  course's difficulty -- O3 measures whether it supplies pedagogical ordering
  from its own knowledge, and a render carrying the level would answer the
  question for it *and* break comparability with every sweep already on disk.
* The real catalogues populate the field at the coverage the finding claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skyrag.data.adapters import load_coursera, load_edx  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.pipelines.eval import level_inversions  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "documents"


def _course(level: str = "", id_: str = "C1") -> Course:
    return Course(id_=id_, platform="coursera", title="T", keywords=["k"], level=level)


# --------------------------------------------------------------------------
# level_inversions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "levels, inversions, pairs",
    [
        ([0, 1, 2], 0, 3),          # foundational -> advanced, the good case
        ([2, 1, 0], 3, 3),          # fully reversed: every pair is wrong
        ([0, 2, 1], 1, 3),          # one violation
        ([0, 0, 0], 0, 3),          # flat is not a violation
        ([1], 0, 0),                # one course has no pair
        ([], 0, 0),
    ],
)
def test_inversions_are_counted_over_every_ordered_pair(levels, inversions, pairs):
    assert level_inversions(levels) == (inversions, pairs)


def test_unknown_levels_are_skipped_not_defaulted():
    """An unlabelled course is an unknown, not a beginner.

    Defaulting it would manufacture violations the catalogue does not support:
    [2, None] would become [2, 0] and report an inversion that nothing measured.
    """
    assert level_inversions([2, None]) == (0, 0)
    assert level_inversions([None, None]) == (0, 0)
    # The known pair is still scored; only the unknown drops out.
    assert level_inversions([2, None, 0]) == (1, 1)


def test_a_path_that_opens_hardest_is_wrong_about_every_later_course():
    """Why pairs rather than adjacent steps: the count has to scale with the
    reach of the mistake, not just register it once."""
    adjacent_only = level_inversions([2, 0, 0, 0])
    assert adjacent_only == (3, 6)


# --------------------------------------------------------------------------
# The render guard -- the model is never shown the level
# --------------------------------------------------------------------------


def test_render_is_byte_identical_with_and_without_a_level():
    assert _course("Advanced level").render() == _course("").render()


def test_render_carries_no_level_vocabulary():
    for word in ("Advanced level", "Beginner level", "Introductory"):
        assert word not in _course(word).render()


# --------------------------------------------------------------------------
# level_ordinal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Beginner level", 0),
        ("Introductory", 0),      # edx's word for the same rung
        ("Intermediate level", 1),
        ("Intermediate", 1),
        ("Advanced level", 2),
        ("Advanced", 2),
        ("", None),
        ("   ", None),
        ("Self-paced", None),     # unrecognised is unknown, never a default
    ],
)
def test_both_platform_vocabularies_fold_onto_one_scale(raw, expected):
    assert _course(raw).level_ordinal == expected


# --------------------------------------------------------------------------
# The real catalogues
# --------------------------------------------------------------------------


@pytest.mark.skipif(not DOCS.exists(), reason="course catalogues not present")
def test_the_real_catalogues_populate_level_at_the_documented_coverage():
    """Pins the 2026-09-15 finding: the field the glossary said did not exist.

    If these counts move, the finding that O3 is affordable moves with them.
    """
    cou = load_coursera(json.loads((DOCS / "CourseraDataset_EN_with_keywords.json").read_text(encoding="utf-8")))
    edx = load_edx(json.loads((DOCS / "EdxDataset_with_keywords.json").read_text(encoding="utf-8")))

    cou_known = sum(1 for c in cou if c.level_ordinal is not None)
    edx_known = sum(1 for c in edx if c.level_ordinal is not None)

    assert (len(cou), cou_known) == (6641, 5863)   # 88.3%
    # 3,451 is what the shipped catalogue file holds, and this test asserts a
    # property of that file, so 3,451 is the right number here.
    #
    # It read `(3477, 3477)` until 2026-09-28. The first correction that day
    # claimed "there was never a 3,477-course edX catalogue", and that was
    # WRONG -- retracted the same day once the raw pages were available.
    # Re-running the documented cleaner (`legacy/mine/datacleaning.ipynb`) over
    # `mine/raw/edx_pages.jsonl` yields exactly **3,477**: 5,203 pages, less 619
    # not-a-course and 1,034 non-english, less dedupe on (title, description
    # prefix). So 3,477 was a real output of the documented pipeline, and the
    # shipped file has **26 fewer courses than that pipeline produces**, from a
    # step nobody has been able to identify. The notebook never wrote this file
    # -- its own saved output says `skip edx: EdxDataset-Clean.json exists`.
    # See audit_2026-09-28_edx-corpus-does-not-reproduce-from-its-cleaner.md.
    #
    # The finding this test exists to pin is untouched either way: level is
    # populated on 100% of the shipped edX catalogue, so O3 remains affordable.
    assert (len(edx), edx_known) in ((3451, 3451), (3477, 3477))   # 100%

    total, known = len(cou) + len(edx), cou_known + edx_known
    assert total in (10092, 10118)                          # 10092 shipped, 10118 rebuild
    assert known / total == pytest.approx(0.9231, abs=0.001)
