"""T1 -- "arm" means one thing, and a grid unit is a "cell".

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-20_dev-task_consolidated-build-list.md (T1)

`eval.py` used "arm" for a **cell** of the sweep grid -- one
(query x model x pipeline) unit -- in `_run_arm`, `arm_failure`, `arm_model`,
`failed_arms` and `STATUS_ARM_FAILED = "arm_failed"`. Every research document in
this project uses "arm" for the **intervention axis**: Arm 1 baseline 70B, Arm 2
naive 8B, Arm 3 skill schemas. **B0 introduces `--arm` with the
second meaning into the same module that already used the first.**

Two senses of one word in one results table is the kind of defect that is a
rename today and an ambiguous column in a published table later, which is why
T1 sits ahead of B0 rather than beside it.

**Why this is a test and not a convention.** The collision arose in the first
place because nothing stopped it, and B0 will be written by someone reading the
specification -- where "arm" means the intervention -- into a module where it
historically meant a cell. A convention recorded in a comment is exactly what
was in force while the collision accumulated five identifiers.

`naive` is not an arm and never becomes one (D3): all four arms run the `skyrag`
pipeline, and `naive` is a reference condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.pipelines import base as base_module  # noqa: E402
from skyrag.pipelines import eval as eval_module  # noqa: E402


def test_the_failure_status_names_a_cell_not_an_arm():
    """The value, not just the constant -- it is what reaches the `status` column."""
    assert base_module.STATUS_CELL_FAILED == "cell_failed"


def test_the_old_name_is_gone_rather_than_aliased():
    """An alias would keep both spellings live, which is the state T1 ends.

    Pre-rename artifacts (`outputs/eval_20260818_arm1-2_llama70b-8b_4q_live-openrouter/`) still carry `arm_failed`;
    that is recorded in `base.py` beside the constant. Nothing reads it
    programmatically, so a compatibility alias would be an unexercised code path
    asserting a support commitment this project does not have.
    """
    assert not hasattr(base_module, "STATUS_ARM_FAILED")


@pytest.mark.parametrize(
    "name", ["_run_arm", "arm_failure", "arm_model", "failed_arms"]
)
def test_no_grid_level_identifier_spells_itself_arm(name):
    """The five holdouts, pinned individually so a revert names which one."""
    assert not hasattr(eval_module, name)
    assert not hasattr(eval_module.BenchmarkMetric, name)


@pytest.mark.parametrize(
    "name", ["_run_cell", "_stage_total"]
)
def test_the_cell_level_replacements_exist(name):
    """The other half. Asserting only the absence above would pass on a typo."""
    assert hasattr(eval_module, name)


def test_the_metric_carries_a_cell_failure_constructor():
    assert hasattr(eval_module.BenchmarkMetric, "cell_failure")


def test_a_failed_cell_row_reports_the_cell_status():
    """End to end: the constructor, the constant and the column agree.

    Separate from the constant test because the two could drift -- a
    `cell_failure` that stamped some other status would satisfy both assertions
    above and still export a row nobody can filter on.
    """
    metric = eval_module.BenchmarkMetric.cell_failure(
        model="arm-8b",
        pipeline="skyrag",
        query="learn python",
        error=RuntimeError("endpoint refused"),
    )

    assert metric.status == "cell_failed"
    assert metric.status == base_module.STATUS_CELL_FAILED
    # And it still cannot enter any mean -- the property the row exists for.
    assert metric.grounding_accuracy is None
    assert metric.grounding_accuracy_n == 0
    assert metric.tokens_complete is False


# --------------------------------------------------------------------------
# R6r -- retrieval depth reaches the artifact
#
# Not a naming item, but it lands here because it is the same failure shape the
# tests above guard: a fact about a run that lived only in someone's memory of
# how the run was configured.
#
# Closing the outstanding "re-run the two-arm sweep at `per_keyword=7`" item
# required *inferring* the depth of `outputs/ablation_20260819_kw-sub_llama8b-kw-70b-gen_10q_coreweave/` from
# its retrieved counts -- 253 courses is impossible at `per_keyword=1` and
# ordinary at 7. That inference is sound and it is not a record. `per_keyword`
# and `max_keywords` decide `retrieved_count` and therefore the size of every
# generation prompt, so a sweep that cannot state them cannot be compared with
# one that used different values.
# --------------------------------------------------------------------------


def test_provenance_stamps_the_retrieval_depth():
    from skyrag.config import Config
    from skyrag.pipelines import RunProvenance

    config = Config()
    provenance = RunProvenance.capture(config, "echo")

    assert provenance.per_keyword == config.retrieval.per_keyword
    assert provenance.max_keywords == config.retrieval.max_keywords


def test_retrieval_depth_reaches_every_exported_row():
    """Per row, not per file -- rows from several sweeps get merged for analysis.

    The moment that happens a run-level header no longer travels with the row it
    describes, which is the argument `max_context_courses` was already stamped
    on. These two belong beside it for the same reason.
    """
    import csv

    import numpy as np

    from skyrag.config import Config
    from skyrag.data.schema import Course
    from skyrag.embedding import EmbeddingStore
    from skyrag.llm import EchoBackend
    from skyrag.pipelines import (
        RunProvenance,
        export_benchmark_csv,
        run_multi_model_eval,
    )

    config = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(config.llm, attr, "stub")

    class FixedEmbedder:
        def encode_one(self, text):
            return np.ones(8, dtype=np.float32)

    store = EmbeddingStore(
        courses=[Course(id_="COU1", platform="coursera", title="Py", keywords=["python"])],
        matrix=np.ones((1, 8), dtype=np.float32),
    )
    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["skyrag"],
        backend=EchoBackend(responses=['{"python": 0.9}'] * 16),
        config=config,
        stores={"coursera": store},
        embedder=FixedEmbedder(),
        provenance=RunProvenance.capture(config, "echo"),
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.csv"
        export_benchmark_csv(metrics, path)
        row = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))[0]

    assert int(row["per_keyword"]) == config.retrieval.per_keyword
    assert int(row["max_keywords"]) == config.retrieval.max_keywords
