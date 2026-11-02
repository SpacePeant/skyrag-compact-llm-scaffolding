"""The generated learning path reaches an artifact, and survives being read back.

Cross-reference:
- project_knowledge/research_directions/research_backed_metrics_recommendations.md

Until 2026-08-20 `skyrag eval` exported `BenchmarkMetric` rows only. The generated
path was used to compute `cited_ids()` and then discarded with the
`PipelineResult`, so every number in every CSV described a text nobody could read
back. That blocked every claim-level metric and it blocked **B3** -- a judge
harness cannot score post-hoc from an export carrying no output.

The tests that matter here are not "is the column present". They are:

* **It survives the lossy reading idiom.** `csv.DictReader(text.splitlines())` is
  what every helper in this repository uses, and on a field containing a newline
  it keeps the row count and silently deletes the newline. A test that reads the
  CSV with pandas would pass while the project's own idiom corrupted the data.

* **The count and the ID list cannot disagree.** `cited_count` and
  `cited_course_ids` come from one call; if they ever diverge, one of them is
  lying about the same run.

* **Nothing is truncated.** A length cap would corrupt exactly the runs worth
  looking at.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.config import Config  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    BenchmarkMetric,
    export_benchmark_csv,
    export_benchmark_json,
    run_multi_model_eval,
)

DIM = 8

MULTILINE_PATH = (
    'Step 1: COU1 -- "Foundations", the entry point.\n'
    "Step 2: COU2 -- builds on it.\n"
    "Step 3: EDX1 -- capstone.\\end"
)


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _run(tmp_path, *, generated: str):
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    courses = [
        Course(id_="COU1", platform="coursera", title="Foundations", keywords=["python"]),
        Course(id_="COU2", platform="coursera", title="Next", keywords=["python"]),
    ]
    store = EmbeddingStore(
        courses=courses, matrix=np.ones((len(courses), DIM), dtype=np.float32)
    )
    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["naive"],
        # `run_naive` calls the backend twice -- step-back principles, then
        # generation. Both are scripted so `generated` lands on the generation
        # call; leaving the second unscripted makes `EchoBackend` fall back to
        # echoing the prompt, which contains the course IDs and quietly turns
        # every citation assertion into a test of the echo path.
        backend=EchoBackend(responses=["principles", generated]),
        config=cfg,
        stores={"coursera": store},
        embedder=FixedEmbedder(),
    )
    csv_path = tmp_path / "benchmark_eval.csv"
    json_path = tmp_path / "benchmark_eval.json"
    export_benchmark_csv(metrics, csv_path)
    export_benchmark_json(metrics, json_path)
    return metrics[0], csv_path, json_path


# --------------------------------------------------------------------------
# The path reaches an artifact at all
# --------------------------------------------------------------------------


def test_the_generated_path_reaches_the_metric_row(tmp_path):
    metric, _, _ = _run(tmp_path, generated=MULTILINE_PATH)
    assert metric.learning_path == MULTILINE_PATH


def test_the_json_export_carries_the_path_raw(tmp_path):
    """Only the flat format needs escaping. JSON is already structured."""
    _, _, json_path = _run(tmp_path, generated=MULTILINE_PATH)
    record = json.loads(json_path.read_text(encoding="utf-8"))[0]
    assert record["learning_path"] == MULTILINE_PATH


# --------------------------------------------------------------------------
# Surviving the way this repository actually reads a CSV
# --------------------------------------------------------------------------


def test_the_exported_path_survives_a_csv_round_trip(tmp_path):
    """The property the JSON-encoding exists for.

    `splitlines()` on a raw multi-line field keeps the row count and welds the
    lines together, so the corruption is invisible to every check anyone runs.
    Encoded, the field is single-line and comes back byte-identical.
    """
    _, csv_path, _ = _run(tmp_path, generated=MULTILINE_PATH)
    text = csv_path.read_text(encoding="utf-8")

    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 1
    assert json.loads(rows[0]["learning_path"]) == MULTILINE_PATH


def test_the_encoded_path_occupies_exactly_one_csv_line(tmp_path):
    """A newline in the field would make one record span several lines.

    Asserted directly rather than inferred from the round trip, because this is
    the property every line-oriented tool -- `wc -l`, `head`, `split` -- depends
    on, and none of them will complain when it stops holding.
    """
    _, csv_path, _ = _run(tmp_path, generated=MULTILINE_PATH)
    lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2          # header + one record


def test_the_exported_path_also_survives_the_correct_reader(tmp_path):
    """Encoding must not break the readers that were never broken."""
    _, csv_path, _ = _run(tmp_path, generated=MULTILINE_PATH)
    text = csv_path.read_text(encoding="utf-8")
    row = next(iter(csv.DictReader(io.StringIO(text))))
    assert json.loads(row["learning_path"]) == MULTILINE_PATH


@pytest.mark.parametrize("payload", [
    'He said "take COU1" first',          # quotes
    "back\\slash and COU1",               # backslash
    "comma, separated, COU1",             # the delimiter itself
    "tab\tand COU1",                      # control character
    "",                                   # empty generation
])
def test_awkward_characters_round_trip_unchanged(tmp_path, payload):
    """Every one of these is why the escape is `json.dumps` and not a newline swap."""
    _, csv_path, _ = _run(tmp_path, generated=payload)
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert json.loads(rows[0]["learning_path"]) == payload


def test_a_long_path_is_not_truncated(tmp_path):
    """A length cap would corrupt exactly the runs worth reading."""
    long_path = "Step 1: COU1. " + ("filler words repeated. " * 2000)
    metric, csv_path, _ = _run(tmp_path, generated=long_path)

    assert metric.learning_path == long_path
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert json.loads(rows[0]["learning_path"]) == long_path


# --------------------------------------------------------------------------
# The ID columns
# --------------------------------------------------------------------------


def test_cited_ids_are_exported_in_order_and_match_the_count(tmp_path):
    """`cited_count` says how many; without this column nobody can say which."""
    metric, csv_path, _ = _run(tmp_path, generated="Step 1: COU2 then Step 2: COU1")

    assert metric.cited_course_ids == "COU2;COU1"
    assert len(metric.cited_course_ids.split(";")) == metric.cited_count

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["cited_course_ids"] == "COU2;COU1"


def test_hallucinated_ids_are_exported_so_m3_1_is_auditable(tmp_path):
    """A bare count cannot distinguish a plausible invention from garbage."""
    metric, _, _ = _run(tmp_path, generated="Step 1: COU1 then Step 2: COU9")

    assert metric.hallucinated_course_ids == "COU9"
    assert metric.hallucinated_count == 1


def test_a_run_that_cited_nothing_exports_empty_id_columns(tmp_path):
    """Empty, not absent, and not a stray separator."""
    metric, _, _ = _run(tmp_path, generated="No suitable courses were found.")

    assert metric.cited_course_ids == ""
    assert metric.hallucinated_course_ids == ""
    assert metric.cited_count == 0


def test_an_arm_failure_row_carries_empty_artifact_columns():
    """A row for an arm that never ran has no path, and must not invent one."""
    metric = BenchmarkMetric.cell_failure(
        model="arm", pipeline="skyrag", query="q", error=RuntimeError("boom")
    )
    assert metric.learning_path == ""
    assert metric.cited_course_ids == ""
    assert metric.hallucinated_course_ids == ""
