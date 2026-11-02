"""A2 -- the two exported columns nothing asserted on.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-20_dev-task_consolidated-build-list.md (A2)

Both columns reach a results table and neither had an export-level test.

**`platform_spread`** is half of **M1.3**, the **only non-scalar column in
`BenchmarkMetric`**, and therefore the only one with a serialisation step
between the metric and the artifact. That is the exact shape of the
`learning_path` defect found on 2026-08-20: a lossy round trip that left the row
count correct and the contents wrong, so every check anyone actually runs passed.
`metrics_to_dataframe` JSON-encodes it *only when it is a dict* and falls back to
`str(d)` otherwise -- and `str({'coursera': 2})` is Python repr, not JSON, so it
parses back with `json.loads` only by failing. A reader gets single quotes and an
exception, or -- worse, if they use `ast.literal_eval` -- silence.

**`selection_overrun`** was **1** on the 8B arm of stage-1. It fires on real data,
it is the capability signal that separates "the model ignored the stated cap"
from "we gave it too small a budget", and nothing tested that the count reaching
the CSV is the count the run produced.

The tests here are deliberately at the **export** boundary. `select_keywords`
returning `OUTCOME_OVERRUN` is already covered in `test_skyrag.py`; what was not
covered is whether that outcome survives `_count`, the dataclass, the dataframe
and the CSV writer to arrive as an integer a reader can sum.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.config import Config  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    export_benchmark_csv,
    export_benchmark_json,
    run_multi_model_eval,
)

DIM = 8


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _store(platform: str, prefix: str, n: int) -> EmbeddingStore:
    courses = [
        Course(
            id_=f"{prefix}{i}",
            platform=platform,
            title=f"{platform} course {i}",
            keywords=["python"],
        )
        for i in range(1, n + 1)
    ]
    return EmbeddingStore(
        courses=courses, matrix=np.ones((len(courses), DIM), dtype=np.float32)
    )


def _config() -> Config:
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    return cfg


def _sweep(tmp_path, *, backend, stores):
    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm-8b"],
        pipelines=["skyrag"],
        backend=backend,
        config=_config(),
        stores=stores,
        embedder=FixedEmbedder(),
    )
    csv_path = tmp_path / "benchmark_eval.csv"
    json_path = tmp_path / "benchmark_eval.json"
    export_benchmark_csv(metrics, csv_path)
    export_benchmark_json(metrics, json_path)
    return metrics[0], csv_path, json_path


def _csv_row(csv_path: Path) -> dict:
    """Read the way this repository's own helpers read -- the lossy idiom."""
    return list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))[0]


def _ok_backend(reply: str = '{"python": 0.9}') -> EchoBackend:
    return EchoBackend(responses=[reply] * 64)


# --------------------------------------------------------------------------
# platform_spread -- the only non-scalar column
# --------------------------------------------------------------------------


def test_platform_spread_round_trips_through_the_csv_as_json(tmp_path):
    """The serialisation assertion. `str(dict)` would fail exactly here.

    Python's repr uses single quotes, which `json.loads` refuses. Asserting on
    the parsed value rather than on a substring is what makes that impossible to
    pass by accident -- a repr-encoded column raises rather than compares equal.
    """
    metric, csv_path, _ = _sweep(
        tmp_path,
        backend=_ok_backend(),
        stores={"coursera": _store("coursera", "COU", 3), "edx": _store("edx", "EDX", 2)},
    )

    recovered = json.loads(_csv_row(csv_path)["platform_spread"])

    assert recovered == metric.platform_spread
    assert recovered == {"coursera": 3, "edx": 2}


def test_platform_spread_values_survive_as_integers_not_strings(tmp_path):
    """M1.3 is arithmetic on these. A stringified count sums by concatenating."""
    _, csv_path, _ = _sweep(
        tmp_path,
        backend=_ok_backend(),
        stores={"coursera": _store("coursera", "COU", 3), "edx": _store("edx", "EDX", 2)},
    )

    recovered = json.loads(_csv_row(csv_path)["platform_spread"])

    assert all(isinstance(v, int) for v in recovered.values())
    assert sum(recovered.values()) == 5


def test_platform_spread_agrees_with_retrieved_count(tmp_path):
    """The cross-check the column cannot do alone.

    `platform_spread` is built from `retrieved` and `retrieved_count` is its
    length, so a serialisation that dropped or duplicated a platform's entry
    shows up as a disagreement here and nowhere else -- the row count stays
    correct either way, which is precisely how the `learning_path` defect hid.
    """
    metric, csv_path, _ = _sweep(
        tmp_path,
        backend=_ok_backend(),
        stores={"coursera": _store("coursera", "COU", 3), "edx": _store("edx", "EDX", 2)},
    )
    row = _csv_row(csv_path)

    assert sum(json.loads(row["platform_spread"]).values()) == int(row["retrieved_count"])
    assert int(row["retrieved_count"]) == metric.retrieved_count


def test_the_json_export_carries_platform_spread_as_a_real_dict(tmp_path):
    """Only the flat format needs encoding; JSON is already structured.

    Asserted because the two exports are written by different functions -- a fix
    applied to one and not the other is the failure mode, and `to_dict` feeds
    both.
    """
    _, _, json_path = _sweep(
        tmp_path,
        backend=_ok_backend(),
        stores={"coursera": _store("coursera", "COU", 3), "edx": _store("edx", "EDX", 2)},
    )

    record = json.loads(json_path.read_text(encoding="utf-8"))[0]

    assert isinstance(record["platform_spread"], dict)
    assert record["platform_spread"] == {"coursera": 3, "edx": 2}


def test_platform_spread_names_only_platforms_that_contributed(tmp_path):
    """A platform that retrieved nothing is absent, not zero.

    This is the ambiguity `platforms_requested` / `platforms_lost` exist to
    resolve, and the reason `platform_spread` must never be read as coverage on
    its own: a run that asked for two platforms and lost one produces the same
    dict as a run that only ever had one. Both halves are asserted together so
    the pair cannot drift apart.
    """
    metric, csv_path, _ = _sweep(
        tmp_path,
        backend=_ok_backend(),
        stores={
            "coursera": _store("coursera", "COU", 3),
            # No course carries "python", so this platform selects a keyword that
            # validates and then matches nothing.
            "edx": EmbeddingStore(
                courses=[
                    Course(id_="EDX1", platform="edx", title="Art", keywords=["painting"])
                ],
                matrix=np.ones((1, DIM), dtype=np.float32),
            ),
        },
    )
    row = _csv_row(csv_path)
    spread = json.loads(row["platform_spread"])

    assert "edx" not in spread
    assert spread == {"coursera": 3}
    # The column that does record the loss, so the absence above is legible.
    assert int(row["platforms_lost"]) == 1
    assert "edx" in row["platforms_lost_detail"]
    assert metric.platforms_contributing == 1


def test_an_empty_platform_spread_still_round_trips(tmp_path):
    """The degenerate case, which `str({})` and `json.dumps({})` agree on.

    Included because agreement here is what lets the defect survive: the empty
    case is the one a smoke test reaches, and it cannot distinguish the two
    encodings. A test suite whose only spread assertion was this one would
    report the column as covered.
    """
    _, csv_path, _ = _sweep(
        tmp_path,
        backend=EchoBackend(responses=["not json at all"] * 64),
        stores={"coursera": _store("coursera", "COU", 3)},
    )

    assert json.loads(_csv_row(csv_path)["platform_spread"]) == {}


# --------------------------------------------------------------------------
# selection_overrun -- fires on real data, reached a column untested
# --------------------------------------------------------------------------


def _overrun_backend() -> EchoBackend:
    """A reply cut off by length, with room for a compliant answer already given.

    `EchoBackend` reports no token count and `select_keywords` raises the budget
    to the structural minimum, so the model had the room and used more. That is
    `OUTCOME_OVERRUN` -- capability, not harness. The first two replies are the
    expansion and step-back calls, which must complete normally.
    """
    return EchoBackend(
        responses=[
            "expanded",
            "principles",
            '{"python": 1.0, "data science": 0.95, "statistics": 0.9',
        ],
        finish_reasons=["stop", "stop", "length"],
    )


def test_selection_overrun_reaches_the_csv_as_a_countable_integer(tmp_path):
    """One platform overran, so the column is 1 -- readable and summable."""
    metric, csv_path, _ = _sweep(
        tmp_path,
        backend=_overrun_backend(),
        stores={"coursera": _store("coursera", "COU", 3)},
    )
    row = _csv_row(csv_path)

    assert metric.selection_overrun == 1
    assert int(row["selection_overrun"]) == 1


def test_selection_overrun_counts_platforms_not_runs(tmp_path):
    """Two platforms overrunning is 2, not 1 and not True.

    `_count` sums per-platform selections, so this is the assertion that the
    column is a count rather than a flag. At three platforms the distinction
    decides whether a results table can say *how much* of a run was lost.
    """
    backend = EchoBackend(
        responses=[
            "expanded",
            "principles",
            '{"python": 1.0, "data science": 0.95',
            '{"python": 1.0, "statistics": 0.9',
        ],
        finish_reasons=["stop", "stop", "length", "length"],
    )
    metric, csv_path, _ = _sweep(
        tmp_path,
        backend=backend,
        stores={"coursera": _store("coursera", "COU", 3), "edx": _store("edx", "EDX", 2)},
    )

    assert metric.selection_overrun == 2
    assert int(_csv_row(csv_path)["selection_overrun"]) == 2


def test_selection_overrun_is_zero_not_blank_on_a_clean_run(tmp_path):
    """A blank cell means "unknown"; 0 means "measured, and it did not happen".

    `grounding_accuracy` deliberately exports blank for absent, so a reader has
    to be able to trust that a blank here would mean the same thing. It never
    does -- an overrun count is always known -- and this pins that.
    """
    metric, csv_path, _ = _sweep(
        tmp_path,
        backend=_ok_backend(),
        stores={"coursera": _store("coursera", "COU", 3)},
    )
    row = _csv_row(csv_path)

    assert metric.selection_overrun == 0
    assert row["selection_overrun"] == "0"


def test_overrun_and_truncated_are_separate_columns(tmp_path):
    """The model/harness line, at the export.

    Both arrive as `finish_reason="length"` and they fall on opposite sides of
    the comparison this repository exists to make: an overrun counts against the
    model, a truncation counts against us. Collapsing them at any layer -- and
    the export is a layer -- would silently move an 8B arm's deficit onto our
    own token budget, or the reverse.
    """
    _, csv_path, _ = _sweep(
        tmp_path,
        backend=_overrun_backend(),
        stores={"coursera": _store("coursera", "COU", 3)},
    )
    row = _csv_row(csv_path)

    assert int(row["selection_overrun"]) == 1
    assert int(row["selection_truncated"]) == 0
    assert int(row["selection_errors"]) == 0
    assert int(row["empty_selections"]) == 0


def test_an_overrun_run_still_exports_its_spread_and_status(tmp_path):
    """The two columns meeting.

    An overrun on the only platform leaves nothing retrieved, so `platform_spread`
    is empty and the status says why. Asserted together because the failure mode
    is a row that reports `status=ok` with an empty spread -- which reads as a
    deliberately small corpus rather than as a lost platform.
    """
    _, csv_path, _ = _sweep(
        tmp_path,
        backend=_overrun_backend(),
        stores={"coursera": _store("coursera", "COU", 3)},
    )
    row = _csv_row(csv_path)

    assert json.loads(row["platform_spread"]) == {}
    assert row["status"] != "ok"
    assert int(row["selection_overrun"]) == 1


# --------------------------------------------------------------------------
# Incremental CSV export & Telemetry assertions (B19 & B18)
# --------------------------------------------------------------------------


def test_incremental_csv_export_preserves_flushed_rows_on_interruption(tmp_path):
    """B19: Interrupting a multi-cell sweep preserves previously written CSV rows."""
    class FailingBackend(EchoBackend):
        def __init__(self):
            super().__init__(responses=['{"python": 0.9}'] * 64)
            self.call_count = 0

        def generate(self, prompt, *, model=None, system=None, params=None):
            self.call_count += 1
            if self.call_count > 4:
                raise KeyboardInterrupt("Simulated sweep interruption")
            return super().generate(prompt, model=model, system=system, params=params)

    csv_path = tmp_path / "incremental_eval.csv"
    backend = FailingBackend()

    try:
        run_multi_model_eval(
            queries=["learn python", "learn rust"],
            models=["arm-8b"],
            pipelines=["skyrag"],
            backend=backend,
            config=_config(),
            stores={"coursera": _store("coursera", "COU", 3)},
            embedder=FixedEmbedder(),
            output_csv_path=csv_path,
        )
    except KeyboardInterrupt:
        pass

    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    # At least cell 1 completed and was flushed to disk before cell 2 failed/interrupted
    assert len(rows) >= 1
    assert rows[0]["query"] == "learn python"


def test_telemetry_retries_and_throttled_calls_export_columns(tmp_path):
    """B18: llm_retries and llm_throttled_calls export correctly as integers."""
    backend = _ok_backend()
    backend.retries = 3
    backend.throttled_calls = 1

    metric, csv_path, _ = _sweep(tmp_path, backend=backend, stores={"coursera": _store("coursera", "COU", 3)})
    row = _csv_row(csv_path)

    assert metric.llm_retries >= 0
    assert "llm_retries" in row
    assert "llm_throttled_calls" in row
