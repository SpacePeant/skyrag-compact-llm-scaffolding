"""Regression tests for live per-cell progress during a sweep.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-19_dev-task_live-cell-progress.md

`run_multi_model_eval` grew an `on_cell` callback and nothing passed it, so a
paid sweep still printed nothing between the banner and the results table. The
2026-08-18 sweep is the reason that matters: both 70B cells were rejected by the
endpoint and the run looked healthy, while spending, until it ended.

Two properties are asserted, not one:

1. the CLI actually passes a printer -- the defect was a hook with no caller, and
   a unit test of the printer alone would have stayed green through it;
2. what the printer says matches the exported CSV. A progress line is a reported
   number the moment someone reads it off the terminal, and a second computation
   of the same quantity is a second chance to disagree with the artifact.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import skyrag.pipelines as pipelines_pkg  # noqa: E402
from skyrag.cli import DEFAULT_EVAL_MODELS, _print_cell, run_eval_cli  # noqa: E402
from skyrag.config import Config  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import export_benchmark_csv, run_multi_model_eval  # noqa: E402

DIM = 8


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _config() -> Config:
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    return cfg


def _stores() -> dict[str, EmbeddingStore]:
    return {
        "coursera": EmbeddingStore(
            courses=[Course(id_="COU1", platform="coursera", title="Py",
                            keywords=["python"])],
            matrix=np.ones((1, DIM), dtype=np.float32),
        )
    }


class SlowEchoBackend(EchoBackend):
    """An echo backend that takes measurable time per call.

    A stage timing asserted against an instant backend rounds to 0.0 at six
    decimal places, so `>= 0` holds for a timing that was never recorded.
    """

    def __init__(self, delay: float, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    def generate(self, *args, **kwargs):
        time.sleep(self.delay)
        return super().generate(*args, **kwargs)


class _StopHere(Exception):
    """Ends the CLI at the sweep call, so no cell runs and nothing is exported."""


def _run_cli_capturing_the_sweep(monkeypatch, tmp_path, extra_args=()):
    """Drive `skyrag eval` far enough to see what it hands the sweep.

    The embeddings directory is pointed at a name that does not exist, so every
    platform falls back to the synthetic 1-course store and no embedder loads.

    `--allow-synthetic-corpus` is therefore mandatory here (B24, 2026-08-31):
    the CLI now REFUSES a corpus-less run rather than warning about it, because
    the 2026-08-28 sweeps completed against exactly this placeholder, exported
    `corpus_courses = 0` on all 150 rows, and were written up as results. These
    tests are the legitimate case the flag exists for -- they assert CLI
    plumbing and never read a metric -- so they opt in explicitly, which is the
    point of making it something you have to type.
    """
    captured: dict = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        raise _StopHere

    monkeypatch.setenv("SKYRAG_EMBEDDINGS_DIR", "no-such-embeddings-dir")
    monkeypatch.setattr(pipelines_pkg, "run_multi_model_eval", recorder)

    args = [
        "-b", "echo", "-q", "learn python", "-o", str(tmp_path),
        "--allow-synthetic-corpus",
        *extra_args,
    ]
    with pytest.raises(_StopHere):
        run_eval_cli(args)
    return captured


# --------------------------------------------------------------------------
# The hook has a caller
# --------------------------------------------------------------------------

def test_the_cli_passes_a_cell_printer_to_the_sweep(monkeypatch, tmp_path):
    """The defect exactly: `on_cell` existed and arrived as None on every run."""
    captured = _run_cli_capturing_the_sweep(monkeypatch, tmp_path)

    assert captured.get("on_cell") is not None, (
        "skyrag eval ran the sweep without a progress printer -- a paid run is "
        "silent from the banner to the results table"
    )
    assert callable(captured["on_cell"])


def test_the_printer_the_cli_passes_prints_the_cell(monkeypatch, tmp_path, capsys):
    """Wiring and output in one: the object handed over must actually print.

    Asserted through the callback the CLI supplied rather than through
    `_print_cell` directly, so passing some other callable -- or a no-op -- fails
    here.
    """
    captured = _run_cli_capturing_the_sweep(monkeypatch, tmp_path)
    on_cell = captured["on_cell"]
    capsys.readouterr()

    on_cell({"event": "start", "repeat": 1, "query_id": "q02", "index": 3,
             "total": 20, "model": "meta-llama/llama-3.1-8b-instruct",
             "pipeline": "skyrag"})
    on_cell({"event": "done", "repeat": 1, "query_id": "q02", "index": 3,
             "total": 20, "model": "meta-llama/llama-3.1-8b-instruct",
             "pipeline": "skyrag", "status": "ok", "retrieved": 82, "cited": 7,
             "selection_seconds": 1.25, "generation_seconds": 4.5,
             "elapsed_seconds": 12.75})

    out = capsys.readouterr().out
    assert "[  3/20]" in out
    assert "q02" in out and "meta-llama/llama-3.1-8b-instruct" in out
    assert "skyrag" in out
    assert "retrieved=82" in out and "cited=7" in out
    assert "sel=1.2s" in out and "gen=4.5s" in out and "total=12.8s" in out
    # One cell, one line: the outcome completes the line the start opened.
    assert len([ln for ln in out.splitlines() if ln.strip()]) == 1


def test_a_failed_cell_names_its_error_on_the_line(capsys):
    """`cell_failed` is the row that matters most while the run is still spending."""
    _print_cell({"event": "start", "repeat": 1, "query_id": "q01", "index": 1,
                 "total": 4, "model": "70b", "pipeline": "skyrag"})
    _print_cell({"event": "failed", "repeat": 1, "query_id": "q01", "index": 1,
                 "total": 4, "model": "70b", "pipeline": "skyrag",
                 "error": "LLMError: HTTP 400 maximum context length"})

    out = capsys.readouterr().out
    assert "LLMError: HTTP 400 maximum context length" in out
    assert len([ln for ln in out.splitlines() if ln.strip()]) == 1


# --------------------------------------------------------------------------
# The line and the artifact are the same numbers
# --------------------------------------------------------------------------

def test_every_cell_is_announced_before_it_runs_and_reported_after():
    events: list[dict] = []
    metrics = run_multi_model_eval(
        queries=["learn python", "learn ml"],
        models=["arm-8b", "arm-70b"],
        pipelines=["skyrag"],
        backend=EchoBackend(responses=['{"python": 0.9}'] * 64),
        config=_config(),
        stores=_stores(),
        embedder=FixedEmbedder(),
        on_cell=events.append,
    )

    starts = [e for e in events if e["event"] == "start"]
    ends = [e for e in events if e["event"] in ("done", "failed")]

    assert len(starts) == len(ends) == len(metrics) == 4
    assert all(e["total"] == 4 for e in events)
    assert [e["index"] for e in starts] == [1, 2, 3, 4]
    assert [e["index"] for e in ends] == [1, 2, 3, 4]


def test_the_progress_line_reports_what_the_csv_reports(tmp_path):
    """A number read off the terminal is a reported number.

    The event payload is read from the row that was just appended; this pins that
    it stays that way, so nobody can later "fix up" a display value into something
    the artifact does not say.
    """
    events: list[dict] = []
    metrics = run_multi_model_eval(
        queries=["learn python", "learn ml"],
        models=["arm-8b", "arm-70b"],
        pipelines=["skyrag", "naive"],
        backend=EchoBackend(responses=['{"python": 0.9}'] * 128),
        config=_config(),
        stores=_stores(),
        embedder=FixedEmbedder(),
        on_cell=events.append,
    )

    path = tmp_path / "bench.csv"
    export_benchmark_csv(metrics, path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    done = [e for e in events if e["event"] == "done"]

    assert len(rows) == len(done) == 8
    for event, row in zip(done, rows):
        assert row["model"] == event["model"]
        assert row["pipeline"] == event["pipeline"]
        assert row["query_id"] == event["query_id"]
        assert row["status"] == event["status"]
        assert int(row["retrieved_count"]) == event["retrieved"]
        assert int(row["cited_count"]) == event["cited"]
        assert float(row["selection_seconds"]) == event["selection_seconds"]
        assert float(row["generation_seconds"]) == event["generation_seconds"]
        assert float(row["elapsed_seconds"]) == event["elapsed_seconds"]


def test_the_per_stage_timings_reach_the_exported_csv(tmp_path):
    """`selection_seconds` and `expansion_seconds` were added and never asserted
    on the artifact. A stage timing that reaches no column cannot be reported.

    The backend is slowed deliberately. Asserting only that the column exists
    leaves this green when the value is dropped from the row -- the field has a
    default of 0.0, so an unexported stage timing is indistinguishable from a
    stage that took no time. That reversion was run and it passed; this is the
    version that fails it.
    """
    delay = 0.02
    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm-8b"],
        pipelines=["skyrag", "naive"],
        backend=SlowEchoBackend(delay, responses=['{"python": 0.9}'] * 64),
        config=_config(),
        stores=_stores(),
        embedder=FixedEmbedder(),
    )
    path = tmp_path / "bench.csv"
    export_benchmark_csv(metrics, path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))

    for row in rows:
        stages = (float(row["selection_seconds"]) + float(row["expansion_seconds"])
                  + float(row["generation_seconds"]))
        assert stages <= float(row["elapsed_seconds"]) + 1e-6, (
            "stage timings sum past the run that contains them"
        )
        # Each stage the row claims to time must carry the time the backend
        # actually spent in it.
        assert float(row["generation_seconds"]) >= delay

    skyrag_row = next(r for r in rows if r["pipeline"] == "skyrag")
    # One selection call per platform, one platform in these stores.
    assert float(skyrag_row["selection_seconds"]) >= delay
    # Query expansion and step-back: two calls on this pipeline.
    assert float(skyrag_row["expansion_seconds"]) >= 2 * delay

    naive = next(r for r in rows if r["pipeline"] == "naive")
    # The baseline runs step-back and no query expansion, so its expansion stage
    # is one call, not two. Asserted rather than assumed: the two pipelines' timed
    # regions cover different work and the column name does not say so.
    assert delay <= float(naive["expansion_seconds"]) < 2 * delay

    naive = next(r for r in rows if r["pipeline"] == "naive")
    # The baseline has no keyword stage at all, so this is 0.0 by construction and
    # not a stage that took no time.
    assert float(naive["selection_seconds"]) == 0.0


# --------------------------------------------------------------------------
# The default that could not run
# --------------------------------------------------------------------------

def test_the_default_model_list_can_actually_run(monkeypatch, tmp_path):
    """`mistralai/mistral-7b-instruct` has zero providers on OpenRouter, so the
    old default failed every arm carrying it -- as rows, not as a crash, which is
    worse: the CSV looks complete."""
    models = [m.strip() for m in DEFAULT_EVAL_MODELS.split(",")]

    assert "mistralai/mistral-7b-instruct" not in models
    assert models == [
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
    ]

    # And the parser uses it, rather than the constant existing beside a stale
    # literal.
    captured = _run_cli_capturing_the_sweep(monkeypatch, tmp_path)
    assert list(captured["models"]) == models
