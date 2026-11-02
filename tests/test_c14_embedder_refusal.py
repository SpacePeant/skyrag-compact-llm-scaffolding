"""C14/C15: a run that cannot embed must refuse, and a CSV must not double its
line terminator.

Cross-reference:
- project_knowledge/task_log/backlog_2026-09-23_experiment_stamped-factorial-and-the-silent-embedder-fallback.md
- project_knowledge/task_log/backlog_2026-09-24_dev-task_candidate-frequency-corrections-and-the-half-of-c-prime-that-works.md

**C14.** On 2026-09-23, lane `i_l0` of the stamped 2x3 factorial lost torch to
`OSError: [WinError 1455]` under three-way concurrency. The harness substituted
a constant query vector, scored every course identically, and ran the cell to
completion: 50 well-formed rows, no cell error, no circuit-breaker trip, exit 0.
The only trace was one line on stdout. The directory survives as
`outputs/INVALID_embedder-fallback_fact_20260923_v5fp8_i_l0_r1/` because a human
re-read a lane log, which is not a control.

Two properties are asserted, not one, for the same reason B24 needed both:

1. the run **refuses** rather than degrading, before any money is spent;
2. the one path that still reaches a row -- an operator who typed
   `--allow-degraded-embedder` -- **stamps the row**, so the artifact carries its
   own disqualification instead of depending on a log that may not travel with
   it.

A test of (1) alone would stay green through a regression that silently dropped
the stamp, and the stamp is the half that survives being copied to another
machine.

**C15** rides along because it is the same class of defect one layer down: an
export that looks fine to every reader that tolerates it. `to_csv` already
terminates rows with `os.linesep`, and both writers then handed the result to a
text-mode handle that translated the newline again, so every
`benchmark_eval.csv` on this machine carries CR CR LF.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import skyrag.embedding as embedding_pkg  # noqa: E402
import skyrag.pipelines as pipelines_pkg  # noqa: E402
from skyrag.cli import run_eval_cli  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import CACHE_OK, CacheStatus, EmbeddingStore  # noqa: E402
from skyrag.pipelines.eval import (  # noqa: E402
    BenchmarkMetric,
    RunProvenance,
    _append_metric_to_csv,
    export_benchmark_csv,
)

DIM = 8

# The exception the real failure raised, verbatim enough to be recognisable in
# the refusal text. The message matters: WinError 1455 is the Windows commit
# limit, and an operator who reads "out of memory" will free RAM and try again.
WINERROR_1455 = OSError(
    "[WinError 1455] The paging file is too small for this operation to complete"
)


class _StopHere(Exception):
    """Ends the CLI at the sweep call, so no cell runs and nothing is exported."""


def _a_real_store() -> EmbeddingStore:
    return EmbeddingStore(
        courses=[Course(id_="COU1", platform="coursera", title="Py",
                        keywords=["python"])],
        matrix=np.ones((1, DIM), dtype=np.float32),
    )


def _pretend_a_real_cache_is_on_disk(monkeypatch) -> None:
    """Make `has_real_caches` true without writing a cache.

    The embedder branch under test is reached only when at least one real cache
    loaded -- which is correct, and is why the synthetic-corpus tests elsewhere
    never touch it. Stubbing the inspection is narrower than building a cache on
    disk and keeps this file's subject to the one branch it is about.
    """
    monkeypatch.setattr(
        embedding_pkg,
        "inspect_cache",
        lambda directory, name, *, model_name: CacheStatus(CACHE_OK, "stubbed"),
    )
    monkeypatch.setattr(
        embedding_pkg.EmbeddingStore,
        "load",
        classmethod(lambda cls, directory, name, *, expect_model: _a_real_store()),
    )


def _break_the_embedder(monkeypatch, exc: Exception = WINERROR_1455) -> None:
    class Exploding:
        def __init__(self, *args, **kwargs):
            raise exc

    monkeypatch.setattr(embedding_pkg, "Embedder", Exploding)


def _working_embedder(monkeypatch) -> None:
    class Working:
        def __init__(self, *args, **kwargs):
            pass

        def encode_one(self, text: str) -> np.ndarray:
            return np.ones(DIM, dtype=np.float32)

    monkeypatch.setattr(embedding_pkg, "Embedder", Working)


def _run_cli_capturing_the_sweep(monkeypatch, tmp_path, extra_args=()) -> dict:
    captured: dict = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        raise _StopHere

    monkeypatch.setattr(pipelines_pkg, "run_multi_model_eval", recorder)
    args = [
        "-b", "echo", "-q", "learn python", "-o", str(tmp_path),
        *extra_args,
    ]
    with pytest.raises(_StopHere):
        run_eval_cli(args)
    return captured


# --------------------------------------------------------------------------
# C14, property 1: it refuses
# --------------------------------------------------------------------------

def test_a_run_whose_embedder_will_not_load_is_refused(monkeypatch, tmp_path):
    """The defect exactly: this run used to reach the sweep and spend money."""
    _pretend_a_real_cache_is_on_disk(monkeypatch)
    _break_the_embedder(monkeypatch)

    def recorder(**kwargs):
        raise AssertionError(
            "the sweep was reached with a constant query vector -- this is the "
            "2026-09-23 i_l0 failure, which exits 0 and writes a publishable-"
            "looking directory"
        )

    monkeypatch.setattr(pipelines_pkg, "run_multi_model_eval", recorder)

    with pytest.raises(SystemExit) as excinfo:
        run_eval_cli(["-b", "echo", "-q", "learn python", "-o", str(tmp_path)])

    message = str(excinfo.value)
    assert "REFUSED" in message
    assert "C14" in message, "the refusal must name the finding it closes"
    assert "constant query vector" in message, (
        "the refusal must say what would have been wrong with the numbers, not "
        "merely that something failed"
    )


def test_the_refusal_repeats_the_underlying_error_and_names_the_way_out(
    monkeypatch, tmp_path
):
    """A refusal that hides the cause makes the operator re-run to find it."""
    _pretend_a_real_cache_is_on_disk(monkeypatch)
    _break_the_embedder(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        run_eval_cli(["-b", "echo", "-q", "learn python", "-o", str(tmp_path)])

    message = str(excinfo.value)
    assert "WinError 1455" in message
    assert "--allow-degraded-embedder" in message
    assert "fewer concurrent lanes" in message, (
        "WinError 1455 is the commit limit, so the actionable remedy is "
        "concurrency, not RAM"
    )


def test_the_run_still_prints_the_banner_the_launchers_grep_for(
    monkeypatch, tmp_path, capsys
):
    """`launch_injection_lanes.sh` greps lane logs for this exact string.

    The refusal must not replace the banner: a lane launcher auditing logs after
    the fact still has to be able to find a degraded run, and the two defences
    are independent.
    """
    _pretend_a_real_cache_is_on_disk(monkeypatch)
    _break_the_embedder(monkeypatch)

    with pytest.raises(SystemExit):
        run_eval_cli(["-b", "echo", "-q", "learn python", "-o", str(tmp_path)])

    assert "NOT VALID AS RESULTS" in capsys.readouterr().out


def test_a_missing_cache_is_still_b24_and_not_this(monkeypatch, tmp_path):
    """The embedder branch must stay behind `has_real_caches`.

    With no real cache the corpus itself is the problem and B24 owns the
    refusal. Asserting this keeps C14 from quietly widening into a second,
    differently-worded refusal for a case that already has one.
    """
    monkeypatch.setenv("SKYRAG_EMBEDDINGS_DIR", "no-such-embeddings-dir")
    _break_the_embedder(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        run_eval_cli(["-b", "echo", "-q", "learn python", "-o", str(tmp_path)])

    message = str(excinfo.value)
    assert "B24" in message
    assert "C14" not in message


# --------------------------------------------------------------------------
# C14, property 2: the allowed path stamps the row
# --------------------------------------------------------------------------

def test_the_typed_escape_hatch_lets_the_run_through(monkeypatch, tmp_path):
    _pretend_a_real_cache_is_on_disk(monkeypatch)
    _break_the_embedder(monkeypatch)

    captured = _run_cli_capturing_the_sweep(
        monkeypatch, tmp_path, ["--allow-degraded-embedder"]
    )

    assert captured, "the flag must reach the sweep, as --allow-synthetic-corpus does"


def test_an_allowed_degraded_run_stamps_every_row(monkeypatch, tmp_path):
    """The half that survives being copied away from its log."""
    _pretend_a_real_cache_is_on_disk(monkeypatch)
    _break_the_embedder(monkeypatch)

    captured = _run_cli_capturing_the_sweep(
        monkeypatch, tmp_path, ["--allow-degraded-embedder"]
    )
    provenance = captured["provenance"]

    assert provenance.embedder_degraded is True
    assert "WinError 1455" in provenance.embedder_degraded_reason, (
        "the reason is the column that tells a later reader whether this was a "
        "deliberate smoke test or a lane that lost its model"
    )


def test_a_clean_run_stamps_not_degraded(monkeypatch, tmp_path):
    """The negative case, so the column cannot be True by construction."""
    _pretend_a_real_cache_is_on_disk(monkeypatch)
    _working_embedder(monkeypatch)

    captured = _run_cli_capturing_the_sweep(monkeypatch, tmp_path)
    provenance = captured["provenance"]

    assert provenance.embedder_degraded is False
    assert provenance.embedder_degraded_reason == ""


def test_the_stamp_is_a_real_boolean_not_an_empty_string(tmp_path):
    """C10's defect class, avoided deliberately.

    A single string column holding "" for a clean run reads as falsy in Python
    but comes back from `read_csv` as NaN, and `bool(nan)` is True -- so a filter
    for degraded rows would select every clean one instead. The boolean and the
    reason are therefore separate columns, and this pins that apart.
    """
    clean = RunProvenance(
        run_id="r", run_timestamp="t", skyrag_version="v", backend="echo",
        provider="", temperature=0.0, seed=1, max_tokens=1,
        embedding_model="m", embedding_device="cpu",
    )
    columns = clean.as_columns()

    assert columns["embedder_degraded"] is False
    assert isinstance(columns["embedder_degraded"], bool)
    assert columns["embedder_degraded_reason"] == ""


def test_the_stamp_reaches_the_exported_csv(tmp_path):
    """A provenance field that does not become a column is not a record."""
    from dataclasses import replace

    degraded = RunProvenance(
        run_id="r", run_timestamp="t", skyrag_version="v", backend="echo",
        provider="", temperature=0.0, seed=1, max_tokens=1,
        embedding_model="m", embedding_device="cpu",
        embedder_degraded=True,
        embedder_degraded_reason="OSError: [WinError 1455]",
    )
    metric = replace(
        BenchmarkMetric(model="m", pipeline="p", query="q", elapsed_seconds=1.0),
        **degraded.as_columns(),
    )

    csv_path = tmp_path / "benchmark_eval.csv"
    export_benchmark_csv([metric], csv_path)
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    assert rows[0]["embedder_degraded"] == "True"
    assert "WinError 1455" in rows[0]["embedder_degraded_reason"]


def test_provenance_columns_can_be_splatted_onto_a_metric():
    """`_stamp` does `replace(metric, **provenance.as_columns())`.

    Any provenance field without a matching `BenchmarkMetric` field raises
    TypeError there, at sweep time, after the spend. This is that check, for
    free, before it.
    """
    from dataclasses import replace

    provenance = RunProvenance(
        run_id="r", run_timestamp="t", skyrag_version="v", backend="echo",
        provider="", temperature=0.0, seed=1, max_tokens=1,
        embedding_model="m", embedding_device="cpu",
    )
    stamped = replace(
        BenchmarkMetric(model="m", pipeline="p", query="q", elapsed_seconds=1.0),
        **provenance.as_columns(),
    )
    assert stamped.run_id == "r"


# --------------------------------------------------------------------------
# C15: the line terminator
# --------------------------------------------------------------------------

def _metric() -> BenchmarkMetric:
    return BenchmarkMetric(model="m", pipeline="p", query="q", elapsed_seconds=1.0)


def test_the_full_export_does_not_double_the_line_terminator(tmp_path):
    path = tmp_path / "benchmark_eval.csv"
    export_benchmark_csv([_metric(), _metric()], path)
    raw = path.read_bytes()

    assert b"\r\r\n" not in raw, (
        "every committed benchmark_eval.csv carries CR CR LF and renders with a "
        "blank line between rows"
    )
    assert raw.count(os.linesep.encode()) == 3, "header plus two rows, each terminated once"


def test_the_incremental_export_does_not_double_the_line_terminator(tmp_path):
    """The appending writer had the same defect, and C15 named only the other.

    This is the writer a run uses while it is still going, so a sweep killed
    part-way through left the doubling in the rows that survived.
    """
    path = tmp_path / "benchmark_eval.csv"
    _append_metric_to_csv(_metric(), path)
    _append_metric_to_csv(_metric(), path)
    raw = path.read_bytes()

    assert b"\r\r\n" not in raw
    assert raw.count(os.linesep.encode()) == 3


def test_a_csv_with_one_row_reads_back_as_one_row(tmp_path):
    """The defect's actual consequence, stated as behaviour rather than bytes."""
    path = tmp_path / "benchmark_eval.csv"
    export_benchmark_csv([_metric()], path)

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
