"""B8 — findings 18 and 22 of the post-remediation sweep, closed with tests.

Both fixes existed in some form before this file and neither was a closed
finding: 18's `strict` branch was written and unreachable, and 22's guard
arrived incidentally during the harness work and owed a regression test. Under
this project's protocol an untested fix is not a closed finding.

**Finding 18** — `resolve_device(strict=True)` refuses a silent downgrade, and
`Embedder.model` resolves through `EmbeddingConfig.resolved_device()`, which
passed no `strict` and had no field through which one could be requested. So no
run could demand an accelerator and be refused, and `LAST_DEVICE_FALLBACK` --
the *reason* a downgrade happened -- was read by nothing outside `config.py` and
reached no artifact. Every run on this machine is CPU, so the defect cannot move
a number today; it becomes live the moment the work moves to a GPU machine,
which is when `elapsed_seconds` starts feeding the O4 cost axis with two
incomparable regimes and nothing saying which is which.

The backlog records a specific requirement for this test, because it is how the
defect survived the sweep that closed its sibling: **the regression must go
through `Embedder`, not call `resolve_device` directly.** A test that calls
`resolve_device(strict=True)` passes against the unfixed code -- that function
was never broken. What was broken is that nothing could reach it.

**Finding 22** — `--models ""` filtered to an empty list, `metrics_to_dataframe([])`
returned a frame with no columns, and the CLI's column select raised
`KeyError: "None of [Index([...])] are in the [columns]"` -- a pandas error for
a plain argument-validation problem.

Assertions on the exported CSV wherever the fix could reach a reported number,
per the discipline these backlogs establish: `embedding_device_fallback` is a
provenance column, and a provenance value that reaches only a module global is
the exact shape of the defect being closed.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyrag import config as config_mod  # noqa: E402
from skyrag.config import Config, EmbeddingConfig  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import Embedder, EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import export_benchmark_csv, run_multi_model_eval  # noqa: E402
from skyrag.pipelines.eval import RunProvenance  # noqa: E402

DIM = 8


class FixedEmbedder:
    """Stands in for the real embedder so no test needs torch to spin up a model."""

    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), DIM), dtype=np.float32)


def _stores() -> dict[str, EmbeddingStore]:
    out = {}
    for platform, prefix in (("coursera", "COU"), ("edx", "EDX")):
        courses = [
            Course(
                id_=f"{prefix}1",
                platform=platform,
                title="Intro Python",
                what_you_will_learn="python basics",
                keywords=["python"],
            )
        ]
        out[platform] = EmbeddingStore(
            courses=courses, matrix=np.ones((1, DIM), dtype=np.float32)
        )
    return out


def _config() -> Config:
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    return cfg


def _exported_row(cfg: Config) -> dict[str, str]:
    """One row of the real exported CSV, through the real sweep controller."""
    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["skyrag"],
        backend=EchoBackend(
            responses=["expanded", "principles", '{"python": 0.9}',
                       '{"python": 0.9}', "Step 1: COU1."] * 4
        ),
        config=cfg,
        stores=_stores(),
        embedder=FixedEmbedder(),
        provenance=RunProvenance.capture(cfg, "echo"),
    )
    return list(csv.DictReader(io.StringIO(export_benchmark_csv(metrics))))[0]


# --------------------------------------------------------------------------
# Finding 18 — a run can now demand an accelerator, and a downgrade is recorded
# --------------------------------------------------------------------------


def test_an_embedder_asked_for_an_absent_accelerator_strictly_refuses_to_run():
    """Through `Embedder`, which is the whole finding.

    `Embedder.model` is the only path that resolves a device for embedding, and
    it calls `resolved_device()` with whatever the config carries. Before the
    fix that call had no way to pass `strict`, so this raise was unreachable
    however the caller was configured -- the guard existed and could not be
    asked for.

    Touching `.model` is what forces the resolution. It raises before
    `SentenceTransformer` is imported, so the refusal does not depend on torch
    being installed or on which build of it is.
    """
    embedder = Embedder(EmbeddingConfig(device="cuda", strict_device=True))

    with pytest.raises(RuntimeError, match="Refusing to fall back to CPU"):
        _ = embedder.model


def test_the_strict_request_survives_the_environment_it_is_set_in(monkeypatch):
    """`SKYRAG_DEVICE_STRICT=1` is the route a GPU run will actually take.

    A field with no environment loader is reachable from Python and not from
    `.env`, which is finding 13's shape one field over: the type advertises an
    option the loader cannot express.
    """
    monkeypatch.setenv("SKYRAG_DEVICE_STRICT", "1")
    monkeypatch.setenv("SKYRAG_DEVICE", "cuda")

    with pytest.raises(RuntimeError, match="Refusing to fall back to CPU"):
        _ = Embedder(EmbeddingConfig()).model


def test_a_misspelled_strict_flag_is_refused_rather_than_read_as_true(monkeypatch):
    """`SKYRAG_DEVICE_STRICT=flase` must not switch strictness on.

    `bool(os.environ.get(...))` is the obvious spelling and every non-empty
    string is truthy under it, so `false` would enable the very refusal it asks
    to disable. A setting whose job is to refuse a silent downgrade may not
    itself be enabled by a misread value.
    """
    monkeypatch.setenv("SKYRAG_DEVICE_STRICT", "flase")

    with pytest.raises(ValueError, match="SKYRAG_DEVICE_STRICT"):
        EmbeddingConfig()


def test_a_false_flag_reads_as_false_and_not_as_a_non_empty_string(monkeypatch):
    monkeypatch.setenv("SKYRAG_DEVICE_STRICT", "false")
    assert EmbeddingConfig().strict_device is False

    monkeypatch.setenv("SKYRAG_DEVICE_STRICT", "on")
    assert EmbeddingConfig().strict_device is True


def test_a_downgrade_reason_reaches_the_exported_row(monkeypatch):
    """The half of finding 18 that touches an artifact.

    Non-strict is the default and stays the default, so a downgrade still
    happens -- what changes is that the run says so in its own export instead of
    leaving the reason in a module global nothing reads. Asserted on the CSV,
    not on `LAST_DEVICE_FALLBACK`, because reading the global back is the test
    that passes against the unfixed code.
    """
    monkeypatch.setenv("SKYRAG_DEVICE", "cuda")
    cfg = _config()
    cfg.embedding = EmbeddingConfig()

    row = _exported_row(cfg)

    assert "embedding_device_fallback" in row, (
        "the reason a run did not get the device it asked for reaches no column"
    )
    assert "torch reports no CUDA device" in row["embedding_device_fallback"]


def test_a_clean_resolution_exports_an_empty_reason_rather_than_a_stale_one():
    """An absent reason is the empty one, and it must not be another run's.

    Coverage of the exported value, and **not** the reversion evidence for the
    resolve-before-read rule -- see the test below for why that had to be
    written separately.
    """
    config_mod.LAST_DEVICE_FALLBACK = "a stale reason from some earlier call"

    cfg = _config()
    cfg.embedding = EmbeddingConfig(device="cpu")
    row = _exported_row(cfg)

    assert row["embedding_device_fallback"] == ""
    assert row["embedding_device"] == "cpu"


def test_the_fallback_reason_is_this_configs_and_not_the_previous_calls():
    """`device_fallback()` must resolve *first*, then read the global.

    **This test exists because the reversion above passed.** Deleting the
    `self.resolved_device()` line from `device_fallback()` -- so it reads
    `LAST_DEVICE_FALLBACK` cold -- left
    `test_a_clean_resolution_exports_an_empty_reason_rather_than_a_stale_one`
    green, because `RunProvenance.capture` resolves the device on the line
    immediately before it reads the reason. The neighbouring call cleared the
    global, so the export was correct whether or not this method did its own
    job, and an assertion on the artifact could not see the difference. That is
    the fifth time in this repository a test has passed against unfixed code,
    and the fourth where the mechanism was a value that happened to be right for
    a reason the test was not measuring.

    `device_fallback()` is a public method on a config object, so the hazard is
    real for every caller that is *not* `capture()` -- read it after any other
    resolution and it reports that resolution's answer. Poisoning the global
    through a genuine failed resolution, then asking a clean config, is the only
    arrangement where the two implementations differ.
    """
    # A real resolution that leaves a real reason behind, not a hand-set string:
    # this is exactly the state a preceding GPU probe puts the module in.
    EmbeddingConfig(device="cuda").resolved_device()
    assert config_mod.LAST_DEVICE_FALLBACK is not None, (
        "precondition: the global must be carrying another call's reason"
    )

    assert EmbeddingConfig(device="cpu").device_fallback() is None, (
        "a clean resolution reported the previous call's downgrade reason"
    )


def test_the_fallback_reason_reaches_a_failed_arm_too():
    """Stamped from provenance, so the row carrying no measurement carries it.

    An arm that died is exactly the row someone returns to asking what the run
    was configured as, and it is the row with no pipeline result to read a
    setting off.
    """
    prov = RunProvenance.capture(_config(), "echo")
    assert "embedding_device_fallback" in prov.as_columns()


# --------------------------------------------------------------------------
# Finding 22 — an empty --models is an argument error, not a pandas KeyError
# --------------------------------------------------------------------------


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Drive the real CLI in a subprocess, with its output directory sandboxed.

    `--output-dir` is redirected on purpose rather than as tidiness. These two
    tests assert that an empty `--models` stops *before* the sweep, and the way
    that assertion fails is that the sweep runs -- writing over
    `outputs/benchmark_eval.csv` and `.json`, real artifacts. A test whose
    failure mode is destroying the evidence it was written to protect is not an
    acceptable test, and this is verified rather than assumed: reverting the
    guard during the reversion pass did exactly that, twice.

    It is `--output-dir` and not `SKYRAG_OUTPUT_DIR`. The first attempt at this
    sandbox set the environment variable, which `PathConfig` reads -- and the
    eval exporter does not consult `PathConfig` at all: it takes the argparse
    flag, whose default is the literal `"outputs"`. The env var was accepted and
    ignored, so the sandbox looked correct and was not, which is the same shape
    as the defect this file's finding 18 closes.
    """
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.run(
        [sys.executable, "-m", "skyrag.cli", *args,
         "--output-dir", str(tmp_path / "outputs")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(SRC.parent),
    )


def test_an_empty_model_list_names_the_flag_instead_of_raising_from_pandas(tmp_path):
    """The fix is which error, not whether there is one.

    `--models ""` filtered to `[]`, `metrics_to_dataframe([])` returned a frame
    with no columns at all, and the CLI's column select raised
    `KeyError: "None of [Index([...])] are in the [columns]"`. A reader of that
    traceback is looking at pandas internals for what is a plain
    argument-validation problem, and nothing in the message names `--models`.
    """
    proc = _run_cli(tmp_path, "eval", "-m", "", "--backend", "echo")

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "--models" in combined, "the error must name the flag that caused it"
    assert "KeyError" not in combined, "a pandas KeyError is the defect, not the fix"
    assert "Index(" not in combined


def test_a_model_list_that_is_only_separators_is_empty_too(tmp_path):
    """`-m ",,,"` splits to four empty strings, all stripped away.

    The guard has to sit after the strip-and-filter, not on the raw string --
    `",,,"` is a non-empty argument that resolves to no models, which is exactly
    the state that reached pandas.
    """
    proc = _run_cli(tmp_path, "eval", "-m", ",,,", "--backend", "echo")

    assert proc.returncode != 0
    assert "--models" in proc.stdout + proc.stderr


def test_an_unknown_pipeline_is_refused_before_any_cell_is_billed(tmp_path):
    """`--pipelines` validates up front, not inside the sweep loop.

    `run_multi_model_eval` already raises `ValueError: Unknown pipeline` -- but
    it raises it *per cell*, after the cells before it have run. On a live
    backend a typo in the last of three pipeline names would therefore be
    reported only after a third of the sweep had been paid for, and reported as
    a traceback rather than as an argument error. The guard belongs beside the
    `--models` one above, for the same reason.
    """
    proc = _run_cli(tmp_path, "eval", "-P", "skyrag,nieve", "--backend", "echo")

    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "--pipelines" in combined, "the error must name the flag that caused it"
    assert "Traceback" not in combined


def test_restricting_to_one_pipeline_runs_only_that_pipeline(tmp_path):
    """The flag exists to stop paying for `naive` rows on an arm sweep.

    Per D3 only `skyrag` is an arm, so an arm sweep that leaves the default in
    place buys a second generation call per query and then filters those rows
    out of the results table. Asserted on the exported CSV rather than on the
    terminal line, because the CSV is what the results table is built from.
    """
    proc = _run_cli(
        tmp_path, "eval", "-P", "skyrag", "--backend", "echo",
        "-q", "learn data science", "-m", "echo-model",
        # A fresh clone has no embeddings/, and the B24 guard would refuse.
        "--allow-synthetic-corpus",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = list(csv.DictReader(
        (tmp_path / "outputs" / "benchmark_eval.csv").read_text(encoding="utf-8").splitlines()
    ))
    assert rows, "the sweep exported no rows"
    assert {r["pipeline"] for r in rows} == {"skyrag"}
