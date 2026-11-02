"""Regression tests for the experiment-instrument work stream.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-18_dev-task_experiment-harness-controls.md

These cover the controls a benchmark run needs that the harness did not have:
per-role model pinning, a query set larger than one, repeated passes over the
grid, and a record on every row of how that row was produced.

Assertions are on the exported CSV or JSON wherever a defect could reach a
reported number, per the discipline the preceding backlogs established: a unit
test calling the fixed function directly stayed green in this repository while
the wiring feeding it was broken, three separate times.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.cli import load_query_set  # noqa: E402
from skyrag.config import Config  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    RunProvenance,
    export_benchmark_csv,
    export_benchmark_json,
    run_multi_model_eval,
)

DIM = 8


class FixedEmbedder:
    """A query vector in the stores' space. Ranking is uniform, which is fine:
    these tests assert on identity and provenance bookkeeping, not on ordering."""

    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _store(platform: str, *courses: Course) -> EmbeddingStore:
    return EmbeddingStore(
        courses=list(courses),
        matrix=np.ones((len(courses), DIM), dtype=np.float32),
    )


def _config(model: str = "stub") -> Config:
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, model)
    return cfg


def _stores() -> dict[str, EmbeddingStore]:
    return {
        "coursera": _store(
            "coursera",
            Course(id_="COU1", platform="coursera", title="Py", keywords=["python"]),
        )
    }


def _sweep(**kwargs):
    """A two-arm sweep on the echo backend. No network, no spend."""
    params = {
        "queries": ["learn python"],
        "models": ["arm-8b", "arm-70b"],
        "pipelines": ["skyrag"],
        "backend": EchoBackend(responses=['{"python": 0.9}'] * 64),
        "config": _config(),
        "stores": _stores(),
        "embedder": FixedEmbedder(),
    }
    params.update(kwargs)
    return run_multi_model_eval(**params)


def _rows(metrics, tmp_path) -> list[dict]:
    path = tmp_path / "bench.csv"
    export_benchmark_csv(metrics, path)
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


# --------------------------------------------------------------------------
# Per-role model pinning
# --------------------------------------------------------------------------

def test_roles_default_to_the_arms_own_model(tmp_path):
    """Passing no pins leaves the pre-existing whole-model comparison untouched.

    The four-arm specification wants exactly this shape -- one model per arm, used
    at every stage -- so the default must not change when pinning becomes possible.
    """
    rows = _rows(_sweep(), tmp_path)

    for row in rows:
        assert row["keyword_model"] == row["model"]
        assert row["expansion_model"] == row["model"]
        assert row["generation_model"] == row["model"]


def test_generation_can_be_pinned_while_the_selection_model_varies(tmp_path):
    """The condition grid's core requirement.

    "The final generation stage is identical in every cell -- same prompt, same
    model, one call. Only keyword selection varies." Without this the selection
    model and the generation model move together, so a difference in path quality
    between two cells cannot be attributed to either one.
    """
    rows = _rows(_sweep(generation_model="pinned-70b"), tmp_path)

    assert {r["generation_model"] for r in rows} == {"pinned-70b"}
    assert {r["keyword_model"] for r in rows} == {"arm-8b", "arm-70b"}
    assert {r["expansion_model"] for r in rows} == {"arm-8b", "arm-70b"}


def test_the_arm_label_survives_generation_pinning(tmp_path):
    """The axis under comparison must not vanish from the results table.

    `BenchmarkMetric.from_pipeline_result` takes `model` from `res.model`, which is
    whichever model served the *generation* call. That equals the arm label only
    while every role runs on the arm's model -- so with generation pinned, every row
    in the sweep reported the pinned model as its arm and the two cells became
    indistinguishable in the export.
    """
    rows = _rows(_sweep(generation_model="pinned-70b"), tmp_path)

    assert {r["model"] for r in rows} == {"arm-8b", "arm-70b"}


def test_keyword_selection_can_be_pinned_independently(tmp_path):
    """The mirror of the above: hold selection fixed and vary generation."""
    rows = _rows(_sweep(keyword_model="pinned-8b"), tmp_path)

    assert {r["keyword_model"] for r in rows} == {"pinned-8b"}
    assert {r["generation_model"] for r in rows} == {"arm-8b", "arm-70b"}


def test_an_empty_pin_means_no_pin(tmp_path):
    """An empty string is not a model id, and must not become one."""
    rows = _rows(_sweep(generation_model=""), tmp_path)

    for row in rows:
        assert row["generation_model"] == row["model"]


# --------------------------------------------------------------------------
# Repeats
# --------------------------------------------------------------------------

def test_repeats_run_the_whole_grid_again(tmp_path):
    """One run per cell gives a number with no measurable run-to-run spread.

    The outcome metrics ask for inter-run variance, and it cannot be recovered
    after the fact from a single row per cell.
    """
    rows = _rows(_sweep(repeats=3), tmp_path)

    assert len(rows) == 6                                   # 2 arms x 3 passes
    assert sorted({r["repeat"] for r in rows}) == ["1", "2", "3"]
    for rep in ("1", "2", "3"):
        assert {r["model"] for r in rows if r["repeat"] == rep} == {"arm-8b", "arm-70b"}


def test_a_repeat_is_a_whole_pass_not_a_repeated_cell(tmp_path):
    """Repeat is the outer loop, so a killed sweep leaves complete passes behind
    rather than a grid that is deeper in one corner than another."""
    metrics = _sweep(repeats=2)
    seen = [m.repeat for m in metrics]

    assert seen == [1, 1, 2, 2]


def test_repeats_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        _sweep(repeats=0)


# --------------------------------------------------------------------------
# Query identity
# --------------------------------------------------------------------------

def test_query_ids_pair_the_same_question_across_arms(tmp_path):
    rows = _rows(
        _sweep(queries=["learn python", "learn sql"], query_ids=["Q-PY", "Q-SQL"]),
        tmp_path,
    )

    assert {r["query_id"] for r in rows} == {"Q-PY", "Q-SQL"}
    paired = [r for r in rows if r["query_id"] == "Q-PY"]
    assert {r["model"] for r in paired} == {"arm-8b", "arm-70b"}


def test_query_ids_are_assigned_when_none_are_supplied(tmp_path):
    rows = _rows(_sweep(queries=["a", "b", "c"]), tmp_path)

    assert {r["query_id"] for r in rows} == {"q01", "q02", "q03"}


def test_mismatched_query_ids_are_refused():
    with pytest.raises(ValueError, match="one to one"):
        _sweep(queries=["a", "b"], query_ids=["only-one"])


# --------------------------------------------------------------------------
# The query set file
# --------------------------------------------------------------------------

def test_a_txt_query_set_skips_blanks_and_comments(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("# a comment\n\nlearn python\n  learn sql  \n", encoding="utf-8")

    queries, ids = load_query_set(path)

    assert queries == ["learn python", "learn sql"]
    assert ids == ["q01", "q02"]


def test_a_jsonl_query_set_carries_its_own_ids(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"id": "Q-PY", "query": "learn python"}\n'
        '{"id": "Q-SQL", "query": "learn sql"}\n',
        encoding="utf-8",
    )

    queries, ids = load_query_set(path)

    assert queries == ["learn python", "learn sql"]
    assert ids == ["Q-PY", "Q-SQL"]


def test_a_json_query_set_accepts_objects_or_plain_strings(tmp_path):
    objects = tmp_path / "objects.json"
    objects.write_text(json.dumps([{"id": "A", "query": "learn python"}]), encoding="utf-8")
    strings = tmp_path / "strings.json"
    strings.write_text(json.dumps(["learn python"]), encoding="utf-8")

    assert load_query_set(objects) == (["learn python"], ["A"])
    assert load_query_set(strings) == (["learn python"], ["q01"])


def test_duplicate_query_ids_are_refused(tmp_path):
    """Ids exist to pair rows across arms; duplicates silently mis-pair them."""
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"id": "SAME", "query": "learn python"}\n'
        '{"id": "SAME", "query": "learn sql"}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="duplicate query ids"):
        load_query_set(path)


def test_resolve_benchmark_output_dir_custom_path(tmp_path):
    from skyrag.cli import resolve_benchmark_output_dir

    custom = tmp_path / "my_custom_sweep"
    res = resolve_benchmark_output_dir(str(custom))
    assert res == custom
    assert res.exists()


def test_resolve_benchmark_output_dir_auto_format(tmp_path, monkeypatch):
    from datetime import datetime
    from skyrag.cli import resolve_benchmark_output_dir

    # The auto path creates `outputs/<name>/` relative to the working directory,
    # so without this the test leaves three empty run folders in the real outputs/.
    monkeypatch.chdir(tmp_path)

    date_stamp = datetime.now().strftime("%Y%m%d")

    # Multi-arm pilot auto-naming
    res = resolve_benchmark_output_dir("outputs", arms=["2", "3"], sweep_version="v2", query_count=10)
    assert res.name == f"pilot_v2_{date_stamp}_arm2-3_llama8b_10q"

    # Single arm sweep auto-naming with provider
    res2 = resolve_benchmark_output_dir(None, arm="3", sweep_version="v1", query_count=50, provider="deepinfra")
    assert res2.name == f"sweep_v1_{date_stamp}_arm3_llama8b_50q_deepinfra"

    # Single query smoke test with custom model
    res3 = resolve_benchmark_output_dir(
        "auto",
        arms=["2", "3"],
        model="qwen/qwen-2.5-7b-instruct",
        sweep_version="v2",
        query_count=1,
        provider="openrouter",
    )
    assert res3.name == f"smoke_v2_{date_stamp}_arm2-3_qwen7b_1q_openrouter"



def test_an_empty_or_missing_query_set_is_refused(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing but a comment\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="empty"):
        load_query_set(empty)
    with pytest.raises(SystemExit, match="not found"):
        load_query_set(tmp_path / "absent.txt")


def test_an_entry_without_a_query_field_names_the_entry(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"id": "A", "text": "wrong field"}]), encoding="utf-8")

    with pytest.raises(SystemExit, match="entry 1"):
        load_query_set(path)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def test_provenance_reaches_both_exported_artifacts(tmp_path):
    """The exports carried scores and nothing about how they were produced.

    No seed, temperature, backend, embedder, device, version or timestamp -- so two
    sweeps run under different settings produced byte-comparable artifacts with
    nothing to tell them apart.
    """
    config = _config()
    config.llm.temperature = 0.25
    config.llm.seed = 7
    provenance = RunProvenance.capture(config, "echo", run_id="RUN-1")

    metrics = _sweep(config=config, provenance=provenance)

    csv_rows = _rows(metrics, tmp_path)
    json_rows = json.loads(export_benchmark_json(metrics))

    for row in csv_rows:
        assert row["run_id"] == "RUN-1"
        assert row["seed"] == "7"
        assert row["temperature"] == "0.25"
        assert row["backend"] == "echo"
        assert row["embedding_model"] == config.embedding.model_name
        assert row["embedding_device"]              # resolved, never blank
        assert row["run_timestamp"]
        assert row["skyrag_version"]

    assert {r["run_id"] for r in json_rows} == {"RUN-1"}


def test_provenance_reaches_a_failed_arm_too(tmp_path):
    """The row that carries no measurement is the one someone will come back to
    asking which configuration produced it."""
    from skyrag.llm.base import LLMError
    from skyrag.pipelines.base import STATUS_CELL_FAILED

    class Failing:
        name = "failing"

        def generate(self, prompt, *, model=None, system=None, params=None):
            raise LLMError("503 upstream unavailable")

    provenance = RunProvenance.capture(_config(), "echo", run_id="RUN-2")
    rows = _rows(_sweep(backend=Failing(), provenance=provenance), tmp_path)

    failed = [r for r in rows if r["status"] == STATUS_CELL_FAILED]
    assert failed
    for row in failed:
        assert row["run_id"] == "RUN-2"
        assert row["query_id"]
        assert row["generation_model"]


def test_the_recorded_device_is_the_resolved_one():
    """`SKYRAG_DEVICE=auto` is not an answer to "which device ran this"."""
    provenance = RunProvenance.capture(_config(), "echo")

    assert provenance.embedding_device != "auto"
    assert provenance.embedding_device


def test_version_agrees_with_the_installed_distribution():
    """`__version__` read "0.1.0" while the distribution declared "0.2.0", so a
    stamped artifact recorded a version that never existed."""
    from importlib.metadata import version

    import skyrag

    assert skyrag.__version__ == version("skyrag")


# --------------------------------------------------------------------------
# Provider routing
# --------------------------------------------------------------------------

def test_a_pinned_provider_reaches_the_request_payload():
    """Routing was free, so part of the cost axis measured which company served
    the request rather than which model answered it."""
    from skyrag.llm.openrouter import OpenRouterBackend

    backend = OpenRouterBackend(api_key="k", provider="DeepInfra")
    _, payload = backend._build_payload("hello")

    assert payload["provider"] == {"order": ["DeepInfra"], "allow_fallbacks": False}


def test_an_unpinned_provider_sends_no_routing_block():
    """Automatic routing has to stay the default; an empty pin is not a pin."""
    from skyrag.llm.openrouter import OpenRouterBackend

    assert "provider" not in OpenRouterBackend(api_key="k")._build_payload("hi")[1]
    assert "provider" not in OpenRouterBackend(api_key="k", provider="")._build_payload("hi")[1]


def test_the_provider_pin_is_wired_from_config():
    """A field the backend accepts and `get_backend` never passes is a setting that
    appears to exist and does nothing -- the shape of four defects already recorded."""
    from skyrag.llm import get_backend

    config = _config()
    config.llm.backend = "openrouter"
    config.llm.openrouter_api_key = "k"
    config.llm.provider = "Together"

    assert get_backend(config.llm).provider == "Together"


# --------------------------------------------------------------------------
# Post-remediation sweep, findings 10 and 15
# --------------------------------------------------------------------------

def test_the_cited_id_pattern_is_derived_from_the_platform_constant():
    """Finding 10.

    The prefixes were spelled out inline in `cited_ids`, duplicating
    `PLATFORM_PREFIX`. A fourth platform added to the constant would not have
    reached the pattern, so its courses would be invisible to the grounding check:
    a run citing them reports zero citations, which suppresses its grounding score
    entirely, and any ID the model invented for that platform escapes the
    hallucination check written to catch it.
    """
    from skyrag.data.schema import PLATFORM_PREFIX
    from skyrag.pipelines.base import _CITED_ID_PATTERN

    for prefix in PLATFORM_PREFIX.values():
        assert prefix in _CITED_ID_PATTERN, f"{prefix} unreachable by the grounding check"


def test_a_new_platform_becomes_citable_without_touching_the_regex(monkeypatch):
    """The point of deriving it: adding a platform must be a one-line change.

    Asserts on the behaviour a new platform would get, not on the pattern string --
    a test reading the constant back would pass against the hardcoded version too.
    """
    import importlib

    from skyrag.data import schema

    monkeypatch.setitem(schema.PLATFORM_PREFIX, "futurelearn", "FUT")
    base = importlib.reload(importlib.import_module("skyrag.pipelines.base"))
    try:
        result = base.PipelineResult(
            pipeline="skyrag", query="q", learning_path="Start with FUT42, then COU1."
        )
        assert result.cited_ids() == ["FUT42", "COU1"]
    finally:
        monkeypatch.undo()
        importlib.reload(base)


def test_the_existing_platforms_still_match_exactly_as_before():
    """A refactor that changes behaviour is not a refactor."""
    from skyrag.pipelines.base import PipelineResult

    result = PipelineResult(
        pipeline="skyrag", query="q",
        learning_path="Take COU12, then edx99 and UDE3. Ignore FAKE7 and COU.",
    )

    assert result.cited_ids() == ["COU12", "EDX99", "UDE3"]


def test_the_dashboard_reuses_one_backend_rather_than_leaking_one_per_click():
    """Finding 15.

    The dashboard built an `OpenRouterBackend` per model per click and never closed
    it, leaking a connection pool each time -- the same defect fixed *inside* the
    backend and reintroduced from outside by constructing a new one per call site.

    Asserted on the source rather than by driving Streamlit: the leak is a
    construction-site property, and importing the app module starts a Streamlit
    runtime this suite has no business starting.
    """
    import re
    from pathlib import Path

    # Scans the whole `app/` tree. On 2026-09-10 the dashboard became a thin
    # router over `app/skyrag_ui/`, so reading only the entry point would let a
    # per-click backend construction reappear in a submodule with this guard
    # still green. The leak is a construction-site property wherever it sits.
    app_dir = Path(__file__).resolve().parent.parent / "app"
    if not app_dir.is_dir():
        # This release ships no app/; an empty rglob would pass vacuously.
        pytest.skip("app/ is not part of this checkout")
    blob = "\n".join(
        p.read_text(encoding="utf-8")
        for p in app_dir.rglob("*.py")
        if "__pycache__" not in p.parts
    )

    constructions = re.findall(r"OpenRouterBackend\(", blob)

    # AT MOST one, not exactly one. The read-only UI (Phase A) makes no inference
    # calls and constructs none; requiring one would force a backend into a page
    # that has no use for it. The invariant being protected is "never one per
    # click", which is violated by two or more, not by zero.
    assert len(constructions) <= 1, (
        f"{len(constructions)} OpenRouterBackend constructions across app/; at most one "
        "is allowed, inside a cached factory -- more than one is the per-click leak"
    )
    if constructions:
        assert "@st.cache_resource" in blob, (
            "app/ constructs an OpenRouterBackend outside a cached factory; "
            "Streamlit reruns the script on every interaction, so this leaks a "
            "connection pool per click"
        )


# --------------------------------------------------------------------------
# Seed handling across repeats
# --------------------------------------------------------------------------

def test_the_seed_is_fixed_across_passes_by_default(tmp_path):
    """The default is one seed for every pass and every arm.

    That is what fairness requires for the headline comparison: neither arm gets a
    luckier draw. Opting into variation must not happen by accident.
    """
    config = _config()
    config.llm.seed = 34
    provenance = RunProvenance.capture(config, "echo")
    rows = _rows(_sweep(config=config, repeats=3, provenance=provenance), tmp_path)

    assert {r["seed"] for r in rows} == {"34"}


def test_seed_per_repeat_walks_the_seed_with_the_pass(tmp_path):
    """Opt-in: pass 1 gets the base seed, pass 2 base+1, and so on."""
    config = _config()
    config.llm.seed = 34
    provenance = RunProvenance.capture(config, "echo")
    rows = _rows(
        _sweep(config=config, repeats=3, seed_per_repeat=True, provenance=provenance),
        tmp_path,
    )

    assert sorted({(r["repeat"], r["seed"]) for r in rows}) == [
        ("1", "34"), ("2", "35"), ("3", "36"),
    ]


def test_every_arm_within_a_pass_shares_one_seed(tmp_path):
    """The sequence is the same for every arm, so the arms stay paired.

    A seed that advanced per *arm* rather than per pass would hand one model a
    different draw from the other and quietly break the comparison.
    """
    config = _config()
    config.llm.seed = 34
    provenance = RunProvenance.capture(config, "echo")
    rows = _rows(
        _sweep(config=config, repeats=2, seed_per_repeat=True, provenance=provenance),
        tmp_path,
    )

    for rep in ("1", "2"):
        in_pass = [r for r in rows if r["repeat"] == rep]
        assert {r["model"] for r in in_pass} == {"arm-8b", "arm-70b"}
        assert len({r["seed"] for r in in_pass}) == 1


def test_the_recorded_seed_is_the_one_that_ran(tmp_path):
    """The provenance column has to name the seed in force for that row.

    Stamped once per run, it would claim the base seed for every row of passes 2
    and 3 -- a column that lies is worse than a column that is absent, because a
    reader has no way to tell.
    """
    config = _config()
    config.llm.seed = 100
    provenance = RunProvenance.capture(config, "echo")
    metrics = _sweep(config=config, repeats=2, seed_per_repeat=True, provenance=provenance)

    by_repeat = {m.repeat: m.seed for m in metrics}
    assert by_repeat == {1: 100, 2: 101}


def test_an_unseeded_run_is_not_given_one_by_walking(tmp_path):
    """`None` means unseeded; arithmetic on it must not invent a seed."""
    config = _config()
    config.llm.seed = None
    provenance = RunProvenance.capture(config, "echo")
    metrics = _sweep(config=config, repeats=2, seed_per_repeat=True, provenance=provenance)

    assert all(m.seed is None for m in metrics)
