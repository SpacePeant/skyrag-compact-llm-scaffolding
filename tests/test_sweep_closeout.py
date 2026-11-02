"""Regression tests for the six findings that closed the post-remediation sweep.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-17_bugfix-post-remediation-project-sweep.md
  findings 9, 13, 20, 21, 23, 25

None of the six can alter a reported number -- that was established when they
were triaged, and closing them does not change it. What they alter is what
happens to a person: a diagnostic that names the wrong file, a checkpoint that
silently re-pays for a catalog, a documented rule that was never in force.

Each test below was shown to fail with only its own fix reverted, with one
exception that is called out where it appears: finding 9 is a documentation
defect over dead code, so its behavioural test passes against the unfixed code
too, and the reversion evidence rests on the docstring assertion beside it. That
is the same shape as finding 10 in the preceding round, and it is recorded here
rather than left for a reader to discover by reverting and seeing green.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.config import Config, LLMConfig, RetrievalConfig  # noqa: E402
from skyrag.data.keywords import generate_keywords  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import (  # noqa: E402
    CACHE_CORRUPT,
    CACHE_OK,
    EmbeddingStore,
    inspect_cache,
)
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    RunProvenance,
    export_benchmark_csv,
    export_benchmark_json,
    run_multi_model_eval,
)
from skyrag.retrieval import keyword_buckets, score_all  # noqa: E402

DIM = 8


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _store(*courses: Course) -> EmbeddingStore:
    return EmbeddingStore(
        courses=list(courses),
        matrix=np.ones((len(courses), DIM), dtype=np.float32),
    )


# --------------------------------------------------------------------------
# Finding 9 -- keyword attribution: the documented rule is the rule in force
# --------------------------------------------------------------------------

def test_a_course_is_attributed_to_the_first_keyword_that_reaches_it():
    """Attribution follows the order of `selected_keywords`, not the score.

    Recorded honestly: this test passes against the unfixed code as well. The
    `score > best[i].score` comparison that finding 9 removed could never fire --
    scores are course-level, so a course carries the same score under every
    keyword -- which is exactly why the docstring and the behaviour had drifted
    apart with nothing failing. It is kept because it pins the contract that the
    corrected docstring now states, and it would fail if someone later made
    attribution genuinely score-dependent.
    """
    course = Course(
        id_="COU1", platform="coursera", title="Py", keywords=["python", "ml"]
    )
    store = _store(course)
    query = np.ones(DIM, dtype=np.float32)

    forward = keyword_buckets(query, store, ["python", "ml"])
    reverse = keyword_buckets(query, store, ["ml", "python"])

    assert [r.via_keyword for r in forward] == ["python"]
    assert [r.via_keyword for r in reverse] == ["ml"]


def test_the_attribution_rule_in_the_docstring_is_the_rule_in_force():
    """The reversion evidence for finding 9.

    `via_keyword` is rendered in the Markdown report as the explanation of *why*
    a course was retrieved, so a docstring promising "the keyword where it scored
    best" was a claim about the output a reader sees. Asserting on prose is
    unusual and is done deliberately: the defect was in the prose.
    """
    doc = keyword_buckets.__doc__ or ""
    assert "scored best" not in doc.replace("\n", " ")
    assert "first" in doc.lower()


# --------------------------------------------------------------------------
# Finding 13 -- an unseeded run can be requested from the environment
# --------------------------------------------------------------------------

def test_an_unset_seed_still_takes_the_default(monkeypatch):
    monkeypatch.delenv("SKYRAG_SEED", raising=False)
    assert LLMConfig().seed == 34


def test_an_explicit_seed_is_still_read(monkeypatch):
    monkeypatch.setenv("SKYRAG_SEED", "7")
    assert LLMConfig().seed == 7


@pytest.mark.parametrize("value", ["", "none", "NULL", " off "])
def test_a_seed_switched_off_in_the_environment_reaches_the_config_as_none(
    monkeypatch, value
):
    """`seed` is typed `int | None` and every backend sends no seed for None.

    `_env_int` returned the default for an empty variable and for an absent one
    alike, so the None half of the field's own type had no environment route.
    """
    monkeypatch.setenv("SKYRAG_SEED", value)
    assert LLMConfig().seed is None


def test_an_unseeded_run_says_so_in_its_exported_rows(tmp_path, monkeypatch):
    """The artifact-level half: a run with no seed must not export a seed.

    Absent and empty are kept distinct in the loader, and the safety net for a
    variable emptied by accident is that the provenance stamp carries the seed
    that actually ran onto every row -- so the export says "no seed" rather than
    leaving it to be discovered by re-reading `.env` months later.
    """
    monkeypatch.setenv("SKYRAG_SEED", "")
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    assert cfg.llm.seed is None

    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm-8b"],
        pipelines=["skyrag"],
        backend=EchoBackend(responses=['{"python": 0.9}'] * 32),
        config=cfg,
        stores={"coursera": _store(
            Course(id_="COU1", platform="coursera", title="Py", keywords=["python"])
        )},
        embedder=FixedEmbedder(),
        provenance=RunProvenance.capture(cfg, "echo"),
    )

    csv_path, json_path = tmp_path / "b.csv", tmp_path / "b.json"
    export_benchmark_csv(metrics, csv_path)
    export_benchmark_json(metrics, json_path)

    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert rows and all(row["seed"] == "" for row in rows)
    assert all(row["seed"] is None for row in json.loads(json_path.read_text("utf-8")))


# --------------------------------------------------------------------------
# Finding 23 -- a mistyped numeric setting names itself
# --------------------------------------------------------------------------

def test_a_mistyped_integer_setting_names_the_variable_and_the_value(monkeypatch):
    monkeypatch.setenv("SKYRAG_TOP_K", "twenty")
    with pytest.raises(ValueError) as excinfo:
        RetrievalConfig()

    message = str(excinfo.value)
    assert "SKYRAG_TOP_K" in message
    assert "twenty" in message
    assert ".env" in message
    # The bare failure this replaced. Seven settings are read the same way, so a
    # message naming none of them left the reader with seven candidates.
    assert "invalid literal for int()" not in message


def test_a_mistyped_float_setting_names_the_variable_and_the_value(monkeypatch):
    monkeypatch.setenv("SKYRAG_TEMPERATURE", "warm")
    with pytest.raises(ValueError) as excinfo:
        LLMConfig()

    message = str(excinfo.value)
    assert "SKYRAG_TEMPERATURE" in message
    assert "warm" in message


def test_an_off_literal_is_still_an_error_for_a_setting_that_cannot_be_off(monkeypatch):
    """`_OFF_LITERALS` belongs to optional settings only.

    There is no such thing as a run with no top-k, so `SKYRAG_TOP_K=none` is a
    typo and has to be reported as one rather than quietly meaning something.
    """
    monkeypatch.setenv("SKYRAG_TOP_K", "none")
    with pytest.raises(ValueError, match="SKYRAG_TOP_K"):
        RetrievalConfig()


# --------------------------------------------------------------------------
# Finding 20 -- the keyword checkpoint survives a catalog that changed size
# --------------------------------------------------------------------------

def _coursera(title: str, keyword: str = "") -> dict:
    return {"Course Title": title, "What you will learn": f"about {title}", "Keyword": keyword}


def _checkpoint(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def test_a_grown_catalog_resumes_instead_of_re_paying(tmp_path, capsys):
    """One course added must cost one call, not a whole catalog.

    The guard was `if len(existing) == len(records)`, and on a mismatch the
    checkpoint was not merely ignored -- the fresh list was written over it. At
    one LLM call per course over 6,641 courses this is the most expensive silent
    failure in the project.
    """
    out = tmp_path / "coursera.json"
    _checkpoint(out, [_coursera(t, "python, data") for t in ("A", "B", "C")])

    catalog = [_coursera(t) for t in ("A", "B", "C", "D")]
    backend = EchoBackend(responses=["python, data science"] * 8)

    result = generate_keywords(
        backend, catalog, "coursera", output_path=out, progress=False, workers=1
    )

    assert len(backend.calls) == 1, "only the new course should have been extracted"
    assert [r["Course Title"] for r in result] == ["A", "B", "C", "D"]
    assert [r["Keyword"] for r in result[:3]] == ["python, data"] * 3
    assert result[3]["Keyword"] == "python, data science"
    assert "resumed 3 by identity" in capsys.readouterr().out


def test_an_unchanged_catalog_still_resumes_wholesale(tmp_path):
    """The pre-existing path is untouched: equal counts adopt the checkpoint."""
    out = tmp_path / "coursera.json"
    _checkpoint(out, [_coursera(t, "python, data") for t in ("A", "B", "C")])

    backend = EchoBackend(responses=["python, data science"] * 8)
    generate_keywords(
        backend,
        [_coursera(t) for t in ("A", "B", "C")],
        "coursera",
        output_path=out,
        progress=False,
        workers=1,
    )

    assert backend.calls == []


def test_a_checkpoint_matching_nothing_raises_rather_than_re_paying(tmp_path):
    """A total mismatch is a wrong-file mistake, not an incremental change.

    Continuing would re-pay for the catalog and then overwrite the checkpoint
    with the result, destroying the record of what had already been bought.
    """
    out = tmp_path / "coursera.json"
    paid = [_coursera(t, "python, data") for t in ("A", "B", "C")]
    _checkpoint(out, paid)
    before = out.read_text(encoding="utf-8")

    backend = EchoBackend(responses=["python, data science"] * 8)
    with pytest.raises(ValueError, match="none of which matches"):
        generate_keywords(
            backend,
            [_coursera(t) for t in ("X", "Y", "Z", "W")],
            "coursera",
            output_path=out,
            progress=False,
            workers=1,
        )

    assert backend.calls == [], "nothing may be paid for before the refusal"
    assert out.read_text(encoding="utf-8") == before, "the checkpoint must survive"


def test_overwrite_still_discards_the_checkpoint_deliberately(tmp_path):
    """The escape hatch the refusal points at has to work."""
    out = tmp_path / "coursera.json"
    _checkpoint(out, [_coursera(t, "python, data") for t in ("A", "B", "C")])

    backend = EchoBackend(responses=["python, data science"] * 8)
    generate_keywords(
        backend,
        [_coursera(t) for t in ("X", "Y")],
        "coursera",
        output_path=out,
        progress=False,
        workers=1,
        overwrite=True,
    )

    assert len(backend.calls) == 2


# --------------------------------------------------------------------------
# Finding 21 -- a row-count-inconsistent cache is corrupt, not ok
# --------------------------------------------------------------------------

def _write_cache(directory: Path, name: str, rows: int) -> None:
    courses = [
        Course(id_=f"C{i}", platform=name, title=f"t{i}", keywords=["python"])
        for i in range(rows)
    ]
    EmbeddingStore(
        courses=courses, matrix=np.ones((rows, DIM), dtype=np.float32)
    ).save(directory, name, model_name="test-embedder")


def test_a_healthy_cache_is_still_reported_ok(tmp_path):
    _write_cache(tmp_path, "coursera", 3)
    status = inspect_cache(tmp_path, "coursera", model_name="test-embedder")
    assert status.state == CACHE_OK and bool(status)


def test_a_truncated_metadata_file_is_reported_corrupt(tmp_path):
    """`CACHE_CORRUPT` means "manifest disagrees with the matrix beside it".

    `inspect_cache` compared dimensionality only, so this cache returned `ok` --
    readable as usable by every caller -- and then raised out of
    `EmbeddingStore.load` several frames later. The eval CLI branches on
    `if status:` and builds its disclosed synthetic placeholder in the `else`, so
    a corrupt cache took the *success* branch and killed the process instead of
    degrading with a stated reason.
    """
    _write_cache(tmp_path, "coursera", 3)
    jsonl = tmp_path / "coursera.jsonl"
    first = jsonl.read_text(encoding="utf-8").splitlines()[0]
    jsonl.write_text(first + "\n", encoding="utf-8")

    status = inspect_cache(tmp_path, "coursera", model_name="test-embedder")
    assert status.state == CACHE_CORRUPT
    assert not status
    assert "1" in status.detail and "3" in status.detail


def test_a_matrix_shorter_than_its_manifest_is_reported_corrupt(tmp_path):
    _write_cache(tmp_path, "coursera", 3)
    np.save(tmp_path / "coursera.npy", np.ones((2, DIM), dtype=np.float32))

    status = inspect_cache(tmp_path, "coursera", model_name="test-embedder")
    assert status.state == CACHE_CORRUPT


def test_a_corrupt_cache_does_not_claim_to_exist(tmp_path):
    """`exists()` is what callers branch on before deciding to re-embed."""
    _write_cache(tmp_path, "coursera", 3)
    jsonl = tmp_path / "coursera.jsonl"
    jsonl.write_text(jsonl.read_text(encoding="utf-8").splitlines()[0] + "\n", "utf-8")

    assert EmbeddingStore.exists(tmp_path, "coursera", model_name="test-embedder") is False


# --------------------------------------------------------------------------
# Finding 25 -- a dead-vector diagnostic names the side that died
# --------------------------------------------------------------------------

def test_a_dead_query_vector_blames_the_query_encoder(tmp_path):
    """The guard was right; the instruction pointed at the wrong file.

    A caller whose query encoder returned nothing was told to "re-embed the
    affected rows" -- untrue and unactionable, because there are no rows to
    re-embed and the cache is fine.
    """
    store = _store(Course(id_="COU1", platform="coursera", title="Py"))

    with pytest.raises(ValueError) as excinfo:
        score_all(np.zeros(DIM, dtype=np.float32), store)

    message = str(excinfo.value)
    assert "query vector" in message
    assert "query encoder" in message
    assert "Re-embed the affected rows" not in message


def test_a_dead_corpus_row_still_says_re_embed():
    """The other side keeps the instruction that is correct for it."""
    store = EmbeddingStore(
        courses=[
            Course(id_="COU1", platform="coursera", title="a"),
            Course(id_="COU2", platform="coursera", title="b"),
        ],
        matrix=np.vstack(
            [np.ones(DIM, dtype=np.float32), np.zeros(DIM, dtype=np.float32)]
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        score_all(np.ones(DIM, dtype=np.float32), store)

    message = str(excinfo.value)
    assert "corpus row" in message
    assert "Re-embed the affected rows" in message
