"""Regression tests for the unbounded generation prompt.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-18_bugfix_live-run-context-overflow.md

The defect these cover is not an outage. `max_keywords` x `per_keyword` x
platforms admitted 560 courses at a mean 801-token `Course.render()`, so the
generation prompt reached ~441,000 tokens against a 131,072-token endpoint limit
and was refused with HTTP 400. The arm that selected valid keywords on both
platforms retrieved more courses, overflowed and returned `cell_failed`; the cell
that hallucinated keywords retrieved less and passed. Selection quality was
inversely correlated with completing the run, which inverts the headline
comparison by a mechanism unrelated to either model's ability.

Two properties have to hold, and they fail differently:

  1. the *prompt* must be bounded -- covered by the render tests;
  2. `Course.render()` must NOT be what got bounded -- covered by
     `test_render_is_left_alone_because_the_embedder_reads_it`. `render()` feeds
     `Embedder.embed_courses`, and `CacheManifest` records the model name,
     dimension and row count but no render format, so a shortened `render()`
     invalidates every cached matrix on disk while every check in this repository
     goes on passing. That is the one failure mode with no detector anywhere else.

Assertions are on the exported CSV/JSON wherever the defect could reach a
reported number, per the discipline the preceding backlogs established.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.config import Config  # noqa: E402
from skyrag.data.schema import (  # noqa: E402
    CONTEXT_MAX_CHARS,
    CONTEXT_SUMMARY_CHARS,
    Course,
)
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    RunProvenance,
    export_benchmark_csv,
    export_benchmark_json,
    run_multi_model_eval,
)
from skyrag.pipelines.base import to_markdown  # noqa: E402
from skyrag.pipelines.skyrag import run_skyrag  # noqa: E402
from skyrag.prompts import build_generation_prompt  # noqa: E402
from skyrag.retrieval import (  # noqa: E402
    RetrievedCourse,
    cap_context,
    format_context,
    merge_platforms,
)

DIM = 8

# The endpoint limit the live sweep was refused by, and what is left for the input
# once `max_tokens` is reserved for the reply.
ENDPOINT_LIMIT = 131_072

# chars/4 as the token proxy, and deliberately the *conservative* one. Measured
# against the live backend on 2026-08-18: OpenRouter's tokenizer reported 17,055
# tokens for a 91,473-character prompt and 8,593 for a 44,302-character one -- 5.4
# and 5.2 characters per token. chars/4 therefore overestimates by roughly 25-30%,
# so a prompt this proxy says fits is a prompt that fits.
CHARS_PER_TOKEN = 4


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _course(i: int, platform: str = "coursera", prefix: str = "COU") -> Course:
    """A course of roughly corpus-typical size at the full render."""
    return Course(
        id_=f"{prefix}{i}",
        platform=platform,
        title=f"Course number {i} on applied topics",
        keywords=["python", "data", "stats", "ml", "viz", "sql", "etl", "nlp"],
        what_you_will_learn="learning outcome text. " * 60,
        description="description paragraph. " * 60,
        skills="skill, " * 40,
        modules="module, " * 40,
        url=f"https://example.test/course/{i}",
    )


def _store(*courses: Course) -> EmbeddingStore:
    return EmbeddingStore(
        courses=list(courses), matrix=np.ones((len(courses), DIM), dtype=np.float32)
    )


def _retrieved(courses) -> list[RetrievedCourse]:
    return [RetrievedCourse(course=c, score=1.0) for c in courses]


def _tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


# --------------------------------------------------------------------------
# 1. The prompt is bounded
# --------------------------------------------------------------------------

def test_the_configured_ceiling_fits_the_endpoint_limit():
    """The reproduction of the live failure, run against the current code.

    560 courses is what `max_keywords=40 x per_keyword=7 x 2 platforms` admits.
    Against the full `render()` this built ~441,000 tokens and was refused; the
    assertion is that the configuration's own worst case now fits, with the
    reply budget already subtracted.
    """
    cfg = Config()
    ceiling = cfg.retrieval.max_keywords * cfg.retrieval.per_keyword * 2
    budget = ENDPOINT_LIMIT - cfg.llm.max_tokens

    prompt = build_generation_prompt(
        "learn data science",
        format_context(_retrieved(_course(i) for i in range(ceiling))),
        "",
    )

    assert _tokens(prompt) < budget, (
        f"{ceiling} courses render to {_tokens(prompt):,.0f} tokens against a "
        f"{budget:,} budget -- the configuration can still overflow the endpoint."
    )


def test_the_backstop_cap_fits_even_at_the_longest_render_the_trim_can_produce():
    """The bound that holds without appealing to an average.

    The test above uses one representative course; a mean is exactly the reasoning
    that produced the falsified comment in `config.py` ("far more than a generation
    prompt consumes"). This asks the adversarial question instead: if every course
    admitted were the longest the trim can emit, does the *cap* still fit?

    This test is why `CONTEXT_MAX_CHARS` exists. Written against a trim that
    clipped only the summary, it failed: titles and keyword lists have no
    structural limit, so 600 courses of unbounded title still overflowed. The bound
    now has to hold for *any* input, not for the corpus currently on disk.
    """
    cfg = Config()
    budget = ENDPOINT_LIMIT - cfg.llm.max_tokens
    worst = Course(
        id_="COU99999",
        platform="coursera",
        title="A" * 4000,
        keywords=["k" * 400] * 5,
        what_you_will_learn="w" * 5000,
        description="d" * 5000,
    )
    per_course = len(worst.render_for_context())

    prompt = build_generation_prompt(
        "learn data science",
        format_context(_retrieved([worst] * cfg.retrieval.max_context_courses)),
        "principles " * 200,
    )

    assert per_course <= CONTEXT_MAX_CHARS + len(" ..."), (
        f"a single trimmed course reached {per_course} chars against a "
        f"{CONTEXT_MAX_CHARS} ceiling -- the block is not bounded"
    )
    assert "COU99999" in worst.render_for_context(), (
        "the ceiling ate the course ID -- a course the model cannot cite"
    )
    assert _tokens(prompt) < budget, (
        f"{cfg.retrieval.max_context_courses} worst-case courses render to "
        f"{_tokens(prompt):,.0f} tokens against a {budget:,} budget"
    )


def test_a_long_title_does_not_crowd_out_the_summary():
    """The title clip earns its own test because the ceiling already hides it.

    Reverting the title clip alone leaves the test above green: the 700-character
    block ceiling catches an unbounded title and the arithmetic still holds. What
    it does not catch is *what the ceiling eats* -- with the title unclipped, a
    long-titled course spends its whole budget on the title and reaches the model
    with no description at all, which is a silent quality loss inside a bound that
    still passes.

    Recorded rather than left implicit, because "the fix is covered by the other
    fix's test" is how this repository has twice shipped a fix whose test passed
    against unfixed code.
    """
    course = Course(
        id_="COU1",
        platform="coursera",
        title="Title " * 200,                       # 1,200 characters
        what_you_will_learn="The distinguishing summary sentence. " * 20,
    )
    rendered = course.render_for_context()

    assert "About: The distinguishing summary sentence." in rendered
    assert "COU1" in rendered


def test_a_course_costs_an_order_of_magnitude_less_in_the_prompt_than_in_the_index():
    """The trim is what makes the ceiling fit, so its magnitude is the assertion.

    A weaker check -- "the trimmed render is shorter" -- passes against a trim
    that saves twenty characters, which would not have prevented the overflow.
    """
    course = _course(1)

    assert len(course.render_for_context()) * 5 < len(course.render())


def test_the_trimmed_render_keeps_what_grounding_is_scored_on():
    """The ID must survive, or the trim breaks the metrics it was meant to protect.

    `cited_ids` matches `COU\\d+` against the learning path and
    `hallucinated_ids` compares that to the retrieved set. Drop the ID from the
    prompt and a model can no longer cite anything: the run reports zero
    citations, its grounding score is suppressed, and any ID it does invent
    escapes the hallucination check. Both errors would land on the model.
    """
    course = _course(7)
    rendered = course.render_for_context()

    assert "COU7" in rendered
    assert course.title in rendered
    assert "coursera" in rendered


def test_a_clipped_summary_is_marked_as_clipped():
    """A truncated course must not read as a short one.

    Silent truncation is the failure shape this project has corrected repeatedly:
    a value that is legal, plausible and wrong. A reader of the prompt -- or a
    model -- can tell 400 characters of a long description from a genuinely brief
    one only if the cut announces itself.
    """
    long_course = Course(
        id_="COU1", platform="coursera", title="T",
        what_you_will_learn="word " * 400,
    )
    short_course = Course(
        id_="COU2", platform="coursera", title="T", what_you_will_learn="brief.",
    )

    assert long_course.render_for_context().rstrip().endswith("...")
    assert not short_course.render_for_context().rstrip().endswith("...")
    assert len(long_course.render_for_context()) < CONTEXT_SUMMARY_CHARS + 200


# --------------------------------------------------------------------------
# 2. render() is NOT what got bounded
# --------------------------------------------------------------------------

def test_render_is_left_alone_because_the_embedder_reads_it():
    """The fix that would have passed every other check in this repository.

    `Embedder.embed_courses` calls `Course.render()` to build the matrix, and
    `CacheManifest` records model name, dimension and row count -- not the render
    format. Shortening `render()` to fix the prompt would therefore leave every
    cached matrix on disk looking valid while its vectors described text that no
    longer exists, and nothing anywhere would report it.

    So `render()` must keep the fields the trim drops. This asserts on what makes
    the embedding, deliberately: `format_context` is free to send less.
    """
    course = _course(1)
    rendered = course.render()

    for fragment in (course.skills[:20], course.modules[:20], course.url,
                     course.description[:30]):
        assert fragment in rendered

    # ...and every keyword, not the first five.
    for keyword in course.keywords:
        assert keyword in rendered


def test_the_generation_prompt_no_longer_prints_each_course_id_twice():
    """`format_context` prefixed an ID that `render()` already carried."""
    context = format_context(_retrieved([_course(3)]))

    assert context.count("COU3") == 1


# --------------------------------------------------------------------------
# 3. The backstop cap
# --------------------------------------------------------------------------

def test_the_cap_does_not_bind_at_the_shipped_configuration():
    """The property the methods section needs, asserted rather than assumed.

    A cap that binds on one arm and not another is a confound: truncating the
    context is itself a treatment, and two arms shown different amounts of corpus
    were not asked the same question. The shipped backstop sits *above* the 560
    courses the configuration can produce, so it binds on no arm at all.
    """
    cfg = Config()
    ceiling = cfg.retrieval.max_keywords * cfg.retrieval.per_keyword * 2

    kept, dropped = cap_context(
        _retrieved(_course(i) for i in range(ceiling)),
        cfg.retrieval.max_context_courses,
    )

    assert dropped == 0
    assert len(kept) == ceiling
    assert cfg.retrieval.max_context_courses > ceiling


def test_the_cap_preserves_the_round_robin_interleave():
    """Applied after the merge, never per platform.

    Capping each platform's list before the interleave would take the top N of
    whichever platform sorted first and starve the other -- the exact
    single-platform collapse the round-robin merge exists to prevent, reintroduced
    by the guard against a different failure.
    """
    merged = merge_platforms({
        "coursera": _retrieved(_course(i, "coursera", "COU") for i in range(10)),
        "edx": _retrieved(_course(i, "edx", "EDX") for i in range(10)),
    })

    kept, dropped = cap_context(merged, 6)

    assert dropped == 14
    assert [r.course.platform for r in kept] == [
        "coursera", "edx", "coursera", "edx", "coursera", "edx",
    ]


def test_switching_the_cap_off_is_expressible():
    """`SKYRAG_MAX_CONTEXT_COURSES=none` means no backstop, not the default."""
    kept, dropped = cap_context(_retrieved(_course(i) for i in range(50)), None)

    assert dropped == 0 and len(kept) == 50


def test_a_bound_cap_is_recorded_rather_than_applied_silently():
    """Grounding is scored against what the model saw, and the trim is exported.

    Two things have to be true at once: `retrieved` must be the capped list, or
    grounding would be scored against courses the model was never shown; and the
    drop must be visible, or a run that retrieved 30 and showed 5 would export
    identically to one that retrieved 5.
    """
    cfg = Config()
    cfg.retrieval.max_context_courses = 5
    cfg.retrieval.per_keyword = 30      # so all 30 courses reach the merge
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")

    stores = {
        "coursera": _store(*[_course(i) for i in range(30)]),
    }
    result = run_skyrag(
        query="learn python",
        stores=stores,
        backend=EchoBackend(responses=['{"python": 0.9}'] * 8),
        config=cfg,
        embedder=FixedEmbedder(),
    )

    assert len(result.retrieved) == 5
    assert result.context_courses_dropped == 25
    assert result.context_courses_cap == 5
    assert result.context_capped

    report = to_markdown(result)
    assert "CONTEXT CAPPED" in report
    assert "25 of 30" in report


def test_a_binding_cap_still_leaves_every_platform_represented():
    """The interleave has to survive the cap *through the pipeline*, not only in
    `cap_context`.

    `test_the_cap_preserves_the_round_robin_interleave` calls `cap_context` on an
    already-merged list, so it stays green against a pipeline that caps each
    platform *before* merging -- the "fix landing off the executing path" family
    these backlogs have closed four instances of. This asserts on the list the
    generation prompt was actually built from.

    Per-platform capping is not merely untidy: at a cap of 4 it would take 4
    Coursera and 4 edX courses and merge 8, so the cap silently fails to bind at
    all, and at any cap below the per-platform count it re-creates the
    single-platform collapse the round-robin merge exists to prevent.
    """
    cfg = Config()
    cfg.retrieval.max_context_courses = 4
    cfg.retrieval.per_keyword = 10
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")

    result = run_skyrag(
        query="learn python",
        stores={
            "coursera": _store(*[_course(i, "coursera", "COU") for i in range(10)]),
            "edx": _store(*[_course(i, "edx", "EDX") for i in range(10)]),
        },
        backend=EchoBackend(responses=['{"python": 0.9}'] * 16),
        config=cfg,
        embedder=FixedEmbedder(),
    )

    platforms = [r.course.platform for r in result.retrieved]

    assert len(result.retrieved) == 4, "the cap did not bind on the merged list"
    assert set(platforms) == {"coursera", "edx"}, (
        f"the cap starved a platform: {platforms}"
    )
    assert platforms == ["coursera", "edx", "coursera", "edx"]
    assert result.context_courses_dropped == 16


# --------------------------------------------------------------------------
# 4. It reaches the exported artifacts
# --------------------------------------------------------------------------

def _sweep_rows(tmp_path, cfg=None):
    cfg = cfg or Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")

    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm-8b"],
        pipelines=["skyrag", "naive"],
        backend=EchoBackend(responses=['{"python": 0.9}'] * 64),
        config=cfg,
        stores={"coursera": _store(*[_course(i) for i in range(12)])},
        embedder=FixedEmbedder(),
        provenance=RunProvenance.capture(cfg, "echo"),
    )
    csv_path, json_path = tmp_path / "b.csv", tmp_path / "b.json"
    export_benchmark_csv(metrics, csv_path)
    export_benchmark_json(metrics, json_path)
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return rows, payload


def test_the_export_says_how_much_context_each_row_sent(tmp_path):
    """The column that would have made the live failure legible in the artifact.

    Every column the export carried on 2026-08-18 was consistent with an ordinary
    failure. Nothing recorded how large the prompt was, so the one number that
    identified the cause had to be read out of an HTTP error string.
    """
    rows, payload = _sweep_rows(tmp_path)

    for row in rows:
        assert int(row["context_chars"]) > 0
        assert row["context_courses_dropped"] == "0"
        assert int(row["max_context_courses"]) == Config().retrieval.max_context_courses
        assert row["context_render"].startswith("trimmed-v1:")

    for row in payload:
        assert row["context_chars"] > 0
        assert row["max_context_courses"] == Config().retrieval.max_context_courses


def test_a_bound_cap_reaches_the_exported_csv(tmp_path):
    """A cap that binds is a treatment, so it has to be recoverable from the CSV."""
    cfg = Config()
    cfg.retrieval.max_context_courses = 4
    rows, _ = _sweep_rows(tmp_path, cfg)

    skyrag_row = next(r for r in rows if r["pipeline"] == "skyrag")
    assert int(skyrag_row["context_courses_dropped"]) > 0
    assert int(skyrag_row["retrieved_count"]) == 4
    assert int(skyrag_row["max_context_courses"]) == 4


def test_the_context_setting_reaches_a_failed_cell_too(tmp_path):
    """The row where it matters most carries no pipeline result to read it off.

    An arm that died is exactly the row someone returns to asking how it was
    configured -- and the failure this column explains *is* an over-large
    generation prompt, so the setting has to be stamped rather than derived.
    """
    from skyrag.llm.base import LLMError

    class Failing(EchoBackend):
        def generate(self, *a, **kw):
            raise LLMError("HTTP 400: maximum context length is 131072 tokens")

    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")

    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm-70b"],
        pipelines=["skyrag"],
        backend=Failing(responses=["{}"]),
        config=cfg,
        stores={"coursera": _store(_course(1))},
        embedder=FixedEmbedder(),
        provenance=RunProvenance.capture(cfg, "echo"),
    )
    path = tmp_path / "b.csv"
    export_benchmark_csv(metrics, path)
    row = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))[0]

    assert row["status"] == "cell_failed"
    assert int(row["max_context_courses"]) == Config().retrieval.max_context_courses
    assert row["context_render"].startswith("trimmed-v1:")


def test_the_render_format_is_stamped_so_a_changed_trim_is_visible(tmp_path):
    """`CacheManifest` records no render format; this export does.

    That omission is why `Course.render()` cannot be touched. The generation
    context has the same exposure -- a quietly retuned trim changes what every
    model saw and every number downstream -- so the format travels with the rows.
    """
    _, payload = _sweep_rows(tmp_path)

    stamped = {r["context_render"] for r in payload}
    assert len(stamped) == 1
    assert str(CONTEXT_SUMMARY_CHARS) in stamped.pop()
