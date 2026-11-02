"""X1 (per-stage tokens) and R22 (generation length).

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-20_dev-task_consolidated-build-list.md (X1, R22)
- project_knowledge/task_log/backlog_2026-08-20_events_judge-credibility-and-d4-dependency.md §2

**X1.** `prompt_tokens` and `completion_tokens` came from the final
`backend.generate` alone. Query expansion, step-back and every per-platform
keyword selection were timed and their tokens counted nowhere, so `total_tokens`
was one stage of four presented as a run's cost, and **R11** (Token Efficiency
Ratio) would have priced a quarter of the pipeline.

**R22.** Judges reward longer answers, so an O1 gain that tracks length is
verbosity rather than capability -- and with no length column the two are
indistinguishable after the fact.

Both are unrecoverable once a sweep has run, which is what earns them a place
ahead of the arms under D12's carve-out.

**The load-bearing assertion in this file is the None-propagation one.** Getting
the columns to exist is easy and gets the arithmetic wrong in the direction that
flatters a cost comparison: `sum(c or 0)` turns "one call did not report usage"
into a confident total short by one call, on the axis where an undercount is
worse than an absence because it can be averaged and quoted.
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
from skyrag.llm.base import GenerationParams, LLMResponse  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    export_benchmark_csv,
    run_multi_model_eval,
    summarize_metrics,
)
from skyrag.pipelines.base import sum_tokens  # noqa: E402

DIM = 8


class CountingBackend:
    """A backend that reports usage, so per-stage attribution has something to add.

    `EchoBackend` reports no token counts at all -- which is correct for it, and
    is exactly why it cannot test this: every stage sum would be None and the
    tests would pass without the arithmetic ever running.

    Each call returns a distinct, prime-ish token count so a mis-attributed
    stage cannot coincidentally produce the right total. `usage` maps a call
    index to `(prompt, completion)`; `None` means "this call reported no usage",
    which is the case the None-propagation rule exists for.
    """

    name = "counting"

    def __init__(self, replies: list[str], usage: list[tuple[int, int] | None]):
        self.default_model = "counting/stub"
        self._replies = list(replies)
        self._usage = list(usage)
        self.calls: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        params: GenerationParams | None = None,
    ) -> LLMResponse:
        index = len(self.calls)
        self.calls.append(prompt)
        text = self._replies[index] if index < len(self._replies) else "{}"
        usage = self._usage[index] if index < len(self._usage) else (1, 1)
        prompt_tokens, completion_tokens = usage if usage else (None, None)
        return LLMResponse(
            text=text,
            model=model or self.default_model,
            backend=self.name,
            raw={},
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _stores(*platforms: str) -> dict[str, EmbeddingStore]:
    out = {}
    for platform in platforms:
        prefix = {"coursera": "COU", "edx": "EDX"}[platform]
        courses = [
            Course(
                id_=f"{prefix}{i}",
                platform=platform,
                title=f"{platform} {i}",
                keywords=["python"],
            )
            for i in (1, 2)
        ]
        out[platform] = EmbeddingStore(
            courses=courses, matrix=np.ones((2, DIM), dtype=np.float32)
        )
    return out


def _config() -> Config:
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    return cfg


# The call order `run_skyrag` makes, which every fixture below depends on:
#   0  expansion
#   1  step-back
#   2  keyword selection, platform 1
#  (3  keyword selection, platform 2 -- two-platform runs only)
#   n  generation
SELECTION_REPLY = '{"python": 0.9}'


def _one_platform_run(usage):
    return run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["skyrag"],
        backend=CountingBackend(
            replies=["expanded", "principles", SELECTION_REPLY, "Step 1: COU1"],
            usage=usage,
        ),
        config=_config(),
        stores=_stores("coursera"),
        embedder=FixedEmbedder(),
    )[0]


# --------------------------------------------------------------------------
# sum_tokens -- the rule, isolated from the pipeline
# --------------------------------------------------------------------------


def test_sum_tokens_adds_known_counts():
    assert sum_tokens([3, 5, 7]) == 15


def test_sum_tokens_refuses_a_partial_total():
    """The `or 0` defect, at the unit level.

    A stage where two of three calls reported usage has an unknown cost, not a
    cost of two calls. On the O4 axis the partial figure is worse than the
    absence: it looks complete, and it can be averaged.
    """
    assert sum_tokens([3, None, 7]) is None
    assert sum_tokens([None]) is None


def test_sum_tokens_of_no_calls_is_zero_not_none():
    """A stage that made no calls spent no tokens. That is a measurement.

    `expand_query=False` is a legitimate configuration, and reporting its cost
    as unknown would make every run using it unquotable for no reason.
    """
    assert sum_tokens([]) == 0


# --------------------------------------------------------------------------
# X1 -- the stages are attributed separately
# --------------------------------------------------------------------------


def test_each_stage_carries_its_own_tokens():
    """Expansion is calls 0-1, selection is call 2, generation is call 3."""
    metric = _one_platform_run(
        usage=[(100, 10), (200, 20), (3000, 30), (400, 40)]
    )

    assert metric.expansion_prompt_tokens == 300      # 100 + 200
    assert metric.expansion_completion_tokens == 30   # 10 + 20
    assert metric.selection_prompt_tokens == 3000
    assert metric.selection_completion_tokens == 30
    # Unprefixed == generation, unchanged from before X1.
    assert metric.prompt_tokens == 400
    assert metric.completion_tokens == 40


def test_the_selection_stage_sums_across_platforms():
    """Selection runs once per platform, so the sum has to span platforms.

    A per-platform figure would understate the treatment's cost by however many
    platforms the run covered -- and the arms differ in what they spend *per
    platform*, which is precisely the quantity being compared.
    """
    metric = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["skyrag"],
        backend=CountingBackend(
            replies=[
                "expanded",
                "principles",
                SELECTION_REPLY,
                SELECTION_REPLY,
                "Step 1: COU1 and EDX1",
            ],
            usage=[(100, 10), (200, 20), (3000, 30), (5000, 50), (400, 40)],
        ),
        config=_config(),
        stores=_stores("coursera", "edx"),
        embedder=FixedEmbedder(),
    )[0]

    assert metric.selection_prompt_tokens == 8000     # 3000 + 5000
    assert metric.selection_completion_tokens == 80   # 30 + 50


def test_the_pipeline_total_is_every_stage_not_just_generation():
    """The headline number X1 exists to produce.

    Asserted against the generation-only total on the same row, because the
    whole defect was that the two were the same value under a name that claimed
    otherwise.
    """
    metric = _one_platform_run(usage=[(100, 10), (200, 20), (3000, 30), (400, 40)])

    assert metric.pipeline_prompt_tokens == 3700      # 300 + 3000 + 400
    assert metric.pipeline_completion_tokens == 100   # 30 + 30 + 40
    assert metric.pipeline_total_tokens == 3800
    assert metric.pipeline_tokens_complete is True

    # The number that used to stand in for a run's cost, on the same row.
    assert metric.total_tokens == 440
    assert metric.pipeline_total_tokens > metric.total_tokens


def test_selection_dominates_generation_in_token_cost():
    """Not arithmetic -- the finding the column makes visible.

    The selection prompt carries the vocabulary window (up to 1,000 keywords),
    which is the largest prompt in the pipeline. A cost comparison built on the
    generation call alone was measuring the smallest stage. The fixture's
    proportions are the real ones in miniature.
    """
    metric = _one_platform_run(usage=[(100, 10), (200, 20), (3000, 30), (400, 40)])

    assert metric.selection_prompt_tokens > metric.prompt_tokens


# --------------------------------------------------------------------------
# X1 -- the None-propagation rule, which is where this gets it wrong quietly
# --------------------------------------------------------------------------


def test_a_stage_with_one_unreported_call_is_none_not_a_partial_sum():
    """Call 1 (step-back) reports nothing, so the whole expansion stage is unknown.

    100 would be the plausible-looking wrong answer here, and it is wrong in the
    direction that makes a treatment look cheaper than it was.
    """
    metric = _one_platform_run(usage=[(100, 10), None, (3000, 30), (400, 40)])

    assert metric.expansion_prompt_tokens is None
    assert metric.expansion_completion_tokens is None
    # The stages that did report are unaffected -- the rule is per stage.
    assert metric.selection_prompt_tokens == 3000


def test_one_unknown_stage_makes_the_pipeline_total_unknown():
    """And the completeness flag says so, rather than the blank being ambiguous."""
    metric = _one_platform_run(usage=[(100, 10), None, (3000, 30), (400, 40)])

    assert metric.pipeline_total_tokens is None
    assert metric.pipeline_tokens_complete is False
    # The generation-only columns still report, so the flags are not redundant:
    # a row can have a complete generation count and an incomplete pipeline one.
    assert metric.tokens_complete is True
    assert metric.total_tokens == 440


def test_the_naive_pipeline_reports_no_selection_tokens_rather_than_zero():
    """It has no keyword stage. Zero would claim the stage ran and cost nothing.

    This is the same distinction `selection_seconds=0.0` documents on the timing
    side, resolved the other way on purpose: a duration of 0.0 is
    self-evidently "did not run", where a token count of 0 would be summed into
    a cost comparison as though the baseline got its keywords free.
    """
    metric = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["naive"],
        backend=CountingBackend(
            replies=["principles", "Step 1: COU1"],
            usage=[(200, 20), (400, 40)],
        ),
        config=_config(),
        stores=_stores("coursera"),
        embedder=FixedEmbedder(),
    )[0]

    assert metric.selection_prompt_tokens is None
    assert metric.selection_completion_tokens is None
    assert metric.expansion_prompt_tokens == 200      # step-back only
    assert metric.pipeline_total_tokens is None
    assert metric.pipeline_tokens_complete is False


# --------------------------------------------------------------------------
# X1 -- reaching the artifact
# --------------------------------------------------------------------------


def test_the_stage_columns_reach_the_csv(tmp_path):
    metric = _one_platform_run(usage=[(100, 10), (200, 20), (3000, 30), (400, 40)])
    path = tmp_path / "bench.csv"
    export_benchmark_csv([metric], path)
    row = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))[0]

    assert int(row["expansion_prompt_tokens"]) == 300
    assert int(row["selection_prompt_tokens"]) == 3000
    assert int(row["pipeline_total_tokens"]) == 3800


def test_an_unknown_stage_exports_blank_not_zero(tmp_path):
    """0 and "unknown" must not print the same, or the column cannot be summed."""
    metric = _one_platform_run(usage=[(100, 10), None, (3000, 30), (400, 40)])
    path = tmp_path / "bench.csv"
    export_benchmark_csv([metric], path)
    row = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))[0]

    assert row["expansion_prompt_tokens"] == ""
    assert row["pipeline_total_tokens"] == ""


def test_the_sweep_summary_reports_each_stage_with_its_denominator():
    """A stage total without the row count is the aggregate form of the defect."""
    complete = _one_platform_run(usage=[(100, 10), (200, 20), (3000, 30), (400, 40)])
    partial = _one_platform_run(usage=[(100, 10), None, (3000, 30), (400, 40)])

    summary = summarize_metrics([complete, partial])

    # Both rows reported selection; only one reported expansion.
    assert summary["selection_tokens_total"] == {"tokens": 6060, "rows": 2}
    assert summary["expansion_tokens_total"] == {"tokens": 330, "rows": 1}
    assert summary["pipeline_tokens_total"] == 3800
    assert summary["pipeline_tokens_n"] == 1


# --------------------------------------------------------------------------
# R22 -- generation length
# --------------------------------------------------------------------------

PATH_TEXT = "Step 1: COU1 foundations.\nStep 2: COU2 builds on it."


def _length_run():
    return run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["skyrag"],
        backend=CountingBackend(
            replies=["expanded", "principles", SELECTION_REPLY, PATH_TEXT],
            usage=[(100, 10), (200, 20), (3000, 30), (400, 40)],
        ),
        config=_config(),
        stores=_stores("coursera"),
        embedder=FixedEmbedder(),
    )[0]


def test_path_length_is_measured_on_the_exported_path():
    """The length column and the judged artifact must describe one text.

    Measuring length anywhere other than `learning_path` -- on the raw response,
    say, or before a strip -- would let the verbosity check and the judge
    disagree about what they scored, which is worse than not checking.
    """
    metric = _length_run()

    assert metric.learning_path == PATH_TEXT
    assert metric.path_chars == len(PATH_TEXT)
    assert metric.path_words == len(PATH_TEXT.split())


def test_path_length_is_recorded_even_when_the_backend_reports_no_tokens():
    """Why characters are not redundant with `completion_tokens`.

    `completion_tokens` is the token half of R22 and it is None whenever a
    provider omits its usage block. A verbosity control with holes in it cannot
    be run, so the character count -- which is always available -- is what makes
    the correlation survive a provider that stops reporting mid-sweep.
    """
    metric = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["skyrag"],
        backend=CountingBackend(
            replies=["expanded", "principles", SELECTION_REPLY, PATH_TEXT],
            usage=[None, None, None, None],
        ),
        config=_config(),
        stores=_stores("coursera"),
        embedder=FixedEmbedder(),
    )[0]

    assert metric.completion_tokens is None
    assert metric.path_chars == len(PATH_TEXT)
    assert metric.path_words == len(PATH_TEXT.split())


def test_a_run_that_generated_nothing_has_zero_length():
    """An aborted run has a real length of zero, not an unknown one.

    Distinct from the token columns deliberately: `completion_tokens` is None
    here because no generation call was made, while the path genuinely is empty.
    Conflating them would put an empty path into a length correlation as a gap
    rather than as the zero it is.
    """
    metric = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["skyrag"],
        # An unparseable selection reply means no keywords, so nothing is
        # retrieved and the pipeline aborts before generation.
        backend=CountingBackend(
            replies=["expanded", "principles", "not json"],
            usage=[(100, 10), (200, 20), (3000, 30)],
        ),
        config=_config(),
        stores=_stores("coursera"),
        embedder=FixedEmbedder(),
    )[0]

    assert metric.learning_path == ""
    assert metric.path_chars == 0
    assert metric.path_words == 0
    assert metric.completion_tokens is None


def test_path_length_reaches_the_csv_and_the_summary(tmp_path):
    metric = _length_run()
    path = tmp_path / "bench.csv"
    export_benchmark_csv([metric], path)
    row = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))[0]

    assert int(row["path_chars"]) == len(PATH_TEXT)
    assert int(row["path_words"]) == len(PATH_TEXT.split())

    summary = summarize_metrics([metric])
    assert summary["path_chars_mean"] == float(len(PATH_TEXT))
    assert summary["path_chars_max"] == len(PATH_TEXT)


def test_length_and_the_judged_path_survive_the_export_together(tmp_path):
    """R22 is only usable if length and the text it measures are on one row.

    The correlation is computed per row after the sweep, so a length column that
    reached the CSV while the path did not -- or the reverse -- would leave the
    verbosity check uncomputable for exactly the runs already paid for.
    """
    metric = _length_run()
    path = tmp_path / "bench.csv"
    export_benchmark_csv([metric], path)
    row = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))[0]

    recovered = json.loads(row["learning_path"])
    assert recovered == PATH_TEXT
    assert len(recovered) == int(row["path_chars"])
