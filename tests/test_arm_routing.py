"""Tests for B0 -- `--arm` as a first-class execution axis.

Coverage:
  1. CLI helper: parse_arms() expansion and rejection.
  2. resolve_arm(): model IDs, intervention, schema flag per arm.
  3. BenchmarkMetric: schema_version field exists and exports to CSV/DataFrame.
  4. run_multi_model_eval(arms=[...]): end-to-end with echo backend.
  5. Legacy path: arms=None preserves prior models-list behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skyrag.cli import arm_cell_models, parse_arms
from skyrag.config import Config
from skyrag.data.schema import Course
from skyrag.embedding import EmbeddingStore
from skyrag.llm.echo import EchoBackend
from skyrag.pipelines import eval as eval_module
from skyrag.schemas import SCHEMA_VERSION
from skyrag.pipelines.eval import (
    MODEL_8B,
    MODEL_70B,
    VALID_ARMS,
    ArmConfig,
    BenchmarkMetric,
    export_benchmark_csv,
    metrics_to_dataframe,
    resolve_arm,
    run_multi_model_eval,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _MockEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        v = np.ones(384, dtype=np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.encode_one(t) for t in texts])


@pytest.fixture
def sample_stores():
    course = Course(
        id_="COU1",
        platform="coursera",
        title="Intro to Python",
        keywords=["python", "programming"],
        description="Learn Python.",
        url="https://coursera.org/python",
    )
    matrix = _MockEmbedder().encode([course.title])
    return {"coursera": EmbeddingStore(courses=[course], matrix=matrix)}


# ---------------------------------------------------------------------------
# 1. parse_arms() -- CLI helper
# ---------------------------------------------------------------------------

class TestParseArms:
    def test_single_digit(self):
        assert parse_arms("1") == ["1"]

    def test_comma_list(self):
        assert parse_arms("2,3") == ["2", "3"]

    def test_all_keyword(self):
        assert parse_arms("all") == ["1", "2", "3"]

    def test_all_case_insensitive(self):
        assert parse_arms("ALL") == ["1", "2", "3"]

    def test_deduplication_and_sort(self):
        # "3,1,3" -> ["1", "3"] (deduped, sorted)
        assert parse_arms("3,1,3") == ["1", "3"]

    def test_whitespace_tolerance(self):
        assert parse_arms("  2 , 3  ") == ["2", "3"]

    @pytest.mark.parametrize("raw", ["4", "5", "6", "2,4"])
    def test_unknown_arm_raises(self, raw):
        with pytest.raises(SystemExit):
            parse_arms(raw)

    def test_empty_raises(self):
        with pytest.raises(SystemExit):
            parse_arms("")

    def test_all_three_arms_individually(self):
        for a in ("1", "2", "3"):
            assert parse_arms(a) == [a]


# ---------------------------------------------------------------------------
# 1b. arm_cell_models() -- what the banner and the cell count describe
# ---------------------------------------------------------------------------


class TestArmCellModels:
    """The run description must match the run.

    Both lines that describe an arm sweep before it starts -- the "Models"
    banner and the announced cell count -- were computed from the `--models`
    default, which arm mode never reads. That produced two wrong statements:
    `--arm 1` named the 8B it would never call, and `--arm 1,2 -m a,b`
    announced 2 cells and ran 4, under-reporting spend by the number of models.
    These pin the resolution against `run_multi_model_eval`'s arm branch.
    """

    def test_single_arm_names_only_its_own_model(self):
        assert arm_cell_models(["1"], None) == [MODEL_70B]
        assert arm_cell_models(["2"], None) == [MODEL_8B]

    def test_scale_contrast_keeps_both_models_distinct(self):
        # The Arm 1 vs Arm 2 contrast IS the 70B/8B difference; collapsing it
        # to one model is the failure `-m` causes and this must not hide.
        assert arm_cell_models(["1", "2"], None) == [MODEL_70B, MODEL_8B]

    def test_override_replaces_every_arm_model(self):
        assert arm_cell_models(["1", "2"], MODEL_8B) == [MODEL_8B, MODEL_8B]

    def test_override_builds_the_models_x_arms_cross_product(self):
        # Four ArmConfigs, not two: this length is the cell multiplier, and
        # announcing two here is the bug that billed double.
        got = arm_cell_models(["1", "2"], f"{MODEL_70B},{MODEL_8B}")
        assert got == [MODEL_70B, MODEL_70B, MODEL_8B, MODEL_8B]
        assert len(got) == 4

    def test_length_matches_the_configs_the_executor_builds(self):
        """The count line divides by len(arms); that must stay exact."""
        for arms, override, expected in (
            (["1"], None, 1),
            (["1", "2"], None, 2),
            (["1", "2", "3"], None, 3),
            (["1", "2"], f"{MODEL_70B},{MODEL_8B}", 4),
            (["1", "2", "3"], f"{MODEL_70B},{MODEL_8B}", 6),
        ):
            assert len(arm_cell_models(arms, override)) == expected

    def test_whitespace_and_empty_entries_in_the_override_are_dropped(self):
        assert arm_cell_models(["1"], f" {MODEL_8B} , ") == [MODEL_8B]


# ---------------------------------------------------------------------------
# 2. resolve_arm() -- model IDs, intervention, flags
# ---------------------------------------------------------------------------

class TestResolveArm:
    def test_valid_arms_set(self):
        assert VALID_ARMS == {"1", "2", "3"}

    @pytest.mark.parametrize("arm", ["4", "5", "6"])
    def test_unknown_raises(self, arm):
        with pytest.raises(ValueError, match="not a valid arm"):
            resolve_arm(arm)

    def test_arm1_uses_70b_on_all_roles(self):
        cfg = resolve_arm("1")
        assert cfg.model == MODEL_70B
        assert cfg.keyword_model == MODEL_70B
        assert cfg.expansion_model == MODEL_70B
        assert cfg.generation_model == MODEL_70B

    def test_arm1_intervention_none(self):
        cfg = resolve_arm("1")
        assert cfg.intervention == "none"
        assert not cfg.schema_enabled
        assert cfg.schema_version == ""

    def test_arm2_uses_8b_on_all_roles(self):
        cfg = resolve_arm("2")
        assert cfg.model == MODEL_8B
        assert cfg.keyword_model == MODEL_8B
        assert cfg.expansion_model == MODEL_8B
        assert cfg.generation_model == MODEL_8B

    def test_arm2_intervention_none(self):
        cfg = resolve_arm("2")
        assert cfg.intervention == "none"
        assert not cfg.schema_enabled

    def test_arm3_uses_8b_schemas(self):
        cfg = resolve_arm("3")
        assert cfg.model == MODEL_8B
        assert cfg.intervention == "schemas"
        assert cfg.schema_enabled
        # B1, 2026-08-26: was `== ""` while the schema text did not exist. A row
        # carrying a live schema and a blank version column is unauditable --
        # nothing in the artifact would say which schema produced it.
        assert cfg.schema_version == SCHEMA_VERSION

    def test_arm_field_is_the_digit_string(self):
        for arm in ("1", "2", "3"):
            assert resolve_arm(arm).arm == arm

    def test_returns_armconfig_type(self):
        assert isinstance(resolve_arm("1"), ArmConfig)

    def test_whitespace_stripped(self):
        cfg = resolve_arm("  2  ")
        assert cfg.arm == "2"

    def test_model_constants_are_correct_ids(self):
        assert MODEL_70B == "meta-llama/llama-3.1-70b-instruct"
        assert MODEL_8B == "meta-llama/llama-3.1-8b-instruct"

    def test_custom_model_override_across_all_arms(self):
        custom_model = "qwen/qwen-2.5-7b-instruct"
        for arm in ("1", "2", "3"):
            cfg = resolve_arm(arm, model=custom_model)
            assert cfg.model == custom_model
            assert cfg.keyword_model == custom_model
            assert cfg.expansion_model == custom_model
            assert cfg.generation_model == custom_model

    def test_custom_model_override_preserves_treatment_flags(self):
        cfg3 = resolve_arm("3", model="meta-llama/llama-3.1-70b-instruct")
        assert cfg3.model == "meta-llama/llama-3.1-70b-instruct"
        assert cfg3.intervention == "schemas"
        assert cfg3.schema_enabled

        cfg2 = resolve_arm("2", model="qwen/qwen-2.5-7b-instruct")
        assert cfg2.model == "qwen/qwen-2.5-7b-instruct"
        assert cfg2.intervention == "none"
        assert not cfg2.schema_enabled


# ---------------------------------------------------------------------------
# 3. BenchmarkMetric: schema_version field
# ---------------------------------------------------------------------------

class TestBenchmarkMetricSchemaVersion:
    def _metric(self, **kwargs) -> BenchmarkMetric:
        """Minimal valid BenchmarkMetric with optional field overrides."""
        from dataclasses import replace as dc_replace

        from skyrag.llm.base import LLMError
        base = BenchmarkMetric.cell_failure(
            model="test-model",
            pipeline="skyrag",
            query="test query",
            error=LLMError("stub"),
        )
        return dc_replace(base, **kwargs)

    def test_field_exists_and_defaults_empty(self):
        m = self._metric()
        assert hasattr(m, "schema_version")
        assert m.schema_version == ""

    def test_schema_version_in_dataframe_columns(self):
        m = self._metric(arm="3", intervention="schemas", schema_version="v0.1")
        df = metrics_to_dataframe([m])
        assert "schema_version" in df.columns
        assert df["schema_version"].iloc[0] == "v0.1"

    def test_schema_version_in_csv(self, tmp_path):
        m = self._metric(arm="3", intervention="schemas", schema_version="v0.1")
        csv_path = tmp_path / "test.csv"
        export_benchmark_csv([m], csv_path)
        content = csv_path.read_text()
        assert "schema_version" in content
        assert "v0.1" in content


# ---------------------------------------------------------------------------
# 4. run_multi_model_eval(arms=[...]) -- end-to-end with echo backend
# ---------------------------------------------------------------------------

class TestRunMultiModelEvalArmMode:
    """Echo-backend runs that check arm-mode plumbing."""

    def _echo_responses(self):
        return [
            "Expanded: learn Python",
            "Step back: programming basics",
            "python\nprogramming",
            "A good Python course is COU1.",
        ]

    def test_arm2_row_has_correct_arm_and_intervention(self, sample_stores):
        backend = EchoBackend(responses=self._echo_responses())
        config = Config.load()
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arms=["2"],
        )
        assert len(metrics) == 1
        m = metrics[0]
        assert m.arm == "2", f"expected arm='2', got {m.arm!r}"
        assert m.intervention == "none"
        assert m.schema_version == ""

    def test_arm2_row_has_8b_model_columns(self, sample_stores):
        backend = EchoBackend(responses=self._echo_responses())
        config = Config.load()
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arms=["2"],
        )
        m = metrics[0]
        assert m.keyword_model == MODEL_8B
        assert m.expansion_model == MODEL_8B
        assert m.generation_model == MODEL_8B

    def _schema_responses(self):
        """Schema-compliant replies for the four calls an Arm 3 cell makes."""
        return [
            json.dumps({
                "expansions": ["learn python", "python for beginners"],
                "expanded_query": "An introductory Python programming course.",
            }),
            json.dumps({
                "step_back": "programming fundamentals",
                "principles": ["Start simple", "Build up", "Practise", "Review"],
            }),
            json.dumps({
                "platform": "coursera",
                "keywords": [
                    {"keyword": "python", "score": 0.9, "copied_verbatim": True},
                ],
            }),
            json.dumps({
                "path": [
                    {"step": 1, "course_id": "COU1", "rationale": "Starts at zero."},
                ],
            }),
        ]

    def test_arm3_runs_and_stamps_its_schema_version(self, sample_stores):
        """B1, 2026-08-26. This test asserted a refusal until today.

        The refusal was correct while no schema existed: `--arm 3` ran the
        Arm 2 zero-shot pipeline and stamped the row `arm=3`, and a
        mislabelled CSV row outlives everyone's memory of the caveat. Now that
        `schemas.py` exists and `run_skyrag` reads `schema_enabled`, the arm
        produces a real row -- and the row must carry the schema version, or
        nothing in the artifact says which treatment text produced it.
        """
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=EchoBackend(responses=self._schema_responses()),
            config=Config.load(),
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arms=["3"],
            # Pinned to the V1 payload deliberately: this test asserts the
            # `copied_verbatim` self-report below, a field the V5 flat map does
            # not carry. The default is now per cell, so the version an
            # assertion depends on has to be stated rather than inherited.
            sweep_version="v2",
        )
        assert len(metrics) == 1
        m = metrics[0]
        assert m.arm == "3"
        assert m.intervention == "schemas"
        assert m.schema_version == SCHEMA_VERSION
        assert m.schema_violations == 0
        assert m.selection_schema_violation == 0
        # The model claimed it copied its one keyword, and that claim exported.
        assert m.keywords_claimed_copied == 1
        assert m.keywords_claimed_recalled == 0
        assert m.sweep_version == "v2"

    def test_arm3_defaults_to_the_flat_map_payload(self, sample_stores):
        """With no version passed, Arm 3 must validate against V5, not V1.

        The old single default of "v3" gave Arm 3 the V1 three-field payload
        unless `--sweep-version v5` was typed, and forgetting it cost a measured
        3.2x in retrieval while the run reported success.
        """
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=EchoBackend(responses=[
                json.dumps({"expansions": ["learn python"],
                            "expanded_query": "An introductory Python course."}),
                json.dumps({"step_back": "programming fundamentals",
                            "principles": ["Start simple", "Build up"]}),
                json.dumps({"platform": "coursera", "keywords": {"python": 0.9}}),
                json.dumps({"path": [{"step": 1, "course_id": "COU1",
                                      "rationale": "Starts at zero."}]}),
            ]),
            config=Config.load(),
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arms=["3"],
        )
        m = metrics[0]
        assert m.sweep_version == "v5", "the arm-aware default did not reach the row"
        # A flat map is accepted, so the default and the payload agree.
        assert m.selection_schema_violation == 0

    def test_arm3_records_violations_without_failing_the_cell(self, sample_stores):
        """A rejected reply is Arm 3 data, not an exception and not a retry.

        The responses here are the *Arm 2* ones -- plain prose and a bare
        keyword list -- which is exactly what a model ignoring the schema
        emits. The cell must still complete and export, with the violations
        counted and their stages named. A crash here would delete the finding;
        a silent retry would turn a failure rate into a latency figure.
        """
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=EchoBackend(responses=self._echo_responses()),
            config=Config.load(),
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arms=["3"],
        )
        m = metrics[0]
        assert m.arm == "3"
        assert m.schema_violations > 0
        # Every stage that spoke is named, so the count can be attributed.
        assert "query_expansion" in m.schema_violation_stages
        assert "keyword_selection" in m.schema_violation_stages
        # The keyword stage refused the reply rather than salvaging keywords
        # out of it -- salvage would manufacture the failure being measured.
        assert m.selection_schema_violation == 1

    def test_an_unimplemented_arm_blocks_the_whole_request(
        self, sample_stores, monkeypatch
    ):
        """B16. A half-built arm fails at the top, not after the hundredth cell.

        `UNIMPLEMENTED_ARMS` is empty as of B1 -- every registered arm has a
        treatment. The guard still has to work, so the set is patched here
        rather than the test deleted. Refusing per-cell would
        leave 100 good rows and 50 mislabelled ones in one CSV, which is worse
        than either failing or succeeding: the file looks complete.
        """
        monkeypatch.setattr(eval_module, "UNIMPLEMENTED_ARMS", frozenset({"3"}))
        with pytest.raises(ValueError, match="no implementation"):
            run_multi_model_eval(
                queries=["Learn Python"],
                models=["placeholder-ignored"],
                pipelines=["skyrag"],
                backend=EchoBackend(responses=self._echo_responses() * 4),
                config=Config.load(),
                stores=sample_stores,
                embedder=_MockEmbedder(),
                arms=["1", "2", "3"],
            )

    def test_multi_arm_produces_one_row_per_arm(self, sample_stores):
        responses = self._echo_responses() * 2
        backend = EchoBackend(responses=responses)
        config = Config.load()
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=sample_stores,
            embedder=_MockEmbedder(),
            # Was ["2", "3"] until 2026-08-26. Arm 3 now refuses (B16), and this
            # test is about the multi-arm loop, not about which arms exist --
            # so it uses two arms whose treatments are both real.
            arms=["1", "2"],
        )
        assert len(metrics) == 2
        arm_values = {m.arm for m in metrics}
        assert arm_values == {"1", "2"}

    def test_arm_column_in_csv_export(self, tmp_path, sample_stores):
        backend = EchoBackend(responses=self._echo_responses())
        config = Config.load()
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arms=["2"],
        )
        csv_path = tmp_path / "arm_test.csv"
        export_benchmark_csv(metrics, csv_path)
        df = metrics_to_dataframe(metrics)
        assert "arm" in df.columns
        assert "intervention" in df.columns
        assert "schema_version" in df.columns
        assert df["arm"].iloc[0] == "2"
        assert df["intervention"].iloc[0] == "none"


# ---------------------------------------------------------------------------
# 5. Legacy path: arms=None preserves prior behaviour
# ---------------------------------------------------------------------------

class TestLegacyPath:
    def _echo_responses(self):
        return [
            "Expanded: learn Python",
            "Step back: programming basics",
            "python\nprogramming",
            "A good Python course is COU1.",
        ]

    def test_legacy_arm_column_is_empty_string(self, sample_stores):
        backend = EchoBackend(responses=self._echo_responses())
        config = Config.load()
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=[MODEL_8B],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=sample_stores,
            embedder=_MockEmbedder(),
        )
        assert len(metrics) == 1
        m = metrics[0]
        assert m.arm == ""
        assert m.intervention == ""
        assert m.schema_version == ""

    def test_legacy_arm_kwarg_still_stamps(self, sample_stores):
        """The old single-value arm= kwarg still stamps the column correctly."""
        backend = EchoBackend(responses=self._echo_responses())
        config = Config.load()
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=[MODEL_8B],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arm="2",
            intervention="none",
        )
        m = metrics[0]
        assert m.arm == "2"
        assert m.intervention == "none"


class TestModelOverrideArmMode:
    """Tests for model override functionality across arms."""

    def test_resolve_arm_with_model_override(self):
        cfg3 = resolve_arm("3", model="meta-llama/llama-3.1-70b-instruct")
        assert cfg3.model == "meta-llama/llama-3.1-70b-instruct"
        assert cfg3.keyword_model == "meta-llama/llama-3.1-70b-instruct"
        assert cfg3.generation_model == "meta-llama/llama-3.1-70b-instruct"
        assert cfg3.schema_enabled is True

        cfg2 = resolve_arm("2", model="qwen/qwen-2.5-7b-instruct")
        assert cfg2.model == "qwen/qwen-2.5-7b-instruct"
        assert cfg2.keyword_model == "qwen/qwen-2.5-7b-instruct"
        assert cfg2.generation_model == "qwen/qwen-2.5-7b-instruct"
        assert cfg2.schema_enabled is False

    def test_run_multi_model_eval_with_model_override(self, sample_stores):
        responses = [
            "Expanded: learn Python",
            "Step back: programming basics",
            "python\nprogramming",
            "A good Python course is COU1.",
        ]
        backend = EchoBackend(responses=responses)
        config = Config.load()
        metrics = run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=sample_stores,
            embedder=_MockEmbedder(),
            arms=["2"],
            model_override="qwen/qwen-2.5-7b-instruct",
        )
        assert len(metrics) == 1
        m = metrics[0]
        assert m.arm == "2"
        assert m.model == "qwen/qwen-2.5-7b-instruct"
        assert m.keyword_model == "qwen/qwen-2.5-7b-instruct"
        assert m.generation_model == "qwen/qwen-2.5-7b-instruct"
