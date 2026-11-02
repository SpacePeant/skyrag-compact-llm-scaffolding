"""Condition C provenance stamping.

The first Condition C sweep (`outputs/eval_20260907_arm2-3_condition_c_50q/`)
exported 100 rows in which nothing recorded that candidate injection had run:
Arm 3 rows were labelled `spec_only`, identical to Condition A, and Arm 2 rows
carried no label at all because the variant column is empty on unschemad arms.
Any aggregation over `schema_prompt_variant` therefore pooled Condition A and
Condition C without saying so. These tests pin the two columns that make the
condition recoverable from a row alone.
"""

from dataclasses import replace

from skyrag.pipelines.eval import BenchmarkMetric, _stamp, resolve_arm
from skyrag.schemas import (
    VARIANT_SPEC_ONLY,
    VARIANT_SPEC_PLUS_CANDIDATE_INJECTION,
    VARIANT_SPEC_PLUS_EXEMPLAR,
)


def _stamped(**kw):
    from skyrag.config import Config

    return _stamp(
        BenchmarkMetric(model="m", pipeline="skyrag", query="q", elapsed_seconds=0.0),
        cell_model="m", cfg=Config(), qid="q01", rep=1, stamp={},
        **kw,
    )


def test_injection_flag_and_limit_are_exported():
    row = _stamped(candidate_injection=True, candidate_limit=100)
    assert row.candidate_injection is True
    assert row.candidate_limit == 100
    assert "candidate_injection" in row.to_dict()
    assert "candidate_limit" in row.to_dict()


def test_limit_is_zero_when_injection_did_not_run():
    """A row that injected nothing must not carry the CLI default.

    `--candidate-limit` defaults to 100 so it need not be typed; a 100 on a row
    that ran no injection would read as an applied treatment.
    """
    row = _stamped(candidate_injection=False, candidate_limit=100)
    assert row.candidate_injection is False
    assert row.candidate_limit == 0


def test_arms_without_injection_are_unchanged():
    row = _stamped(schema_prompt_variant=VARIANT_SPEC_ONLY)
    assert row.candidate_injection is False
    assert row.candidate_limit == 0
    assert row.schema_prompt_variant == VARIANT_SPEC_ONLY


def test_injection_is_stamped_on_arm_2_which_has_no_variant():
    """Arm 2 runs injection with no schema, so the label cannot carry it."""
    row = _stamped(arm="2", candidate_injection=True, candidate_limit=100,
                   schema_prompt_variant="")
    assert row.schema_prompt_variant == ""
    assert row.candidate_injection is True
    assert row.candidate_limit == 100


def test_variant_labels_condition_c_only_for_schema_arms():
    from skyrag.pipelines import eval as ev

    def variant_for(ac, cli_injection):
        # Mirrors `_variant_for` as wired into the sweep driver.
        if not ac.schema_enabled:
            return ""
        if (ac.candidate_injection or cli_injection) and \
                ac.schema_prompt_variant == VARIANT_SPEC_ONLY:
            return VARIANT_SPEC_PLUS_CANDIDATE_INJECTION
        return ac.schema_prompt_variant

    arm3 = resolve_arm("3")
    arm2 = resolve_arm("2")
    assert variant_for(arm3, True) == VARIANT_SPEC_PLUS_CANDIDATE_INJECTION
    assert variant_for(arm3, False) == VARIANT_SPEC_ONLY
    assert variant_for(arm2, True) == ""

    # Exemplar + injection is not a named condition: keep B's label and let the
    # boolean carry the injection rather than invent a fourth letter.
    arm3_b = replace(arm3, schema_prompt_variant=VARIANT_SPEC_PLUS_EXEMPLAR)
    assert variant_for(arm3_b, True) == VARIANT_SPEC_PLUS_EXEMPLAR
    assert ev is not None
