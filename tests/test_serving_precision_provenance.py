"""Serving precision provenance.

Arm 1's precision appears in no artifact. It was inferred from the provider pin
after the fact -- an argument rather than a record -- and the inference matters
because precision is not a minor factor here: moving `llama-3.1-8b` from bf16 to
fp8 shifted invention by +0.194, larger than any effect the schema intervention
was measured to produce. A cross-arm comparison whose arms differ in precision is
therefore confounded by the bigger of the two factors, and nothing in the export
said which arms those were.

These tests pin the column that makes precision recoverable from a row alone,
and -- more importantly -- pin that an undeclared precision exports as *empty*
rather than as an assumed default. Hosted APIs do not report their dtype, so
there is no value this can fall back to that would not be a guess wearing the
costume of a measurement.
"""

from skyrag.config import Config
from skyrag.pipelines.eval import RunProvenance


def _capture(**kw) -> RunProvenance:
    return RunProvenance.capture(Config(), "test-backend", run_id="TESTRUN", **kw)


def test_declared_precision_is_stamped_and_exported():
    prov = _capture(serving_precision="bf16")
    assert prov.serving_precision == "bf16"
    assert prov.as_columns()["serving_precision"] == "bf16"


def test_undeclared_precision_is_empty_not_a_default():
    """The absence must be visible as an absence.

    A default of "bf16" here would have made every unstamped historical run look
    like a declared bf16 run, which is precisely the failure this column exists
    to end: a row that does not say bf16 is a row where nobody wrote it down,
    not a row that ran at some assumed house default.
    """
    prov = _capture()
    assert prov.serving_precision == ""
    assert prov.as_columns()["serving_precision"] == ""


def test_precision_travels_on_the_row_not_a_run_header():
    """Rows from several sweeps get merged into one frame for analysis.

    The moment that happens a run-level header stops travelling with the row it
    describes, which is the reasoning `RunProvenance` was built on. Precision is
    subject to it more than most fields, because the comparison it protects --
    fp8 against bf16 -- is by construction a comparison *across* runs.
    """
    fp8 = _capture(serving_precision="fp8").as_columns()
    bf16 = _capture(serving_precision="bf16").as_columns()
    assert fp8["serving_precision"] != bf16["serving_precision"]
    assert {"serving_precision"} <= set(fp8) & set(bf16)
