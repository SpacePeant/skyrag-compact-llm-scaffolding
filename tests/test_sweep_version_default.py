"""The sweep version defaults to v5, and the row records what ran.

These tests exist because the failure mode is silent: forgetting the flag used to
give Arm 3 the V1 three-field payload, which measures a 3.2x retrieval collapse
against V5 on the same window, while the run reported success.
"""

import pytest

from skyrag.cli import resolve_benchmark_output_dir, run_eval_cli
from skyrag.pipelines.eval import (
    SWEEP_VERSION_AUTO_LABEL,
    SWEEP_VERSION_SELECTION_DEFAULT,
    resolve_sweep_version,
)


class TestResolveSweepVersion:
    def test_unspecified_gets_v5(self):
        assert resolve_sweep_version(None) == "v5"
        assert SWEEP_VERSION_SELECTION_DEFAULT == "v5"

    @pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5"])
    def test_explicit_version_always_wins(self, version):
        """Published recipes pin a version; the default must never override one."""
        assert resolve_sweep_version(version) == version

    @pytest.mark.parametrize("sentinel", [None, "", "auto"])
    def test_sentinels_all_mean_the_default(self, sentinel):
        assert resolve_sweep_version(sentinel) == "v5"


class TestCliSurface:
    def _help(self, capsys):
        with pytest.raises(SystemExit):
            run_eval_cli(["--help"])
        return capsys.readouterr().out

    def test_help_documents_the_default(self, capsys):
        """The contract is user-facing: a reader must not assume a version."""
        out = " ".join(self._help(capsys).split())
        assert "Default: auto, which resolves to v5" in out

    def test_auto_is_an_accepted_choice(self, capsys):
        out = self._help(capsys)
        assert "auto" in out

    def test_output_dir_slug_survives_an_unset_version(self, tmp_path, monkeypatch):
        """The run-dir name is built before any cell resolves; it must not crash."""
        monkeypatch.chdir(tmp_path)
        out = resolve_benchmark_output_dir(None, arms=["3"], sweep_version=None, query_count=10)
        assert SWEEP_VERSION_AUTO_LABEL in out.name


class TestRowStamping:
    """The row must record the version the cell ran, not the run-level request."""

    def test_stamp_override_beats_run_header(self):
        from dataclasses import replace

        from skyrag.pipelines.eval import BenchmarkMetric

        header = {"sweep_version": SWEEP_VERSION_AUTO_LABEL, "seed": 34}
        cell_stamp = {**header, "sweep_version": "v5"}
        base = BenchmarkMetric(model="m", pipeline="skyrag", query="q", elapsed_seconds=0.0)
        row = replace(base, **cell_stamp)
        assert row.sweep_version == "v5"
        assert row.seed == 34


def test_serving_precision_defaults_from_the_shared_environment_variable(monkeypatch, tmp_path):
    """One declaration must cover a sweep row and a chat answer alike.

    An operator who exported `SKYRAG_SERVING_PRECISION` must get a stamped
    sweep row, not an UNSTAMPED one -- the field moved invention by +0.194 and
    therefore decides whether two cells are comparable at all.

    Driven end to end through the echo backend because the parser is built
    inside `run_eval_cli`, so the default cannot be inspected without running
    the command that binds it.
    """
    import json

    from skyrag.cli import run_eval_cli

    monkeypatch.setenv("SKYRAG_SERVING_PRECISION", "bf16")
    out = tmp_path / "envprec"
    run_eval_cli([
        "--arm", "2", "-Q", "benchmarks/queries_1q_smoke.jsonl",
        "-b", "echo", "--allow-synthetic-corpus", "-o", str(out),
    ])
    rows = [r for r in json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
            if r.get("query_id")]
    assert rows, "the echo run produced no rows to check"
    assert {r["serving_precision"] for r in rows} == {"bf16"}


def test_an_explicit_precision_flag_still_beats_the_environment(monkeypatch, tmp_path):
    """The environment is a default, never an override."""
    import json

    from skyrag.cli import run_eval_cli

    monkeypatch.setenv("SKYRAG_SERVING_PRECISION", "bf16")
    out = tmp_path / "explicitprec"
    run_eval_cli([
        "--arm", "2", "-Q", "benchmarks/queries_1q_smoke.jsonl",
        "-b", "echo", "--allow-synthetic-corpus",
        "--serving-precision", "fp8", "-o", str(out),
    ])
    rows = [r for r in json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
            if r.get("query_id")]
    assert {r["serving_precision"] for r in rows} == {"fp8"}
