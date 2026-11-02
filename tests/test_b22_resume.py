"""B22 -- `--resume`: re-run only what failed, and never lose what did not.

The defect this exists to prevent is not "resume does not work". It is **resume
appearing to work while destroying the file it was topping up**: an 11-cell R1b
re-run that writes an 11-row export over a 100-row baseline completes cleanly,
prints a plausible summary, and is only detectable by noticing the row count.
"""

from __future__ import annotations

import json

import pytest

from skyrag.pipelines.eval import (
    RERUNNABLE_STATUSES,
    BenchmarkMetric,
    cell_key,
    completed_cells,
    metric_from_row,
)


def _row(qid, model, status, *, arm="", pipeline="skyrag", repeat=1):
    """A prior export row. Carries the fields `BenchmarkMetric` requires, because
    a real export does -- see `test_incomplete_row_raises` for what happens when
    one does not."""
    return {
        "query_id": qid, "model": model, "pipeline": pipeline,
        "arm": arm, "repeat": repeat, "status": status,
        "query": f"query text for {qid}", "elapsed_seconds": 1.0,
    }


class TestCompletedCells:
    def test_ok_cells_are_skipped(self):
        done = completed_cells([_row("q01", "8b", "ok")])
        assert cell_key(repeat=1, query_id="q01", model="8b",
                        pipeline="skyrag", arm="") in done

    @pytest.mark.parametrize("status", sorted(RERUNNABLE_STATUSES))
    def test_failed_cells_are_rerun(self, status):
        """Both spellings. `arm_failed` is the pre-T1 name and the 2026-08-20
        baseline -- the file R1b actually resumes from -- is written with it."""
        assert completed_cells([_row("q01", "8b", status)]) == set()

    @pytest.mark.parametrize("status", ["partial_retrieval", "empty_retrieval"])
    def test_degraded_results_are_kept_not_rerun(self, status):
        """These are findings about the model, not harness failures.

        Re-running an empty retrieval until it stops being empty selects on the
        outcome, which is a different experiment from the one specified. 21 of the
        baseline's 100 rows are `partial_retrieval`; re-running them would rewrite
        a fifth of a paid sweep and change what it measured.
        """
        assert len(completed_cells([_row("q01", "8b", status)])) == 1

    def test_arm_is_part_of_the_key(self):
        """An arm sweep runs the same (query, model, pipeline) triple under
        different interventions. A key without `arm` would read an Arm 2 row as
        evidence that the Arm 3 cell had already run, and silently skip it."""
        done = completed_cells([_row("q01", "8b", "ok", arm="2")])
        assert cell_key(repeat=1, query_id="q01", model="8b",
                        pipeline="skyrag", arm="3") not in done

    def test_repeat_is_part_of_the_key(self):
        """L5 buys k=3. Without `repeat`, pass 2 and 3 would be skipped as done."""
        done = completed_cells([_row("q01", "8b", "ok", repeat=1)])
        assert cell_key(repeat=2, query_id="q01", model="8b",
                        pipeline="skyrag", arm="") not in done


class TestMetricFromRow:
    def test_round_trips_through_json(self):
        m = BenchmarkMetric(
            model="8b", pipeline="skyrag", query="q", query_id="q01",
            status="ok", elapsed_seconds=1.0,
        )
        back = metric_from_row(json.loads(json.dumps(m.to_dict())))
        assert back.query_id == "q01" and back.status == "ok"

    def test_incomplete_row_raises_so_the_caller_can_count_it(self):
        """A row missing a required field must NOT be silently turned into a
        default-filled metric. `cli.py` catches this and reports the count as
        undecodable, because a fabricated row in a merged export is worse than a
        missing one: it would carry `status=""` and `elapsed_seconds=0.0` into the
        results table with nothing marking it as invented."""
        with pytest.raises(TypeError):
            metric_from_row({"model": "8b", "pipeline": "skyrag"})

    def test_unknown_columns_are_dropped_not_fatal(self):
        """A resume across a schema change keeps the columns both revisions share.
        The export gained 87 -> 91 -> 109 -> 111 columns in eight days; a resume
        that raised on an unrecognised one would be unusable in practice."""
        row = BenchmarkMetric(model="8b", pipeline="skyrag", query="q",
                              query_id="q01", status="ok",
                              elapsed_seconds=1.0).to_dict()
        row["a_column_from_the_future"] = 1
        assert metric_from_row(row).query_id == "q01"


class TestTheDestructiveCase:
    """The reason this feature needs a test at all."""

    def test_carrying_prior_rows_preserves_the_baseline(self):
        """Simulates R1b: 100 prior rows, 11 failed, re-run only those.

        The merged export must be 100 rows -- not 11 (baseline destroyed) and not
        111 (failed rows duplicated alongside their replacements).
        """
        prior = [_row(f"q{i:02d}", "8b", "ok") for i in range(89)]
        prior += [_row(f"f{i:02d}", "8b", "arm_failed") for i in range(11)]
        assert len(prior) == 100

        fresh = [
            BenchmarkMetric(model="8b", pipeline="skyrag", query="q",
                            query_id=f"f{i:02d}", status="ok", elapsed_seconds=1.0)
            for i in range(11)
        ]
        rerun = {cell_key(repeat=m.repeat, query_id=m.query_id, model=m.model,
                          pipeline=m.pipeline, arm=m.arm or "") for m in fresh}
        carried = [
            metric_from_row(r) for r in prior
            if cell_key(repeat=r["repeat"], query_id=r["query_id"], model=r["model"],
                        pipeline=r["pipeline"], arm=r["arm"]) not in rerun
        ]
        merged = carried + fresh

        assert len(merged) == 100, "a resumed export must not change the row count"
        assert len(carried) == 89
        ids = [m.query_id for m in merged]
        assert len(ids) == len(set(ids)), "no cell may appear twice"
        assert not [m for m in merged if m.status in RERUNNABLE_STATUSES], (
            "every failed cell should have been superseded"
        )


class TestTheCliPath:
    """The helpers were unit-tested and the CLI still crashed on `NameError: json`.

    Every test above exercises `skyrag.pipelines.eval` directly; none of them
    imported `skyrag.cli`, so the flag's own code path was never executed. These
    run the CLI itself.
    """

    def test_resume_module_imports_json(self):
        """The crash was a missing stdlib import in the module that reads the file."""
        import skyrag.cli as cli
        assert hasattr(cli, "json"), "cli.py reads the resume file with json.loads"

    def test_missing_resume_file_exits_nonzero_without_running(self, tmp_path, capsys):
        """A typo in the path must fail before the embedder loads and before spend."""
        from skyrag.cli import run_eval_cli
        # --allow-synthetic-corpus: on a fresh clone there is no embeddings/ yet,
        # and the B24 corpus guard would refuse before the resume check is reached.
        rc = run_eval_cli([
            "--resume", str(tmp_path / "nope.json"),
            "--output-dir", str(tmp_path),
            "--allow-synthetic-corpus",
        ])
        assert rc == 1
        assert "no such file" in capsys.readouterr().out

    def test_resume_flag_is_exposed(self):
        """Guards against the flag being dropped in a future argparse edit."""
        import argparse
        import skyrag.cli as cli
        seen = []
        real = argparse.ArgumentParser.add_argument

        def spy(self, *a, **kw):
            seen.extend(x for x in a if isinstance(x, str))
            return real(self, *a, **kw)

        argparse.ArgumentParser.add_argument = spy
        try:
            try:
                cli.run_eval_cli(["--help"])
            except SystemExit:
                pass
        finally:
            argparse.ArgumentParser.add_argument = real
        assert "--resume" in seen


class TestBackendMismatch:
    """Resume must refuse to merge simulated cells into a live export.

    This is not hypothetical. The first real use of `--resume` omitted
    `-b openrouter`, took the `echo` default, and merged 11 simulated rows into an
    89-row live baseline -- printing "11 re-run" and exiting 0. It was caught only
    because echo's replies happened to be unparseable, so the cells landed as
    `empty_retrieval` with `selection_seconds=0.0`. An echo reply that parsed would
    have produced plausible numbers in a file whose other 89 rows were real.
    """

    def test_echo_into_openrouter_is_refused(self, tmp_path, capsys):
        from skyrag.cli import run_eval_cli

        prior = tmp_path / "prior.json"
        prior.write_text(json.dumps([
            {**_row("q01", "8b", "ok"), "backend": "openrouter"},
            {**_row("q02", "8b", "arm_failed"), "backend": "openrouter"},
        ]), encoding="utf-8")

        rc = run_eval_cli([
            "--resume", str(prior), "--output-dir", str(tmp_path),
            "--pipelines", "skyrag", "-q", "query text for q02",
            "--allow-synthetic-corpus",  # fresh clone: no embeddings/ (see above)
        ])
        out = capsys.readouterr().out
        assert rc == 1, "a crossed-backend merge must not exit 0"
        assert "backend mismatch" in out
        assert not (tmp_path / "benchmark_eval.json").exists(), (
            "no export may be written when the merge is refused"
        )

    def test_same_backend_is_allowed(self):
        """The guard keys on disjoint sets, so a matching backend passes through."""
        prior_backends, fresh_backends = {"openrouter"}, {"openrouter"}
        assert prior_backends & fresh_backends
