"""Regression tests for what the pre-evaluation gate actually certifies.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-18_bugfix_live-run-context-overflow.md,
  finding 2

The gate's live step ran `--per-keyword 1` unconditionally: a ceiling of
40 x 1 x 2 = 80 courses, 33 retrieved in practice, against the 560 the
configuration permits. The defect that broke the 2026-08-18 sweep was therefore
structurally unreachable from the gate, which printed `CLEARED` fifteen minutes
before both 70B cells failed.

The word is the defect. `CLEARED` from a run seven times smaller than the
reportable one is a true statement about the plumbing read as a statement about
the experiment -- the same "legal value that means less than it appears to"
shape as every other finding in these backlogs.

These tests assert on the command the gate builds and on the verdict it returns,
because those are the two things a reader acts on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _preflight():
    """Import `scripts/preflight.py` by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "preflight_under_test", ROOT / "scripts" / "preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def preflight():
    return _preflight()


# --------------------------------------------------------------------------
# The command the gate runs
# --------------------------------------------------------------------------

def test_the_live_step_runs_the_configured_retrieval_by_default(preflight):
    """The reproduction of finding 2, asserted on the command line itself.

    `--per-keyword 1` was hardcoded, so no amount of `.env` said what the gate
    ran. The default now carries no override at all, which is what makes the
    live `skyrag eval` take the configured value.
    """
    _, cmd, _ = preflight.live_step(cheap=False)

    assert not any("SKYRAG_PER_KEYWORD" in part for part in cmd)
    assert cmd[1:3] == ["-m", "skyrag.cli"]


def test_the_cheap_run_is_still_available_and_still_explicit(preflight):
    """The old behaviour is a deliberate cost control, not a mistake to delete.

    It caught the truncation-attribution defect and the silent platform dropout.
    What it must not do is arrive by default and report itself as a full gate.
    """
    _, cmd, _ = preflight.live_step(cheap=True)

    # The release's live step is `skyrag eval`, which has no --per-keyword flag;
    # the cheap run sets SKYRAG_PER_KEYWORD=1 for the child process instead.
    assert "os.environ['SKYRAG_PER_KEYWORD'] = '1'" in cmd[2]
    assert "eval" in cmd


# --------------------------------------------------------------------------
# What the gate says it ran under
# --------------------------------------------------------------------------

def test_the_gate_reports_the_parameters_it_will_run_under(preflight, monkeypatch):
    """"Nearly free, and removes the false assurance" -- the backlog's own words.

    A reader who cannot see the course ceiling cannot tell a gate that exercised
    the experiment from one that exercised 6% of it.
    """
    # Pinned here, not left to .env: an unpinned provider is itself a reported
    # divergence, so on a fresh clone without .env this test would fail for it.
    monkeypatch.setenv("SKYRAG_OPENROUTER_PROVIDER", "deepinfra")
    params, divergences = preflight.live_parameters(cheap=False)

    assert params["course ceiling"] == (
        params["max_keywords"] * params["per_keyword"] * len(params["platforms"].split(","))
    )
    assert "generation model" in params
    assert divergences == []


def test_a_reduced_live_run_names_the_divergence(preflight):
    """A gate that covers less than the experiment has to say so, in numbers.

    "Runs a smaller configuration" is not actionable; "80 courses instead of 560"
    is the sentence that would have stopped `CLEARED` being believed.
    """
    _, divergences = preflight.live_parameters(cheap=True)

    assert divergences, "a reduced live run reported no divergence at all"
    joined = " ".join(divergences)
    assert "per_keyword=1" in joined
    full, _ = preflight.live_parameters(cheap=False)
    assert str(full["course ceiling"]) in joined


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

def _run_main(preflight, monkeypatch, capsys, argv, results):
    """Drive `main()` with the steps stubbed, so nothing is spent and nothing runs."""
    monkeypatch.setattr(sys, "argv", ["preflight.py", *argv])
    monkeypatch.setenv("OPENROUTER_API_KEY", "stub-key-not-used")
    # Pinned so the verdict does not depend on whether a .env exists (see above).
    monkeypatch.setenv("SKYRAG_OPENROUTER_PROVIDER", "deepinfra")
    monkeypatch.setattr(preflight, "run", lambda step, verbose: (results, 0.1, ""))
    code = preflight.main()
    return code, capsys.readouterr().out


def test_a_full_live_run_that_passes_is_cleared(preflight, monkeypatch, capsys):
    code, out = _run_main(preflight, monkeypatch, capsys, ["--live"], True)

    assert code == 0
    assert "CLEARED" in out
    assert "NOT a cleared gate" not in out


def test_a_cheap_live_run_that_passes_is_not_cleared(preflight, monkeypatch, capsys):
    """The finding, in one assertion.

    Every step passed. The gate must still refuse the word, because what it
    exercised is not what will be reported. Exit 2 is the code the gate already
    uses for "ran, and does not cover the experiment" -- the skipped-live path --
    so a reduced run is treated as the same kind of gap rather than as a pass.
    """
    code, out = _run_main(preflight, monkeypatch, capsys, ["--live", "--cheap-live"], True)

    assert code == 2
    assert "NOT a cleared" in out
    assert "per_keyword=1" in out
    # The verdict must carry the numbers, not only the earlier parameter banner.
    verdict = out[out.index("GATE RESULT"):]
    assert "per_keyword=1" in verdict


def test_a_failing_step_still_blocks_regardless_of_configuration(preflight, monkeypatch, capsys):
    """The divergence check must not shadow an outright failure."""
    code, out = _run_main(preflight, monkeypatch, capsys, ["--live", "--cheap-live"], False)

    assert code == 1
    assert "BLOCKED" in out
