"""The gate's exit code must match the gate's verdict.

Written 2026-09-29 to settle "C18", which I had recorded in RUNNING.md as a live
defect: that `preflight.py` prints `BLOCKED -- N step(s) failed` and still exits
0, so `preflight.py && run` would launch through a failing gate. On the strength
of that claim two launches were gated on `pytest` instead.

**The claim was wrong.** `main()` returns 1 on BLOCKED and the module ends in
`raise SystemExit(main())`. The original observation was almost certainly an
exit code read through a pipe, where `$?` belongs to the last command in the
pipeline and not to preflight.

These tests exist because a wrong belief about a gate is as dangerous as a wrong
gate, and because the belief was plausible enough to survive a week: nothing
pinned the exit codes, so the only way to check was to read the source, and I
had written the warning instead.

The three codes carry distinct meanings and a caller may act on them:
    0  cleared -- offline checks and a live run at the configured parameters
    1  BLOCKED -- a step failed
    2  not cleared -- checks passed but no qualifying live run happened
Note that `&&` treats 2 as failure, which is the safe direction: a gate that was
never fully exercised does not launch anything.
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _preflight():
    spec = importlib.util.spec_from_file_location("preflight", _ROOT / "scripts" / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["preflight"] = module
    spec.loader.exec_module(module)
    return module


def _run_main(monkeypatch, argv, step_ok):
    pf = _preflight()
    monkeypatch.setattr(pf, "run", lambda step, verbose: (step_ok, 0.1, ""))
    monkeypatch.setattr(sys, "argv", argv)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = pf.main()
    return code, buf.getvalue()


def test_blocked_exits_non_zero(monkeypatch):
    """The one that matters: a printed BLOCKED must not be a zero exit."""
    code, out = _run_main(monkeypatch, ["preflight.py"], step_ok=False)
    assert "BLOCKED" in out, "expected the gate to report BLOCKED when a step fails"
    assert code == 1, (
        f"preflight printed BLOCKED and returned {code}. A launcher chained with "
        "`preflight.py && run` would launch through a failing gate."
    )


def test_passing_offline_only_is_not_a_cleared_gate(monkeypatch):
    """Offline checks passing is not clearance, and must not exit 0.

    Exit 2 keeps `&&` from launching, which is the whole point: steps 1-3 cannot
    reach the defects a live run finds.
    """
    code, out = _run_main(monkeypatch, ["preflight.py"], step_ok=True)
    assert "NOT a cleared gate" in out
    assert code == 2, f"offline-only gate returned {code}, expected 2"


def test_the_module_propagates_main_to_the_process(monkeypatch):
    """`main()`'s return value must reach the shell, not just the caller."""
    source = (_ROOT / "scripts" / "preflight.py").read_text(encoding="utf-8")
    assert "raise SystemExit(main())" in source, (
        "preflight no longer propagates main()'s return code to the process exit "
        "status, so every exit-code test above is vacuous at the shell"
    )


@pytest.mark.parametrize("code,launches", [(0, True), (1, False), (2, False)])
def test_shell_and_semantics_hold(code, launches):
    """Pin the contract a caller relies on: only 0 permits `&&` to proceed."""
    assert (code == 0) is launches
