"""`--control-only` is the instrument behind the long-prompt reproducibility result.

`scripts/compare_context_render.py` had no test coverage at all, which was
tolerable while it only ever produced a between-render comparison nobody reported.
It is not tolerable now: `--control-only` is what made the `per_keyword=7` regime
measurable, and the headline of
`backlog_2026-08-19_experiment_8b-vs-70b-generation.md` -- 3 of 8 pairs failing to
reproduce -- is read straight off the line it prints.

Two ways that headline could be wrong without anything crashing:

* the full-render arm quietly still runs, so `per_keyword` is capped and the
  "long prompt" control is not long;
* the control pair is taken from the wrong two arms, so the number printed under
  `CONTROL` is the between-render *effect* wearing the floor's label -- which
  inverts the reading, because the whole point of the floor is that effects are
  only interpretable above it.

Both are silent. Neither raises. Hence these.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_context_render.py"


def _load():
    spec = importlib.util.spec_from_file_location("compare_context_render", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for a module that is still
    # executing and not yet registered.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


class _Arm:
    """Stand-in for ArmResult -- control_pair only ever indexes."""

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"_Arm({self.label!r})"


# --- which arms run -------------------------------------------------------


def test_control_only_drops_the_full_render_arm(mod):
    """The full render is the arm that overflows, so it is the arm that forces
    per_keyword down to 1. If it survives --control-only, every long-prompt number
    in the 8B-vs-70B backlog was measured at a prompt length that cannot occur."""
    renders = mod.renders_for(control=False, control_only=True)
    labels = [label for label, _ in renders]
    assert "full render()" not in labels
    assert labels == ["render_for_context()", "render_for_context() again"]


def test_control_only_sends_the_same_render_twice(mod):
    """Both arms must be the *same* renderer. Two different renderers compared
    against each other is a treatment, not a control, however it is labelled."""
    renders = mod.renders_for(control=False, control_only=True)
    renderers = {renderer for _, renderer in renders}
    assert renderers == {mod.render_trimmed}


def test_plain_mode_is_two_arms_and_control_adds_a_third(mod):
    """The pre-existing behaviour, pinned so --control-only cannot regress it."""
    plain = mod.renders_for(control=False, control_only=False)
    assert [label for label, _ in plain] == ["full render()", "render_for_context()"]

    with_control = mod.renders_for(control=True, control_only=False)
    assert [label for label, _ in with_control] == [
        "full render()",
        "render_for_context()",
        "render_for_context() again",
    ]
    assert with_control[0][1] is mod.render_full
    assert with_control[1][1] is with_control[2][1] is mod.render_trimmed


# --- which arms the control is computed over ------------------------------


def test_the_control_pair_is_the_last_two_arms_in_control_only_mode(mod):
    """Under --control-only the pair is arms 0 and 1. Indexing from position 2 --
    the shape this code had while there was only one mode -- would raise here, or
    worse, silently pick up a stale arm."""
    arms = [_Arm("render_for_context()"), _Arm("render_for_context() again")]
    first, repeat = mod.control_pair(arms)
    assert first is arms[0]
    assert repeat is arms[1]


def test_the_control_pair_is_the_last_two_arms_in_full_mode(mod):
    """Under --control the pair is arms 1 and 2 -- the trimmed render against its
    own repeat. If it were arms 0 and 1 the printed CONTROL figure would be the
    between-render overlap: an effect presented as the floor that effect is
    supposed to be read against."""
    arms = [_Arm("full render()"), _Arm("render_for_context()"),
            _Arm("render_for_context() again")]
    first, repeat = mod.control_pair(arms)
    assert first is arms[1]
    assert repeat is arms[2]
    assert first.label != "full render()"


def test_the_control_pair_never_includes_the_full_render(mod):
    """The invariant both cases above are instances of, stated once directly."""
    for arms in (
        [_Arm("full render()"), _Arm("render_for_context()"),
         _Arm("render_for_context() again")],
        [_Arm("render_for_context()"), _Arm("render_for_context() again")],
    ):
        assert all(a.label != "full render()" for a in mod.control_pair(arms))


# --- the flag wiring ------------------------------------------------------


def test_control_only_is_treated_as_a_control_run():
    """--control-only IS a control run, and two independent places have to agree
    on that or it silently measures nothing: `main()` normalises the flag right
    after parsing, and the reporting block's gate accepts either flag.

    Asserted against the source because both live inside `main()`, which cannot be
    called without loading an embedder and a corpus. A source assertion is weaker
    than a behavioural one and is recorded as such -- it catches the flag being
    dropped, not the logic being subtly rewritten.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    normalisation = "if args.control_only:\n        args.control = True"
    assert normalisation in source
    assert "args.control or args.control_only" in source


def test_the_script_exposes_control_only_on_its_parser():
    """Guards against the flag being renamed or dropped while the reproduction
    commands in the backlog keep quoting it."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"--control-only"' in source
    assert "args.control_only" in source


# --- the provenance stamp (B9) -------------------------------------------


def test_the_header_stamps_the_provenance_a_determinism_claim_needs():
    """Every determinism number this script produced before 2026-08-19 was
    evidenced by `.env` plus a separate Config.load() check made at the same time,
    which is one step short of an artifact that proves what it ran under. An
    unpinned run must also say so: `provider: None` printed without comment is
    indistinguishable, to a reader, from a pin they assumed was in force."""
    source = SCRIPT.read_text(encoding="utf-8")
    for field in ("provider", "temperature", "seed", "max_tokens"):
        assert f'"  {field}' in source or f"{field} :" in source, field
    assert "UNPINNED" in source
