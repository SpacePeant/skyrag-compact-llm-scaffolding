"""Help is two levels, and each level has a different job.

`--help` is the FRONT PAGE: a command list and where to look next. Its job is to
be read, so what these pin is that it stays short and keeps the command list.
`--help-all` is the REFERENCE: its job is completeness, so what these pin is that
every flag is actually in it.
"""

from __future__ import annotations

import pytest

from skyrag import help_text

COMMANDS = ("eval", "show")

EVAL_FLAGS = (
    "-b, --backend", "--arm", "-m, --models", "-P, --pipelines", "--max-queries",
    "-r, --repeats", "--seed-per-repeat", "-o, --output-dir", "--resume",
    "--candidate-injection", "--candidate-limit", "--candidate-ranker",
    "--candidate-source", "--schema-exemplar", "--expansion-mode",
    "--sweep-version", "--serving-precision", "--keyword-model",
    "--expansion-model", "--generation-model", "--intervention",
    "--allow-synthetic-corpus", "--allow-degraded-embedder",
)


class TestRouting:
    @pytest.mark.parametrize("args", [["--help"], ["-h"], ["help"]])
    def test_bare_help_is_the_front_page(self, args):
        assert help_text.resolve(args) == help_text.front_page()

    @pytest.mark.parametrize("args", [["--help-all"], ["--helpall"], ["--help", "all"]])
    def test_everything_spellings(self, args):
        assert help_text.resolve(args) == help_text.everything()

    @pytest.mark.parametrize("args", [[], ["eval"], ["show"]])
    def test_non_help_args_route_nowhere(self, args):
        """resolve() must not swallow a subcommand."""
        assert help_text.resolve(args) is None


class TestFrontPageStaysReadable:
    """The front page exists to be read. Length is the feature."""

    def test_fits_one_screen(self):
        assert len(help_text.front_page().splitlines()) <= 32

    def test_fits_eighty_columns(self):
        over = [ln for ln in help_text.front_page().splitlines() if len(ln) > 80]
        assert not over, f"lines over 80 columns: {over[:3]}"

    def test_lists_every_command(self):
        page = help_text.front_page()
        for cmd in COMMANDS:
            assert f"\n  {cmd}" in page, f"{cmd} missing from the command list"

    def test_points_at_the_deeper_help(self):
        page = help_text.front_page()
        assert "eval --help" in page
        assert "--help-all" in page

    def test_warns_about_the_mock_backend(self):
        """The one default that silently produces a meaningless export."""
        assert "mock" in help_text.front_page()

    def test_is_not_a_flag_reference(self):
        """If the front page grows a flag table it has stopped being a front page."""
        page = help_text.front_page()
        assert "--allow-synthetic-corpus" not in page
        assert "--expansion-model" not in page

    def test_names_the_scale_contrast_trap(self):
        assert "Do not pass -m" in help_text.front_page()


class TestReferenceIsComplete:
    """`--help-all` is where detail lives, so it has to actually hold all of it."""

    @pytest.mark.parametrize("flag", EVAL_FLAGS)
    def test_every_eval_flag_is_documented(self, flag):
        assert flag in help_text.everything()

    def test_carries_the_no_sampling_flags_surprise(self):
        assert "--temperature" in help_text.everything()

    def test_is_longer_than_the_front_page(self):
        full = len(help_text.everything().splitlines())
        assert full > len(help_text.front_page().splitlines())

    def test_fits_a_standard_terminal(self):
        over = [ln for ln in help_text.everything().splitlines() if len(ln) > 84]
        assert not over, f"lines over 84 columns: {over[:3]}"

    def test_sections_are_blank_line_separated(self):
        """Headings jammed against the block above are unreadable -- this regressed once."""
        page = help_text.everything()
        assert "\n\nTRAPS" in page
        assert "\n\nARMS" in page
