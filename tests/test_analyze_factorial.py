"""The 2x3 analysis must degrade honestly, not quietly.

Most of these pin refusals and `None`s rather than numbers, because the ways
this script can be wrong are all ways it could still print a plausible table:
a missing cell silently dropped, a historical static run swept into an injected
cell, ITT and As-Treated pooled into one n, macro reported as micro.

The statistical machinery is scipy's and is not re-tested here. What is tested
is the wiring around it: which rows reach which test, which n it is computed
on, and what happens when a cell is absent.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fac = _load("analyze_factorial")


def _row(query_id, *, arm="2", variant="", inject=False, emitted=10, invented=3, repeat=1, **kw):
    row = {
        "query_id": query_id,
        "repeat": repeat,
        "arm": arm,
        "schema_prompt_variant": variant,
        "keywords_emitted": emitted,
        "hallucinated_keywords": invented,
        "retrieved_count": 40,
        "cited_count": 5,
    }
    if inject is not None:
        row["candidate_injection"] = inject
    row.update(kw)
    return row


def _dir(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")
    return d


def _cell_rows(context, level, queries, *, invented=3, emitted=10, repeats=1, spread=True):
    """Rows for one cell.

    `spread` varies invention per query by index. Without it every query in a
    cell scores identically, every paired difference is zero, and the tests
    exercise only the degenerate branch -- which is how the first draft of this
    file "passed" while asserting nothing about the paired path.
    """
    arm, variant = {"L0": ("2", ""), "L1": ("3", "spec_only"), "L2": ("3", "spec_plus_exemplar")}[level]
    rows = []
    for i, q in enumerate(sorted(queries)):
        n = invented + (i % 3 if spread else 0)
        for r in range(1, repeats + 1):
            rows.append(_row(q, arm=arm, variant=variant, inject=(context == "inject"),
                             emitted=emitted, invented=min(n, emitted), repeat=r))
    return rows


def _full_design(tmp_path, queries=("q01", "q02", "q03", "q04", "q05", "q06")):
    """All six cells, with an interaction deliberately present in the data."""
    args = []
    for context in fac.CONTEXTS:
        for level in fac.LEVELS:
            # inject lowers invention much more at L1/L2 than at L0 -> interaction
            invented = {"L0": 6, "L1": 5, "L2": 5}[level]
            if context == "inject":
                invented = {"L0": 5, "L1": 1, "L2": 2}[level]
            d = _dir(tmp_path, f"{context}_{level}", _cell_rows(context, level, queries, invented=invented))
            args += ["--cell", f"{context}:{level}={d}"]
    return args


# --- cell assignment ---------------------------------------------------------


def test_level_is_derived_from_arm_and_variant():
    assert fac.level_of(_row("q", arm="2", variant="")) == "L0"
    assert fac.level_of(_row("q", arm="3", variant="spec_only")) == "L1"
    assert fac.level_of(_row("q", arm="3", variant="spec_plus_exemplar")) == "L2"


def test_the_conflated_variant_reads_as_its_format_half():
    """`spec_plus_candidate_injection` names both axes; only L1 is its format."""
    row = _row("q", arm="3", variant="spec_plus_candidate_injection", inject=True)
    assert fac.level_of(row) == "L1"
    assert fac.context_of(row) == "inject"


def test_an_unstamped_row_is_refused_not_defaulted_to_static():
    """The whole reason --discover exists in a guarded form.

    `arm3_A_spec_only_20260831` (static) and the Arm 3 half of
    `eval_20260907_arm2-3_condition_c_50q` (inject) carry identical stamps.
    Defaulting absent to static would merge them.
    """
    row = _row("q", arm="3", variant="spec_only", inject=None)
    with pytest.raises(fac.Ambiguous, match="does not stamp `candidate_injection`"):
        fac.context_of(row)


def test_discover_refuses_a_run_that_predates_the_stamp(tmp_path):
    d = _dir(tmp_path, "historical", [_row("q01", arm="3", variant="spec_only", inject=None)])
    with pytest.raises(SystemExit, match="cannot place a row by its stamps"):
        fac.main(["--discover", str(d)])


def test_an_explicit_assignment_contradicted_by_the_stamps_is_refused(tmp_path):
    d = _dir(tmp_path, "actually_inject", _cell_rows("inject", "L1", ["q01"]))
    with pytest.raises(SystemExit, match="Refusing to file a row under a context it contradicts"):
        fac.main(["--cell", f"static:L1={d}"])


def test_unrecognised_variant_is_refused():
    with pytest.raises(fac.Ambiguous, match="unrecognised schema_prompt_variant"):
        fac.level_of(_row("q", arm="3", variant="some_new_thing"))


def test_arm2_with_a_variant_is_refused():
    """L0 is the unschema'd level by definition; a variant there is a mis-stamp."""
    with pytest.raises(fac.Ambiguous, match="L0 is the unschema'd level"):
        fac.level_of(_row("q", arm="2", variant="spec_only"))


# --- repeats and cohorts -----------------------------------------------------


def test_repeats_are_averaged_per_query_before_anything_pairs_on_them():
    cell = fac.Cell("static", "L0")
    cell.rows = [
        _row("q01", emitted=10, invented=2, repeat=1),
        _row("q01", emitted=10, invented=8, repeat=2),
        _row("q02", emitted=10, invented=5, repeat=1),
    ]
    per_query = cell.per_query(fac.agg.itt)
    assert set(per_query) == {"q01", "q02"}          # 3 rows -> 2 observations
    assert per_query["q01"] == pytest.approx(0.5)    # (0.2 + 0.8) / 2


def test_as_treated_cohort_is_smaller_than_itt_and_is_not_zero_filled():
    """A muted query is 1.0 under ITT and *absent* under As-Treated."""
    cell = fac.Cell("static", "L1")
    cell.rows = [_row("q01", emitted=10, invented=4), _row("q02", emitted=0, invented=0)]
    itt = cell.per_query(fac.agg.itt)
    at = cell.per_query(fac.agg.both_spoke)
    assert itt["q02"] == 1.0
    assert "q02" not in at
    assert (len(itt), len(at)) == (2, 1)


def test_micro_is_pooled_and_differs_from_macro():
    """Ledger 3.2: the two must be reported separately because they disagree."""
    cell = fac.Cell("inject", "L0")
    cell.rows = [_row("q01", emitted=90, invented=45), _row("q02", emitted=10, invented=1)]
    macro = sum(cell.per_query(fac.agg.itt).values()) / 2      # (0.5 + 0.1) / 2 = 0.30
    micro = cell.micro("hallucinated_keywords", "keywords_emitted")  # 46/100 = 0.46
    assert macro == pytest.approx(0.30)
    assert micro == pytest.approx(0.46)


def test_micro_is_none_not_zero_when_nothing_was_emitted():
    cell = fac.Cell("inject", "L0")
    cell.rows = [_row("q01", emitted=0, invented=0)]
    assert cell.micro("hallucinated_keywords", "keywords_emitted") is None


# --- missing cells -----------------------------------------------------------


def test_an_empty_cell_makes_its_tests_not_computable_rather_than_zero(tmp_path):
    """T0 may find `inject x L2` does not exist; round 2 may not finish."""
    cells = {k: fac.Cell(*k) for k in fac.CELLS}
    for key in fac.CELLS:
        if key == ("inject", "L2"):
            continue
        invented = {"L0": 6, "L1": 5, "L2": 5}[key[1]] if key[0] == "static" else {"L0": 5, "L1": 1, "L2": 2}[key[1]]
        cells[key].rows = _cell_rows(key[0], key[1], ["q01", "q02", "q03", "q04", "q05"], invented=invented)

    inter = fac.interaction(cells, fac.agg.itt)
    by_name = {r.name: r for r in inter}
    l1 = by_name["interaction: context x (L1 - L0)"]
    l2 = by_name["interaction: context x (L2 - L0)"]

    assert l1.computable                       # untouched by the missing cell
    assert not l2.computable
    assert l2.p is None and l2.statistic is None
    assert "inject.L2" in l2.note

    # And a main effect that spans the missing cell says so rather than
    # averaging the five cells it does have.
    ctx = fac.main_effect_context(cells, fac.agg.itt)
    assert not ctx.computable
    assert "inject.L2" in ctx.note


def test_unbalanced_queries_across_cells_pair_on_the_intersection(tmp_path):
    cells = {k: fac.Cell(*k) for k in fac.CELLS}
    for key in fac.CELLS:
        queries = ["q01", "q02", "q03", "q04"] if key[0] == "static" else ["q01", "q02", "q03"]
        invented = 6 if key[0] == "static" else 2
        cells[key].rows = _cell_rows(key[0], key[1], queries, invented=invented)
    result = fac.main_effect_context(cells, fac.agg.itt)
    assert result.n == 3      # q04 is absent from the inject cells
    assert result.computable


def test_all_identical_pairs_report_not_computable_rather_than_p_equals_one():
    """"Nothing to test" must not read as "tested, no difference"."""
    r = fac._wilcoxon([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], "flat")
    assert not r.computable
    assert "all paired differences are zero" in r.note


# --- corrections -------------------------------------------------------------


def test_holm_only_counts_the_tests_that_actually_ran():
    """A test that could not run is not a test that passed.

    Counting it would inflate m and make the correction weaker for the ones
    that did run -- the opposite of conservative.
    """
    ran = [fac.Result("a", 10, 1.0, 0.01), fac.Result("b", 10, 1.0, 0.04)]
    skipped = fac.Result("c", None, None, None, note="cell empty")
    fac.holm(ran + [skipped])
    assert "Holm p=0.02" in ran[0].note     # 2 * 0.01, not 3 * 0.01
    assert "Holm p=0.04" in ran[1].note     # (2-1) * 0.04
    assert "Holm" not in skipped.note


def test_holm_is_monotone():
    results = [fac.Result("a", 9, 1.0, 0.04), fac.Result("b", 9, 1.0, 0.04), fac.Result("c", 9, 1.0, 0.05)]
    fac.holm(results)
    adjusted = [float(r.note.split("Holm p=")[1]) for r in results]
    assert adjusted == sorted(adjusted)


# --- end to end --------------------------------------------------------------


def test_full_design_runs_and_finds_the_planted_interaction(tmp_path, capsys):
    assert fac.main(_full_design(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "2x3 factorial analysis" in out
    assert "Main effect: context [ITT]" in out
    assert "Interaction [ITT]" in out
    assert "Interaction [AT]" in out      # both bases, always
    assert "NOT COMPUTABLE" not in out.split("## ITT")[1].split("## AT")[0]


def test_a_missing_cell_is_announced_before_any_number(tmp_path, capsys):
    args = _full_design(tmp_path)
    args = [a for i, a in enumerate(args) if not (a.startswith("inject:L2=") or (a == "--cell" and args[i + 1].startswith("inject:L2=")))]
    fac.main(args)
    out = capsys.readouterr().out
    assert "1 cell(s) empty: inject.L2" in out
    assert out.index("cell(s) empty") < out.index("Cell descriptives")


def test_nothing_to_analyse_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="nothing to analyse"):
        fac.main([])


def test_malformed_cell_spec_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="malformed"):
        fac.main(["--cell", "garbage"])
    with pytest.raises(SystemExit, match="context must be one of"):
        fac.main(["--cell", "sideways:L0=x"])


# --- one directory, several cells -------------------------------------------


def test_a_two_arm_directory_is_sliced_by_the_declared_level(tmp_path):
    """An `--arm 2,3` sweep is two cells in one file.

    Found by running the script on the real tree: `inject.L0` and `inject.L1`
    were both assigned `eval_20260907_arm2-3_condition_c_50q` and both came
    back 0.3543 — each had swallowed the whole two-arm run, so the two cells
    were the same number and the interaction contrast was structurally zero.
    """
    rows = _cell_rows("inject", "L0", ["q01", "q02"]) + _cell_rows("inject", "L1", ["q01", "q02"])
    d = _dir(tmp_path, "two_arm_sweep", rows)

    cells, _ = fac.load_cells({("inject", "L0"): [d], ("inject", "L1"): [d]}, [])

    assert len(cells[("inject", "L0")].rows) == 2
    assert len(cells[("inject", "L1")].rows) == 2
    assert {fac.level_of(r) for r in cells[("inject", "L0")].rows} == {"L0"}
    assert {fac.level_of(r) for r in cells[("inject", "L1")].rows} == {"L1"}


def test_the_skipped_rows_are_reported_not_silently_dropped(tmp_path):
    rows = _cell_rows("inject", "L0", ["q01"]) + _cell_rows("inject", "L1", ["q01"])
    d = _dir(tmp_path, "mixed", rows)
    cells, _ = fac.load_cells({("inject", "L0"): [d]}, [])
    assert "of another level skipped" in cells[("inject", "L0")].sources[0]


def test_a_cell_left_with_no_rows_after_slicing_is_refused(tmp_path):
    d = _dir(tmp_path, "only_l1", _cell_rows("inject", "L1", ["q01"]))
    with pytest.raises(SystemExit, match="no rows are L2"):
        fac.load_cells({("inject", "L2"): [d]}, [])


def test_model_filter_slices_a_two_model_directory(tmp_path):
    """The Arm 1/2 runs hold both models and stamp `arm` as ''.

    Their only distinguishing field is `keyword_model`, which is why the
    filter is a substring match on it — the same rule `aggregate_arms.load`
    documents for exactly these runs.
    """
    rows = [
        _row("q01", arm="", keyword_model="meta-llama/llama-3.1-8b-instruct", invented=7),
        _row("q01", arm="", keyword_model="meta-llama/llama-3.1-70b-instruct", invented=2),
    ]
    d = _dir(tmp_path, "two_model", rows)

    cells, _ = fac.load_cells({("static", "L0"): [d]}, [], "8b")
    assert len(cells[("static", "L0")].rows) == 1
    assert cells[("static", "L0")].per_query(fac.agg.itt)["q01"] == pytest.approx(0.7)


def test_a_model_filter_matching_nothing_is_refused(tmp_path):
    d = _dir(tmp_path, "run", _cell_rows("static", "L0", ["q01"]))
    with pytest.raises(SystemExit, match="no rows match --model"):
        fac.load_cells({("static", "L0"): [d]}, [], "qwen")
