"""Lane merging must renumber a repeat split and refuse everything else.

The script exists to satisfy `aggregate_arms.load`'s duplicate-key refusal, so
the load-bearing test here is the integration one: a merged artifact must pass
through the real aggregator without tripping it. The rest pin the refusals,
which are the part that can silently do damage -- a wrong merge produces a file
that loads cleanly and reports a repeat axis that was never run.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


merge_lane_repeats = _load_script("merge_lane_repeats")
agg = _load_script("aggregate_arms")


TREATMENT = {
    "arm": "2",
    "keyword_model": "meta-llama/llama-3.1-8b-instruct",
    "backend": "openrouter",
    "provider": "deepinfra",
    "serving_precision": "fp8",
    "sweep_version": "v5",
    "schema_prompt_variant": "",
    "candidate_injection": False,
    "candidate_limit": 100,
    "temperature": 0.0,
    "max_tokens": 4096,
    "embedding_model": "all-mpnet-base-v2",
    "context_render": "compact",
    "max_keywords": 30,
    "per_keyword": 5,
    "max_context_courses": 200,
    # Recorded empty string: what the static (non-injection) cells actually
    # carry. Not the same thing as the field being absent.
    "candidate_ranker": "",
}


def _row(query_id, repeat=1, **overrides):
    row = {
        "query_id": query_id,
        "repeat": repeat,
        "keywords_emitted": 10,
        "hallucinated_count": 1,
        "retrieved_count": 40,
        "cited_count": 5,
        **TREATMENT,
    }
    row.update(overrides)
    return row


def _lane(tmp_path, name, rows):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")
    return directory


def _two_lanes(tmp_path):
    a = _lane(tmp_path, "lane_a", [_row("q01"), _row("q02")])
    b = _lane(tmp_path, "lane_b", [_row("q01"), _row("q02")])
    return a, b


# --- the repeat split, which is the case that should work -------------------


def test_identical_query_sets_renumber_by_lane_order(tmp_path):
    a, b = _two_lanes(tmp_path)
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert sorted((r["query_id"], r["repeat"]) for r in rows) == [
        ("q01", 1),
        ("q01", 2),
        ("q02", 1),
        ("q02", 2),
    ]


def test_lane_order_on_the_command_line_decides_the_numbering(tmp_path):
    """Not mtime, not name sort -- a rerun must renumber identically."""
    a = _lane(tmp_path, "zzz_first", [_row("q01")])
    b = _lane(tmp_path, "aaa_second", [_row("q01")])
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    by_source = {r["lane_source"]: r["repeat"] for r in rows}
    assert by_source == {"zzz_first": 1, "aaa_second": 2}


def test_multi_repeat_lanes_offset_rather_than_overwrite(tmp_path):
    """A lane already carrying 1-3 keeps its structure; the next starts at 4."""
    a = _lane(tmp_path, "lane_a", [_row("q01", repeat=r) for r in (1, 2, 3)])
    b = _lane(tmp_path, "lane_b", [_row("q01", repeat=r) for r in (1, 2, 3)])
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert sorted(r["repeat"] for r in rows) == [1, 2, 3, 4, 5, 6]


def test_original_repeat_is_preserved_so_the_merge_is_reversible(tmp_path):
    a, b = _two_lanes(tmp_path)
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    for row in rows:
        assert row["lane_original_repeat"] == 1
        assert row["lane_source"] in {"lane_a", "lane_b"}


def test_merged_artifact_loads_through_the_real_aggregator(tmp_path):
    """The whole point. Unmerged lanes trip the duplicate-key refusal; merged do not."""
    a, b = _two_lanes(tmp_path)
    out = tmp_path / "merged"

    # Precondition: concatenating the lanes naively is what the aggregator
    # refuses. If this ever stops raising, the script has no reason to exist.
    naive = _lane(tmp_path, "naive", [_row("q01"), _row("q02"), _row("q01"), _row("q02")])
    with pytest.raises(SystemExit, match="Refusing to guess which row wins"):
        agg.load(naive)

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    cell = agg.load(out)
    assert cell.n == 4
    assert agg._repeats_per_cell(cell) == 2


def test_csv_is_written_alongside_the_json(tmp_path):
    a, b = _two_lanes(tmp_path)
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    csv_text = (out / "benchmark_eval.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("query_id,repeat,")
    assert len(csv_text.strip().splitlines()) == 5  # header + 4 rows


def test_csv_does_not_carry_the_doubled_line_ending(tmp_path):
    """`export_benchmark_csv` writes `\\r\\r\\n`; a merged run must not copy it.

    `to_csv` emits `\\r\\n` and text-mode `write_text` translates the `\\n`
    again. Every native artifact on disk has it. See `write`'s docstring -- the
    divergence is deliberate, and this pins it so nobody "fixes" the merge to
    match the defect.
    """
    a, b = _two_lanes(tmp_path)
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    raw = (out / "benchmark_eval.csv").read_bytes()
    assert b"\r\r\n" not in raw


# --- the refusals ------------------------------------------------------------


def test_disjoint_query_sets_are_refused_as_a_query_split(tmp_path):
    """q01-02 and q03-04 are one pass over a partition, not two passes."""
    a = _lane(tmp_path, "lane_a", [_row("q01"), _row("q02")])
    b = _lane(tmp_path, "lane_b", [_row("q03"), _row("q04")])

    with pytest.raises(SystemExit, match="QUERY split"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


def test_partial_overlap_is_refused_rather_than_resolved(tmp_path):
    a = _lane(tmp_path, "lane_a", [_row("q01"), _row("q02")])
    b = _lane(tmp_path, "lane_b", [_row("q02"), _row("q03")])

    with pytest.raises(SystemExit, match="neither a clean repeat split"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


@pytest.mark.parametrize(
    "field,value",
    [
        ("arm", "3"),
        ("keyword_model", "qwen/qwen-2.5-7b-instruct"),
        ("sweep_version", "v2"),
        ("schema_prompt_variant", "spec_only"),
        ("candidate_injection", True),
        ("candidate_limit", 1000),
        ("serving_precision", "bf16"),
        ("backend", "echo"),
        ("provider", "phala"),
        ("temperature", 0.7),
        ("max_tokens", 1408),
        ("embedding_model", "BAAI/bge-small-en-v1.5"),
        ("context_render", "verbose"),
        ("max_keywords", 60),
        ("per_keyword", 10),
        ("max_context_courses", 800),
    ],
)
def test_lanes_from_different_cells_are_refused(tmp_path, field, value):
    """Between-cell differences must not be stacked as within-cell variance."""
    a = _lane(tmp_path, "lane_a", [_row("q01")])
    b = _lane(tmp_path, "lane_b", [_row("q01", **{field: value})])

    with pytest.raises(SystemExit, match=f"lanes disagree on '{field}'"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


def test_seed_may_differ_across_lanes(tmp_path):
    """`--seed-per-repeat` varies it by design; refusing on it would refuse the point."""
    a = _lane(tmp_path, "lane_a", [_row("q01", seed=11)])
    b = _lane(tmp_path, "lane_b", [_row("q01", seed=22)])
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert sorted(r["seed"] for r in rows) == [11, 22]


def test_the_pool_size_pilot_is_not_merged_as_repeats(tmp_path):
    """Regression, found by running over the real `outputs/` tree.

    `scratch_inject100_10q` and `scratch_inject1000_10q` match on every field
    the first draft checked. They are the 2026-09-21 pool-size pilot, where
    k=1000 collapsed retrieval 172.4 -> 44.4 per query. Merged as repeats of
    one cell, a measured contrast becomes within-cell variance and the finding
    disappears into an average.
    """
    a = _lane(tmp_path, "inject100", [_row("q01", candidate_limit=100, retrieved_count=172)])
    b = _lane(tmp_path, "inject1000", [_row("q01", candidate_limit=1000, retrieved_count=44)])

    with pytest.raises(SystemExit, match="lanes disagree on 'candidate_limit'"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


def test_an_unrecorded_field_is_named_as_unrecorded(tmp_path):
    """Absent is unknown, never equal -- and the message must say which it is."""
    row_without = _row("q01")
    del row_without["candidate_limit"]
    a = _lane(tmp_path, "lane_old", [row_without])
    b = _lane(tmp_path, "lane_new", [_row("q01", candidate_limit=100)])

    with pytest.raises(SystemExit, match="lane_old does not stamp 'candidate_limit' at all"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


def test_two_lanes_that_both_omit_a_field_are_refused(tmp_path):
    """Regression, found by running over the real `outputs/` tree.

    `arm3_A_spec_only_20260831` and `eval_20260908_arm3_inject_v1_50q` both
    predate the provenance stamp, so both omit `candidate_injection`. Under a
    `.get(field, "")` fold they compare equal and merge cleanly -- while their
    names say one injected candidates and the other did not. Two runs that
    both decline to say are not thereby the same cell.
    """
    rows = []
    for _ in range(2):
        row = _row("q01")
        del row["candidate_injection"]
        rows.append(row)
    a = _lane(tmp_path, "lane_old_a", [rows[0]])
    b = _lane(tmp_path, "lane_old_b", [rows[1]])

    with pytest.raises(SystemExit, match="neither lane_old_a nor lane_old_b stamps"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


def test_a_recorded_empty_string_is_not_treated_as_missing(tmp_path):
    """`provider` is genuinely `""` on echo; `schema_prompt_variant` on Arms 1-2.

    Those are measurements, not silence, and two lanes that agree on them must
    still merge. This is the other half of the UNRECORDED distinction -- the
    half that keeps the refusal from swallowing legitimate merges.
    """
    a = _lane(tmp_path, "lane_a", [_row("q01", provider="", schema_prompt_variant="")])
    b = _lane(tmp_path, "lane_b", [_row("q01", provider="", schema_prompt_variant="")])
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert sorted(r["repeat"] for r in rows) == [1, 2]


def test_an_internally_mixed_lane_is_refused(tmp_path):
    """A --arm 2,3 lane needs a repeat axis per arm, not one shared offset."""
    a = _lane(tmp_path, "lane_a", [_row("q01"), _row("q01", arm="3")])
    b = _lane(tmp_path, "lane_b", [_row("q01")])

    with pytest.raises(SystemExit, match="not a single cell"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


def test_existing_artifact_is_never_overwritten(tmp_path):
    a, b = _two_lanes(tmp_path)
    out = _lane(tmp_path, "already_here", [_row("q99")])

    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    # And the existing run is untouched.
    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert [r["query_id"] for r in rows] == ["q99"]


def test_a_single_lane_is_refused(tmp_path):
    a = _lane(tmp_path, "lane_a", [_row("q01")])

    with pytest.raises(SystemExit, match="at least two lanes"):
        merge_lane_repeats.main([str(a), "--output", str(tmp_path / "merged")])


def test_a_directory_without_an_artifact_is_named(tmp_path):
    a, _ = _two_lanes(tmp_path)
    missing = tmp_path / "not_a_run"
    missing.mkdir()

    with pytest.raises(SystemExit, match="no benchmark_eval.json"):
        merge_lane_repeats.main([str(a), str(missing), "--output", str(tmp_path / "m")])


def test_dry_run_writes_nothing(tmp_path):
    a, b = _two_lanes(tmp_path)
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out), "--dry-run"])

    assert not out.exists()


# --- the repeat rule, pinned against the aggregator's -----------------------


@pytest.mark.parametrize("stored", [None, 0, 1, "1"])
def test_absent_none_and_zero_all_mean_the_first_pass(tmp_path, stored):
    """`int(r.get("repeat") or 1)` -- the same fold `aggregate_arms.load` applies.

    Duplicated rather than imported, so this test is what keeps the two in
    agreement. It asserts against the aggregator directly rather than against a
    copy of the rule.
    """
    row = _row("q01")
    if stored is None:
        row.pop("repeat")
    else:
        row["repeat"] = stored

    assert merge_lane_repeats._repeat_of(row) == 1

    lane = _lane(tmp_path, f"lane_{stored}", [row])
    assert list(agg.load(lane).rows) == [("q01", 1)]


# --- C17: the ranker is a treatment, and the merge must say so --------------


def test_lanes_differing_only_in_candidate_ranker_are_refused(tmp_path):
    """C17. `candidate_ranker` selects WHICH courses reach the model.

    It was absent from `TREATMENT_FIELDS`, so two lanes that ranked candidates
    by different strategies merged silently into one cell and reported a repeat
    axis that was never run. This is not hypothetical: the corpus carries four
    distinct stamped values -- `lexical` (550 rows), `floor` (203), `freq` (50)
    and unstamped (659) -- and E3 exists precisely to contrast floor against
    lexical. Merging those two is merging the experiment with its control.

    The guard belongs here rather than in review, because the failure is silent:
    the merged artifact loads cleanly through `aggregate_arms` and every number
    it reports is a mean over two different systems.

    Both lanes set `candidate_injection=True`, which is the only condition under
    which the ranker acts at all. With injection recorded off the guard is
    deliberately narrowed; see
    `test_ranker_is_not_a_treatment_when_injection_is_off`.
    """
    a = _lane(tmp_path, "lane_floor",
              [_row("q01", candidate_injection=True, candidate_ranker="floor")])
    b = _lane(tmp_path, "lane_lexical",
              [_row("q01", candidate_injection=True, candidate_ranker="lexical")])
    out = tmp_path / "merged"

    with pytest.raises(SystemExit) as excinfo:
        merge_lane_repeats.main([str(a), str(b), "--output", str(out)])
    assert "candidate_ranker" in str(excinfo.value), (
        "the merge did not name candidate_ranker as the differing treatment"
    )


def test_lanes_agreeing_on_candidate_ranker_still_merge(tmp_path):
    """The guard must not refuse a legitimate repeat split."""
    a = _lane(tmp_path, "lane_a", [_row("q01", candidate_ranker="floor")])
    b = _lane(tmp_path, "lane_b", [_row("q01", candidate_ranker="floor")])
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert sorted(r["repeat"] for r in rows) == [1, 2]


def test_lanes_that_both_predate_the_ranker_are_refused_not_assumed(tmp_path):
    """Absent is unknown, never a value -- and unknown does not merge.

    I first assumed the opposite: that a field neither lane stamps would fall
    through as UNRECORDED and merge anyway. It does not, and that is the whole
    point of this script. When neither artifact records the ranker, whether the
    lanes ran the same one is simply not known, and guessing is how a merged
    cell comes to average the experiment with its control.

    The cost is bounded: every lane in the 2026-09-23 factorial stamps the
    field, so this refusal only reaches lanes predating 2026-09-08, whose
    provenance has to come from their task logs regardless.
    """
    a = _lane(tmp_path, "lane_a", [_row("q01", candidate_injection=True)])
    b = _lane(tmp_path, "lane_b", [_row("q01", candidate_injection=True)])
    for lane in (a, b):
        rows = json.loads((lane / "benchmark_eval.json").read_text(encoding="utf-8"))
        for row in rows:
            row.pop("candidate_ranker", None)
        (lane / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])
    assert "candidate_ranker" in str(excinfo.value)


def test_ranker_is_not_a_treatment_when_injection_is_off(tmp_path):
    """C17, refined 2026-09-29. The ranker only ranks INJECTED candidates.

    The guard added earlier today correctly refuses two lanes that disagree on
    `candidate_ranker`. It also refused a legitimate Arm 1 merge: the 70B scale
    control never injects candidates, so one pass predating the field (absent)
    and three stamping the no-ranker empty string could not be merged, although
    no ranker ran in any of them.

    `eval.py` states this itself, refusing `--candidate-ranker` when no cell
    enables injection because "no ranker will run". A field that provably cannot
    have acted is not a treatment, and blocking on it buys no protection while
    costing a real merge.

    The narrowing is deliberately tight: injection must be **recorded** False on
    every row of every lane. An absent or None `candidate_injection` leaves it
    unknown, and unknown keeps the guard armed.
    """
    a = _lane(tmp_path, "lane_a", [_row("q01", candidate_injection=False)])
    b = _lane(tmp_path, "lane_b", [_row("q01", candidate_injection=False)])
    for lane in (a, b):
        rows = json.loads((lane / "benchmark_eval.json").read_text(encoding="utf-8"))
        for row in rows:
            row.pop("candidate_ranker", None)      # lane_a style: predates the field
        (lane / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")
    b_rows = json.loads((b / "benchmark_eval.json").read_text(encoding="utf-8"))
    for row in b_rows:
        row["candidate_ranker"] = ""               # lane_b style: records "no ranker"
    (b / "benchmark_eval.json").write_text(json.dumps(b_rows), encoding="utf-8")

    merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])

    rows = json.loads((tmp_path / "merged" / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert sorted(r["repeat"] for r in rows) == [1, 2]


def test_ranker_still_guarded_when_injection_is_on(tmp_path):
    """The narrowing must not disarm the guard where it matters."""
    a = _lane(tmp_path, "lane_floor",
              [_row("q01", candidate_injection=True, candidate_ranker="floor")])
    b = _lane(tmp_path, "lane_lexical",
              [_row("q01", candidate_injection=True, candidate_ranker="lexical")])

    with pytest.raises(SystemExit) as excinfo:
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])
    assert "candidate_ranker" in str(excinfo.value)


def test_unknown_injection_keeps_the_ranker_guard_armed(tmp_path):
    """Absent injection is unknown, and unknown does not license the narrowing.

    The refusal here names `candidate_injection` rather than `candidate_ranker`,
    because injection is itself a guarded treatment and is missing from both
    lanes, so it trips first. That is the correct outcome and the reason this
    test asserts a refusal rather than a particular field: what must hold is
    that an unrecorded injection state never silently waves the ranker through.
    """
    a = _lane(tmp_path, "lane_a", [_row("q01", candidate_ranker="floor")])
    b = _lane(tmp_path, "lane_b", [_row("q01", candidate_ranker="lexical")])
    for lane in (a, b):
        rows = json.loads((lane / "benchmark_eval.json").read_text(encoding="utf-8"))
        for row in rows:
            row.pop("candidate_injection", None)
        (lane / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])
    message = str(excinfo.value)
    assert "candidate_injection" in message or "candidate_ranker" in message, message


# --- expansion_mode (2026-10-01) ---------------------------------------------


def test_lanes_differing_in_expansion_mode_are_refused(tmp_path):
    """The Stage 1 decomposition's cells share every other treatment field."""
    a = _lane(tmp_path, "lane_plain", [_row("q01", expansion_mode="plain")])
    b = _lane(tmp_path, "lane_schema", [_row("q01", expansion_mode="schema")])

    with pytest.raises(SystemExit, match="lanes disagree on 'expansion_mode'"):
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])


def test_one_stamped_lane_against_one_unstamped_is_refused(tmp_path):
    a = _lane(tmp_path, "lane_a", [_row("q01", expansion_mode="schema")])
    b = _lane(tmp_path, "lane_b", [_row("q01")])

    with pytest.raises(SystemExit) as excinfo:
        merge_lane_repeats.main([str(a), str(b), "--output", str(tmp_path / "merged")])
    assert "expansion_mode" in str(excinfo.value)


def test_two_lanes_that_predate_expansion_mode_still_merge(tmp_path):
    """No override existed before the column, so both ran the arm's own rewrite."""
    a, b = _two_lanes(tmp_path)
    out = tmp_path / "merged"

    merge_lane_repeats.main([str(a), str(b), "--output", str(out)])

    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert len(rows) == 4
