"""`concat_query_chunks.py`: one pass split by query, stitched back with refusals."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("concat_query_chunks", _ROOT / "scripts" / "concat_query_chunks.py")
cqc = importlib.util.module_from_spec(_spec)
sys.modules["concat_query_chunks"] = cqc
_spec.loader.exec_module(cqc)

TREATMENT = {
    "arm": "3", "keyword_model": "meta-llama/llama-3.1-8b-instruct", "backend": "openrouter",
    "provider": "deepinfra", "serving_precision": "fp8", "sweep_version": "v2",
    "schema_prompt_variant": "spec_only", "candidate_injection": False, "candidate_limit": 0,
    "candidate_ranker": "", "expansion_mode": "plain", "temperature": 0.0, "max_tokens": 4096,
    "embedding_model": "all-mpnet-base-v2", "context_render": "trimmed", "max_keywords": 40,
    "per_keyword": 7, "max_context_courses": 600,
}


def _lane(tmp_path, name, qids, **over):
    d = tmp_path / name
    d.mkdir()
    rows = [{"query_id": q, "repeat": 1, **TREATMENT, **over} for q in qids]
    (d / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")
    return d


def _expect(tmp_path, qids):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps({"id": q, "query": "x"}) for q in qids), encoding="utf-8")
    return p


def test_disjoint_chunks_stitch_into_one_pass(tmp_path):
    a = _lane(tmp_path, "q1", ["q01", "q02"])
    b = _lane(tmp_path, "q2", ["q03"])
    out = tmp_path / "pass"
    cqc.main([str(a), str(b), "--output", str(out), "--expect", str(_expect(tmp_path, ["q01", "q02", "q03"]))])
    rows = json.loads((out / "benchmark_eval.json").read_text(encoding="utf-8"))
    assert sorted(r["query_id"] for r in rows) == ["q01", "q02", "q03"]
    assert {r["repeat"] for r in rows} == {1}


def test_overlapping_chunks_are_refused(tmp_path):
    a = _lane(tmp_path, "q1", ["q01", "q02"])
    b = _lane(tmp_path, "q2", ["q02"])
    with pytest.raises(SystemExit, match="in both"):
        cqc.main([str(a), str(b), "--output", str(tmp_path / "pass")])


def test_chunks_of_different_cells_are_refused(tmp_path):
    a = _lane(tmp_path, "q1", ["q01"])
    b = _lane(tmp_path, "q2", ["q02"], expansion_mode="schema")
    with pytest.raises(SystemExit, match="expansion_mode"):
        cqc.main([str(a), str(b), "--output", str(tmp_path / "pass")])


def test_a_missing_query_is_refused(tmp_path):
    a = _lane(tmp_path, "q1", ["q01"])
    b = _lane(tmp_path, "q2", ["q02"])
    with pytest.raises(SystemExit, match="missing"):
        cqc.main([str(a), str(b), "--output", str(tmp_path / "pass"),
                  "--expect", str(_expect(tmp_path, ["q01", "q02", "q03"]))])


def test_a_chunk_with_two_repeats_is_refused(tmp_path):
    a = _lane(tmp_path, "q1", ["q01"])
    b = tmp_path / "q2"
    b.mkdir()
    rows = [{"query_id": "q02", "repeat": r, **TREATMENT} for r in (1, 2)]
    (b / "benchmark_eval.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(SystemExit, match="repeats"):
        cqc.main([str(a), str(b), "--output", str(tmp_path / "pass")])
