"""`scripts/cost_and_answer.py`: a query with no complete token total has no cost, never a zero cost."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "cost_and_answer", Path(__file__).resolve().parents[1] / "scripts" / "cost_and_answer.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_cost_is_list_price_times_tokens_of_all_stages():
    row = {"pipeline_prompt_tokens": 1_000_000, "pipeline_completion_tokens": 500_000, "pipeline_tokens_complete": True}
    assert mod.cost_fn((0.02, 0.04))(row) == pytest.approx(0.04)
    assert mod.cost_fn((0.40, 0.40))(row) == pytest.approx(0.60)


@pytest.mark.parametrize("row", [
    {"pipeline_prompt_tokens": 10, "pipeline_completion_tokens": 5, "pipeline_tokens_complete": False},
    {"pipeline_prompt_tokens": None, "pipeline_completion_tokens": 5, "pipeline_tokens_complete": True},
    {"pipeline_prompt_tokens": 10, "pipeline_completion_tokens": None, "pipeline_tokens_complete": True},
    {},
])
def test_incomplete_token_total_is_absent_not_free(row):
    assert mod.cost_fn((0.02, 0.04))(row) is None
    assert mod.tokens({**row, "pipeline_total_tokens": 15} if row.get("pipeline_tokens_complete") is False else row) is None


def test_prices_match_the_manuscript_table():
    assert mod.PRICE == {"70b": (0.40, 0.40), "8b": (0.02, 0.04), "qwen": (0.10, 0.20)}
