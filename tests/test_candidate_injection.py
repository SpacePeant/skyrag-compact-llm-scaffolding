"""Tests for in-context candidate keyword injection (RAG-for-Keywords)."""

from __future__ import annotations

from dataclasses import dataclass
from skyrag.data.adapters import Vocabulary
from skyrag.llm import GenerationParams, LLMResponse
from skyrag.selection import rank_candidate_keywords, select_keywords


@dataclass
class DummyBackend:
    response_text: str = '{"python": 0.95, "machine learning": 0.85}'
    last_prompt: str = ""

    def generate(self, prompt: str, model: str | None = None, params: GenerationParams | None = None) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(text=self.response_text, model=model or "dummy", backend="dummy")


def test_rank_candidate_keywords_prioritizes_query_matches():
    eligible = {
        "organic chemistry": 100,
        "python": 50,
        "machine learning": 80,
        "statistics": 90,
        "deep learning": 30,
    }
    query = "I want to learn python programming and machine learning"

    ranked = rank_candidate_keywords(query, eligible, top_k=3)

    assert "python" in ranked
    assert "machine learning" in ranked
    assert len(ranked) == 3


def test_select_keywords_with_candidate_injection():
    eligible = {
        "organic chemistry": 100,
        "python": 50,
        "machine learning": 80,
        "statistics": 90,
    }
    vocab = Vocabulary(eligible=eligible, corpus=eligible, removed={})

    backend = DummyBackend(response_text='{"python": 0.95, "machine learning": 0.85}')
    query = "Need courses on python and machine learning algorithms"

    result = select_keywords(
        backend=backend,
        query=query,
        platform="coursera",
        vocabulary=vocab,
        candidate_injection=True,
        candidate_limit=2,
    )

    assert result.outcome == "ok"
    assert "python" in result.keywords
    assert "machine learning" in result.keywords
    assert "python" in backend.last_prompt
    assert "machine learning" in backend.last_prompt
