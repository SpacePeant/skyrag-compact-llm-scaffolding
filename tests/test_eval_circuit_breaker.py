"""Unit and integration tests for the evaluation sweep circuit breaker."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence
import numpy as np
import pytest

from skyrag.config import Config
from skyrag.embedding import EmbeddingStore
from skyrag.llm.base import LLMBackend, LLMError, LLMResponse
from skyrag.pipelines.eval import (
    MAX_CONSECUTIVE_CELL_FAILURES,
    SweepCircuitBreakerError,
    run_multi_model_eval,
)


class FailingBackend(LLMBackend):
    """Backend that raises LLMError (e.g. 404 or backend failure) on specific queries."""

    name = "failing_backend"

    def __init__(self, failing_queries: set[str] | None = None, error_message: str = "HTTP 404: Model Not Found"):
        self.failing_queries = failing_queries or set()
        self.error_message = error_message

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        for q in self.failing_queries:
            if q in prompt:
                raise LLMError(self.error_message)
        return LLMResponse(
            text="Step 1: Introduction to Data Science\nKeywords: data science, python\n",
            model=kwargs.get("model", "echo-model"),
            backend=self.name,
            raw={},
        )


def test_circuit_breaker_halts_after_threshold(tmp_path: Path):
    """Sweep raises SweepCircuitBreakerError after hitting consecutive failures threshold."""
    queries = [f"Query_{i}" for i in range(10)]
    backend = FailingBackend(failing_queries=set(queries), error_message="HTTP 404: Model Not Found")
    config = Config()
    stores = {"test": EmbeddingStore(courses=[], matrix=np.empty((0, 384), dtype=np.float32))}
    csv_path = tmp_path / "test_eval.csv"

    with pytest.raises(SweepCircuitBreakerError) as excinfo:
        run_multi_model_eval(
            queries=queries,
            models=["test-model"],
            pipelines=["skyrag"],
            backend=backend,
            config=config,
            stores=stores,
            output_csv_path=csv_path,
            max_consecutive_failures=3,
        )

    assert "Sweep halted by circuit breaker: encountered 3 consecutive cell failure(s)" in str(excinfo.value)
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3
        for row in reader:
            assert row["status"] == "cell_failed"
            assert "HTTP 404" in row["failure_detail"]


def test_intermittent_failures_reset_circuit_breaker(tmp_path: Path):
    """A successful cell resets the consecutive failure counter."""
    # Q0 and Q1 fail, Q2 succeeds (resets counter), Q3 succeeds
    queries = ["TargetQuery_0", "TargetQuery_1", "TargetQuery_2", "TargetQuery_3"]
    backend = FailingBackend(failing_queries={"TargetQuery_0", "TargetQuery_1"}, error_message="Transient 503")
    config = Config()
    stores = {"test": EmbeddingStore(courses=[], matrix=np.empty((0, 384), dtype=np.float32))}
    csv_path = tmp_path / "test_intermittent.csv"

    metrics = run_multi_model_eval(
        queries=queries,
        models=["test-model"],
        pipelines=["skyrag"],
        backend=backend,
        config=config,
        stores=stores,
        output_csv_path=csv_path,
        max_consecutive_failures=3,
    )

    assert len(metrics) == 4
    assert metrics[0].status == "cell_failed"
    assert metrics[1].status == "cell_failed"
    assert metrics[2].status in ("ok", "empty_retrieval", "partial_retrieval")
    assert metrics[3].status in ("ok", "empty_retrieval", "partial_retrieval")
