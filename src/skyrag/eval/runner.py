"""Evaluation runner module for SkyRag benchmark sweeps and multi-model evaluation.

Provides the benchmark sweep execution and runner orchestration interface,
re-exporting and wrapping core evaluation controllers from `src/skyrag/pipelines/eval.py`.
"""

from __future__ import annotations

from typing import Any, Sequence
from pathlib import Path

from ..pipelines.eval import (
    ArmConfig,
    BenchmarkMetric,
    RunProvenance,
    SweepCircuitBreakerError,
    _run_cell,
    cell_key,
    completed_cells,
    export_benchmark_csv,
    export_benchmark_json,
    metric_from_row,
    metrics_to_dataframe,
    resolve_arm,
    run_multi_model_eval,
    summarize_metrics,
)

__all__ = [
    "ArmConfig",
    "BenchmarkMetric",
    "RunProvenance",
    "SweepCircuitBreakerError",
    "_run_cell",
    "cell_key",
    "completed_cells",
    "export_benchmark_csv",
    "export_benchmark_json",
    "metric_from_row",
    "metrics_to_dataframe",
    "resolve_arm",
    "run_multi_model_eval",
    "summarize_metrics",
]
