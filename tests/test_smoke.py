"""Smoke test.

Verifies that the core pipelines (naive, skyrag), schemas and config
import cleanly, and that the release carries Arms 1-3 only.
"""

from __future__ import annotations


def test_core_package_imports():
    """Verify that all core modules can be imported without errors."""
    import skyrag
    import skyrag.config
    import skyrag.embedding
    import skyrag.llm
    import skyrag.metrics
    import skyrag.pipelines
    import skyrag.retrieval
    import skyrag.schemas
    import skyrag.selection
    import skyrag.stats
    from skyrag.pipelines import run_naive, run_skyrag

    assert skyrag.__version__ is not None
    assert callable(run_naive)
    assert callable(run_skyrag)


def test_release_carries_arms_one_to_three_only():
    from skyrag.pipelines.eval import VALID_ARMS

    assert VALID_ARMS == {"1", "2", "3"}
