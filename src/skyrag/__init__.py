"""SKYRAG -- Separated Keyword Retrieval Augmented Generation.

A .py port of the notebook pipelines, with a pluggable LLM backend so the
system can run against a hosted API instead of a local 70B model.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from typing import Any

from .config import Config, EmbeddingConfig, LLMConfig, PathConfig, RetrievalConfig
from .llm import GenerationParams, LLMBackend, LLMResponse, get_backend

# Derived, not restated. The literal here read "0.1.0" while `pyproject.toml`
# declared "0.2.0", so anything stamping a result artifact with `skyrag.__version__`
# recorded a version that never existed -- and stamping provenance onto a reported
# number is what the benchmark exports now do. Deriving it means the two cannot
# drift again. The fallback covers a source checkout that was never
# installed, where there is no distribution metadata to read.
try:
    __version__ = _dist_version("skyrag")
except PackageNotFoundError:      # pragma: no cover - only in an uninstalled tree
    __version__ = "0.0.0+unknown"

_LAZY_MODULES = {
    "pipelines": "skyrag.pipelines",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        target = _LAZY_MODULES[name]
        import importlib

        mod = importlib.import_module(target)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Config",
    "EmbeddingConfig",
    "GenerationParams",
    "LLMBackend",
    "LLMConfig",
    "LLMResponse",
    "PathConfig",
    "RetrievalConfig",
    "get_backend",
    "__version__",
]

