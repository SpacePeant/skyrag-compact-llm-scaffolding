"""Central configuration for SKYRAG.

Everything that was hardcoded in the notebooks (model ids, device, paths,
top-k values, the Ollama URL) lives here and can be overridden by environment
variables, so the same code runs on a 2x A100 box and on a CPU laptop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root = three levels up from src/skyrag/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _coerce(key: str, raw: str, caster, kind: str):
    """Parse an environment value, naming the setting when it will not parse.

    The bare `int(raw)` this replaced raised
    `ValueError: invalid literal for int() with base 10: 'twenty'` -- a message
    that names neither the variable that carried the value nor the file it most
    likely came from. Seven numeric settings are read this way, so the reader of
    that traceback had seven candidates and a stack that pointed at a dataclass
    field factory rather than at their own typo.
    """
    try:
        return caster(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{key}={raw!r} is not a valid {kind}. Environment settings are read "
            f"from the process environment and from `.env` at the repository root, "
            f"in that order of precedence -- correct {key} in whichever of the two "
            f"is setting it, or unset it to take the default."
        ) from None


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return _coerce(key, raw, int, "integer") if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return _coerce(key, raw, float, "number") if raw else default


_TRUE_LITERALS = {"1", "true", "yes", "on"}
_FALSE_LITERALS = {"0", "false", "no", "off", ""}


def _env_bool(key: str, default: bool) -> bool:
    """A flag, refusing a value it cannot read rather than guessing.

    `bool(os.environ.get(key))` is the obvious spelling and it is wrong in the
    direction that matters here: every non-empty string is truthy, so
    `SKYRAG_DEVICE_STRICT=false` would switch strictness *on*. A setting whose
    whole job is to refuse a silent downgrade must not itself be enabled by a
    misread value, so anything outside the two literal sets raises through
    `_coerce`, naming the variable the way the seven numeric settings do.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default

    def _cast(value: str) -> bool:
        folded = value.strip().lower()
        if folded in _TRUE_LITERALS:
            return True
        if folded in _FALSE_LITERALS:
            return False
        raise ValueError(value)

    return _coerce(key, raw, _cast, "boolean (true/false, 1/0, yes/no, on/off)")


# Values that mean "this setting is deliberately switched off", as distinct from
# "this setting was not mentioned". Only settings whose type is genuinely
# optional read these; `SKYRAG_TOP_K=none` is still an error, because there is no
# such thing as a run with no top-k.
_OFF_LITERALS = {"", "none", "null", "off"}


def _env_int_or_none(key: str, default: int | None) -> int | None:
    """An int, None when the variable is present and explicitly switched off.

    `_env_int` cannot express None: it returns the default for an unset variable
    and for an empty one alike, so a field typed `int | None` had no environment
    route to the None half of its own type. For `seed` that meant an unseeded run
    -- the thing every backend already implements by sending no seed -- could be
    requested from Python and not from `.env`.

    Absent and empty are kept distinct on purpose. Unset means "no opinion, take
    the default"; `SKYRAG_SEED=` means "no seed", which is a statement about the
    run. The seed that actually ran reaches every exported row through the
    provenance stamp, so a run switched off by accident says so in its own
    artifact rather than being discoverable only by re-reading `.env`.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    if raw.strip().lower() in _OFF_LITERALS:
        return None
    return _coerce(key, raw, int, "integer")


def resolve_device(requested: str = "auto", *, strict: bool = False) -> str:
    """Resolve 'auto' to cuda/mps/cpu, reporting what it settled for and why.

    The notebooks hardcoded "cuda:1", which fails on any machine without two
    GPUs. Default is now auto-detection.

    `strict` exists because wall-clock time is a reported metric. A run that
    asked for CUDA and quietly got CPU produces timings three orders of
    magnitude apart from the run it will be compared against, with nothing in
    the result to say so -- the same "failure encoded as a legal success value"
    shape as the defects this module's callers were audited for. With
    `strict=True` an unavailable accelerator raises instead of downgrading.

    The fallback reason is recorded in `LAST_DEVICE_FALLBACK` either way, so a
    non-strict caller can still report what happened rather than guess.
    """
    global LAST_DEVICE_FALLBACK

    if requested != "auto":
        # An explicit request is honoured, but a request for an accelerator that
        # is not there must not pass silently.
        family = requested.split(":", 1)[0].lower()
        if family in {"cuda", "mps"}:
            available, why = _accelerator_available(family)
            if not available:
                LAST_DEVICE_FALLBACK = f"requested {requested!r} but {why}"
                if strict:
                    raise RuntimeError(
                        f"Device {requested!r} was requested explicitly but {why}. "
                        "Refusing to fall back to CPU: wall-clock time is a reported "
                        "metric here, so a silent downgrade would make this run "
                        "incomparable with the runs it is measured against. Pass "
                        "device='auto' to accept whatever is present, or 'cpu' to "
                        "state the intent outright."
                    )
        else:
            LAST_DEVICE_FALLBACK = None
        return requested

    try:
        import torch
    except ImportError:
        LAST_DEVICE_FALLBACK = "torch is not installed; using cpu"
        if strict:
            raise RuntimeError(
                "device='auto' with strict=True cannot confirm an accelerator: "
                "torch is not installed, so only cpu is available."
            )
        return "cpu"

    if torch.cuda.is_available():
        LAST_DEVICE_FALLBACK = None
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        LAST_DEVICE_FALLBACK = None
        return "mps"

    LAST_DEVICE_FALLBACK = "no CUDA or MPS device is available; using cpu"
    if strict:
        raise RuntimeError(
            "device='auto' with strict=True found no CUDA or MPS device. "
            "Pass device='cpu' to run on CPU deliberately."
        )
    return "cpu"


# Why the last resolve_device() call did not get an accelerator, or None when it
# did. Module-level rather than returned, so adding the explanation did not have
# to change every call site's signature.
LAST_DEVICE_FALLBACK: str | None = None


def _accelerator_available(family: str) -> tuple[bool, str]:
    """(is_available, human reason when it is not) for 'cuda' or 'mps'."""
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"

    if family == "cuda":
        try:
            if torch.cuda.is_available():
                return True, ""
            return False, "torch reports no CUDA device"
        except Exception as exc:      # a driver-level probe failure, not absence
            return False, f"the CUDA probe itself failed ({type(exc).__name__}: {exc})"

    backend = getattr(torch.backends, "mps", None)
    if backend is not None and backend.is_available():
        return True, ""
    return False, "torch reports no MPS device"


@dataclass
class LLMConfig:
    """Which LLM backend to use, and with what defaults.

    backend: "openrouter" | "ollama" | "echo"
    """

    backend: str = field(default_factory=lambda: _env("SKYRAG_LLM_BACKEND", "openrouter"))

    # OpenRouter
    openrouter_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    # Sent as optional attribution headers; OpenRouter uses them for rankings.
    openrouter_referer: str | None = field(
        default_factory=lambda: os.environ.get("OPENROUTER_REFERER")
    )
    openrouter_title: str | None = field(
        default_factory=lambda: os.environ.get("OPENROUTER_TITLE", "SKYRAG")
    )

    # Which *upstream* provider OpenRouter may route to -- a serving company such
    # as "DeepInfra" or "Together", not a backend name. Unset means automatic
    # routing, and the run records that it was automatic.
    #
    # This matters twice over on a benchmark: `seed` support is provider-dependent,
    # so determinism was never guaranteed while routing was free, and latency and
    # price differ between providers serving the same model, so part of the cost
    # axis was measuring routing luck rather than the model.
    #
    # Deliberately NOT named SKYRAG_PROVIDER. That variable already exists in the
    # local `.env`, holds "openrouter" -- the backend name, not a provider slug --
    # and is read by nothing. Giving a new meaning to a variable whose current
    # value is wrong for that meaning would turn a dead setting into a live one
    # that pins routing to a provider that does not exist, and with fallbacks
    # disabled that fails every request. A new name leaves the dead one dead.
    provider: str | None = field(
        default_factory=lambda: os.environ.get("SKYRAG_OPENROUTER_PROVIDER")
    )

    # Ollama (the original local path, kept working)
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Model ids are per-role so you can use a cheap model for keyword
    # selection and a stronger one for the final learning path.
    model: str = field(default_factory=lambda: _env("SKYRAG_MODEL", "meta-llama/llama-3.1-70b-instruct"))
    keyword_model: str | None = field(default_factory=lambda: os.environ.get("SKYRAG_KEYWORD_MODEL"))
    expansion_model: str | None = field(default_factory=lambda: os.environ.get("SKYRAG_EXPANSION_MODEL"))
    # Declared rather than attached at runtime. `run_multi_model_eval` assigned
    # `cfg.llm.generation_model = model` to override the model under test, and on a
    # plain dataclass that assignment succeeds while `model_for("generation")` goes
    # on returning `self.model` -- so every arm of a multi-model sweep silently ran
    # the same generation model and the comparison measured nothing.
    generation_model: str | None = field(default_factory=lambda: os.environ.get("SKYRAG_GENERATION_MODEL"))

    temperature: float = field(default_factory=lambda: _env_float("SKYRAG_TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("SKYRAG_MAX_TOKENS", 4096))
    # `_env_int_or_none`, not `_env_int`: this field is typed optional and every
    # backend treats None as "send no seed", so the loader has to be able to say
    # it. `SKYRAG_SEED=` (or `none`/`null`/`off`) requests an unseeded run.
    seed: int | None = field(default_factory=lambda: _env_int_or_none("SKYRAG_SEED", 34))
    timeout: int = field(default_factory=lambda: _env_int("SKYRAG_TIMEOUT", 300))
    max_retries: int = field(default_factory=lambda: _env_int("SKYRAG_MAX_RETRIES", 4))

    # Ollama-only: context window. Ignored by OpenRouter (server-side).
    num_ctx: int = field(default_factory=lambda: _env_int("SKYRAG_NUM_CTX", 22000))

    def model_for(self, role: str) -> str:
        """Model id for a given call site: 'expansion' | 'keyword' | 'generation'.

        An unknown role falls back to `self.model`, which is the right default --
        but 'generation' used to reach that fallback despite being one of the three
        roles named above, so an override set for it was accepted and discarded.
        """
        if role == "keyword" and self.keyword_model:
            return self.keyword_model
        if role == "expansion" and self.expansion_model:
            return self.expansion_model
        if role == "generation" and self.generation_model:
            return self.generation_model
        return self.model


@dataclass
class EmbeddingConfig:
    model_name: str = field(
        default_factory=lambda: _env("SKYRAG_EMBEDDING_MODEL", "all-mpnet-base-v2")
    )
    device: str = field(default_factory=lambda: _env("SKYRAG_DEVICE", "auto"))
    batch_size: int = field(default_factory=lambda: _env_int("SKYRAG_BATCH_SIZE", 32))
    # Finding 18. `resolve_device` grew a `strict` parameter so a run could demand
    # an accelerator and be refused rather than silently downgraded -- and then no
    # caller could ask for it, because `Embedder.model` resolves through
    # `resolved_device()` and this dataclass had no field to carry the request.
    # Half a guard is not a guard: the branch existed, was tested directly, and was
    # unreachable from the only path that resolves a device for embedding.
    #
    # Default False, because the shipped `.env` runs `SKYRAG_DEVICE=auto` on a
    # CPU-only torch build and defaulting to True would refuse every run on this
    # machine. It is the *GPU* run that needs the refusal, and that run is the one
    # that will set it.
    strict_device: bool = field(
        default_factory=lambda: _env_bool("SKYRAG_DEVICE_STRICT", False)
    )

    def resolved_device(self) -> str:
        return resolve_device(self.device, strict=self.strict_device)

    def device_fallback(self) -> str | None:
        """Why the last resolution did not get an accelerator, or None when it did.

        Resolves first and reads the module global immediately afterwards, because
        `LAST_DEVICE_FALLBACK` records whichever `resolve_device` call ran most
        recently: reading it without resolving reports some other caller's answer.
        Exposed as a method rather than left as a global for the same reason
        finding 18 was raised at all -- nothing outside `config.py` read it, so a
        downgrade reached no artifact.
        """
        self.resolved_device()
        return LAST_DEVICE_FALLBACK


@dataclass
class PathConfig:
    root: Path = PROJECT_ROOT
    documents: Path = field(default_factory=lambda: PROJECT_ROOT / _env("SKYRAG_DOCUMENTS_DIR", "documents"))
    embeddings: Path = field(default_factory=lambda: PROJECT_ROOT / _env("SKYRAG_EMBEDDINGS_DIR", "embeddings"))
    outputs: Path = field(default_factory=lambda: PROJECT_ROOT / _env("SKYRAG_OUTPUT_DIR", "outputs"))

    def ensure(self) -> None:
        self.embeddings.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)


@dataclass
class RetrievalConfig:
    # Courses kept per selected keyword (notebook used nlargest(7)).
    per_keyword: int = field(default_factory=lambda: _env_int("SKYRAG_PER_KEYWORD", 7))
    # Top-k for the naive pipeline (notebook used 10-20).
    top_k: int = field(default_factory=lambda: _env_int("SKYRAG_TOP_K", 20))
    # Drop LLM-selected keywords scoring below this.
    min_keyword_score: float = field(
        default_factory=lambda: _env_float("SKYRAG_MIN_KEYWORD_SCORE", 0.0)
    )
    # How many of the most frequent keywords to show the LLM when it picks.
    # Real catalogs have a long tail -- Coursera's `Skills` yields 8,901
    # distinct keywords, 6,330 of which occur exactly once. The whole list
    # will not fit in a prompt, and keywords outside the cut are unreachable,
    # so this trades prompt size against catalog coverage: at 600, 76% of
    # keyworded Coursera courses are reachable; at 1000, 81%; at 2000, 87%.
    max_vocabulary: int = field(default_factory=lambda: _env_int("SKYRAG_MAX_VOCABULARY", 1000))
    # Cap on keywords the LLM may return per platform. Bounding the reply is what
    # lets the token budget be sized in advance, so a compliant answer cannot be
    # truncated mid-object.
    #
    # This comment used to continue: "At per_keyword=7 this still admits ~280
    # candidate courses per platform, far more than a generation prompt consumes,
    # so the cap is not expected to bind in practice." The arithmetic was right and
    # the conclusion was false -- the generation prompt consumed *all* of them,
    # because nothing trimmed the merged list. 40 x 7 x 2 platforms = 560 courses at
    # a mean 801-token `Course.render()` is a 441,000-token prompt against a
    # 131,072-token endpoint limit, and the arm that selected keywords well was the
    # arm that overflowed. The sentence is recorded here rather than deleted because
    # it is exactly the kind of reasoning that stops someone from checking.
    # Bounded now by `Course.render_for_context` and by `max_context_courses` below.
    max_keywords: int = field(default_factory=lambda: _env_int("SKYRAG_MAX_KEYWORDS", 40))
    # Backstop on how many courses may reach a generation prompt, applied after the
    # round-robin merge. 600 is deliberately *above* the 560 the settings above can
    # produce, so at the shipped configuration it binds on no arm at all -- which is
    # the property the experiment needs. A cap that binds on one arm and not another
    # is a confound: truncating the context is itself a treatment, and two arms
    # seeing different amounts of it are not being asked the same question. Lower it
    # only deliberately, and record that it bound: `context_courses_dropped` is
    # exported per row for exactly that reason. `SKYRAG_MAX_CONTEXT_COURSES=` (or
    # none/off) removes the backstop entirely.
    max_context_courses: int | None = field(
        default_factory=lambda: _env_int_or_none("SKYRAG_MAX_CONTEXT_COURSES", 600)
    )
    # Platforms the run expects, declared rather than inferred from whichever
    # dataset files happen to be on disk. A hardcoded ["coursera", "edx"] in the
    # CLI previously made a missing Udemy dataset indistinguishable from a
    # deliberate two-platform design; the corpus deviation from the source paper's
    # three platforms has to be a stated decision, not an accident of the
    # filesystem.
    platforms: list[str] = field(
        default_factory=lambda: [
            p.strip() for p in _env("SKYRAG_PLATFORMS", "coursera,edx").split(",") if p.strip()
        ]
    )


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def load(cls) -> "Config":
        """Build config from the environment, reading .env first if present."""
        load_dotenv_if_present()
        return cls()


def load_dotenv_if_present(path: Path | None = None) -> None:
    """Minimal .env loader so we don't take a python-dotenv dependency.

    Existing environment variables always win.
    """
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)
