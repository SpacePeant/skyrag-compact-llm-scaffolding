"""Query-time keyword selection -- SKYRAG's distinguishing step.

The LLM is shown one platform's controlled vocabulary and asked which
keywords are relevant. Two properties matter:

1. Answers are validated against the real vocabulary, so a hallucinated
   keyword cannot reach retrieval.
2. Parsing tolerates the several shapes models actually return.

The notebook used `ast.literal_eval` on raw output, which raised the moment a
model added a sentence of preamble or wrapped its answer in a ```json fence.
Since the whole point of this port is swapping models, that had to be
hardened.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .data.adapters import Vocabulary
from .llm import GenerationParams, LLMBackend, LLMError
from .schemas import (
    STAGE_SELECTION,
    STAGE_SELECTION_V5,
    keyword_selection_schema,
    keyword_selection_schema_v4,
    keyword_selection_schema_v5,
    parse_and_validate,
)
from .schemas import render as render_schema

if TYPE_CHECKING:  # pragma: no cover - import cycle and torch cost at runtime
    from .embedding import Embedder

PROMPT = """You are an expert curriculum designer.

Below is the complete list of topic keywords available on {platform}.
Choose the keywords relevant to the learner's request.

Rules:
- Choose ONLY from the list below. Never invent a keyword.
- Score each from 0.0 to 1.0 by how relevant it is.
- Return AT MOST {max_keywords} keywords, the most relevant ones.
- Return ONLY a JSON object mapping keyword to score. No prose, no code fence.

Example:
{{"machine learning": 0.95, "statistics": 0.72}}

Available keywords on {platform}:
{vocabulary}

Learner's request:
{query}
"""

# B1 / Arm 3. The same request under a skill schema (S2).
#
# Deliberately the same instructions in the same order as PROMPT above, with
# the shape swapped and the schema appended. Arm 3 is meant to differ from
# Arm 2 in the *form* of the answer and in nothing else, so any rewording here
# that is not forced by the schema puts a second variable in the arm.
#
# Enforcement is prompt-and-parse per DEC-6: the schema is shown to the model
# and validated on the way back, never sent as a wire `response_format`.
PROMPT_SCHEMA_V1 = """You are an expert curriculum designer.

Below is the complete list of topic keywords available on {platform}.
Choose the keywords relevant to the learner's request.

Rules:
- Choose ONLY from the list below. Never invent a keyword.
- Score each from 0.0 to 1.0 by how relevant it is.
- Return AT MOST {max_keywords} keywords, the most relevant ones.
- Copy each keyword EXACTLY as written below. Do not correct, pluralise,
  expand or tidy the characters.
- Set `copied_verbatim` to true only if you copied the string from the list.
  If you wrote it from memory, set it to false. Answer honestly: this is
  recorded, and it does not change whether the keyword is accepted.
- Return ONLY a JSON object matching this schema. No prose, no code fence.

Schema:
{schema}

Example:
{{"platform": "{platform}", "keywords": [{{"keyword": "machine learning", "score": 0.95, "copied_verbatim": true}}]}}

Available keywords on {platform}:
{vocabulary}

Learner's request:
{query}
"""

# B1 / Arm 3 under V5. Deliberately `PROMPT_SCHEMA_V1` with exactly two edits,
# both forced by the payload shape: the `copied_verbatim` rule is gone because
# the field is gone, and the example is a flat map. Every other instruction is
# the same text in the same order, so a V1-vs-V5 comparison varies payload shape
# and nothing else. The "Copy each keyword EXACTLY" rule is KEPT -- V1 has it and
# Arm 2 does not, so removing it here would reintroduce the confound that made
# the V4 comparison uninterpretable.
PROMPT_SCHEMA_V5 = """You are an expert curriculum designer.

Below is the complete list of topic keywords available on {platform}.
Choose the keywords relevant to the learner's request.

Rules:
- Choose ONLY from the list below. Never invent a keyword.
- Score each from 0.0 to 1.0 by how relevant it is.
- Return AT MOST {max_keywords} keywords, the most relevant ones.
- Copy each keyword EXACTLY as written below. Do not correct, pluralise,
  expand or tidy the characters.
- Return ONLY a JSON object matching this schema. No prose, no code fence.

Schema:
{schema}

Example:
{{"platform": "{platform}", "keywords": {{"machine learning": 0.95, "statistics": 0.72}}}}

Available keywords on {platform}:
{vocabulary}

Learner's request:
{query}
"""

PROMPT_SCHEMA_V4 = """Choose relevant keywords for the query from the list below.
Return ONLY JSON matching the schema. No markdown code blocks, no commentary.

Rules:
- Choose ONLY from Available keywords on {platform}. Never invent keywords.
- Return AT MOST {max_keywords} keywords.

Schema:
{schema}

Available keywords on {platform}:
{vocabulary}

Learner's request:
{query}
"""


# Selection outcomes. Kept as values rather than inferred from empty fields,
# because three of these look identical downstream -- an empty `keywords` dict --
# while meaning completely different things for the measurement.
OUTCOME_OK = "ok"
OUTCOME_EMPTY = "empty"                  # model answered, chose nothing: a real result
OUTCOME_BACKEND_ERROR = "backend_error"  # never answered: infrastructure
OUTCOME_TRUNCATED = "truncated"          # cut off by a budget too small: harness
OUTCOME_OVERRUN = "overrun"              # cut off after ignoring the cap: capability
OUTCOME_UNPARSEABLE = "unparseable"      # answered fully, emitted garbage: capability
OUTCOME_NO_VOCABULARY = "no_vocabulary"  # never asked: the platform has no keywords
# B1 / Arm 3 only. Answered fully with something the harness could read, but not
# in the shape the schema asked for.
#
# Kept apart from OUTCOME_UNPARSEABLE rather than folded into it, for the reason
# this taxonomy exists at all: they are different findings. "Could not be read"
# and "could be read, but ignored the shape it was given" say different things
# about what a schema does to an 8B, and Arm 3's entire purpose is to measure
# that. Collapsing them would make neither countable.
#
# It is a capability signal, so it is NOT in HARNESS_OUTCOMES: the model was
# shown the schema and did not follow it. `KeywordSelection.schema_violations`
# carries the specific reasons, so the count can always be traced to a cause.
OUTCOME_SCHEMA_VIOLATION = "schema_violation"

# Outcomes not attributable to the model's ability to select keywords. These are
# excluded from capability metrics the same way an HTTP 503 would be.
#
# OUTCOME_OVERRUN is deliberately NOT here. Both it and OUTCOME_TRUNCATED arrive
# as finish_reason="length", and the temptation is to treat the wire signal as
# the whole story -- but a reply cut off after being given three times the room a
# compliant answer needs is the model disregarding an instruction, which is the
# capability being measured. Excluding it under-counts the smaller model's
# deficit, which is the same error as the original defect with its sign flipped.
HARNESS_OUTCOMES = frozenset(
    {OUTCOME_BACKEND_ERROR, OUTCOME_TRUNCATED, OUTCOME_NO_VOCABULARY}
)

# Tokens to allow per returned keyword, plus fixed overhead for the braces and
# whitespace. Sizing the budget from the requested cap is what makes truncation
# structurally impossible rather than merely less likely: raising max_tokens by
# hand only moves the cliff, because output length scales with how many keywords
# the model decides to emit.
_TOKENS_PER_KEYWORD = 32
# B1 / Arm 3. One S2 entry is `{"keyword": "...", "score": 0.95,
# "copied_verbatim": true}` -- roughly twice the tokens of a `"key": score`
# pair once the field names are counted. Kept generous for the same reason the
# Arm 2 figure is: the budget exists to make truncation of a *compliant* answer
# structurally impossible, and the cost of being generous is nothing, because
# what a run pays for is tokens emitted, not tokens allowed.
_TOKENS_PER_KEYWORD_SCHEMA = 64
_TOKEN_OVERHEAD = 128


@dataclass
class KeywordSelection:
    """One platform's keyword-selection outcome.

    Four outcomes are deliberately kept apart, because the thesis measures
    different things with each:

    * `keywords`      -- usable: present in the window shown to the model.
    * `hallucinated`  -- invented: absent from the platform's corpus entirely,
                         under exact match AND after normalisation. This is the
                         model-capability signal.
    * `near_miss`     -- metric M2.2: fails exact match, matches once case,
                         punctuation and plurals are folded. The model had the
                         concept and botched the transcription, which is a
                         different failure from not knowing it. Kept out of
                         `hallucinated` so M2.1 measures inventions only.

                         **Measured, not repaired.** These keywords do not reach
                         retrieval. Correcting them would delete the failure mode
                         under study and move every downstream number toward the
                         hypothesis.
    * `out_of_window` -- real, but outside the top-`max_vocabulary` slice the
                         model was shown. A retrieval-coverage limitation of
                         this harness, NOT a model failure; counting it as
                         hallucination overstates the hallucination rate.
    * `removed_by_cleaning`
                      -- real in the corpus, and deliberately excluded from the
                         vocabulary as a publisher rather than a topic. Also not
                         a model failure, and for the same reason: the keyword
                         exists in the data. Kept apart from `out_of_window`
                         because the two are fixed by different things -- one by
                         raising `max_vocabulary`, the other only by revisiting
                         what cleaning removes.
    * `error`         -- the backend never answered. Infrastructure, not
                         capability. An empty `keywords` with `error=None` means
                         the model genuinely chose nothing, which is a real and
                         different measurement.

    `outcome` disambiguates the three separate events that all present as an
    empty `keywords` dict:

    * OUTCOME_EMPTY       -- answered, selected nothing. A model result.
    * OUTCOME_BACKEND_ERROR -- never answered. Infrastructure.
    * OUTCOME_TRUNCATED   -- answered, but max_tokens cut the reply mid-object so
                             it could not be parsed, AND the budget granted was
                             below what a compliant answer needs. Ours.
    * OUTCOME_OVERRUN     -- cut off the same way, but with room for a compliant
                             answer already granted. The model disregarded the
                             stated cap, which is capability, not budget. This is
                             what dropped 6,641 coursera courses from the
                             2026-08-17 live run while it reported status ok.
    * OUTCOME_UNPARSEABLE -- answered completely, emitted something unparseable.
                             Like OVERRUN this IS a capability signal and must
                             count against the model.
    """

    platform: str
    keywords: dict[str, float]      # validated, usable for retrieval
    hallucinated: list[str] = field(default_factory=list)
    # Metric M2.2. Maps what the model emitted to the corpus keyword it resolves
    # to once case, punctuation and plurals are folded. A dict rather than a list
    # because the pair is the diagnosis: "machine-learning" -> "machine learning"
    # says what went wrong, where the emitted string alone does not.
    near_miss: dict[str, str] = field(default_factory=dict)
    # Metric M1.2. Where each usable keyword sat in the frequency-sorted window
    # the model was shown -- 0 is the most frequent keyword in the platform.
    #
    # Rank is knowable without a reference set because the window is deterministic:
    # `build_vocabulary` returns `most_common()` order and the trim takes a prefix
    # of it, so a keyword's position is a property of the corpus, not of the run.
    # Selections clustered at low ranks are the signature of position bias -- a
    # model reading the head of a 1,000-item list and skimming the tail, which is
    # long-context recall failing in a specific and diagnosable way rather than
    # merely failing.
    selected_ranks: list[int] = field(default_factory=list)
    out_of_window: list[str] = field(default_factory=list)
    removed_by_cleaning: list[str] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None
    outcome: str = OUTCOME_OK

    # --- B1 / Arm 3. Empty on every arm that runs no schema. --------------
    #
    # What the schema rejected, in the reply's own terms. A violation count with
    # no reasons beside it is a number nobody can check, and this is the field
    # that makes `selection_schema_violation` auditable after the run.
    schema_violations: list[str] = field(default_factory=list)
    # The model's own claim, per keyword, about whether it copied the string or
    # recalled it. Keyed by the keyword as emitted, before normalisation, so it
    # still joins to `hallucinated` and `near_miss` -- which is the entire point:
    # M2.1 split by this field answers "when the model invents a keyword, does it
    # know it is inventing?"
    #
    # An INSTRUMENT, never a control. Nothing in this module or downstream of it
    # may drop, correct or deprioritise a keyword because of this flag. Doing so
    # would delete the failure mode Arm 3 exists to observe and move every
    # downstream number toward the hypothesis, exactly as repairing near-misses
    # would.
    copied_verbatim: dict[str, bool] = field(default_factory=dict)

    # X1 -- what this one selection call cost.
    #
    # Until these fields existed the instrument could not see the selection
    # stage's cost -- `BenchmarkMetric.prompt_tokens` comes from the final
    # `generate` call
    # alone, so expansion, step-back and every per-platform selection were timed
    # and their tokens counted nowhere.
    #
    # None means *not reported*, never zero. A backend that never answered and a
    # backend that answered without a usage block are both unknown, and a stage
    # sum containing an unknown is unknown -- the rule blocker 3.4 exists to
    # enforce. Zero would be a measurement, and a plausible undercount of cost
    # is worse than an absent one because it can be averaged.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def failed(self) -> bool:
        """Backend failure, as opposed to a legitimately empty selection."""
        return self.error is not None

    @property
    def harness_failure(self) -> bool:
        """True when the empty result is our fault, not the model's.

        Capability metrics must exclude these rows; including them attributes a
        token-budget or network problem to the model under test, which is the
        exact comparison this repository exists to make.
        """
        return self.outcome in HARNESS_OUTCOMES

    def ranked(self, min_score: float = 0.0) -> list[str]:
        """Usable keywords, best first. Ties broken by keyword (L1).

        This list is not merely displayed: `keyword_buckets` consumes it in
        order, and first-keyword-wins decides every retrieved course's
        `via_keyword`. Models return round numbers -- 0.9, 0.8, 0.95 -- so ties
        here are common rather than exotic, and without a tie-break the winner
        was whichever key the model's JSON happened to list first. That makes an
        exported attribution column depend on reply *formatting* rather than on
        the scores the model assigned.
        """
        return [
            k
            for k, v in sorted(self.keywords.items(), key=lambda kv: (-kv[1], kv[0]))
            if v >= min_score
        ]


def try_parse_keyword_response(text: str) -> tuple[dict[str, float], bool]:
    """Parse a keyword response, reporting whether it was parseable at all.

    The boolean is the point of this function. `{}` is an ambiguous result: it is
    what a model returns when it deliberately selects nothing, AND what a failed
    parse returns. Those are opposite findings -- one is a model result, the other
    is a defect -- so the caller needs them told apart.

    Handles: a bare JSON object, a ```json fence, a list of single-key dicts
    (the shape the notebook's prompt actually asked for), and a plain
    comma-separated list.
    """
    if not text or not text.strip():
        return {}, False

    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()

    # Each balanced {...} or [...] span, in the order they open, tried until one
    # parses. This replaced a pair of greedy regexes (`\{.*\}`, `\[.*\]`) that
    # matched from the first opening delimiter to the *last* closing one anywhere
    # in the text, so one sentence of trailing prose containing a brace -- or a
    # second fenced block -- swallowed the real object into an unparseable span and
    # the reply was recorded as OUTCOME_UNPARSEABLE against the model.
    #
    # Scanning rather than making the regex non-greedy is deliberate. `\{.*?\}`
    # fixes the trailing-prose case and breaks a nested one: it stops at the first
    # inner `}`, cutting `{"a": {"b": 1}}` down to `{"a": {"b": 1}`. Only matching
    # delimiters can tell those apart, and a model that emits nested JSON is
    # exactly the non-compliant behaviour under measurement.
    # A span that parses to nothing does not end the search -- a stray `{}` in
    # prose must not shadow a real object later in the text -- but it is still a
    # successful parse, and if nothing better turns up the answer is "the model
    # returned a well-formed empty selection", not "the reply was garbage". Those
    # are OUTCOME_EMPTY and OUTCOME_UNPARSEABLE respectively: a real model result
    # against a model failure, so collapsing them would recreate the ambiguity this
    # function's boolean exists to remove.
    saw_empty = False
    for blob in _balanced_spans(cleaned):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = _coerce(loader(blob))
            except (ValueError, SyntaxError, TypeError):
                continue
            if parsed:
                return parsed, True
            saw_empty = True
            break
    if saw_empty:
        return {}, True

    # Last resort: a plain comma-separated list, scored neutrally -- for models
    # that answer "python, machine learning" instead of JSON.
    #
    # It must NOT fire on text carrying JSON structure. Reaching here with braces
    # or brackets present means structured output was attempted and failed to
    # parse, and splitting that on commas manufactures partial garbage out of a
    # failed stage: '{"a": 1.0, "b": 0.5' would yield the keys '{"a": 1.0' and
    # 'b": 0.5'. A length gate does not catch this -- a short truncation slips
    # under any threshold -- so the test is structural.
    #
    # The structural test cannot save a truncated *plain list*: "python, data
    # science, statis" is indistinguishable, by inspection, from a complete list
    # whose last entry happens to be short, and salvaging it fabricates the
    # keyword 'statis' which then scores as a model hallucination. Nothing in
    # this function can decide that -- the deciding evidence is the backend's
    # finish_reason, so `select_keywords` returns OUTCOME_TRUNCATED before ever
    # reaching here. Callers that parse a reply without checking `.truncated`
    # first reinstate the defect.
    structural = set("{}[]")
    if "," in cleaned and len(cleaned) < 2000 and not (structural & set(cleaned)):
        parts = [p.strip().strip("\"'-*").lower() for p in cleaned.split(",")]
        salvaged = {p: 1.0 for p in parts if p and len(p) < 60}
        if salvaged:
            return salvaged, True

    return {}, False


def _balanced_spans(text: str) -> list[str]:
    """Every balanced `{...}` or `[...]` span in `text`, outermost first.

    Delimiters inside string literals are skipped, so a keyword containing a brace
    cannot unbalance the scan. Quote tracking handles both `"` and `'` because the
    fallback loader is `ast.literal_eval`, for which single-quoted output is valid.

    Quote tracking only applies **inside** a span, and that restriction is the whole
    correctness argument. Honouring quotes outside one made any English contraction
    in a model's preamble fatal: the apostrophe in `Here's the JSON: {"python": 0.9}`
    opened a string that never closed, the scanner consumed the rest of the reply as
    string contents, and the function returned zero spans. `select_keywords` then
    booked a perfectly good answer as OUTCOME_UNPARSEABLE -- "the model answered
    fully and emitted garbage" -- which is a capability signal, deliberately excluded
    from HARNESS_OUTCOMES. So a harness parsing defect was charged to the model under
    test, and charged *preferentially to whichever model is chattier*, biasing the
    scale comparison in the direction that flatters the hypothesis.
    Prose outside a span contains no delimiters that need protecting, so there is
    nothing for quote tracking to do there.

    Nested spans are not returned separately: the outermost balanced span is the
    candidate, and if it fails to parse the next *sibling* span is tried. Returning
    inner spans as well would let `{"a": {"b": 1}}` fall back to parsing `{"b": 1}`
    and report `b` as the model's selection -- salvaging a keyword out of a reply
    that was never a valid answer, which is the fabrication this module's comma
    fallback is already gated to prevent.
    """
    spans: list[str] = []
    openers = {"{": "}", "[": "]"}
    depth = 0
    start = -1
    quote: str | None = None
    escaped = False

    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        # Only once a span is open. See the docstring: outside one, an apostrophe is
        # just an apostrophe, and treating it as a delimiter discarded the reply.
        if depth > 0 and ch in "\"'":
            quote = ch
            continue
        if ch in openers:
            if depth == 0:
                start = i
            depth += 1
        elif ch in openers.values():
            if depth:
                depth -= 1
                if depth == 0 and start != -1:
                    spans.append(text[start : i + 1])
                    start = -1

    return spans


def balanced_json_spans(text: str) -> list[str]:
    """Public name for the balanced-brace scanner above.

    Exists so `schemas.parse_and_validate` can extract JSON with the *same*
    scanner Arm 2's keyword parser uses. DEC-6's
    resolution is that every arm shares one enforcement mechanism; a second
    extractor written for Arm 3 would be a second mechanism, and the arms would
    then differ by how their replies are read as well as by treatment.
    """
    return _balanced_spans(text)


def parse_keyword_response(text: str) -> dict[str, float]:
    """Best-effort parse of an LLM keyword response.

    Thin wrapper over `try_parse_keyword_response` for callers that do not need
    to distinguish "selected nothing" from "could not be parsed".
    """
    return try_parse_keyword_response(text)[0]


def _coerce(obj) -> dict[str, float]:
    """Normalize the parsed shapes into {keyword: score}."""
    out: dict[str, float] = {}

    def add(key, value=1.0) -> None:
        key = str(key).strip().lower()
        if not key:
            return
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = 1.0

    if isinstance(obj, dict):
        for k, v in obj.items():
            add(k, v)
    elif isinstance(obj, (list, tuple)):
        # tuple arises from literal_eval on "{'a': 1}, {'b': 2}".
        for item in obj:
            if isinstance(item, dict):
                for k, v in item.items():   # [{'Art': 0.42}, ...]
                    add(k, v)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                add(item[0], item[1])
            else:
                add(item)
    return out


def validate(selected: dict[str, float], vocabulary: dict[str, int]) -> tuple[dict, list[str]]:
    """Split selections into those present in the vocabulary and those not.

    This is the anti-hallucination guarantee: retrieval only ever sees
    keywords that exist in the data.
    """
    keep, rejected = {}, []
    for keyword, score in selected.items():
        if keyword in vocabulary:
            keep[keyword] = score
        else:
            rejected.append(keyword)
    return keep, rejected


# Characters that carry no meaning for a keyword's identity. Hyphens, slashes and
# punctuation are how the same concept gets written two ways -- "machine-learning",
# "machine learning", "machine/learning" -- and none of those differences is a
# statement about what the model knew.
_SEPARATORS = re.compile(r"[^a-z0-9]+")


def _singularise(token: str) -> str:
    """Crude English singular. Deliberately crude, and only for matching.

    This never touches a keyword that reaches retrieval; it exists solely to
    decide whether two strings name the same concept. A false merge costs one
    keyword moved from the invented bucket to the near-miss bucket, which is a
    conservative error: it understates hallucination rather than inflating it,
    and inflating it is the direction that would flatter this project's thesis.
    """
    if len(token) < 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalise_keyword(keyword: str) -> str:
    """Fold a keyword to its concept, for near-miss comparison only.

    The ladder, in order: case and surrounding whitespace (already applied by
    `_coerce` for model output, applied again here because corpus keys do not go
    through it), then punctuation and separators to spaces, then whitespace
    collapse, then a per-token singular fold.
    """
    tokens = _SEPARATORS.sub(" ", str(keyword).strip().lower()).split()
    return " ".join(_singularise(t) for t in tokens)


def split_near_misses(
    hallucinated: list[str], vocabulary: Vocabulary
) -> tuple[dict[str, str], list[str]]:
    """Separate transcription failures from inventions. Metric M2.2.

    Returns `(near_miss, still_hallucinated)`, where `near_miss` maps what the
    model emitted to the corpus keyword it resolves to under `normalise_keyword`.

    **This must run before anything reads `hallucinated` as final.** M2.1 and
    M2.2 measure different capabilities -- "did not know the concept" versus "knew
    it and botched the transcription". Run this after the hallucination
    count is taken and both numbers are wrong: M2.1 keeps the near-misses it
    should have shed, and M2.2 reports zero.

    **A near-miss is measured, not repaired.** The keyword stays out of retrieval
    exactly as before. Silently correcting the model's spelling would remove the
    failure mode this experiment exists to measure, and would change every
    downstream number in the direction of the hypothesis -- the retrieved set, the
    citations, the grounding rate. The bucket moves; the pipeline does not.

    The index is built lazily and only over the corpus, because on a real run this
    sits inside the timed region that finding 14 already measured.
    """
    if not hallucinated:
        return {}, hallucinated

    index: dict[str, str] = {}
    for keyword in vocabulary.corpus:
        index.setdefault(normalise_keyword(keyword), keyword)

    near_miss: dict[str, str] = {}
    still: list[str] = []
    for keyword in hallucinated:
        resolved = index.get(normalise_keyword(keyword))
        # `resolved != keyword` guards the degenerate case: a keyword that folds
        # onto itself and is in the corpus would already have matched exactly
        # upstream, so reaching here means the two really are different strings.
        if resolved is not None and resolved != keyword:
            near_miss[keyword] = resolved
        else:
            still.append(keyword)
    return near_miss, still


def classify(
    selected: dict[str, float],
    trimmed: dict[str, int],
    vocabulary: Vocabulary | dict[str, int],
) -> tuple[dict[str, float], list[str], list[str], list[str]]:
    """Four-way split of a model's keyword choices.

    `validate` alone cannot distinguish these, because it sees only the trimmed
    window: every keyword outside it simply fails the membership test, whatever
    the reason. Widening the check to the eligible pool separates a real keyword
    below the cut from an invented one -- and widening it again to the raw corpus
    separates both from a keyword that cleaning deliberately removed.

    That last test is the one whose absence mattered. `full` used to receive an
    already-cleaned dict, so `aws` -- present in thousands of real courses and
    stripped as a publisher -- was in neither argument and landed in the invented
    bucket by elimination, inflating the headline hallucination rate in the
    direction that flatters the finding.

    Returns (usable, hallucinated, out_of_window, removed_by_cleaning).

    **`hallucinated` is provisional.** This function tests exact membership only,
    so a keyword the model spelled differently from the corpus lands here. Pass it
    through `split_near_misses` before reading it as an invention count -- that is
    what separates M2.1 from M2.2, and `select_keywords` does it for you.
    """
    vocab = vocabulary if isinstance(vocabulary, Vocabulary) else Vocabulary.uncleaned(vocabulary)

    keep, hallucinated, out_of_window, removed = {}, [], [], []
    for keyword, score in selected.items():
        if keyword in trimmed:
            keep[keyword] = score
        elif keyword in vocab.eligible:
            out_of_window.append(keyword)   # eligible, just below the window cut
        elif keyword in vocab.removed:
            removed.append(keyword)         # in the corpus, cleaned out on purpose
        else:
            hallucinated.append(keyword)    # nowhere in the corpus
    return keep, hallucinated, out_of_window, removed


# --- Candidate ranking strategies (2026-09-22) -------------------------------
#
# Paper #1 4.7.3 claims the candidate-injection result is "retriever-agnostic"
# across dense embeddings, sparse lexical scoring, RRF hybrid and lexical
# heuristics. Only the last has ever run. These names make the claim testable
# at k=100 -- the depth the 2026-09-21 pool-size pilot established as the only
# one worth testing, having shown k=1000 collapses retrieval 74% through
# overruns whatever the ranker.
RANKER_LEXICAL = "lexical"
RANKER_DENSE = "dense"
RANKER_HYBRID = "hybrid"
# Frequency-weighted lexical ranking -- the section 2.7 correction, C'.
#
# The lexical ranker scores relevance and consults catalog frequency only as a
# *tiebreak*, so a keyword attached to one course outranks a keyword attached to
# two hundred whenever it matches the query marginally better. Measured against
# the live Coursera vocabulary, the 100 candidates it injects have a **median
# corpus frequency of 1**, and 83.6% attach to 7 or fewer courses. With
# `per_keyword = 7` such a keyword can return at most one course however well the
# model chooses it: the retrieval ceiling is set before the model acts. That is
# the measured cause of Condition C keeping *more* valid keywords (989 vs 764)
# while retrieving *fewer* courses (3,378 vs 3,991).
#
# This strategy makes frequency a term in the objective rather than a tiebreak,
# scoring `relevance * log1p(frequency)`. It is a **separate strategy and not an
# edit to the lexical branch**, because every published Condition C cell ranked
# lexically and changing that branch would rewrite what those artifacts mean.
RANKER_FREQ = "freq"
# The other half of the C' proposal -- "or drop candidates below a frequency
# floor" -- and, measured, the half that works.
#
# `RANKER_FREQ` only reorders; it cannot conjure common keywords into a
# vocabulary that has few. And this vocabulary has few: **95.6% of Coursera's
# 12,854 eligible keywords attach to 7 or fewer courses, median 1.** The
# published lexical ranker already improves on that baseline (82.2% of its
# injected set is <= 7, against the vocabulary's 95.6%) because it breaks
# relevance ties by frequency descending. Weighting relevance by log1p(frequency)
# on top of that moves the injected set from 82.2% to 79.1%. A floor moves it to
# **0.0%**, because it changes the eligible pool rather than its order.
#
# The floor is `per_keyword + 1`, not an arbitrary constant: with `per_keyword =
# 7`, a keyword attached to 7 or fewer courses cannot fill its own retrieval slot,
# so excluding it removes exactly the candidates that are structurally incapable
# of contributing. Measured over the 50 benchmark queries, injected-set retrieval
# ceiling goes 307.9 -> 700.0, which is the theoretical maximum of 100 x 7.
#
# **A ceiling is not a prediction.** The floor shrinks Coursera's eligible pool
# from 12,854 to ~565, so the surviving candidates necessarily match the query
# *less* well. Whether a higher ceiling on worse-matched keywords retrieves more
# usable courses, and what it does to grounding, is what the run decides. Nothing
# here claims it will.
RANKER_FLOOR = "floor"
# `per_keyword` is fixed at 7 for every cell in the study, so the floor is a
# constant rather than a plumbed parameter -- threading a value that never varies
# would add a stamp nobody reads and a knob nobody turns.
RANKER_FLOOR_MIN_FREQUENCY = 8
RANKER_STRATEGIES = (
    RANKER_LEXICAL,
    RANKER_DENSE,
    RANKER_HYBRID,
    RANKER_FREQ,
    RANKER_FLOOR,
)
# The two strategies that need an embedder, and the two that do not. Named
# because three call sites branch on the distinction and `!= RANKER_LEXICAL` is
# no longer the same question now that a fourth strategy exists.
RANKER_SEMANTIC_STRATEGIES = (RANKER_DENSE, RANKER_HYBRID)
RANKER_LEXICAL_STRATEGIES = (RANKER_LEXICAL, RANKER_FREQ, RANKER_FLOOR)

# Which text the injected candidates are ranked against. Every published cell
# ranked against the Stage 1 rewrite, the same text the selector then reads, so
# a different rewrite changed both what the selector reads and which candidates
# it is shown. The 2026-10-21 pre-submission review (A7) points out that the rewrite's
# effect under injection is therefore not attributable to the text alone.
# `query` ranks against the learner's original query instead, which holds the
# candidate list fixed across rewrites while the selector still reads the
# rewrite, so the two paths can be separated.
CANDIDATE_SOURCE_REWRITE = "rewrite"
CANDIDATE_SOURCE_QUERY = "query"
CANDIDATE_SOURCES = (CANDIDATE_SOURCE_REWRITE, CANDIDATE_SOURCE_QUERY)

# Reciprocal-rank-fusion constant -- Cormack, Clarke & Buettcher, "Reciprocal
# Rank Fusion outperforms Condorcet and individual Rank Learning Methods",
# SIGIR 2009. 60 is the value that paper uses and the one every later
# implementation inherits; it is named rather than inlined so a run that
# changes it cannot do so invisibly.
RRF_K = 60


def _rrf(rank: int) -> float:
    """Reciprocal rank fusion weight for a 0-based rank."""
    return 1.0 / (RRF_K + rank + 1)


def _rank_candidates_semantic(
    query: str,
    eligible: dict[str, int],
    top_k: int,
    *,
    strategy: str,
    embedder: "Embedder | None",
) -> dict[str, int]:
    """Dense or hybrid candidate ranking. See `rank_candidate_keywords`.

    Dense scores every eligible keyword by cosine against the query embedding.
    Hybrid fuses that ordering with the lexical one by RRF, which needs no
    score calibration between the two -- the lexical scorer's 10.0/2.0/5.0
    weights are not on a cosine scale and normalising them would invent a
    correspondence that does not exist.

    Returns the same shape as the lexical branch: `keyword -> catalog
    frequency`, capped at `top_k`. The value stays the frequency and not the
    relevance score, because that is what the prompt renderer downstream reads
    and what every existing artifact stamps.
    """
    if strategy not in RANKER_SEMANTIC_STRATEGIES:
        raise ValueError(
            f"{strategy!r} is not a semantic ranker; expected one of "
            f"{RANKER_SEMANTIC_STRATEGIES}"
        )
    if embedder is None:
        raise ValueError(
            f"candidate ranker {strategy!r} needs an embedder. Falling back to "
            "lexical here would export a ranker stamp the numbers contradict."
        )

    import numpy as np

    keywords = list(eligible)
    matrix = embedder.encode(keywords, show_progress=False)
    query_vector = embedder.encode_one(query)

    # Cosine, computed explicitly rather than through a helper, because the
    # keyword matrix is not the course matrix and does not go through
    # `EmbeddingStore`'s normalisation.
    norms = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query_vector))
    # A zero vector has no direction, so its similarity is undefined; scoring
    # it 0.0 ranks it last without claiming it is orthogonal to the query.
    denominator = norms * query_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = np.where(denominator > 0, (matrix @ query_vector) / denominator, 0.0)

    dense_order = sorted(
        range(len(keywords)),
        # Frequency then keyword break ties, so two keywords at identical
        # cosine order deterministically rather than by argsort accident.
        key=lambda i: (-float(cosines[i]), -eligible[keywords[i]], keywords[i]),
    )
    if strategy == RANKER_DENSE:
        chosen = dense_order[:top_k]
        return {keywords[i]: eligible[keywords[i]] for i in chosen}

    # Hybrid: fuse against the lexical ordering over the same candidate set.
    lexical = rank_candidate_keywords(
        query, eligible, top_k=len(eligible), strategy=RANKER_LEXICAL
    )
    lexical_rank = {kw: r for r, kw in enumerate(lexical)}
    dense_rank = {keywords[i]: r for r, i in enumerate(dense_order)}

    fused = sorted(
        keywords,
        key=lambda kw: (
            -(_rrf(lexical_rank[kw]) if kw in lexical_rank else 0.0)
            - (_rrf(dense_rank[kw]) if kw in dense_rank else 0.0),
            -eligible[kw],
            kw,
        ),
    )
    return {kw: eligible[kw] for kw in fused[:top_k]}


def rank_candidate_keywords(
    query: str,
    eligible: dict[str, int],
    top_k: int = 100,
    *,
    strategy: str = RANKER_LEXICAL,
    embedder: "Embedder | None" = None,
) -> dict[str, int]:
    """Rank `eligible` catalog keywords by relevance to `query`.

    Uses lexical token overlap and keyword containment scoring against query terms,
    falling back to global catalog frequency to fill remaining slots up to `top_k`.

    **`strategy` defaults to lexical and every published cell used that default.**
    Paper #1 section 4.7.3 asserts that the candidate-injection finding is
    "retriever-agnostic" and lists dense embeddings, BM25, RRF hybrid and
    lexical heuristics -- but only this lexical ranker has ever been run, so the
    claim is an assertion rather than a result. `RANKER_DENSE` and
    `RANKER_HYBRID` exist so it can be tested; they change nothing until asked
    for by name, and the lexical scoring below is what it was.

    `RANKER_FREQ` is C', the section 2.7 correction: the same relevance scorer,
    but weighted by `log1p(catalog frequency)` so that a keyword attached to one
    course stops outranking a keyword attached to two hundred. It reaches the
    same scoring loop rather than a branch of its own, which is what makes the
    two orderings comparable -- they differ in the sort key and in nothing else.

    The dense strategies need an `embedder`. They raise rather than silently
    falling back to lexical: a run that asked for dense ranking and got lexical
    would export a `candidate_ranker` stamp that its own numbers contradict,
    which is the provenance failure `serving_precision` exists to prevent.
    """
    if not eligible:
        return {}

    if strategy in RANKER_SEMANTIC_STRATEGIES:
        return _rank_candidates_semantic(
            query, eligible, top_k, strategy=strategy, embedder=embedder
        )
    if strategy not in RANKER_LEXICAL_STRATEGIES:
        raise ValueError(
            f"unknown candidate ranker {strategy!r}; expected one of {RANKER_STRATEGIES}"
        )

    import math
    import re

    if strategy == RANKER_FLOOR:
        # Applied to the *pool*, before scoring, which is the whole difference
        # between this and RANKER_FREQ. If every candidate is below the floor the
        # pool is not emptied: an empty injected set would silently turn Condition
        # C back into unassisted Arm 2 while still stamping `candidate_injection`.
        # Falling back to the unfloored pool keeps the stamp honest.
        floored = {k: c for k, c in eligible.items() if c >= RANKER_FLOOR_MIN_FREQUENCY}
        eligible = floored or eligible

    query_lower = query.lower()
    query_tokens = set(re.findall(r"\b\w+\b", query_lower))

    scored_items = []
    for kw, count in eligible.items():
        kw_lower = kw.lower()
        kw_tokens = set(re.findall(r"\b\w+\b", kw_lower))

        rel_score = 0.0
        if kw_lower and kw_lower in query_lower:
            rel_score += 10.0

        common = query_tokens.intersection(kw_tokens)
        if common:
            rel_score += len(common) * 2.0
            if kw_tokens.issubset(query_tokens):
                rel_score += 5.0

        # C': frequency enters the objective instead of breaking ties. log1p and
        # not the raw count, so a 200-course keyword is worth ~5x a 1-course one
        # rather than 200x -- unweighted frequency would stop ranking relevance
        # and start ranking popularity, which is the opposite failure.
        #
        # The relevance *tier* is deliberately preserved: a keyword with no query
        # match never outranks one with a match, however common it is. Only the
        # order *within* the matched tier changes. Below that tier both
        # strategies fall back to frequency, so they agree on every query whose
        # matches do not fill `top_k`.
        rank_score = rel_score * math.log1p(count) if strategy == RANKER_FREQ else rel_score

        scored_items.append((rel_score > 0.0, rank_score, count, len(kw), kw, count))

    scored_items.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    return {item[4]: item[5] for item in scored_items[:top_k]}


def select_keywords(
    backend: LLMBackend,
    query: str,
    platform: str,
    vocabulary: Vocabulary | dict[str, int],
    *,
    model: str | None = None,
    max_vocabulary: int = 1000,
    max_keywords: int = 40,
    params: GenerationParams | None = None,
    schema_enabled: bool = False,
    schema_exemplar: bool = False,
    platforms: list[str] | None = None,
    sweep_version: str = "v1",
    candidate_injection: bool = False,
    candidate_limit: int = 100,
    candidate_ranker: str = RANKER_LEXICAL,
    embedder: "Embedder | None" = None,
    candidate_query: str | None = None,
) -> KeywordSelection:
    """Ask the LLM which of `vocabulary`'s keywords fit `query`.

    `candidate_query`, when given, is the text the injected candidates are
    ranked against in place of `query`; the selector prompt still uses `query`.
    See `CANDIDATE_SOURCES`.

    Pass a `Vocabulary` to have cleaning-removed keywords scored as what they
    are. A plain dict is accepted and treated as a vocabulary with no cleaning
    applied -- true for callers that never cleaned, and the reason
    `build_platform_vocabulary` exists for the ones that did.

    `schema_enabled` is the Arm 3 treatment (B1): the S2 skill schema is
    rendered into the prompt and the reply is validated against it. Everything
    else -- the model, the vocabulary window, the cap, the token budget, the
    scoring, what counts as hallucinated -- is held identical to Arm 2, because
    the arm measures the effect of the schema and nothing may vary beside it.

    `candidate_injection` enables dynamic candidate ranking against `query`
    (RAG-for-Keywords) before slicing to `candidate_limit` terms.
    `candidate_ranker` chooses which ranker does that slicing; it is read
    **only** when `candidate_injection` is on, because nothing ranks otherwise.
    The dense strategies need `embedder` and raise without one rather than
    quietly serving lexical results under a dense stamp.

    `platforms` is the full set of loaded platforms, used for S2's `platform`
    enum. Defaults to this call's platform alone, which is correct for a
    single-platform caller and wrong only if a caller passes nothing while
    expecting the model to be told about its siblings.
    """
    vocab = vocabulary if isinstance(vocabulary, Vocabulary) else Vocabulary.uncleaned(vocabulary)

    if not vocab.eligible:
        # No keywords exist for this platform. A corpus condition, not a model
        # choice -- so it must not be recorded as one.
        return KeywordSelection(
            platform=platform,
            keywords={},
            error=f"platform {platform!r} has an empty vocabulary",
            outcome=OUTCOME_NO_VOCABULARY,
        )

    # Very large vocabularies blow the context window; rank by candidate relevance
    # or keep the most frequent.
    if candidate_injection:
        trimmed = rank_candidate_keywords(
            candidate_query if candidate_query is not None else query,
            vocab.eligible,
            top_k=candidate_limit,
            strategy=candidate_ranker,
            embedder=embedder,
        )
    else:
        trimmed = dict(list(vocab.eligible.items())[:max_vocabulary])
    # `flat_payload` decides two things that must move together: how the reply is
    # read, and how the completion budget is sized. A V5 answer is Arm 2-shaped,
    # so sizing it at the schema rate would hand Arm 3 twice the room it needs
    # and make its overrun count incomparable with Arm 2's.
    flat_payload = False
    exemplar_stage = STAGE_SELECTION
    if schema_enabled:
        if sweep_version == "v5":
            active_schema = keyword_selection_schema_v5(platforms or [platform], max_keywords)
            prompt_template = PROMPT_SCHEMA_V5
            flat_payload = True
            exemplar_stage = STAGE_SELECTION_V5
        elif sweep_version == "v4":
            active_schema = keyword_selection_schema_v4(platforms or [platform], max_keywords)
            prompt_template = PROMPT_SCHEMA_V4
        else:
            active_schema = keyword_selection_schema(platforms or [platform], max_keywords)
            prompt_template = PROMPT_SCHEMA_V1
    else:
        active_schema = None
        prompt_template = PROMPT

    if active_schema is not None:
        prompt = prompt_template.format(
            platform=platform,
            vocabulary=", ".join(trimmed),
            query=query,
            max_keywords=max_keywords,
            schema=render_schema(
                active_schema,
                exemplar_stage=exemplar_stage if schema_exemplar else None,
            ),
        )
    else:
        prompt = prompt_template.format(
            platform=platform,
            vocabulary=", ".join(trimmed),
            query=query,
            max_keywords=max_keywords,
        )


    # Bound the reply, then size the budget to that bound. The prompt caps how
    # many keywords may be returned, so the maximum well-formed reply length is
    # known in advance and truncation cannot occur for a compliant answer.
    # A schema entry is an object with three fields, not a `"key": score` pair,
    # so a compliant answer is structurally longer. Sizing the budget from the
    # Arm 2 per-keyword figure would reinstate the 2026-08-17 truncation defect
    # in Arm 3 only -- and truncation is a HARNESS outcome, so the arm would
    # quietly lose cells to a budget error while reporting nothing against the
    # model. It would also break the overrun/truncation split below, which
    # decides model-fault versus ours by comparing emitted tokens against
    # exactly this number.
    # V5 is the exception the comment above anticipates: it carries a schema but
    # its compliant answer is `"keyword": score`, the same shape Arm 2 emits, so
    # the Arm 2 rate is the correct structural minimum. Budgeting it at the
    # schema rate would give V5 room V1 never had and turn a payload-shape
    # comparison into a budget comparison.
    per_keyword_tokens = (
        _TOKENS_PER_KEYWORD
        if active_schema is None or flat_payload
        else _TOKENS_PER_KEYWORD_SCHEMA
    )
    budget = _TOKEN_OVERHEAD + max_keywords * per_keyword_tokens
    if params is None:
        params = GenerationParams(temperature=0.0, max_tokens=budget)
    elif params.max_tokens < budget:
        # Honouring a caller budget below the structural minimum would reinstate
        # the truncation defect, so raise it instead.
        params = params.merged(max_tokens=budget)

    try:
        response = backend.generate(prompt, model=model, params=params)
    except LLMError as exc:
        # Record the failure rather than returning a silent empty selection.
        # An API outage and a model that selected nothing are different events;
        # aggregating them together would attribute infrastructure failures to
        # model capability, which is the comparison this repository exists to
        # measure.
        return KeywordSelection(
            platform=platform,
            keywords={},
            error=f"{type(exc).__name__}: {exc}",
            outcome=OUTCOME_BACKEND_ERROR,
        )

    if response.truncated:
        # A reply cut off mid-object parses to nothing, which previously looked
        # identical to a model declining to select. On 2026-08-17 this silently
        # removed 6,641 coursera courses from a live run that reported status ok.
        #
        # Two different events reach here wearing the same finish_reason, and
        # they fall on opposite sides of the model/harness line. `budget` is the
        # length a compliant answer cannot exceed, derived from the cap this
        # prompt states. Having been given at least that much room and still
        # running out of it is not a budget failure -- it is the model ignoring
        # "Return AT MOST {max_keywords} keywords", and instruction-following is
        # precisely what a scale comparison is measuring.
        # The deciding evidence is how much the model actually emitted, not what
        # we asked for. Asking is not granting: a provider's own output cap or a
        # context limit can stop a reply well below the budget in our request,
        # and that ceiling is ours to answer for. Comparing against
        # `params.max_tokens` instead would make this branch unreachable, since
        # the budget above is always raised to at least `budget` -- an outcome
        # value that cannot occur is worse than none, because it reads as
        # coverage that does not exist.
        emitted = response.completion_tokens
        emitted_note = f"{emitted} tokens, " if emitted is not None else ""
        stopped_early = emitted is not None and emitted < budget
        if stopped_early or (emitted is None and params.max_tokens < budget):
            return KeywordSelection(
                platform=platform,
                keywords={},
                raw_response=response.text,
                error=(
                    f"reply cut off after {emitted_note}{len(response.text)} chars, "
                    f"below the {budget} tokens a compliant {max_keywords}-keyword "
                    f"answer needs (max_tokens={params.max_tokens} requested) -- a "
                    "ceiling lower than the one asked for, so this is ours"
                ),
                outcome=OUTCOME_TRUNCATED,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        return KeywordSelection(
            platform=platform,
            keywords={},
            raw_response=response.text,
            error=(
                f"reply overran the {max_keywords}-keyword cap: emitted "
                f"{emitted_note}{len(response.text)} chars against "
                f"max_tokens={params.max_tokens}, where a compliant answer needs "
                f"at most {budget}"
            ),
            outcome=OUTCOME_OVERRUN,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    claimed_copied: dict[str, bool] = {}
    if active_schema is not None:
        # Arm 3. Validation is binary and there is deliberately no salvage path.
        #
        # Falling back to `try_parse_keyword_response` on a schema-shaped reply
        # would be actively harmful, not merely lenient: `_coerce` maps a dict's
        # top-level keys to keywords, so `{"platform": ..., "keywords": [...]}`
        # would enter retrieval as the two keywords "platform" and "keywords",
        # both of which then score as model hallucinations. The harness would be
        # manufacturing the failure it is measuring.
        #
        # Salvaging just the well-formed entries out of a rejected `keywords`
        # array was the other option and is rejected for a subtler reason: it is
        # a *different* leniency from the one Arm 2's parser applies, so the two
        # arms would differ in how their replies are read as well as in
        # treatment -- which is the confound DEC-6 exists to prevent.
        result = parse_and_validate(response.text, active_schema, STAGE_SELECTION)
        if not result.ok:
            return KeywordSelection(
                platform=platform,
                keywords={},
                raw_response=response.text,
                outcome=OUTCOME_SCHEMA_VIOLATION,
                schema_violations=result.violations,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        parsed = {}
        if flat_payload:
            # V5. `keywords` is a map, so there is no per-entry object to read
            # and `claimed_copied` stays empty -- the field does not exist in
            # this variant, and an empty dict is what the export already means
            # by "the model made no verbatim claims", not a missing measurement.
            for raw_key, raw_score in result.value["keywords"].items():
                key = str(raw_key).strip().lower()
                if not key:
                    continue
                parsed[key] = float(raw_score)
        else:
            for entry in result.value["keywords"]:
                # Lower-cased to match `_coerce`, so a keyword's route into
                # `classify` is identical in both arms. A schema-mode keyword
                # that normalised differently from an Arm 2 one would move M2.1
                # by casing rather than by capability.
                key = str(entry["keyword"]).strip().lower()
                if not key:
                    continue
                parsed[key] = float(entry["score"])
                claimed_copied[key] = bool(entry["copied_verbatim"])
        parsed_ok = True
    else:
        parsed, parsed_ok = try_parse_keyword_response(response.text)
    if not parsed_ok:
        # Finished cleanly but emitted something unparseable. Unlike truncation
        # this is the model's failure and stays in the capability metrics.
        return KeywordSelection(
            platform=platform,
            keywords={},
            raw_response=response.text,
            outcome=OUTCOME_UNPARSEABLE,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    keep, hallucinated, out_of_window, removed = classify(parsed, trimmed, vocab)
    # Positions in the window as shown. Built here rather than downstream because
    # `trimmed` is the only place the order the model actually saw still exists.
    ranks = {keyword: i for i, keyword in enumerate(trimmed)}
    # M2.2 before M2.1 is read. `classify` tests exact membership, so a keyword the
    # model spelled differently is sitting in `hallucinated` at this point; moving
    # it out is what makes the two metrics mean what they say.
    near_miss, hallucinated = split_near_misses(hallucinated, vocab)
    return KeywordSelection(
        platform=platform,
        keywords=keep,
        hallucinated=hallucinated,
        near_miss=near_miss,
        selected_ranks=sorted(ranks[k] for k in keep if k in ranks),
        out_of_window=out_of_window,
        removed_by_cleaning=removed,
        raw_response=response.text,
        outcome=OUTCOME_OK if parsed else OUTCOME_EMPTY,
        # Every keyword the model emitted, not only the ones that validated.
        # Restricting this to `keep` would answer "does the model know when it
        # copied a *real* keyword", which is the easy half; the question the
        # field exists for is what the model claims about the ones it invented.
        copied_verbatim=claimed_copied,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
    )
