"""Skill schemas -- the Arm 3 treatment (B1).

Arm 3 is Arm 2 **plus structured skill schemas**: same models, same call
count, same retrieval, same corpus. The only difference is the *form* the
model is asked to answer in at each stage it speaks. See
`project_knowledge/research_directions/arms/arm3_skill_schemas.md`.

Three properties of this module are load-bearing and none of them are
stylistic:

1. **The schema text is part of the treatment.** It is versioned as `v1`,
   stamped onto every row via `schema_version`, and must never be edited
   mid-sweep. A schema edited without a version bump is indistinguishable, in
   the artifact, from an unedited one.
2. **A violation is a recorded outcome, not a crash and not a retry.** The
   caller records it and carries on with whatever the model actually said. A
   silent retry converts a failure rate into a latency figure.
3. **Nothing here filters or repairs model output.** `copied_verbatim` in
   particular is an instrument, never a control -- see S2 below.

Enforcement is **validated prompt-and-parse**, per DEC-6 (resolved
2026-08-21): the schema is rendered into the prompt and the reply is parsed
and validated here. It is not sent as a wire `response_format`, because
`json_schema` returns HTTP 404 for llama-3.1-8b on DeepInfra, and because
Arm 2's replies are already parsed this way, so a different mechanism here
would make the arms differ by wire protocol rather than by treatment.

This module imports nothing from the rest of `skyrag` at module scope; the one
function that needs the shared JSON extractor imports it locally, so that
`selection` can import these schemas without a cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Version of the schema *text*. Bump on any edit to the three schemas or to the
# prompt fragments built from them, and never within a sweep.
SCHEMA_VERSION = "v1"

# Stage labels, used for `schema_violation_stages` so a violation says which
# call produced it. Kept as constants because they reach an exported column.
STAGE_EXPANSION = "query_expansion"
STAGE_STEP_BACK = "step_back"
STAGE_SELECTION = "keyword_selection"
STAGE_PATH = "learning_path"
# V5 selects the same stage under a different payload shape, so it needs its own
# exemplar: rendering the array-of-objects example beside a flat-map schema would
# teach the model to violate the schema it was just given -- the harness inducing
# the failure it measures.
STAGE_SELECTION_V5 = "keyword_selection_v5"


# --- S1 -- the expansion stage, split across the two calls it already makes --
#
# The specification writes S1 as a single object with `expansions` and
# `step_back`. The pipeline makes **two** expansion-stage calls (EXPAND_QUERY
# then STEP_BACK), and collapsing them into one would change the call count and
# therefore the X1 per-stage token attribution -- a difference between Arm 2 and
# Arm 3 that is not the treatment. So S1 is split along the boundary that
# already exists, and both fields the specification named are present and
# validated.
#
# `expanded_query` and `principles` are additions to the specification's shape,
# and they are there for parity, not decoration. `prompts.py` explains why the
# expansion is a paragraph: "short queries embed poorly against long course
# blobs." Replacing that paragraph with an array of short phrases would change
# the retrieval input itself, so Arm 3 would differ from Arm 2 in what gets
# embedded as well as in output format. Keeping both means the schema
# constrains the form while retrieval sees exactly what it saw in Arm 2.
QUERY_EXPANSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["expansions", "expanded_query"],
    "properties": {
        "expansions": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {"type": "string"},
            "description": (
                "Reformulations of the learner's query, each a standalone "
                "search phrase."
            ),
        },
        "expanded_query": {
            "type": "string",
            "description": (
                "One detailed paragraph of 3-5 sentences suitable for searching "
                "a course catalog. Make subject matter, prerequisite level and "
                "target duration explicit."
            ),
        },
    },
}

STEP_BACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["step_back", "principles"],
    "properties": {
        "step_back": {
            "type": "string",
            "description": (
                "The broader concept this query sits inside. One phrase, not a "
                "sentence."
            ),
        },
        "principles": {
            "type": "array",
            "minItems": 4,
            "maxItems": 6,
            "items": {"type": "string"},
            "description": (
                "Educational principles that should guide the learning path. "
                "One concise point each. Do not name courses."
            ),
        },
    },
}


# --- S2 -- keyword selection. The load-bearing one. ------------------------
#
# `maxItems` mirrors `max_keywords`, so overrunning the cap becomes a schema
# violation the harness *reads* rather than a truncation it has to infer from
# a finish_reason. Stage 1 recorded `selection_overrun = 1` on the 8B, so this
# fires on real data.
#
# `score` is an addition to the specification's shape, for the same
# arm-parity reason as `expanded_query` above. Arm 2's selection prompt asks
# for `{keyword: score}` and `KeywordSelection.ranked()` orders retrieval by
# that score; `keyword_buckets` consumes the order, and first-keyword-wins
# decides every retrieved course's `via_keyword`. A schema without a score
# would push ordering onto the alphabetical tie-break, changing which courses
# are retrieved for reasons that have nothing to do with the treatment.
#
# `copied_verbatim` is an INSTRUMENT, not a control. Nothing acts on it. Its
# value is exported so that M2.1 can be split by the model's own claim: if
# invented keywords mostly arrive marked `true`, the model cannot tell recall
# from reading, which is a finding about the failure mode that Arm 3 produces
# for free.
#
# It must NEVER filter keywords. Dropping the ones the model flagged as
# uncopied would delete the failure mode under study and move every downstream
# number toward the hypothesis -- the same reason near-misses are measured and
# never repaired.
def keyword_selection_schema(
    platforms: list[str], max_keywords: int
) -> dict[str, Any]:
    """S2, bound to the platforms actually loaded and the cap actually in force.

    Built rather than declared because both bounds are run configuration. A
    hard-coded `["coursera", "edx"]` enum would reject a valid answer on any
    corpus loaded with a third platform, and a hard-coded cap would drift from
    `config.retrieval.max_keywords` the first time it was tuned -- turning a
    configuration change into a spurious violation rate.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["platform", "keywords"],
        "properties": {
            "platform": {"type": "string", "enum": list(platforms)},
            "keywords": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_keywords,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["keyword", "score", "copied_verbatim"],
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": (
                                "EXACTLY as it appears in the vocabulary list "
                                "above. Copy the characters; do not correct, "
                                "pluralise, expand or tidy them."
                            ),
                        },
                        "score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "How relevant this keyword is, 0.0 to 1.0.",
                        },
                        "copied_verbatim": {
                            "type": "boolean",
                            "description": (
                                "True only if you copied this string from the "
                                "list rather than recalling it. If you are "
                                "writing it from memory, set false."
                            ),
                        },
                    },
                },
            },
        },
    }


# --- S3 -- the learning path ----------------------------------------------
#
# `step` exists so O1's ordering criterion has an explicit field to score
# rather than a judge inferring sequence from the row order of a Markdown
# table. Since O3 folded into O1's rubric, this is the only structured signal
# ordering has.
LEARNING_PATH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["path"],
    "properties": {
        "path": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step", "course_id", "rationale"],
                "properties": {
                    "step": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Position in the sequence. Earlier steps must be "
                            "prerequisites of later ones."
                        ),
                    },
                    "course_id": {
                        "type": "string",
                        "description": (
                            "A course ID copied from the retrieved list. Never "
                            "construct one."
                        ),
                    },
                    "rationale": {"type": "string", "maxLength": 300},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class SchemaResult:
    """The outcome of asking one stage for a schema-shaped answer.

    `value` is None exactly when the reply could not be validated. `violations`
    is non-empty in that case and names what was wrong, in the reply's own
    terms, so the exported count can be traced back to a cause rather than
    merely counted.

    `raw` always holds what the model actually said, including on the failure
    path: a violation the harness cannot show is a number nobody can check.
    """

    stage: str
    value: dict[str, Any] | None
    violations: list[str]
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


# --- Arm 3 prompt variants (2026-08-31) -----------------------------------
#
# The B1 prompts show the model a JSON Schema and ask for "a JSON object
# matching this schema". Measured live on 2026-08-31, the 8B answers by
# **echoing the schema back** -- returning `{"type": "object", "required": [...],
# "properties": {...}}` with its actual answer placed in the `description`
# fields. Validation rejects it, so the arm emits zero keywords.
#
# That is a real behaviour, not a harness defect: prompt parity with Arm 2 is
# exact (`prompts.py` mirrors them instruction for instruction) and neither arm
# carries a worked example. But it makes Arm 3 measure *"can an 8B read a
# schema specification"* rather than *"can an 8B produce structured output"*,
# and those are different claims with different practical consequences.
#
# So the exemplar is a **second level of the schema treatment**, not a fix, and
# both levels are run. `spec_only` is the original; `spec_plus_exemplar` adds
# one valid instance. The variant is stamped per exported row, because it
# changes what the arm measures.
VARIANT_SPEC_ONLY = "spec_only"
VARIANT_SPEC_PLUS_EXEMPLAR = "spec_plus_exemplar"
# Condition C (2026-09-07). Not a third prompt *text* the way the exemplar is:
# the schema and its instructions are byte-identical to `spec_only`, and what
# changes is the vocabulary window the prompt is rendered against -- a static
# slice of the platform vocabulary becomes a query-matched candidate set
# (`selection.rank_candidate_keywords`). It is stamped as a variant anyway
# because it is a level of the treatment that changes what the arm measures,
# and the first Condition C sweep (2026-09-07) exported rows labelled
# `spec_only`, indistinguishable from Condition A in the artifact -- any
# aggregation over this column silently pooled the two.
#
# `candidate_injection` on the row is the machine truth and is stamped on
# **every** arm, including Arm 2, which runs injection with no schema at all.
# This label is the Arm 3 taxonomy on top of it. Injection combined with the
# exemplar is not a named condition: those rows keep `spec_plus_exemplar` and
# carry the boolean, rather than inventing a fourth condition letter here.
VARIANT_SPEC_PLUS_CANDIDATE_INJECTION = "spec_plus_candidate_injection"

# Hand-written rather than generated from the schema. A generated instance
# would reproduce the schema's own vocabulary and risk teaching the model to
# copy the example instead of the shape; these use unrelated subject matter
# (music theory, marine biology) for that reason -- none of the ten benchmark
# domains, so no exemplar leaks a plausible answer for any scored query.
EXEMPLARS: dict[str, Any] = {
    STAGE_EXPANSION: {
        "expansions": [
            "beginner music theory for songwriting",
            "reading sheet music and rhythm fundamentals",
        ],
        "expanded_query": (
            "An introductory course sequence in Western music theory for a "
            "learner with no prior formal training. Covers notation, key "
            "signatures, intervals and basic harmony over roughly three "
            "months, ending with simple four-part writing."
        ),
    },
    STAGE_STEP_BACK: {
        "step_back": "foundational music literacy",
        "principles": [
            "Establish notation fluency before introducing harmony.",
            "Introduce aural training alongside written theory from the start.",
            "Keep cognitive load low by separating rhythm from pitch initially.",
            "Reinforce each concept with short composition exercises.",
        ],
    },
    STAGE_SELECTION: {
        "platform": "coursera",
        "keywords": [
            {"keyword": "marine biology", "score": 0.9, "copied_verbatim": True},
            {"keyword": "oceanography", "score": 0.7, "copied_verbatim": True},
        ],
    },
    # Same subject matter as STAGE_SELECTION, same distance from the ten
    # benchmark domains, differing only in payload shape -- so V5's exemplar
    # cannot leak anything STAGE_SELECTION's does not.
    STAGE_SELECTION_V5: {
        "platform": "coursera",
        "keywords": {"marine biology": 0.9, "oceanography": 0.7},
    },
    STAGE_PATH: {
        "path": [
            {
                "step": 1,
                "course_id": "COU0000",
                "rationale": "Establishes the vocabulary the later steps assume.",
            },
            {
                "step": 2,
                "course_id": "EDX0000",
                "rationale": "Applies that vocabulary to a practical project.",
            },
        ],
    },
}


def render(schema: dict[str, Any], *, exemplar_stage: str | None = None) -> str:
    """The schema as it appears inside a prompt.

    `sort_keys=False` deliberately: the key order in the declarations above is
    the order the model reads, and reordering it between runs would change the
    treatment without changing the version string.

    `exemplar_stage` selects the `spec_plus_exemplar` variant and appends one
    valid instance. The example is **illustrative of shape only** -- its subject
    matter is drawn from outside the ten benchmark domains so it cannot supply
    a usable answer to any scored query, which would convert the variant from a
    formatting aid into a content leak.
    """
    text = json.dumps(schema, indent=2, sort_keys=False)
    if exemplar_stage is None:
        return text
    example = EXEMPLARS.get(exemplar_stage)
    if example is None:
        return text
    return (
        f"{text}\n\n"
        "Example of a valid response. Copy this SHAPE, not its content -- "
        "your answer must address the learner's actual request:\n"
        f"{json.dumps(example, indent=2, sort_keys=False)}"
    )


def _type_name(value: Any) -> str:
    # bool before int: `isinstance(True, int)` is True in Python, and a schema
    # that accepted `true` where it asked for an integer would silently admit
    # a malformed `step`.
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _type_ok(value: Any, expected: str) -> bool:
    actual = _type_name(value)
    if expected == "number":
        # An integer is a valid number; a boolean is not, per _type_name.
        return actual in ("number", "integer")
    return actual == expected


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate `value` against the subset of JSON Schema these schemas use.

    Hand-written rather than delegated to `jsonschema` for two reasons. The
    schemas here are three fixed shapes using nine keywords between them, so
    the dependency would buy very little; and the treatment must be
    byte-stable across a sweep, which a library free to change its error
    strings between patch releases is not -- those strings reach an exported
    column.

    Returns a list of human-readable violations, empty when the value
    conforms. Every message names its location, because "expected string" with
    no path is not something anyone can act on.
    """
    errors: list[str] = []

    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        # Stop at the first type mismatch on this node. Recursing into a value
        # of the wrong type produces a cascade of derived complaints that bury
        # the one real cause.
        return [f"{path}: expected {expected}, got {_type_name(value)}"]

    if expected == "object":
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties")
        if extra is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(extra, dict):
            # A map whose *keys* are data, used by S2 under V5 where a keyword
            # is a key rather than a field value. Every entry not named in
            # `properties` is validated against this one sub-schema, which is
            # what lets the flat map constrain its scores the way the
            # array-of-objects form constrains `score`.
            for key, item in value.items():
                if key not in props:
                    errors.extend(validate(item, extra, f"{path}.{key}"))
        # Object-arity, the flat-map counterpart of minItems/maxItems. Without
        # these the V5 keyword cap would be unenforceable, and V5 would differ
        # from V1 in whether the cap is part of the contract rather than only
        # in payload shape.
        min_props = schema.get("minProperties")
        max_props = schema.get("maxProperties")
        if min_props is not None and len(value) < min_props:
            errors.append(f"{path}: has {len(value)} properties, minimum is {min_props}")
        if max_props is not None and len(value) > max_props:
            errors.append(f"{path}: has {len(value)} properties, maximum is {max_props}")
        for key, sub in props.items():
            if key in value:
                errors.extend(validate(value[key], sub, f"{path}.{key}"))

    elif expected == "array":
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path}: has {len(value)} items, minimum is {min_items}")
        if max_items is not None and len(value) > max_items:
            errors.append(f"{path}: has {len(value)} items, maximum is {max_items}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))

    elif expected == "string":
        max_length = schema.get("maxLength")
        if max_length is not None and len(value) > max_length:
            errors.append(f"{path}: {len(value)} characters, maximum is {max_length}")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")

    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if numeric and "minimum" in schema and value < schema["minimum"]:
        errors.append(f"{path}: {value} is below minimum {schema['minimum']}")
    if numeric and "maximum" in schema and value > schema["maximum"]:
        errors.append(f"{path}: {value} is above maximum {schema['maximum']}")

    return errors


def parse_and_validate(text: str, schema: dict[str, Any], stage: str) -> SchemaResult:
    """Extract the first JSON object in `text` and validate it against `schema`.

    Extraction uses `selection.balanced_json_spans` -- the same balanced-brace
    scanner Arm 2's keyword parser relies on -- because DEC-6's whole point is
    that the arms share one enforcement mechanism. Imported inside the
    function so `selection` can import this module at its top without a cycle.

    Never raises for bad model output. Everything a model can do wrong here is
    a measurement, and a measurement that terminates the sweep is not one.
    """
    from .selection import balanced_json_spans

    text = (text or "").strip()
    if not text:
        return SchemaResult(stage, None, [f"{stage}: empty reply"], text)

    # The whole reply first: a model that answered with nothing but the object
    # is the compliant case, and it should not depend on the scanner agreeing
    # with `json.loads` about where the object starts.
    for span in [text, *balanced_json_spans(text)]:
        try:
            obj = json.loads(span)
        except (ValueError, TypeError):
            continue
        errors = validate(obj, schema)
        if not errors:
            return SchemaResult(stage, obj, [], text)
        # Stop at the first span that is valid JSON but the wrong shape. Its
        # violations describe what the model actually got wrong, which is the
        # reportable signal; continuing would report on some fragment nested
        # inside the reply instead.
        return SchemaResult(stage, None, [f"{stage}: {e}" for e in errors], text)

    return SchemaResult(stage, None, [f"{stage}: no JSON object found in reply"], text)


# --- V4 Ultra-Compact Skill Schemas ($S1, S2, S3$) ---

QUERY_EXPANSION_SCHEMA_V4: dict[str, Any] = {
    "type": "object",
    "required": ["expansions", "expanded_query"],
    "properties": {
        "expansions": {
            "type": "array",
            "minItems": 2,
            "maxItems": 4,
            "items": {"type": "string"},
        },
        "expanded_query": {
            "type": "string",
        },
    },
    "additionalProperties": False,
}

STEP_BACK_SCHEMA_V4: dict[str, Any] = {
    "type": "object",
    "required": ["step_back", "principles"],
    "properties": {
        "step_back": {
            "type": "string",
        },
        "principles": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
}


def keyword_selection_schema_v5(
    platforms: list[str], max_keywords: int
) -> dict[str, Any]:
    """S2 under V5: the same contract, with Arm 2's payload shape.

    V1 asks for an array of three-field objects, so one keyword costs roughly
    three times what the same keyword costs Arm 2 -- which is why the harness
    budgets `_TOKENS_PER_KEYWORD_SCHEMA` (64) against Arm 2's 32. Measured
    consequence: Arm 3 emits far fewer keywords than Arm 2 and retrieval falls
    with it, in both model families tested.

    That cost is **not** intrinsic to having a schema. It is a property of the
    payload V1 asks for. V5 keeps every other element of the contract -- typed
    object, required properties, platform enum, closed top level, an enforced
    keyword cap, a declared score range -- and makes the per-keyword payload
    byte-comparable to Arm 2's `"keyword": score`.

    So V5 vs V1 isolates one variable: **does the schema damage retrieval
    because it is a schema, or because of what it asks the model to write?**

    `copied_verbatim` has no home in a flat map and is dropped. It is the one
    unavoidable secondary change, and it costs nothing measurable: the field was
    flagged `true` on at least 70.9% of invented keywords in condition A, so it
    never functioned as a signal, and `validate()` in `selection` already
    answers the same question deterministically against the real vocabulary.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["platform", "keywords"],
        "properties": {
            "platform": {"type": "string", "enum": list(platforms)},
            "keywords": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": max_keywords,
                # The fidelity instruction V1 hangs on its `keyword` property
                # has no property to hang on here, so it moves to the map
                # itself. Dropping it would confound payload shape with the
                # presence of the copy-exactly instruction -- which is the
                # confound that made the V4 comparison uninterpretable.
                "description": (
                    "Each key is a keyword copied EXACTLY as it appears in the "
                    "vocabulary list above. Do not correct, pluralise, expand "
                    "or tidy the characters. Each value is that keyword's "
                    "relevance, 0.0 to 1.0."
                ),
                "additionalProperties": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
        },
    }


def keyword_selection_schema_v4(
    platforms: list[str], max_keywords: int = 20
) -> dict[str, Any]:
    """Compact S2 keyword selection schema for V4."""
    return {
        "type": "object",
        "required": ["platform", "keywords"],
        "properties": {
            "platform": {
                "type": "string",
                "enum": list(platforms),
            },
            "keywords": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_keywords,
                "items": {
                    "type": "object",
                    "required": ["keyword", "score", "copied_verbatim"],
                    "properties": {
                        "keyword": {"type": "string"},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "copied_verbatim": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


KEYWORD_SELECTION_SCHEMA_V4: dict[str, Any] = keyword_selection_schema_v4(["coursera", "edx"], 20)

LEARNING_PATH_SCHEMA_V4: dict[str, Any] = {
    "type": "object",
    "required": ["path"],
    "properties": {
        "path": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["step", "course_id", "rationale"],
                "properties": {
                    "step": {"type": "integer", "minimum": 1},
                    "course_id": {"type": "string"},
                    "rationale": {"type": "string", "maxLength": 250},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

