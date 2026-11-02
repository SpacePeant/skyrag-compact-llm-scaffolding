"""Prompt templates.

Kept in one module so prompts can be revised without touching pipeline code
-- prompt wording is the main experimental variable in this project.

The all-caps coercion from the notebooks (threats, sales-person roleplay,
"YOUR LIFE WILL BE TERMINATED") is deliberately gone. It existed to force two
behaviours that are now handled structurally: platform diversity comes from
per-platform keyword selection plus round-robin merging, and grounding comes
from validating every cited Course ID against the retrieved set.
"""

from __future__ import annotations

# Step 1 of step-back prompting: elicit principles before applying them.
STEP_BACK = """You are an experienced curriculum designer.

Before proposing any specific courses, identify the core educational
principles that should guide a learning path for this learner. Consider prior
knowledge, cognitive load, pacing, motivation, and how skills should build on
one another.

Answer as 4-6 concise bullet points. Do not recommend courses yet.

Learner's request:
{query}
"""

# Query expansion: short queries embed poorly against long course blobs.
EXPAND_QUERY = """Rewrite the learner's request below as a single detailed
paragraph of 3-5 sentences suitable for searching a course catalog.

Make the subject matter, prerequisite level, and target duration explicit, and
name the specific skills and topics involved. Return only the paragraph.

Learner's request:
{query}
"""

# --- B1 / Arm 3. The same three prompts under skill schemas. ---------------
#
# Each V1 prompt mirrors the Arm 2 prompt above it instruction for instruction,
# with the output section replaced by a schema. The wording is held as close as
# the shape allows on purpose: Arm 3 is Arm 2 plus a schema, so any rewording
# that the schema does not force is a second variable inside a single-factor arm.
#
# That holds for V1 only. The V4 prompts further down do not mirror Arm 2:
# EXPAND_QUERY_SCHEMA_V4 asks for three sentences on prerequisite level and
# duration and drops "name the specific skills and topics involved". The Stage 1
# decomposition (2026-10-01) and Run A (2026-10-02) found that this wording
# difference, not the schema, carries most of the flat arm's invention
# reduction: the V1 schema rewrite here leaves Arm 2's invention unchanged on
# LLaMA 3.1-8B. See EXPAND_QUERY_SCHEMA_WORDING.
#
# The schema itself is interpolated by the caller from `schemas.py`, which is
# where the versioned treatment text lives.

EXPAND_QUERY_SCHEMA_V1 = """Rewrite the learner's request below for searching a
course catalog.

Make the subject matter, prerequisite level, and target duration explicit, and
name the specific skills and topics involved.

Return ONLY a JSON object matching this schema. No prose, no code fence.

Schema:
{schema}

Learner's request:
{query}
"""

STEP_BACK_SCHEMA_V1 = """You are an experienced curriculum designer.

Before proposing any specific courses, identify the core educational
principles that should guide a learning path for this learner. Consider prior
knowledge, cognitive load, pacing, motivation, and how skills should build on
one another. Do not recommend courses yet.

Return ONLY a JSON object matching this schema. No prose, no code fence.

Schema:
{schema}

Learner's request:
{query}
"""

GENERATE_PATH_SCHEMA_V1 = """You are an experienced curriculum designer building a
personalized learning path.

## Available courses

These are the ONLY courses you may recommend. Every course you name must
appear below, cited by its exact Course ID.

{context}

## Educational principles

{principles}

## Learner's request

{query}

## Instructions

1. Recommend courses drawn only from the list above, in a sensible order,
   progressing from foundational to advanced.
2. Cover the full duration the learner asked for, with at least one course per
   period.
3. Cite the exact Course ID for every course. Copy it from the list above;
   never construct one.
4. `step` is the position in the sequence, starting at 1. Earlier steps must
   be prerequisites of later ones.
5. Briefly explain in `rationale` why each course follows from the previous
   one.
6. If the available courses do not fully cover the request, return the courses
   that do fit rather than inventing one.

## Output format

Return ONLY a JSON object matching this schema. No prose, no code fence.

Schema:
{schema}
"""

GENERATE_PATH = """You are an experienced curriculum designer building a
personalized learning path.

## Available courses

These are the ONLY courses you may recommend. Every course you name must
appear below, cited by its exact Course ID.

{context}

## Educational principles

{principles}

## Learner's request

{query}

## Instructions

1. Recommend courses drawn only from the list above, in a sensible order,
   progressing from foundational to advanced.
2. Cover the full duration the learner asked for, with at least one course per
   period.
3. Cite the exact Course ID for every course.
4. Briefly explain why each course follows from the previous one.
5. If the available courses do not fully cover the request, say so plainly
   rather than inventing a course.

## Output format

First a Markdown table with columns:
`Period | Course ID | Course Title | Platform | Why this course`

Then a short paragraph on how the path fits together.
"""


def build_generation_prompt(
    query: str,
    context: str,
    principles: str = "",
    *,
    schema: str = "",
) -> str:
    """The generation prompt, schema-shaped when `schema` is non-empty.

    `schema` rather than a boolean so the rendered treatment text arrives from
    `schemas.py` and this module never holds a second copy of it. Two copies of
    a versioned string is one copy too many: the version stamped on the row
    would stop being evidence of what the model was actually shown.
    """
    if schema:
        return GENERATE_PATH_SCHEMA_V1.format(
            query=query.strip(),
            context=context.strip(),
            principles=(
                principles.strip() or "Apply standard curriculum-design practice."
            ),
            schema=schema,
        )
    return GENERATE_PATH.format(
        query=query.strip(),
        context=context.strip(),
        principles=(principles.strip() or "Apply standard curriculum-design practice."),
    )


# --- V4 prompt scaffolding (Arm 3; also used by v5) ---

EXPAND_QUERY_SCHEMA_V4 = """Rewrite the query for catalog search: 2-4 search phrases and a 3-sentence expanded query specifying prerequisite level and duration.
Return ONLY JSON matching the schema. No markdown code blocks, no commentary.

Schema:
{schema}

Query: {query}
"""

# Stage 1 decomposition control (2026-10-01). EXPAND_QUERY_SCHEMA_V4's
# instruction text with the JSON removed, answered as plain text.
#
# The Arm 3 rewrite differs from Arm 2's EXPAND_QUERY in two ways at once: it
# carries a schema, and its instructions are different -- three sentences on
# prerequisite level and duration, where EXPAND_QUERY asks for 3-5 sentences
# that "name the specific skills and topics involved". The 5090 2x2 found the
# schema arm's invention reduction in the Stage 1 rewrite, and this prompt is
# what separates those two causes: if it reduces invention as much as the
# schema rewrite does, the cause is the wording, not the schema.
#
# The 2-4 search phrases are dropped because nothing downstream reads them; only
# `expanded_query` reaches selection. The remaining difference from the schema
# rewrite is that the model there writes the phrases before the expanded query.
EXPAND_QUERY_SCHEMA_WORDING = """Rewrite the query for catalog search: a 3-sentence expanded query specifying prerequisite level and duration.
Return ONLY the expanded query as plain text. No markdown, no commentary.

Query: {query}
"""

STEP_BACK_SCHEMA_V4 ="""Identify the broad concept and 3-5 core curriculum design principles for this learner. Do not name courses.
Return ONLY JSON matching the schema. No markdown code blocks, no commentary.

Schema:
{schema}

Query: {query}
"""

GENERATE_PATH_SCHEMA_V4 = """Design a sequenced learning path from foundational to advanced.
Cite ONLY Course IDs from Available Courses.

Available Courses:
{context}

Principles:
{principles}

Query: {query}

Return ONLY JSON matching the schema. No markdown code blocks, no commentary.

Schema:
{schema}
"""
