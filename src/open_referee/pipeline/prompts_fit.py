"""Prompts for the journal-fit / desk-reject estimator product.

Two-step protocol (agreed with the project owner): the LLM editor writes the
six factor analyses FIRST; the decision model then takes the venue profile
plus those factor texts as input and directly scores the desk-screen
predicate. The LLM never produces the final number.
"""

from __future__ import annotations

FACTOR_NAMES = (
    "scope_fit",
    "aim_alignment",
    "methods_format_fit",
    "novelty_threshold",
    "presentation_readiness",
    "policy_ethics",
)

FACTOR_LABELS = {
    "scope_fit": "Scope fit",
    "aim_alignment": "Aim alignment",
    "methods_format_fit": "Methods & format fit",
    "novelty_threshold": "Novelty vs venue threshold",
    "presentation_readiness": "Presentation readiness",
    "policy_ethics": "Policy & ethics compliance",
}
FACTOR_ORDER = list(FACTOR_NAMES)

# hard cap per factor analysis — the user wants brevity and bounded tokens
FACTOR_TOKEN_CAP = 600

EDITOR_SYSTEM = f"""\
You are a desk editor at an academic journal. A user has supplied a free-form \
description of the venue — aim, scope, profile, metrics, and possibly \
institutional or opportunity considerations from a working-paper collection. \
Assess how the attached manuscript fares at the desk-screening stage.

Write six factor analyses. Each is a tight editorial note of AT MOST \
{FACTOR_TOKEN_CAP} tokens: what the evidence shows for this factor, grounded \
in the manuscript and the profile (quote briefly where decisive). Weigh what \
the profile actually emphasizes. Be honest, not encouraging.

You DO NOT produce the final probability — a decision model will judge your \
analyses afterwards.

Reply with ONLY a JSON object:
{{"factors": [
  {{"name": "scope_fit", "analysis": "..."}},
  {{"name": "aim_alignment", "analysis": "..."}},
  {{"name": "methods_format_fit", "analysis": "..."}},
  {{"name": "novelty_threshold", "analysis": "..."}},
  {{"name": "presentation_readiness", "analysis": "..."}},
  {{"name": "policy_ethics", "analysis": "..."}}],
 "reasoning": "markdown: overall editorial reading (3-6 sentences)",
 "risks": ["short bullet", ...],
 "suggestions": ["concrete, actionable improvements", ...]}}
All six factors are required. If the profile says nothing about a dimension, \
say so in that factor's analysis."""

EDITOR_USER_TEMPLATE = """\
## Journal / venue profile (free-form, from the user)

{journal_profile}

## Manuscript: {title}

{manuscript}
"""

# fallback (no decision model configured): the LLM also estimates probability
EDITOR_FALLBACK_ADDENDUM = """

Additionally include "llm_estimate": 0.0-1.0 — your own holistic probability
that the paper passes this venue's desk screen (used ONLY because no decision
model is configured)."""

# the decision model's predicate over the editor's output
DECISION_PREDICATE = (
    "Based on the editor's six factor analyses for this venue profile, this "
    "paper would pass the journal's desk screen and be sent to review."
)


def editor_user(
    journal_profile: str, title: str, manuscript: str, *, fallback: bool = False
) -> str:
    user = EDITOR_USER_TEMPLATE.format(
        journal_profile=journal_profile[:20_000], title=title, manuscript=manuscript
    )
    if fallback:
        user += EDITOR_FALLBACK_ADDENDUM
    return user
