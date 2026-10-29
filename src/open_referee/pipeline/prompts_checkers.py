"""Prompts for the hybrid recomputation checker (extract -> sympy -> verify)."""

RECOMPUTE_EXTRACT_SYSTEM = """\
You extract derived numerical quantities from an academic manuscript so a \
computer algebra system can recompute them.

For each derived value the paper REPORTS (a number computed from other \
numbers in the paper via a stated formula — not raw measurements, not \
counts of observations, not parameters taken from outside sources), emit:

{"quantities": [{
  "symbol": "short name, e.g. 'optimal k' or 'total cost'",
  "formula": "sympy-parsable expression using the input names, e.g. 'sqrt(v/q)'",
  "inputs": {"v": 1.3, "q": 0.44},
  "reported_value": 12.5,
  "quote": "the manuscript sentence/table cell reporting the value (verbatim, <=200 chars)",
  "section": "section title or number where it appears"
}]}

Rules:
- formulas use ONLY arithmetic (+ - * / **), sqrt, log, ln, exp, abs, Min, \
Max, binomial, factorial, sin, cos, tan, pi, E — anything else must be \
approximated or skipped.
- inputs must be numbers that appear in the paper (or simple derivations of \
them); never invent values.
- 3-10 quantities maximum, preferring central results over incidental ones.
- if nothing qualifies, return {"quantities": []}.
JSON only."""

RECOMPUTE_EXTRACT_USER = """\
Manuscript title: {title}

Excerpt (equations, numeric passages, tables):
{excerpt}

Extract the recomputable derived quantities now. JSON only."""

RECOMPUTE_VERIFY_SYSTEM = """\
You are verifying a numerical inconsistency flagged by a computer algebra \
system. The paper's own formula and inputs were evaluated symbolically; the \
result disagrees with the paper's reported value.

Given the formula, inputs, computed value, reported value and the quoting \
passage, decide whether this is a REAL error in the manuscript — as opposed \
to a mis-extraction (wrong formula, wrong inputs, different units, rounding, \
or the quote not actually being the output of that formula).

Respond with JSON:
{{"confirmed": true|false,
  "title": "short finding title",
  "message": "2-4 sentences: what the paper reports, what the formula gives, \
and what the authors should check/fix",
  "severity": 0.3-0.9}}
Confirm ONLY when the extraction is faithful and the mismatch exceeds any \
plausible rounding. JSON only."""

RECOMPUTE_VERIFY_USER = """\
Quantity: {symbol}
Formula (sympy): {formula}
Inputs: {inputs}
Computed value: {computed}
Reported value: {reported}
Manuscript quote: {quote}
Section: {section}

Is this a real numerical error in the manuscript? JSON only."""
