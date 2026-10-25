"""Prompt contracts established by the seeded-error A/B eval (local_test/ab_*).

Each test pins a binding behavior that measurably changed the funnel:
- inventory coverage duties route overclaim/significance/artifact-description
  sentences into the per-claim engine (variant B1);
- the figure_text worker must articulate caption-prose contradictions as
  consistency findings, not unverifiability complaints (variant B2);
- the meta merge must never drop distinct findings (variant B3: on the RMI
  paper the shipped prompt deleted ~130 distinct findings, 51 of 60 major).
"""

from __future__ import annotations

from open_referee.pipeline import prompts, prompts_claims


def test_inventory_prompt_demands_coverage_families():
    sys = prompts_claims.CLAIM_INVENTORY_SYSTEM
    assert "Coverage duties" in sys
    for phrase in ("Overclaims and hedge removals", "Significance statements",
                   "Figure/table descriptions", "Table-prose comparisons"):
        assert phrase in sys, f"coverage family missing: {phrase}"


def test_figure_text_contract_demands_concrete_contradictions():
    instr = prompts_claims.TYPE_SPECIFIC["figure_text"]
    assert "PAIR of descriptions" in instr
    # the two anti-patterns the eval showed kill true findings downstream
    assert "Do NOT soften" in instr
    assert '"consistency"' in instr


def test_meta_prompt_has_preservation_contract():
    sys = prompts.META_SYSTEM
    assert "Preservation contract" in sys
    assert "NEVER drop a finding that is distinct" in sys
    # severity recalibration must not change list membership
    assert "changes \\\nscores, never membership" in sys or \
        "changes scores, never membership" in sys.replace("\\\n", "")


def test_figure_text_instruction_instantiates_cleanly():
    instantiated = prompts_claims.claim_verify_system("figure_text")
    assert "FIGURE_TEXT VERIFICATION WORKER" in instantiated
    assert "PAIR of descriptions" in instantiated


def test_validator_prompt_has_evidence_based_drop_contract():
    sys = prompts.VALIDATOR_SYSTEM
    assert "DROP rules" in sys and "ONLY when you can cite" in sys
    for phrase in ("does not appear in the manuscript", "contradicts the criticism",
                   "exact duplicate"):
        assert phrase in sys, f"drop ground missing: {phrase}"
    # severity taste must never remove; drops must be accountable
    assert "recalibrated, never removed" in sys
    assert "List every dropped title" in sys
