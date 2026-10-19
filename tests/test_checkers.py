"""Deterministic checkers, sandbox, recomputation, and reference resolution."""

from __future__ import annotations

import json

import pytest

from open_referee.ingestion.readers import document_from_markdown
from open_referee.literature.reference_resolution import (
    _score,
    _title_guess,
    split_references,
)
from open_referee.pipeline.checkers import (
    cross_reference_check,
    number_consistency_check,
    run_deterministic_checks,
    table_arithmetic_check,
)
from open_referee.pipeline.sandbox import evaluate


def _doc(md: str):
    return document_from_markdown(md, source_format="md", title_hint="t")


# --------------------------------------------------------------- cross-ref --


def test_dangling_reference_flagged():
    doc = _doc(
        "# T\n\n## 1 Methods\n\nThe estimator is described in Figure 3 and the "
        "results follow from Theorem 2. "
        + "body " * 80
        + "\n\n## Figure 1\n\nA real figure caption.\n\n"
        + "more body " * 80
    )
    findings = cross_reference_check(doc)
    dangling = [f for f in findings if f.meta.get("kind") == "dangling"]
    labels = {f.meta["label"] for f in dangling}
    assert "3" in labels  # Figure 3 does not exist
    assert all(f.category == "reference" for f in dangling)


def test_existing_reference_not_flagged_and_duplicates_flagged():
    doc = _doc(
        "# T\n\n## 1 Methods\n\nSee Figure 1 and Figure 2 for details. "
        + "body " * 80
        + "\n\n## Figure 1\n\ncaption one.\n\n## Figure 2\n\ncaption two.\n\n"
        + "## Figure 2\n\ncaption two again.\n\n" + "tail " * 80
    )
    findings = cross_reference_check(doc)
    assert not [f for f in findings if f.meta.get("kind") == "dangling"]
    assert any(f.meta.get("kind") == "duplicate" and f.meta["class"] == "Figure" for f in findings)


# ------------------------------------------------------- number consistency --


def test_conflicting_quantity_values_flagged():
    doc = _doc(
        "# T\n\n## Abstract\n\nThe model achieves a mean absolute error of 0.313 "
        "percentage points overall. " + "filler " * 60
        + "\n\n## 2 Results\n\nTable 2 reports a mean absolute error of 0.418 "
        "percentage points for the same setting. " + "filler " * 60
    )
    findings = number_consistency_check(doc)
    assert findings, "expected a consistency finding"
    f = findings[0]
    assert f.category == "consistency"
    assert "0.313" in f.title and "0.418" in f.title
    assert "0.313" in f.quote and "0.418" in f.quote


def test_rounding_consistency_not_flagged():
    doc = _doc(
        "# T\n\nThe error is 0.313 percentage points here and elsewhere reported "
        "as 0.31 percentage points. " + "filler " * 60
    )
    assert number_consistency_check(_doc(doc.full_text())) == []


def test_plain_counts_and_years_ignored():
    doc = _doc(
        "# T\n\nWe use 24 classes in 2019 and 24 classes again in 2020 with "
        "18 specifications. " + "filler " * 60
    )
    assert number_consistency_check(doc) == []


# --------------------------------------------------------- table arithmetic --


GOOD_TABLE = """| Class | Products | Share % |
|---|---:|---:|
| Food | 120 | 40.0 |
| Energy | 180 | 60.0 |
| Total | 300 | 100.0 |"""

BAD_TOTAL_TABLE = """| Class | Products | Share % |
|---|---:|---:|
| Food | 120 | 40.0 |
| Energy | 180 | 60.0 |
| Total | 480 | 100.0 |"""


def test_table_total_row_mismatch_flagged():
    doc = _doc("# T\n\n## 4 Tables\n\n" + BAD_TOTAL_TABLE + "\n\n" + "tail " * 40)
    findings = table_arithmetic_check(doc)
    assert any(f.check == "table-arithmetic" and f.meta["reported"] == 480.0 for f in findings)


def test_table_correct_totals_pass():
    doc = _doc("# T\n\n## 4 Tables\n\n" + GOOD_TABLE + "\n\n" + "tail " * 40)
    assert table_arithmetic_check(doc) == []


# ---------------------------------------------------------------- sandbox --


def test_sandbox_arithmetic_and_symbols():
    res = evaluate(
        [
            {"expression": "sqrt(v/q)", "subs": {"v": 2.0, "q": 8.0}},
            {"expression": "gamma*k + 0.5", "subs": {"gamma": 0.2, "k": 10}},
        ]
    )
    assert res[0]["ok"] and abs(res[0]["value"] - 0.5) < 1e-9
    assert res[1]["ok"] and abs(res[1]["value"] - 2.5) < 1e-9


def test_sandbox_rejects_injection_and_errors_softly():
    res = evaluate(
        [
            {"expression": "__import__('os').system('touch /tmp/pwned')"},
            {"expression": "lambda: 1"},
            {"expression": "1/0"},
        ]
    )
    assert all(not r["ok"] for r in res)
    assert "rejected" in res[0]["error"]
    assert "rejected" in res[1]["error"]


def test_sandbox_timeout_degrades():
    res = evaluate([{"expression": "factorial(10**7)"}], timeout_s=2)
    assert len(res) == 1
    assert not res[0]["ok"]


# ------------------------------------------------------------- recompute --


class _FakePool:
    def __init__(self, extract: dict, verify: dict | None):
        self.extract = extract
        self.verify = verify
        self.calls: list[str] = []

    async def complete_json(self, role, system, user, call_type="other"):
        self.calls.append(call_type)
        if call_type == "recompute_extract":
            return self.extract
        if call_type == "recompute_verify":
            assert self.verify is not None, "unexpected verify call"
            return self.verify
        raise AssertionError(f"unexpected call_type {call_type}")


class _FakePipeline:
    cfg = None

    def plan_knobs(self):
        from open_referee.pipeline.budget import PlanKnobs

        return PlanKnobs(recompute=True)

    def budget_allows(self, ct):
        return True

    class _Emit:
        async def __call__(self, *a, **k):
            return None

    _emit = _Emit()


@pytest.mark.asyncio
async def test_recompute_confirms_real_mismatch():
    from open_referee.pipeline.recompute import run_recompute

    doc = _doc(
        "# T\n\n## 3 Model\n\nThe optimal frequency follows k* = sqrt(v/q) "
        "with v=2.0 and q=8.0, giving k* = 0.9 percentage points. "
        + "filler " * 40
    )
    extract = {
        "quantities": [
            {
                "symbol": "optimal k",
                "formula": "sqrt(v/q)",
                "inputs": {"v": 2.0, "q": 8.0},
                "reported_value": 0.9,
                "quote": "giving k* = 0.9 percentage points",
                "section": "3 Model",
            }
        ]
    }
    verify = {
        "confirmed": True,
        "title": "Reported k* disagrees with its formula",
        "message": "The formula evaluates to 0.5, not 0.9.",
        "severity": 0.7,
    }
    pool = _FakePool(extract, verify)
    findings = await run_recompute(_FakePipeline(), pool, doc, None)
    assert len(findings) == 1
    assert findings[0].check == "recompute"
    assert findings[0].meta["computed"] == pytest.approx(0.5)
    assert pool.calls == ["recompute_extract", "recompute_verify"]


@pytest.mark.asyncio
async def test_recompute_consistent_value_needs_no_verification():
    from open_referee.pipeline.recompute import run_recompute

    doc = _doc("# T\n\n## 3 Model\n\nk* = sqrt(v/q) with v=2.0, q=8.0 → 0.5. " + "f " * 40)
    extract = {
        "quantities": [
            {
                "symbol": "k",
                "formula": "sqrt(v/q)",
                "inputs": {"v": 2.0, "q": 8.0},
                "reported_value": 0.5,
                "quote": "→ 0.5",
                "section": "3",
            }
        ]
    }
    pool = _FakePool(extract, None)
    findings = await run_recompute(_FakePipeline(), pool, doc, None)
    assert findings == []
    assert pool.calls == ["recompute_extract"]


# ------------------------------------------------- reference resolution (offline) --


def test_split_references_handles_numbered_and_blank():
    text = (
        "[1] Smith, J. (2001). Classical widgets. Journal of Widgets, 1(2), 3-14.\n"
        "[2] Doe, A. (2019). Modern widgets. Widgets Today, 5(1), 1-20.\n"
        "\n"
        "Berki, A. and Others (2025). Machine learning nowcasting. Journal of\n"
        "Applied Econometrics 40(2), 121-140."
    )
    refs = split_references(text)
    assert len(refs) == 3
    assert refs[0].startswith("[1]")
    assert "Applied Econometrics" in refs[2]


def test_title_guess_and_score():
    entry = (
        "[4] Cavallo, A. and Rigobon, R. (2016). The billion prices project: "
        "Using online prices for measurement and research. NBER WP."
    )
    guess = _title_guess(entry)
    assert "billion prices" in guess.lower()
    from open_referee.literature.context import LiteratureItem

    good = LiteratureItem(
        source="crossref",
        title="The Billion Prices Project: Using Online Prices for Measurement and Research",
        year=2016,
    )
    bad = LiteratureItem(
        source="crossref", title="Completely Different Paper About Fish", year=2016
    )
    assert _score(entry, guess, good) > 0.7
    assert _score(entry, guess, bad) < 0.5


# ------------------------------------------------------------ end-to-end wiring --


def test_deterministic_checks_run_on_documented_pipeline():
    md = (
        "# Broken paper\n\n## Abstract\n\nOur accuracy is 88.4 percent overall. "
        + "filler " * 60
        + "\n\n## 2 Results\n\nAs reported, our accuracy is 61.2 percent overall "
        "(see Figure 9). " + "filler " * 60
    )
    findings = run_deterministic_checks(_doc(md))
    checks = {f.check for f in findings}
    assert "consistency" in checks  # 88.4 vs 61.2
    assert "xref" in checks  # Figure 9 undefined
    # every finding serializes cleanly for the run state
    for f in findings:
        d = f.to_dict()
        json.dumps(d)
        assert d["provenance"] in ("deterministic", "recompute")
