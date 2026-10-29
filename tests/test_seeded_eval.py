"""Seeded-error benchmark, CI variant: the deterministic checkers must catch
every mechanically-checkable seed, and the clean baseline must stay quiet.

(The full harness with LLM-found seeds lives in local_test/seed_errors.py —
uncommitted by design; this test keeps the wiring regression-covered.)
"""

from __future__ import annotations

from open_referee.ingestion.readers import document_from_markdown
from open_referee.pipeline.checkers import run_deterministic_checks

SEEDED = """# Seeded Methods Paper

## Abstract

We evaluate a nowcasting procedure on 24 classes. The procedure achieves a
mean absolute error of 0.314 percentage points pooled across classes, with
directional accuracy of 81.5 percent.

## 1 Introduction

Nowcasting with high-frequency data is trivial for any dataset.

## 2 Data and protocol

| Class | Products | Share % |
|---|---:|---:|
| Food | 120 | 40.0 |
| Energy | 180 | 60.0 |
| Total | 480 | 100.0 |

Table 1: class counts by category.

## 3 Results

The pooled mean absolute error is 0.418 percentage points (see Table 1).
Figure 9 plots the error by class.

Figure 1: pooled error by class.
"""

CLEAN = """# Seeded Methods Paper

## Abstract

We evaluate a nowcasting procedure on 24 classes. The procedure achieves a
mean absolute error of 0.314 percentage points pooled across classes.

## 1 Introduction

Nowcasting with high-frequency data requires care.

## 2 Data and protocol

| Class | Products | Share % |
|---|---:|---:|
| Food | 120 | 40.0 |
| Energy | 180 | 60.0 |
| Total | 300 | 100.0 |

Table 1: class counts by category.

## 3 Results

The pooled mean absolute error is 0.314 percentage points (see Table 1).
Figure 1 plots the error by class.

Figure 1: pooled error by class.
"""


def test_seeded_errors_all_caught_by_deterministic_checkers():
    doc = document_from_markdown(SEEDED, source_format="md", title_hint="seeded")
    findings = run_deterministic_checks(doc)
    checks = [f.check for f in findings]
    quotes = " ".join(f.quote + f.title for f in findings)
    # S1: table total row 480 != 120+180
    assert "table-arithmetic" in checks and "480" in quotes
    # S2: abstract 0.314 vs body 0.418
    assert "consistency" in checks and "0.314" in quotes and "0.418" in quotes
    # S3: Figure 9 referenced, never defined
    assert "xref" in checks and "Figure 9" in quotes


def test_clean_baseline_stays_quiet():
    doc = document_from_markdown(CLEAN, source_format="md", title_hint="clean")
    findings = run_deterministic_checks(doc)
    # the clean paper has no broken refs, no contradictions, correct sums
    assert [f for f in findings if f.check in ("consistency", "table-arithmetic")] == []
    assert not [f for f in findings if "Figure 9" in f.title]
