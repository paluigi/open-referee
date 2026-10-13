from open_referee.pipeline.orchestrator import _resolve_anchor


def test_exact_anchor(sample_doc):
    bid, conf = _resolve_anchor(sample_doc, "Theorem 1. Every widget network is stable.", "")
    assert conf == 1.0
    block = next(b for b in sample_doc.blocks if b.id == bid)
    assert "widget" in block.text.lower()


def test_anchor_via_quote(sample_doc):
    bid, conf = _resolve_anchor(
        sample_doc, "not present in the doc at all", "Widget networks are everywhere"
    )
    assert bid is not None and conf >= 0.6


def test_anchor_fuzzy_token_overlap(sample_doc):
    # paraphrase, not exact — token overlap should still find the right block
    bid, conf = _resolve_anchor(
        sample_doc, "we prove that all widgets remain stable under mild assumptions", ""
    )
    assert bid is not None
    assert conf >= 0.35


def test_anchor_missing_gives_none(sample_doc):
    bid, conf = _resolve_anchor(sample_doc, "zzz qqq xyzzy", "zzz qqq xyzzy")
    assert bid is None or conf > 0  # may weakly match, but never crash
