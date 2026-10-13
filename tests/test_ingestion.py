import hashlib

from open_referee.ingestion.document import normalize_text
from open_referee.ingestion.readers import _pdf_text_to_markdown, ingest_document


def test_markdown_ingestion_blocks_and_hashes(sample_doc):
    assert sample_doc.title == "On the Stability of Widget Networks"
    types = {b.type.value for b in sample_doc.blocks}
    assert "title" in types and "heading" in types
    for b in sample_doc.blocks:
        assert b.content_hash == hashlib.sha256(normalize_text(b.text).encode()).hexdigest()[:16]


def test_find_blocks_exact_and_normalized(sample_doc):
    hits = sample_doc.find_blocks("Every widget network is stable")
    assert len(hits) == 1
    # punctuation/whitespace-insensitive matching
    hits2 = sample_doc.find_blocks("every  widget  NETWORK is stable.")
    assert hits2


def test_references_detected(sample_doc):
    assert sample_doc.references_text and "Smith" in sample_doc.references_text


def test_md_file_ingestion(sample_paper_file):
    doc = ingest_document(sample_paper_file)
    assert doc.title.startswith("On the Stability")


def test_pdf_text_heuristics():
    raw = """On the Stability of Widget Networks
3. Main result

We now state the theorem.
Theorem 1. Every widget is stable."""
    md = _pdf_text_to_markdown(raw)
    assert "## 3. Main result" in md


def test_unsupported_format_rejected(tmp_path):
    p = tmp_path / "x.doc"
    p.write_text("nope")
    try:
        ingest_document(p)
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "Unsupported" in str(e)
