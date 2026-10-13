"""Shared literature types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LiteratureItem(BaseModel):
    source: str  # "openalex" | "crossref" | "web" | "user_pdf" | "review_community"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    cited_by: int | None = None
    abstract: str | None = None
    url: str | None = None
    why_relevant: str | None = None  # filled by the LLM surveyor or scout


class BibEntryCheck(BaseModel):
    """Verification result for one bibliography entry."""

    raw_entry: str
    matched_title: str | None = None
    matched_doi: str | None = None
    matched_year: int | None = None
    status: str = "unverified"  # verified | year_mismatch | title_mismatch | not_found
    note: str | None = None


class ContextPack(BaseModel):
    paper_title: str
    paper_abstract: str | None = None
    field_hint: str | None = None
    items: list[LiteratureItem] = Field(default_factory=list)
    bib_checks: list[BibEntryCheck] = Field(default_factory=list)
    user_docs: list[str] = Field(default_factory=list)  # titles of user-supplied PDFs

    def render(self, max_items: int = 25, max_abstract_chars: int = 700) -> str:
        """Compact text rendering for prompt injection."""
        parts = [f"# Field context pack (paper: {self.paper_title})"]
        if self.field_hint:
            parts.append(f"Detected field: {self.field_hint}")
        if self.user_docs:
            parts.append(
                "Author-supplied related work (prioritized reading): " + "; ".join(self.user_docs)
            )
        for i, it in enumerate(self.items[:max_items], 1):
            line = f"{i}. {it.title}"
            if it.authors:
                line += f" — {', '.join(it.authors[:3])}"
            if it.year:
                line += f" ({it.year})"
            if it.venue:
                line += f" [{it.venue}]"
            if it.doi:
                line += f" doi:{it.doi}"
            if it.cited_by is not None:
                line += f" cited_by:{it.cited_by}"
            parts.append(line)
            if it.abstract:
                a = it.abstract[:max_abstract_chars]
                parts.append(
                    f"   abstract: {a}{'…' if len(it.abstract) > max_abstract_chars else ''}"
                )
            if it.why_relevant:
                parts.append(f"   relevance: {it.why_relevant}")
        if self.bib_checks:
            suspicious = [b for b in self.bib_checks if b.status != "verified"]
            if suspicious:
                parts.append("\nBibliography entries needing attention:")
                for b in suspicious:
                    parts.append(f"- [{b.status}] {b.raw_entry[:160]} — {b.note or ''}")
        return "\n".join(parts)
