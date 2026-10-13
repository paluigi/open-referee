"""Canonical document model with anchored blocks."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import Enum

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    EQUATION = "equation"
    TABLE = "table"
    CAPTION = "caption"
    LIST_ITEM = "list_item"
    CODE = "code"
    ABSTRACT = "abstract"


class Block(BaseModel):
    id: str
    type: BlockType = BlockType.PARAGRAPH
    text: str
    section_path: list[str] = Field(default_factory=list)  # e.g. ["3", "Methods"]
    order: int = 0
    content_hash: str = ""
    page: int | None = None  # 1-based PDF page, when known

    def model_post_init(self, __context) -> None:  # noqa: N805
        if not self.content_hash:
            self.content_hash = hashlib.sha256(normalize_text(self.text).encode()).hexdigest()[:16]


class Figure(BaseModel):
    id: str
    page: int | None = None
    image_data_url: str | None = None
    caption: str | None = None
    caption_block_id: str | None = None


class TableArtifact(BaseModel):
    """A table extracted for dedicated verification (text + optional image)."""

    id: str
    page: int | None = None
    markdown: str | None = None  # parsed cell content, when extraction worked
    image_data_url: str | None = None  # page-region render for the vision role
    caption: str | None = None
    caption_block_id: str | None = None


class TheoremEnvironment(BaseModel):
    """A theorem/lemma/proposition/definition/corollary with optional proof.

    Isolated so the math verifiers see the statement, its proof, and ONLY the
    definitions it uses — not the whole section.
    """

    kind: str  # theorem | lemma | proposition | corollary | definition
    label: str | None = None  # "Theorem 1", "Lemma 2.3"
    statement: str
    proof: str | None = None
    statement_block_id: str | None = None
    proof_block_id: str | None = None


class Section(BaseModel):
    path: list[str]
    title: str
    start_block: int
    end_block: int  # exclusive


class Document(BaseModel):
    title: str = "Untitled"
    source_format: str = "md"
    blocks: list[Block] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    tables: list[TableArtifact] = Field(default_factory=list)
    theorems: list[TheoremEnvironment] = Field(default_factory=list)
    references_text: str | None = None  # raw bibliography section text if detected

    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def find_blocks(self, quote: str) -> list[Block]:
        """Blocks whose normalized text contains the normalized quote (case-insensitive)."""
        q = normalize_text(quote).lower()
        if not q:
            return []
        return [b for b in self.blocks if q in normalize_text(b.text).lower()]

    def section_titles(self) -> list[str]:
        return [s.title for s in self.sections]


_WS = re.compile(r"\s+")


def normalize_text(t: str) -> str:
    """NFKC-normalize, unify whitespace/hyphens/quotes: robust for anchor matching."""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u00ad", "")  # soft hyphen
    t = (
        t.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    return _WS.sub(" ", t).strip()
