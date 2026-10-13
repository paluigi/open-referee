"""Artifact extraction: tables, theorem environments, definitions pack.

Works on the canonical Document (post-ingestion). PDF table extraction and
page-region rendering happen in readers (pymupdf-specific); this module keeps
the format-independent logic so markdown/LaTeX documents get the same
treatment.
"""

from __future__ import annotations

import re

from open_referee.ingestion.document import (
    Block,
    BlockType,
    Document,
    TableArtifact,
    TheoremEnvironment,
)

_THEOREM_RE = re.compile(
    r"^(Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Claim)\s*"
    r"([0-9]+(?:\.[0-9]+)*|[A-Z])?\s*[\.\:]?",
    re.I,
)
_PROOF_RE = re.compile(r"^(Proof|Proof\.|Proof:|Demonstration)\b", re.I)
_QED_RE = re.compile(r"(□|\$\\blacksquare\$|Q\.?E\.?D\.?\s*$)", re.I)
_DEF_RE = re.compile(
    r"^([A-Za-z][A-Za-z\- ]{0,40}?)\s+(?:is|are|denotes?|is defined as|shall be)\b"
)


def extract_theorems(doc: Document) -> list[TheoremEnvironment]:
    """Walk blocks, pair theorem statements with their proofs, capture labels."""
    out: list[TheoremEnvironment] = []
    i = 0
    blocks = doc.blocks
    while i < len(blocks):
        b = blocks[i]
        m = _THEOREM_RE.match(b.text.strip())
        if m:
            kind = m.group(1).lower()
            label = (m.group(1) + (" " + m.group(2) if m.group(2) else "")).strip()
            statement = b.text.strip()
            # absorb until Proof, a new theorem, or a heading
            j = i + 1
            while j < len(blocks):
                t = blocks[j].text.strip()
                if (
                    _PROOF_RE.match(t)
                    or _THEOREM_RE.match(t)
                    or blocks[j].type is BlockType.HEADING
                ):
                    break
                statement += "\n\n" + t
                j += 1
            env = TheoremEnvironment(
                kind=kind,
                label=label,
                statement=statement,
                statement_block_id=b.id,
            )
            # proof: immediately following blocks starting with Proof
            if j < len(blocks) and _PROOF_RE.match(blocks[j].text.strip()):
                proof = blocks[j].text.strip()
                k = j + 1
                while k < len(blocks):
                    t = blocks[k].text.strip()
                    if _THEOREM_RE.match(t) or blocks[k].type is BlockType.HEADING:
                        break
                    proof += "\n\n" + t
                    k += 1
                    if _QED_RE.search(t):
                        break
                env.proof = proof
                env.proof_block_id = blocks[j].id
                i = k
            else:
                i = j
            out.append(env)
        else:
            i += 1
    return out


def extract_tables(doc: Document) -> list[TableArtifact]:
    """Markdown pipe-table blocks become TableArtifacts (PDF tables are
    attached by the PDF reader)."""
    out: list[TableArtifact] = []
    n = 0
    for b in doc.blocks:
        if b.type is BlockType.TABLE and b.text.strip().startswith("|"):
            n += 1
            out.append(
                TableArtifact(
                    id=f"tbl_md_{n:02d}",
                    page=b.page,
                    markdown=b.text,
                    caption=_caption_near(doc, b),
                    caption_block_id=b.id,
                )
            )
    return out


def _caption_near(doc: Document, block: Block, window: int = 2) -> str | None:
    for b in doc.blocks[max(0, block.order - window) : block.order + window + 1]:
        t = b.text.strip()
        if re.match(r"^(Table|Tab\.)\s*\d+", t, re.I) or b.type is BlockType.CAPTION:
            return t
    return None


DEF_BLOCK_TYPES = {BlockType.EQUATION, BlockType.PARAGRAPH, BlockType.ABSTRACT}


def build_definitions_pack(doc: Document, env: TheoremEnvironment) -> str:
    """Compact pack of definitions/notation the theorem environment uses.

    Heuristic symbol collection: symbols appearing in the statement/proof are
    matched against blocks that define things (the _DEF_RE pattern, equation
    blocks, and earlier theorem environments of kind definition/assumption).
    Kept small — this is context, not the full paper.
    """
    body = env.statement + "\n" + (env.proof or "")
    symbols = _collect_symbols(body)
    lines: list[str] = []
    seen: set[str] = set()

    # explicit definition/assumption environments first
    for other in doc.theorems:
        if other.kind in ("definition", "assumption") and other.label and other.label not in seen:
            if other is not env and (
                _overlap(symbols, other.statement) >= 2 or _same_section(doc, env, other)
            ):
                lines.append(f"[{other.label}] {other.statement[:600]}")
                seen.add(other.label)

    # prose definitions and numbered equations
    for b in doc.blocks:
        t = b.text.strip()
        if not t or t in seen:
            continue
        is_def = bool(_DEF_RE.match(t))
        is_eq = b.type is BlockType.EQUATION or re.match(r"^\(\d+\)", t)
        if not (is_def or is_eq):
            continue
        if _overlap(symbols, t) >= 2:
            label = b.id
            lines.append(f"[{label}] {t[:400]}")
            seen.add(t)
        if len(lines) >= 25:
            break
    if not lines:
        return "(no definitions matched)"
    return "\n\n".join(lines)


def _same_section(doc: Document, a: TheoremEnvironment, b: TheoremEnvironment) -> bool:
    def first_section(bid: str | None) -> str | None:
        if not bid:
            return None
        for blk in doc.blocks:
            if blk.id == bid:
                return blk.section_path[0] if blk.section_path else None
        return None

    sa, sb = first_section(a.statement_block_id), first_section(b.statement_block_id)
    return sa is not None and sa == sb


def _collect_symbols(text: str) -> set[str]:
    """Symbols worth tracking: math tokens, subscripted names, call-style names."""
    syms: set[str] = set()
    for m in re.finditer(r"\b[A-Za-z](?:_[A-Za-z0-9]+)?\b", text):
        syms.add(m.group(0))
    for m in re.finditer(r"\\\\[A-Za-z]+", text):  # latex commands surviving extraction
        syms.add(m.group(0))
    return {s for s in syms if len(s) <= 12}


def _overlap(symbols: set[str], text: str) -> int:
    toks = set(re.findall(r"\b[A-Za-z](?:_[A-Za-z0-9]+)?\b|\\\\[A-Za-z]+", text))
    return len(symbols & toks)
