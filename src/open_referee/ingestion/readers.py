"""Format readers: PDF (pymupdf), DOCX (mammoth), LaTeX (pylatexenc), Markdown."""

from __future__ import annotations

import re
from pathlib import Path

from open_referee.ingestion.document import (
    Block,
    BlockType,
    Document,
    Figure,
    Section,
    TableArtifact,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(\d+(?:\.\d+)*)\.?|Appendix\s+([A-Z]))\s+([A-Z][^\n]{2,120})$"
)
_ABSTRACT_RE = re.compile(r"^\*{0,2}abstract\*{0,2}[:.]?\s*", re.I)
_EQ_RE = re.compile(r"^\$\$(.+)\$\$$", re.S)
_REF_RE = re.compile(r"^\*{0,2}(references|bibliography)\*{0,2}[:.]?\s*$", re.I)


def ingest_document(path: str | Path, *, extract_figures: bool = True) -> Document:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _ingest_pdf(p, extract_figures=extract_figures)
    if suffix in {".docx"}:
        return _ingest_docx(p)
    if suffix in {".tex"}:
        return _ingest_latex(p)
    if suffix in {".md", ".markdown", ".txt"}:
        return _ingest_markdown(p)
    raise ValueError(f"Unsupported input format: {suffix} (supported: .pdf .docx .tex .md)")


# ---------------------------------------------------------------- markdown --


def _ingest_markdown(p: Path) -> Document:
    text = p.read_text(errors="replace")
    return document_from_markdown(text, source_format="md", title_hint=p.stem)


def document_from_markdown(text: str, *, source_format: str, title_hint: str = "") -> Document:
    blocks: list[Block] = []
    sections: list[Section] = []
    current_section: list[str] = []  # hierarchical path, e.g. ["2", "2.3 Methods"]
    section_start = 0
    current_title = title_hint or "Untitled"
    doc_title = title_hint or "Untitled"
    references_start: int | None = None

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        hm = _HEADING_RE.match(para)
        if hm and "\n" not in para:
            level, title = len(hm.group(1)), hm.group(2).strip()
            # close previous section
            if blocks and (blocks[-1].order - section_start) > 0:
                sections.append(
                    Section(
                        path=list(current_section) or [current_title],
                        title=current_title,
                        start_block=section_start,
                        end_block=len(blocks),
                    )
                )
            if level == 1 and doc_title == (title_hint or "Untitled"):
                doc_title = title
                current_section = []
                current_title = title
                section_start = len(blocks)
                blocks.append(
                    Block(
                        id=_bid(len(blocks)),
                        type=BlockType.TITLE,
                        text=title,
                        section_path=[],
                        order=len(blocks),
                    )
                )
                continue
            current_section = [title]
            current_title = title
            section_start = len(blocks)
            blocks.append(
                Block(
                    id=_bid(len(blocks)),
                    type=BlockType.HEADING,
                    text=title,
                    section_path=[title],
                    order=len(blocks),
                )
            )
            if _REF_RE.match(title):
                references_start = len(blocks)
            continue

        btype = BlockType.PARAGRAPH
        if _ABSTRACT_RE.match(para):
            btype = BlockType.ABSTRACT
        elif _EQ_RE.match(para):
            btype = BlockType.EQUATION
        elif para.startswith(("- ", "* ", "+ ")) or re.match(r"^\d+\.\s", para):
            btype = BlockType.LIST_ITEM
        elif para.startswith(("```", "    ")):
            btype = BlockType.CODE
        if para.startswith("|") and para.rstrip().endswith("|"):
            btype = BlockType.TABLE
        blocks.append(
            Block(
                id=_bid(len(blocks)),
                type=btype,
                text=para,
                section_path=[current_title] if current_title else [],
                order=len(blocks),
            )
        )
        if references_start is None and _REF_RE.match(para):
            references_start = len(blocks)

    if blocks:
        sections.append(
            Section(
                path=[current_title],
                title=current_title,
                start_block=section_start,
                end_block=len(blocks),
            )
        )
    refs = None
    if references_start is not None:
        refs = "\n\n".join(b.text for b in blocks[references_start:])
    return Document(
        title=doc_title,
        source_format=source_format,
        blocks=blocks,
        sections=sections,
        references_text=refs,
    )


def _bid(i: int) -> str:
    return f"b{i:05d}"


# --------------------------------------------------------------------- pdf --


def _ingest_pdf(p: Path, *, extract_figures: bool) -> Document:
    import base64

    import fitz  # pymupdf

    doc = fitz.open(p)
    lines: list[str] = []
    figures: list[Figure] = []
    tables: list[TableArtifact] = []
    for pno in range(len(doc)):
        page = doc[pno]
        lines.append(page.get_text("text"))
        if extract_figures:
            for i, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:  # CMYK etc.
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                data = pix.tobytes("png")
                figures.append(
                    Figure(
                        id=f"fig_p{pno + 1}_{i}",
                        page=pno + 1,
                        image_data_url="data:image/png;base64," + base64.b64encode(data).decode(),
                    )
                )
        # tables: pymupdf finder; keep parsed cells + a rendered region image
        try:
            tabs = page.find_tables()
            for ti, tab in enumerate(tabs.tables, start=1):
                markdown = _table_to_markdown(tab.extract())
                bbox = tab.bbox
                clip = fitz.Rect(bbox)
                pix = page.get_pixmap(clip=clip, dpi=150)
                img_url = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
                tables.append(
                    TableArtifact(
                        id=f"tbl_p{pno + 1}_{ti}",
                        page=pno + 1,
                        markdown=markdown,
                        image_data_url=img_url,
                    )
                )
        except Exception:
            pass  # table detection is best-effort; never block ingestion
    raw = "\n".join(lines)
    n_pages = len(doc)
    doc.close()
    md = _pdf_text_to_markdown(raw)
    d = document_from_markdown(md, source_format="pdf", title_hint=p.stem)
    d.figures = figures
    d.tables = tables
    # approximate block->page mapping by proportional position
    for b in d.blocks:
        b.page = max(1, min(n_pages, 1 + (b.order * n_pages) // max(1, len(d.blocks))))
    return d


def _pdf_text_to_markdown(text: str) -> str:
    """Heuristic PDF text -> markdown. Paragraphs from blank-line/indent gaps;
    headings from numbered-section lines."""
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            out.append("")
            continue
        m = _NUMBERED_HEADING_RE.match(line.strip())
        if m and len(line.strip()) < 130:
            out.append(f"## {line.strip()}")
        else:
            out.append(line)
    # join consecutive non-empty, non-heading lines into paragraphs
    paras: list[str] = []
    buf: list[str] = []
    for line in out:
        if line.startswith("## ") or not line.strip():
            if buf:
                paras.append(" ".join(buf))
                buf = []
            if line.strip():
                paras.append(line)
        else:
            buf.append(line.strip())
    if buf:
        paras.append(" ".join(buf))
    return "\n\n".join(paras)


# -------------------------------------------------------------------- docx --


def _ingest_docx(p: Path) -> Document:
    import mammoth

    with open(p, "rb") as f:
        result = mammoth.convert_to_markdown(f)
    md = result.value
    # mammoth emits headings as `# Title` (ATX is default with style_map below not
    # needed for Word built-in styles)
    d = document_from_markdown(md, source_format="docx", title_hint=p.stem)
    return d


# ------------------------------------------------------------------- latex --


def _ingest_latex(p: Path) -> Document:
    from pylatexenc.latex2text import LatexNodes2Text

    text = _inline_latex_inputs(p)
    # strip preamble: everything before \begin{document} if present
    if r"\begin{document}" in text:
        text = text.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    md = LatexNodes2Text(math_mode="verbatim").latex_to_text(text)
    md = _latex_text_to_markdown(md)
    return document_from_markdown(md, source_format="latex", title_hint=p.stem)


def _inline_latex_inputs(p: Path, _depth: int = 0) -> str:
    if _depth > 5:
        return ""
    text = p.read_text(errors="replace")

    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        if not name.endswith(".tex"):
            name += ".tex"
        sub = p.parent / name
        if sub.exists():
            return _inline_latex_inputs(sub, _depth + 1)
        return ""

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


_LATEX_SECTION_RE = re.compile(r"\\(sub)*section\*?\{([^}]*)\}")


def _latex_text_to_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        m = _LATEX_SECTION_RE.search(line)
        if m:
            level = "##" if m.group(1) is None else "###"
            out.append(f"{level} {m.group(2).strip()}")
            continue
        out.append(line)
    return "\n".join(out)


def _table_to_markdown(cells) -> str:
    """pymupdf table.extract() cell matrix -> markdown rows (None -> empty)."""
    if not cells:
        return ""
    rows = []
    for row in cells:
        vals = ["" if c is None else str(c).replace("\n", " ").replace("|", "\\-") for c in row]
        rows.append("| " + " | ".join(vals) + " |")
    if len(rows) >= 1:
        n_cols = rows[0].count("|") - 1
        rows.insert(1, "|" + "---|" * n_cols)
    return "\n".join(rows)
