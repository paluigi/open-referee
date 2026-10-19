"""Deterministic checkers: cross-references, number consistency, table arithmetic.

These run with zero LLM cost and produce :class:`CheckFinding` records that
join the candidate-comment stream (provenance ``deterministic``). All
thresholds are conservative: a finding that is wrong is worse than a missing
one, so ambiguous structures (unparsable tables, unit-less plain integers)
are skipped rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from open_referee.ingestion.document import Document

# ----------------------------------------------------------------- findings --


@dataclass
class CheckFinding:
    check: str  # xref | consistency | table-arithmetic | doi-resolution | recompute
    title: str
    message: str
    quote: str = ""
    section: str | None = None
    category: str = "consistency"
    severity: float = 0.5
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["provenance"] = "deterministic" if self.check != "recompute" else "recompute"
        return d


# ------------------------------------------------------ cross-reference check --

_LABEL_DEF_RE = re.compile(
    r"^(figure|table|theorem|lemma|proposition|corollary|definition|"
    r"algorithm|appendix)\s+([0-9]+[A-Z]?|[A-Z])(?![0-9A-Za-z])",
    re.I,
)
_LABEL_REF_RE = re.compile(
    r"\b(figure|fig\.|table|theorem|thm\.|lemma|proposition|prop\.|corollary|"
    r"definition|algorithm)\s*~?\s*([0-9]+[A-Z]?)\b"
    r"|\b(appendix)\s*~?\s+([A-Z])(?![a-z])",
    re.I,
)
_EQ_DEF_RE = re.compile(r"(?:\\tag\{([0-9]+[a-z]?)\}|\(([0-9]+[a-z]?)\)\s*$)")
_EQ_REF_RE = re.compile(
    r"\b(?:equation|eq\.?s?|expression)\s*\(?([0-9]+[a-z]?)\)?", re.I
)
_SECTION_REF_RE = re.compile(r"\b(?:section|sect\.?)\s+([0-9]+(?:\.[0-9]+)*)\b", re.I)
_NUM_MONOTONE_RE = re.compile(r"^[0-9]+")
# a line anywhere in the text that looks like a numbered (sub)section heading
_ANY_NUMBERED_LINE_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,2}[.:]?\s+[A-Z]\S.*$")

_CLASS_CANON = {
    "figure": "Figure", "fig.": "Figure", "table": "Table", "theorem": "Theorem",
    "thm.": "Theorem", "lemma": "Lemma", "proposition": "Proposition",
    "prop.": "Proposition", "corollary": "Corollary", "definition": "Definition",
    "algorithm": "Algorithm", "appendix": "Appendix", "equation": "Equation",
    "eq.": "Equation", "eqs.": "Equation", "expression": "Equation",
}
_SKIP_CLASS_FOR_REFS = {"Definition"}  # definitions are rarely referenced by number
_MAX_XREF_FINDINGS = 8


def cross_reference_check(doc: Document) -> list[CheckFinding]:
    findings: list[CheckFinding] = []
    defined: dict[str, list[tuple[str, str]]] = {}  # class -> [(label, block_id)]

    def define(cls: str, label_id: str, block_id: str) -> None:
        defined.setdefault(cls, []).append((label_id, block_id))

    for th in doc.theorems:
        m = re.match(r"([A-Za-z]+)\s+(.+)", th.label or "")
        if m and m.group(1).lower() in _CLASS_CANON:
            cls = _CLASS_CANON[m.group(1).lower()]
            define(cls, m.group(2), th.statement_block_id or "")
    for fig in doc.figures:
        if fig.caption:
            m = re.match(r"(?:figure|fig\.?)\s*([0-9]+[A-Z]?)", fig.caption.strip(), re.I)
            if m:
                define("Figure", m.group(1), fig.caption_block_id or "")
    for tab in doc.tables:
        if tab.caption:
            m = re.match(r"(?:table)\s*([0-9]+[A-Z]?)", tab.caption.strip(), re.I)
            if m:
                define("Table", m.group(1), tab.caption_block_id or "")
    for b in doc.blocks:
        # label definitions may sit at the start of any line inside a block
        # (captions, bold pseudo-headings, theorem statements)
        for line in b.text.split("\n")[:3]:
            s = line.strip()
            if not (3 < len(s) < 220):
                continue
            if b.type.value == "paragraph" and len(s) >= 120:
                continue  # long body paragraphs are prose, not labels
            m = _LABEL_DEF_RE.match(s)
            if m and _looks_like_definition(s, m.end()):
                define(_CLASS_CANON[m.group(1).lower()], m.group(2), b.id)
                break
            if b.type.value == "equation":
                em = _EQ_DEF_RE.search(s)
                if em:
                    define("Equation", em.group(1) or em.group(2), b.id)
                    break

    # section-number registry: parsed sections plus ANY numbered heading-like
    # line in the text (bold/plain subsections that never parsed as headings)
    section_numbers: set[str] = set()
    for s in doc.sections:
        m = re.match(r"^([0-9]+(?:\.[0-9]+)*)\.?\s", s.title)
        if m:
            section_numbers.add(m.group(1))
    for b in doc.blocks:
        for line in b.text.split("\n"):
            if _ANY_NUMBERED_LINE_RE.match(line) and len(line.strip()) < 120:
                m = re.match(r"\s*(\d{1,2}(?:\.\d{1,2}){0,2})", line)
                if m:
                    section_numbers.add(m.group(1))

    # duplicate labels + non-monotone numbering per class
    duplicated_classes: set[str] = set()
    for cls, items in defined.items():
        seen: dict[str, str] = {}
        for label_id, block_id in items:
            if label_id in seen:
                if cls not in duplicated_classes:
                    duplicated_classes.add(cls)
                    findings.append(
                        CheckFinding(
                            check="xref",
                            title=f"Duplicate label {cls} {label_id}",
                            message=(
                                f"{cls} {label_id} is defined more than once; "
                                "references to it are ambiguous. This usually "
                                "means a renumbering after edits missed a copy."
                            ),
                            quote=_quote_block(doc, block_id),
                            category="reference",
                            severity=0.45,
                            meta={"class": cls, "label": label_id, "kind": "duplicate"},
                        )
                    )
                break  # one report per class is enough
            seen[label_id] = block_id
        if cls in duplicated_classes:
            continue
        nums = [
            int(m.group(0))
            for m in (_NUM_MONOTONE_RE.match(lid) for lid, _ in items)
            if m
        ]
        if len(nums) >= 3 and nums != sorted(nums):
            findings.append(
                CheckFinding(
                    check="xref",
                    title=f"{cls} numbering is not monotone",
                    message=(
                        f"{cls} numbers appear out of order in the document "
                        f"({nums}); check for renumbering after edits."
                    ),
                    category="reference",
                    severity=0.3,
                    meta={"class": cls, "kind": "ordering", "sequence": nums},
                )
            )

    # dangling references, aggregated per (class,label) and per section-number.
    # classes with zero definitions anywhere (extraction loss) collapse into
    # one low-severity note instead of one finding per label.
    dangling: dict[tuple[str, str], dict] = {}
    orphan_classes: dict[str, int] = {}
    missing_sections: dict[str, dict] = {}
    for b in doc.blocks:
        if b.type.value == "heading":
            continue
        for m in _LABEL_REF_RE.finditer(b.text):
            cls_raw, label_id = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            cls = _CLASS_CANON.get(cls_raw.lower())
            if cls is None or cls in _SKIP_CLASS_FOR_REFS:
                continue
            if label_id in {lid for lid, _ in defined.get(cls, [])}:
                continue
            if not defined.get(cls):
                orphan_classes[cls] = orphan_classes.get(cls, 0) + 1
                continue
            key = (cls, label_id)
            slot = dangling.setdefault(key, {"count": 0, "quote": "", "section": None})
            slot["count"] += 1
            if not slot["quote"]:
                slot["quote"] = _window(b.text, m.start())
                slot["section"] = _section_of(doc, b.id)
        for m in _EQ_REF_RE.finditer(b.text):
            eq_id = m.group(1)
            if not re.fullmatch(r"[0-9]+", eq_id):
                continue
            if defined.get("Equation") and eq_id not in {lid for lid, _ in defined["Equation"]}:
                key = ("Equation", eq_id)
                slot = dangling.setdefault(key, {"count": 0, "quote": "", "section": None})
                slot["count"] += 1
                if not slot["quote"]:
                    slot["quote"] = _window(b.text, m.start())
                    slot["section"] = _section_of(doc, b.id)
        for m in _SECTION_REF_RE.finditer(b.text):
            if section_numbers and m.group(1) not in section_numbers:
                slot = missing_sections.setdefault(
                    m.group(1), {"count": 0, "quote": "", "section": None}
                )
                slot["count"] += 1
                if not slot["quote"]:
                    slot["quote"] = _window(b.text, m.start())
                    slot["section"] = _section_of(doc, b.id)

    for (cls, label_id), slot in sorted(dangling.items()):
        findings.append(
            CheckFinding(
                check="xref",
                title=f"Reference to undefined {cls} {label_id}"
                + (f" ({slot['count']}×)" if slot["count"] > 1 else ""),
                message=(
                    f"The text refers to {cls} {label_id}, but no {cls} with "
                    "that number is defined anywhere in the document"
                    + (f" — {slot['count']} references in total" if slot["count"] > 1 else "")
                    + "."
                ),
                quote=slot["quote"],
                section=slot["section"],
                category="reference",
                severity=0.35,
                meta={"class": cls, "label": label_id, "kind": "dangling",
                      "occurrences": slot["count"]},
            )
        )
    if missing_sections:
        worst = sorted(missing_sections.items(), key=lambda kv: -kv[1]["count"])[0]
        nums_str = ", ".join(
            f"{n} ({s['count']}×)" if s["count"] > 1 else n
            for n, s in sorted(
                missing_sections.items(), key=lambda kv: kv[1]["count"], reverse=True
            )[:6]
        )
        findings.append(
            CheckFinding(
                check="xref",
                title="References to sections without matching headings",
                message=(
                    "The text cites section numbers that do not match any "
                    f"section heading or numbered block: {nums_str}. Check "
                    "whether a renumbering or a formatting loss in conversion "
                    "broke the internal structure."
                ),
                quote=worst[1]["quote"],
                section=worst[1]["section"],
                category="reference",
                severity=0.35,
                meta={
                    "kind": "section-refs",
                    "missing": {n: s["count"] for n, s in missing_sections.items()},
                },
            )
        )
    if orphan_classes:
        for cls, n in sorted(orphan_classes.items()):
            findings.append(
                CheckFinding(
                    check="xref",
                    title=f"No {cls.lower()} captions found for {n} reference(s)",
                    message=(
                        f"The text references {cls.lower()}s {n} times, but the "
                        "document contains no "
                        f"{cls.lower()} captions or definitions at all — most "
                        "likely the figures were images lost in conversion. "
                        "Internal reference checking for this class is "
                        "therefore not possible; treat as unverified."
                    ),
                    category="reference",
                    severity=0.2,
                    meta={"class": cls, "kind": "orphan-class", "references": n},
                )
            )
    return findings[:_MAX_XREF_FINDINGS]


_DEF_PUNCT = (":", ".", ",", ";", ")", "—", "–", "-")


def _looks_like_definition(line: str, match_end: int) -> bool:
    """A label at line start is a DEFINITION only when caption punctuation
    follows it ('Figure 1:', 'Theorem 2.', 'Appendix A.'). A letter follows
    in prose ('Figure 9 plots...', 'Table 1 reports...') — that is a
    reference, not a definition."""
    rest = line[match_end:].lstrip()
    if not rest:
        return True
    return rest[0] in _DEF_PUNCT


def _quote_block(doc: Document, block_id: str) -> str:
    for b in doc.blocks:
        if b.id == block_id:
            return b.text[:240]
    return ""


def _section_of(doc: Document, block_id: str) -> str | None:
    for b in doc.blocks:
        if b.id == block_id:
            return b.section_path[0] if b.section_path else None
    return None


def _window(text: str, pos: int, width: int = 110) -> str:
    lo = max(0, pos - width // 2)
    return text[lo : pos + width].strip()[:240]


# ----------------------------------------------------- number consistency --

_NUM_RE = re.compile(
    r"(?<![0-9A-Za-z.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?![0-9])"
    r"\s?(%|pp\b|percentage points|per cent\b)?",
    re.I,
)
_UNIT_MENTION = ("%", "pp", "percentage points", "per cent")
_DOI_TOKEN_RE = re.compile(r"^10\.\d{4,}")
_LATEX_PREFIX_RE = re.compile(r"(\\[a-z]+|\barray\b|\bbegin\b|\bend\b|\bfrac\b|\\\$)")
_GENERIC_TAIL = {
    "estimate", "estimates", "value", "values", "number", "numbers",
    "result", "results", "increase", "decrease", "level", "levels",
    "rate", "rates", "size", "sizes", "sample", "samples",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
}
_CONNECTORS = {
    "of", "is", "as", "was", "at", "in", "to", "and", "the", "a", "an",
    "for", "with", "by", "on", "from", "are", "were", "has", "have",
    "equals", "equal", "per", "about", "over", "under",
}
_PROSE_BLOCKS = ("paragraph", "abstract", "caption")
_STRUCTURAL_TAIL = {
    "section", "sections", "figure", "figures", "table", "tables",
    "equation", "equations", "appendix", "page", "pages", "fig",
    "chapter", "step", "steps", "period", "wave",
}


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def _value(tok: str) -> float:
    return float(tok.replace(",", ""))


def _consistent(a_tok: str, b_tok: str) -> bool:
    a, b = _value(a_tok), _value(b_tok)
    min_dec = min(_decimals(a_tok), _decimals(b_tok))
    if min_dec:
        if abs(a - b) <= 0.6 * 10 ** -min_dec:
            return True
    return abs(a - b) <= max(1e-9, 0.01 * max(abs(a), abs(b)))


def number_consistency_check(doc: Document) -> list[CheckFinding]:
    """Flag the same named quantity reported with different values.

    Scope is deliberately narrow to keep the false-positive rate near zero:
    mentions in the abstract (or the opening paragraphs when no abstract
    block exists) are compared against mentions anywhere else in the prose.
    Two mentions refer to the same quantity when their preceding word
    templates share their last two words ("our accuracy is" ~ "reported our
    accuracy is"). Table blocks, LaTeX, DOIs, years and unit-less integers
    are excluded; generic or structural templates (a "value of", "section
    5.1") never match."""
    abstract_blocks, body_blocks = _split_abstract(doc)
    if not abstract_blocks:
        return []
    head = _mentions(abstract_blocks)
    if not head:
        return []
    tail = _mentions([b for b in body_blocks if b.type.value in _PROSE_BLOCKS])

    findings: list[CheckFinding] = []
    for hm in head:
        candidates = [
            tm
            for tm in tail
            if _same_quantity(hm["words"], tm["words"]) and not _consistent(hm["tok"], tm["tok"])
        ]
        # contradictions inside the abstract itself count too
        inside = [
            om
            for om in head
            if om is not hm and _same_quantity(hm["words"], om["words"])
            and not _consistent(hm["tok"], om["tok"])
        ]
        conflicts = candidates + inside
        if not conflicts:
            continue
        other = conflicts[0]
        a, b = sorted([hm["tok"], other["tok"]])
        template = " ".join(hm["words"][-3:])
        dup = any(
            f.meta.get("template") == template
            and f.meta.get("values", [None, None])[:2] == [a, b]
            for f in findings
        )
        if dup:
            continue
        quotes = [
            f"[{hm['block'].type.value}] …{_window(hm['block'].text, hm['pos'], 90)}…",
            f"[{other['block'].type.value}] …{_window(other['block'].text, other['pos'], 90)}…",
        ]
        findings.append(
            CheckFinding(
                check="consistency",
                title=f'Conflicting values for "{template}" ({a} vs {b})',
                message=(
                    f'The quantity described as "{template}" is reported as {a} '
                    f"in one place and {b} in another; at most one can be "
                    "correct. Verify which value is right and align the others."
                ),
                quote="\n\n".join(quotes),
                category="consistency",
                severity=0.7,
                meta={"template": template, "values": [a, b]},
            )
        )
    return findings


def _split_abstract(doc: Document) -> tuple[list, list]:
    abstract = [b for b in doc.blocks if b.type.value == "abstract"]
    if abstract:
        rest = [b for b in doc.blocks if b.type.value != "abstract"]
        return abstract, rest
    # no explicit abstract block: the first few paragraphs before the first
    # real section heading act as the abstract proxy
    head: list = []
    rest: list = []
    seen_heading = False
    for b in doc.blocks:
        if b.type.value == "title":
            rest.append(b)
            continue
        if b.type.value == "heading":
            if (
                not seen_heading
                and b.text.strip().lower().rstrip(":.") in ("abstract", "summary")
            ):
                continue  # abstract heading itself — what follows is still front matter
            seen_heading = True
            rest.append(b)
            continue
        (rest if (seen_heading or len(head) >= 3) else head).append(b)
    return head, rest


def _mentions(blocks: list) -> list[dict]:
    out: list[dict] = []
    for b in blocks:
        for m in _NUM_RE.finditer(b.text):
            tok, unit = m.group(1), m.group(2)
            if "." not in tok and unit is None:
                continue  # plain integers: counts/years — not quantity claims
            if _decimals(tok) == 0 and abs(_value(tok)) < 10 and unit is None:
                continue
            if re.fullmatch(r"(19|20)\d\d", tok):
                continue  # years
            if _DOI_TOKEN_RE.match(tok):
                continue  # DOIs in references
            prefix_raw = b.text[max(0, m.start() - 60) : m.start()]
            if _LATEX_PREFIX_RE.search(prefix_raw):
                continue  # inside math/LaTeX
            prefix = re.sub(r"[\d,.%]+", " # ", prefix_raw.lower())
            words = re.findall(r"[a-z][a-z-]{1,}", prefix)
            if len(words) < 2:
                continue
            content_words = [w for w in words if w not in _CONNECTORS]
            if not content_words or content_words[-1] in _GENERIC_TAIL | _STRUCTURAL_TAIL:
                continue  # names no specific quantity (a value of / section 5.1)
            out.append({"tok": tok, "unit": unit, "words": words, "block": b, "pos": m.start()})
    return out


def _same_quantity(words_a: list[str], words_b: list[str]) -> bool:
    """Two mention prefixes name the same quantity when their last two
    content words (connectors ignored) agree — "mean absolute error of" and
    "pooled mean absolute error is" both reduce to (absolute, error)."""
    ca = [w for w in words_a if w not in _CONNECTORS]
    cb = [w for w in words_b if w not in _CONNECTORS]
    return len(ca) >= 2 and len(cb) >= 2 and ca[-2:] == cb[-2:]


# ------------------------------------------------------- table arithmetic --

_TOTAL_RE = re.compile(r"^(total|sum|all)\b", re.I)
_ROW_SPLIT = re.compile(r"\s*\|\s*")


def _parse_md_table(md: str) -> list[list[str]] | None:
    rows: list[list[str]] = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in _ROW_SPLIT.split(line.strip("|"))]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue  # separator row
        rows.append(cells)
    if len(rows) < 3:
        return None
    return rows


def _num(cell: str) -> float | None:
    s = cell.strip().replace(",", "").replace("%", "").replace("−", "-")
    m = re.fullmatch(r"\(?\s*-?\d+(?:\.\d+)?\s*\)?", s)
    if not m:
        return None
    neg = s.startswith("(") and s.endswith(")")
    v = float(s.strip("()"))
    return -v if neg else v


def table_arithmetic_check(doc: Document) -> list[CheckFinding]:
    findings: list[CheckFinding] = []
    tables = [(t.id, t.caption, t.markdown) for t in doc.tables if t.markdown]
    tables += [
        (b.id, None, b.text) for b in doc.blocks if b.type.value == "table"
    ]
    for tid, caption, md in tables:
        rows = _parse_md_table(md)
        if not rows:
            continue
        header, body = rows[0], rows[1:]
        # Total row: each numeric cell should equal the column sum above it
        for i, row in enumerate(body):
            if not row or not _TOTAL_RE.match(row[0] or ""):
                continue
            for col in range(1, len(row)):
                total = _num(row[col])
                if total is None:
                    continue
                above = [
                    _num(r[col]) for r in body[:i] if col < len(r)
                ]
                above = [v for v in above if v is not None]
                if len(above) < 2:
                    continue
                s = sum(above)
                tol = max(
                    0.5 * 10 ** -_decimals(f"{total:.6f}".rstrip("0")),
                    abs(total) * 0.005,
                    1e-9,
                )
                if abs(s - total) > tol:
                    label = header[col] if col < len(header) else f"column {col}"
                    findings.append(
                        CheckFinding(
                            check="table-arithmetic",
                            title=f'Total row does not sum: {label} ({tid})',
                            message=(
                                f'The "{row[0]}" row reports {total:g} for '
                                f'"{label}", but the rows above sum to {s:g}. '
                                "Recomputed from the table itself."
                                + (f" ({caption})" if caption else "")
                            ),
                            quote=_table_quote(md, row),
                            category="consistency",
                            severity=0.7,
                            meta={"table": tid, "column": label,
                                  "reported": total, "computed": s},
                        )
                    )
        # "total"-named column: each row's cell should equal the row's part sum
        for col, h in enumerate(header):
            if col == 0 or not re.search(r"\b(total|sum)\b", h, re.I):
                continue
            for row in body:
                if _TOTAL_RE.match(row[0] or ""):
                    continue
                total = _num(row[col]) if col < len(row) else None
                if total is None:
                    continue
                parts = [
                    _num(row[c])
                    for c in range(1, len(row))
                    if c != col and c < len(row) and not re.search(
                        r"\b(total|share|sum|%)\b", header[c], re.I
                    )
                ]
                parts = [v for v in parts if v is not None]
                if len(parts) < 2:
                    continue
                s = sum(parts)
                if abs(s - total) > max(abs(total) * 0.005, 0.01):
                    findings.append(
                        CheckFinding(
                            check="table-arithmetic",
                            title=f'Row total mismatch in {tid}: {row[0]}',
                            message=(
                                f'Row "{row[0]}": the "{h}" column reports '
                                f"{total:g} but the row's other numeric columns "
                                f"sum to {s:g}."
                            ),
                            quote=_table_quote(md, row),
                            category="consistency",
                            severity=0.7,
                            meta={"table": tid, "row": row[0],
                                  "reported": total, "computed": s},
                        )
                    )
    return findings


def _table_quote(md: str, row: list[str]) -> str:
    line = next(
        (
            ln
            for ln in md.splitlines()
            if ln.strip().startswith("|") and row and row[0] and row[0] in ln
        ),
        "",
    )
    return line.strip()[:240] or " | ".join(row)[:240]


# ------------------------------------------------------------------- entry --


def run_deterministic_checks(doc: Document) -> list[CheckFinding]:
    out: list[CheckFinding] = []
    out += cross_reference_check(doc)
    out += number_consistency_check(doc)
    out += table_arithmetic_check(doc)
    return out
