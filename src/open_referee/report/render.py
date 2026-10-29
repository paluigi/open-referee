"""Report renderers: markdown and PDF (weasyprint if available, reportlab-fallback)."""

from __future__ import annotations

import re
from pathlib import Path

from open_referee.report.models import (
    TOOL_NAME,
    TOOL_URL,
    FitReport,
    PlanSummary,
    ReviewReport,
)


def _fmt_int(n: float) -> str:
    return f"{int(n):,}"


def _usage_table(usage, model_roles: dict[str, str]) -> list[str]:
    """Markdown table: one row per role/model + totals. Empty if no usage."""
    if not usage.tokens_by_role:
        return []
    rows = [
        "## Build metadata",
        "",
        "| Role | Model | Calls | Input tokens | Output tokens | Est. cost |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for role in sorted(usage.tokens_by_role):
        agg = usage.tokens_by_role[role]
        model = model_roles.get(role, "?")
        rows.append(
            f"| {role} | `{model}` | {int(agg.get('calls', 0))} | "
            f"{_fmt_int(agg.get('input_tokens', 0))} | {_fmt_int(agg.get('output_tokens', 0))} | "
            f"${agg.get('cost_usd', 0.0):.4f} |"
        )
    rows.append(
        f"| **Total** | | {sum(int(a.get('calls', 0)) for a in usage.tokens_by_role.values())} | "
        f"**{_fmt_int(usage.total_input_tokens)}** | **{_fmt_int(usage.total_output_tokens)}** | "
        f"**${usage.estimated_cost_usd:.4f}** |"
    )
    rows.append("")
    return rows


def _plan_lines(plan: PlanSummary | None) -> list[str]:
    if plan is None:
        return []
    out: list[str] = []
    bits = [f"Budget: ${plan.budget_usd:.2f}" if plan.budget_usd is not None else None]
    if plan.target_usd is not None:
        bits.append(f"planned target ${plan.target_usd:.2f}")
    bits.append(f"spent ${plan.spent_usd:.2f}")
    if plan.control_spent_usd:
        bits.append(f"control-plane overhead ${plan.control_spent_usd:.2f}")
    if plan.projected_final_usd is not None and plan.mode != "fit":
        bits.append(f"projected ${plan.projected_final_usd:.2f}")
    if plan.mode == "auto":
        bits.append(f"{plan.planner_calls_used} planner call(s)")
    out.append("_" + " · ".join(b for b in bits if b) + f" ({plan.mode} mode)_")
    for adj in plan.adjustments[-8:]:  # keep the report readable
        out.append(f"- adjustment: {adj}")
    if plan.notes:
        out.append(f"- {plan.notes}")
    out.append("")
    return out


def _attribution_lines() -> list[str]:
    return [
        "---",
        "",
        f"Built with [{TOOL_NAME}]({TOOL_URL})",
        "",
    ]


_DETERMINISTIC_PROVENANCE = ("deterministic", "recompute")


def _comment_header(i: int, c) -> str:
    badges = [f"severity {c.score:.2f}", str(c.category or "general")]
    if c.provenance:
        badges.append(c.provenance)
    tag = " · ".join(f"[{b}]" for b in badges)
    sec = f" — _{c.section}_" if c.section else ""
    return f"### {i}. {c.title} {sec}", tag


def render_markdown(report: ReviewReport) -> str:
    parts = [f"# Review report — {report.paper_title}", ""]
    parts.append(f"_Generated {report.generated_at:%Y-%m-%d %H:%M} UTC · run `{report.run_id}`_")
    if report.model_roles:
        roles = ", ".join(f"{role}: `{spec}`" for role, spec in sorted(report.model_roles.items()))
        parts.append(f"_Models — {roles}_")
    parts += ["", "## Paper summary", "", report.paper_summary or "_(none)_", ""]
    parts += ["## Overall feedback", "", report.overall_feedback or "_(none)_", ""]

    # findings by relevance (severity), deterministic ones in their own section
    llm_comments = [
        c for c in report.sorted_comments() if (c.provenance or "") not in _DETERMINISTIC_PROVENANCE
    ]
    det_comments = [
        c for c in report.sorted_comments() if (c.provenance or "") in _DETERMINISTIC_PROVENANCE
    ]
    parts += [f"## Findings ({len(llm_comments)})", ""]
    parts.append(
        "_Sorted by relevance (severity). Provenance tags show how each finding "
        "was verified: `claim_verify 2-of-2` means it reproduced in two "
        "independent passes; `arbitrated` findings passed a strong-model gate; "
        "`auditor` entries were produced or cross-checked by a second model._"
    )
    parts.append("")
    for i, c in enumerate(llm_comments, 1):
        header, tag = _comment_header(i, c)
        parts.append(header)
        parts.append(f"_{tag}_")
        if c.quote:
            parts += ["", f"> {c.quote}"]
        parts += ["", c.message, ""]
    if det_comments:
        parts += [f"## Deterministic checks ({len(det_comments)})", ""]
        parts.append(
            "_Produced by code, not language models: cross-reference resolution, "
            "number consistency, table arithmetic, formula recomputation "
            "(sympy), and reference resolution against Crossref/OpenAlex._"
        )
        parts.append("")
        for i, c in enumerate(det_comments, 1):
            header, tag = _comment_header(i, c)
            parts.append(header)
            parts.append(f"_{tag}_")
            if c.quote:
                parts += ["", f"> {c.quote}"]
            parts += ["", c.message, ""]
    if report.validator_notes:
        parts += ["## Validator notes", "", report.validator_notes, ""]
    if report.copyedit_items:
        parts += [
            f"## Suggested language fixes ({len(report.copyedit_items)})",
            "",
            "_Mechanical language items from the copyedit pass — typos, grammar, "
            "phrasing, terminology. Each quote is verified to appear in the "
            "manuscript; these are copyedits, not findings, and carry no "
            "severity._",
            "",
            "| # | Kind | Location | Original | Suggested fix |",
            "| --- | --- | --- | --- | --- |",
        ]
        for i, it in enumerate(report.copyedit_items, 1):
            q = it.quote.replace("|", "\\|").replace("\n", " ")
            s = it.suggestion.replace("|", "\\|").replace("\n", " ")
            parts.append(f"| {i} | {it.kind} | {it.section or '—'} | {q} | {s} |")
        parts.append("")
    if report.plan_summary:
        parts += _plan_lines(report.plan_summary)
    parts += _usage_table(report.usage, report.model_roles)
    parts += _attribution_lines()
    return "\n".join(parts)


def render_fit_markdown(report: FitReport) -> str:
    p = report.probability_not_desk_rejected
    parts = [f"# Journal fit estimate — {report.paper_title}", ""]
    parts.append(f"_Generated {report.generated_at:%Y-%m-%d %H:%M UTC} · run `{report.run_id}`_")
    if report.model_roles:
        roles = ", ".join(f"{role}: `{spec}`" for role, spec in sorted(report.model_roles.items()))
        parts.append(f"_Models — {roles}_")
    parts += [""]

    parts += ["## Probability of passing the desk screen", ""]
    parts.append(f"**{p:.0%}**")
    if report.decision is not None:
        src = (
            "decision model over the editor's factor analyses"
            if report.decision.source == "decision-model"
            else "LLM fallback (no decision model configured)"
        )
        conf = (
            f" · confidence {report.decision.confidence:.2f}"
            if report.decision.confidence is not None
            else ""
        )
        parts.append(f"_Verdict by {src}{conf}_")
    parts += [""]
    if report.decision is not None and report.decision.question:
        parts += [f"> Question judged: _{report.decision.question}_", ""]

    parts += ["## Factor assessment (editor's analyses)", ""]
    for f in report.factors:
        parts.append(f"### {f.label}")
        parts += ["", f.analysis or "_(not assessed)_", ""]

    if report.reasoning:
        parts += ["## Editor reasoning", "", report.reasoning, ""]
    if report.risks:
        parts += ["## Key risks", ""]
        parts += [f"- {r}" for r in report.risks] + [""]
    if report.suggestions:
        parts += ["## Suggestions to improve fit", ""]
        parts += [f"- {s}" for s in report.suggestions] + [""]

    if report.journal_profile:
        parts += [
            "## Venue profile (as provided)",
            "",
            "> " + report.journal_profile.replace("\n", "\n> "),
            "",
        ]

    if report.plan_summary:
        parts += _plan_lines(report.plan_summary)
    parts += _usage_table(report.usage, report.model_roles)
    parts += _attribution_lines()
    return "\n".join(parts)


def render_pdf(report: ReviewReport | FitReport, out_path: str | Path) -> Path:
    """PDF export of a review or fit report. Uses weasyprint when available."""
    md = (
        render_markdown(report) if isinstance(report, ReviewReport) else render_fit_markdown(report)
    )
    html = _markdown_to_html(md)
    out = Path(out_path)
    try:
        from weasyprint import HTML

        HTML(string=html, base_url=".").write_pdf(out)
        return out
    except ImportError:
        pass
    return _render_pdf_fallback(report, md, out)


def render_fit_pdf(report: FitReport, out_path: str | Path) -> Path:
    return render_pdf(report, out_path)


def _markdown_to_html(md: str) -> str:
    from markdown_it import MarkdownIt

    body = MarkdownIt("commonmark", {"typographer": True}).enable("table").render(md)
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body {{ font-family: Georgia, serif; margin: 2.2em; line-height: 1.45; color: #1a1a1a; }}
h1 {{ font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: .3em; }}
h2 {{ font-size: 1.25em; border-bottom: 1px solid #bbb; padding-bottom: .2em; margin-top: 1.6em; }}
h3 {{ font-size: 1.05em; margin-top: 1.2em; }}
blockquote {{ border-left: 3px solid #999; margin: .6em 0; padding: .2em 1em;
    color: #444; font-style: italic; }}
code {{ background: #f4f4f4; padding: .1em .3em; border-radius: 3px; font-size: .9em; }}
table {{ border-collapse: collapse; }} th, td {{ border: 1px solid #999; padding: .3em .6em; }}
</style></head><body>{body}</body></html>"""


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _render_pdf_fallback(report, md: str, out: Path) -> Path:
    """Minimal plaintext PDF via reportlab if weasyprint is missing."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4, title=f"Report — {report.paper_title}")
    flow = []
    for line in md.splitlines():
        if not line.strip():
            flow.append(Spacer(1, 6))
            continue
        txt = line.replace("&", "&amp;").replace("<", "&lt;")
        # reportlab paragraphs have no markdown links: [text](url) -> text (url)
        txt = _MD_LINK_RE.sub(r"\1 (\2)", txt)
        if line.startswith("# "):
            flow.append(Paragraph(f"<b><font size=16>{txt[2:]}</font></b>", styles["Normal"]))
        elif line.startswith("## "):
            flow.append(Paragraph(f"<b><font size=13>{txt[3:]}</font></b>", styles["Normal"]))
        elif line.startswith("### "):
            flow.append(Paragraph(f"<b>{txt[4:]}</b>", styles["Normal"]))
        elif line.startswith("> "):
            flow.append(Paragraph(f"<i>{txt[2:]}</i>", styles["Normal"]))
        else:
            flow.append(Paragraph(txt, styles["Normal"]))
    doc.build(flow)
    return out
