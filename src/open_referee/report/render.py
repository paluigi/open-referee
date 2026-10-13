"""Report renderers: markdown and PDF (weasyprint if available, reportlab-fallback)."""

from __future__ import annotations

from pathlib import Path

from open_referee.report.models import ReviewReport


def render_markdown(report: ReviewReport) -> str:
    parts = [f"# Review report — {report.paper_title}", ""]
    parts.append(f"_Generated {report.generated_at:%Y-%m-%d %H:%M UTC} · run `{report.run_id}`_")
    if report.model_roles:
        roles = ", ".join(f"{role}: `{spec}`" for role, spec in sorted(report.model_roles.items()))
        parts.append(f"_Models — {roles}_")
    parts += ["", "## Paper summary", "", report.paper_summary or "_(none)_", ""]
    parts += ["## Overall feedback", "", report.overall_feedback or "_(none)_", ""]

    parts += [f"## Detailed comments ({len(report.comments)})", ""]
    for i, c in enumerate(report.sorted_comments(), 1):
        sev = f"[severity {c.score:.2f}]"
        sec = f" — _{c.section}_" if c.section else ""
        parts.append(f"### {i}. {c.title} {sev}{sec}")
        if c.quote:
            parts += ["", f"> {c.quote}"]
        parts += ["", c.message, ""]
    if report.validator_notes:
        parts += ["## Validator notes", "", report.validator_notes, ""]
    if report.usage.estimated_cost_usd or report.usage.total_input_tokens:
        u = report.usage
        parts += [
            "## Usage",
            "",
            f"Input tokens: {u.total_input_tokens:,} · Output tokens: {u.total_output_tokens:,} · "
            f"Estimated cost: ${u.estimated_cost_usd:.4f}",
            "",
        ]
    return "\n".join(parts)


def render_pdf(report: ReviewReport, out_path: str | Path) -> Path:
    """PDF export of the review report. Uses weasyprint when available."""
    md = render_markdown(report)
    html = _markdown_to_html(md)
    out = Path(out_path)
    try:
        from weasyprint import HTML

        HTML(string=html, base_url=".").write_pdf(out)
        return out
    except ImportError:
        pass
    return _render_pdf_fallback(report, md, out)


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


def _render_pdf_fallback(report: ReviewReport, md: str, out: Path) -> Path:
    """Minimal plaintext PDF via reportlab if weasyprint is missing."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=A4, title=f"Review report — {report.paper_title}")
    flow = []
    for line in md.splitlines():
        if not line.strip():
            flow.append(Spacer(1, 6))
            continue
        txt = line.replace("&", "&amp;").replace("<", "&lt;")
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
