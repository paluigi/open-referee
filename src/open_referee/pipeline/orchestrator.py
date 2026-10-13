"""The review pipeline orchestrator.

Stages: ingest -> triage -> survey -> scout -> verify (parallel per section)
-> challenge (parallel per section) -> bibliography -> meta -> validate ->
assemble. Every stage persists its artifact so a run resumes where it left
off. LLM role resolution and budget enforcement flow through a RolePool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from open_referee.config import Config, ModelRole
from open_referee.ingestion import Document, Figure, ingest_document
from open_referee.literature import (
    ContextPack,
    CrossrefClient,
    OpenAlexClient,
    PeerReviewScout,
    SearchRouter,
)
from open_referee.pipeline import prompts
from open_referee.pipeline.state import RunState, Stage, StageEvent, StageStatus
from open_referee.providers import (
    BudgetExceeded,
    ChatMessage,
    ChatProvider,
    LLMError,
    ModelSpec,
    UsageLedger,
)
from open_referee.report import ReviewComment, ReviewReport
from open_referee.report.models import UsageSummary
from open_referee.report.render import render_markdown, render_pdf

logger = logging.getLogger(__name__)

MAX_MANUSCRIPT_CHARS = 120_000


def _runs_dir() -> Path:
    from open_referee.config import DEFAULT_CONFIG_DIR

    return DEFAULT_CONFIG_DIR / "runs"


MAX_CITATION_SCANS = 6


@dataclass
class RolePool:
    """Resolved providers per role + shared ledger. Built once per run."""

    cfg: Config
    ledger: UsageLedger
    providers: dict[ModelRole, ChatProvider] = field(default_factory=dict)
    specs: dict[ModelRole, ModelSpec] = field(default_factory=dict)

    @classmethod
    def build(cls, cfg: Config, roles: set[ModelRole]) -> RolePool:
        from open_referee.providers.factory import build_provider

        pool = cls(cfg=cfg, ledger=UsageLedger(cfg.review.max_cost_usd, cfg.llm.pricing))
        for role in roles:
            role_cfg = cfg.llm.role(role)
            pool.providers[role] = build_provider(cfg.llm, role_cfg)
            pcfg = cfg.llm.providers[role_cfg.provider]
            pool.specs[role] = ModelSpec(
                provider_name=role_cfg.provider,
                provider_type=pcfg.type.value,
                model=role_cfg.model,
                base_url=pcfg.base_url,
                api_key=pcfg.resolved_api_key(),
                temperature=role_cfg.temperature,
                max_tokens=role_cfg.max_tokens,
                timeout_s=cfg.llm.request_timeout_s,
                max_retries=cfg.llm.max_retries,
            )
        return pool

    async def complete_json(self, role: ModelRole, system: str, user: str) -> dict:
        """JSON-mode completion with parse-repair retry, ledger check, usage record."""
        provider = self.providers[role]
        spec = self.specs[role]
        est_in = (len(system) + len(user)) // 3
        # rough output estimate: proportional to input, bounded — avoids
        # pre-charging max_tokens/2 on every call and tripping the cap early
        est_out = min(spec.max_tokens, max(1_000, est_in // 4))
        await self.ledger.check_budget(spec, est_in, est_out)
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]
        resp = await provider.complete(messages, json_mode=True)
        await self.ledger.record(spec, role.value, resp)
        from open_referee.providers.adapters import extract_json

        try:
            return extract_json(resp.text)
        except json.JSONDecodeError:
            repair_user = (
                f"{user}\n\nYour previous reply was not valid JSON. Reply again with ONLY the "
                f"required JSON object, no prose, no code fences."
            )
            resp2 = await provider.complete(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=repair_user),
                ],
                json_mode=True,
            )
            await self.ledger.record(spec, role.value, resp2)
            return extract_json(resp2.text)

    async def close(self) -> None:
        for p in self.providers.values():
            await p.close()


class ReviewPipeline:
    def __init__(self, cfg: Config, run_id: str | None = None):
        self.cfg = cfg
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.state: RunState | None = None
        self._subscribers: list[asyncio.Queue[StageEvent]] = []
        self._pool: RolePool | None = None

    # ---------------------------------------------------------------- events --

    def subscribe(self) -> asyncio.Queue[StageEvent]:
        q: asyncio.Queue[StageEvent] = asyncio.Queue()
        self._subscribers.append(q)
        return q

    async def _emit(
        self, stage: Stage, status: StageStatus, message: str = "", progress: float | None = None
    ):
        ev = StageEvent(
            run_id=self.run_id, stage=stage, status=status, message=message, progress=progress
        )
        for q in self._subscribers:
            await q.put(ev)
        if self.state is not None:
            self.state.events.append(ev)

    # ------------------------------------------------------------------- run --

    async def run(
        self, paper_path: str | Path, literature_paths: list[str | Path] | None = None
    ) -> ReviewReport:
        state = RunState(
            run_id=self.run_id,
            paper_path=str(paper_path),
            literature_paths=[str(p) for p in (literature_paths or [])],
        )
        self.state = state
        state.save()
        try:
            pool = await self._ensure_pool()
            # 1. ingest
            await self._emit(Stage.INGEST, StageStatus.RUNNING)
            doc = await asyncio.to_thread(ingest_document, paper_path)
            self._ingested_doc = doc
            # artifacts: theorem environments + markdown tables (PDF ones come
            # from the reader)
            from open_referee.ingestion import extract_tables, extract_theorems

            doc.theorems = await asyncio.to_thread(extract_theorems, doc)
            doc.tables = doc.tables + await asyncio.to_thread(extract_tables, doc)
            lit_docs = [
                await asyncio.to_thread(ingest_document, p)
                for p in (literature_paths or [])[: self.cfg.review.max_user_literature_docs]
            ]
            state.paper_title = doc.title
            state.artifacts["ingest"] = {
                "title": doc.title,
                "n_blocks": len(doc.blocks),
                "n_figures": len(doc.figures),
                "lit_titles": [d.title for d in lit_docs],
            }
            state.set_stage(Stage.INGEST, StageStatus.DONE)
            state.save()
            await self._emit(Stage.INGEST, StageStatus.DONE, doc.title)

            # 2. triage
            triage = await self._stage_or_resume(Stage.TRIAGE, lambda: self._triage(pool, doc))
            queries = triage.get("search_queries", [])[:8]
            key_citations = triage.get("key_citations", [])

            # 2b. claim inventory (atomic checkable claims) + artifact/theorem extraction
            claims = await self._stage_or_resume(
                Stage.CLAIMS, lambda: self._claim_inventory(pool, doc, triage), skip_key="claims"
            )

            # 3+4. literature survey + community scout (concurrent)
            pack = await self._stage_or_resume(
                Stage.SURVEY,
                lambda: self._literature(pool, doc, triage, queries, key_citations, lit_docs),
                skip_key="survey",
            )
            await self._stage_or_resume(
                Stage.SCOUT, lambda: self._scout(pool, doc.title, key_citations, pack)
            )

            # 5. verify: per-claim workers + section lenses (by depth) + artifacts + whole-paper
            candidates = await self._stage_or_resume(
                Stage.VERIFY,
                lambda: self._verify(pool, doc, pack, triage, claims),
                skip_key="candidates",
            )
            # 6. challenge
            validated = await self._stage_or_resume(
                Stage.CHALLENGE,
                lambda: self._challenge(pool, doc, pack, candidates),
                skip_key="validated",
            )
            # 7. bibliography
            bib = await self._stage_or_resume(
                Stage.BIBLIOGRAPHY,
                lambda: self._bibliography(pool, doc, pack),
                skip_key="bibliography",
            )
            # 8. meta
            draft = await self._stage_or_resume(
                Stage.META,
                lambda: self._meta(pool, doc, triage, validated, bib, pack),
                skip_key="draft_review",
            )
            # 9. validate
            final = await self._stage_or_resume(
                Stage.VALIDATE, lambda: self._validate(pool, doc, draft), skip_key="final_review"
            )
            # 10. assemble
            report = await self._assemble(final, doc)
            state.status = "completed"
            state.save()
            await self._emit(Stage.ASSEMBLE, StageStatus.DONE, "report ready")
            return report
        except BudgetExceeded as e:
            # Drain gracefully: assemble a partial report from whatever the
            # pipeline managed to produce before the spend cap tripped.
            if state:
                state.status = "failed"
                state.error = str(e)
                state.save()
            await self._emit(Stage.ASSEMBLE, StageStatus.FAILED, f"budget cap reached: {e}")
            doc: Document | None = getattr(self, "_ingested_doc", None)
            if doc is not None and state is not None:
                try:
                    draft = state.artifacts.get("draft_review")
                    if not isinstance(draft, dict):
                        draft = {}
                    draft.setdefault("paper_summary", "")
                    draft.setdefault("overall_feedback", "")
                    draft.setdefault("comments", [])
                    draft.setdefault(
                        "validator_notes",
                        f"Partial report: the run stopped early ({e}). "
                        "Stages completed: "
                        + ", ".join(
                            k
                            for k, v in (state.stages.items() if state else {})
                            if v.value == "done"
                        ),
                    )
                    report = await self._assemble(draft, doc)
                    return report
                except Exception:
                    logger.exception("partial-report assembly failed")
            raise
        except Exception as e:
            if state:
                state.status = "failed"
                state.error = str(e)
                state.save()
            logger.exception("pipeline failed")
            raise
        finally:
            if self._pool:
                await self._pool.close()

    # ---------------------------------------------------------------- helpers --

    async def _ensure_pool(self) -> RolePool:
        if self._pool is None:
            roles = {ModelRole.STRONG, ModelRole.SMALL}
            try:
                self.cfg.llm.role(ModelRole.VISION)
                roles.add(ModelRole.VISION)
            except KeyError:
                pass
            self._pool = RolePool.build(self.cfg, roles)
        return self._pool

    async def _stage_or_resume(self, stage: Stage, coro_fn, skip_key: str | None = None):
        assert self.state is not None, "run() must be invoked first"
        status = self.state.stage_status(stage)
        if status is StageStatus.DONE and skip_key and skip_key in self.state.artifacts:
            await self._emit(stage, StageStatus.DONE, "resumed from saved state", 1.0)
            return self.state.artifacts[skip_key]
        self.state.set_stage(stage, StageStatus.RUNNING)
        self.state.save()
        await self._emit(stage, StageStatus.RUNNING)
        try:
            result = await coro_fn()
        except (LLMError, BudgetExceeded):
            self.state.set_stage(stage, StageStatus.FAILED)
            self.state.save()
            raise
        self.state.set_stage(stage, StageStatus.DONE)
        if skip_key:
            self.state.artifacts[skip_key] = result
        self.state.save()
        await self._emit(stage, StageStatus.DONE, progress=1.0)
        return result

    def _manuscript_trunc(self, doc: Document) -> str:
        t = doc.full_text()
        return t[:MAX_MANUSCRIPT_CHARS]

    def _pack_for_section(self, pack: ContextPack | None) -> str:
        if not pack:
            return ""
        return pack.render(max_items=15)[:6000]

    # ----------------------------------------------------------------- stages --

    async def _triage(self, pool: RolePool, doc: Document) -> dict:
        user = prompts.TRIAGE_USER.format(title=doc.title, manuscript=self._manuscript_trunc(doc))
        return await pool.complete_json(ModelRole.STRONG, prompts.TRIAGE_SYSTEM, user)

    async def _claim_inventory(self, pool: RolePool, doc: Document, triage: dict) -> list[dict]:
        from open_referee.pipeline.claims import run_claim_inventory

        claims = await run_claim_inventory(pool, doc, triage, self.cfg)
        await self._emit(Stage.CLAIMS, StageStatus.RUNNING, f"{len(claims)} claims catalogued")
        return claims

    async def _literature(self, pool, doc, triage, queries, key_citations, lit_docs) -> ContextPack:
        pack = ContextPack(paper_title=doc.title, field_hint=triage.get("domain"))
        pack.user_docs = [d.title for d in lit_docs]
        items: list = []
        async with (
            OpenAlexClient(self.cfg.openalex) as oa,
            CrossrefClient(self.cfg.crossref) as cr,
            SearchRouter(self.cfg.search) as router,
        ):
            try:
                for q in queries:
                    items.extend(await oa.search_works(q, limit=5))
                    if len(items) >= 30:
                        break
                # user-supplied literature: try to match in OpenAlex for metadata
                for d in lit_docs:
                    hit = await oa.search_works(d.title, limit=1)
                    for h in hit:
                        h.source = "user_pdf"
                        h.why_relevant = "Author-supplied related work"
                        items.append(h)
                # web search for very recent / non-scholarly context
                for q in queries[:3]:
                    for hit in await router.search(q, limit=4):
                        items.append(
                            _hit_to_item(hit, why="Web result (recent or non-scholarly source)")
                        )
            except Exception as e:
                # literature enrichment is best-effort; never fail the run for it
                logger.warning("literature collection degraded: %s", e)
            # dedupe by normalized title
            seen: set[str] = set()
            for it in items:
                key = re.sub(r"\W+", "", it.title.lower())[:80]
                if key and key not in seen:
                    seen.add(key)
                    pack.items.append(it)
            # surveyor refines relevance + missing refs (small role) — still inside
            # the client contexts so Crossref lookups below remain valid
            try:
                survey = await pool.complete_json(
                    ModelRole.SMALL, prompts.SURVEY_SYSTEM, pack.render(max_items=30)
                )
                for sel in survey.get("selected", []):
                    idx = sel.get("index")
                    if isinstance(idx, int) and 1 <= idx <= len(pack.items):
                        pack.items[idx - 1].why_relevant = sel.get("why_relevant")
                for m in survey.get("missing_references", [])[:5]:
                    extra = await cr.lookup(m.get("title", "")) if m.get("title") else None
                    if extra:
                        extra.why_relevant = m.get("why_important")
                        pack.items.append(extra)
                pack.field_hint = (pack.field_hint or "") + (
                    f" — {survey.get('state_of_the_art_notes', '')}"[:400]
                )
            except (LLMError, BudgetExceeded):
                raise
            except Exception as e:
                logger.warning("surveyor stage degraded: %s", e)
        return pack

    async def _scout(self, pool, title: str, key_citations: list, pack: ContextPack) -> None:
        router = SearchRouter(self.cfg.search)
        scout = PeerReviewScout(self.cfg.peer_review_sources, router)
        try:
            found = await scout.scan_title(title)
            for c in key_citations[:MAX_CITATION_SCANS]:
                cit_title = re.sub(r"^.*?:\s*", "", str(c))
                found.extend(await scout.scan_citation(cit_title))
            pack.items.extend(found[:15])
        finally:
            await router.close()

    def _reviewable_sections(self, doc: Document) -> list[tuple[str, str]]:
        """(title, text) for chunks worth verifying; falls back to one big chunk."""
        out: list[tuple[str, str]] = []
        if doc.sections:
            for s in doc.sections:
                text = "\n\n".join(b.text for b in doc.blocks[s.start_block : s.end_block])
                if len(text) > 400:
                    out.append((s.title, text))
        if not out:
            out.append(("Full manuscript", doc.full_text()))
        return out

    def _lenses_for(self, triage: dict) -> list[tuple[str, str]]:
        """Select verifier lenses based on triage: (name, system prompt)."""
        math_density = str(triage.get("mathematical_density", "")).lower()
        stat_content = str(triage.get("statistical_content", "")).lower()
        style = str(triage.get("methodological_style", "")).lower()

        lenses: list[tuple[str, str]] = [("prose", prompts.PROSE_VERIFIER_SYSTEM)]
        if (
            math_density in ("light", "moderate", "heavy")
            or "theorem" in style
            or math_density in ("moderate", "heavy")
        ):
            lenses.append(("math", prompts.MATH_VERIFIER_SYSTEM))
        if stat_content not in ("", "none") or "empirical" in style or "causal" in stat_content:
            lenses.append(("empirical", prompts.EMPIRICAL_VERIFIER_SYSTEM))
        # literature coherence always runs: cheap and central to the review
        lenses.append(("lit", prompts.LIT_VERIFIER_SYSTEM))
        return lenses

    async def _verify(
        self,
        pool: RolePool,
        doc: Document,
        pack: ContextPack | None,
        triage: dict,
        claims: list[dict] | None = None,
    ) -> list[dict]:
        from open_referee.pipeline.claims import verify_artifacts, verify_claims

        preset = self.cfg.review.depth_preset()
        candidates: list[dict] = []

        # per-claim verification workers (the primary engine)
        if claims:
            await self._emit(Stage.VERIFY, StageStatus.RUNNING, "per-claim verification")
            candidates.extend(await verify_claims(self, pool, doc, claims, self.cfg))

        sections = self._reviewable_sections(doc)
        lenses = self._lenses_for(triage) if preset.per_section_lenses else []
        sem = asyncio.Semaphore(self.cfg.review.max_parallel_calls)

        async def run_lens_section(lens_name: str, system: str, title: str, text: str):
            async with sem:
                user = prompts.VERIFIER_USER.format(
                    section_title=f"{title} — {lens_name} lens",
                    context_block=self._pack_for_section(pack),
                    section_text=text[:40_000],
                )
                try:
                    res = await pool.complete_json(ModelRole.SMALL, system, user)
                except LLMError as e:
                    logger.warning("verifier (%s) failed on section %s: %s", lens_name, title, e)
                    return []
                await self._emit(
                    Stage.VERIFY, StageStatus.RUNNING, f"verified [{lens_name}]: {title}"
                )
                return res.get("comments", [])

        tasks = [
            run_lens_section(lens, system, t, txt)
            for (t, txt) in sections
            for (lens, system) in lenses
        ]
        results = await asyncio.gather(*tasks)
        for r in results:
            candidates.extend(r)

        # artifact verification: ALL figures and ALL tables vs prose (vision role)
        if ModelRole.VISION in pool.providers and (doc.figures or doc.tables):
            await self._emit(Stage.VERIFY, StageStatus.RUNNING, "artifact verification")
            art_comments = await verify_artifacts(self, pool, doc, self.cfg)
            candidates.extend(art_comments)

        # whole-paper pass: cross-section coherence (strong model, full text)
        if not preset.whole_paper_passes:
            return candidates
        try:
            await self._emit(Stage.VERIFY, StageStatus.RUNNING, "whole-paper coherence pass")
            wp_user = prompts.WHOLE_PAPER_VERIFIER_USER.format(
                title=doc.title, manuscript=self._manuscript_trunc(doc)
            )
            wp = await pool.complete_json(
                ModelRole.STRONG, prompts.WHOLE_PAPER_VERIFIER_SYSTEM, wp_user
            )
            candidates.extend(wp.get("comments", []))
            await self._emit(
                Stage.VERIFY,
                StageStatus.RUNNING,
                f"whole-paper pass: {len(wp.get('comments', []))} findings",
            )
        except LLMError as e:
            logger.warning("whole-paper verifier failed: %s", e)
        return candidates

    async def _challenge(self, pool, doc, pack, candidates: list[dict]) -> list[dict]:
        sections = self._reviewable_sections(doc)
        sem = asyncio.Semaphore(self.cfg.review.max_parallel_calls)
        validated: list[dict] = []

        async def run_section(title: str, text: str):
            async with sem:
                cands = [
                    c
                    for c in candidates
                    if c.get("paragraph_anchor", "") and _anchor_in(c["paragraph_anchor"], text)
                ]
                user = prompts.CHALLENGER_USER.format(
                    section_title=title,
                    context_block=self._pack_for_section(pack),
                    section_text=text[:40_000],
                    candidates_json=json.dumps(cands, indent=1)[:20_000],
                )
                try:
                    res = await pool.complete_json(
                        ModelRole.STRONG, prompts.CHALLENGER_SYSTEM, user
                    )
                except LLMError as e:
                    logger.warning("challenger failed on %s: %s", title, e)
                    return cands
                kept = [c for c in res.get("validated", []) if c.get("verdict") != "dropped"]
                return kept + res.get("new_comments", [])

        tasks = [run_section(t, txt) for t, txt in sections]
        results = await asyncio.gather(*tasks)
        for r in results:
            validated.extend(r)

        # whole-paper challenge: validate cross-section candidates (those no
        # section challenger claimed) and hunt missed global weaknesses
        try:
            claimed = {
                _norm(c.get("paragraph_anchor", ""))[:120]
                for r in results
                for c in (r if isinstance(r, list) else [])
                if isinstance(c, dict)
            }
            wp_candidates = [
                c
                for c in candidates
                if isinstance(c, dict) and _norm(c.get("paragraph_anchor", ""))[:120] not in claimed
            ]
            await self._emit(Stage.CHALLENGE, StageStatus.RUNNING, "whole-paper challenge")
            wp_user = prompts.WHOLE_PAPER_CHALLENGER_USER.format(
                title=doc.title,
                manuscript=self._manuscript_trunc(doc),
                candidates_json=json.dumps(wp_candidates, indent=1)[:30_000],
            )
            wp = await pool.complete_json(
                ModelRole.STRONG, prompts.WHOLE_PAPER_CHALLENGER_SYSTEM, wp_user
            )
            kept = [c for c in wp.get("validated", []) if c.get("verdict") != "dropped"]
            validated.extend(kept + wp.get("new_comments", []))
            await self._emit(Stage.CHALLENGE, StageStatus.RUNNING, "whole-paper challenge done")
        except LLMError as e:
            logger.warning("whole-paper challenger failed: %s", e)

        # defense-adjudication quality gate (per comment, standard/deep only)
        from open_referee.pipeline.claims import defense_round

        validated = await defense_round(self, pool, doc, validated, self.cfg)
        return validated

    async def _bibliography(self, pool, doc, pack) -> dict:
        empty = {
            "entries": [],
            "in_text_issues": [],
            "missing_key_references": [],
        }
        if not doc.references_text:
            return empty
        intext = _extract_intext_citations(doc)
        user = prompts.BIBLIOGRAPHY_USER.format(
            references=doc.references_text[:30_000],
            intext=intext[:15_000],
            context=pack.render(max_items=25),
        )
        try:
            result = await pool.complete_json(ModelRole.SMALL, prompts.BIBLIOGRAPHY_SYSTEM, user)
            result.setdefault("in_text_issues", [])
            return result
        except LLMError as e:
            logger.warning("bibliography audit failed: %s", e)
            return empty

    async def _meta(self, pool, doc, triage, validated, bib, pack) -> dict:
        bib_summary = {
            "entries": bib.get("entries", []),
            "in_text_issues": bib.get("in_text_issues", []),
        }
        user = prompts.META_USER.format(
            title=doc.title,
            triage_json=json.dumps(triage)[:4000],
            comments_json=json.dumps(validated, indent=1)[:60_000],
            bib_summary=json.dumps(bib_summary)[:8000],
            sota_notes=(pack.field_hint or "")[:1500],
        )
        return await pool.complete_json(ModelRole.STRONG, prompts.META_SYSTEM, user)

    async def _validate(self, pool, doc, draft: dict) -> dict:
        user = prompts.VALIDATOR_USER.format(
            title=doc.title,
            manuscript=self._manuscript_trunc(doc),
            comments_json=json.dumps(draft.get("comments", []), indent=1)[:60_000],
            overall=draft.get("overall_feedback", "")[:20_000],
        )
        result = await pool.complete_json(ModelRole.STRONG, prompts.VALIDATOR_SYSTEM, user)
        # merge any fields the validator omitted from the draft
        result.setdefault("paper_summary", draft.get("paper_summary", ""))
        result.setdefault("overall_feedback", draft.get("overall_feedback", ""))
        result.setdefault("comments", draft.get("comments", []))
        return result

    async def _assemble(self, final: dict, doc: Document) -> ReviewReport:
        comments: list[ReviewComment] = []
        for i, c in enumerate(final.get("comments", [])):
            try:
                score = max(0.0, min(1.0, float(c.get("score", 0.3))))
            except (TypeError, ValueError):
                score = 0.3
            anchor = c.get("paragraph_anchor") or ""
            quote = c.get("quote") or ""
            block_id, confidence = _resolve_anchor(doc, anchor, quote)
            comments.append(
                ReviewComment(
                    id=f"c{i:03d}",
                    title=c.get("title") or "Untitled comment",
                    section=_section_of_block(doc, block_id),
                    block_id=block_id,
                    paragraph_anchor=anchor or None,
                    quote=quote or None,
                    message=c.get("message") or "",
                    score=score,
                    category=c.get("category") or "general",
                    anchor_confidence=confidence,
                )
            )
        u = self._pool.ledger if self._pool else None
        by_role = u.by_role() if u else {}
        usage = UsageSummary(
            tokens_by_role=by_role,
            estimated_cost_usd=round(u.total_cost(), 4) if u else 0.0,
            total_input_tokens=int(u.total_tokens()[0]) if u else 0,
            total_output_tokens=int(u.total_tokens()[1]) if u else 0,
        )
        report = ReviewReport(
            run_id=self.run_id,
            paper_title=doc.title,
            paper_summary=final.get("paper_summary", ""),
            overall_feedback=final.get("overall_feedback", ""),
            comments=comments,
            usage=usage,
            model_roles={
                r.value: f"{p.provider_name}/{p.model}"
                for r, p in (self._pool.specs if self._pool else {}).items()
            },
            validator_notes=final.get("validator_notes"),
        )
        # persist report + exports into the run dir
        rdir = _runs_dir() / self.run_id
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "report.json").write_text(report.model_dump_json(indent=2))
        (rdir / "report.md").write_text(render_markdown(report))
        try:
            render_pdf(report, rdir / "report.pdf")
        except Exception as e:  # PDF export must never fail the run
            logger.warning("PDF export failed: %s", e)
        return report


# ------------------------------------------------------------------ anchor --


def _extract_intext_citations(doc: Document, max_citations: int = 60) -> str:
    """Extract in-text citation mentions with claim context for the audit.

    Matches the common author-year styles: (Smith, 2001), Smith (2001),
    (Smith and Jones, 2001; Doe, 2019), [1], [Smith2001]. For each match,
    keeps the surrounding sentence as the claim context.
    """
    text = doc.full_text()
    patterns = [
        # Author (1999) / Author and Other (1999) — runs FIRST so it claims the
        # full span before the bare-paren pattern can reduce it to "(1999)"
        r"([A-Z][A-Za-z'’\-]+(?:\s+(?:and|&|et al\.?)\s+[A-Z][A-Za-z'’\-]+)"
        r"{0,2}\s*\(\d{4}[a-z]?\))",
        # numeric [1] / [1, 2] / [1-3]
        r"(\[[0-9][0-9,\s\-–]*\])",
        # (Author, 1999) / (Author and Other, 1999; Third, 2020)
        r"\(([^()]{0,80}?\d{4}[a-z]?[^()]{0,80}?)\)",
        # natbib \citep-style leftovers: Smith2001 / Jones2020
        r"\b([A-Z][a-z]{2,}\d{2,4})\b",
    ]
    hits: list[tuple[str, str]] = []
    seen_spans: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < be and e > bs for bs, be in seen_spans)

    for pat in patterns:
        for m in re.finditer(pat, text):
            if overlaps(m.start(), m.end()):
                continue
            # must contain either a year or a bracket number to count as a citation
            cite = m.group(0)
            if not (re.search(r"\d{4}", cite) or cite.startswith("[")):
                continue
            sent_lo = max(0, text.rfind(". ", 0, m.start()) + 1)
            sent_hi = text.find(". ", m.end())
            if sent_hi == -1:
                sent_hi = min(len(text), m.end() + 200)
            claim = " ".join(text[sent_lo:sent_hi].split())[:400]
            hits.append((cite, claim))
            seen_spans.append((m.start(), m.end()))
            if len(hits) >= max_citations:
                break
        if len(hits) >= max_citations:
            break
    if not hits:
        return "(no in-text citations detected)"
    return "\n".join(f"- {cite} :: {claim}" for cite, claim in hits)


def _figure_context(doc: Document, fig: Figure, window: int = 30) -> str:
    """Text around a figure: prefers the caption block, else blocks near the
    figure's page (approximated by proportional position in the block list)."""
    if fig.caption_block_id:
        for i, b in enumerate(doc.blocks):
            if b.id == fig.caption_block_id:
                lo, hi = max(0, i - window // 2), min(len(doc.blocks), i + window // 2)
                return "\n".join(b.text for b in doc.blocks[lo:hi])
    if len(doc.blocks) <= window:
        return doc.full_text()
    # pages are 1-based; approximate block index proportionally
    if fig.page:
        frac = (fig.page - 1) / max(1, fig.page)  # crude; falls back to middle
        center = min(len(doc.blocks) - window, int(frac * (len(doc.blocks) - window)))
        center = max(0, center)
    else:
        center = len(doc.blocks) // 2
    return "\n".join(b.text for b in doc.blocks[center : center + window])


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def _anchor_in(anchor: str, text: str) -> bool:
    return _norm(anchor)[:120] in _norm(text)


def _resolve_anchor(doc: Document, anchor: str, quote: str) -> tuple[str | None, float]:
    """Fuzzy-resolve a comment to a block. Returns (block_id, confidence)."""
    if anchor:
        hits = doc.find_blocks(anchor)
        if len(hits) == 1:
            return hits[0].id, 1.0
        if hits:
            return hits[0].id, 0.7
    if quote:
        hits = doc.find_blocks(quote)
        if hits:
            return hits[0].id, 0.6
    if anchor:
        # token-overlap fallback
        a_tokens = set(_norm(anchor).split())
        best, best_j = None, 0.0
        for b in doc.blocks:
            tokens = set(_norm(b.text).split())
            if not tokens:
                continue
            j = len(a_tokens & tokens) / max(1, len(a_tokens | tokens))
            if j > best_j:
                best, best_j = b, j
        if best is not None and best_j > 0.35:
            return best.id, round(best_j, 2)
    return None, 0.0


def _section_of_block(doc: Document, block_id: str | None) -> str | None:
    if not block_id:
        return None
    for b in doc.blocks:
        if b.id == block_id:
            return b.section_path[0] if b.section_path else None
    return None


def _hit_to_item(hit, why: str):
    from open_referee.literature.context import LiteratureItem

    return LiteratureItem(
        source="web", title=hit.title, url=hit.url, abstract=hit.snippet, why_relevant=why
    )
