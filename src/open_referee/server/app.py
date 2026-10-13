"""FastAPI application: PWA pages, upload, run management, SSE, settings."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from open_referee.config import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    Config,
    load_config,
    save_config,
)
from open_referee.pipeline import RunState, Stage, StageEvent, StageStatus
from open_referee.pipeline.orchestrator import ReviewPipeline
from open_referee.report.models import ReviewReport

SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent
TEMPLATES_DIR = SRC_DIR / "templates"
STATIC_DIR = SRC_DIR / "static"
# packaged fallback (hatch force-include puts static/templates inside the package)
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
    STATIC_DIR = Path(__file__).resolve().parent / "static"

UPLOAD_DIR = DEFAULT_CONFIG_DIR / "uploads"


def _safe_filename(name: str | None) -> str:
    """Sanitize an upload filename: basename only, no traversal, no weird chars."""
    if not name:
        return "upload"
    name = Path(name).name  # strips any directory components
    cleaned = re.sub(r"[^\w.\- ]+", "_", name).strip("._ ")
    return cleaned or "upload"


def create_app() -> FastAPI:
    application = FastAPI(title="Open Referee", version="0.1.0")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Single source of truth for the config path; tests may override it.
    application.state.config_path = DEFAULT_CONFIG_PATH

    def get_cfg(request: Request) -> Config:
        return load_config(request.app.state.config_path)

    # in-memory live pipelines: run_id -> ReviewPipeline
    application.state.pipelines = {}

    # Completed/failed background tasks, bounded to avoid unbounded growth
    _finished_tasks: dict[str, asyncio.Task] = {}
    _MAX_FINISHED_TASKS = 50

    # ------------------------------------------------------------- pages --

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request, cfg: Config = Depends(get_cfg)):
        runs = RunState.list_runs()
        return templates.TemplateResponse(request, "index.html", {"runs": runs, "cfg": cfg})

    @application.get("/review/{run_id}", response_class=HTMLResponse)
    async def review_page(request: Request, run_id: str, cfg: Config = Depends(get_cfg)):
        try:
            state = RunState.load(run_id)
        except FileNotFoundError:
            raise HTTPException(404, "run not found") from None
        report = _load_report(run_id)
        return templates.TemplateResponse(
            request, "review.html", {"state": state, "report": report, "run_id": run_id}
        )

    @application.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request, cfg: Config = Depends(get_cfg)):
        return templates.TemplateResponse(request, "settings.html", {"cfg": cfg, "saved": False})

    # ---------------------------------------------------------------- api --

    @application.post("/api/reviews")
    async def start_review(
        paper: UploadFile = File(...),
        literature: list[UploadFile] = File(default=[]),
        cfg: Config = Depends(get_cfg),
    ):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:12]
        paper_path = UPLOAD_DIR / f"{run_id}_{_safe_filename(paper.filename)}"
        paper_path.write_bytes(await paper.read())
        lit_paths: list[Path] = []
        for lf in literature[: cfg.review.max_user_literature_docs]:
            if lf.filename:
                p = UPLOAD_DIR / f"{run_id}_lit_{_safe_filename(lf.filename)}"
                p.write_bytes(await lf.read())
                lit_paths.append(p)
        pipeline = ReviewPipeline(cfg, run_id=run_id)
        application.state.pipelines[run_id] = pipeline

        def _cleanup(_task: asyncio.Task) -> None:
            # keep a bounded number of finished task refs so exceptions are
            # retrievable for debugging without leaking forever
            application.state.pipelines.pop(run_id + ":task", None)
            _finished_tasks[run_id] = _task
            while len(_finished_tasks) > _MAX_FINISHED_TASKS:
                _finished_tasks.pop(next(iter(_finished_tasks)))

        task = asyncio.create_task(_run_pipeline(pipeline, paper_path, lit_paths, run_id))
        task.add_done_callback(_cleanup)
        application.state.pipelines[run_id + ":task"] = task  # keep a ref while running
        return {"run_id": run_id}

    async def _run_pipeline(pipeline, paper_path, lit_paths, run_id):
        try:
            await pipeline.run(paper_path, lit_paths)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("run %s failed", run_id)

    @application.get("/api/reviews/{run_id}/events")
    async def events(run_id: str):
        pipeline = application.state.pipelines.get(run_id)

        async def sse(ev: dict | StageEvent) -> str:
            payload = ev if isinstance(ev, dict) else json.loads(ev.model_dump_json())
            return f"data: {json.dumps(payload)}\n\n"

        async def gen():
            # 1) Finished / unknown runs: replay persisted events, ending with
            #    the run's REAL terminal status (never a synthetic "done").
            if pipeline is None:
                try:
                    state = RunState.load(run_id)
                except FileNotFoundError:
                    state = None
                if state is not None:
                    for ev in state.events:
                        yield await sse(ev)
                    terminal = {
                        "run_id": run_id,
                        "stage": "assemble",
                        "status": "done" if state.status == "completed" else "failed",
                        "message": state.error or "",
                    }
                    yield await sse(terminal)
                else:
                    yield await sse(
                        {
                            "run_id": run_id,
                            "stage": "assemble",
                            "status": "failed",
                            "message": "run not found",
                        }
                    )
                return
            # 2) Live run: replay events already emitted, then stream new ones.
            #    Covers the race where the run finishes between page load and
            #    SSE connect (no hang: state.events already holds the terminal
            #    event because _emit appends before returning).
            for ev in list(pipeline.state.events if pipeline.state else []):
                yield await sse(ev)
            if pipeline.state is not None and pipeline.state.status in ("completed", "failed"):
                terminal = {
                    "run_id": run_id,
                    "stage": "assemble",
                    "status": "done" if pipeline.state.status == "completed" else "failed",
                    "message": pipeline.state.error or "",
                }
                yield await sse(terminal)
                return
            q = pipeline.subscribe()
            while True:
                ev = await q.get()
                yield await sse(ev)
                if ev.stage is Stage.ASSEMBLE and ev.status in (
                    StageStatus.DONE,
                    StageStatus.FAILED,
                ):
                    break

        return StreamingResponse(gen(), media_type="text/event-stream")

    @application.get("/api/reviews/{run_id}/report")
    async def report_json(run_id: str):
        report = _load_report(run_id)
        if report is None:
            raise HTTPException(404, "report not ready")
        if isinstance(report, ReviewReport):
            report = report.model_dump(mode="json")
        return JSONResponse(report)

    @application.get("/api/reviews/{run_id}/report.pdf")
    async def report_pdf(run_id: str):
        p = DEFAULT_CONFIG_DIR / "runs" / run_id / "report.pdf"
        if not p.exists():
            raise HTTPException(404, "pdf not ready")
        return FileResponse(p, media_type="application/pdf", filename=f"review-{run_id}.pdf")

    @application.get("/api/reviews/{run_id}/report.md")
    async def report_md(run_id: str):
        p = DEFAULT_CONFIG_DIR / "runs" / run_id / "report.md"
        if not p.exists():
            raise HTTPException(404, "markdown not ready")
        return FileResponse(p, media_type="text/markdown", filename=f"review-{run_id}.md")

    # ---------------------------------------------------------- settings --

    @application.post("/api/settings")
    async def save_settings(request: Request):
        form = await request.form()
        cfg = _config_from_form(form, load_config(request.app.state.config_path))
        save_config(cfg, request.app.state.config_path)
        return templates.TemplateResponse(request, "frag_settings_saved.html", {"ok": True})

    @application.get("/api/settings/test-llm")
    async def test_llm(role: str = "small", cfg: Config = Depends(get_cfg)):
        """Round-trip a tiny completion to verify role/provider wiring."""
        from open_referee.config import ModelRole
        from open_referee.pipeline.orchestrator import RolePool

        try:
            pool = RolePool.build(cfg, {ModelRole(role)})
            try:
                res = await pool.complete_json(
                    ModelRole(role), "Reply with JSON only.", '{"ok": true}'
                )
                return {"ok": True, "reply": res}
            finally:
                await pool.close()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @application.get("/sw.js", include_in_schema=False)
    async def service_worker():
        """Serve the service worker at root scope so it controls all pages
        (a worker under /static/ would only control /static/*)."""
        p = STATIC_DIR / "sw.js"
        return FileResponse(
            p,
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/"},
        )

    @application.get("/manifest.webmanifest")
    async def manifest():
        return JSONResponse(
            {
                "name": "Open Referee",
                "short_name": "Referee",
                "start_url": "/",
                "display": "standalone",
                "background_color": "#101418",
                "theme_color": "#101418",
                "icons": [{"src": "/static/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
            }
        )

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    return application


def _load_report(run_id: str) -> dict | ReviewReport | None:
    p = DEFAULT_CONFIG_DIR / "runs" / run_id / "report.json"
    if not p.exists():
        return None
    return ReviewReport.model_validate_json(p.read_text())


def _config_from_form(form, base: Config) -> Config:
    """Map the settings form onto the config model. Only env-var NAMES are saved."""

    def s(key: str, default=None):
        v = form.get(key)
        return v if v not in (None, "") else default

    def f(key: str, default):
        try:
            return float(form.get(key, default))
        except (TypeError, ValueError):
            return default

    def i(key: str, default):
        try:
            return int(form.get(key, default))
        except (TypeError, ValueError):
            return default

    from open_referee.config import LLMConfig, ModelRole, ProviderConfig, ProviderType, RoleConfig

    # start from the existing providers so hand-edited custom entries survive a
    # settings-form save (the form only manages the five well-known names)
    providers: dict[str, ProviderConfig] = dict(base.llm.providers)
    for name in ("openrouter", "anthropic", "gemini", "ollama", "vllm"):
        ptype = s(f"provider_{name}_type")
        if not ptype:
            providers.pop(name, None)  # explicitly disabled in the form
            continue
        providers[name] = ProviderConfig(
            type=ProviderType(ptype),
            base_url=s(f"provider_{name}_base_url"),
            api_key_env=s(f"provider_{name}_api_key_env"),
        )
    roles: dict[ModelRole, RoleConfig] = {}
    for role in ("strong", "small", "vision"):
        prov, model = s(f"role_{role}_provider"), s(f"role_{role}_model")
        if prov and model:
            roles[ModelRole(role)] = RoleConfig(provider=prov, model=model)
    base.llm = LLMConfig(
        providers=providers,
        roles=roles,
        pricing=base.llm.pricing,
        request_timeout_s=base.llm.request_timeout_s,
        max_retries=base.llm.max_retries,
    )
    depth = s("review_depth", base.review.depth)
    if depth in ("fast", "standard", "deep"):
        base.review.depth = depth
    base.review.max_cost_usd = f("review_max_cost_usd", base.review.max_cost_usd)
    base.review.max_parallel_calls = i("review_max_parallel_calls", base.review.max_parallel_calls)
    base.openalex.api_key_env = (
        s("openalex_api_key_env", base.openalex.api_key_env) or "OPENALEX_API_KEY"
    )
    base.openalex.email = s("openalex_email")
    base.crossref.email = s("crossref_email")
    order = s("search_order", "tavily,tinyfish,brave") or "tavily,tinyfish,brave"
    base.search.order = [x.strip() for x in order.split(",") if x.strip()]
    for eng in ("tavily", "tinyfish", "brave"):
        eng_cfg = getattr(base.search, eng)
        eng_cfg.api_key_env = s(f"search_{eng}_api_key_env", eng_cfg.api_key_env)
        eng_cfg.enabled = bool(form.get(f"search_{eng}_enabled"))
    base.peer_review_sources.pubpeer = bool(form.get("prs_pubpeer"))
    base.peer_review_sources.openreview = bool(form.get("prs_openreview"))
    base.peer_review_sources.prereview = bool(form.get("prs_prereview"))
    return base
