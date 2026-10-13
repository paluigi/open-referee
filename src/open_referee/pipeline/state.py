"""Pipeline models: run state, stage events, resumable artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _runs_dir() -> Path:
    from open_referee.config import DEFAULT_CONFIG_DIR

    return DEFAULT_CONFIG_DIR / "runs"


class Stage(str, Enum):
    INGEST = "ingest"
    TRIAGE = "triage"
    CLAIMS = "claims"
    SURVEY = "survey"
    SCOUT = "scout"
    VERIFY = "verify"
    CHALLENGE = "challenge"
    BIBLIOGRAPHY = "bibliography"
    META = "meta"
    VALIDATE = "validate"
    ASSEMBLE = "assemble"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageEvent(BaseModel):
    run_id: str
    stage: Stage
    status: StageStatus
    message: str = ""
    progress: float | None = None  # 0..1 within stage
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict, exclude=True)


class RunState(BaseModel):
    run_id: str
    paper_path: str
    paper_title: str = ""
    literature_paths: list[str] = Field(default_factory=list)
    stages: dict[str, StageStatus] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)  # stage -> JSON-able artifact
    # events are part of the persisted state so finished runs can replay them
    events: list[StageEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: str = "running"  # running | completed | failed
    error: str | None = None

    def model_post_init(self, __context) -> None:
        # normalize stage dict entries back to StageStatus after JSON round-trip
        self.stages = {k: StageStatus(v) for k, v in self.stages.items()}

    def set_stage(self, stage: Stage, status: StageStatus) -> None:
        self.stages[stage.value] = status
        self.updated_at = datetime.now(UTC)

    def stage_status(self, stage: Stage) -> StageStatus:
        return StageStatus(self.stages.get(stage.value, StageStatus.PENDING))

    def save(self, runs_dir: Path | None = None) -> Path:  # noqa: F811
        d = (runs_dir or _runs_dir()) / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(UTC)
        (d / "state.json").write_text(self.model_dump_json(indent=2))
        return d / "state.json"

    @classmethod
    def load(cls, run_id: str, runs_dir: Path | None = None) -> RunState:  # noqa: F811
        p = (runs_dir or _runs_dir()) / run_id / "state.json"
        return cls.model_validate_json(p.read_text())

    @classmethod
    def list_runs(cls, runs_dir: Path | None = None) -> list[RunState]:
        base = runs_dir or _runs_dir()
        if not base.exists():
            return []
        out = []
        for d in sorted(base.iterdir(), reverse=True):
            if (d / "state.json").exists():
                out.append(cls.model_validate_json((d / "state.json").read_text()))
        return out
