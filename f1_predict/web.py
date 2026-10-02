from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from f1_predict.live.service import RaceIntelligenceService

STATIC_DIR = Path(__file__).resolve().parent / "web_static"

app = FastAPI(
    title="F1 Predict — Race Intelligence",
    version="0.1.0",
    description="Historical ML priors combined with live/replay F1 race state.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
service = RaceIntelligenceService()


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "f1-predict"}


@app.get("/api/snapshot")
def snapshot() -> dict:
    result = service.snapshot()
    return {
        "mode": result.mode,
        "source": result.source,
        "session": result.session,
        "updated_at": result.updated_at,
        "drivers": result.drivers,
        "note": result.note,
    }
