"""FastAPI application: JSON REST API + a small web UI."""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config, prompts
from app.rag import RAGService
from app.security import User, audit, authenticate, require_admin
from app.vectorstore import can_see

app = FastAPI(title="Enterprise Knowledge Assistant", version="1.0.0")
service = RAGService()
STATIC = Path(__file__).resolve().parent.parent / "static"


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    prompt_version: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "mode": config.LLM_MODE}


@app.get("/prompts")
def list_prompts(user: User = Depends(authenticate)):
    return [
        {"name": p.name, "description": p.description, "top_k": p.top_k, "min_score": p.min_score}
        for p in prompts.load_all().values()
    ]


@app.post("/query")
def query(body: QueryIn, user: User = Depends(authenticate)):
    try:
        result = service.ask(body.question, role=user.role, prompt_version=body.prompt_version)
    except KeyError as e:
        raise HTTPException(400, str(e).strip("\"'"))
    audit(user, "query", question=body.question, prompt=result["prompt_version"],
          sources=[s["source"] for s in result["sources"]], abstained=result["abstained"])
    return result


@app.get("/documents")
def documents(user: User = Depends(authenticate)):
    """List only the documents the caller is allowed to see."""
    service.ensure_index()
    docs = {}
    for ch in service.store.chunks:
        if can_see(user.role, ch["allowed_roles"]):
            d = docs.setdefault(ch["source"], {"source": ch["source"], "title": ch["title"],
                                               "department": ch["department"], "chunks": 0})
            d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source"])


@app.post("/ingest")
def ingest(user: User = Depends(require_admin)):
    report = service.ingest()
    audit(user, "ingest", **report)
    return report


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")
