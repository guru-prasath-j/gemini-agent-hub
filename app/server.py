"""FastAPI service that fronts the ADK agents. Deployed to Cloud Run.

Endpoints
  POST /v1/chat            one turn, JSON response (reply, routing, tool trace, latency)
  POST /v1/chat/stream     same turn as Server-Sent Events
  POST /v1/documents       upload txt/md/pdf/image -> multimodal ingestion into the index (admin)
  GET  /v1/audit           recent guardrail / tool audit records (admin)
  GET  /healthz            liveness + index size

Identity: in production the service sits behind IAP or Cloud Run IAM and the
role comes from a verified identity claim. For the demo the caller passes it.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from app.agents.root import build_root_agent
from app.bootstrap import bootstrap
from app.config import get_settings
from app.governance import audit as audit_mod
from app.rag.ingest import ingest_bytes
from app.tools import knowledge

Role = Literal["guest", "employee", "admin"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    user_id: str = "demo-user"
    session_id: str | None = None
    role: Role = "employee"


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    handled_by: str
    tool_calls: list[dict[str, Any]]
    latency_ms: int


def _session_service():
    url = get_settings().session_db_url
    if url:
        from google.adk.sessions import DatabaseSessionService

        return DatabaseSessionService(db_url=url)
    return InMemorySessionService()


class AgentService:
    def __init__(self, agent=None) -> None:
        self.settings = get_settings()
        self.sessions = _session_service()
        self.runner = Runner(
            app_name=self.settings.app_name,
            agent=agent or build_root_agent(),
            session_service=self.sessions,
        )

    async def _ensure_session(self, user_id: str, session_id: str | None, role: str) -> str:
        if session_id:
            existing = await self.sessions.get_session(
                app_name=self.settings.app_name, user_id=user_id, session_id=session_id
            )
            if existing:
                return session_id
        s = await self.sessions.create_session(
            app_name=self.settings.app_name,
            user_id=user_id,
            session_id=session_id or uuid.uuid4().hex,
            state={"user:role": role},
        )
        return s.id

    async def events(self, req: ChatRequest) -> AsyncIterator[tuple[str, Any]]:
        sid = await self._ensure_session(req.user_id, req.session_id, req.role)
        yield "session", sid
        msg = types.Content(role="user", parts=[types.Part(text=req.message)])
        async for ev in self.runner.run_async(
            user_id=req.user_id, session_id=sid, new_message=msg, state_delta={"user:role": req.role}
        ):
            for call in ev.get_function_calls() or []:
                yield "tool_call", {"agent": ev.author, "tool": call.name, "args": dict(call.args or {})}
            if ev.content and ev.content.parts:
                text = "".join(p.text or "" for p in ev.content.parts if not getattr(p, "thought", False))
                if text and ev.author != "user":
                    yield "text", {"agent": ev.author, "text": text, "final": ev.is_final_response()}

    async def chat(self, req: ChatRequest) -> ChatResponse:
        t0 = time.perf_counter()
        sid, reply, author, calls = "", "", "concierge", []
        async for kind, data in self.events(req):
            if kind == "session":
                sid = data
            elif kind == "tool_call":
                calls.append(data)
            elif kind == "text" and data["final"]:
                reply, author = data["text"], data["agent"]
        latency = int((time.perf_counter() - t0) * 1000)
        audit_mod.audit("chat_turn", session=sid, role=req.role, handled_by=author,
                        tools=[c["tool"] for c in calls], latency_ms=latency)
        return ChatResponse(session_id=sid, reply=reply, handled_by=author, tool_calls=calls, latency_ms=latency)


def create_app(service: AgentService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            bootstrap()
        app.state.svc = service or AgentService()
        yield

    app = FastAPI(title="Gemini Agent Hub", version="1.0.0", lifespan=lifespan)
    static = Path(__file__).parent / "static"

    def require_admin(role: str | None) -> None:
        if role != "admin":
            raise HTTPException(status_code=403, detail="admin role required")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        idx = knowledge.get_index()
        return {"status": "ok", "chunks": len(idx) if idx else 0, "model": get_settings().model}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        return await app.state.svc.chat(req)

    @app.post("/v1/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        async def sse() -> AsyncIterator[str]:
            async for kind, data in app.state.svc.events(req):
                yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/documents")
    async def upload(file: UploadFile = File(...), x_user_role: str | None = Header(default=None)) -> dict[str, Any]:
        require_admin(x_user_role)
        idx = knowledge.get_index()
        if idx is None:
            raise HTTPException(status_code=503, detail="index not ready")
        data = await file.read()
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="file too large (15 MB max)")
        s = get_settings()
        try:
            chunks = ingest_bytes(file.filename or "upload.txt", data, s.model, multimodal=s.embedder == "gemini")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        idx.add(chunks)
        audit_mod.audit("document_ingested", file=file.filename, chunks=len(chunks))
        return {"status": "ok", "file": file.filename, "chunks_added": len(chunks), "total_chunks": len(idx)}

    @app.get("/v1/audit")
    async def audit_log(limit: int = 50, x_user_role: str | None = Header(default=None)) -> list[dict]:
        require_admin(x_user_role)
        return list(audit_mod.RECENT)[-limit:]

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static / "index.html")

    return app


app = create_app()
