"""
routes.py
━━━━━━━━━
All API endpoints.
POST /query  — Main RAG query (fixed from GET)
GET  /status — Health check
GET  /       — Serves frontend HTML
"""
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional

from backend.services.retrieval_service import rag_answer_hybrid_service

router = APIRouter()

# ── Base dir for locating index.html ────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    q:           str
    top_k:       int            = 5
    profile:     Optional[dict] = None
    current_day: Optional[dict] = None

    # ── Prevents 422 / 400 from bad inputs ───────────────
    @field_validator("q")
    @classmethod
    def q_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("q must not be empty")
        if len(v) > 2000:
            raise ValueError("q must be under 2000 characters")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_range(cls, v: int) -> int:
        if not (1 <= v <= 20):
            raise ValueError("top_k must be between 1 and 20")
        return v


# ─────────────────────────────────────────────────────────────
# POST /query — Main RAG Endpoint
# ─────────────────────────────────────────────────────────────

@router.post("/query")
async def query_rag(body: QueryRequest, request: Request):
    """
    Accepts a nutrition question, runs hybrid RAG pipeline,
    and returns a grounded answer with source context.

    Errors handled:
    - 503 if system not ready or retrievers uninitialised
    - 400 if query is invalid
    - 500 for unexpected server errors
    """
    # ── 503: system still starting ───────────────────────
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "service_unavailable",
                "message": "System is still initialising — please retry in 15 seconds",
            },
        )

    try:
        answer, context = rag_answer_hybrid_service(
            body.q, top_k=body.top_k
        )
        return {
            "answer":      answer,
            "context":     context,
            "query":       body.q,
            "chunks_used": len(context),
            "status":      "ok",
        }

    # ── 400: bad input (empty query, etc.) ───────────────
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_request", "message": str(e)},
        )

    # ── 503: retrievers not ready ────────────────────────
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "message": str(e)},
        )

    # ── 500: anything unexpected ─────────────────────────
    except Exception as e:
        print(f"[ERROR] /query failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "internal_error",
                "message": "Unexpected server error — check server logs",
            },
        )


# ─────────────────────────────────────────────────────────────
# GET /status — Health Check
# ─────────────────────────────────────────────────────────────

@router.get("/status")
async def status(request: Request):
    """Returns system readiness — poll this after deploy."""
    ready = getattr(request.app.state, "ready", False)
    return {
        "status":  "ready" if ready else "starting",
        "service": "nutriai-rag-api",
        "version": "2.0.0",
    }


# ─────────────────────────────────────────────────────────────
# GET / — Serve Frontend
# ─────────────────────────────────────────────────────────────

@router.get("/")
async def serve_frontend():
    """
    Serve index.html so the whole app runs from a single Replit URL.
    Frontend at: /Frontend/index.html (relative to project root)
    """
    html_path = os.path.join(_ROOT, "Frontend", "index.html")

    if not os.path.exists(html_path):
        raise HTTPException(
            status_code=404,
            detail={
                "error":   "not_found",
                "message": f"index.html not found at {html_path}",
                "hint":    "Ensure the Frontend/index.html file exists in your project root",
            },
        )
    return FileResponse(html_path, media_type="text/html")
