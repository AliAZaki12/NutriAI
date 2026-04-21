"""
main.py
━━━━━━━
FastAPI application entry point.

Startup sequence:
  1. Run pipeline (only if data missing)
  2. Connect to Pinecone
  3. Load chunks CSV
  4. Initialise dense + sparse retrievers
  5. Mark app as ready → serve requests
"""
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "meta",       "chunks_meta.csv")
EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "embeddings.npy")


# ─────────────────────────────────────────────────────────────
# Lifespan — runs on startup and shutdown
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "═" * 55)
    print("  🚀  NUTRIAI RAG API — STARTING UP")
    print("═" * 55)

    app.state.ready = False

    try:
        # ── 1. Run pipeline if data is missing ───────────
        if not os.path.exists(CSV_PATH) or not os.path.exists(EMB_PATH):
            print("\n📄 Data not found — running pipeline automatically…")
            from backend.pipeline.main import run_pipeline
            run_pipeline(force=False)
        else:
            print("\n⏩ Data found — skipping pipeline")

        # ── 2. Connect to Pinecone ───────────────────────
        print("🔌 Connecting to Pinecone…")
        from backend.vectorstore.pinecone_client import get_index, _get_vector_count
        index = get_index()

        # ── 3. Load chunks ───────────────────────────────
        print(f"📦 Loading chunks from CSV…")
        df     = pd.read_csv(CSV_PATH)
        chunks = df.to_dict(orient="records")

        if not chunks:
            raise ValueError("CSV is empty — re-run pipeline with --force flag")

        print(f"   ✓ {len(chunks)} chunks loaded")

        # ── 4. Init retrievers ───────────────────────────
        from backend.services.retrieval_service import init_retrievers
        init_retrievers(index, chunks)

        # ── 5. Pinecone health check ─────────────────────
        total = _get_vector_count(index)
        print(f"   ✓ Pinecone vectors: {total}")

        if total == 0:
            print("   ⚠ WARNING: Pinecone index is empty!")
            print("   ⚠ Run: python -m backend.pipeline.main --force")

        # ── Ready ────────────────────────────────────────
        app.state.ready = True
        print("\n" + "═" * 55)
        print("  ✅  SYSTEM READY — Accepting requests")
        print("═" * 55 + "\n")

    except Exception as e:
        print(f"\n❌ Startup error: {type(e).__name__}: {e}")
        print("⚠  Running in DEGRADED MODE — /query will return 503")
        print("   Fix the error above and restart.\n")
        app.state.ready = False

    yield  # ← app runs here

    print("\n🛑 Shutting down NutriAI RAG API…")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "NutriAI RAG API",
    description = "Hybrid RAG-powered nutrition assistant",
    version     = "2.0.0",
    lifespan    = lifespan,
)


# ─────────────────────────────────────────────────────────────
# CORS Middleware
# Allows the frontend (same or different domain) to call the API
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # In production: replace with your Replit URL
    allow_credentials = False,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["Content-Type", "Authorization"],
)


# ─────────────────────────────────────────────────────────────
# Global Error Handlers
# ─────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error":   "not_found",
            "path":    str(request.url.path),
            "message": "This endpoint does not exist",
            "hint":    "Available endpoints: GET /, GET /status, POST /query",
        },
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={
            "error":   "method_not_allowed",
            "method":  request.method,
            "path":    str(request.url.path),
            "message": "Wrong HTTP method — POST /query, GET /status, GET /",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error":   "internal_server_error",
            "message": "An unexpected error occurred — check server logs",
        },
    )


# ─────────────────────────────────────────────────────────────
# Serve static assets (CSS / JS / images if added later)
# ─────────────────────────────────────────────────────────────

_FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "Frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=_FRONTEND_DIR),
        name="static",
    )

# ─────────────────────────────────────────────────────────────
# Include API routes
# ─────────────────────────────────────────────────────────────

from backend.routes import router
app.include_router(router)
