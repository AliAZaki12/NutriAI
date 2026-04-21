"""
pipeline/main.py
━━━━━━━━━━━━━━━━
One-time setup pipeline:
  PDF → Chunks → Embeddings → Pinecone
Run once before starting the API server.
"""
import os
import sys

# ── مسارات ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "meta",       "chunks_meta.csv")
EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "embeddings.npy")
PDF_PATH = os.path.join(BASE_DIR, "data", "raw",        "nutrition.pdf")

PDF_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"


def run_pipeline(force: bool = False) -> None:
    print("\n" + "═" * 55)
    print("  🚀  NUTRIAI RAG PIPELINE")
    print("═" * 55 + "\n")

    # ── Step 1: Ingestion ────────────────────────────────
    if force or not os.path.exists(CSV_PATH):
        print("📄 [1/3] Running PDF ingestion…")
        from backend.ingestion.ingest_pdf import ingest_pdf
        ingest_pdf(pdf_path=PDF_PATH, download_url=PDF_URL, chunk_size=8)
    else:
        print("⏩ [1/3] Ingestion skipped — CSV already exists")

    # ── Step 2: Embedding ────────────────────────────────
    if force or not os.path.exists(EMB_PATH):
        print("\n🔢 [2/3] Running embedding…")
        from backend.embedding.embed_chunks import embed_chunks
        embed_chunks()
    else:
        print("⏩ [2/3] Embedding skipped — embeddings.npy already exists")

    # ── Step 3: Pinecone Upsert ──────────────────────────
    print("\n📤 [3/3] Checking Pinecone…")
    from backend.vectorstore.pinecone_client import upsert_embeddings
    upsert_embeddings(force=force)

    print("\n" + "═" * 55)
    print("  ✅  PIPELINE COMPLETE — Ready to start the API")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run_pipeline(force=force_flag)
