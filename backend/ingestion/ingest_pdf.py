import os
import requests
import fitz  # PyMuPDF
import pandas as pd

from backend.ingestion.utils import (
    text_formatter,
    split_sentences_spacy,
    create_sentence_chunks,
    filter_chunks,
)

# ── مسارات موحَّدة ──────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(BASE_DIR, "data", "parquet")
META_DIR    = os.path.join(BASE_DIR, "data", "meta")


def download_pdf(url: str, pdf_path: str) -> None:
    """Download PDF from URL if it doesn't exist locally."""
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    if os.path.exists(pdf_path):
        print(f"⏩ PDF already exists: {pdf_path}")
        return
    print(f"⬇️  Downloading PDF from {url}…")
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with open(pdf_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ PDF downloaded successfully")
    except Exception as e:
        raise RuntimeError(f"PDF download failed: {e}")


def load_chunks(parquet_path: str) -> list[dict]:
    """Load chunks from an existing parquet file."""
    df = pd.read_parquet(parquet_path)
    return df.to_dict(orient="records")


def ingest_pdf(
    pdf_path: str,
    download_url: str = None,
    chunk_size: int = 8,
) -> list[dict]:
    """
    Full ingestion pipeline:
    1. Download PDF (if URL provided and file missing)
    2. Extract text page by page
    3. Split into sentence chunks
    4. Filter short chunks
    5. Save to parquet + CSV
    Returns: list of chunk dicts
    """
    # ── 1. Download ──────────────────────────────────────
    if download_url and not os.path.exists(pdf_path):
        download_pdf(download_url, pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── 2. Extract text ──────────────────────────────────
    print("📄 Reading PDF and extracting text…")
    doc   = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        raw_text  = page.get_text()
        if not raw_text.strip():
            continue
        text      = text_formatter(raw_text)
        sentences = split_sentences_spacy(text)
        chunks    = create_sentence_chunks(sentences, i, chunk_size)
        pages.extend(chunks)

    # ── 3. Filter ────────────────────────────────────────
    chunks = filter_chunks(pages)
    if not chunks:
        raise ValueError("No valid chunks extracted from PDF")

    # ── 4. Save ─────────────────────────────────────────
    df = pd.DataFrame(chunks)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(META_DIR,    exist_ok=True)

    df.to_parquet(os.path.join(PARQUET_DIR, "chunks.parquet"), index=False)
    df.to_csv(os.path.join(META_DIR, "chunks_meta.csv"),       index=False)

    print(f"✅ Ingestion complete: {len(chunks)} chunks saved")
    return chunks
