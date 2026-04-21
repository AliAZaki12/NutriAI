import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── مسارات ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "meta", "chunks_meta.csv")
EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "embeddings.npy")

MODEL     = os.getenv("VOYAGE_MODEL", "voyage-3")


def _get_client():
    from voyageai import Client
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("❌ VOYAGE_API_KEY is not set")
    return Client(api_key=api_key)


def embed_chunks() -> np.ndarray:
    """
    Read chunks from CSV → embed in batches → save as .npy
    Returns: numpy array of shape (n_chunks, embedding_dim)
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH} — run ingest_pdf first")

    df    = pd.read_csv(CSV_PATH)
    texts = df["sentence_chunk"].dropna().tolist()

    if not texts:
        raise ValueError("No texts found for embedding")

    client     = _get_client()
    embeddings = []
    BATCH_SIZE = 32

    print(f"🔢 Embedding {len(texts)} chunks with model '{MODEL}'…")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        for attempt in range(3):
            try:
                r = client.embed(texts=batch, model=MODEL)
                embeddings.extend(r.embeddings)
                print(f"   ✓ Batch {i // BATCH_SIZE + 1} / {-(-len(texts) // BATCH_SIZE)}")
                break
            except Exception as e:
                wait = 2 * (attempt + 1)
                print(f"   ⚠ Retry {attempt + 1} after {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Embedding failed after 3 retries at batch {i}")
        time.sleep(0.5)  # rate-limit buffer

    arr = np.array(embeddings, dtype=np.float32)

    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.save(EMB_PATH, arr)

    print(f"✅ Embeddings saved → {EMB_PATH} | shape {arr.shape}")
    return arr
