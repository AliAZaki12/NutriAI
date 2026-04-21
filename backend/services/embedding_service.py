import os
import time
import numpy as np
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")

# ── Lazy client — يتعمل أول مرة محتاجينه فقط ───────────────
_client = None


def _get_client():
    global _client
    if _client is None:
        from voyageai import Client
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("❌ VOYAGE_API_KEY is not set in environment")
        _client = Client(api_key=api_key)
    return _client


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts and return numpy array."""
    client     = _get_client()
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        for attempt in range(3):
            try:
                r = client.embed(texts=batch, model=VOYAGE_MODEL)
                embeddings.extend(r.embeddings)
                break
            except Exception as e:
                wait = 2 * (attempt + 1)
                print(f"⚠ Embed retry {attempt + 1}: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError("❌ Embedding failed after retries")

    return np.array(embeddings, dtype=np.float32)


@lru_cache(maxsize=512)
def cached_embed(query: str) -> list[float]:
    """
    Cache repeated queries to avoid redundant Voyage API calls.
    Returns: list[float] — ready for Pinecone query
    """
    arr = embed_texts([query])
    return arr[0].tolist()
