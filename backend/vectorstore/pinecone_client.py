import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nutrition-rag-project")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE",  "default")
CLOUD      = os.getenv("PINECONE_CLOUD",      "aws")
REGION     = os.getenv("PINECONE_REGION",     "eu-west-1")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings", "embeddings.npy")
CSV_PATH = os.path.join(BASE_DIR, "data", "meta", "chunks_meta.csv")

# ── Lazy Pinecone client ─────────────────────────────────────
_pc = None


def _get_pc():
    global _pc
    if _pc is None:
        from pinecone import Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("❌ PINECONE_API_KEY is not set in environment")
        _pc = Pinecone(api_key=api_key)
    return _pc


def get_index():
    """Return the Pinecone index object."""
    return _get_pc().Index(INDEX_NAME)


def _get_vector_count(index) -> int:
    """Safely get total vector count — compatible with Pinecone SDK v3+."""
    try:
        stats = index.describe_index_stats()
        # SDK v3 returns an object, not a dict
        if hasattr(stats, "total_vector_count"):
            return stats.total_vector_count or 0
        if isinstance(stats, dict):
            return stats.get("total_vector_count", 0)
    except Exception as e:
        print(f"⚠ Could not fetch index stats: {e}")
    return 0


def upsert_embeddings(force: bool = False) -> None:
    """
    Upload embeddings to Pinecone.
    Skips if vectors already exist (unless force=True).
    """
    if not os.path.exists(EMB_PATH):
        raise FileNotFoundError(f"Embeddings not found: {EMB_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    from pinecone import ServerlessSpec

    pc         = _get_pc()
    embeddings = np.load(EMB_PATH)
    df         = pd.read_csv(CSV_PATH)

    if len(embeddings) != len(df):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(df)} CSV rows"
        )

    dim = embeddings.shape[1]

    # ── Create index if it doesn't exist ─────────────────
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"📦 Creating Pinecone index '{INDEX_NAME}' (dim={dim})…")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        print(f"✅ Index created")

    index = pc.Index(INDEX_NAME)

    # ── Skip if already populated ────────────────────────
    if not force:
        count = _get_vector_count(index)
        if count >= len(df):
            print(f"⏩ Skipping upsert — {count} vectors already in Pinecone")
            return

    # ── Batch upsert ─────────────────────────────────────
    BATCH = 100
    print(f"📤 Uploading {len(df)} vectors to Pinecone…")

    for i in range(0, len(df), BATCH):
        batch = []
        for j in range(min(BATCH, len(df) - i)):
            idx = i + j
            row = df.iloc[idx]
            batch.append(
                {
                    "id":     str(row["id"]),
                    "values": embeddings[idx].tolist(),
                    "metadata": {
                        "id":   str(row["id"]),
                        "text": str(row["sentence_chunk"]),
                        "page": int(row["page_number"]),
                    },
                }
            )
        index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"   ✓ Uploaded batch {i // BATCH + 1} / {-(-len(df) // BATCH)}")

    final_count = _get_vector_count(index)
    print(f"✅ Upsert complete. Total vectors in Pinecone: {final_count}")
