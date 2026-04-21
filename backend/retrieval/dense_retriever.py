import os

NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")


def dense_search(index, query_embedding: list, top_k: int = 5) -> list[dict]:
    """
    Search Pinecone using dense (vector) similarity.
    Returns list of dicts with id, text, page, score.
    """
    try:
        res = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE,
        )
        results = []
        for m in res.matches:
            results.append(
                {
                    "id":    m.metadata.get("id", m.id),
                    "text":  m.metadata.get("text", ""),
                    "page":  m.metadata.get("page", "-"),
                    "score": float(m.score),
                }
            )
        return results
    except Exception as e:
        print(f"⚠ Dense search error: {e}")
        return []
