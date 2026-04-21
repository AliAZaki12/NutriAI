def hybrid_search(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float = 0.7,
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) — fuses dense and sparse rankings.

    alpha: weight for dense scores (1-alpha goes to sparse)
    k: RRF smoothing constant
    top_k: number of results to return
    """
    scores: dict[str, float] = {}

    # Dense — higher weight
    for rank, r in enumerate(dense_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + alpha * (1.0 / (k + rank + 1))

    # Sparse — lower weight
    for rank, r in enumerate(sparse_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0.0) + (1 - alpha) * (1.0 / (k + rank + 1))

    # Merge metadata — prefer dense source
    merged: dict[str, dict] = {}
    for r in dense_results + sparse_results:
        rid = r["id"]
        if rid not in merged:
            merged[rid] = r

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {**merged[rid], "score": score}
        for rid, score in ranked[:top_k]
        if rid in merged
    ]
