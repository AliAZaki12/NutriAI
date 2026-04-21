"""
retrieval_service.py
━━━━━━━━━━━━━━━━━━━━
Core RAG orchestration service.
Combines dense + sparse retrieval, MMR deduplication,
context compression, and LLM generation.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from backend.utils.prompt_formatter import prompt_formatter
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.sparse_retriever import SparseRetriever
from backend.retrieval.hybrid import hybrid_search
from backend.llm.llm_openrouter import generate_answer
from backend.services.embedding_service import cached_embed

# ── Global singletons ────────────────────────────────────────
sparse:      SparseRetriever | None = None
dense_index: object | None          = None


# ─────────────────────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────────────────────

def init_retrievers(index, documents: list[dict]) -> None:
    """
    Called once at startup from main.py lifespan.
    Initialises both dense (Pinecone) and sparse (BM25) retrievers.
    """
    global sparse, dense_index

    if not documents:
        raise ValueError(
            "❌ documents list is empty — run the ingestion pipeline first"
        )

    sparse      = SparseRetriever(documents)
    dense_index = index
    print(f"✅ Retrievers initialised with {len(documents)} chunks")


# ─────────────────────────────────────────────────────────────
# Context Compression (Fix #5)
# ─────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "for", "it", "this", "that", "with", "be", "by",
    "has", "have", "had", "but", "not", "from", "as", "by", "its",
}


def _query_tokens(query: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b\w+\b", query)} - _STOP_WORDS


def compress_context(
    query: str, chunks: list[dict], max_sentences: int = 3
) -> list[dict]:
    """
    Keep only the most query-relevant sentences from each chunk.
    Reduces prompt length → lower cost, better focus, fewer hallucinations.
    """
    q_tokens   = _query_tokens(query)
    compressed = []

    for chunk in chunks:
        text      = chunk.get("text", "")
        sentences = re.split(r"(?<=[.!?])\s+", text)

        scored = sorted(
            sentences,
            key=lambda s: len(
                {w.lower() for w in re.findall(r"\b\w+\b", s)} & q_tokens
            ),
            reverse=True,
        )

        best = " ".join(scored[:max_sentences]).strip()
        compressed.append({**chunk, "text": best if best else text})

    return compressed


# ─────────────────────────────────────────────────────────────
# MMR-lite Smart Chunk Selection (Fix #6)
# ─────────────────────────────────────────────────────────────

def _trigrams(text: str) -> frozenset:
    words = text.lower().split()
    if len(words) < 3:
        return frozenset()
    return frozenset(tuple(words[i: i + 3]) for i in range(len(words) - 2))


def smart_chunk_selection(chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    MMR-lite: select high-score chunks while skipping near-duplicates.
    If > 60% of a chunk's trigrams already appear in selected chunks → drop it.
    """
    selected: list[dict]  = []
    seen:     frozenset   = frozenset()

    for chunk in sorted(chunks, key=lambda x: x.get("score", 0), reverse=True):
        tg = _trigrams(chunk.get("text", ""))

        if selected and tg:
            overlap = len(tg & seen) / len(tg)
            if overlap > 0.60:
                continue

        selected.append(chunk)
        seen = seen | tg

        if len(selected) >= top_k:
            break

    return selected


# ─────────────────────────────────────────────────────────────
# Main RAG Service
# ─────────────────────────────────────────────────────────────

def rag_answer_hybrid_service(
    query: str, top_k: int = 5
) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline:
    1. Embed query (cached)
    2. Dense + sparse search in parallel
    3. Hybrid RRF fusion
    4. MMR-lite deduplication
    5. Context compression
    6. LLM generation
    Returns: (answer: str, context_chunks: list[dict])
    """
    if sparse is None or dense_index is None:
        raise RuntimeError(
            "Retrievers not initialised — pipeline must run first"
        )

    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    # ── 1. Embed query ───────────────────────────────────
    q_emb = cached_embed(query)

    # ── 2. Parallel search ──────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_f  = executor.submit(dense_search,  dense_index, q_emb, top_k)
        sparse_f = executor.submit(sparse.search, query,       top_k)
        dense_results  = dense_f.result()
        sparse_results = sparse_f.result()

    # ── 3. Hybrid fusion ────────────────────────────────
    hybrid = hybrid_search(
        dense_results, sparse_results, alpha=0.7, top_k=top_k * 2
    )

    # ── 4. Deduplicate ───────────────────────────────────
    smart = smart_chunk_selection(hybrid, top_k=top_k)

    # ── 5. Compress context ──────────────────────────────
    compressed = compress_context(query, smart, max_sentences=3)

    # ── 6. Generate answer ───────────────────────────────
    prompt = prompt_formatter(query, compressed)
    answer = generate_answer(prompt)

    return answer, compressed
