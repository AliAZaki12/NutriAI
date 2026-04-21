# 🥗 NutriAI — Hybrid RAG Nutrition Assistant

---

## Description

**NutriAI** is a production-ready AI nutrition assistant that generates **personalized 4-week meal plans** using a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline.

The system combines **dense vector search (Pinecone)** with **sparse keyword retrieval (BM25)** to ground responses in a verified academic source: *Human Nutrition (OER Hawaii)*.

It delivers accurate, context-aware nutritional guidance while maintaining **low latency, high relevance, and strong factual grounding**.

> Designed as a **scalable AI health assistant**, this architecture can be extended to medical Q&A, fitness coaching, or personalized diet optimization systems.

---

## Features

* **Hybrid Retrieval (Dense + Sparse):**
  Combines semantic search (embeddings) with keyword matching for maximum recall and precision.

* **Personalized Meal Planning:**
  Generates structured 4-week plans based on user profile (age, goals, diet, conditions, allergies).

* **Advanced Ranking Pipeline:**

  * Reciprocal Rank Fusion (RRF)
  * MMR-lite deduplication
  * Context compression (top sentences per chunk)

* **Low-Latency Embedding:**
  Uses cached embeddings (`lru_cache`) to reduce API calls and cost.

* **LLM Fallback Chain:**
  Ensures reliability via multiple models (Llama → Mistral → DeepSeek → Gemma).

* **Single-Origin Deployment:**
  Frontend and backend served from the same FastAPI app → **no CORS complexity**.

* **Production-Ready API:**
  Includes validation, structured errors, health checks, and readiness guards.

---

## Architecture Overview

```
Browser
   │
   ├── GET /        → FastAPI → Frontend (index.html)
   ├── POST /query  → RAG Pipeline → JSON Response
   └── GET /status  → Health Check

────────────────────────────────────────────

RAG Pipeline:

User Query
    │
    ▼
Embedding (Voyage AI - cached)
    │
    ├───────────────┐
    ▼               ▼
Dense Search     Sparse Search
(Pinecone)       (BM25)
    │               │
    └───────┬───────┘
            ▼
   Hybrid RRF Fusion (α=0.7)
            ▼
   MMR-lite Deduplication
            ▼
   Context Compression
            ▼
   Prompt Formatter
            ▼
   LLM (OpenRouter + fallback)
            ▼
        JSON Output
```

---

## Technical Details

| Component           | Implementation                       | Notes / Benefits                |
| ------------------- | ------------------------------------ | ------------------------------- |
| Dense Retrieval     | Pinecone (cosine similarity)         | High semantic understanding     |
| Sparse Retrieval    | BM25 (rank-bm25)                     | Strong keyword matching         |
| Fusion Strategy     | Reciprocal Rank Fusion (k=60, α=0.7) | Balanced ranking                |
| Deduplication       | Trigram overlap (60% threshold)      | Removes redundant chunks        |
| Context Compression | Top-3 sentences per chunk            | Reduces token usage             |
| Embeddings          | Voyage AI (`voyage-3`, 1024-dim)     | High-quality semantic vectors   |
| LLM                 | OpenRouter (Llama 3.2 + fallbacks)   | Reliable response generation    |
| Caching             | `lru_cache(maxsize=512)`             | Reduces repeated embedding cost |
| Deployment          | Replit (single service)              | Simplified architecture         |

---

## Project Structure

```
NutriAI/
│
├── Frontend/
│   └── index.html              # Full SPA (no build step)
│
└── backend/
    ├── main.py                # FastAPI app (startup + lifecycle)
    ├── routes.py              # API endpoints
    ├── requirements.txt
    │
    ├── pipeline/
    │   └── main.py            # End-to-end data pipeline
    │
    ├── ingestion/
    │   ├── ingest_pdf.py
    │   └── utils.py
    │
    ├── embedding/
    │   └── embed_chunks.py
    │
    ├── retrieval/
    │   ├── dense_retriever.py
    │   ├── sparse_retriever.py
    │   └── hybrid.py
    │
    ├── services/
    │   ├── retrieval_service.py
    │   └── embedding_service.py
    │
    ├── llm/
    │   └── llm_openrouter.py
    │
    ├── utils/
    │   └── prompt_formatter.py
    │
    └── vectorstore/
        └── pinecone_client.py
```

---

## Requirements

* **Python:** 3.10+
* **Core Libraries:**

```bash
pip install fastapi uvicorn pinecone-client voyageai rank-bm25 pymupdf pandas numpy requests nltk
```

---

## Configuration

### Environment Variables

```
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_NAMESPACE
PINECONE_CLOUD
PINECONE_REGION

VOYAGE_API_KEY
VOYAGE_MODEL=voyage-3

OPENROUTER_API_KEY
```

---

## Setup & Execution

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Download NLP Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. Run Pipeline (One-Time)

```bash
python -m backend.pipeline.main
```

### 4. Start Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8080
```

---

## API Reference

### `POST /query`

**Request:**

```json
{
  "q": "What foods are high in omega-3?",
  "top_k": 5,
  "profile": { ... }
}
```

**Response:**

```json
{
  "answer": "...",
  "context": [...],
  "chunks_used": 5,
  "status": "ok"
}
```

---

### `GET /status`

```json
{
  "status": "ready",
  "service": "nutriai-rag-api",
  "version": "2.0.0"
}
```

---

## Data Pipeline

### Source

* **Human Nutrition (OER Hawaii)**
* ~500 pages → ~3000+ chunks

### Processing Steps

1. PDF ingestion & cleaning
2. Sentence-based chunking (8 sentences)
3. Embedding generation
4. Pinecone indexing

---

## Retrieval Strategy

| Layer  | Method   | Weight |
| ------ | -------- | ------ |
| Dense  | Pinecone | 0.7    |
| Sparse | BM25     | 0.3    |

---

## LLM Strategy

| Priority | Model          |
| -------- | -------------- |
| Primary  | Llama 3.2 (3B) |
| Fallback | Mistral Small  |
| Fallback | DeepSeek Chat  |
| Fallback | Gemma 3        |

---

## Error Handling

| Code | Scenario       | Handling        |
| ---- | -------------- | --------------- |
| 400  | Invalid input  | Validation      |
| 404  | Unknown route  | Custom handler  |
| 405  | Wrong method   | Method hint     |
| 422  | Schema error   | Pydantic        |
| 500  | Internal error | Trace logging   |
| 503  | Not ready      | Readiness guard |

---

## Design Principles

* **Single Deployment Unit** (Frontend + Backend)
* **Stateless API Layer**
* **Deterministic Retrieval > Generative Guessing**
* **Latency-Optimized Pipeline**
* **Fail-safe LLM execution**

---

## Use Cases

* Personalized nutrition assistants
* Medical RAG systems
* Fitness & diet planning apps
* Health-focused conversational AI

---

## Author

**Eng. Ali Zaki**
AI Engineer — RAG Systems & Applied Intelligence


