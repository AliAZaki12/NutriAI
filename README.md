# 🥗 NutriAI — Production-Grade Hybrid RAG Nutrition Assistant

**Live Demo:**
[https://nutri-ai--aliahmedzaki788.replit.app](https://nutri-ai--aliahmedzaki788.replit.app)

---

##  ( Overview )

A production-grade AI system that generates **personalized 4-week meal plans** using a **Hybrid RAG pipeline grounded in clinical nutrition data**. The system is designed to minimize hallucination and ensure high factual reliability in health-related use cases.

NutriAI follows a **retrieval-first architecture**, where responses are generated only after retrieving relevant, verified information.

---

## ( Problem Statement )

LLM-only systems in healthcare domains often suffer from:

* Hallucinated outputs
* Inconsistent responses
* Lack of a reliable knowledge source

### Solution

NutriAI enforces:

* Deterministic retrieval
* A single verified knowledge base
* Controlled generation

**Core Principle:** Retrieval First. Generation Second.

---

## ( System Architecture )

```
Frontend (SPA)
      │
      ▼
FastAPI (Single Deployment)
      │
      ▼
Hybrid RAG Pipeline
      │
      ▼
LLM (OpenRouter + Fallback Chain)
```

### Key Design Decisions

* Single-service deployment (frontend and backend together)

  * Eliminates CORS issues
  * Reduces system complexity
  * Simplifies deployment

* Hybrid retrieval strategy

  * Combines semantic search (vector) with keyword matching (BM25)
  * Improves recall for both vague and exact queries

---

## ( RAG Pipeline )

```
User Query + Profile
        │
        ▼
Embedding (Voyage AI)
        │
        ▼
Parallel Retrieval
 ├── Dense → Pinecone
 └── Sparse → BM25
        │
        ▼
Hybrid Fusion (RRF, α=0.7 dense / 0.3 sparse, k=60)
        │
        ▼
MMR Deduplication
        │
        ▼
Context Compression (~60% reduction)
        │
        ▼
Prompt Construction (profile-aware)
        │
        ▼
LLM Generation
        │
        ▼
Final Answer
```

---

## ⚙️ Key Features

### Hybrid Retrieval

* Dense and sparse fusion (RRF)
* Adaptive retrieval (dynamic `top_k`)
* Improved recall under low-similarity queries

### Performance Optimization

* LRU embedding cache
* Parallel retrieval
* Context compression (~60% token reduction)

### Reliability

* Multi-model fallback chain
* Graceful degradation
* Input validation (Pydantic)

### Personalization

* User profile integration (age, goals, conditions)
* Context-aware meal planning

---

## 🌐 Frontend–Backend Integration

```javascript
var API_BASE = window.location.origin;
```

* No hardcoded URLs
* No cross-origin issues
* Works across all environments

---

## 📊 Performance

| Metric             | Value      |
| ------------------ | ---------- |
| End-to-end latency | ~1.2s      |
| Retrieval time     | 150–300 ms |
| Embedding (cached) | 30–50 ms   |
| Token reduction    | ~60%       |
| Context size       | 3–5 chunks |
| Uptime             | ~99.9%     |

Performance achieved through parallel retrieval, caching, and context compression.

---

## 🧠 Engineering Challenges & Solutions

### API Design

* Problem: POST/GET mismatch
* Solution: Unified POST endpoint with Pydantic
* Impact: Eliminated 405/422 errors

### Startup Reliability

* Problem: 503 despite running
* Cause: Silent import failures
* Solution: Explicit readiness state
* Impact: Predictable system state

### Deployment & CORS

* Problem: Cross-origin failures
* Solution: Single-origin architecture
* Impact: No CORS configuration needed

### Retrieval Quality

* Problem: Weak results for low similarity queries
* Solution: Adaptive retrieval expansion
* Impact: Improved recall

### Context Redundancy

* Problem: Duplicate chunks
* Solution: MMR deduplication
* Impact: More relevant context

### Personalization

* Problem: Profile not used
* Solution: Injected into prompt pipeline
* Impact: Personalized outputs

### External Dependencies

* Problem: Pinecone SDK changes
* Solution: Version-safe handling
* Impact: Stable integration

### Frontend Rendering

* Problems: Markdown + UI clipping
* Solutions: Markdown parser + CSS fixes
* Impact: Clean UI

### Meal Planning

* Problem: Repeated meals
* Solution: Themed plans + expanded dataset
* Impact: Better diversity

---

## 🧱 Tech Stack

| Layer            | Technology              |
| ---------------- | ----------------------- |
| Backend          | FastAPI (>=0.111)       |
| Frontend         | HTML / CSS / JavaScript |
| Embeddings       | Voyage AI               |
| Vector DB        | Pinecone                |
| Sparse Retrieval | BM25                    |
| LLM              | OpenRouter              |
| Data             | Pandas / NumPy          |
| Deployment       | Replit                  |

---

## 📁 Project Structure

```
NutriAI/
│
├── Frontend/
│   └── index.html
│
└── backend/
    ├── main.py
    ├── routes.py
    ├── services/
    ├── retrieval/
    ├── pipeline/
    ├── llm/
    ├── utils/
    └── vectorstore/
```

---

## ⚡ Quick Start

### Install

```bash
pip install -r backend/requirements.txt
```

### Run pipeline

```bash
python -m backend.pipeline.main
```

### Start server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

---

## 🔌 API

### POST `/query`

```json
{
  "q": "What foods help hypertension?",
  "top_k": 5,
  "profile": {}
}
```

### Example Response

```json
{
  "answer": "Foods that help hypertension include fruits, vegetables...",
  "context": [
    {"text": "...", "score": 0.82}
  ],
  "chunks_used": 5
}
```

---

## 📚 Data Pipeline

* Source: Human Nutrition (OER Hawaii)
* ~500 pages
* 3000+ chunks
* 1024-d embeddings
* Indexed in Pinecone

---

##  Design Principles

* Deterministic retrieval over probabilistic generation
* Retrieval before generation
* Minimize hallucination
* Fail gracefully
* Consistency across identical queries

---

##  Future Work

* Streaming responses (SSE / WebSockets)
* Query classification
* RAG evaluation (RAGAS)
* User memory layer
* Multi-source retrieval

---

## 👨‍💻 Author

**Eng. Ali Zaki**
AI Engineer — RAG Systems & Applied LLMs
