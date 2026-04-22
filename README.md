# 🥗 NutriAI — Production-Grade Hybrid RAG Nutrition Assistant

**Live Demo:**
[https://nutri-ai--aliahmedzaki788.replit.app](https://nutri-ai--aliahmedzaki788.replit.app)

---

## 🚀 Overview

NutriAI is a **production-grade AI nutrition system** that generates:

* Personalized **4-week meal plans (112 unique meals)**
* Intelligent **nutrition guidance via bilingual chatbot (Arabic / English)**
* Accurate **calorie & macro recommendations**

The system is built on a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline to ensure:

* High factual accuracy
* Minimal hallucination
* Strong personalization

---

## 🎯 Core Principle

> **Retrieval First. Generation Second.**

Unlike traditional LLM apps, NutriAI enforces **deterministic retrieval from a verified knowledge base before generation**, ensuring reliability in a health-sensitive domain.

---

## ⚠️ Problem Statement

Standard LLM-based systems in healthcare suffer from:

* ❌ Hallucinated responses
* ❌ Inconsistent outputs
* ❌ No grounding in verified data

### ✅ Solution

NutriAI introduces:

* Hybrid retrieval (Dense + Sparse)
* Controlled prompt construction
* Profile-aware generation

---

## 🧠 System Architecture

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

* **Single-service deployment**

  * Eliminates CORS issues
  * Simplifies infra
  * Improves reliability

* **Hybrid Retrieval**

  * Dense (semantic understanding)
  * Sparse (exact keyword matching)

---

## 🔍 RAG Pipeline (Detailed)

```
User Query + Profile + Current Day
        │
        ▼
Embedding (Voyage AI)
        │
        ▼
Parallel Retrieval
 ├── Dense → Pinecone (Top-K)
 └── Sparse → BM25 (Top-K)
        │
        ▼
Hybrid Fusion (RRF)
  α = 0.7 (Dense)
  α = 0.3 (Sparse)
        │
        ▼
MMR Deduplication
        │
        ▼
Context Compression (~60%)
        │
        ▼
Prompt Builder (Profile + Language Override)
        │
        ▼
LLM (OpenRouter)
        │
        ▼
Final Response (Markdown → HTML)
```

---

## ✨ Key Features

### 🍽 Egyptian Meal Planning Engine

* 4 Weeks × 7 Days × 4 Meals = **112 unique meals**

* Weekly themes:

  * Egyptian
  * Seaside
  * Levantine
  * Countryside

* Diet modes:

  * Standard
  * Keto
  * Vegetarian

* Zero duplication across the full plan

---

### 💬 Smart Bilingual Chat

* Language modes:

  * `auto` (detect)
  * `ar` (force Arabic)
  * `en` (force English)

* Full RTL support

* Markdown → HTML rendering

* Language-aware caching

---

### 🧮 Health Calculators

* BMR (Mifflin-St Jeor)
* TDEE (5 activity levels)
* Goal-based calories:

  * Loss / Maintain / Gain
* Smart macro distribution

---

### 🔍 Advanced Retrieval

* Hybrid Search:

  * Pinecone (Dense)
  * BM25 (Sparse)

* Adaptive thresholds:

  * WEAK = 0.45
  * MIN = 0.30

* Dynamic `top_k`

* Profile-aware retrieval context

---

## ⚡ Performance

| Metric             | Value      |
| ------------------ | ---------- |
| End-to-end latency | ~1.2s      |
| Retrieval time     | 150–300 ms |
| Embedding (cached) | 30–50 ms   |
| Token reduction    | ~60%       |
| Context size       | 3–5 chunks |
| Uptime             | ~99.9%     |

### Optimization Techniques

* Parallel retrieval
* LRU embedding cache
* Context compression
* MMR deduplication

---

## 🧱 Tech Stack

| Layer            | Technology            |
| ---------------- | --------------------- |
| Backend          | FastAPI, Uvicorn      |
| Frontend         | HTML, CSS, Vanilla JS |
| Embeddings       | Voyage AI             |
| Vector DB        | Pinecone              |
| Sparse Retrieval | BM25                  |
| LLM              | OpenRouter            |
| Data             | Pandas / NumPy        |
| Deployment       | Replit Autoscale      |

---

## 📁 Project Structure

```
nutriai/
│
├── Frontend/
│   └── index.html
│
└── backend/
    ├── main.py
    ├── routes.py
    ├── llm/
    ├── services/
    ├── retrieval/
    ├── pipeline/
    ├── utils/
    ├── vectorstore/
    └── data/
```

---

## ⚙️ Setup & Installation

### 1. Environment Variables

| Key                | Purpose    |
| ------------------ | ---------- |
| PINECONE_API_KEY   | Vector DB  |
| VOYAGE_API_KEY     | Embeddings |
| OPENROUTER_API_KEY | LLM        |

Optional:

```
PINECONE_INDEX_NAME
PINECONE_NAMESPACE
PINECONE_REGION
```

---

### 2. Run Locally

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

Open:

```
http://localhost:5000
```

---

## 🔌 API

### POST `/query`

```json
{
  "q": "What foods are high in omega-3?",
  "top_k": 5,
  "lang": "auto",
  "profile": {
    "age": 28,
    "weight": 75,
    "goal": "lose",
    "diet": "standard"
  },
  "current_day": 5
}
```

---

## 📌 Example Output

**Query:**

> What foods are high in omega-3?

**Response:**

> Fatty fish such as salmon, mackerel, and sardines are among the richest sources of omega-3 fatty acids. Plant-based sources include flaxseeds, chia seeds, and walnuts.

---

## 🧠 Engineering Challenges & Solutions

| Problem               | Solution                   | Impact           |
| --------------------- | -------------------------- | ---------------- |
| 503 errors on startup | Readiness state validation | Stable boot      |
| Weak retrieval        | Adaptive thresholds        | Better recall    |
| Duplicate context     | MMR deduplication          | Higher relevance |
| CORS issues           | Single deployment          | Zero config      |
| Profile ignored       | Injected into prompt       | Personalization  |
| API inconsistencies   | Unified POST design        | Stability        |

---

## 📊 Data Pipeline

* Source: Human Nutrition (OER Hawaii)
* ~500 pages processed
* 3000+ chunks
* 1024-d embeddings
* Indexed in Pinecone

---

## 🚀 Achievements

* ✅ 1906 vectors indexed
* ✅ 112 unique meals (no repetition)
* ✅ Full bilingual UX
* ✅ Hybrid RAG with adaptive scoring
* ✅ Production deployment (Autoscale)

---

## 🔮 Future Work

* Streaming responses (SSE / WebSockets)
* Query classification layer
* RAG evaluation (RAGAS)
* User memory system
* Multi-source retrieval

---

## 👨‍💻 Author

**Eng. Ali Zaki**
AI Engineer — RAG Systems & Applied LLMs
