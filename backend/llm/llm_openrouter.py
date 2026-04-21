import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Primary model
PRIMARY_MODEL = "meta-llama/llama-3.2-3b-instruct:free"

# Fallback models — tried in order if primary fails
FALLBACK_MODELS = [
    "mistralai/mistral-small-2603",
    "deepseek/deepseek-chat",
    "google/gemma-3-4b-it:free",
]


def call_openrouter(model: str, prompt: str, api_key: str) -> str | None:
    """
    Single model call to OpenRouter.
    Returns the response text or None on failure.
    """
    headers = {
        "Authorization":  f"Bearer {api_key}",
        "HTTP-Referer":   "https://nutriai.replit.app",
        "X-Title":        "NutriAI RAG",
        "Content-Type":   "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are NutriAI, a clinical nutrition expert. "
                    "Answer ONLY using the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  800,
    }
    try:
        r = requests.post(BASE_URL, headers=headers, json=payload, timeout=45)
        if r.status_code != 200:
            print(f"[LLM ERROR] Model: {model} | Status: {r.status_code}")
            print(f"[LLM ERROR] Response: {r.text[:300]}")
            return None
        content = r.json()["choices"][0]["message"]["content"]
        return content.strip() if content else None
    except Exception as e:
        print(f"[LLM EXCEPTION] Model: {model} → {type(e).__name__}: {e}")
        return None


def generate_answer(prompt: str) -> str:
    """
    Generate answer with automatic fallback chain.
    Tries primary model first, then fallbacks.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("❌ OPENROUTER_API_KEY is not set in environment")

    # Try primary
    response = call_openrouter(PRIMARY_MODEL, prompt, api_key)
    if response:
        return response

    # Try fallbacks
    for model in FALLBACK_MODELS:
        print(f"🔁 Falling back to: {model}")
        response = call_openrouter(model, prompt, api_key)
        if response:
            return response

    return (
        "I'm sorry, I'm unable to generate a response right now. "
        "All language models are temporarily unavailable. Please try again later."
    )
