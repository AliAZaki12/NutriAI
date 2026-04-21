import re
import nltk
from nltk.tokenize import sent_tokenize

# ── تحميل NLTK data بشكل آمن ──────────────────────────────
def _ensure_nltk():
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

_ensure_nltk()


def text_formatter(text: str) -> str:
    """Clean raw PDF text — remove newlines and extra spaces."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_sentences_spacy(text: str) -> list[str]:
    """Sentence-split using NLTK (no spaCy model needed)."""
    try:
        sentences = sent_tokenize(text)
    except Exception:
        # Fallback: split on punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def split_list(lst: list, size: int) -> list[list]:
    return [lst[i: i + size] for i in range(0, len(lst), size)]


def create_sentence_chunks(
    sentences: list[str], page_number: int, chunk_size: int = 8
) -> list[dict]:
    chunks = []
    for group in split_list(sentences, chunk_size):
        joined = " ".join(group).strip()
        joined = re.sub(r"\.([A-Z])", r". \1", joined)
        if not joined:
            continue
        chunks.append(
            {
                "id": f"page-{page_number}-chunk-{len(chunks)}",
                "page_number": page_number,
                "sentence_chunk": joined,
                "chunk_token_count": len(joined) / 4,
            }
        )
    return chunks


def filter_chunks(
    chunks: list[dict], min_token_length: int = 30
) -> list[dict]:
    return [c for c in chunks if c.get("chunk_token_count", 0) > min_token_length]
