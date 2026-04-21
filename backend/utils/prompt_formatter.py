def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """
    Build the final prompt sent to the LLM.
    Each chunk is labelled with its source page for traceability.
    """
    context_block = ""
    for item in context_items:
        source = item.get("page", "-")
        text   = item.get("text", "").strip()
        if text:
            context_block += f"[Source — Page {source}]:\n{text}\n\n"

    if not context_block.strip():
        context_block = "No relevant context found in the knowledge base.\n"

    prompt = f"""You are NutriAI, an expert clinical nutritionist and dietitian.
Answer the question STRICTLY based on the provided context from a verified nutrition textbook.

Rules:
- Be detailed, structured, and helpful.
- Use bullet points or numbered lists where appropriate.
- Include specific foods, quantities, and practical advice when available.
- If the answer is NOT in the context, say exactly: "I don't have enough information in my knowledge base to answer this accurately."
- Never invent facts or numbers not present in the context.

CONTEXT FROM NUTRITION KNOWLEDGE BASE:
{context_block.strip()}

QUESTION:
{query.strip()}

DETAILED ANSWER:"""

    return prompt
