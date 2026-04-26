def _format_profile(profile: dict | None) -> str:
    if not profile:
        return ""
    fields = [
        ("Age",        profile.get("age")),
        ("Gender",     profile.get("gender")),
        ("Height",     profile.get("height")),
        ("Weight",     profile.get("weight")),
        ("Activity",   profile.get("activity")),
        ("Goal",       profile.get("goal")),
        ("Diet",       profile.get("diet")),
        ("Allergies",  profile.get("allergies")),
        ("Conditions", profile.get("conditions")),
        ("Notes",      profile.get("notes")),
    ]
    lines = [f"- {k}: {v}" for k, v in fields if v not in (None, "", [], {})]
    return "USER PROFILE:\n" + "\n".join(lines) + "\n\n" if lines else ""


def _format_current_day(day: dict | None) -> str:
    if not day:
        return ""
    parts = [f"CURRENT DAY ({day.get('day_name', 'today')}):"]
    if day.get("total_kcal"):
        parts.append(f"- Total: {day['total_kcal']} kcal")
    for slot in ("breakfast", "lunch", "snack", "dinner"):
        meal = day.get(slot)
        if meal and meal.get("name"):
            macros = ", ".join(
                f"{k}: {meal[k]}"
                for k in ("kcal", "protein", "carbs", "fat")
                if meal.get(k)
            )
            parts.append(f"- {slot.title()}: {meal['name']} ({macros})")
    return "\n".join(parts) + "\n\n" if len(parts) > 1 else ""


def prompt_formatter(
    query: str,
    context_items: list[dict],
    profile: dict | None = None,
    current_day: dict | None = None,
    lang: str = "auto",
) -> str:
    """
    Build the final prompt sent to the LLM.
    Includes retrieved textbook context plus the user's profile and current day
    when available, so personal questions can be answered concretely.
    """
    context_block = ""
    for item in context_items:
        source = item.get("page", "-")
        text   = item.get("text", "").strip()
        if text:
            context_block += f"[Textbook — Page {source}]:\n{text}\n\n"

    has_context = bool(context_block.strip())
    if not has_context:
        context_block = "(No directly matching passages — use general nutrition expertise.)"

    profile_block = _format_profile(profile)
    day_block     = _format_current_day(current_day)

    lang_norm = (lang or "auto").lower()
    if lang_norm == "ar":
        language_block = (
            "LANGUAGE (HARD RULE — OVERRIDES THE QUESTION'S LANGUAGE):\n"
            "- Reply ENTIRELY in fluent natural Arabic, even if the question is in English.\n"
            "- Use Arabic headers, Arabic bullet points, Arabic phrasing.\n"
            "- Use Arabic / Egyptian food names (فول، كشري، ملوخية، فراخ، سمك، عدس...).\n"
            "- Do NOT include any English sentences. English food names in parentheses are OK only when needed.\n"
            "- Write as a native Arabic-speaking Egyptian nutritionist."
        )
    elif lang_norm == "en":
        language_block = (
            "LANGUAGE (HARD RULE — OVERRIDES THE QUESTION'S LANGUAGE):\n"
            "- Reply ENTIRELY in clear English, even if the question is in Arabic.\n"
            "- Use English headers, English bullet points, English phrasing.\n"
            "- Do NOT include Arabic sentences. Arabic food names in parentheses are OK."
        )
    else:
        language_block = (
            "LANGUAGE (CRITICAL):\n"
            "- Detect the language of the USER QUESTION and reply in the SAME language.\n"
            "- If the question is in Arabic (any dialect — Egyptian, Levantine, MSA),\n"
            "  reply ENTIRELY in fluent natural Arabic: Arabic headers, Arabic bullets,\n"
            "  Arabic food names (فول، كشري، ملوخية، فراخ، سمك، عدس...). Do NOT translate\n"
            "  word-for-word from English — write as a native Arabic-speaking nutritionist.\n"
            "- If the question is in English, reply in English.\n"
            "- Never mix languages within one reply (unless the user mixed them first)."
        )

    prompt = f"""You are NutriAI, a friendly clinical nutritionist and dietitian.

Style:
- Open with a short helpful sentence (NEVER start with "I don't know" or any apology).
- Then use clear sections with **bold headers** and concise bullet points.
- Mention specific foods, portions, and macros (kcal / protein / carbs / fat) when relevant.
- Keep total length under ~250 words unless the user asks for detail.
- If the user provided a profile or current day, reference it directly
  (e.g. their goal, calorie target, or specific meals) instead of generic advice.
- Prefer the textbook context when it directly answers the question.
- If the textbook context is missing or unrelated, answer from your general
  nutrition expertise — but stay strictly within nutrition / food / dietetics.
- Do not invent scientific citations or page numbers.

{language_block}

{profile_block}{day_block}TEXTBOOK CONTEXT:
{context_block.strip()}

USER QUESTION:
{query.strip()}

ANSWER:"""

    return prompt
