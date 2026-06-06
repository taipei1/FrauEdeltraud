import os
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

log = logging.getLogger("critic")


def _llm_invoke(messages: list, model: str, api_key: str, temperature: float, max_tokens: int) -> str:
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens, groq_api_key=api_key)
        return llm.invoke(messages).content
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        return llm.invoke(messages).content


def critique_translation(
    original_text: str,
    translated_text: str,
    user_message: str = "",
) -> dict:
    provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    else:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError(f"API key not set for provider: {provider}")

    system_prompt = f"""You are a strict quality critic for an English tutor bot.

The bot's response MUST follow this structure:
1. CORRECTIONS — Either "All correct!" (if no mistakes) or specific corrections for the learner's last message
2. ANSWER — An informative response to the learner
3. Exactly ONE follow-up question at the end

The learner's last message was: "{user_message}"

Compare the ORIGINAL text (before any modification) with the FINAL text and evaluate:

CRITICAL CHECK (MUST pass):
- Does the final text contain evaluation of the learner's message?
  Look for "All correct!" or corrections of mistakes. If missing, score MUST be < 7.

OTHER CHECKS:
- Is meaning preserved?
- Is it grammatically correct?
- Is it natural-sounding English?
- Does it end with exactly one question?

Respond with ONLY valid JSON:
{{
    "approved": true/false,
    "score": 1-10,
    "issues": ["issue1", "issue2"],
    "fixed_text": "corrected version (only if score < 7)"
}}

RULES:
- Score >= 7: approved, use final text as-is
- Score < 7: not approved, provide fixed_text that fixes issues
- If corrections evaluation is missing, score CANNOT be >= 7
- fixed_text must include all required sections"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"ORIGINAL:\n{original_text}\n\nFINAL:\n{translated_text}"),
    ]

    try:
        raw = _llm_invoke(messages, model_name, api_key, 0.2, 500).strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        approved = result.get("approved", True)
        score = result.get("score", 5)
        issues = result.get("issues", [])
        fixed_text = result.get("fixed_text", translated_text)

        log.info("Critic: score=%d/10, approved=%s, issues=%d", score, approved, len(issues))

        return {
            "approved": approved,
            "score": score,
            "issues": issues,
            "fixed_text": fixed_text if not approved else translated_text,
        }

    except (json.JSONDecodeError, Exception) as e:
        log.warning("Critic parse error: %s — NOT approving", e)
        return {
            "approved": False,
            "score": 3,
            "issues": [f"Critic parse error: {e}"],
            "fixed_text": translated_text,
        }
