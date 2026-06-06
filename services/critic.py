"""
critic.py — Critic agent that checks if the translated text preserves
the original meaning after vocabulary substitution.

If meaning is lost, the critic fixes the translation.
"""
import os
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

log = logging.getLogger("critic")


def critique_translation(
    original_text: str,
    translated_text: str,
) -> dict:
    """Check if translation preserves meaning and fix if needed.
    
    Args:
        original_text: The original English text from the LLM.
        translated_text: The simplified version using known vocabulary.
    
    Returns:
        dict with keys:
            - approved: bool
            - fixed_text: str (the final text, fixed if needed)
            - score: int (1-10)
            - issues: list[str]
    """
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    
    system_prompt = """You are a translation quality critic for an English learner at CEFR A2 level.

Compare the ORIGINAL text with the SIMPLIFIED text and evaluate:
1. Does the simplified version preserve the core MEANING of the original?
2. Does the simplified version contain an evaluation of the learner's last message?
   It must either say "All correct!" or provide corrections. If missing, it is an error.
3. Is the simplified version grammatically correct?
4. Is it natural-sounding English?

Respond with ONLY valid JSON (no markdown, no backticks):
{
    "approved": true/false,
    "score": 1-10,
    "issues": ["issue1", "issue2"],
    "fixed_text": "corrected version if not approved, otherwise same as simplified"
}

RULES:
- If score >= 7, set approved = true
- If score < 7, set approved = false and provide fixed_text that:
  a) Preserves the original meaning better
  b) Includes evaluation of the learner's message ("All correct!" or corrections)
  c) Still uses simple A2-level vocabulary
  d) Keeps sentences simple
- The fixed_text should be ONLY the corrected text, nothing else
- Keep the same style and length as the simplified version"""

    from services.agent import _llm_invoke
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"ORIGINAL:\n{original_text}\n\nSIMPLIFIED:\n{translated_text}"),
    ]
    
    try:
        raw = _llm_invoke(messages, model_name, api_key, 0.2, 500).strip()
        
        # Clean up markdown code blocks if present
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
        
        log.info(
            "Critic: score=%d/10, approved=%s, issues=%d",
            score, approved, len(issues),
        )
        
        return {
            "approved": approved,
            "score": score,
            "issues": issues,
            "fixed_text": fixed_text if not approved else translated_text,
        }
        
    except (json.JSONDecodeError, Exception) as e:
        log.warning("Critic parse error: %s, approving by default", e)
        return {
            "approved": True,
            "score": 5,
            "issues": [f"Parse error: {e}"],
            "fixed_text": translated_text,
        }
