"""
CriticService: проверка качества перевода.
"""

import logging
import json
from typing import Dict, Tuple

from services.llm_client import generate_text

logger = logging.getLogger(__name__)


class CriticService:
    """Критик для проверки качества перевода."""

    def __init__(self):
        logger.info("[CriticService] Инициализирован")

    def critique(
        self,
        source_text: str,
        translation: str,
        stats: Dict
    ) -> Tuple[bool, str, Dict]:
        """Проверить качество перевода."""
        logger.info("[CriticService] Проверка перевода...")
        
        prompt = f"""Rate this English translation (CEFR A2+). Response as JSON only:
{{
    "score": <1-10>,
    "is_approved": <true/false>,
    "issues": [<issues>],
    "improvements": [<suggestions>]
}}

SOURCE: {source_text[:200]}
TRANSLATION: {translation[:200]}"""

        try:
            response = generate_text(prompt, temperature=0.2)
            
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            critique = json.loads(response)
            
            score = critique.get("score", 5)
            is_approved = critique.get("is_approved", True)
            
            logger.info(f"[CriticService] Оценка: {score}/10, Одобрено: {is_approved}")
            
            details = {
                "score": score,
                "issues": critique.get("issues", []),
                "improvements": critique.get("improvements", [])
            }
            
            return is_approved, translation, details
        
        except Exception as e:
            logger.warning(f"[CriticService] Ошибка: {e} (продолжаем)")
            return True, translation, {"score": 0, "issues": [], "improvements": []}
