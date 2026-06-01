"""
TranslatorService: перевод на английский A2+.

Использует грамматические темы из learned_topics.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from config import Settings
from services.llm_client import generate_text

logger = logging.getLogger(__name__)


class TranslatorService:
    """Переводчик с приоритизацией тем."""

    def __init__(self):
        self.learned_topics = self._load_learned_topics()
        logger.info("[TranslatorService] Инициализирован")

    def _load_learned_topics(self) -> Dict:
        """Загрузить грамматические темы."""
        try:
            path = Path(Settings.LEARNED_TOPICS_PATH)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    topics = data.get("topics", [])
                    logger.info(f"[TranslatorService] Загружено {len(topics)} тем")
                    return data
        except Exception as e:
            logger.warning(f"[TranslatorService] Не удалось загрузить темы: {e}")
        
        return {"topics": []}

    def _build_topics_prompt(self) -> str:
        """Собрать подсказку о темах."""
        topics = self.learned_topics.get("topics", [])
        if not topics:
            return ""
        
        lines = ["LEARNED GRAMMAR TOPICS:"]
        for topic in topics[:10]:
            name = topic.get("name", "")
            desc = topic.get("description", "")
            lines.append(f"- {name}: {desc}")
        
        return "\n".join(lines)

    def _translate_chunks(self, text: str, topics_prompt: str) -> str:
        """Перевести текст чанками."""
        chunk_size = Settings.TRANSLATION_CHUNK_SIZE
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        translated_parts: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            logger.info(f"[TranslatorService] Перевод чанка {idx}/{len(chunks)}")
            prompt = self._build_prompt(chunk, topics_prompt, idx, len(chunks))
            translated = generate_text(prompt)
            translated_parts.append(translated)

        return " ".join(translated_parts)

    def _build_prompt(self, chunk: str, topics_prompt: str, idx: int, total: int) -> str:
        """Собрать промпт."""
        base = f"""Translate to simple English (CEFR A2+).
Output ONLY the translation, no comments.

"""
        
        if topics_prompt:
            base += f"""{topics_prompt}

Try to use these constructions naturally where appropriate.

"""

        base += f"""TEXT (chunk {idx}/{total}):
{chunk}"""

        return base

    def process(self, raw_text: str) -> Tuple[str, Dict]:
        """Перевести текст. Возвращает (translation, stats)."""
        logger.info("[TranslatorService] Начало перевода...")
        
        topics_prompt = self._build_topics_prompt()
        translated = self._translate_chunks(raw_text, topics_prompt)
        
        logger.info(f"[TranslatorService] Перевод завершён: {len(translated)} символов")
        
        stats = {
            "total_vocabulary_words": 0,
            "used_count": 0,
            "vocabulary_percentage": 0,
        }
        
        return translated, stats
