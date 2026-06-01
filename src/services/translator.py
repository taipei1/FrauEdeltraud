"""Translator service: перевод на английский A2+ с приоритетом словаря."""
import logging
from typing import Dict, List, Tuple

from config import Settings
from services.db_service import get_db_service
from services.llm_client import generate_text

logger = logging.getLogger(__name__)


def _word_in_text(word: str, text: str) -> bool:
    """Проверить, что слово из словаря встречается в тексте как отдельное слово."""
    if not word:
        return False
    word_l = word.lower()
    text_l = text.lower()
    return (
        f" {word_l} " in f" {text_l} "
        or text_l.startswith(f"{word_l} ")
        or text_l.endswith(f" {word_l}")
    )


class TranslatorService:
    """Переводчик с приоритетом словаря ученика."""

    def __init__(self):
        self.db = get_db_service()
        logger.info("[OK] Translator инициализирован")

    def _build_vocabulary_prompt(self) -> Tuple[str, List[Dict]]:
        """Собрать словарь для промпта и вернуть сырой список слов."""
        hot_words = self.db.get_hot_vocabulary(limit=Settings.MAX_VOCABULARY_WORDS)
        if not hot_words:
            return "", []

        lines = []
        for w in hot_words[:50]:
            word = w.get("word", "")
            translation = w.get("translation", "")
            if word and translation:
                lines.append(f'- "{word}" = {translation}')

        prompt = "\n".join(lines)
        logger.info(f"[OK] Словарь для промпта: {len(lines)} слов")
        return prompt, hot_words

    def _translate_chunks(self, text: str, vocab_prompt: str) -> str:
        """Разбить текст на чанки и перевести каждый."""
        chunk_size = Settings.TRANSLATION_CHUNK_SIZE
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        translated_parts: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            logger.info(f"[TRANSLATOR] Перевод чанка {idx}/{len(chunks)}")
            prompt = self._build_prompt(chunk, vocab_prompt, idx, len(chunks))
            translated_parts.append(generate_text(prompt))
            logger.info(f"[OK] Чанк {idx} переведён")

        return " ".join(translated_parts)

    def _build_prompt(self, chunk: str, vocab_prompt: str, idx: int, total: int) -> str:
        if vocab_prompt:
            return f"""Translate the following text into simple English (CEFR A2+).

VOCABULARY PRIORITY — use these words from the learner's dictionary whenever they fit (max coverage, do NOT distort meaning):
{vocab_prompt}

RULES:
1. If a vocabulary word fits EXACTLY (100%) — use it.
2. If it fits 80%+ — use it.
3. If it fits < 60% — do not use, pick a simpler alternative.
4. Keep sentences short and natural for A2+.
5. Output ONLY the translation, no commentary.

TEXT (chunk {idx}/{total}):
{chunk}"""
        return f"""Translate the following text into simple English (CEFR A2+).
Output ONLY the translation, no commentary.

TEXT (chunk {idx}/{total}):
{chunk}"""

    def _analyze_usage(
        self, translated_text: str, hot_words: List[Dict]
    ) -> Tuple[Dict[str, str], Dict]:
        """Сопоставить использованные слова словаря с переводом."""
        usage: Dict[str, str] = {}
        used_words: List[str] = []

        for w in hot_words:
            word = w.get("word", "")
            translation = w.get("translation", "")
            if word and _word_in_text(word, translated_text):
                usage[word] = translation
                used_words.append(word)

        total = len(hot_words)
        used = len(used_words)
        pct = round((used / total * 100) if total else 0, 1)

        stats = {
            "total_vocabulary_words": total,
            "used_words": used_words,
            "used_count": used,
            "vocabulary_percentage": pct,
        }
        logger.info(f"[OK] Словарь: {used}/{total} ({pct}%)")
        return usage, stats

    def process(self, raw_text: str) -> Tuple[str, Dict[str, str], Dict]:
        """Перевести текст и вернуть (translation, vocabulary_usage, stats)."""
        logger.info("[TRANSLATOR] Перевод текста...")

        vocab_prompt, hot_words = self._build_vocabulary_prompt()
        translated = self._translate_chunks(raw_text, vocab_prompt)
        usage, stats = self._analyze_usage(translated, hot_words)

        logger.info("[OK] Перевод завершён")
        return translated, usage, stats
