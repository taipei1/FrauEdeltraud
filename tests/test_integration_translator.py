"""Интеграционные тесты: перевод через LLM.

Требуют GOOGLE_API_KEY (или GROQ_API_KEY как fallback).
"""
import pytest

pytestmark = pytest.mark.integration


def test_translator_smoke(sample_text):
    """Базовый тест: перевод возвращает непустую строку."""
    from services.translator import TranslatorService

    tr = TranslatorService()
    translated, usage, stats = tr.process("Привет! Как дела?")
    assert translated and len(translated) > 3
    assert isinstance(usage, dict)
    assert isinstance(stats, dict)
    assert "used_count" in stats and "total_vocabulary_words" in stats
