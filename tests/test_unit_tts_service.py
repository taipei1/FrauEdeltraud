"""Unit-тесты TTSService: проверяет исправленный код синтеза речи.

Тесты требуют интернет для edge-tts, но НЕ требуют API-ключей.
"""
import os

import pytest

from services.tts_service import TTSService

pytestmark = [pytest.mark.unit, pytest.mark.integration]


def test_synthesize_returns_word_timings(short_text, temp_output_dir):
    """Главный регрессионный тест: word_timings не пустой."""
    tts = TTSService()
    audio_path, duration, words = tts.synthesize(short_text)

    assert os.path.exists(audio_path)
    assert os.path.getsize(audio_path) > 0
    assert duration > 0
    assert len(words) > 0, (
        "BUG: edge-tts не отдал WordBoundary. "
        "Проверь, что TTSService передаёт boundary='WordBoundary'."
    )

    for w in words:
        assert "word" in w and "start" in w and "end" in w
        assert w["end"] > w["start"]


def test_synthesize_long_text_has_many_words(sample_text, temp_output_dir):
    """Длинный текст даёт много слов с монотонно растущими таймингами."""
    tts = TTSService()
    _, duration, words = tts.synthesize(sample_text)

    assert len(words) >= 10
    starts = [w["start"] for w in words]
    assert starts == sorted(starts), "тайминги должны быть упорядочены"
    assert words[-1]["end"] <= duration + 0.5


def test_synthesize_overwrites_existing_file(short_text, temp_output_dir):
    """Повторный вызов перезаписывает старый файл, а не падает."""
    tts = TTSService()
    tts.synthesize(short_text)
    tts.synthesize(short_text)
    _, _, words = tts.synthesize(short_text)
    assert len(words) > 0
