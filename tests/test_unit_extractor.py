"""Unit-тесты ExtractorService (без сети/БД): только текстовый вход."""
import pytest

from services.extractor import ExtractorService

pytestmark = pytest.mark.unit


def test_text_input_russian():
    text = "Привет! Как дела?"
    text_out, lang, vp, ap = ExtractorService().process(text)
    assert text_out == text
    assert lang == "ru"
    assert vp == "" and ap == ""


def test_text_input_english():
    text = "Hello! How are you?"
    text_out, lang, vp, ap = ExtractorService().process(text)
    assert text_out == text
    assert lang == "en"


def test_text_input_mixed_defaults_to_cyrillic():
    """Если есть хоть одна кириллица — язык ru."""
    text_out, lang, _, _ = ExtractorService().process("Hello, привет")
    assert lang == "ru"


def test_clean_subtitle_text_strips_timestamps():
    raw = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello world

00:00:03.500 --> 00:00:05.000
How are you
"""
    out = ExtractorService._clean_subtitle_text(raw)
    assert "Hello world" in out
    assert "How are you" in out
    assert "-->" not in out
    assert "00:00" not in out
