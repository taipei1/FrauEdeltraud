"""Unit-тесты TranslatorService._analyze_usage (без сети/БД)."""
import pytest

from services.translator import _word_in_text

pytestmark = pytest.mark.unit


def test_word_in_text_exact_match():
    assert _word_in_text("brave", "i am brave today")


def test_word_in_text_substring_no_match():
    """'brave' не должно сматчиться с 'bravely' (это разные слова)."""
    assert not _word_in_text("brave", "he acted bravely today")


def test_word_in_text_case_insensitive():
    assert _word_in_text("Kitchen", "in the kitchen")


def test_word_in_text_empty_word():
    assert not _word_in_text("", "any text")


def test_word_in_text_at_start():
    assert _word_in_text("hello", "hello world")


def test_word_in_text_at_end():
    assert _word_in_text("world", "hello world")
