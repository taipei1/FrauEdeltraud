"""Общие фикстуры для автотестов FrauEdeltraud.

Структура маркеров:
- unit       : чистые тесты, без сети/БД (запускаются всегда)
- integration: нужны сеть и API-ключи
- slow       : долгие (YouTube, рендер видео)
- requires_db: нужен PostgreSQL на DATABASE_URL

Запуск:
  pytest tests/ -v                          # всё подряд (часть упадёт без API)
  pytest tests/ -m unit -v                  # только быстрые
  pytest tests/ -m "unit or integration" -v # без slow
  pytest tests/ -m "not slow" -v            # всё кроме долгих
"""
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

TEMP_OUTPUT = PROJECT_ROOT / "tests" / "test_outputs"
TEMP_OUTPUT.mkdir(exist_ok=True)


@pytest.fixture
def temp_output_dir() -> Path:
    """Изолированная папка для артефактов теста (аудио/видео/txt)."""
    import shutil

    folder = TEMP_OUTPUT / "current"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    yield folder


@pytest.fixture
def sample_text() -> str:
    """Короткий текст для быстрых тестов TTS/видео."""
    return (
        "I often lose my coin purse. "
        "I am brave, so I use an iron in the kitchen."
    )


@pytest.fixture
def short_text() -> str:
    return "Hello world."
