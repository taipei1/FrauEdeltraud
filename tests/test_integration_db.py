"""Интеграционные тесты: подключение к PostgreSQL + словарь."""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


def test_database_available():
    """Проверить что PostgreSQL отвечает и таблица cards/vocabulary существует."""
    from config import Settings
    from services.db_service import get_db_service

    db = get_db_service()
    if not db.available:
        pytest.skip(f"PostgreSQL недоступен: {Settings.DATABASE_URL}")
    words = db.get_hot_vocabulary(limit=10)
    assert isinstance(words, list)
