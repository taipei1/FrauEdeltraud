"""
Database service: работа с PostgreSQL + pgvector.
Ищет "горячие" слова (которые нужно повторить) для приоритизации перевода.
"""

import logging
from typing import List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
from config import Settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Работа с pgvector для поиска слов ученика."""

    def __init__(self):
        self.connection = None
        self.available = False

        try:
            self.connection = psycopg2.connect(Settings.DATABASE_URL)
            self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            # Проверяем существует ли таблица vocabulary
            self.cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'vocabulary')"
            )
            exists = self.cursor.fetchone()
            if exists and list(exists.values())[0]:
                self.available = True
                logger.info("[OK] Подключение к БД, таблица vocabulary найдена")
            else:
                logger.warning("[WARN] Таблица vocabulary не найдена, БД работает в режиме заглушки")
        except Exception as e:
            logger.warning(f"[WARN] БД недоступна: {e}. Будет использоваться простой перевод.")
            self.available = False

    def get_hot_vocabulary(self, limit: int = 100) -> List[Dict]:
        """Получить слова для повторения (просроченные, нестабильные)."""
        if not self.available:
            return []

        try:
            query = """
            SELECT id, word, translation, embedding, stability, difficulty, review_date
            FROM vocabulary
            ORDER BY
                CASE
                    WHEN review_date IS NULL THEN 0
                    WHEN review_date <= NOW() THEN 1
                    ELSE 2
                END DESC,
                stability ASC NULLS LAST,
                COALESCE(review_date, created_at) ASC
            LIMIT %s
            """
            self.cursor.execute(query, (limit,))
            results = self.cursor.fetchall()
            logger.info(f"[OK] Получено {len(results)} слов из БД")
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"[FAIL] Ошибка запроса к БД: {e}")
            return []

    def search_by_translation(self, russian_word: str) -> str:
        """Найти английское слово по русскому переводу."""
        if not self.available:
            return ""

        try:
            query = "SELECT word FROM vocabulary WHERE LOWER(translation) = LOWER(%s) LIMIT 1"
            self.cursor.execute(query, (russian_word,))
            row = self.cursor.fetchone()
            return row["word"] if row else ""
        except Exception:
            return ""

    def close(self):
        """Закрыть подключение."""
        try:
            if self.connection:
                self.connection.close()
        except Exception:
            pass


# Глобальный экземпляр
_db_service = None


def get_db_service() -> DatabaseService:
    """Получить экземпляр DatabaseService (синглтон)."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
