"""
Ручной тест CriticService - проверка качества перевода.
"""

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.critic_service import CriticService


def test_critique_good_translation():
    """Тест 1: Хороший перевод."""
    print("\n" + "="*80)
    print("TEST 1: Критика хорошего перевода")
    print("="*80)
    
    service = CriticService()
    
    source = "Привет, как дела?"
    translation = "Hello, how are you?"
    stats = {"total_vocabulary_words": 5, "used_count": 3, "vocabulary_percentage": 60}
    
    print(f"\nИсходный текст:")
    print(f"  {source}")
    print(f"\nПеревод:")
    print(f"  {translation}")
    
    is_approved, final_translation, details = service.critique(source, translation, stats)
    
    print(f"\nРезультат критики:")
    print(f"  Оценка: {details.get('score', 0)}/10")
    print(f"  Одобрено: {is_approved}")
    print(f"  Проблемы: {details.get('issues', [])}")
    print(f"  Рекомендации: {details.get('improvements', [])}")


def test_critique_bad_translation():
    """Тест 2: Плохой перевод."""
    print("\n" + "="*80)
    print("TEST 2: Критика плохого перевода")
    print("="*80)
    
    service = CriticService()
    
    source = "Мне нравится учить новые слова каждый день."
    translation = "Me like studying new words each day."  # Ошибка: "Me like" вместо "I like"
    stats = {"total_vocabulary_words": 10, "used_count": 5, "vocabulary_percentage": 50}
    
    print(f"\nИсходный текст:")
    print(f"  {source}")
    print(f"\nПеревод (плохой):")
    print(f"  {translation}")
    
    is_approved, final_translation, details = service.critique(source, translation, stats)
    
    print(f"\nРезультат критики:")
    print(f"  Оценка: {details.get('score', 0)}/10")
    print(f"  Одобрено: {is_approved}")
    if details.get('issues'):
        print(f"  Проблемы:")
        for issue in details.get('issues', []):
            print(f"    - {issue}")
    if details.get('improvements'):
        print(f"  Рекомендации:")
        for imp in details.get('improvements', []):
            print(f"    - {imp}")


def test_critique_technical():
    """Тест 3: Технический текст."""
    print("\n" + "="*80)
    print("TEST 3: Критика технического текста")
    print("="*80)
    
    service = CriticService()
    
    source = "Переменная это контейнер для хранения данных."
    translation = "Variable is a container for store data."  # Ошибка: "for store" вместо "for storing"
    stats = {"total_vocabulary_words": 8, "used_count": 4, "vocabulary_percentage": 50}
    
    print(f"\nИсходный текст:")
    print(f"  {source}")
    print(f"\nПеревод:")
    print(f"  {translation}")
    
    is_approved, final_translation, details = service.critique(source, translation, stats)
    
    print(f"\nРезультат критики:")
    print(f"  Оценка: {details.get('score', 0)}/10")
    print(f"  Одобрено: {is_approved}")
    print(f"  Проблемы: {details.get('issues', [])}")


if __name__ == "__main__":
    print("\n" + "▶"*40)
    print("РУЧНЫЕ ТЕСТЫ: CriticService")
    print("▶"*40)
    
    # Раскомментируй нужный тест:
    test_critique_good_translation()
    # test_critique_bad_translation()
    # test_critique_technical()
