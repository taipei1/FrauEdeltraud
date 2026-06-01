"""
Ручной тест TranslatorService - вызови сам с нужным текстом.
"""

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.translator import TranslatorService


def test_translate_simple():
    """Тест 1: Простой текст."""
    print("\n" + "="*80)
    print("TEST 1: Простой перевод")
    print("="*80)
    
    service = TranslatorService()
    
    source = "Привет! Как твои дела? Мне нравится учить английский язык."
    
    print(f"\nИсходный текст:")
    print(f"  {source}")
    
    translated, stats = service.process(source)
    
    print(f"\nПеревод:")
    print(f"  {translated}")
    print(f"\nСтатистика:")
    print(f"  Длина: {len(translated)} символов")


def test_translate_long():
    """Тест 2: Длинный текст."""
    print("\n" + "="*80)
    print("TEST 2: Длинный текст")
    print("="*80)
    
    service = TranslatorService()
    
    source = """Вчера я пошел в парк с друзьями. Было хорошую погоду. 
    Мы играли в футбол и пили лимонад. После этого мы пошли в кафе и поговорили о наших планах на лето.
    Мне очень понравился этот день."""
    
    print(f"\nИсходный текст:")
    print(f"  {source[:100]}...")
    
    translated, stats = service.process(source)
    
    print(f"\nПеревод:")
    print(f"  {translated[:100]}...")
    print(f"\nСтатистика:")
    print(f"  Длина: {len(translated)} символов")


def test_translate_technical():
    """Тест 3: Технический текст."""
    print("\n" + "="*80)
    print("TEST 3: Технический текст")
    print("="*80)
    
    service = TranslatorService()
    
    source = "Переменная это именованная область памяти. Функция это блок кода который решает определенную задачу."
    
    print(f"\nИсходный текст:")
    print(f"  {source}")
    
    translated, stats = service.process(source)
    
    print(f"\nПеревод:")
    print(f"  {translated}")


if __name__ == "__main__":
    print("\n" + "▶"*40)
    print("РУЧНЫЕ ТЕСТЫ: TranslatorService")
    print("▶"*40)
    
    # Раскомментируй нужный тест:
    test_translate_simple()
    # test_translate_long()
    # test_translate_technical()
