"""
Ручной тест ExtractorService - вызови сам функцию с данными.
"""

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.extractor import ExtractorService


def test_extract_russian_text():
    """Тест 1: Русский текст."""
    print("\n" + "="*80)
    print("TEST 1: Русский текст")
    print("="*80)
    
    service = ExtractorService()
    
    source = "Привет! Это русский текст. Как дела?"
    
    text, language, video_path, audio_path = service.process(source)
    
    print(f"\nРезультат:")
    print(f"  Текст: {text}")
    print(f"  Язык: {language}")
    print(f"  Длина: {len(text)} символов")


def test_extract_english_text():
    """Тест 2: Английский текст."""
    print("\n" + "="*80)
    print("TEST 2: Английский текст")
    print("="*80)
    
    service = ExtractorService()
    
    source = "Hello world! This is English text. How are you?"
    
    text, language, video_path, audio_path = service.process(source)
    
    print(f"\nРезультат:")
    print(f"  Текст: {text}")
    print(f"  Язык: {language}")
    print(f"  Длина: {len(text)} символов")


def test_extract_youtube():
    """Тест 3: YouTube URL (нужен GOOGLE_API_KEY)."""
    print("\n" + "="*80)
    print("TEST 3: YouTube URL")
    print("="*80)
    
    service = ExtractorService()
    
    # Замени на реальный URL
    source = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        text, language, video_path, audio_path = service.process(source)
        
        print(f"\nРезультат:")
        print(f"  Текст: {text[:100]}...")
        print(f"  Язык: {language}")
        print(f"  Видео: {video_path}")
        print(f"  Аудио: {audio_path}")
    except Exception as e:
        print(f"\n⚠ Ошибка: {e}")


if __name__ == "__main__":
    print("\n" + "▶"*40)
    print("РУЧНЫЕ ТЕСТЫ: ExtractorService")
    print("▶"*40)
    
    # Вызови тест который нужен:
    test_extract_russian_text()
    # test_extract_english_text()
    # test_extract_youtube()
