"""
Ручной тест VideoGeneratorService - генерация видео.
"""

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.video_service import VideoGeneratorService


def test_generate_simple_video():
    """Тест 1: Простое видео."""
    print("\n" + "="*80)
    print("TEST 1: Генерация простого видео")
    print("="*80)
    
    service = VideoGeneratorService()
    
    text = "Hello world! How are you today?"
    
    print(f"\nТекст для видео:")
    print(f"  {text}")
    
    print(f"\nНастройки видео:")
    print(f"  Размер: {service.width}x{service.height}")
    print(f"  FPS: {service.fps}")
    print(f"  Шрифт: {service.font_size}pt")
    
    try:
        video_path, duration = service.process(text)
        
        print(f"\nРезультат:")
        print(f"  Видео создано: {video_path}")
        print(f"  Длительность: {duration:.1f} сек")
    except Exception as e:
        print(f"\n⚠ Ошибка: {e}")


def test_generate_russian_video():
    """Тест 2: Русский текст в видео."""
    print("\n" + "="*80)
    print("TEST 2: Видео с русским текстом")
    print("="*80)
    
    service = VideoGeneratorService()
    
    text = "Привет! Это видео с русским текстом."
    
    print(f"\nТекст для видео:")
    print(f"  {text}")
    
    try:
        video_path, duration = service.process(text)
        
        print(f"\nРезультат:")
        print(f"  Видео создано: {video_path}")
        print(f"  Длительность: {duration:.1f} сек")
    except Exception as e:
        print(f"\n⚠ Ошибка: {e}")


def test_generate_long_video():
    """Тест 3: Длинный текст в видео."""
    print("\n" + "="*80)
    print("TEST 3: Видео с длинным текстом")
    print("="*80)
    
    service = VideoGeneratorService()
    
    text = "Today is a beautiful day. The sun is shining brightly. Birds are singing in the trees. People are walking in the park. Everyone is happy and smiling."
    
    print(f"\nТекст для видео:")
    print(f"  {text[:80]}...")
    
    try:
        video_path, duration = service.process(text)
        
        print(f"\nРезультат:")
        print(f"  Видео создано: {video_path}")
        print(f"  Длительность: {duration:.1f} сек")
    except Exception as e:
        print(f"\n⚠ Ошибка: {e}")


if __name__ == "__main__":
    print("\n" + "▶"*40)
    print("РУЧНЫЕ ТЕСТЫ: VideoGeneratorService")
    print("▶"*40)
    
    # Раскомментируй нужный тест:
    test_generate_simple_video()
    # test_generate_russian_video()
    # test_generate_long_video()
