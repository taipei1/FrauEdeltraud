"""
Полный ручной тест ВСЕГО PIPELINE - вызови сам с данными.
"""

import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from services.extractor import ExtractorService
from services.translator import TranslatorService
from services.critic_service import CriticService
from services.video_service import VideoGeneratorService


def test_full_pipeline_simple():
    """Полный pipeline: текст → перевод → видео."""
    print("\n" + "="*80)
    print("ПОЛНЫЙ PIPELINE: Русский текст → Перевод → Видео")
    print("="*80)
    
    # 1. EXTRACTOR
    print("\n[1/4] EXTRACTOR")
    print("-" * 80)
    extractor = ExtractorService()
    source = "Привет! Как дела? Это русский текст для теста."
    raw_text, language, _, _ = extractor.process(source)
    print(f"✓ Текст извлечён: {raw_text[:60]}...\n")
    
    # 2. TRANSLATOR
    print("[2/4] TRANSLATOR")
    print("-" * 80)
    translator = TranslatorService()
    translated, stats = translator.process(raw_text)
    print(f"✓ Переведено: {translated[:60]}...\n")
    
    # 3. CRITIC (опционально)
    print("[3/4] CRITIC")
    print("-" * 80)
    critic = CriticService()
    is_approved, final_translation, details = critic.critique(raw_text, translated, stats)
    print(f"✓ Оценка: {details.get('score', 0)}/10, Одобрено: {is_approved}\n")
    
    # 4. VIDEO GENERATOR
    print("[4/4] VIDEO GENERATOR")
    print("-" * 80)
    try:
        video_gen = VideoGeneratorService()
        video_path, duration = video_gen.process(final_translation)
        print(f"✓ Видео создано: {duration:.1f} сек\n")
    except Exception as e:
        print(f"⚠ Видео ошибка: {e}\n")


def test_full_pipeline_youtube():
    """Полный pipeline: YouTube → Перевод → Видео."""
    print("\n" + "="*80)
    print("ПОЛНЫЙ PIPELINE: YouTube → Перевод → Видео")
    print("="*80)
    
    # 1. EXTRACTOR
    print("\n[1/4] EXTRACTOR")
    print("-" * 80)
    extractor = ExtractorService()
    # Замени на реальный URL
    source = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    try:
        raw_text, language, _, _ = extractor.process(source)
        print(f"✓ Текст извлечён: {raw_text[:60]}...\n")
    except Exception as e:
        print(f"✗ Ошибка: {e}\n")
        return
    
    # 2. TRANSLATOR
    print("[2/4] TRANSLATOR")
    print("-" * 80)
    translator = TranslatorService()
    translated, stats = translator.process(raw_text)
    print(f"✓ Переведено: {translated[:60]}...\n")
    
    # 3. CRITIC
    print("[3/4] CRITIC")
    print("-" * 80)
    critic = CriticService()
    is_approved, final_translation, details = critic.critique(raw_text, translated, stats)
    print(f"✓ Оценка: {details.get('score', 0)}/10\n")
    
    # 4. VIDEO GENERATOR
    print("[4/4] VIDEO GENERATOR")
    print("-" * 80)
    try:
        video_gen = VideoGeneratorService()
        video_path, duration = video_gen.process(final_translation)
        print(f"✓ Видео создано: {duration:.1f} сек\n")
    except Exception as e:
        print(f"⚠ Видео ошибка: {e}\n")


def test_pipeline_custom():
    """Полный pipeline с собственными данными."""
    print("\n" + "="*80)
    print("ПОЛНЫЙ PIPELINE: Собственные данные")
    print("="*80)
    
    # РЕДАКТИРУЙ ЭТИ ДАННЫЕ:
    custom_source = "Мне очень нравится программирование на Python. Это красивый и мощный язык."
    skip_video = False  # True если не нужно видео
    
    # 1. EXTRACTOR
    print("\n[1/4] EXTRACTOR")
    print("-" * 80)
    extractor = ExtractorService()
    raw_text, language, _, _ = extractor.process(custom_source)
    print(f"✓ Текст: {raw_text[:60]}...\n")
    
    # 2. TRANSLATOR
    print("[2/4] TRANSLATOR")
    print("-" * 80)
    translator = TranslatorService()
    translated, stats = translator.process(raw_text)
    print(f"✓ Перевод: {translated[:60]}...\n")
    
    # 3. CRITIC
    print("[3/4] CRITIC")
    print("-" * 80)
    critic = CriticService()
    is_approved, final_translation, details = critic.critique(raw_text, translated, stats)
    print(f"✓ Оценка: {details.get('score', 0)}/10\n")
    
    # 4. VIDEO GENERATOR
    if not skip_video:
        print("[4/4] VIDEO GENERATOR")
        print("-" * 80)
        try:
            video_gen = VideoGeneratorService()
            video_path, duration = video_gen.process(final_translation)
            print(f"✓ Видео: {duration:.1f} сек\n")
        except Exception as e:
            print(f"⚠ Видео ошибка: {e}\n")
    else:
        print("[4/4] VIDEO GENERATOR - ПРОПУЩЕНО\n")


if __name__ == "__main__":
    print("\n" + "▶"*40)
    print("РУЧНЫЕ ТЕСТЫ: ПОЛНЫЙ PIPELINE")
    print("▶"*40)
    
    # Раскомментируй нужный тест:
    test_full_pipeline_simple()
    # test_full_pipeline_youtube()
    # test_pipeline_custom()
