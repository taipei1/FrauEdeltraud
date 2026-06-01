"""
TTS service: озвучка текста через edge-tts.
Мужской голос, стандартная скорость.
Возвращает время каждого слова для синхронизации субтитров (без librosa)!
"""

import logging
import os
import asyncio
from typing import Tuple, List, Dict
import edge_tts

logger = logging.getLogger(__name__)


class TTSService:
    """Генерирует аудио из текста через edge-tts, а также возвращает время каждого слова!"""

    def __init__(self):
        self.voice = "en-US-GuyNeural"  # Мужской голос
        self.rate = "+0%"  # Стандартная скорость (формат edge-tts: +0%, -10%, +20%)
        logger.info(f"[OK] TTS: голос={self.voice}, скорость={self.rate}")

    async def generate_audio_with_word_timings(self, text: str) -> Tuple[str, float, List[Dict]]:
        """
        Асинхронно сгенерировать MP3 И получить время каждого слова!
        Возвращает: (путь_к_аудио, длительность, список_слов_с_временем)
        """
        output_path = os.path.join(Settings.TEMP_DIR, "tts_audio.mp3")

        # Удаляем старый файл
        if os.path.exists(output_path):
            os.remove(output_path)

        logger.info(f"TTS: генерация аудио ({len(text)} символов)...")

        word_timings = []

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate
        )

        # Собираем аудио и события
        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    # Сохраняем информацию о слове
                    word_timings.append({
                        "word": chunk["text"],
                        "start": chunk["offset"] / 10_000_000,  # из 100-наносекунд в секунды
                        "end": (chunk["offset"] + chunk["duration"]) / 10_000_000,
                        "index": len(word_timings),
                        "duration": chunk["duration"] / 10_000_000
                    })

        file_size = os.path.getsize(output_path)
        logger.info(f"[OK] Аудио готово: {file_size / 1024:.1f} KB, {len(word_timings)} слов с синхронизацией")

        # Получаем общую длительность
        total_duration = word_timings[-1]["end"] if word_timings else len(text) / 4.0

        return output_path, total_duration, word_timings

    def generate_audio(self, text: str) -> Tuple[str, float, List[Dict]]:
        """Синхронная обёртка для генерации аудио с временем слов."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.generate_audio_with_word_timings(text))
            return result
        finally:
            loop.close()

    def process(self, text: str) -> Tuple[str, float, List[Dict]]:
        """
        Главный метод: генерирует аудио, длительность И ВРЕМЯ КАЖДОГО СЛОВА!
        Возвращает: (путь_к_аудио, длительность, список_слов_с_временем)
        """
        logger.info("[AGENT 3] TTS: озвучка перевода...")

        audio_path, duration, word_timings = self.generate_audio(text)

        logger.info(f"[OK] TTS готово: {duration:.1f} сек")
        return audio_path, duration, word_timings


from config import Settings
