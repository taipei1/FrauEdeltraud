"""
VideoGeneratorService: генерация вертикального видео с озвучкой.
"""

import logging
import os
import subprocess
import shutil
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

from config import Settings
from services.tts_service import TTSService

logger = logging.getLogger(__name__)


class VideoGeneratorService:
    """Генератор вертикального видео."""

    def __init__(self):
        self.width = Settings.VIDEO_WIDTH
        self.height = Settings.VIDEO_HEIGHT
        self.fps = Settings.VIDEO_FPS
        self.font_size = Settings.VIDEO_FONT_SIZE
        self.tts = TTSService()
        logger.info(f"[VideoGeneratorService] {self.width}x{self.height} @ {self.fps} FPS")

    def _create_frame(self, text: str, word_timings: List[Dict], word_idx: int = -1) -> np.ndarray:
        """Создать кадр с текстом."""
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
        except Exception:
            font = ImageFont.load_default()

        words = [t["word"] for t in word_timings]
        if not words:
            return np.array(img)

        # Простое размещение
        text_str = " ".join(words)
        bbox = draw.textbbox((0, 0), text_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (self.width - text_width) // 2
        y = (self.height - text_height) // 2

        draw.text((x, y), text_str, fill=(255, 255, 255), font=font)

        return np.array(img)

    def process(self, text: str) -> Tuple[str, float]:
        """Генерировать видео."""
        logger.info("[VideoGeneratorService] Генерация видео...")

        try:
            # Генерируем аудио
            audio_path, duration, word_timings = self.tts.process(text)
            logger.info(f"[VideoGeneratorService] Аудио: {duration:.1f} сек, {len(word_timings)} слов")

            # Генерируем видеокадры
            temp_video = os.path.join(Settings.TEMP_DIR, "temp_video.mp4")
            video_path = os.path.join(Settings.TEMP_DIR, "video_final.mp4")

            for path in [temp_video, video_path]:
                if os.path.exists(path):
                    os.remove(path)

            logger.info(f"[VideoGeneratorService] Генерация кадров...")
            writer = imageio.get_writer(temp_video, fps=self.fps, codec="libx264", pixelformat="yuv420p")

            total_frames = int(duration * self.fps)
            for frame_idx in range(total_frames):
                current_time = frame_idx / self.fps
                
                word_idx = -1
                for timing in word_timings:
                    if timing["start"] <= current_time < timing["end"]:
                        word_idx = timing["index"]
                        break

                frame = self._create_frame(text, word_timings, word_idx)
                writer.append_data(frame)

                if (frame_idx + 1) % 60 == 0:
                    progress = (frame_idx + 1) / total_frames * 100
                    logger.info(f"[VideoGeneratorService] {progress:.0f}% ({frame_idx + 1}/{total_frames})")

            writer.close()
            logger.info(f"[VideoGeneratorService] Видео готово: {total_frames} кадров")

            # Объединяем с аудио
            logger.info(f"[VideoGeneratorService] Добавление аудио...")
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                cmd = [ffmpeg, "-i", temp_video, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", "-y", video_path]
                subprocess.run(cmd, capture_output=True, timeout=300)
            else:
                shutil.copy(temp_video, video_path)

            if os.path.exists(temp_video):
                os.remove(temp_video)

            file_size = os.path.getsize(video_path) / 1024 / 1024
            logger.info(f"[VideoGeneratorService] Финальное видео: {file_size:.2f} MB")

            return video_path, duration

        except Exception as e:
            logger.error(f"[VideoGeneratorService] Ошибка: {e}")
            raise
