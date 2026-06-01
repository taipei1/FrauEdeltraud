"""
Video service: генерация вертикального видео с субтитрами и аудиодорожкой.
Вертикальное видео (9:16), большой текст, ТОЧНАЯ синхронизация из edge-tts (БЕЗ LIBSOSA)!
"""

import logging
import os
import subprocess
import shutil
import numpy as np
from typing import Tuple, List, Dict
from PIL import Image, ImageDraw, ImageFont
import imageio
from services.tts_service import TTSService
from config import Settings

logger = logging.getLogger(__name__)


class VideoService:
    """Генерирует вертикальное видео с субтитрами, подсветкой слов и аудиодорожкой (БЕЗ librosa)!"""

    def __init__(self):
        # ВЕРТИКАЛЬНОЕ видео (9:16 - как TikTok/Instagram Reels)
        # Размеры кратны 16 для совместимости с видеокодеками
        self.width = 544  # Кратно 16 (544 / 16 = 34)
        self.height = 960  # Кратно 16 (960 / 16 = 60)
        self.fps = 15
        # БОЛЬШИЕ размеры шрифта
        self.font_size = 56  # Было 32
        self.highlight_font_size = 64  # Было 36
        self.background_color = (0, 0, 0)  # Чёрный
        self.text_color = (255, 255, 255)  # Белый
        self.highlight_color = (255, 255, 0)  # Жёлтый (подсветка)
        self.tts = TTSService()
        
        # Ищем ffmpeg
        self.ffmpeg_path = self._find_ffmpeg()
        logger.info(f"[OK] VideoService: {self.width}x{self.height} @ {self.fps} FPS (9:16 вертикальное)")
        logger.info(f"[OK] Шрифт: {self.font_size}pt (обычный), {self.highlight_font_size}pt (подсвеченный)")
        logger.info(f"[OK] FFmpeg: {self.ffmpeg_path if self.ffmpeg_path else 'не найден'}")

    def _find_ffmpeg(self) -> str:
        """Найти ffmpeg в системе."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        
        return None

    def create_frame_with_highlighting(
        self,
        text: str,
        word_timings: List[Dict],
        highlighted_word_index: int = -1
    ) -> np.ndarray:
        """
        Создать кадр с полным текстом и подсветкой активного слова.
        БОЛЬШОЙ РАЗМЕР ТЕКСТА, ВЕРТИКАЛЬНЫЙ ФОРМАТ.
        """
        img = Image.new("RGB", (self.width, self.height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Загружаем шрифты (БОЛЬШИЕ)
        try:
            normal_font = ImageFont.truetype("arial.ttf", self.font_size)
            highlight_font = ImageFont.truetype("arial.ttf", self.highlight_font_size)
        except Exception:
            logger.warning("Не удалось загрузить arial.ttf")
            normal_font = ImageFont.load_default()
            highlight_font = ImageFont.load_default()

        # Создаём список слов из word_timings
        words = [t["word"] for t in word_timings]
        if not words:
            return np.array(img)

        # Параметры размещения для вертикального видео
        margin_left = 20
        margin_right = 20
        max_width = self.width - margin_left - margin_right
        line_height = self.font_size + 20
        max_lines = 10  # Больше строк в вертикальном формате

        # Формируем строки текста
        lines = []
        current_line = []
        current_width = 0

        for i, word in enumerate(words):
            font = highlight_font if i == highlighted_word_index else normal_font
            bbox = draw.textbbox((0, 0), word + " ", font=font)
            word_width = bbox[2] - bbox[0]

            if current_width + word_width > max_width and current_line:
                lines.append(current_line)
                current_line = [(word, i)]
                current_width = word_width
            else:
                current_line.append((word, i))
                current_width += word_width

        if current_line:
            lines.append(current_line)

        # Центрируем текст по вертикали
        total_height = len(lines) * line_height
        y_start = (self.height - total_height) // 2
        y_start = max(y_start, 50)

        # Рисуем текст
        for line_idx, line in enumerate(lines[:max_lines]):
            # Вычисляем общую ширину строки
            line_width = 0
            for word, word_idx in line:
                font = highlight_font if word_idx == highlighted_word_index else normal_font
                bbox = draw.textbbox((0, 0), word + " ", font=font)
                line_width += bbox[2] - bbox[0]

            # Центрируем строку по горизонтали
            x_start = (self.width - line_width) // 2
            x_start = max(x_start, margin_left)
            x_start = min(x_start, self.width - line_width - margin_right)

            # Рисуем слова в строке
            x = x_start
            y = y_start + line_idx * line_height

            for word, word_idx in line:
                is_highlighted = (word_idx == highlighted_word_index)
                font = highlight_font if is_highlighted else normal_font
                color = self.highlight_color if is_highlighted else self.text_color

                # Если слово подсвечено, добавляем фон
                if is_highlighted:
                    bbox = draw.textbbox((x, y), word, font=font)
                    draw.rectangle(
                        [bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5],
                        fill=(50, 50, 0)
                    )

                draw.text((x, y), word, fill=color, font=font)

                bbox = draw.textbbox((x, y), word + " ", font=font)
                x = bbox[2]

        return np.array(img)

    def process(self, text: str) -> Tuple[str, float]:
        """
        Главный метод: генерирует ВЕРТИКАЛЬНОЕ видео с озвучкой и подсветкой слов (БЕЗ librosa)!
        """
        logger.info("[VIDEO SERVICE] Генерация ВЕРТИКАЛЬНОГО видео с синхронизацией из edge-tts...")

        try:
            # Генерируем аудио И получаем время каждого слова из edge-tts!
            audio_path, audio_duration, word_timings = self.tts.process(text)
            logger.info(f"Аудио готово: {audio_duration:.1f} сек, путь: {audio_path}, {len(word_timings)} слов с синхронизацией")

            # Генерируем видео
            video_path = os.path.join(Settings.TEMP_DIR, "video_with_subtitles.mp4")
            temp_video_path = os.path.join(Settings.TEMP_DIR, "temp_video.mp4")

            # Удаляем старые файлы
            for path in [video_path, temp_video_path]:
                if os.path.exists(path):
                    os.remove(path)

            # Генерируем видеокадры без аудио
            logger.info("Генерация видеокадров...")
            self._generate_video_without_audio(
                text, 
                audio_duration, 
                word_timings, 
                temp_video_path
            )

            # Объединяем видео с аудио
            logger.info("Добавление аудиодорожки...")
            self._merge_video_and_audio(temp_video_path, audio_path, video_path)

            # Удаляем временное видео
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

            file_size = os.path.getsize(video_path) / 1024 / 1024
            logger.info(f"[OK] Финальное видео готово: {file_size:.2f} MB")

            return video_path, audio_duration

        except Exception as e:
            logger.error(f"[FAIL] Ошибка при генерации видео: {e}")
            raise

    def _generate_video_without_audio(
        self, 
        text: str, 
        audio_duration: float, 
        word_timings: List[Dict], 
        output_path: str
    ):
        """
        Генерировать видео без аудио, используя imageio и timings из edge-tts.
        """
        try:
            writer = imageio.get_writer(
                output_path,
                fps=self.fps,
                codec="libx264",
                pixelformat="yuv420p",
            )

            total_frames = int(audio_duration * self.fps)
            current_word_idx = -1
            logger.info(f"Генерация {total_frames} кадров ({audio_duration:.1f} сек @ {self.fps} FPS)...")

            for frame_idx in range(total_frames):
                current_time = frame_idx / self.fps

                # Определяем текущее слово по точному времени из edge-tts
                for timing in word_timings:
                    if timing["start"] <= current_time < timing["end"]:
                        current_word_idx = timing["index"]
                        break

                # Создаём кадр
                frame_array = self.create_frame_with_highlighting(
                    text,
                    word_timings,
                    highlighted_word_index=current_word_idx
                )
                writer.append_data(frame_array)

                if (frame_idx + 1) % 60 == 0:
                    progress = (frame_idx + 1) / total_frames * 100
                    logger.info(f"  Прогресс: {progress:.0f}% ({frame_idx + 1}/{total_frames} кадров)")

            writer.close()
            logger.info(f"[OK] Видео (без аудио) готово: {total_frames} кадров")

        except Exception as e:
            logger.error(f"[FAIL] Ошибка при генерации видео: {e}")
            raise

    def _merge_video_and_audio(self, video_path: str, audio_path: str, output_path: str):
        """
        Объединить видео с аудиодорожкой через ffmpeg или PyAV.
        """
        try:
            if self.ffmpeg_path:
                logger.info(f"Использование ffmpeg: {self.ffmpeg_path}")
                cmd = [
                    self.ffmpeg_path,
                    "-i", video_path,
                    "-i", audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    "-y",
                    output_path
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode != 0:
                    logger.warning(f"ffmpeg вернул ошибку: {result.stderr}")
                else:
                    logger.info("[OK] Аудио успешно добавлено через ffmpeg")
                    return

            logger.info("Пересоздание видео с аудио через PyAV...")
            self._merge_with_pyav(video_path, audio_path, output_path)

        except Exception as e:
            logger.error(f"[FAIL] Ошибка при объединении видео и аудио: {e}")
            if not os.path.exists(output_path):
                shutil.copy(video_path, output_path)

    def _merge_with_pyav(self, video_path: str, audio_path: str, output_path: str):
        """
        Объединить видео и аудио через PyAV.
        """
        try:
            import av
            
            video_container = av.open(video_path)
            video_stream = video_container.streams.video[0]
            
            audio_container = av.open(audio_path)
            audio_stream = audio_container.streams.audio[0]
            
            output_container = av.open(output_path, "w")
            
            video_out = output_container.add_stream(
                video_stream.codec_context.name,
                rate=video_stream.rates[0]
            )
            video_out.width = video_stream.width
            video_out.height = video_stream.height
            
            audio_out = output_container.add_stream(
                audio_stream.codec_context.name,
                rate=audio_stream.sample_rate
            )
            
            for frame in video_container.decode(video_stream):
                for packet in video_out.encode(frame):
                    output_container.mux(packet)
            
            for packet in video_out.encode():
                output_container.mux(packet)
            
            for frame in audio_container.decode(audio_stream):
                for packet in audio_out.encode(frame):
                    output_container.mux(packet)
            
            for packet in audio_out.encode():
                output_container.mux(packet)
            
            output_container.close()
            logger.info("[OK] Видео с аудио создано через PyAV")
            
        except Exception as e:
            logger.error(f"[FAIL] Ошибка PyAV: {e}")
            raise
