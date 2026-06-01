"""
ExtractorService: извлечение текста из различных источников.
"""

import logging
import os
import re
from typing import Tuple

import yt_dlp

from config import Settings
from services.llm_client import generate_multimodal

logger = logging.getLogger(__name__)


class ExtractorService:
    """Извлекает текст из видео, YouTube или текста."""

    def __init__(self):
        self.temp_dir = Settings.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info("[ExtractorService] Инициализирован")

    @staticmethod
    def _detect_language(text: str) -> str:
        """Определить язык."""
        cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04FF")
        latin = sum(1 for c in text if "a" <= c.lower() <= "z")
        return "ru" if cyrillic > latin else "en"

    @staticmethod
    def _clean_subtitles(content: str) -> str:
        """Очистить субтитры."""
        lines = []
        prev = ""
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.isdigit() or "-->" in line or re.match(r"^\d+:\d+", line):
                continue
            line = re.sub(r"<[^>]+>", "", line)
            if line and line != prev:
                lines.append(line)
                prev = line
        
        return " ".join(lines)

    def _try_subtitles(self, url: str) -> str:
        """Загрузить субтитры."""
        out_tmpl = os.path.join(self.temp_dir, "subs.%(ext)s")
        
        try:
            logger.info("[ExtractorService] Загрузка субтитров...")
            with yt_dlp.YoutubeDL({
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["ru", "en"],
                "outtmpl": out_tmpl,
                "quiet": True,
            }) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.warning(f"[ExtractorService] Не удалось загрузить субтитры: {e}")
            return ""

        for fname in os.listdir(self.temp_dir):
            if fname.startswith("subs") and fname.endswith((".vtt", ".srt")):
                path = os.path.join(self.temp_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    text = self._clean_subtitles(f.read())
                try:
                    os.remove(path)
                except:
                    pass
                if text:
                    logger.info(f"[ExtractorService] Субтитры загружены: {len(text)} символов")
                    return text
        
        return ""

    def _download_audio(self, url: str) -> str:
        """Загрузить аудио."""
        for ext in ("m4a", "mp3", "webm"):
            out_tmpl = os.path.join(self.temp_dir, f"audio.{ext}")
            try:
                with yt_dlp.YoutubeDL({
                    "format": f"bestaudio[ext={ext}]/bestaudio",
                    "outtmpl": out_tmpl,
                    "quiet": True,
                }) as ydl:
                    ydl.download([url])
            except:
                continue
            
            for fname in os.listdir(self.temp_dir):
                if fname.startswith("audio.") and fname.endswith(f".{ext}"):
                    audio_path = os.path.join(self.temp_dir, fname)
                    logger.info(f"[ExtractorService] Аудио загружено")
                    return audio_path
        
        return ""

    def _download_video(self, url: str) -> str:
        """Загрузить видео."""
        out_path = os.path.join(self.temp_dir, "video.mp4")
        
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
            
            logger.info("[ExtractorService] Загрузка видео...")
            with yt_dlp.YoutubeDL({
                "format": "best[ext=mp4][filesize<=100M]/best[ext=mp4]/best",
                "outtmpl": out_path,
                "quiet": True,
            }) as ydl:
                ydl.download([url])
            
            if os.path.exists(out_path):
                logger.info(f"[ExtractorService] Видео загружено")
                return out_path
        except Exception as e:
            logger.warning(f"[ExtractorService] Не удалось загрузить видео: {e}")
        
        return ""

    def _transcribe_audio(self, audio_path: str) -> Tuple[str, str]:
        """Распознать речь в аудио."""
        logger.info("[ExtractorService] Распознавание аудио через Gemini...")
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        prompt = "Transcribe verbatim in original language. End with line: LANGUAGE: <name>"
        parts = [prompt, {"inline_data": {"mime_type": "audio/mp3", "data": audio_bytes}}]
        result = generate_multimodal(parts)

        language = "unknown"
        m = re.search(r"LANGUAGE[=:]\s*([^\n]+)", result, re.IGNORECASE)
        if m:
            language = m.group(1).strip()
            result = result[:m.start()].strip()

        try:
            os.remove(audio_path)
        except:
            pass

        logger.info(f"[ExtractorService] Текст распознан: {len(result)} символов, язык: {language}")
        return result, language

    def _transcribe_video(self, video_path: str) -> Tuple[str, str]:
        """Распознать речь в видео."""
        logger.info("[ExtractorService] Распознавание видео через Gemini...")
        
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        prompt = "Transcribe speech verbatim. End with line: LANGUAGE: <name>"
        parts = [prompt, {"inline_data": {"mime_type": "video/mp4", "data": video_bytes}}]
        result = generate_multimodal(parts)

        text = result
        language = "unknown"
        for line in result.splitlines():
            if "LANGUAGE:" in line.upper():
                language = line.split(":", 1)[1].strip()
                text = result[:result.index(line)].strip()
                break

        logger.info(f"[ExtractorService] Текст распознан: {len(text)} символов, язык: {language}")
        return text, language

    def process(self, source: str) -> Tuple[str, str, str, str]:
        """Извлечь текст. Возвращает (text, language, video_path, audio_path)."""
        logger.info(f"[ExtractorService] Обработка: {source[:80]}...")

        if source.startswith(("http://", "https://")):
            # YouTube URL
            subs = self._try_subtitles(source)
            if subs:
                lang = self._detect_language(subs)
                return subs, lang, "", ""

            audio = self._download_audio(source)
            if audio:
                text, lang = self._transcribe_audio(audio)
                return text, lang, "", ""

            video = self._download_video(source)
            if video:
                text, lang = self._transcribe_video(video)
                return text, lang, "", ""

            raise RuntimeError("Не удалось извлечь текст из URL")
        
        elif os.path.exists(source):
            # Локальный файл
            text, lang = self._transcribe_video(source)
            return text, lang, "", ""
        
        else:
            # Прямой текст
            lang = self._detect_language(source)
            logger.info(f"[ExtractorService] Текст: {len(source)} символов, язык: {lang}")
            return source, lang, "", ""
