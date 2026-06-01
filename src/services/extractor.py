"""Extractor service: извлечение текста из текста, YouTube или видео.

Приоритет: готовые субтитры → аудио (Gemini) → видео (Gemini).
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
    """Извлекает текст из видео/аудио/текста."""

    def __init__(self):
        self.temp_dir = Settings.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info("[OK] ExtractorService инициализирован")

    @staticmethod
    def _detect_language(text: str) -> str:
        return "ru" if any("\u0400" <= c <= "\u04FF" for c in text) else "en"

    @staticmethod
    def _clean_subtitle_text(content: str) -> str:
        """Очистить VTT/SRT от меток времени и дублей."""
        lines_out: list[str] = []
        prev = ""
        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            if (
                line.isdigit()
                or "-->" in line
                or re.match(r"^\d+:\d+", line)
            ):
                continue
            line = re.sub(r"<[^>]+>", "", line)
            if not line or line == prev:
                continue
            lines_out.append(line)
            prev = line
        return " ".join(lines_out)

    def _try_subtitles(self, url: str) -> str:
        """Скачать готовые субтитры (если есть) и вернуть очищенный текст."""
        out_tmpl = os.path.join(self.temp_dir, "youtube_subs.%(ext)s")
        try:
            with yt_dlp.YoutubeDL(
                {
                    "skip_download": True,
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": ["ru", "en"],
                    "outtmpl": out_tmpl,
                    "quiet": True,
                    "no_warnings": True,
                }
            ) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.warning(f"Не удалось скачать субтитры: {e}")
            return ""

        for fname in os.listdir(self.temp_dir):
            if not fname.startswith("youtube_subs"):
                continue
            if not fname.endswith((".vtt", ".srt")):
                continue
            full = os.path.join(self.temp_dir, fname)
            with open(full, "r", encoding="utf-8") as f:
                text = self._clean_subtitle_text(f.read())
            try:
                os.remove(full)
            except OSError:
                pass
            if text:
                return text
        return ""

    def _download_audio(self, url: str) -> str:
        """Скачать аудиодорожку. Возвращает путь или ''."""
        for ext in ("m4a", "mp3", "webm", "opus"):
            outtmpl = os.path.join(self.temp_dir, f"youtube_audio.{ext}.%(ext)s")
            try:
                with yt_dlp.YoutubeDL(
                    {
                        "format": f"bestaudio[ext={ext}]/bestaudio",
                        "outtmpl": outtmpl,
                        "quiet": True,
                        "no_warnings": True,
                    }
                ) as ydl:
                    ydl.download([url])
            except Exception:
                continue
            for fname in os.listdir(self.temp_dir):
                if fname.startswith("youtube_audio.") and fname.endswith(f".{ext}"):
                    return os.path.join(self.temp_dir, fname)
        return ""

    def _download_video(self, url: str) -> str:
        """Скачать видео файл (fallback). Возвращает путь или ''."""
        out = os.path.join(self.temp_dir, "video.mp4")
        try:
            if os.path.exists(out):
                os.remove(out)
            with yt_dlp.YoutubeDL(
                {
                    "format": "best[ext=mp4][filesize<=100M]/best[ext=mp4]/best",
                    "outtmpl": out,
                    "quiet": True,
                    "no_warnings": True,
                }
            ) as ydl:
                ydl.download([url])
        except Exception as e:
            logger.warning(f"Не удалось скачать видео: {e}")
            return ""
        return out if os.path.exists(out) else ""

    def _transcribe_audio(self, audio_path: str) -> Tuple[str, str]:
        """Отправить аудио в Gemini Multimodal, вернуть (текст, язык)."""
        logger.info("Транскрибируем аудио через Gemini...")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        prompt = (
            "Transcribe the speech in this audio verbatim in its original language. "
            'At the end write a separate line: "LANGUAGE: <name>".'
        )
        parts = [prompt, {"inline_data": {"mime_type": "audio/mp3", "data": audio_bytes}}]
        result = generate_multimodal(parts)

        language = "unknown"
        m = re.search(r"LANGUAGE[=::\s]+([^\n]+)", result, re.IGNORECASE)
        if m:
            language = m.group(1).strip()
            result = result[: m.start()].strip()
        try:
            os.remove(audio_path)
        except OSError:
            pass
        return result, language

    def _transcribe_video(self, video_path: str) -> Tuple[str, str]:
        """Отправить видео в Gemini Multimodal, вернуть (текст, язык)."""
        logger.info("Транскрибируем видео через Gemini...")
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        prompt = (
            "Listen to this video and transcribe the speech verbatim in its original language. "
            'After the text write two separate lines:\n"TEXT: <text>"\n"LANGUAGE: <name>"'
        )
        parts = [prompt, {"inline_data": {"mime_type": "video/mp4", "data": video_bytes}}]
        result = generate_multimodal(parts)

        text = ""
        language = "unknown"
        for line in result.splitlines():
            s = line.strip()
            if s.upper().startswith("TEXT:"):
                text = s[5:].strip()
            elif s.upper().startswith("LANGUAGE:"):
                language = s[8:].strip()
        if not text:
            text = result
        return text, language

    def _from_url(self, url: str) -> Tuple[str, str, str, str]:
        """Обработать URL: попробовать субтитры → аудио → видео."""
        logger.info(f"Вход: URL {url}")

        subtitles = self._try_subtitles(url)
        if subtitles:
            return subtitles, self._detect_language(subtitles), "", ""

        audio = self._download_audio(url)
        if audio:
            text, lang = self._transcribe_audio(audio)
            return text, lang, "", audio

        video = self._download_video(url)
        if video:
            text, lang = self._transcribe_video(video)
            return text, lang, video, ""

        raise RuntimeError("Не удалось извлечь текст из URL ни одним из способов")

    @staticmethod
    def _from_text(text: str) -> Tuple[str, str, str, str]:
        return text, ExtractorService._detect_language(text), "", ""

    def process(self, source: str) -> Tuple[str, str, str, str]:
        """Точка входа: текст или URL → (text, language, video_path, audio_path)."""
        if source.startswith(("http://", "https://")):
            return self._from_url(source)
        return self._from_text(source)
