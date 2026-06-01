"""Telegram service: отправка видео со статистикой словаря."""
import asyncio
import logging
import os
from typing import Dict, Optional

from telegram import Bot
from telegram.error import TelegramError

from config import Settings

logger = logging.getLogger(__name__)


class TelegramService:
    def __init__(self):
        self.bot = Bot(token=Settings.TELEGRAM_TOKEN)
        self.chat_id = Settings.TELEGRAM_CHAT_ID
        logger.info("[OK] TelegramService инициализирован")

    async def _send_video(self, path: str, caption: str) -> Optional[int]:
        size_mb = os.path.getsize(path) / 1024 / 1024
        logger.info(f"Отправка видео в Telegram ({size_mb:.2f} MB)...")
        with open(path, "rb") as f:
            msg = await self.bot.send_video(
                chat_id=self.chat_id,
                video=f,
                caption=caption,
                supports_streaming=True,
                read_timeout=120,
                write_timeout=120,
            )
        return msg.message_id

    def send_video(self, path: str, caption: str) -> Optional[int]:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._send_video(path, caption))
        except TelegramError as e:
            logger.error(f"[FAIL] Telegram: {e}")
            return None
        except Exception as e:
            logger.error(f"[FAIL] {e}")
            return None
        finally:
            try:
                loop.close()
            except Exception:
                pass

    @staticmethod
    def build_caption(stats: Dict) -> str:
        used = stats.get("used_count", 0)
        total = stats.get("total_vocabulary_words", 0)
        pct = stats.get("vocabulary_percentage", 0)
        words = stats.get("used_words", [])

        lines = [
            "Перевод готов!",
            "",
            f"Словарь: {used}/{total} ({pct}%)",
            "",
        ]
        for w in words[:15]:
            lines.append(f"  - {w}")
        if len(words) > 15:
            lines.append(f"  ... +{len(words) - 15} ещё")
        lines.append("")
        lines.append("Слушай и учи слова!")
        return "\n".join(lines)

    def process_video(self, path: str, stats: Dict) -> bool:
        caption = self.build_caption(stats)
        msg_id = self.send_video(path, caption)
        if msg_id:
            logger.info(f"[OK] Видео отправлено, message_id={msg_id}")
            return True
        logger.error("[FAIL] Не удалось отправить видео")
        return False
