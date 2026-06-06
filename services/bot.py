import os
import re
import shutil
import tempfile
import logging
from pathlib import Path
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
)

from services.stt import transcribe_audio
from services.tts import generate_tts
from services.agent import reply, greet, reset, generate_story
from services.vocabulary import reindex_embeddings, count_missing_embeddings

log = logging.getLogger("bot")

ALLOWED_CHAT_IDS = {
    int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip().isdigit()
}

_VOCAB_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_vocab_tokens: set[str] | None = None

_EXCLUDED_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "and", "or", "but", "not", "no", "yes",
    "if", "when", "then", "so", "because", "also", "too",
    "to", "in", "on", "at", "for", "with", "from", "by", "of", "about",
    "up", "down", "out", "off", "over", "under",
    "what", "where", "who", "whose", "why", "how", "which",
    "here", "there", "now", "then",
    "do", "does", "did", "done",
    "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "very", "much", "many", "some", "any", "all",
    "one", "two", "three",
}


def _get_vocab_tokens() -> set[str]:
    global _vocab_tokens
    if _vocab_tokens is not None:
        return _vocab_tokens
    try:
        from services.vector_vocabulary import get_vector_vocabulary
        v = get_vector_vocabulary("en")
        if not v._loaded:
            v.load()
        all_tokens = v.tokens()
        _vocab_tokens = {t for t in all_tokens if len(t) > 2 or t not in _EXCLUDED_WORDS}
        log.info("Loaded %d vocab tokens for highlighting", len(_vocab_tokens))
    except Exception as e:
        log.warning("Failed to load vocab for highlighting: %s", e)
        _vocab_tokens = set()
    return _vocab_tokens


def _strip_markdown(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)

def _highlight_vocab(text: str) -> str:
    text = _strip_markdown(text)
    tokens = _get_vocab_tokens()
    if not tokens:
        return text

    def _replace(m: re.Match) -> str:
        t = m.group(0)
        low = t.lower()
        if low not in _EXCLUDED_WORDS and low in tokens:
            return f"({t})"
        return t

    return _VOCAB_WORD_RE.sub(_replace, text)


def _allowed(chat_id: int) -> bool:
    return not ALLOWED_CHAT_IDS or chat_id in ALLOWED_CHAT_IDS


async def _action(bot, chat_id: int, kind: ChatAction) -> None:
    try:
        await bot.send_chat_action(chat_id=chat_id, action=kind)
    except Exception:
        pass


async def _send_voice_and_text(bot, chat_id: int, text: str) -> None:
    try:
        mp3_path = await generate_tts(text)
        with open(mp3_path, "rb") as fh:
            await bot.send_voice(chat_id=chat_id, voice=fh)
    except Exception as e:
        log.warning("TTS failed: %s", e)

    try:
        highlighted = _highlight_vocab(text)
        await bot.send_message(chat_id=chat_id, text=highlighted)
    except Exception as e:
        log.warning("Text send failed: %s", e)
        await bot.send_message(chat_id=chat_id, text=text)


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    await _action(context.bot, chat_id, ChatAction.RECORD_VOICE)
    text = greet(chat_id)
    await _send_voice_and_text(context.bot, chat_id, text)


async def on_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    had = reset(chat_id)
    message = "Conversation has been reset!" if had else "No active conversation found."
    await context.bot.send_message(chat_id=chat_id, text=message)


async def on_reindex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    await _action(context.bot, chat_id, ChatAction.TYPING)
    try:
        missing = count_missing_embeddings("en")
        if missing == 0:
            await context.bot.send_message(
                chat_id=chat_id,
                text="All vocabulary words already have embeddings. Nothing to reindex.",
            )
            return
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Reindexing {missing} words... this may take a moment.",
        )
        stats = reindex_embeddings("en")
        # Clear vocab cache so next request picks up new embeddings
        from services.vector_vocabulary import _vocab_cache
        _vocab_cache.pop("en", None)
        global _vocab_tokens
        _vocab_tokens = None
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Done! Processed {stats['processed']} words.",
        )
    except Exception as e:
        log.error("Reindex failed: %s", e)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Reindex failed: {e}",
        )


async def on_story(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    args = context.args
    if not args:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Usage: /story <topic> [word_count]\nExample: /story about space exploration 200",
        )
        return

    text_args = " ".join(args)
    word_count = 200
    topic = text_args

    # Try to extract word count from the end
    parts = text_args.rsplit(None, 1)
    if len(parts) == 2 and parts[1].isdigit():
        topic = parts[0]
        word_count = int(parts[1])
        word_count = max(50, min(word_count, 1000))

    await _action(context.bot, chat_id, ChatAction.TYPING)
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"Generating a story about '{topic}' ({word_count} words)...",
    )

    try:
        story = generate_story(chat_id, topic, word_count)
        await _send_voice_and_text(context.bot, chat_id, story)
    except Exception as e:
        log.error("Story generation failed: %s", e)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Story generation failed: {e}",
        )


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    voice = update.message.voice
    log.info("Voice: chat_id=%s, duration=%ss", chat_id, voice.duration)

    temp_dir = Path(os.getenv("TEMP_DIR", "tmp"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="bot_voice_", dir=temp_dir))
    incoming_audio = work_dir / "input.ogg"

    try:
        await _action(context.bot, chat_id, ChatAction.TYPING)
        tg_file = await context.bot.get_file(voice.file_id)
        await tg_file.download_to_drive(str(incoming_audio))

        if incoming_audio.stat().st_size < 100:
            await context.bot.send_message(chat_id=chat_id, text="Voice message seems empty. Please try again.")
            return

        try:
            user_text = transcribe_audio(incoming_audio)
        except Exception as e:
            log.error("STT failed: %s", e)
            await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't understand the audio.")
            return

        if not user_text.strip():
            await context.bot.send_message(chat_id=chat_id, text="I couldn't hear anything. Please try again.")
            return

        log.info("User: %s", user_text)

        await _action(context.bot, chat_id, ChatAction.TYPING)
        try:
            result = reply(chat_id, user_text)
        except Exception as e:
            log.error("Agent reply failed: %s", e)
            await context.bot.send_message(chat_id=chat_id, text="Sorry, I'm having trouble thinking right now.")
            return

        bot_text = result.get("final_response", "")

        if not bot_text.strip():
            await context.bot.send_message(chat_id=chat_id, text="...")
            return

        log.info("Bot: %s", bot_text)

        await _action(context.bot, chat_id, ChatAction.RECORD_VOICE)
        await _send_voice_and_text(context.bot, chat_id, bot_text)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def start_telegram_bot() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("reset", on_reset))
    app.add_handler(CommandHandler("reindex", on_reindex))
    app.add_handler(CommandHandler("story", on_story))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))

    log.info("Starting Telegram Bot polling...")
    app.run_polling(drop_pending_updates=True)
