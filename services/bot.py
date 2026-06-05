import os
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
from services.agent import reply, greet, reset

log = logging.getLogger("bot")

ALLOWED_CHAT_IDS = {
    int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip().isdigit()
}

def _allowed(chat_id: int) -> bool:
    return not ALLOWED_CHAT_IDS or chat_id in ALLOWED_CHAT_IDS

async def _action(bot, chat_id: int, kind: ChatAction) -> None:
    try:
        await bot.send_chat_action(chat_id=chat_id, action=kind)
    except Exception:
        pass

async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    await _action(context.bot, chat_id, ChatAction.RECORD_VOICE)
    text = greet(chat_id)
    try:
        mp3_path = await generate_tts(text)
        with open(mp3_path, "rb") as fh:
            await context.bot.send_voice(chat_id=chat_id, voice=fh)
    except Exception as e:
        log.error("TTS on /start failed: %s", e)
        await context.bot.send_message(chat_id=chat_id, text=text)

async def on_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    had = reset(chat_id)
    message = "Conversation history has been reset!" if had else "No active conversation was found."
    await context.bot.send_message(chat_id=chat_id, text=message)

async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return
    
    voice = update.message.voice
    log.info("Voice received: chat_id=%s, duration=%ss, size=%s bytes", chat_id, voice.duration, voice.file_size)
    
    temp_dir = Path(os.getenv("TEMP_DIR", "tmp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    work_dir = Path(tempfile.mkdtemp(prefix="bot_voice_", dir=temp_dir))
    incoming_audio = work_dir / "input.ogg"
    
    try:
        await _action(context.bot, chat_id, ChatAction.TYPING)
        tg_file = await context.bot.get_file(voice.file_id)
        await tg_file.download_to_drive(str(incoming_audio))
        
        if incoming_audio.stat().st_size < 100:
            await context.bot.send_message(chat_id=chat_id, text="The voice message seems empty. Please try again.")
            return
            
        # 1. Transcribe audio
        try:
            user_text = transcribe_audio(incoming_audio)
        except Exception as e:
            log.error("STT failed: %s", e)
            await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't understand the audio. Please try again.")
            return
            
        if not user_text.strip():
            await context.bot.send_message(chat_id=chat_id, text="I couldn't hear anything. Please try again.")
            return
            
        log.info("User: %s", user_text)
        
        # 2. Get AI Response
        await _action(context.bot, chat_id, ChatAction.TYPING)
        try:
            bot_text = reply(chat_id, user_text)
        except Exception as e:
            log.error("Agent reply failed: %s", e)
            await context.bot.send_message(chat_id=chat_id, text="Sorry, I'm having trouble thinking right now.")
            return
            
        if not bot_text.strip():
            await context.bot.send_message(chat_id=chat_id, text="...")
            return
            
        log.info("Bot: %s", bot_text)
        
        # 3. Synthesize and Send Voice
        await _action(context.bot, chat_id, ChatAction.RECORD_VOICE)
        try:
            mp3_path = await generate_tts(bot_text)
            with open(mp3_path, "rb") as fh:
                await context.bot.send_voice(chat_id=chat_id, voice=fh)
        except Exception as e:
            log.error("TTS failed: %s", e)
            await context.bot.send_message(chat_id=chat_id, text=bot_text)
            
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

def start_telegram_bot() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment")
        
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("reset", on_reset))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    
    log.info("Starting Telegram Bot polling...")
    app.run_polling(drop_pending_updates=True)
