"""3 агента LangGraph: EXTRACTOR → TRANSLATOR → VIDEO GENERATOR."""
import logging

from services.extractor import ExtractorService
from services.telegram_service import TelegramService
from services.translator import TranslatorService
from services.video_service import VideoService
from langgraph_system.state import AgentState

logger = logging.getLogger(__name__)


def _build_services():
    return {
        "extractor": ExtractorService(),
        "translator": TranslatorService(),
        "video": VideoService(),
        "telegram": TelegramService(),
    }


def agent_extractor(state: AgentState) -> AgentState:
    logger.info("=" * 60)
    logger.info("[AGENT 1] EXTRACTOR")
    logger.info("=" * 60)
    try:
        if "extractor" not in state:
            state["extractor"] = ExtractorService()
        text, lang, video_path, audio_path = state["extractor"].process(state["input_source"])
        state["raw_text"] = text
        state["source_language"] = lang
        state["video_path"] = video_path
        state["audio_path"] = audio_path
        state["status"] = "extraction_done"
        logger.info(f"[OK] Извлечено ({lang}): {text[:120]}...")
    except Exception as e:
        logger.error(f"[FAIL] Extractor: {e}")
        state["status"] = "failed"
        state["error_message"] = str(e)
    return state


def agent_translator(state: AgentState) -> AgentState:
    logger.info("=" * 60)
    logger.info("[AGENT 2] TRANSLATOR")
    logger.info("=" * 60)
    if state.get("status") == "failed":
        return state
    try:
        if "translator" not in state:
            state["translator"] = TranslatorService()
        translated, usage, stats = state["translator"].process(state["raw_text"])
        state["translated_text"] = translated
        state["vocabulary_usage"] = usage
        state["vocabulary_stats"] = stats
        state["status"] = "translation_done"
        logger.info(
            f"[OK] Перевод: {translated[:120]}... "
            f"({stats['used_count']}/{stats['total_vocabulary_words']} слов)"
        )
    except Exception as e:
        logger.error(f"[FAIL] Translator: {e}")
        state["status"] = "failed"
        state["error_message"] = str(e)
    return state


def agent_video_generator(state: AgentState) -> AgentState:
    logger.info("=" * 60)
    logger.info("[AGENT 3] VIDEO GENERATOR")
    logger.info("=" * 60)
    if state.get("status") == "failed":
        return state
    try:
        if "video" not in state:
            state["video"] = VideoService()
        if "telegram" not in state:
            state["telegram"] = TelegramService()

        video_path, duration, _ = state["video"].process(state["translated_text"])
        state["video_output_path"] = video_path
        state["audio_duration"] = duration

        stats = state.get("vocabulary_stats", {})
        ok = state["telegram"].process_video(video_path, stats)
        if ok:
            state["status"] = "completed"
            state["final_message"] = (
                f"Готово! Видео {duration:.0f} сек, отправлено в Telegram."
            )
        else:
            state["status"] = "failed"
            state["error_message"] = "Не удалось отправить видео в Telegram"
    except Exception as e:
        logger.error(f"[FAIL] Video Generator: {e}")
        state["status"] = "failed"
        state["error_message"] = str(e)
    return state
