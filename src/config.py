import os
from pathlib import Path
from dotenv import load_dotenv

# .env проекта имеет приоритет над глобальными переменными окружения
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

class Settings:
    # Google APIs
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")

    # Groq fallback (llama-3.3-70b-versatile)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    TEMPERATURE = 0.3  # Lower for more deterministic translations

    # LangChain Tracing
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true") == "true"
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "frauEdeltraud")

    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:mysecretpassword@localhost:5432/postgres",
    )

    # Telegram
    TELEGRAM_TOKEN = os.getenv(
        "TELEGRAM_TOKEN",
        "6480342406:AAFL9HEv8QU-nJISTxuUuK0_Zk_zzxwWTpw",
    )
    TELEGRAM_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "512504505"))

    # Paths
    TEMP_DIR = "./temp"
    MODELS_DIR = "./models"
    LEARNED_TOPICS_PATH = os.path.join(
        os.path.dirname(__file__), "learned_topics.json"
    )

    # Video Settings
    VIDEO_WIDTH = 544   # Vertical (9:16)
    VIDEO_HEIGHT = 960  # Vertical (9:16)
    VIDEO_FPS = 5       # Extremely lightweight
    VIDEO_BITRATE = "300k"
    VIDEO_CODEC = "libx264"
    AUDIO_BITRATE = "96k"

    # TTS Settings
    TTS_VOICE = "en-US-GuyNeural"
    TTS_RATE = "+0%"

    # Translator Settings
    VOCABULARY_SIMILARITY_THRESHOLD = 0.75
    MAX_VOCABULARY_WORDS = 100
    MAX_RELEVANT_VOCABULARY_WORDS = 30
    TRANSLATION_CHUNK_SIZE = 3000
    MAX_VIDEO_TEXT_CHARS = 2000000
    
    # Video Settings
    SKIP_AUDIO_ANALYSIS = True  # Пропустить анализ аудио, использовать простую синхронизацию

# Warn if Google API key is missing; fallback to Groq will be handled by llm_client
if not Settings.GOOGLE_API_KEY:
    import logging
    logging.getLogger(__name__).warning(
        "GOOGLE_API_KEY not set; Gemini will be disabled, using Groq fallback if available."
    )

# Ensure temp directories exist
os.makedirs(Settings.TEMP_DIR, exist_ok=True)
os.makedirs(Settings.MODELS_DIR, exist_ok=True)