"""Config - минимум параметров."""
import os
from pathlib import Path
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)


class Settings:
    """Настройки."""
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = "gemini-2.5-flash"
    TEMPERATURE = 0.3
    
    TEMP_DIR = "./temp"
    LEARNED_TOPICS_PATH = os.path.join(
        Path(__file__).resolve().parent, "learned_topics.json"
    )
    
    # Video
    VIDEO_WIDTH = 544
    VIDEO_HEIGHT = 960
    VIDEO_FPS = 15
    VIDEO_FONT_SIZE = 56
    
    # TTS
    TTS_VOICE = "en-US-GuyNeural"
    TTS_RATE = "0%"
    
    # Translator
    MAX_VOCABULARY_WORDS = 50
    TRANSLATION_CHUNK_SIZE = 3000
    
    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:mysecretpassword@localhost:5432/postgres",
    )


os.makedirs(Settings.TEMP_DIR, exist_ok=True)
