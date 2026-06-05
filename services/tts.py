import asyncio
import os
import hashlib
from pathlib import Path
import edge_tts

async def synthesize_speech_async(text: str, output_path: Path) -> None:
    voice = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")
    rate = os.getenv("EDGE_TTS_RATE", "+0%")
    pitch = os.getenv("EDGE_TTS_PITCH", "+0Hz")
    
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch
    )
    await communicate.save(str(output_path))

async def generate_tts(text: str) -> Path:
    """Synthesizes text to an MP3 file and returns the path to it. Caches results based on hash."""
    text = text.strip()
    if not text:
        raise ValueError("Empty text provided for TTS")
        
    temp_dir = Path(os.getenv("TEMP_DIR", "tmp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    mp3_path = temp_dir / f"tts_{digest}.mp3"
    
    if mp3_path.exists() and mp3_path.stat().st_size > 0:
        return mp3_path
        
    await synthesize_speech_async(text, mp3_path)
    return mp3_path
