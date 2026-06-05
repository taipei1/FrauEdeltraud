import os
from pathlib import Path
from groq import Groq

def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio from file using Groq Whisper."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment")
    
    whisper_model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
    stt_language = os.getenv("STT_LANGUAGE", "en")
    
    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as fh:
        suffix = audio_path.suffix.lower()
        if suffix == ".mp3":
            mime_type = "audio/mp3"
        elif suffix in (".ogg", ".oga"):
            mime_type = "audio/ogg"
        elif suffix == ".wav":
            mime_type = "audio/wav"
        else:
            mime_type = "application/octet-stream"
            
        r = client.audio.transcriptions.create(
            file=(audio_path.name, fh, mime_type),
            model=whisper_model,
            language=stt_language,
            response_format="text",
        )
    text = r if isinstance(r, str) else (getattr(r, "text", "") or "")
    return text.strip()
