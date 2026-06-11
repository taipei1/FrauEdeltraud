import os
from dataclasses import dataclass, field


def _env_flag(name: str, default: str = "") -> str:
    raw = os.getenv(name, default).strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in {"0", "false", "no", "off", "your_assemblyai_api_key_here", "your_..._here"}:
        return ""
    if lowered.startswith("your_") and lowered.endswith("_here"):
        return ""
    return raw


@dataclass
class DubConfig:
    temp_dir: str = field(default_factory=lambda: os.getenv("DUB_TEMP_DIR", "tmp/video_dub"))
    output_dir: str = field(default_factory=lambda: os.getenv("DUB_OUTPUT_DIR", "results/dubs"))
    max_duration_minutes: int = int(os.getenv("DUB_MAX_DURATION", "30"))
    target_language: str = field(
        default_factory=lambda: os.getenv("DUB_TARGET_LANGUAGE")
        or os.getenv("TARGET_LANGUAGE")
        or "en"
    )
    source_language: str = field(
        default_factory=lambda: os.getenv("DUB_SOURCE_LANGUAGE")
        or os.getenv("STT_LANGUAGE")
        or "en"
    )
    whisper_model: str = field(default_factory=lambda: os.getenv("DUB_WHISPER_MODEL", "whisper-large-v3"))
    llm_temperature: float = float(os.getenv("DUB_LLM_TEMPERATURE", "0.3"))
    llm_max_tokens: int = int(os.getenv("DUB_LLM_MAX_TOKENS", "300"))
    max_parallel_segments: int = int(os.getenv("DUB_MAX_PARALLEL", "4"))
    silence_threshold_sec: float = float(os.getenv("DUB_SILENCE_THRESHOLD", "0.7"))
    min_segment_duration: float = float(os.getenv("DUB_MIN_SEGMENT_SEC", "1.0"))
    max_segment_duration: float = float(os.getenv("DUB_MAX_SEGMENT_SEC", "15.0"))
    use_assemblyai_diarization: bool = _env_flag("ASSEMBLYAI_API_KEY") != ""
    edge_tts_rate: str = field(default_factory=lambda: os.getenv("EDGE_TTS_RATE", "+0%"))
    edge_tts_pitch: str = field(default_factory=lambda: os.getenv("EDGE_TTS_PITCH", "+0Hz"))

    @property
    def voice_pool(self) -> list[dict]:
        lang = self.target_language
        pools = {
            "ru": [
                {"voice": "ru-RU-SvetlanaNeural", "gender": "female", "name": "Sveta"},
                {"voice": "ru-RU-DmitryNeural", "gender": "male", "name": "Dima"},
            ],
            "en": [
                {"voice": "en-US-JennyNeural", "gender": "female", "name": "Jenny"},
                {"voice": "en-US-AriaNeural", "gender": "female", "name": "Aria"},
                {"voice": "en-US-GuyNeural", "gender": "male", "name": "Guy"},
                {"voice": "en-GB-SoniaNeural", "gender": "female", "name": "Sonia"},
                {"voice": "en-GB-RyanNeural", "gender": "male", "name": "Ryan"},
            ],
            "de": [
                {"voice": "de-DE-KatjaNeural", "gender": "female", "name": "Katja"},
                {"voice": "de-DE-ConradNeural", "gender": "male", "name": "Conrad"},
            ],
            "uk": [
                {"voice": "uk-UA-OstapNeural", "gender": "male", "name": "Ostap"},
                {"voice": "uk-UA-PolinaNeural", "gender": "female", "name": "Polina"},
            ],
        }
        return pools.get(lang, pools.get("en", []))

    @property
    def voice_map_fallback(self) -> list[str]:
        return [v["voice"] for v in self.voice_pool]
